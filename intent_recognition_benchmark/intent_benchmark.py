from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


INTENT_LABELS = (
    "zoom_in",
    "reveal",
    "branch_out",
    "reframe",
    "follow",
    "interact",
)
INTENT_SET = set(INTENT_LABELS)
REPO_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(rendered, encoding="utf-8")


def normalized_box(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if not 0.0 <= x1 < x2 <= 1.0 or not 0.0 <= y1 < y2 <= 1.0:
        return None
    return [x1, y1, x2, y2]


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def assign_group_splits(groups: Iterable[str], *, seed: int) -> dict[str, str]:
    ordered = sorted(set(groups))
    if len(ordered) < 3:
        raise ValueError("At least three groups are required for train/val/test splitting")
    random.Random(seed).shuffle(ordered)
    test_count = max(1, round(len(ordered) * 0.15))
    val_count = max(1, round(len(ordered) * 0.15))
    while len(ordered) - test_count - val_count < 1:
        if test_count >= val_count and test_count > 1:
            test_count -= 1
        elif val_count > 1:
            val_count -= 1
        else:
            raise ValueError("Unable to reserve a non-empty training split")
    test_groups = set(ordered[:test_count])
    val_groups = set(ordered[test_count : test_count + val_count])
    return {
        group: (
            "test"
            if group in test_groups
            else "val"
            if group in val_groups
            else "train"
        )
        for group in ordered
    }


def _source_image(source_tree: Path, source_node: str, node: dict[str, Any]) -> Path:
    expected = source_tree.parent / "images" / f"{source_node}.png"
    if expected.is_file():
        return expected
    raw_image = str(node.get("image", "")).replace("\\", "/")
    fallback = source_tree.parent / "images" / Path(raw_image).name
    return fallback


def _human_review_map(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    reviews: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError(f"Human review row is missing sample_id: {row}")
        decision = str(row.get("decision", "")).strip().lower()
        if decision not in {"accept", "exclude"}:
            raise ValueError(f"Human review decision must be accept or exclude: {sample_id}")
        label = str(row.get("closed_intent", "")).strip().lower()
        if decision == "accept" and label not in INTENT_SET:
            raise ValueError(f"Human review has invalid closed_intent for {sample_id}: {label}")
        reviews[sample_id] = row
    return reviews


def prepare_benchmark(
    *,
    labels_dir: Path,
    output_dir: Path,
    protocol: str,
    seed: int = 20260727,
    include_repair_review_queue: bool = False,
    human_review_file: Path | None = None,
    min_eval_per_class: int = 20,
    include_oracle_user_state: bool = False,
) -> dict[str, Any]:
    labels_dir = labels_dir.resolve()
    output_dir = output_dir.resolve()
    if protocol not in {"unseen_user", "unseen_topic"}:
        raise ValueError("protocol must be unseen_user or unseen_topic")

    audits = {
        (str(row["tree_id"]), str(row["edge_id"])): row
        for row in read_jsonl(labels_dir / "label_audit.jsonl")
    }
    repair_review_rows = read_jsonl(labels_dir / "label_review_queue.jsonl")
    repair_review = {
        (str(row["tree_id"]), str(row["edge_id"])): row
        for row in repair_review_rows
    }
    human_reviews = _human_review_map(human_review_file)
    tree_paths = sorted((labels_dir / "trees").glob("user_*/*/tree.json"))
    if not tree_paths:
        raise FileNotFoundError(f"No repaired tree.json files found under {labels_dir / 'trees'}")

    candidates: list[dict[str, Any]] = []
    invalid_samples: list[dict[str, Any]] = []
    excluded_repair_review = 0
    excluded_human_review = 0
    for tree_path in tree_paths:
        tree = read_json(tree_path)
        tree_id = str(tree["tree_id"])
        user_id = str(tree["user_id"])
        topic_id = str(tree["topic_id"])
        nodes = {str(node["node_id"]): node for node in tree.get("nodes", [])}
        for edge in tree.get("edges", []):
            edge_id = str(edge["edge_id"])
            sample_id = f"{tree_id}/{edge_id}"
            key = (tree_id, edge_id)
            audit = audits.get(key)
            if audit is None:
                invalid_samples.append({"sample_id": sample_id, "reason": "missing_label_audit"})
                continue
            if key in repair_review and not include_repair_review_queue:
                excluded_repair_review += 1
                continue
            human_review = human_reviews.get(sample_id)
            if human_review and str(human_review["decision"]).lower() == "exclude":
                excluded_human_review += 1
                continue

            source_node = str(edge["source_node"])
            node = nodes.get(source_node)
            if node is None:
                invalid_samples.append({"sample_id": sample_id, "reason": "missing_source_node"})
                continue
            source_tree = Path(str(audit["source_tree"]))
            source_image = _source_image(source_tree, source_node, node)
            box = normalized_box(edge.get("grounding", {}).get("target_box"))
            if not source_image.is_file():
                invalid_samples.append(
                    {"sample_id": sample_id, "reason": "missing_source_image", "path": str(source_image)}
                )
                continue
            if box is None:
                invalid_samples.append({"sample_id": sample_id, "reason": "invalid_target_box"})
                continue

            repaired_label = str(edge.get("closed_intent", "")).strip().lower()
            label = repaired_label
            human_reviewed = False
            if human_review:
                label = str(human_review["closed_intent"]).strip().lower()
                human_reviewed = True
            if label not in INTENT_SET:
                invalid_samples.append(
                    {"sample_id": sample_id, "reason": "invalid_closed_intent", "label": label}
                )
                continue

            candidates.append(
                {
                    "sample_id": sample_id,
                    "user_id": user_id,
                    "topic_id": topic_id,
                    "tree_id": tree_id,
                    "edge_id": edge_id,
                    "source_node": source_node,
                    "source_image": repo_relative(source_image),
                    "target_box": box,
                    "click_center": [round((box[0] + box[2]) / 2, 6), round((box[1] + box[3]) / 2, 6)],
                    "source_story_state": node.get("story_state", {}),
                    "oracle_user_state_before": {
                        "stable_profile": node.get("stable_profile", {}),
                        "affective_state": node.get("affective_state", {}),
                        "current_state": node.get("current_state", {}),
                        "profile_summary": node.get("profile_summary", ""),
                    },
                    "visual_group_id": (
                        f"topic:{topic_id}:shared_root"
                        if source_node == "n0"
                        else f"tree:{tree_id}:node:{source_node}"
                    ),
                    "closed_intent": label,
                    "repaired_closed_intent": repaired_label,
                    "legacy_closed_intent": str(audit.get("legacy_closed_intent", "")),
                    "target_label": str(edge.get("grounding", {}).get("target_label", "")),
                    "annotation_confidence": float(audit.get("confidence", 0.0)),
                    "annotation_source": (
                        "human_review" if human_reviewed else "model_text_adjudication_no_image"
                    ),
                    "human_reviewed": human_reviewed,
                    "label_changed": bool(audit.get("label_changed")),
                    "repair_review_reasons": list(repair_review.get(key, {}).get("review_reasons", [])),
                }
            )

    group_field = "user_id" if protocol == "unseen_user" else "topic_id"
    split_by_group = assign_group_splits(
        (str(row[group_field]) for row in candidates),
        seed=seed,
    )
    for row in candidates:
        row["split"] = split_by_group[str(row[group_field])]

    inputs = [
        {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "protocol": protocol,
            "user_id": row["user_id"],
            "topic_id": row["topic_id"],
            "tree_id": row["tree_id"],
            "edge_id": row["edge_id"],
            "source_node": row["source_node"],
            "source_image": row["source_image"],
            "interaction": {
                "type": "box",
                "target_box": row["target_box"],
                "click_center": row["click_center"],
            },
            "context": {
                "story_state_before": row["source_story_state"],
                **(
                    {"oracle_user_state_before": row["oracle_user_state_before"]}
                    if include_oracle_user_state
                    else {}
                ),
            },
            "visual_group_id": row["visual_group_id"],
        }
        for row in candidates
    ]
    labels = [
        {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "closed_intent": row["closed_intent"],
            "grounding_target_label": row["target_label"],
            "annotation_confidence": row["annotation_confidence"],
            "annotation_source": row["annotation_source"],
            "human_reviewed": row["human_reviewed"],
            "repaired_closed_intent": row["repaired_closed_intent"],
            "legacy_closed_intent": row["legacy_closed_intent"],
            "label_changed": row["label_changed"],
        }
        for row in candidates
    ]

    class_counts = Counter(row["closed_intent"] for row in candidates)
    split_counts = {
        split: Counter(row["closed_intent"] for row in candidates if row["split"] == split)
        for split in ("train", "val", "test")
    }
    test_rows = [row for row in candidates if row["split"] == "test"]
    missing_test_classes = [
        label for label in INTENT_LABELS if split_counts["test"].get(label, 0) < min_eval_per_class
    ]
    unreviewed_test_count = sum(not row["human_reviewed"] for row in test_rows)
    formal_ready = not missing_test_classes and unreviewed_test_count == 0
    warnings: list[str] = []
    if missing_test_classes:
        warnings.append(
            "Test split does not meet the minimum per-class support for: "
            + ", ".join(missing_test_classes)
        )
    if unreviewed_test_count:
        warnings.append(
            f"{unreviewed_test_count} test labels are model-adjudicated and not human-reviewed."
        )
    if protocol == "unseen_user":
        warnings.append(
            "The unseen_user protocol can repeat shared topic root images across splits; report unseen_topic as the stricter visual-domain result."
        )
    if include_oracle_user_state:
        warnings.append(
            "User state is an oracle snapshot maintained by the dataset simulator at the source node; it is not an online estimate from interaction history."
        )
    warnings.append(
        "OOS is excluded because legacy oos_regions do not identify their source_node and cannot be paired with a source image reliably."
    )

    review_queue = [
        {
            "sample_id": row["sample_id"],
            "split": row["split"],
            "source_image": row["source_image"],
            "target_box": row["target_box"],
            "current_label": row["closed_intent"],
            "legacy_label": row["legacy_closed_intent"],
            "annotation_confidence": row["annotation_confidence"],
            "suggested_review": {
                "decision": "accept | exclude",
                "closed_intent": row["closed_intent"],
                "reviewer": "",
                "notes": "",
            },
        }
        for row in candidates
        if row["split"] in {"val", "test"} and not row["human_reviewed"]
    ]
    review_queue.sort(
        key=lambda row: (
            0 if row["split"] == "test" else 1,
            split_counts[row["split"]].get(row["current_label"], 0),
            row["annotation_confidence"],
            row["sample_id"],
        )
    )

    report = {
        "schema_version": "intent_recognition_benchmark_v1",
        "protocol": protocol,
        "seed": seed,
        "labels_dir": repo_relative(labels_dir),
        "tree_count": len(tree_paths),
        "sample_count": len(candidates),
        "excluded_repair_review_count": excluded_repair_review,
        "excluded_human_review_count": excluded_human_review,
        "invalid_sample_count": len(invalid_samples),
        "class_counts": {label: class_counts.get(label, 0) for label in INTENT_LABELS},
        "split_class_counts": {
            split: {label: split_counts[split].get(label, 0) for label in INTENT_LABELS}
            for split in ("train", "val", "test")
        },
        "split_groups": {
            split: sorted(group for group, assigned in split_by_group.items() if assigned == split)
            for split in ("train", "val", "test")
        },
        "human_reviewed_test_count": len(test_rows) - unreviewed_test_count,
        "unreviewed_test_count": unreviewed_test_count,
        "min_eval_per_class": min_eval_per_class,
        "user_state_mode": "oracle_source_node" if include_oracle_user_state else "none",
        "missing_or_underfilled_test_classes": missing_test_classes,
        "formal_benchmark_ready": formal_ready,
        "warnings": warnings,
        "forbidden_model_inputs": [
            "closed_intent",
            "legacy_closed_intent",
            "intent_ranking",
            "natural_language_intent",
            "branch_label",
            "grounding.target_caption",
            "slots",
            "story_state_after",
            "generation_prompt",
            "target_image",
            "repair rationale",
            "annotation_metadata",
            "private_generation_metadata",
            "private_generation_summary",
        ],
    }
    write_jsonl(output_dir / "inputs.jsonl", inputs)
    write_jsonl(output_dir / "labels.jsonl", labels)
    write_jsonl(output_dir / "human_review_queue.jsonl", review_queue)
    write_json(output_dir / "split_stats.json", report)
    write_json(output_dir / "invalid_samples.json", invalid_samples)
    return report


def _normalize_prediction(row: dict[str, Any]) -> tuple[str, list[str], float | None, bool]:
    raw_ranking = row.get("intent_ranking", row.get("ranking"))
    ranking: list[str] = []
    confidence: float | None = None
    has_ranking = isinstance(raw_ranking, list) and bool(raw_ranking)
    if isinstance(raw_ranking, list):
        for item in raw_ranking:
            if isinstance(item, dict):
                label = str(item.get("intent", item.get("label", ""))).strip().lower()
                if confidence is None and item.get("score") is not None:
                    try:
                        confidence = float(item["score"])
                    except (TypeError, ValueError):
                        confidence = None
            else:
                label = str(item).strip().lower()
            if label in INTENT_SET and label not in ranking:
                ranking.append(label)
    predicted = str(
        row.get("predicted_intent", row.get("closed_intent", ranking[0] if ranking else ""))
    ).strip().lower()
    if predicted not in INTENT_SET:
        raise ValueError(f"Invalid predicted intent: {predicted!r}")
    if predicted in ranking:
        ranking.remove(predicted)
    ranking.insert(0, predicted)
    if confidence is None and row.get("confidence") is not None:
        try:
            confidence = float(row["confidence"])
        except (TypeError, ValueError):
            confidence = None
    if confidence is not None:
        confidence = max(0.0, min(1.0, confidence))
    return predicted, ranking, confidence, has_ranking


def score_predictions(
    *,
    labels: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    split: str = "test",
) -> dict[str, Any]:
    selected = {str(row["sample_id"]): row for row in labels if row.get("split") == split}
    if not selected:
        raise ValueError(f"No labels found for split={split}")
    prediction_by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    for row in predictions:
        sample_id = str(row.get("sample_id", ""))
        if sample_id in prediction_by_id:
            duplicate_ids.append(sample_id)
        prediction_by_id[sample_id] = row
    if duplicate_ids:
        raise ValueError(f"Duplicate prediction sample_ids: {sorted(set(duplicate_ids))[:10]}")
    missing = sorted(set(selected) - set(prediction_by_id))
    if missing:
        raise ValueError(f"Missing {len(missing)} predictions; first ids: {missing[:10]}")

    confusion = {
        actual: {predicted: 0 for predicted in INTENT_LABELS}
        for actual in INTENT_LABELS
    }
    correct = 0
    top3_hits = 0
    reciprocal_rank_sum = 0.0
    ranked_count = 0
    confidence_rows: list[tuple[float, bool]] = []
    for sample_id, label_row in selected.items():
        actual = str(label_row["closed_intent"])
        predicted, ranking, confidence, has_ranking = _normalize_prediction(prediction_by_id[sample_id])
        confusion[actual][predicted] += 1
        is_correct = predicted == actual
        correct += int(is_correct)
        if has_ranking:
            ranked_count += 1
            top3_hits += int(actual in ranking[:3])
            reciprocal_rank_sum += 1.0 / (ranking.index(actual) + 1) if actual in ranking else 0.0
        if confidence is not None:
            confidence_rows.append((confidence, is_correct))

    per_class: dict[str, dict[str, Any]] = {}
    f1_values: list[float] = []
    supported_f1: list[float] = []
    recall_values: list[float] = []
    weighted_f1 = 0.0
    total = len(selected)
    for label in INTENT_LABELS:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted_count = sum(confusion[actual][label] for actual in INTENT_LABELS)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": support,
            "predicted_count": predicted_count,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
        f1_values.append(f1)
        recall_values.append(recall)
        if support:
            supported_f1.append(f1)
            weighted_f1 += f1 * support / total

    ece: float | None = None
    if confidence_rows:
        weighted_gap = 0.0
        for bin_index in range(10):
            low = bin_index / 10
            high = (bin_index + 1) / 10
            bucket = [
                row
                for row in confidence_rows
                if (
                    (low <= row[0] <= high)
                    if bin_index == 9
                    else (low <= row[0] < high)
                )
            ]
            if not bucket:
                continue
            mean_confidence = sum(row[0] for row in bucket) / len(bucket)
            mean_accuracy = sum(row[1] for row in bucket) / len(bucket)
            weighted_gap += len(bucket) / len(confidence_rows) * abs(mean_confidence - mean_accuracy)
        ece = round(weighted_gap, 6)

    return {
        "schema_version": "intent_recognition_metrics_v1",
        "split": split,
        "sample_count": total,
        "top1_accuracy": round(correct / total, 6),
        "macro_precision": round(sum(row["precision"] for row in per_class.values()) / len(INTENT_LABELS), 6),
        "macro_recall_balanced_accuracy": round(sum(recall_values) / len(INTENT_LABELS), 6),
        "macro_f1_all_six": round(sum(f1_values) / len(INTENT_LABELS), 6),
        "macro_f1_supported_classes": round(sum(supported_f1) / len(supported_f1), 6) if supported_f1 else 0.0,
        "weighted_f1": round(weighted_f1, 6),
        "top3_accuracy": round(top3_hits / ranked_count, 6) if ranked_count else None,
        "mrr": round(reciprocal_rank_sum / ranked_count, 6) if ranked_count else None,
        "ranking_coverage": round(ranked_count / total, 6),
        "confidence_coverage": round(len(confidence_rows) / total, 6),
        "ece_10_bin": ece,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def _ranked_prediction(sample_id: str, ordering: list[str], scores: dict[str, float]) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "predicted_intent": ordering[0],
        "confidence": scores[ordering[0]],
        "intent_ranking": [
            {"intent": label, "score": round(scores[label], 8)} for label in ordering
        ],
    }


def baseline_predictions(
    *,
    labels: list[dict[str, Any]],
    split: str,
    baseline: str,
    seed: int,
) -> list[dict[str, Any]]:
    train_counts = Counter(
        str(row["closed_intent"]) for row in labels if row.get("split") == "train"
    )
    train_total = sum(train_counts.values())
    if not train_total:
        raise ValueError("Training split is empty")
    priors = {
        label: (train_counts[label] + 1) / (train_total + len(INTENT_LABELS))
        for label in INTENT_LABELS
    }
    target_rows = [row for row in labels if row.get("split") == split]
    predictions: list[dict[str, Any]] = []
    majority_order = sorted(INTENT_LABELS, key=lambda label: (-priors[label], label))
    for row in target_rows:
        sample_id = str(row["sample_id"])
        rng = random.Random(f"{seed}:{sample_id}:{baseline}")
        if baseline == "majority":
            ordering = list(majority_order)
            scores = dict(priors)
        elif baseline == "uniform_random":
            ordering = list(INTENT_LABELS)
            rng.shuffle(ordering)
            scores = {label: 1.0 / len(INTENT_LABELS) for label in INTENT_LABELS}
        elif baseline == "stratified_random":
            first = rng.choices(list(INTENT_LABELS), weights=[priors[label] for label in INTENT_LABELS], k=1)[0]
            remaining = [label for label in INTENT_LABELS if label != first]
            rng.shuffle(remaining)
            ordering = [first, *remaining]
            scores = dict(priors)
            scores[first] = max(scores.values())
        else:
            raise ValueError(f"Unknown baseline: {baseline}")
        predictions.append(_ranked_prediction(sample_id, ordering, scores))
    return predictions


def _prepare_command(args: argparse.Namespace) -> None:
    report = prepare_benchmark(
        labels_dir=Path(args.labels_dir),
        output_dir=Path(args.output_dir),
        protocol=args.protocol,
        seed=args.seed,
        include_repair_review_queue=args.include_repair_review_queue,
        human_review_file=Path(args.human_review_file) if args.human_review_file else None,
        min_eval_per_class=args.min_eval_per_class,
        include_oracle_user_state=args.include_oracle_user_state,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _evaluate_command(args: argparse.Namespace) -> None:
    report = score_predictions(
        labels=read_jsonl(Path(args.labels)),
        predictions=read_jsonl(Path(args.predictions)),
        split=args.split,
    )
    write_json(Path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _baselines_command(args: argparse.Namespace) -> None:
    labels = read_jsonl(Path(args.labels))
    output_dir = Path(args.output_dir)
    reports: dict[str, Any] = {}
    for baseline in ("majority", "uniform_random", "stratified_random"):
        predictions = baseline_predictions(
            labels=labels,
            split=args.split,
            baseline=baseline,
            seed=args.seed,
        )
        write_jsonl(output_dir / f"{baseline}_predictions.jsonl", predictions)
        report = score_predictions(labels=labels, predictions=predictions, split=args.split)
        write_json(output_dir / f"{baseline}_metrics.json", report)
        reports[baseline] = report
    summary = {
        baseline: {
            "top1_accuracy": report["top1_accuracy"],
            "macro_f1_all_six": report["macro_f1_all_six"],
            "top3_accuracy": report["top3_accuracy"],
            "mrr": report["mrr"],
        }
        for baseline, report in reports.items()
    }
    write_json(output_dir / "baseline_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and score the six-class intent benchmark.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Build leak-safe inputs, private labels, and splits.")
    prepare.add_argument("--labels-dir", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--protocol", choices=("unseen_user", "unseen_topic"), required=True)
    prepare.add_argument("--seed", type=int, default=20260727)
    prepare.add_argument("--min-eval-per-class", type=int, default=20)
    prepare.add_argument("--include-repair-review-queue", action="store_true")
    prepare.add_argument("--include-oracle-user-state", action="store_true")
    prepare.add_argument("--human-review-file", default="")
    prepare.set_defaults(handler=_prepare_command)

    evaluate = subparsers.add_parser("evaluate", help="Score model prediction JSONL.")
    evaluate.add_argument("--labels", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--split", choices=("train", "val", "test", "full"), default="test")
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(handler=_evaluate_command)

    baselines = subparsers.add_parser("baselines", help="Run majority and random diagnostic baselines.")
    baselines.add_argument("--labels", required=True)
    baselines.add_argument("--split", choices=("train", "val", "test", "full"), default="test")
    baselines.add_argument("--seed", type=int, default=20260727)
    baselines.add_argument("--output-dir", required=True)
    baselines.set_defaults(handler=_baselines_command)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
