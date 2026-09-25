from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
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
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def repo_relative(path: Path) -> str:
    absolute = path if path.is_absolute() else (REPO_ROOT / path)
    absolute = absolute.absolute()
    try:
        return absolute.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return absolute.as_posix()


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


def split_group_counts(group_count: int, val_ratio: float, test_ratio: float) -> dict[str, int]:
    if group_count < 3:
        raise ValueError("At least three users are required for train/val/test splitting")
    val_count = max(1, round(group_count * val_ratio))
    test_count = max(1, round(group_count * test_ratio))
    train_count = group_count - val_count - test_count
    if train_count < 1:
        raise ValueError("Split ratios leave no training users")
    return {"train": train_count, "val": val_count, "test": test_count}


def _split_score(
    assignment: dict[str, str],
    per_user: dict[str, Counter[str]],
    global_counts: Counter[str],
    *,
    min_eval_per_class: int,
) -> float:
    split_counts = {split: Counter() for split in ("train", "val", "test")}
    split_totals = Counter()
    for user_id, split in assignment.items():
        split_counts[split].update(per_user[user_id])
        split_totals[split] += sum(per_user[user_id].values())

    total = sum(global_counts.values())
    global_distribution = {label: global_counts[label] / total for label in INTENT_LABELS}
    score = 0.0
    for split in ("train", "val", "test"):
        split_total = split_totals[split]
        for label in INTENT_LABELS:
            observed = split_counts[split][label] / split_total
            expected = global_distribution[label]
            score += ((observed - expected) ** 2) / max(expected, 0.01)

    for split in ("val", "test"):
        for label in INTENT_LABELS:
            shortfall = max(0, min_eval_per_class - split_counts[split][label])
            score += shortfall * shortfall * 10.0
    return score


def stratified_user_split(
    samples: list[dict[str, Any]],
    *,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    search_trials: int,
    min_eval_per_class: int,
) -> dict[str, str]:
    per_user: dict[str, Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    for sample in samples:
        user_id = str(sample["user_id"])
        label = str(sample["closed_intent"])
        per_user[user_id][label] += 1
        global_counts[label] += 1

    users = sorted(per_user)
    counts = split_group_counts(len(users), val_ratio, test_ratio)
    rng = random.Random(seed)
    best_assignment: dict[str, str] | None = None
    best_score = float("inf")
    for _ in range(search_trials):
        ordered = users.copy()
        rng.shuffle(ordered)
        test_end = counts["test"]
        val_end = test_end + counts["val"]
        assignment = {
            user_id: (
                "test" if index < test_end else "val" if index < val_end else "train"
            )
            for index, user_id in enumerate(ordered)
        }
        score = _split_score(
            assignment,
            per_user,
            global_counts,
            min_eval_per_class=min_eval_per_class,
        )
        if score < best_score:
            best_assignment = assignment
            best_score = score
    if best_assignment is None:
        raise RuntimeError("Unable to construct a user split")
    return best_assignment


def source_image_for_node(tree_path: Path, source_node: str, node: dict[str, Any]) -> Path:
    expected = tree_path.parent / "images" / f"{source_node}.png"
    if expected.is_file():
        return expected
    raw = str(node.get("image", "")).replace("\\", "/")
    return tree_path.parent / "images" / Path(raw).name


def sample_from_edge(
    *,
    component: str,
    tree_path: Path,
    tree: dict[str, Any],
    edge: dict[str, Any],
    source_manifest: Path,
    augmentation_id: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    nodes = {str(node["node_id"]): node for node in tree.get("nodes", [])}
    source_node = str(edge.get("source_node", ""))
    node = nodes.get(source_node)
    if node is None:
        return None, {"reason": "missing_source_node", "source_node": source_node}
    label = str(edge.get("closed_intent", "")).strip().lower()
    if label not in INTENT_SET:
        return None, {"reason": "invalid_closed_intent", "closed_intent": label}
    box = normalized_box(edge.get("grounding", {}).get("target_box"))
    if box is None:
        return None, {"reason": "invalid_target_box"}
    source_image = source_image_for_node(tree_path, source_node, node)
    if not source_image.is_file():
        return None, {"reason": "missing_source_image", "path": str(source_image)}

    tree_id = str(tree["tree_id"])
    edge_id = str(edge.get("edge_id", augmentation_id or ""))
    sample_id = f"augmentation:{augmentation_id}" if augmentation_id else f"tree:{tree_id}/{edge_id}"
    grounding = edge.get("grounding", {})
    sample = {
        "sample_id": sample_id,
        "component": component,
        "record_type": "augmentation_branch" if augmentation_id else "tree_edge",
        "user_id": str(tree["user_id"]),
        "topic_id": str(tree["topic_id"]),
        "tree_id": tree_id,
        "edge_id": edge_id,
        "augmentation_id": augmentation_id,
        "source_node": source_node,
        "target_node": str(edge.get("target_node", "")),
        "source_tree": repo_relative(tree_path),
        "source_manifest": repo_relative(source_manifest),
        "source_image": repo_relative(source_image),
        "target_image": str(edge.get("target_image", "")),
        "target_box": box,
        "click_center": [round((box[0] + box[2]) / 2, 6), round((box[1] + box[3]) / 2, 6)],
        "story_state_before": node.get("story_state", {}),
        "oracle_user_state_before": {
            "stable_profile": node.get("stable_profile", {}),
            "affective_state": node.get("affective_state", {}),
            "current_state": node.get("current_state", {}),
            "profile_summary": node.get("profile_summary", ""),
        },
        "closed_intent": label,
        "grounding_target_label": str(grounding.get("target_label", "")),
        "natural_language_intent": str(edge.get("natural_language_intent", "")),
        "slots": edge.get("slots", {}),
        "visual_group_id": (
            f"topic:{tree['topic_id']}:shared_root"
            if source_node == "n0"
            else f"tree:{tree_id}:node:{source_node}"
        ),
    }
    return sample, None


def collect_samples(
    base_release: Path,
    augmentation_dirs: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    tree_manifest_path = base_release / "tree_manifest.jsonl"
    tree_rows = read_jsonl(tree_manifest_path)
    tree_lookup: dict[tuple[str, str, str], tuple[Path, dict[str, Any]]] = {}
    samples: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    source_inventory: list[dict[str, Any]] = []

    for row in tree_rows:
        tree_path = base_release / str(row["tree_path"])
        tree = read_json(tree_path)
        key = (str(tree["user_id"]), str(tree["topic_id"]), str(tree["tree_id"]))
        if key in tree_lookup:
            raise ValueError(f"Duplicate tree identity: {key}")
        tree_lookup[key] = (tree_path, tree)
        for edge in tree.get("edges", []):
            sample, error = sample_from_edge(
                component=str(row.get("release_component", "base_tree")),
                tree_path=tree_path,
                tree=tree,
                edge=edge,
                source_manifest=tree_manifest_path,
            )
            if error:
                invalid.append({**error, "tree_id": tree["tree_id"], "edge_id": edge.get("edge_id")})
            elif sample:
                samples.append(sample)
    source_inventory.append(
        {
            "component": "base_trees",
            "path": repo_relative(tree_manifest_path),
            "sample_count": len(samples),
        }
    )

    sidecar_paths = [base_release / "augmentations" / "accepted99" / "usable_branches.jsonl"]
    sidecar_paths.extend(path / "usable_branches.jsonl" for path in augmentation_dirs)
    seen_augmentation_ids: set[str] = set()
    for sidecar_path in sidecar_paths:
        before = len(samples)
        for row in read_jsonl(sidecar_path):
            augmentation_id = str(row["augmentation_id"])
            if augmentation_id in seen_augmentation_ids:
                raise ValueError(f"Duplicate augmentation_id: {augmentation_id}")
            seen_augmentation_ids.add(augmentation_id)
            key = (str(row["user_id"]), str(row["topic_id"]), str(row["tree_id"]))
            tree_entry = tree_lookup.get(key)
            if tree_entry is None:
                invalid.append(
                    {
                        "reason": "missing_source_tree",
                        "augmentation_id": augmentation_id,
                        "tree_identity": list(key),
                    }
                )
                continue
            tree_path, tree = tree_entry
            edge = row.get("edge", {})
            sample, error = sample_from_edge(
                component=sidecar_path.parent.name,
                tree_path=tree_path,
                tree=tree,
                edge=edge,
                source_manifest=sidecar_path,
                augmentation_id=augmentation_id,
            )
            if error:
                invalid.append({**error, "augmentation_id": augmentation_id})
            elif sample:
                samples.append(sample)
        source_inventory.append(
            {
                "component": sidecar_path.parent.name,
                "path": repo_relative(sidecar_path),
                "sample_count": len(samples) - before,
            }
        )

    sample_ids = [str(row["sample_id"]) for row in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample_id values found")
    return samples, invalid, source_inventory


def build_split(args: argparse.Namespace) -> Path:
    base_release = Path(args.base_release).resolve()
    augmentation_dirs = [Path(path).resolve() for path in args.augmentation_dir]
    output_dir = Path(args.output_dir).resolve()
    samples, invalid, source_inventory = collect_samples(base_release, augmentation_dirs)
    if invalid and not args.allow_invalid:
        raise ValueError(f"Found {len(invalid)} invalid samples; rerun with --allow-invalid to exclude them")

    assignment = stratified_user_split(
        samples,
        seed=args.seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        search_trials=args.search_trials,
        min_eval_per_class=args.min_eval_per_class,
    )
    for sample in samples:
        sample["split"] = assignment[str(sample["user_id"])]

    samples.sort(key=lambda row: (row["split"], row["user_id"], row["sample_id"]))
    inputs = []
    labels = []
    for sample in samples:
        inputs.append(
            {
                "sample_id": sample["sample_id"],
                "split": sample["split"],
                "protocol": "unseen_user",
                "component": sample["component"],
                "record_type": sample["record_type"],
                "user_id": sample["user_id"],
                "topic_id": sample["topic_id"],
                "tree_id": sample["tree_id"],
                "edge_id": sample["edge_id"],
                "source_node": sample["source_node"],
                "source_image": sample["source_image"],
                "interaction": {
                    "type": "box",
                    "target_box": sample["target_box"],
                    "click_center": sample["click_center"],
                },
                "context": {
                    "story_state_before": sample["story_state_before"],
                    "oracle_user_state_before": sample["oracle_user_state_before"],
                },
                "visual_group_id": sample["visual_group_id"],
            }
        )
        labels.append(
            {
                "sample_id": sample["sample_id"],
                "split": sample["split"],
                "closed_intent": sample["closed_intent"],
                "grounding_target_label": sample["grounding_target_label"],
                "natural_language_intent": sample["natural_language_intent"],
                "slots": sample["slots"],
            }
        )

    split_counts = {
        split: Counter(row["closed_intent"] for row in samples if row["split"] == split)
        for split in ("train", "val", "test")
    }
    split_sizes = Counter(row["split"] for row in samples)
    split_users = {
        split: sorted(user for user, assigned in assignment.items() if assigned == split)
        for split in ("train", "val", "test")
    }
    global_counts = Counter(row["closed_intent"] for row in samples)
    underfilled = {
        split: [
            label for label in INTENT_LABELS if split_counts[split][label] < args.min_eval_per_class
        ]
        for split in ("val", "test")
    }
    stats = {
        "schema_version": "augmented_release_unseen_user_split_v1",
        "protocol": "unseen_user",
        "seed": args.seed,
        "split_ratios": {
            "train": round(1.0 - args.val_ratio - args.test_ratio, 6),
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "sample_count": len(samples),
        "user_count": len(assignment),
        "tree_count": len(read_jsonl(base_release / "tree_manifest.jsonl")),
        "class_counts": {label: global_counts[label] for label in INTENT_LABELS},
        "split_sample_counts": {split: split_sizes[split] for split in ("train", "val", "test")},
        "split_user_counts": {split: len(split_users[split]) for split in ("train", "val", "test")},
        "split_class_counts": {
            split: {label: split_counts[split][label] for label in INTENT_LABELS}
            for split in ("train", "val", "test")
        },
        "split_users": split_users,
        "min_eval_per_class": args.min_eval_per_class,
        "underfilled_eval_classes": underfilled,
        "invalid_excluded_count": len(invalid),
        "source_inventory": source_inventory,
        "leakage_checks": {
            "user_disjoint": True,
            "sample_id_unique": True,
            "source_tree_and_sidecars_share_user_split": True,
            "shared_topic_root_images_may_cross_splits": True,
        },
        "forbidden_model_inputs": [
            "closed_intent",
            "natural_language_intent",
            "slots",
            "target_image",
            "generation_prompt",
            "story_state_after",
            "intent_ranking",
        ],
    }

    write_jsonl(output_dir / "inputs.jsonl", inputs)
    write_jsonl(output_dir / "labels.jsonl", labels)
    write_json(output_dir / "split_stats.json", stats)
    write_json(output_dir / "user_split.json", assignment)
    write_json(output_dir / "invalid_samples.json", invalid)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic, user-disjoint split over a logical release and sidecar branches."
    )
    parser.add_argument("--base-release", required=True)
    parser.add_argument("--augmentation-dir", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--search-trials", type=int, default=20000)
    parser.add_argument("--min-eval-per-class", type=int, default=20)
    parser.add_argument("--allow-invalid", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(build_split(parse_args()))


if __name__ == "__main__":
    main()
