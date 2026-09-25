from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset_pipeline import (  # noqa: E402
    DatasetPipeline,
    PipelineConfig,
    public_config_snapshot,
    write_json,
    write_jsonl,
)
from select_profile_grounded_augmentations import (  # noqa: E402
    SELECTION_ORDER,
    heuristic_affordance,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def load_config(path: Path) -> PipelineConfig:
    data = read_json(path)
    allowed = set(PipelineConfig.__dataclass_fields__)
    return PipelineConfig(**{key: value for key, value in data.items() if key in allowed})


def resolve_node_image(tree_path: Path, node: dict[str, Any]) -> Path:
    node_id = str(node["node_id"])
    image = Path(str(node.get("image", f"images/{node_id}.png")))
    path = image if image.is_absolute() else tree_path.parent / image
    if path.is_file():
        return path.resolve()
    fallback = tree_path.parent / "images" / f"{node_id}.png"
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(f"Missing source image for {tree_path}:{node_id}")


def source_node_options(
    *,
    candidate: dict[str, Any],
    tree: dict[str, Any],
    tree_path: Path,
    max_attempts: int,
) -> list[dict[str, Any]]:
    target_intent = str(candidate["target_intent"])
    preferred_id = str(candidate["source_node_id"])
    ranked = sorted(
        tree.get("nodes", []),
        key=lambda node: (
            0 if str(node["node_id"]) == preferred_id else 1,
            -heuristic_affordance(label=target_intent, tree=tree, node=node),
            int(node.get("depth", 0)),
            str(node["node_id"]),
        ),
    )
    options: list[dict[str, Any]] = []
    for node in ranked[: max(1, max_attempts)]:
        enriched = dict(node)
        enriched["image"] = str(resolve_node_image(tree_path, node))
        options.append(enriched)
    return options


def topic_spec_from_tree(tree: dict[str, Any]) -> dict[str, Any]:
    topic_id = str(tree["topic_id"])
    return {
        "topic_id": topic_id,
        "topic": topic_id.replace("_", " ").title(),
        "world_bible": tree.get("world_bible", {}),
    }


def select_canary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for label in SELECTION_ORDER:
        matching = [row for row in rows if row.get("target_intent") == label]
        if not matching:
            continue
        selected.append(
            sorted(
                matching,
                key=lambda row: (
                    -float(row.get("profile_affordance_joint_score", 0.0)),
                    str(row["augmentation_id"]),
                ),
            )[0]
        )
    return selected


def portable_record(
    *,
    candidate: dict[str, Any],
    source_root: Path,
    tree_path: Path,
    source_node: dict[str, Any],
) -> dict[str, Any]:
    try:
        source_tree_relative = str(tree_path.resolve().relative_to(source_root.resolve()))
    except ValueError:
        source_tree_relative = str(tree_path)
    return {
        "schema_version": "profile_grounded_minority_augmentation_v1",
        "augmentation_id": candidate["augmentation_id"],
        "source_dataset": candidate["source_dataset"],
        "user_id": candidate["user_id"],
        "cohort_intent": candidate["cohort_intent"],
        "target_intent": candidate["target_intent"],
        "intent_propensity": candidate["intent_propensity"],
        "cohort_intent_propensity": candidate["cohort_intent_propensity"],
        "profile_affordance_joint_score": candidate["profile_affordance_joint_score"],
        "tree_id": candidate["tree_id"],
        "topic_id": candidate["topic_id"],
        "source_tree_relative": source_tree_relative,
        "source_node_id": source_node["node_id"],
        "source_node_image": str(Path(str(source_node["image"])).name),
        "median_stable_profile": candidate["median_stable_profile"],
        "allocation_policy": candidate["allocation_policy"],
    }


def accepted_visual_calibration(
    calibration: dict[str, Any],
    *,
    min_visual_alignment: float,
    min_continuity_alignment: float,
) -> bool:
    values = calibration["delta_alignment"]
    return bool(
        calibration["calibration"]["intent_matches"]
        and float(values["image_intent_alignment"]) >= min_visual_alignment
        and float(values["continuity_alignment"]) >= min_continuity_alignment
    )


def process_candidate(
    *,
    candidate: dict[str, Any],
    source_root: Path,
    output_dir: Path,
    pipeline: DatasetPipeline,
    min_affordance: float,
    max_node_attempts: int,
    max_image_attempts: int,
    min_visual_alignment: float,
    min_continuity_alignment: float,
    generate_images: bool,
    calibrate: bool,
) -> dict[str, Any]:
    tree_path = Path(str(candidate["source_tree"]))
    tree = read_json(tree_path)
    topic_spec = topic_spec_from_tree(tree)
    target_intent = str(candidate["target_intent"])
    affordance_attempts: list[dict[str, Any]] = []
    chosen_node: dict[str, Any] | None = None
    chosen_assessment: dict[str, Any] | None = None

    for source_node in source_node_options(
        candidate=candidate,
        tree=tree,
        tree_path=tree_path,
        max_attempts=max_node_attempts,
    ):
        assessment = pipeline.assess_intent_affordance(
            tree_id=str(tree["tree_id"]),
            topic_spec=topic_spec,
            source_node=source_node,
        )
        target_score = float(assessment["node_affordance"][target_intent])
        affordance_attempts.append(
            {
                "source_node_id": source_node["node_id"],
                "node_affordance": assessment["node_affordance"],
                "target_intent_score": target_score,
                "target_intent_rationale": assessment["rationale"].get(target_intent, ""),
                "passed": target_score >= min_affordance,
            }
        )
        if target_score >= min_affordance:
            chosen_node = source_node
            chosen_assessment = assessment
            break

    reference_node = chosen_node or source_node_options(
        candidate=candidate,
        tree=tree,
        tree_path=tree_path,
        max_attempts=1,
    )[0]
    record = portable_record(
        candidate=candidate,
        source_root=source_root,
        tree_path=tree_path,
        source_node=reference_node,
    )
    record["thresholds"] = {
        "min_node_affordance": min_affordance,
        "min_visual_alignment": min_visual_alignment,
        "min_continuity_alignment": min_continuity_alignment,
    }
    record["affordance_attempts"] = affordance_attempts

    if chosen_node is None or chosen_assessment is None:
        record["status"] = "rejected_low_affordance"
        return record

    digest = hashlib.sha1(str(candidate["augmentation_id"]).encode("utf-8")).hexdigest()[:12]
    target_id = f"aug_{digest}"
    profile = {
        "user_id": str(candidate["user_id"]),
        "stable_profile": chosen_node["stable_profile"],
    }
    plan = pipeline.plan_branches(
        tree_id=str(tree["tree_id"]),
        topic_spec=topic_spec,
        profile=profile,
        source_node=chosen_node,
        target_ids=[target_id],
        forced_intents=[target_intent],
        affordance_assessment=chosen_assessment,
    )
    edge = plan["edges"][0]
    edge["edge_id"] = f"{chosen_node['node_id']}_to_{target_id}"
    edge["source_node"] = chosen_node["node_id"]
    edge["target_node"] = target_id
    edge["augmentation_id"] = candidate["augmentation_id"]
    target_node = pipeline.realize_target_node(
        tree_id=str(tree["tree_id"]),
        topic_spec=topic_spec,
        source_node=chosen_node,
        target_id=target_id,
        edge=edge,
        run_dir=output_dir,
        generate_images=False,
    )
    record["affordance_assessment"] = chosen_assessment
    record["edge"] = edge
    record["oos_region"] = plan["oos_region"]
    record["target_node"] = target_node
    record["image_attempts"] = []

    if not generate_images:
        record["status"] = "planned"
        return record

    base_prompt = str(edge["generation_prompt"])
    revision = ""
    for attempt in range(1, max(1, max_image_attempts) + 1):
        image_path = (
            output_dir
            / "images"
            / str(candidate["source_dataset"])
            / f"user_{candidate['user_id']}"
            / str(candidate["topic_id"])
            / f"{candidate['augmentation_id']}_v{attempt}.png"
        )
        prompt = base_prompt
        if revision:
            prompt += (
                "\n\nTargeted revision after visual calibration: "
                + revision
                + f"\nThe resulting transformation must remain unambiguously {target_intent}."
            )
        pipeline.maybe_generate_image(
            prompt=prompt,
            output_path=image_path,
            references=[Path(str(chosen_node["image"]))],
        )
        target_node["image"] = str(image_path.resolve())
        edge["target_image"] = target_node["image"]
        attempt_record: dict[str, Any] = {
            "attempt": attempt,
            "image_relative": str(image_path.relative_to(output_dir)),
            "generation_prompt": prompt,
        }
        if not calibrate:
            attempt_record["accepted"] = True
            record["image_attempts"].append(attempt_record)
            record["status"] = "generated_uncalibrated"
            break

        calibration = pipeline.calibrate_edge(
            edge=edge,
            source_node=chosen_node,
            target_node=target_node,
        )
        edge.update(calibration)
        accepted = accepted_visual_calibration(
            calibration,
            min_visual_alignment=min_visual_alignment,
            min_continuity_alignment=min_continuity_alignment,
        )
        attempt_record["calibration"] = calibration
        attempt_record["accepted"] = accepted
        record["image_attempts"].append(attempt_record)
        if accepted:
            record["status"] = "accepted"
            break
        revision = (
            calibration["calibration"].get("revision_suggestion")
            or calibration["calibration"].get("visual_notes")
            or "Increase visual evidence for the assigned intent while preserving scene continuity."
        )
    else:
        record["status"] = "rejected_visual_calibration"

    record["edge"] = edge
    record["target_node"] = target_node
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add profile-grounded minority-intent branches as a non-destructive sidecar dataset."
    )
    parser.add_argument("--plan-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-json", default="")
    parser.add_argument("--canary", action="store_true", help="Run one strongest candidate per non-interact label.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-affordance", type=float, default=0.65)
    parser.add_argument("--max-node-attempts", type=int, default=3)
    parser.add_argument("--max-image-attempts", type=int, default=2)
    parser.add_argument("--min-visual-alignment", type=float, default=0.65)
    parser.add_argument("--min-continuity-alignment", type=float, default=0.65)
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan_dir = Path(args.plan_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_summary = read_json(plan_dir / "summary.json")
    source_root = Path(str(plan_summary["source_merged_dataset"])).resolve()
    rows = read_jsonl(plan_dir / "candidate_branches.jsonl")
    if args.canary:
        rows = select_canary(rows)
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]

    if args.config_json:
        config = load_config(Path(args.config_json).resolve())
    else:
        config = PipelineConfig()
    write_json(
        output_dir / "run_config.json",
        {
            "schema_version": "profile_grounded_minority_augmentation_run_v1",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "plan_dir": str(plan_dir),
            "source_merged_dataset": str(source_root),
            "canary": args.canary,
            "candidate_count": len(rows),
            "workers": max(1, args.workers),
            "generate_images": not args.skip_images,
            "calibrate": not args.skip_images and not args.skip_calibration,
            "thresholds": {
                "min_node_affordance": args.min_affordance,
                "min_visual_alignment": args.min_visual_alignment,
                "min_continuity_alignment": args.min_continuity_alignment,
            },
            "models": public_config_snapshot(config),
            "old_trees_mutated": False,
        },
    )

    record_by_index: dict[int, dict[str, Any]] = {}
    pending: list[tuple[int, dict[str, Any]]] = []
    for index, candidate in enumerate(rows, 1):
        record_path = output_dir / "records" / f"{candidate['augmentation_id']}.json"
        if record_path.is_file() and not args.overwrite:
            record = read_json(record_path)
            if record.get("status") != "error":
                record_by_index[index] = record
                print(
                    f"[{index}/{len(rows)}] resume {candidate['augmentation_id']}: {record['status']}",
                    flush=True,
                )
                continue
        pending.append((index, candidate))

    def execute(item: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
        index, candidate = item
        record_path = output_dir / "records" / f"{candidate['augmentation_id']}.json"
        print(
            f"[{index}/{len(rows)}] {candidate['augmentation_id']} "
            f"cohort={candidate['cohort_intent']} target={candidate['target_intent']}",
            flush=True,
        )
        local_pipeline = DatasetPipeline(
            config=config,
            output_dir=output_dir,
            dry_run=args.dry_run,
        )
        try:
            record = process_candidate(
                candidate=candidate,
                source_root=source_root,
                output_dir=output_dir,
                pipeline=local_pipeline,
                min_affordance=args.min_affordance,
                max_node_attempts=args.max_node_attempts,
                max_image_attempts=args.max_image_attempts,
                min_visual_alignment=args.min_visual_alignment,
                min_continuity_alignment=args.min_continuity_alignment,
                generate_images=not args.skip_images,
                calibrate=not args.skip_calibration,
            )
        except Exception as exc:
            record = {
                "schema_version": "profile_grounded_minority_augmentation_v1",
                "augmentation_id": candidate["augmentation_id"],
                "source_dataset": candidate["source_dataset"],
                "user_id": candidate["user_id"],
                "cohort_intent": candidate["cohort_intent"],
                "target_intent": candidate["target_intent"],
                "tree_id": candidate["tree_id"],
                "topic_id": candidate["topic_id"],
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        write_json(record_path, record)
        print(f"[{index}/{len(rows)}] status={record['status']}", flush=True)
        return index, record

    workers = max(1, int(args.workers))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(execute, item) for item in pending]
        for future in concurrent.futures.as_completed(futures):
            index, record = future.result()
            record_by_index[index] = record
            write_jsonl(
                output_dir / "manifest.jsonl",
                [record_by_index[key] for key in sorted(record_by_index)],
            )

    records = [record_by_index[index] for index in range(1, len(rows) + 1)]
    write_jsonl(output_dir / "manifest.jsonl", records)

    status_counts = Counter(str(record["status"]) for record in records)
    planned_counts = Counter(str(record["target_intent"]) for record in records)
    accepted_counts = Counter(
        str(record["target_intent"])
        for record in records
        if record["status"] in {"accepted", "generated_uncalibrated", "planned"}
    )
    summary = {
        "schema_version": "profile_grounded_minority_augmentation_summary_v1",
        "source_merged_dataset": str(source_root),
        "processed_count": len(records),
        "status_counts": dict(sorted(status_counts.items())),
        "planned_intent_counts": dict(sorted(planned_counts.items())),
        "usable_intent_counts": dict(sorted(accepted_counts.items())),
        "old_trees_mutated": False,
    }
    write_json(output_dir / "summary.json", summary)
    write_jsonl(
        output_dir / "usable_branches.jsonl",
        [
            record
            for record in records
            if record["status"] in {"accepted", "generated_uncalibrated", "planned"}
        ],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
