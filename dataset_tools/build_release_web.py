from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps


MODEL_NAMES = {
    "Qwen3_VL_4B": "Qwen3-VL-4B",
    "Qwen3_VL_8B": "Qwen3-VL-8B",
    "Qwen3_VL_32B": "Qwen3-VL-32B",
    "Ministral_3_8B": "Ministral-3-8B",
    "InternVL3_5_30B": "InternVL3.5-30B",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_image(path_value: str, project_root: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    path.relative_to(project_root.resolve())
    if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ValueError(f"Invalid release image: {path}")
    return path


def make_preview(source: Path, target: Path) -> None:
    if target.is_file() and target.stat().st_mtime >= source.stat().st_mtime:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened)
        image.thumbnail((800, 800), Image.Resampling.LANCZOS)
        image.convert("RGB").save(target, "JPEG", quality=76, optimize=True)


def source_preview(source: Path, web_dir: Path) -> str:
    key = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:16]
    target = web_dir / "assets" / "complete6778" / "source" / f"{key}.jpg"
    make_preview(source, target)
    return target.relative_to(web_dir).as_posix()


def target_preview(source: Path, augmentation_id: str, web_dir: Path) -> str:
    safe_id = "".join(character if character.isalnum() or character in "-_" else "_" for character in augmentation_id)
    target = web_dir / "assets" / "complete6778" / "target" / f"{safe_id}.jpg"
    make_preview(source, target)
    return target.relative_to(web_dir).as_posix()


def build_release_data(split_dir: Path, project_root: Path, web_dir: Path) -> dict[str, Any]:
    stats = read_json(split_dir / "split_stats.json")
    inputs = {row["sample_id"]: row for row in read_jsonl(split_dir / "inputs.jsonl")}
    labels = {row["sample_id"]: row for row in read_jsonl(split_dir / "labels.jsonl")}
    records: list[dict[str, Any]] = []

    for inventory in stats["source_inventory"]:
        if inventory["component"] == "base_trees":
            continue
        manifest_path = project_root / inventory["path"]
        for raw in read_jsonl(manifest_path):
            augmentation_id = str(raw["augmentation_id"])
            sample_id = f"augmentation:{augmentation_id}"
            input_row = inputs[sample_id]
            label_row = labels[sample_id]
            edge = raw.get("edge", {})
            target_node = raw.get("target_node", {})
            source_path = resolve_image(str(input_row["source_image"]), project_root)
            target_path = resolve_image(str(target_node.get("image") or edge.get("target_image")), project_root)
            alignment = edge.get("delta_alignment", {})
            records.append(
                {
                    "id": augmentation_id,
                    "component": str(inventory["component"]),
                    "split": str(input_row["split"]),
                    "user_id": str(raw["user_id"]),
                    "topic_id": str(raw["topic_id"]),
                    "tree_id": str(raw["tree_id"]),
                    "source_node_id": str(raw["source_node_id"]),
                    "target_node_id": str(target_node.get("node_id", edge.get("target_node", ""))),
                    "intent": str(label_row["closed_intent"]),
                    "natural_language_intent": str(label_row.get("natural_language_intent", "")),
                    "target_label": str(label_row.get("grounding_target_label", "")),
                    "slots": label_row.get("slots", {}),
                    "source_image": source_preview(source_path, web_dir),
                    "target_image": target_preview(target_path, augmentation_id, web_dir),
                    "image_intent_alignment": alignment.get("image_intent_alignment"),
                    "continuity_alignment": alignment.get("continuity_alignment"),
                    "profile_affordance_score": raw.get("profile_affordance_joint_score"),
                }
            )

    records.sort(key=lambda row: (row["component"], row["user_id"], row["topic_id"], row["id"]))
    counts = Counter(row["intent"] for row in records)
    component_counts = Counter(row["component"] for row in records)
    return {
        "schema_version": "wordweaver_release_web_v1",
        "release_date": "2026-08-20",
        "dataset": {
            "sample_count": stats["sample_count"],
            "user_count": stats["user_count"],
            "tree_count": stats["tree_count"],
            "tree_edge_count": stats["source_inventory"][0]["sample_count"],
            "augmentation_count": len(records),
            "class_counts": stats["class_counts"],
            "split_sample_counts": stats["split_sample_counts"],
            "split_user_counts": stats["split_user_counts"],
            "augmentation_class_counts": dict(sorted(counts.items())),
            "augmentation_component_counts": dict(sorted(component_counts.items())),
        },
        "records": records,
    }


def build_result_data(split_dir: Path, run_root: Path) -> dict[str, Any]:
    stats = read_json(split_dir / "split_stats.json")
    runs: list[dict[str, Any]] = []
    for model_key, display_name in MODEL_NAMES.items():
        for setting in ("A", "B", "D"):
            run_dir = run_root / model_key / f"setting_{setting}"
            metrics = read_json(run_dir / "metrics.json")
            summary = read_json(run_dir / "run_summary.json")
            runs.append(
                {
                    "model": display_name,
                    "model_key": model_key,
                    "setting": setting,
                    "sample_count": metrics["sample_count"],
                    "top1_accuracy": metrics["top1_accuracy"],
                    "macro_f1": metrics["macro_f1_all_six"],
                    "balanced_accuracy": metrics["macro_recall_balanced_accuracy"],
                    "top3_accuracy": metrics["top3_accuracy"],
                    "mrr": metrics["mrr"],
                    "ece": metrics["ece_10_bin"],
                    "failed": int(summary.get("failed", 0)),
                    "per_class": metrics["per_class"],
                }
            )
    diagnostics = read_json(run_root / "diagnostic_baselines" / "baseline_summary.json")
    return {
        "schema_version": "wordweaver_complete6778_results_v1",
        "evaluation_date": "2026-08-20",
        "protocol": "unseen_user",
        "seed": stats["seed"],
        "dataset": {
            "total_samples": stats["sample_count"],
            "test_samples": stats["split_sample_counts"]["test"],
            "test_users": stats["split_user_counts"]["test"],
            "test_class_counts": stats["split_class_counts"]["test"],
        },
        "run_count": len(runs),
        "prediction_count": sum(row["sample_count"] for row in runs),
        "failure_count": sum(row["failed"] for row in runs),
        "runs": runs,
        "diagnostic_baselines": diagnostics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build lightweight Wordweaver release and benchmark web data.")
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--web-dir", required=True)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parent.parent))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_dir = Path(args.split_dir).resolve()
    run_root = Path(args.run_root).resolve()
    web_dir = Path(args.web_dir).resolve()
    project_root = Path(args.project_root).resolve()
    release = build_release_data(split_dir, project_root, web_dir)
    write_json(web_dir / "complete6778-release-data.json", release)
    write_json(web_dir / "complete6778-test20-results.json", build_result_data(split_dir, run_root))
    print(f"Exported {len(release['records'])} augmentation records to {web_dir}")


if __name__ == "__main__":
    main()
