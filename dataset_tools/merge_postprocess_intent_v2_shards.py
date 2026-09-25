from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def shard_dirs(batch_dir: Path, shard_count: int) -> list[Path]:
    target = batch_dir / "postprocessed_intent_v2"
    return [target, *(batch_dir / f"postprocessed_intent_v2_part_{index}" for index in range(1, shard_count))]


def merge_shards(batch_dir: Path, *, shard_count: int, expected_trees_per_shard: int) -> dict[str, Any]:
    directories = shard_dirs(batch_dir, shard_count)
    target = directories[0]
    manifests: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    public_config: dict[str, Any] | None = None

    for index, directory in enumerate(directories):
        summary = read_json(directory / "summary.json")
        if summary.get("dry_run"):
            raise ValueError(f"Shard {index} is a dry-run result")
        if int(summary.get("tree_count", -1)) != expected_trees_per_shard:
            raise ValueError(
                f"Shard {index} has {summary.get('tree_count')} trees; expected {expected_trees_per_shard}"
            )
        shard_config = read_json(directory / "config.public.json")
        if public_config is None:
            public_config = shard_config
        elif shard_config != public_config:
            raise ValueError(f"Shard {index} public config differs from shard 0")
        manifests.extend(read_jsonl(directory / "manifest.jsonl"))
        audits.extend(read_jsonl(directory / "relabel_audit.jsonl"))
        reviews.extend(read_jsonl(directory / "review_queue.jsonl"))

    tree_ids = [str(row["tree_id"]) for row in manifests]
    if len(tree_ids) != len(set(tree_ids)):
        raise ValueError("Duplicate tree_id found across shard manifests")
    edge_ids = [(str(row["tree_id"]), str(row["edge_id"])) for row in audits]
    if len(edge_ids) != len(set(edge_ids)):
        raise ValueError("Duplicate tree_id/edge_id found across shard audits")

    for directory in directories[1:]:
        for relative in (Path("trees"), Path("cache/text"), Path("cache/vision")):
            source = directory / relative
            if source.exists():
                shutil.copytree(source, target / relative, dirs_exist_ok=True)

    for row in manifests:
        if not (target / str(row["output_tree"])).is_file():
            raise FileNotFoundError(f"Merged tree is missing: {row['output_tree']}")

    legacy_counts = Counter(str(row["legacy_closed_intent"]) for row in audits)
    intended_counts = Counter(str(row["intended_closed_intent"]) for row in audits)
    observed_counts = Counter(str(row["observed_visual_intent"]) for row in audits if row.get("observed_visual_intent"))
    summary = {
        "schema_version": "intent_v2_postprocessed",
        "source_batch": str(batch_dir.resolve()),
        "output_dir": str(target.resolve()),
        "stage": "all",
        "dry_run": False,
        "tree_count": len(manifests),
        "edge_count": len(audits),
        "review_count": len(reviews),
        "changed_label_count": sum(bool(row.get("label_changed")) for row in audits),
        "legacy_class_counts": dict(legacy_counts),
        "intended_v2_class_counts": dict(intended_counts),
        "observed_visual_class_counts": dict(observed_counts),
        "images_modified": False,
    }
    write_jsonl(target / "manifest.jsonl", manifests)
    write_jsonl(target / "relabel_audit.jsonl", audits)
    write_jsonl(target / "review_queue.jsonl", reviews)
    write_json(target / "summary.json", summary)
    write_json(
        target / "merge_metadata.json",
        {
            "merged_at": datetime.now(timezone.utc).isoformat(),
            "shards": [str(path.resolve()) for path in directories],
            "shard_count": shard_count,
            "expected_trees_per_shard": expected_trees_per_shard,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge validated Intent V2 post-processing shards.")
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--shard-count", type=int, default=4)
    parser.add_argument("--expected-trees-per-shard", type=int, default=36)
    args = parser.parse_args()
    if args.shard_count < 1 or args.expected_trees_per_shard < 1:
        raise ValueError("Shard count and expected trees per shard must be positive")
    summary = merge_shards(
        Path(args.batch_dir).resolve(),
        shard_count=args.shard_count,
        expected_trees_per_shard=args.expected_trees_per_shard,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
