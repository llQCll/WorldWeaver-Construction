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


def directories(batch_dir: Path, part_count: int) -> list[Path]:
    target = batch_dir / "closed_intent_only_v1"
    return [target, *(batch_dir / f"closed_intent_only_v1_part_{index}" for index in range(1, part_count))]


def merge_parts(batch_dir: Path, *, part_count: int, expected_trees_per_part: int) -> dict[str, Any]:
    part_dirs = directories(batch_dir, part_count)
    target = part_dirs[0]
    manifests: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    api_calls = 0
    reused = 0
    model = ""
    for index, directory in enumerate(part_dirs):
        summary = read_json(directory / "summary.json")
        if int(summary.get("tree_count", -1)) != expected_trees_per_part:
            raise ValueError(
                f"Part {index} has {summary.get('tree_count')} trees; expected {expected_trees_per_part}"
            )
        if summary.get("images_sent") != 0 or summary.get("other_fields_modified") is not False:
            raise ValueError(f"Part {index} is not a valid label-only result")
        if model and summary.get("model") != model:
            raise ValueError(f"Part {index} used a different model")
        model = str(summary.get("model", model))
        api_calls += int(summary.get("api_call_count", 0))
        reused += int(summary.get("reused_full_cache_count", 0))
        manifests.extend(read_jsonl(directory / "manifest.jsonl"))
        audits.extend(read_jsonl(directory / "label_audit.jsonl"))

    tree_ids = [str(row["tree_id"]) for row in manifests]
    edge_ids = [(str(row["tree_id"]), str(row["edge_id"])) for row in audits]
    if len(tree_ids) != len(set(tree_ids)):
        raise ValueError("Duplicate tree_id found across parts")
    if len(edge_ids) != len(set(edge_ids)):
        raise ValueError("Duplicate tree_id/edge_id found across parts")

    for directory in part_dirs[1:]:
        for relative in (Path("trees"), Path("cache/labels")):
            source = directory / relative
            if source.exists():
                shutil.copytree(source, target / relative, dirs_exist_ok=True)
    for row in manifests:
        if not (target / str(row["output_tree"])).is_file():
            raise FileNotFoundError(f"Merged tree is missing: {row['output_tree']}")

    legacy_counts = Counter(str(row["legacy_closed_intent"]) for row in audits)
    repaired_counts = Counter(str(row["repaired_closed_intent"]) for row in audits)
    summary = {
        "schema_version": "closed_intent_only_v1",
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "source_batch": str(batch_dir.resolve()),
        "output_dir": str(target.resolve()),
        "model": model,
        "tree_count": len(manifests),
        "edge_count": len(audits),
        "changed_label_count": sum(bool(row["label_changed"]) for row in audits),
        "legacy_class_counts": dict(legacy_counts),
        "repaired_class_counts": dict(repaired_counts),
        "api_call_count": api_calls,
        "reused_full_cache_count": reused,
        "images_sent": 0,
        "images_modified": False,
        "other_fields_modified": False,
    }
    write_jsonl(target / "manifest.jsonl", manifests)
    write_jsonl(target / "label_audit.jsonl", audits)
    write_json(target / "summary.json", summary)
    write_json(
        target / "merge_metadata.json",
        {
            "merged_at": datetime.now(timezone.utc).isoformat(),
            "parts": [str(path.resolve()) for path in part_dirs],
            "part_count": part_count,
            "expected_trees_per_part": expected_trees_per_part,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge closed-intent-only processing parts.")
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--part-count", type=int, default=24)
    parser.add_argument("--expected-trees-per-part", type=int, default=6)
    args = parser.parse_args()
    summary = merge_parts(
        Path(args.batch_dir).resolve(),
        part_count=args.part_count,
        expected_trees_per_part=args.expected_trees_per_part,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
