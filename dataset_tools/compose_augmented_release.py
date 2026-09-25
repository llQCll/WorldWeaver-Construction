from __future__ import annotations

import argparse
import json
import math
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def relative_link(link: Path, target: Path) -> None:
    target = target.resolve()
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != target:
            raise ValueError(f"Link target mismatch: {link} -> {link.resolve()}, expected {target}")
        return
    if link.exists():
        raise ValueError(f"Expected absent path or matching symlink: {link}")
    link.symlink_to(Path("..") / Path(target.relative_to(link.parent.parent.resolve())))


def safe_relative_link(link: Path, target: Path) -> None:
    target = target.resolve()
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != target:
            raise ValueError(f"Link target mismatch: {link} -> {link.resolve()}, expected {target}")
        return
    if link.exists():
        raise ValueError(f"Expected absent path or matching symlink: {link}")
    import os

    link.symlink_to(Path(os.path.relpath(target, start=link.parent)), target_is_directory=True)


def count_tree_labels(tree_path: Path) -> tuple[int, int, Counter[str]]:
    tree = read_json(tree_path)
    nodes = tree.get("nodes", [])
    edges = tree.get("edges", [])
    labels: Counter[str] = Counter()
    for edge in edges:
        label = str(edge.get("closed_intent", ""))
        if label not in INTENT_LABELS:
            raise ValueError(f"Invalid closed_intent {label!r}: {tree_path}")
        labels[label] += 1
    return len(nodes), len(edges), labels


def reveal_additions(total: int, reveal: int, target: float) -> int:
    return max(0, math.ceil((target * total - reveal) / (1.0 - target)))


def compose(args: argparse.Namespace) -> Path:
    base_dir = Path(args.base_dir).resolve()
    augmentation_dir = Path(args.augmentation_dir).resolve()
    rare_batch_dir = Path(args.rare_batch_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    base_summary = read_json(base_dir / "summary.json")
    base_rows = read_jsonl(base_dir / "manifest.jsonl")
    augmentation_rows = read_jsonl(augmentation_dir / "usable_branches.jsonl")
    resume_summary = read_json(rare_batch_dir / "resume_summary.json")

    if len(base_rows) != int(base_summary["tree_count"]):
        raise ValueError("Base manifest count does not match its summary")
    if len(augmentation_rows) != 99:
        raise ValueError(f"Expected 99 accepted augmentation branches; got {len(augmentation_rows)}")
    if resume_summary.get("done") != 10 or resume_summary.get("failed") != 0:
        raise ValueError("Rare-user batch is not complete")

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_relative_link(output_dir / "trees" / "natural72", base_dir / "trees")
    safe_relative_link(output_dir / "augmentations" / "accepted99", augmentation_dir)

    labels = Counter({key: int(value) for key, value in base_summary["class_counts"].items()})
    tree_rows: list[dict[str, Any]] = []
    for row in base_rows:
        source_path = Path(str(row["tree_path"]))
        if source_path.parts[0] != "trees":
            raise ValueError(f"Unexpected base tree path: {source_path}")
        release_path = Path("trees/natural72").joinpath(*source_path.parts[1:])
        tree_rows.append(
            {
                **row,
                "record_type": "tree",
                "release_component": "natural72",
                "tree_path": release_path.as_posix(),
                "image_dir": (release_path.parent / "images").as_posix(),
            }
        )

    augmentation_labels: Counter[str] = Counter()
    augmentation_index: list[dict[str, Any]] = []
    for row in augmentation_rows:
        label = str(row.get("edge", {}).get("closed_intent", row.get("target_intent", "")))
        if label not in INTENT_LABELS:
            raise ValueError(f"Invalid augmentation label {label!r}")
        augmentation_labels[label] += 1
        augmentation_index.append(
            {
                "record_type": "augmentation_branch",
                "release_component": "accepted99",
                "augmentation_id": row["augmentation_id"],
                "source_dataset": row["source_dataset"],
                "user_id": str(row["user_id"]),
                "topic_id": row["topic_id"],
                "tree_id": row["tree_id"],
                "source_node_id": row["source_node_id"],
                "closed_intent": label,
                "source_manifest": "augmentations/accepted99/usable_branches.jsonl",
            }
        )
    labels.update(augmentation_labels)

    rare_labels: Counter[str] = Counter()
    rare_nodes = 0
    rare_edges = 0
    rare_tree_count = 0
    rare_users: set[str] = set()
    for result in resume_summary["results"]:
        user_id = str(result["user_id"])
        user_dir = Path(result["run_dir"]) / "users" / f"user_{user_id}"
        safe_relative_link(output_dir / "trees" / "rare10" / f"user_{user_id}", user_dir)
        tree_files = sorted(user_dir.glob("*/tree.json"))
        if len(tree_files) != 4:
            raise ValueError(f"Expected four trees for user {user_id}; got {len(tree_files)}")
        rare_users.add(user_id)
        for tree_path in tree_files:
            nodes, edges, tree_labels = count_tree_labels(tree_path)
            if nodes != 19 or edges != 18:
                raise ValueError(f"Unexpected rare tree shape {nodes}/{edges}: {tree_path}")
            rare_nodes += nodes
            rare_edges += edges
            rare_tree_count += 1
            rare_labels.update(tree_labels)
            tree = read_json(tree_path)
            relative_tree = Path("trees/rare10") / f"user_{user_id}" / tree_path.parent.name / "tree.json"
            tree_rows.append(
                {
                    "record_type": "tree",
                    "release_component": "rare10",
                    "sample_id": f"rare10:user_{user_id}:{tree_path.parent.name}",
                    "source_dataset": "rare10_natural",
                    "user_id": user_id,
                    "topic_id": tree_path.parent.name,
                    "tree_id": tree.get("tree_id", f"{tree_path.parent.name}_user_{user_id}"),
                    "tree_path": relative_tree.as_posix(),
                    "image_dir": (relative_tree.parent / "images").as_posix(),
                    "edge_count": edges,
                    "label_counts": dict(sorted(tree_labels.items())),
                }
            )
    labels.update(rare_labels)

    write_jsonl(output_dir / "tree_manifest.jsonl", tree_rows)
    write_jsonl(output_dir / "augmentation_manifest.jsonl", augmentation_index)

    edge_count = sum(labels.values())
    class_counts = {label: labels[label] for label in INTENT_LABELS}
    class_percent = {label: round(labels[label] / edge_count * 100, 3) for label in INTENT_LABELS}
    summary = {
        "schema_version": "augmented_release_v1",
        "description": "Repaired natural72 trees plus 99 accepted sidecar branches plus the completed rare10 natural-profile batch.",
        "composition_policy": "Logical release using relative symlinks; source images and trees are not duplicated or mutated.",
        "source_datasets": {
            "natural72": str(base_dir),
            "accepted99": str(augmentation_dir),
            "rare10": str(rare_batch_dir),
        },
        "excluded_datasets": [
            "earlier 50-branch pilot augmentation",
            "mechanically balanced 10-user batch",
            "controlled canary batches",
        ],
        "user_count": int(base_summary["user_count"]) + len(rare_users),
        "tree_count": len(tree_rows),
        "base_tree_edge_count": int(base_summary["edge_count"]),
        "augmentation_edge_count": len(augmentation_rows),
        "rare10_edge_count": rare_edges,
        "edge_count": edge_count,
        "node_count_excluding_sidecar_targets": int(base_summary["node_count"]) + rare_nodes,
        "sidecar_target_node_count": len(augmentation_rows),
        "class_counts": class_counts,
        "class_percent": class_percent,
        "component_class_counts": {
            "natural72": {label: int(base_summary["class_counts"].get(label, 0)) for label in INTENT_LABELS},
            "accepted99": {label: augmentation_labels[label] for label in INTENT_LABELS},
            "rare10": {label: rare_labels[label] for label in INTENT_LABELS},
        },
        "reveal_expansion_targets": {
            f"{int(target * 100)}_percent": reveal_additions(edge_count, labels["reveal"], target)
            for target in (0.12, 0.13, 0.14, 0.15)
        },
        "known_compatibility_notes": base_summary.get("known_compatibility_notes", []),
    }
    write_json(output_dir / "summary.json", summary)

    if len(tree_rows) != 328 or edge_count != 5715:
        raise ValueError(f"Unexpected release totals: trees={len(tree_rows)}, edges={edge_count}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose the repaired natural trees, accepted sidecars, and rare-user trees into one logical release."
    )
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--augmentation-dir", required=True)
    parser.add_argument("--rare-batch-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    print(compose(parse_args()))


if __name__ == "__main__":
    main()
