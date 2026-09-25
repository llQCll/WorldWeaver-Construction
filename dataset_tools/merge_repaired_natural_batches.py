from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


INTENT_LABELS = {
    "zoom_in",
    "reveal",
    "branch_out",
    "reframe",
    "follow",
    "interact",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def tree_key_from_parts(parts: tuple[str, ...]) -> tuple[str, str] | None:
    for marker in ("users", "trees"):
        if marker not in parts:
            continue
        index = parts.index(marker)
        if index + 3 >= len(parts):
            continue
        user_part, topic_id, filename = parts[index + 1 : index + 4]
        if user_part.startswith("user_") and filename == "tree.json":
            return user_part.removeprefix("user_"), topic_id
    return None


def validate_member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe ZIP member path: {name}")
    return path


def extract_origin_archive(archive_path: Path, extract_root: Path) -> Path:
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            member = validate_member_path(info.filename)
            destination = extract_root.joinpath(*member.parts)
            if info.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if destination.is_file() and destination.stat().st_size == info.file_size:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
    origin_root = extract_root / "origin_dataset"
    if not origin_root.is_dir():
        raise ValueError(f"Expected origin_dataset/ in {archive_path}")
    return origin_root


def load_repaired_zip(archive_path: Path) -> dict[tuple[str, str], tuple[str, dict[str, Any]]]:
    trees: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if not name.endswith("/tree.json"):
                continue
            member = validate_member_path(name)
            key = tree_key_from_parts(member.parts)
            if key is None:
                continue
            if key in trees:
                raise ValueError(f"Duplicate repaired GRe tree key: {key}")
            trees[key] = (name, json.loads(archive.read(name)))
    return trees


def load_gre_origin_trees(origin_root: Path) -> dict[tuple[str, str], Path]:
    trees: dict[tuple[str, str], Path] = {}
    for path in sorted(origin_root.glob("run_*/users/user_*/*/tree.json")):
        key = tree_key_from_parts(path.parts)
        if key is None:
            continue
        if key in trees:
            raise ValueError(f"Duplicate GRe origin tree key: {key}")
        trees[key] = path.resolve()
    return trees


def load_primary_trees(repaired_dir: Path) -> dict[tuple[str, str], tuple[Path, Path]]:
    manifest_path = repaired_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise ValueError(f"Missing primary manifest: {manifest_path}")
    trees: dict[tuple[str, str], tuple[Path, Path]] = {}
    for line in manifest_path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row["user_id"]), str(row["topic_id"])
        repaired_tree = repaired_dir / str(row["output_tree"])
        source_tree = Path(str(row["source_tree"]))
        if key in trees:
            raise ValueError(f"Duplicate primary tree key: {key}")
        if not repaired_tree.is_file() or not source_tree.is_file():
            raise ValueError(f"Missing primary tree for {key}: {repaired_tree} / {source_tree}")
        trees[key] = (repaired_tree.resolve(), source_tree.resolve())
    return trees


def normalize_image_paths(tree: dict[str, Any]) -> None:
    tree["root_image"] = "images/n0.png"
    for node in tree.get("nodes", []):
        node_id = str(node.get("node_id", ""))
        if node_id:
            node["image"] = f"images/{node_id}.png"
    for edge in tree.get("edges", []):
        target_node = str(edge.get("target_node", ""))
        if target_node:
            edge["target_image"] = f"images/{target_node}.png"


def validate_tree(tree: dict[str, Any], *, key: tuple[str, str]) -> Counter[str]:
    nodes = tree.get("nodes", [])
    edges = tree.get("edges", [])
    if len(nodes) != 18 or len(edges) != 17:
        raise ValueError(f"Expected 18 nodes and 17 edges for {key}; got {len(nodes)} / {len(edges)}")
    labels: Counter[str] = Counter()
    for edge in edges:
        label = str(edge.get("closed_intent", ""))
        if label not in INTENT_LABELS:
            raise ValueError(f"Invalid closed_intent {label!r} in {key}")
        labels[label] += 1
    return labels


def link_topic_assets(*, source_topic_dir: Path, output_topic_dir: Path) -> None:
    images = source_topic_dir / "images"
    if not images.is_dir():
        raise ValueError(f"Missing image directory: {images}")
    image_count = sum(1 for path in images.glob("n*.png") if path.is_file())
    if image_count != 18:
        raise ValueError(f"Expected 18 node images in {images}; found {image_count}")
    output_topic_dir.mkdir(parents=True, exist_ok=True)
    image_link = output_topic_dir / "images"
    if image_link.is_symlink() or image_link.exists():
        if image_link.is_symlink() and image_link.resolve() == images.resolve():
            pass
        else:
            raise ValueError(f"Unexpected existing image path: {image_link}")
    else:
        image_link.symlink_to(images.resolve(), target_is_directory=True)
    for filename in ("root_asset.json", "root_prompt.txt", "synopsis.txt"):
        source = source_topic_dir / filename
        target = output_topic_dir / filename
        if not source.is_file() or target.exists() or target.is_symlink():
            continue
        target.symlink_to(source.resolve())


def add_tree(
    *,
    output_dir: Path,
    source_dataset: str,
    repaired_tree: dict[str, Any],
    repaired_tree_source: str,
    source_tree: Path,
    key: tuple[str, str],
) -> tuple[dict[str, Any], Counter[str]]:
    user_id, topic_id = key
    tree = json.loads(json.dumps(repaired_tree))
    labels = validate_tree(tree, key=key)
    normalize_image_paths(tree)
    tree["merge_metadata"] = {
        "source_dataset": source_dataset,
        "source_tree": str(source_tree),
        "repaired_tree_source": repaired_tree_source,
        "authoritative_label_field": "closed_intent",
        "label_schema_version": "closed_intent_only_v1",
        "image_path_mode": "tree_relative",
    }
    output_topic_dir = output_dir / "trees" / source_dataset / f"user_{user_id}" / topic_id
    link_topic_assets(source_topic_dir=source_tree.parent, output_topic_dir=output_topic_dir)
    output_tree = output_topic_dir / "tree.json"
    write_json(output_tree, tree)
    row = {
        "sample_id": f"{source_dataset}:user_{user_id}:{topic_id}",
        "source_dataset": source_dataset,
        "user_id": user_id,
        "topic_id": topic_id,
        "tree_id": str(tree.get("tree_id") or f"{topic_id}_user_{user_id}"),
        "tree_path": output_tree.relative_to(output_dir).as_posix(),
        "image_dir": (output_topic_dir / "images").relative_to(output_dir).as_posix(),
        "edge_count": len(tree.get("edges", [])),
        "label_counts": dict(sorted(labels.items())),
    }
    return row, labels


def merge_batches(args: argparse.Namespace) -> Path:
    primary_dir = Path(args.primary_repaired_dir).resolve()
    gre_origin_zip = Path(args.gre_origin_zip).resolve()
    gre_repaired_zip = Path(args.gre_repaired_zip).resolve()
    gre_extract_dir = Path(args.gre_extract_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    origin_root = extract_origin_archive(gre_origin_zip, gre_extract_dir)
    primary = load_primary_trees(primary_dir)
    gre_origin = load_gre_origin_trees(origin_root)
    gre_repaired = load_repaired_zip(gre_repaired_zip)
    if set(gre_origin) != set(gre_repaired):
        raise ValueError(
            "GRe origin/repaired keys differ: "
            f"origin_only={sorted(set(gre_origin) - set(gre_repaired))[:5]}, "
            f"repaired_only={sorted(set(gre_repaired) - set(gre_origin))[:5]}"
        )

    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory must be empty or absent: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    labels: Counter[str] = Counter()
    users: set[tuple[str, str]] = set()
    topics: set[str] = set()

    for key, (repaired_path, source_path) in sorted(primary.items()):
        row, tree_labels = add_tree(
            output_dir=output_dir,
            source_dataset="last36_repaired",
            repaired_tree=read_json(repaired_path),
            repaired_tree_source=str(repaired_path),
            source_tree=source_path,
            key=key,
        )
        rows.append(row)
        labels.update(tree_labels)
        users.add((row["source_dataset"], row["user_id"]))
        topics.add(row["topic_id"])

    for key, (member_name, repaired_tree) in sorted(gre_repaired.items()):
        source_path = gre_origin[key]
        row, tree_labels = add_tree(
            output_dir=output_dir,
            source_dataset="gre_repaired",
            repaired_tree=repaired_tree,
            repaired_tree_source=f"{gre_repaired_zip}!/{member_name}",
            source_tree=source_path,
            key=key,
        )
        rows.append(row)
        labels.update(tree_labels)
        users.add((row["source_dataset"], row["user_id"]))
        topics.add(row["topic_id"])

    write_jsonl(output_dir / "manifest.jsonl", rows)
    summary = {
        "schema_version": "merged_natural_repaired_v1",
        "description": "Two natural-distribution batches merged with closed_intent_only_v1 labels authoritative; balanced and controlled-canary batches excluded.",
        "source_datasets": {
            "last36_repaired": str(primary_dir),
            "gre_origin_archive": str(gre_origin_zip),
            "gre_repaired_archive": str(gre_repaired_zip),
        },
        "excluded_batches": ["balanced10", "natural_controlled_canary"],
        "user_count": len(users),
        "topic_count": len(topics),
        "tree_count": len(rows),
        "node_count": len(rows) * 18,
        "edge_count": sum(labels.values()),
        "image_count": len(rows) * 18,
        "class_counts": dict(sorted(labels.items())),
        "known_compatibility_notes": [
            "closed_intent is authoritative; preserved intent_ranking may not have the repaired label at Top-1.",
            "natural_language_intent is absent in the source closed-intent-only overlays.",
            "OOS examples and calibrated oos_score are not provided by these source batches.",
        ],
    }
    write_json(output_dir / "summary.json", summary)
    if len(rows) != 288 or len(users) != 72 or sum(labels.values()) != 4896:
        raise ValueError(f"Unexpected merged totals: {summary}")
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge two repaired natural-distribution comic-tree batches.")
    parser.add_argument("--primary-repaired-dir", required=True)
    parser.add_argument("--gre-origin-zip", required=True)
    parser.add_argument("--gre-repaired-zip", required=True)
    parser.add_argument("--gre-extract-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    output_dir = merge_batches(parse_args())
    print(output_dir)


if __name__ == "__main__":
    main()
