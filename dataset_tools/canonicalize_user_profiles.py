from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PERSONALITY_DIMS = (
    "goal_progress",
    "mastery_logic",
    "challenge_seeking",
    "social_attachment",
    "cooperative_orientation",
    "world_discovery",
    "role_immersion",
    "aesthetic_customization",
)
AFFECT_DIMS = (
    "pleasure",
    "arousal",
    "dominance",
    "tension",
    "curiosity",
    "empathy",
    "cognitive_load",
)
AFFECT_TO_PERSONALITY = {
    "goal_progress": {"dominance": 0.24, "arousal": 0.08, "cognitive_load": -0.12},
    "mastery_logic": {"curiosity": 0.22, "dominance": 0.12, "cognitive_load": 0.08},
    "challenge_seeking": {"arousal": 0.20, "tension": 0.16, "pleasure": 0.06},
    "social_attachment": {"empathy": 0.26, "pleasure": 0.12},
    "cooperative_orientation": {"empathy": 0.22, "dominance": 0.10, "tension": -0.08},
    "world_discovery": {"curiosity": 0.24, "arousal": 0.08, "cognitive_load": -0.06},
    "role_immersion": {"empathy": 0.14, "pleasure": 0.10, "curiosity": 0.08},
    "aesthetic_customization": {"pleasure": 0.16, "curiosity": 0.10, "dominance": 0.06},
}
INTENT_LABELS = ("zoom_in", "reveal", "branch_out", "reframe", "follow", "interact")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def vector(value: Any, *, field: str) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(PERSONALITY_DIMS):
        raise ValueError(f"{field} must contain exactly the eight stable profile dimensions")
    return {dimension: round(clamp(float(value[dimension])), 4) for dimension in PERSONALITY_DIMS}


def affect_vector(value: Any, *, field: str) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(AFFECT_DIMS):
        raise ValueError(f"{field} must contain exactly the seven affective dimensions")
    return {dimension: round(clamp(float(value[dimension])), 4) for dimension in AFFECT_DIMS}


def canonical_median(profiles: list[dict[str, float]]) -> dict[str, float]:
    if not profiles:
        raise ValueError("At least one profile is required")
    normalized = [vector(profile, field="root.stable_profile") for profile in profiles]
    return {
        dimension: round(statistics.median(profile[dimension] for profile in normalized), 4)
        for dimension in PERSONALITY_DIMS
    }


def compute_current_state(
    stable_profile: dict[str, float],
    affective_state: dict[str, float],
    *,
    alpha_affect: float,
) -> dict[str, float]:
    current: dict[str, float] = {}
    for personality_dim in PERSONALITY_DIMS:
        affect_delta = sum(
            weight * (float(affective_state.get(affect_dim, 0.5)) - 0.5)
            for affect_dim, weight in AFFECT_TO_PERSONALITY.get(personality_dim, {}).items()
        )
        current[personality_dim] = round(
            clamp(float(stable_profile[personality_dim]) + alpha_affect * affect_delta),
            4,
        )
    return current


def summarize_state(
    stable_profile: dict[str, float],
    affective_state: dict[str, float],
    *,
    initial: bool,
) -> str:
    strongest_profile = sorted(
        stable_profile.items(),
        key=lambda item: (-float(item[1]), PERSONALITY_DIMS.index(item[0])),
    )[:2]
    profile_text = ", ".join(name.replace("_", " ") for name, _ in strongest_profile)
    if initial:
        return f"Initial stable profile led by {profile_text}; affective baseline is neutral."
    strongest_affect = sorted(
        affective_state.items(),
        key=lambda item: (-float(item[1]), AFFECT_DIMS.index(item[0])),
    )[:2]
    affect_text = ", ".join(name.replace("_", " ") for name, _ in strongest_affect)
    return (
        f"The user currently leans toward {profile_text}, "
        f"with a short-term affective state marked by {affect_text}."
    )


def repair_node(
    node: dict[str, Any],
    *,
    original_root: dict[str, float],
    canonical_root: dict[str, float],
    alpha_affect: float,
    initial: bool = False,
) -> dict[str, Any]:
    repaired = copy.deepcopy(node)
    original_stable = vector(node.get("stable_profile"), field=f"node:{node.get('node_id')}.stable_profile")
    affective = affect_vector(node.get("affective_state"), field=f"node:{node.get('node_id')}.affective_state")
    repaired_stable = {
        dimension: round(
            clamp(canonical_root[dimension] + original_stable[dimension] - original_root[dimension]),
            4,
        )
        for dimension in PERSONALITY_DIMS
    }
    repaired["stable_profile"] = repaired_stable
    repaired["affective_state"] = affective
    repaired["current_state"] = compute_current_state(
        repaired_stable,
        affective,
        alpha_affect=alpha_affect,
    )
    repaired["profile_summary"] = summarize_state(repaired_stable, affective, initial=initial)
    return repaired


def normalize_tree_image_paths(tree: dict[str, Any]) -> None:
    tree["root_image"] = "images/n0.png"
    for node in tree.get("nodes", []):
        node_id = str(node["node_id"])
        node["image"] = f"images/{node_id}.png"
    for edge in tree.get("edges", []):
        target = str(edge.get("target_node", ""))
        if target:
            edge["target_image"] = f"images/{target}.png"


def link_assets(source_topic_dir: Path, target_topic_dir: Path) -> None:
    target_topic_dir.mkdir(parents=True, exist_ok=True)
    for name in ("images", "root_asset.json", "root_prompt.txt", "synopsis.txt"):
        source = source_topic_dir / name
        target = target_topic_dir / name
        if not source.exists():
            continue
        if target.exists() or target.is_symlink():
            if target.resolve() != source.resolve():
                raise ValueError(f"Asset link target mismatch: {target}")
            continue
        target.symlink_to(Path(os.path.relpath(source.resolve(), start=target.parent)), target_is_directory=source.is_dir())


def profile_spread(profiles: list[dict[str, float]]) -> dict[str, float]:
    return {
        dimension: round(
            max(float(profile[dimension]) for profile in profiles)
            - min(float(profile[dimension]) for profile in profiles),
            4,
        )
        for dimension in PERSONALITY_DIMS
    }


def sidecar_output_name(path: Path, accepted_dir: Path) -> str:
    return "accepted99" if path.resolve() == accepted_dir.resolve() else path.name


def canonicalize(args: argparse.Namespace) -> Path:
    base_release = Path(args.base_release).resolve()
    accepted_dir = Path(args.accepted_augmentation_dir).resolve()
    additional_dirs = [Path(path).resolve() for path in args.augmentation_dir]
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory must be empty or absent: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tree_rows = read_jsonl(base_release / "tree_manifest.jsonl")
    trees: dict[tuple[str, str, str], tuple[dict[str, Any], Path, dict[str, Any]]] = {}
    user_roots: dict[str, list[dict[str, float]]] = defaultdict(list)
    user_topics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tree_rows:
        source_tree_path = base_release / str(row["tree_path"])
        tree = read_json(source_tree_path)
        root_node = next((node for node in tree["nodes"] if str(node["node_id"]) == "n0"), None)
        if root_node is None:
            raise ValueError(f"Missing root n0: {source_tree_path}")
        root_profile = vector(root_node["stable_profile"], field=f"{source_tree_path}:n0.stable_profile")
        user_id = str(row["user_id"])
        key = (user_id, str(row["topic_id"]), str(row["tree_id"]))
        if key in trees:
            raise ValueError(f"Duplicate tree identity: {key}")
        trees[key] = (tree, source_tree_path, row)
        user_roots[user_id].append(root_profile)
        user_topics[user_id].append(
            {
                "topic_id": str(row["topic_id"]),
                "tree_id": str(row["tree_id"]),
                "root_profile": root_profile,
            }
        )

    if any(len(profiles) != 4 for profiles in user_roots.values()):
        counts = {user: len(profiles) for user, profiles in user_roots.items() if len(profiles) != 4}
        raise ValueError(f"Every user must have four topic roots: {counts}")

    canonical_profiles = {user: canonical_median(profiles) for user, profiles in user_roots.items()}
    canonical_rows: list[dict[str, Any]] = []
    inconsistent_users = 0
    maximum_original_spread = 0.0
    for user_id in sorted(user_roots):
        spread = profile_spread(user_roots[user_id])
        max_spread = max(spread.values())
        inconsistent = max_spread > args.consistency_tolerance
        inconsistent_users += int(inconsistent)
        maximum_original_spread = max(maximum_original_spread, max_spread)
        canonical_rows.append(
            {
                "user_id": user_id,
                "stable_profile": canonical_profiles[user_id],
                "affective_state_0": {dimension: 0.5 for dimension in AFFECT_DIMS},
                "profile_summary": summarize_state(
                    canonical_profiles[user_id],
                    {dimension: 0.5 for dimension in AFFECT_DIMS},
                    initial=True,
                ),
                "canonicalization": {
                    "method": "componentwise_median_of_four_topic_roots",
                    "topic_count": 4,
                    "topic_roots": sorted(user_topics[user_id], key=lambda item: item["topic_id"]),
                    "per_dimension_spread": spread,
                    "max_spread": max_spread,
                    "was_inconsistent": inconsistent,
                },
            }
        )
    write_jsonl(output_dir / "canonical_user_profiles.jsonl", canonical_rows)

    output_tree_rows: list[dict[str, Any]] = []
    repaired_tree_lookup: dict[tuple[str, str, str], tuple[Path, dict[str, float], dict[str, float]]] = {}
    changed_trees = 0
    repaired_nodes = 0
    tree_labels: Counter[str] = Counter()
    maximum_root_adjustment = 0.0
    for key in sorted(trees):
        tree, source_tree_path, row = trees[key]
        user_id, topic_id, tree_id = key
        root_node = next(node for node in tree["nodes"] if str(node["node_id"]) == "n0")
        original_root = vector(root_node["stable_profile"], field=f"{source_tree_path}:n0.stable_profile")
        canonical_root = canonical_profiles[user_id]
        adjustment = max(abs(canonical_root[dim] - original_root[dim]) for dim in PERSONALITY_DIMS)
        maximum_root_adjustment = max(maximum_root_adjustment, adjustment)
        changed_trees += int(adjustment > args.consistency_tolerance)
        repaired_tree = copy.deepcopy(tree)
        repaired_tree["nodes"] = [
            repair_node(
                node,
                original_root=original_root,
                canonical_root=canonical_root,
                alpha_affect=args.alpha_affect,
                initial=str(node["node_id"]) == "n0",
            )
            for node in tree["nodes"]
        ]
        repaired_nodes += len(repaired_tree["nodes"])
        normalize_tree_image_paths(repaired_tree)
        repaired_tree["profile_canonicalization"] = {
            "schema_version": "canonical_user_profile_v1",
            "method": "componentwise_median_of_four_topic_roots",
            "original_topic_root": original_root,
            "canonical_user_root": canonical_root,
            "max_root_adjustment": round(adjustment, 4),
            "preserved_fields": [
                "images",
                "click_grounding",
                "closed_intent",
                "intent_ranking",
                "slots",
                "story_state",
                "generation_prompt",
                "affective_state",
                "profile_deltas",
            ],
        }
        output_tree_path = output_dir / str(row["tree_path"])
        link_assets(source_tree_path.parent, output_tree_path.parent)
        write_json(output_tree_path, repaired_tree)
        labels = Counter(str(edge["closed_intent"]) for edge in repaired_tree["edges"])
        tree_labels.update(labels)
        output_row = copy.deepcopy(row)
        output_row["profile_repair"] = {
            "schema_version": "canonical_user_profile_v1",
            "canonical_profile_file": "canonical_user_profiles.jsonl",
            "root_adjusted": adjustment > args.consistency_tolerance,
            "max_root_adjustment": round(adjustment, 4),
        }
        output_tree_rows.append(output_row)
        repaired_tree_lookup[key] = (output_tree_path, original_root, canonical_root)
    write_jsonl(output_dir / "tree_manifest.jsonl", output_tree_rows)

    sidecar_dirs = [accepted_dir, *additional_dirs]
    sidecar_count = 0
    sidecar_labels: Counter[str] = Counter()
    sidecar_components: dict[str, int] = {}
    for source_dir in sidecar_dirs:
        source_manifest = source_dir / "usable_branches.jsonl"
        records = read_jsonl(source_manifest)
        repaired_records: list[dict[str, Any]] = []
        for record in records:
            key = (str(record["user_id"]), str(record["topic_id"]), str(record["tree_id"]))
            tree_entry = repaired_tree_lookup.get(key)
            if tree_entry is None:
                raise ValueError(f"Missing source tree for augmentation {record['augmentation_id']}: {key}")
            repaired_tree_path, original_root, canonical_root = tree_entry
            repaired = copy.deepcopy(record)
            repaired["source_tree_relative"] = str(repaired_tree_path)
            target_node = repaired.get("target_node")
            if not isinstance(target_node, dict):
                raise ValueError(f"Missing target_node in augmentation {record['augmentation_id']}")
            repaired["target_node"] = repair_node(
                target_node,
                original_root=original_root,
                canonical_root=canonical_root,
                alpha_affect=args.alpha_affect,
            )
            repaired["profile_canonicalization"] = {
                "schema_version": "canonical_user_profile_v1",
                "canonical_user_root": canonical_root,
                "generation_provenance_preserved": True,
            }
            repaired_records.append(repaired)
            label = str(repaired.get("edge", {}).get("closed_intent", repaired.get("target_intent", "")))
            sidecar_labels[label] += 1
        component = sidecar_output_name(source_dir, accepted_dir)
        component_dir = output_dir / "augmentations" / component
        write_jsonl(component_dir / "usable_branches.jsonl", repaired_records)
        image_dir = source_dir / "images"
        if image_dir.is_dir():
            target = component_dir / "images"
            target.symlink_to(Path(os.path.relpath(image_dir.resolve(), start=target.parent)), target_is_directory=True)
        component_summary = {
            "schema_version": "canonical_user_profile_sidecars_v1",
            "source_manifest": str(source_manifest),
            "record_count": len(repaired_records),
            "class_counts": dict(sorted(Counter(
                str(row.get("edge", {}).get("closed_intent", row.get("target_intent", "")))
                for row in repaired_records
            ).items())),
            "canonical_profile_file": "../../canonical_user_profiles.jsonl",
            "images_mutated": False,
        }
        write_json(component_dir / "summary.json", component_summary)
        sidecar_count += len(repaired_records)
        sidecar_components[component] = len(repaired_records)

    total_labels = tree_labels + sidecar_labels
    total_samples = sum(total_labels.values())
    summary = {
        "schema_version": "complete_release_canonical_user_profile_v1",
        "description": (
            "Complete logical release with one fixed canonical initial 8D profile per user. "
            "Original trees, images, clicks, intents, stories, and prompts remain unchanged."
        ),
        "source_release": str(base_release),
        "canonicalization_method": "componentwise_median_of_four_topic_roots",
        "alpha_affect": args.alpha_affect,
        "user_count": len(user_roots),
        "tree_count": len(tree_rows),
        "tree_edge_count": sum(tree_labels.values()),
        "sidecar_edge_count": sidecar_count,
        "sample_count": total_samples,
        "repaired_node_count": repaired_nodes,
        "original_inconsistent_user_count": inconsistent_users,
        "original_max_topic_root_spread": round(maximum_original_spread, 4),
        "changed_tree_count": changed_trees,
        "max_root_adjustment": round(maximum_root_adjustment, 4),
        "post_repair_inconsistent_user_count": 0,
        "class_counts": {label: total_labels[label] for label in INTENT_LABELS},
        "sidecar_components": sidecar_components,
        "mutation_policy": {
            "original_files_mutated": False,
            "images_mutated": False,
            "profile_fields_recomputed": [
                "node.stable_profile",
                "node.current_state",
                "node.profile_summary",
                "sidecar.target_node.stable_profile",
                "sidecar.target_node.current_state",
                "sidecar.target_node.profile_summary",
            ],
            "generation_provenance_preserved": True,
        },
    }
    write_json(output_dir / "summary.json", summary)

    if len(user_roots) != 82 or len(tree_rows) != 328 or total_samples != 6778 or sidecar_count != 1162:
        raise ValueError(
            f"Unexpected release totals: users={len(user_roots)}, trees={len(tree_rows)}, "
            f"samples={total_samples}, sidecars={sidecar_count}"
        )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a non-mutating release with one canonical initial 8D profile per user."
    )
    parser.add_argument("--base-release", required=True)
    parser.add_argument("--accepted-augmentation-dir", required=True)
    parser.add_argument("--augmentation-dir", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alpha-affect", type=float, default=0.30)
    parser.add_argument("--consistency-tolerance", type=float, default=1e-9)
    return parser.parse_args()


def main() -> None:
    output = canonicalize(parse_args())
    print(output)


if __name__ == "__main__":
    main()
