from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset_pipeline import USER_INTENT_WEIGHTS  # noqa: E402
from select_profile_grounded_augmentations import (  # noqa: E402
    grounding_suggestions,
    heuristic_affordance,
    intent_scores,
)


TARGET_LABELS = ("reveal", "reframe", "zoom_in")


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def user_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["source_dataset"]), str(row["user_id"])


def node_image_path(tree_path: Path, node: dict[str, Any]) -> Path:
    node_id = str(node["node_id"])
    image = Path(str(node.get("image", f"images/{node_id}.png")))
    return image if image.is_absolute() else tree_path.parent / image


def load_profiles_and_trees(
    release_dir: Path,
) -> tuple[
    dict[tuple[str, str], dict[str, float]],
    dict[tuple[str, str], list[dict[str, Any]]],
]:
    rows = read_jsonl(release_dir / "tree_manifest.jsonl")
    dimensions = sorted({dimension for weights in USER_INTENT_WEIGHTS.values() for dimension in weights})
    trees_by_user: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    root_profiles: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        tree_path = release_dir / str(row["tree_path"])
        tree = read_json(tree_path)
        key = user_key(row)
        trees_by_user[key].append({"row": row, "tree_path": tree_path, "tree": tree})
        root_profiles[key].append(tree["nodes"][0]["stable_profile"])
    profiles = {
        key: {
            dimension: round(
                statistics.median(float(profile.get(dimension, 0.5)) for profile in user_profiles),
                6,
            )
            for dimension in dimensions
        }
        for key, user_profiles in root_profiles.items()
    }
    return profiles, trees_by_user


def load_excluded_nodes(run_dirs: list[str]) -> set[tuple[str, str, str, str]]:
    excluded: set[tuple[str, str, str, str]] = set()
    for raw_dir in run_dirs:
        run_dir = Path(raw_dir).resolve()
        manifest = run_dir / "usable_branches.jsonl"
        if not manifest.is_file():
            manifest = run_dir / "manifest.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(f"Missing augmentation manifest in {run_dir}")
        for row in read_jsonl(manifest):
            if row.get("status") not in {None, "accepted", "generated_uncalibrated", "planned"}:
                continue
            excluded.add(
                (
                    str(row["source_dataset"]),
                    str(row["user_id"]),
                    str(row["topic_id"]),
                    str(row["source_node_id"]),
                )
            )
    return excluded


def profile_ranking(
    profiles: dict[tuple[str, str], dict[str, float]], label: str
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], float]]:
    scores = {key: intent_scores(profile)[label] for key, profile in profiles.items()}
    ranked = sorted(scores, key=lambda key: (-scores[key], key[0], key[1]))
    denominator = max(1, len(ranked) - 1)
    percentile = {key: round(1.0 - index / denominator, 6) for index, key in enumerate(ranked)}
    return ranked, percentile


def candidate_options(
    *,
    label: str,
    eligible_users: list[tuple[str, str]],
    percentiles: dict[tuple[str, str], float],
    profiles: dict[tuple[str, str], dict[str, float]],
    trees_by_user: dict[tuple[str, str], list[dict[str, Any]]],
    excluded_nodes: set[tuple[str, str, str, str]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    options: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for key in eligible_users:
        propensity = intent_scores(profiles[key])[label]
        evidence = {
            dimension: profiles[key][dimension]
            for dimension in USER_INTENT_WEIGHTS[label]
        }
        for item in trees_by_user[key]:
            tree = item["tree"]
            tree_path = item["tree_path"]
            topic_id = str(tree["topic_id"])
            for node in tree.get("nodes", []):
                node_id = str(node["node_id"])
                exclusion_key = key[0], key[1], topic_id, node_id
                if exclusion_key in excluded_nodes:
                    continue
                local_score = heuristic_affordance(label=label, tree=tree, node=node)
                joint_score = propensity * (1.0 + math.log1p(max(0.0, local_score)))
                options[key].append(
                    {
                        "source_dataset": key[0],
                        "user_id": key[1],
                        "cohort_intent": label,
                        "target_intent": label,
                        "intent_propensity": propensity,
                        "cohort_intent_propensity": propensity,
                        "profile_percentile": percentiles[key],
                        "profile_evidence": evidence,
                        "median_stable_profile": profiles[key],
                        "tree_id": str(tree.get("tree_id", "")),
                        "topic_id": topic_id,
                        "source_tree": str(tree_path),
                        "source_node_id": node_id,
                        "source_node_depth": int(node.get("depth", 0)),
                        "source_image": str(node_image_path(tree_path, node)),
                        "story_state": node.get("story_state", {}),
                        "source_node_profile": node.get("stable_profile", {}),
                        "source_node_affect": node.get("affective_state", {}),
                        "source_node_current_state": node.get("current_state", {}),
                        "grounding_suggestions": grounding_suggestions(tree, node_id),
                        "local_affordance_score": local_score,
                        "profile_affordance_joint_score": round(joint_score, 6),
                        "status": "needs_llm_affordance_validation",
                    }
                )
        options[key].sort(
            key=lambda row: (
                -float(row["profile_affordance_joint_score"]),
                int(row["source_node_depth"]),
                str(row["topic_id"]),
                str(row["source_node_id"]),
            )
        )
    return options


def allocate(
    *,
    label: str,
    quota: int,
    eligible_users: list[tuple[str, str]],
    options: dict[tuple[str, str], list[dict[str, Any]]],
    max_per_user_label: int,
    max_per_user: int,
    max_per_tree: int,
    topic_cap: int,
    source_caps: dict[str, int],
    user_counts: Counter[tuple[str, str]],
    tree_counts: Counter[tuple[str, str, str]],
    used_nodes: set[tuple[str, str, str, str]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    user_label_counts: Counter[tuple[str, str]] = Counter()
    topic_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    cursors: Counter[tuple[str, str]] = Counter()

    while len(selected) < quota:
        progress = False
        for key in eligible_users:
            if len(selected) >= quota:
                break
            if user_label_counts[key] >= max_per_user_label or user_counts[key] >= max_per_user:
                continue
            rows = options[key]
            while cursors[key] < len(rows):
                row = rows[cursors[key]]
                cursors[key] += 1
                node_key = key[0], key[1], str(row["topic_id"]), str(row["source_node_id"])
                tree_key = key[0], key[1], str(row["topic_id"])
                source = key[0]
                if node_key in used_nodes:
                    continue
                if tree_counts[tree_key] >= max_per_tree:
                    continue
                if topic_counts[str(row["topic_id"])] >= topic_cap:
                    continue
                if source_counts[source] >= source_caps[source]:
                    continue
                selected.append(row)
                used_nodes.add(node_key)
                user_counts[key] += 1
                user_label_counts[key] += 1
                tree_counts[tree_key] += 1
                topic_counts[str(row["topic_id"])] += 1
                source_counts[source] += 1
                progress = True
                break
        if not progress:
            raise ValueError(
                f"Unable to satisfy {label} quota {quota}; selected {len(selected)}. "
                "Relax caps or increase eligible users."
            )
    return selected


def build_plan(args: argparse.Namespace) -> Path:
    release_dir = Path(args.release_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    config = read_json(Path(args.target_config_json).resolve())
    target_config = config["target_config"]
    constraints = config["global_constraints"]
    profiles, trees_by_user = load_profiles_and_trees(release_dir)
    excluded_nodes = load_excluded_nodes(args.exclude_run_dir)
    source_user_counts = Counter(source for source, _user_id in profiles)
    topic_count = len(
        {
            str(item["tree"]["topic_id"])
            for items in trees_by_user.values()
            for item in items
        }
    )

    selected: list[dict[str, Any]] = []
    user_counts: Counter[tuple[str, str]] = Counter()
    tree_counts: Counter[tuple[str, str, str]] = Counter()
    used_nodes: set[tuple[str, str, str, str]] = set(excluded_nodes)
    for label in TARGET_LABELS:
        label_config = target_config[label]
        quota = int(label_config["quota"])
        ranked_users, percentiles = profile_ranking(profiles, label)
        eligible_users = ranked_users[: int(label_config["eligible_users"])]
        options = candidate_options(
            label=label,
            eligible_users=eligible_users,
            percentiles=percentiles,
            profiles=profiles,
            trees_by_user=trees_by_user,
            excluded_nodes=excluded_nodes,
        )
        topic_cap = math.ceil(
            quota / topic_count * float(constraints["topic_cap_multiplier"])
        )
        source_caps = {
            source: math.ceil(
                quota
                * count
                / len(profiles)
                * float(constraints["source_cap_multiplier"])
            )
            for source, count in source_user_counts.items()
        }
        selected.extend(
            allocate(
                label=label,
                quota=quota,
                eligible_users=eligible_users,
                options=options,
                max_per_user_label=int(label_config["max_per_user"]),
                max_per_user=int(constraints["max_per_user"]),
                max_per_tree=int(constraints["max_per_tree"]),
                topic_cap=topic_cap,
                source_caps=source_caps,
                user_counts=user_counts,
                tree_counts=tree_counts,
                used_nodes=used_nodes,
            )
        )

    per_label_index: Counter[str] = Counter()
    for row in selected:
        label = str(row["target_intent"])
        per_label_index[label] += 1
        row["augmentation_id"] = (
            f"hd720_{row['source_dataset']}_user_{row['user_id']}_{row['topic_id']}_"
            f"{row['source_node_id']}_{label}_{per_label_index[label]:03d}"
        )
        row["allocation_policy"] = (
            "high-density counterfactual sidecar; profile-top-cohort plus node affordance; "
            "one new branch per source node; quotas never override live affordance or visual gates"
        )
        row["naturalness_constraints"] = {
            "profile_grounded": True,
            "story_state_grounded": True,
            "single_semantic_transformation": True,
            "preserve_character_world_and_style_continuity": True,
            "no_ui_or_click_markers": True,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "candidate_branches.jsonl", selected)
    label_counts = Counter(str(row["target_intent"]) for row in selected)
    topic_counts = Counter(str(row["topic_id"]) for row in selected)
    source_counts = Counter(str(row["source_dataset"]) for row in selected)
    selected_user_counts = Counter((str(row["source_dataset"]), str(row["user_id"])) for row in selected)
    local_scores = [float(row["local_affordance_score"]) for row in selected]
    percentiles = [float(row["profile_percentile"]) for row in selected]
    write_json(
        output_dir / "summary.json",
        {
            "schema_version": "high_density_rare_augmentation_plan_v1",
            "source_merged_dataset": str(release_dir),
            "target_config": config,
            "candidate_branch_count": len(selected),
            "candidate_counts": dict(sorted(label_counts.items())),
            "selected_user_count": len(selected_user_counts),
            "per_user_count": {
                f"{source}:user_{user_id}": count
                for (source, user_id), count in sorted(selected_user_counts.items())
            },
            "topic_counts": dict(sorted(topic_counts.items())),
            "source_counts": dict(sorted(source_counts.items())),
            "excluded_existing_source_node_count": len(excluded_nodes),
            "profile_percentile": {
                "min": round(min(percentiles), 6),
                "median": round(statistics.median(percentiles), 6),
                "max": round(max(percentiles), 6),
            },
            "local_affordance_score": {
                "min": round(min(local_scores), 6),
                "median": round(statistics.median(local_scores), 6),
                "max": round(max(local_scores), 6),
            },
            "status": "local_selection_complete; no API data sent; live LLM affordance validation required",
        },
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a dense but profile- and story-grounded minority-intent sidecar plan."
    )
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-config-json", required=True)
    parser.add_argument("--exclude-run-dir", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    print(build_plan(parse_args()))


if __name__ == "__main__":
    main()
