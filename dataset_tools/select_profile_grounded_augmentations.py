from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset_pipeline import INTENT_ORDER, USER_INTENT_WEIGHTS  # noqa: E402


TARGET_CONFIG = {
    "reframe": {"users": 4, "branches_per_user": 4},
    "zoom_in": {"users": 4, "branches_per_user": 4},
    "reveal": {"users": 3, "branches_per_user": 2},
    "branch_out": {"users": 3, "branches_per_user": 2},
    "follow": {"users": 3, "branches_per_user": 2},
}

SELECTION_ORDER = ["reframe", "zoom_in", "reveal", "branch_out", "follow"]

AFFORDANCE_TERMS = {
    "zoom_in": {
        "detail": 2.0,
        "symbol": 1.7,
        "inscription": 1.8,
        "compass": 1.5,
        "map": 1.4,
        "mechanism": 1.5,
        "texture": 1.3,
        "face": 1.1,
        "visible": 1.0,
        "pattern": 1.2,
    },
    "reveal": {
        "hidden": 2.0,
        "secret": 2.0,
        "unknown": 1.7,
        "uncover": 1.8,
        "sealed": 1.4,
        "mystery": 1.5,
        "clue": 1.5,
        "inside": 1.2,
        "occluded": 1.8,
        "unexplained": 1.5,
    },
    "branch_out": {
        "door": 1.6,
        "path": 1.5,
        "route": 1.7,
        "gateway": 1.8,
        "portal": 1.8,
        "tunnel": 1.7,
        "beyond": 1.2,
        "new area": 1.8,
        "district": 1.3,
        "horizon": 1.2,
    },
    "reframe": {
        "view": 1.8,
        "perspective": 2.0,
        "angle": 2.0,
        "watch": 1.0,
        "witness": 1.5,
        "reflection": 1.6,
        "mirror": 1.6,
        "expression": 1.2,
        "crowd": 1.0,
        "observer": 1.3,
    },
    "follow": {
        "follow": 2.0,
        "moving": 1.7,
        "trail": 1.8,
        "signal": 1.6,
        "sound": 1.4,
        "footprint": 1.7,
        "shadow": 1.2,
        "track": 1.8,
        "toward": 0.8,
        "through": 0.7,
    },
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_manifest(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def user_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["source_dataset"]), str(row["user_id"])


def profile_medians(
    *, root: Path, rows: list[dict[str, Any]]
) -> tuple[dict[tuple[str, str], dict[str, float]], dict[tuple[str, str], list[dict[str, Any]]]]:
    dimensions = sorted({dimension for weights in USER_INTENT_WEIGHTS.values() for dimension in weights})
    trees_by_user: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    values: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        tree_path = root / str(row["tree_path"])
        tree = read_json(tree_path)
        key = user_key(row)
        trees_by_user[key].append({"row": row, "tree_path": tree_path, "tree": tree})
        values[key].append(tree["nodes"][0]["stable_profile"])
    profiles = {
        key: {
            dimension: round(
                statistics.median(float(profile.get(dimension, 0.5)) for profile in user_profiles),
                6,
            )
            for dimension in dimensions
        }
        for key, user_profiles in values.items()
    }
    return profiles, trees_by_user


def intent_scores(profile: dict[str, float]) -> dict[str, float]:
    raw = {
        label: 0.18
        + sum(weight * float(profile.get(dimension, 0.5)) for dimension, weight in USER_INTENT_WEIGHTS[label].items())
        for label in INTENT_ORDER
    }
    total = sum(raw.values())
    return {label: round(raw[label] / total, 6) for label in INTENT_ORDER}


def select_users(
    *,
    profiles: dict[tuple[str, str], dict[str, float]],
    target_config: dict[str, dict[str, int]] | None = None,
    excluded_keys: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    target_config = target_config or TARGET_CONFIG
    excluded_keys = excluded_keys or set()
    scores = {key: intent_scores(profile) for key, profile in profiles.items()}
    selected_keys: set[tuple[str, str]] = set()
    selections: list[dict[str, Any]] = []
    sources = sorted({source for source, _user_id in profiles})
    for label in SELECTION_ORDER:
        target = int(target_config[label]["users"])
        per_source = target // len(sources)
        chosen: list[tuple[str, str]] = []
        for source in sources:
            candidates = sorted(
                (
                    key
                    for key in profiles
                    if key[0] == source and key not in selected_keys and key not in excluded_keys
                ),
                key=lambda key: (-scores[key][label], key[1]),
            )
            chosen.extend(candidates[:per_source])
        while len(chosen) < target:
            candidates = sorted(
                (
                    key
                    for key in profiles
                    if key not in selected_keys and key not in chosen and key not in excluded_keys
                ),
                key=lambda key: (-scores[key][label], key[0], key[1]),
            )
            if not candidates:
                raise ValueError(f"Insufficient distinct users for {label}")
            chosen.append(candidates[0])
        for key in chosen:
            selected_keys.add(key)
            dimensions = USER_INTENT_WEIGHTS[label]
            selections.append(
                {
                    "source_dataset": key[0],
                    "user_id": key[1],
                    "target_intent": label,
                    "branches_requested": int(target_config[label]["branches_per_user"]),
                    "intent_propensity": scores[key][label],
                    "profile_evidence": {
                        dimension: profiles[key][dimension] for dimension in dimensions
                    },
                    "median_stable_profile": profiles[key],
                    "selection_method": "deterministic median-profile weighted propensity with source coverage",
                }
            )
    return selections


def node_text(tree: dict[str, Any], node: dict[str, Any]) -> str:
    source_id = str(node["node_id"])
    outgoing = [edge for edge in tree.get("edges", []) if str(edge.get("source_node")) == source_id]
    content = [
        tree.get("topic_id", ""),
        tree.get("world_bible", {}),
        node.get("story_state", {}),
    ]
    for edge in outgoing:
        content.extend(
            [
                edge.get("branch_label", ""),
                edge.get("grounding", {}),
                edge.get("slots", {}),
                edge.get("story_state_after", {}),
            ]
        )
    return json.dumps(content, ensure_ascii=False).lower()


def heuristic_affordance(*, label: str, tree: dict[str, Any], node: dict[str, Any]) -> float:
    text = node_text(tree, node)
    score = sum(weight * text.count(term) for term, weight in AFFORDANCE_TERMS[label].items())
    outgoing = [edge for edge in tree.get("edges", []) if edge.get("source_node") == node.get("node_id")]
    visible_targets = sum(1 for edge in outgoing if edge.get("grounding", {}).get("target_box"))
    if label in {"zoom_in", "reframe"}:
        score += min(3, visible_targets) * 0.8
    score += max(0, 4 - int(node.get("depth", 0))) * 0.08
    return round(score, 4)


def grounding_suggestions(tree: dict[str, Any], node_id: str) -> list[str]:
    labels: list[str] = []
    for edge in tree.get("edges", []):
        if str(edge.get("source_node")) != node_id:
            continue
        target = str(edge.get("grounding", {}).get("target_label", "")).strip()
        if target and target not in labels:
            labels.append(target)
    return labels[:6]


def select_nodes(
    *,
    root: Path,
    selections: list[dict[str, Any]],
    trees_by_user: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Choose a mixed-intent set across distinct topics for each selected user."""
    candidates: list[dict[str, Any]] = []
    for selection in selections:
        key = str(selection["source_dataset"]), str(selection["user_id"])
        cohort_intent = str(selection["target_intent"])
        requested = int(selection["branches_requested"])
        profile = selection["median_stable_profile"]
        propensities = intent_scores(profile)
        options_by_tree: list[dict[str, dict[str, Any]]] = []

        for item in trees_by_user[key]:
            tree = item["tree"]
            tree_path = item["tree_path"]
            label_options: dict[str, dict[str, Any]] = {}
            for label in SELECTION_ORDER:
                ranked_nodes = sorted(
                    tree.get("nodes", []),
                    key=lambda node: (
                        -heuristic_affordance(label=label, tree=tree, node=node),
                        int(node.get("depth", 0)),
                        str(node.get("node_id", "")),
                    ),
                )
                node = ranked_nodes[0]
                node_id = str(node["node_id"])
                image_path = tree_path.parent / str(node.get("image", f"images/{node_id}.png"))
                if not image_path.is_file():
                    image_path = tree_path.parent / "images" / f"{node_id}.png"
                local_score = heuristic_affordance(label=label, tree=tree, node=node)
                rarity_emphasis = 1.20 if label in {"reframe", "zoom_in"} else 1.0
                joint_score = propensities[label] * (1.0 + math.log1p(max(0.0, local_score))) * rarity_emphasis
                label_options[label] = {
                    "source_dataset": key[0],
                    "user_id": key[1],
                    "cohort_intent": cohort_intent,
                    "target_intent": label,
                    "intent_propensity": propensities[label],
                    "cohort_intent_propensity": selection["intent_propensity"],
                    "median_stable_profile": profile,
                    "tree_id": str(tree.get("tree_id", "")),
                    "topic_id": str(tree.get("topic_id", "")),
                    "source_tree": str(tree_path.resolve()),
                    "source_node_id": node_id,
                    "source_node_depth": int(node.get("depth", 0)),
                    "source_image": str(image_path.resolve()),
                    "story_state": node.get("story_state", {}),
                    "source_node_profile": node.get("stable_profile", {}),
                    "source_node_affect": node.get("affective_state", {}),
                    "source_node_current_state": node.get("current_state", {}),
                    "grounding_suggestions": grounding_suggestions(tree, node_id),
                    "local_affordance_score": local_score,
                    "profile_affordance_joint_score": round(joint_score, 6),
                    "status": "needs_llm_affordance_validation",
                }
            options_by_tree.append(label_options)

        primary_count = min(requested, max(1, (requested + 1) // 2))
        primary_ranked = sorted(
            (options[cohort_intent] for options in options_by_tree),
            key=lambda row: (-float(row["profile_affordance_joint_score"]), row["topic_id"]),
        )
        chosen = primary_ranked[:primary_count]
        used_topics = {str(row["topic_id"]) for row in chosen}
        used_labels = {cohort_intent}

        while len(chosen) < requested:
            alternatives: list[dict[str, Any]] = []
            for options in options_by_tree:
                if str(next(iter(options.values()))["topic_id"]) in used_topics:
                    continue
                for label, row in options.items():
                    if label == cohort_intent:
                        continue
                    diversity_bonus = 1.15 if label not in used_labels else 1.0
                    alternative = dict(row)
                    alternative["_selection_score"] = round(
                        float(row["profile_affordance_joint_score"]) * diversity_bonus,
                        6,
                    )
                    alternatives.append(alternative)
            if not alternatives:
                break
            best = sorted(
                alternatives,
                key=lambda row: (
                    -float(row["_selection_score"]),
                    row["target_intent"],
                    row["topic_id"],
                ),
            )[0]
            best.pop("_selection_score", None)
            chosen.append(best)
            used_topics.add(str(best["topic_id"]))
            used_labels.add(str(best["target_intent"]))

        for index, row in enumerate(chosen, 1):
            row["augmentation_id"] = (
                f"{key[0]}_user_{key[1]}_{row['topic_id']}_{row['source_node_id']}_"
                f"{row['target_intent']}_{index}"
            )
            row["allocation_policy"] = (
                "mixed non-interact intents across distinct topics; "
                "at most half reserved for the user's cohort intent"
            )
            candidates.append(row)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="Select profile-grounded users and nodes for minority-intent augmentation.")
    parser.add_argument("--merged-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-config-json", default="", help="Optional per-intent user and branch quotas.")
    parser.add_argument(
        "--exclude-run-dir",
        action="append",
        default=[],
        help="Exclude users already present in an augmentation run; may be repeated.",
    )
    args = parser.parse_args()

    root = Path(args.merged_dir).resolve()
    output = Path(args.output_dir).resolve()
    rows = load_manifest(root)
    profiles, trees_by_user = profile_medians(root=root, rows=rows)
    target_config = TARGET_CONFIG
    if args.target_config_json:
        config_data = read_json(Path(args.target_config_json).resolve())
        target_config = config_data.get("target_config", config_data)
        missing = [label for label in SELECTION_ORDER if label not in target_config]
        if missing:
            raise ValueError(f"Target config is missing labels: {missing}")

    excluded_keys: set[tuple[str, str]] = set()
    for raw_run_dir in args.exclude_run_dir:
        run_dir = Path(raw_run_dir).resolve()
        source = run_dir / "usable_branches.jsonl"
        if not source.is_file():
            source = run_dir / "manifest.jsonl"
        if not source.is_file():
            raise FileNotFoundError(f"Missing augmentation manifest in {run_dir}")
        for line in source.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            excluded_keys.add((str(record["source_dataset"]), str(record["user_id"])))

    selections = select_users(
        profiles=profiles,
        target_config=target_config,
        excluded_keys=excluded_keys,
    )
    candidates = select_nodes(root=root, selections=selections, trees_by_user=trees_by_user)

    write_json(
        output / "target_users.json",
        {
            "target_config": target_config,
            "excluded_users": [list(key) for key in sorted(excluded_keys)],
            "users": selections,
        },
    )
    write_jsonl(output / "candidate_branches.jsonl", candidates)
    counts = defaultdict(int)
    for row in candidates:
        counts[str(row["target_intent"])] += 1
    write_json(
        output / "summary.json",
        {
            "schema_version": "profile_grounded_minority_augmentation_plan_v2",
            "source_merged_dataset": str(root),
            "source_tree_count": len(rows),
            "available_user_count": len(profiles),
            "excluded_user_count": len(excluded_keys),
            "selected_user_count": len(selections),
            "target_config": target_config,
            "candidate_branch_count": len(candidates),
            "candidate_counts": dict(sorted(counts.items())),
            "selection_order": SELECTION_ORDER,
            "profile_aggregation": "median of the four topic-root stable profiles per namespaced user",
            "allocation_policy": "mixed non-interact labels across distinct topics; cohort label is capped near half per user",
            "status": "local_selection_complete; live LLM affordance validation not run",
        },
    )
    print(output)


if __name__ == "__main__":
    main()
