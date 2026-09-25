from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from . import select_high_density_rare_augmentations as base
except ImportError:
    import select_high_density_rare_augmentations as base


ORIGINAL_HEURISTIC_AFFORDANCE = base.heuristic_affordance
TARGET_LABELS = ("reveal", "reframe", "zoom_in", "branch_out", "follow", "interact")
INTERACT_TERMS = {
    "touch": 1.5,
    "open": 1.6,
    "repair": 1.7,
    "use": 1.2,
    "help": 1.2,
    "move": 1.1,
    "turn": 1.2,
    "press": 1.5,
    "pull": 1.5,
    "activate": 1.7,
    "talk": 1.0,
    "offer": 1.0,
}


def mixed_heuristic_affordance(
    *, label: str, tree: dict[str, Any], node: dict[str, Any]
) -> float:
    if label != "interact":
        return ORIGINAL_HEURISTIC_AFFORDANCE(label=label, tree=tree, node=node)
    node_id = str(node["node_id"])
    outgoing = [
        edge for edge in tree.get("edges", []) if str(edge.get("source_node")) == node_id
    ]
    text = json.dumps(
        [node.get("story_state", {}), *outgoing], ensure_ascii=False
    ).lower()
    score = sum(weight * text.count(term) for term, weight in INTERACT_TERMS.items())
    score += sum(
        3.0 for edge in outgoing if str(edge.get("closed_intent")) == "interact"
    )
    score += sum(
        0.8 for edge in outgoing if edge.get("grounding", {}).get("target_box")
    )
    score += max(0, 4 - int(node.get("depth", 0))) * 0.08
    return round(score, 4)


def main() -> None:
    base.TARGET_LABELS = TARGET_LABELS
    base.heuristic_affordance = mixed_heuristic_affordance
    args = base.parse_args()
    output_dir = base.build_plan(args)
    candidate_path = Path(output_dir) / "candidate_branches.jsonl"
    candidates = base.read_jsonl(candidate_path)
    for candidate in candidates:
        candidate["augmentation_id"] = str(candidate["augmentation_id"]).replace("hd720_", "hd900_", 1)
    base.write_jsonl(candidate_path, candidates)
    summary_path = Path(output_dir) / "summary.json"
    summary = base.read_json(summary_path)
    summary["schema_version"] = "high_density_mixed_augmentation_plan_v1"
    summary["collection_policy"] = (
        "six-label natural mix with minority enrichment; quotas are candidate ceilings and never override "
        "profile, story-affordance, image-intent, continuity, or profile-alignment rejection gates"
    )
    base.write_json(summary_path, summary)
    print(output_dir)


if __name__ == "__main__":
    main()
