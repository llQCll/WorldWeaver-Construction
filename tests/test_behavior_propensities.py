from __future__ import annotations

from dataset_tools.add_behavior_propensities import (
    INTENT_ORDER,
    PROFILE_DIMS,
    enumerate_paths,
    score_sibling_distribution,
    signed_signal_affinity,
    softmax_probabilities,
)


def _profile(value: float, **overrides: float) -> dict[str, float]:
    profile = {dimension: value for dimension in PROFILE_DIMS}
    profile.update(overrides)
    return profile


def _edge(
    edge_id: str,
    target: str,
    intent: str,
    signal: dict[str, float],
) -> dict:
    return {
        "edge_id": edge_id,
        "source_node": "n0",
        "target_node": target,
        "closed_intent": intent,
        "profile_signal": signal,
        "grounding": {"target_box": [0.1, 0.1, 0.4, 0.4]},
        "intent_ranking": [{"intent": intent, "score": 0.9}],
        "decision": {"confidence": 0.9},
    }


def test_signed_signal_affinity_respects_positive_and_negative_preferences() -> None:
    profile = _profile(0.5, challenge_seeking=0.1, world_discovery=0.9)
    positive, ignored = signed_signal_affinity(
        profile, {"world_discovery": 0.8}
    )
    negative, _ = signed_signal_affinity(profile, {"challenge_seeking": -0.8})
    assert positive == 0.9
    assert negative == 0.9
    assert ignored == 0


def test_softmax_probabilities_are_normalized() -> None:
    probabilities = softmax_probabilities(
        [-1.0, -2.0, -3.0],
        temperature=0.65,
    )
    assert sum(probabilities) == 1.0
    assert probabilities[0] > probabilities[1] > probabilities[2]
    assert softmax_probabilities([4.0], temperature=0.65) == [1.0]


def test_profile_grounding_changes_sibling_order() -> None:
    edges = [
        _edge("e1", "n1", "branch_out", {"world_discovery": 0.9}),
        _edge("e2", "n2", "interact", {"cooperative_orientation": 0.9}),
    ]
    source = {
        "affective_state": {},
        "current_state": _profile(
            0.5,
            world_discovery=0.9,
            cooperative_orientation=0.1,
        ),
    }
    scores = score_sibling_distribution(
        edges=edges,
        source_node=source,
        fixed_user_profile=_profile(
            0.5,
            world_discovery=0.9,
            cooperative_orientation=0.1,
        ),
        topic_affinity={intent: 0.8 for intent in INTENT_ORDER},
        history_intents=[],
        temperature=0.65,
    )
    by_id = {row["edge_id"]: row for row in scores}
    assert (
        by_id["e1"]["conditional_probability"]
        > by_id["e2"]["conditional_probability"]
    )
    assert sum(row["conditional_probability"] for row in scores) == 1.0


def test_root_to_leaf_path_probabilities_sum_to_one() -> None:
    edges = [
        _edge("e1", "n1", "branch_out", {"world_discovery": 0.9}),
        _edge("e2", "n2", "interact", {"cooperative_orientation": 0.9}),
    ]
    tree = {
        "tree_id": "tree",
        "nodes": [
            {"node_id": "n0"},
            {"node_id": "n1"},
            {"node_id": "n2"},
        ],
        "edges": edges,
    }
    scores = {
        "e1": {"conditional_probability": 0.7, "evidence_strength": 0.8},
        "e2": {"conditional_probability": 0.3, "evidence_strength": 0.7},
    }
    paths = enumerate_paths(tree, scores)
    assert sum(row["path_probability"] for row in paths) == 1.0
    assert paths[0]["is_map_path"]
    assert paths[0]["path_probability"] == 0.7
