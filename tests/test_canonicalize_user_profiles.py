from __future__ import annotations

from dataset_tools.canonicalize_user_profiles import (
    AFFECT_DIMS,
    PERSONALITY_DIMS,
    canonical_median,
    compute_current_state,
    repair_node,
)


def profile(**overrides: float) -> dict[str, float]:
    value = {dimension: 0.5 for dimension in PERSONALITY_DIMS}
    value.update(overrides)
    return value


def affect(**overrides: float) -> dict[str, float]:
    value = {dimension: 0.5 for dimension in AFFECT_DIMS}
    value.update(overrides)
    return value


def test_canonical_median_is_componentwise_and_topic_order_independent() -> None:
    roots = [
        profile(goal_progress=0.2, world_discovery=0.8),
        profile(goal_progress=0.8, world_discovery=0.2),
        profile(goal_progress=0.4, world_discovery=0.7),
        profile(goal_progress=0.6, world_discovery=0.3),
    ]

    expected = canonical_median(roots)
    reversed_result = canonical_median(list(reversed(roots)))

    assert expected == reversed_result
    assert expected["goal_progress"] == 0.5
    assert expected["world_discovery"] == 0.5


def test_repair_node_preserves_relative_profile_delta() -> None:
    original_root = profile(goal_progress=0.4, mastery_logic=0.7)
    canonical_root = profile(goal_progress=0.6, mastery_logic=0.5)
    original_node = {
        "node_id": "n3",
        "stable_profile": profile(goal_progress=0.415, mastery_logic=0.694),
        "affective_state": affect(curiosity=0.7, dominance=0.6),
        "current_state": profile(),
        "profile_summary": "old",
    }

    repaired = repair_node(
        original_node,
        original_root=original_root,
        canonical_root=canonical_root,
        alpha_affect=0.30,
    )

    assert repaired["stable_profile"]["goal_progress"] == 0.615
    assert repaired["stable_profile"]["mastery_logic"] == 0.494
    assert round(repaired["stable_profile"]["goal_progress"] - canonical_root["goal_progress"], 4) == 0.015
    assert round(repaired["stable_profile"]["mastery_logic"] - canonical_root["mastery_logic"], 4) == -0.006
    assert repaired["current_state"] == compute_current_state(
        repaired["stable_profile"],
        repaired["affective_state"],
        alpha_affect=0.30,
    )
    assert original_node["stable_profile"]["goal_progress"] == 0.415


def test_repaired_initial_node_equals_canonical_profile() -> None:
    original_root = profile(challenge_seeking=0.3)
    canonical_root = profile(challenge_seeking=0.75)
    root_node = {
        "node_id": "n0",
        "stable_profile": original_root,
        "affective_state": affect(),
        "current_state": original_root,
        "profile_summary": "topic-specific",
    }

    repaired = repair_node(
        root_node,
        original_root=original_root,
        canonical_root=canonical_root,
        alpha_affect=0.30,
        initial=True,
    )

    assert repaired["stable_profile"] == canonical_root
    assert repaired["current_state"] == canonical_root
    assert repaired["profile_summary"].endswith("affective baseline is neutral.")
