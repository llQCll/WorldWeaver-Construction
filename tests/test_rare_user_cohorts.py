from __future__ import annotations

import unittest
from collections import Counter

from dataset_tools.prepare_rare_user_cohorts import (
    PROFILE_DIMS,
    apply_synthetic_fallback,
    evidence_passes,
    select_cohorts,
    synthesize_profile_for_cohort,
)


DIMS = [
    "goal_progress",
    "mastery_logic",
    "challenge_seeking",
    "social_attachment",
    "cooperative_orientation",
    "world_discovery",
    "role_immersion",
    "aesthetic_customization",
]


def profile(user_id: int, **values: float):
    stable = {dimension: 0.45 for dimension in DIMS}
    stable.update(values)
    return {
        "user_id": str(user_id),
        "raw_profile": f"profile {user_id}",
        "stable_profile": stable,
        "profile_summary": f"summary {user_id}",
        "dimension_rationales": {},
    }


class RareUserCohortTests(unittest.TestCase):
    def test_selects_distinct_users_with_requested_cohort_counts(self) -> None:
        rows = [
            profile(1, role_immersion=0.90, aesthetic_customization=0.90, social_attachment=0.70),
            profile(2, role_immersion=0.86, aesthetic_customization=0.88, social_attachment=0.72),
            profile(3, role_immersion=0.82, aesthetic_customization=0.84, social_attachment=0.68),
            profile(4, mastery_logic=0.90, aesthetic_customization=0.80),
            profile(5, mastery_logic=0.86, aesthetic_customization=0.76),
            profile(6, mastery_logic=0.84, world_discovery=0.88),
            profile(7, mastery_logic=0.80, world_discovery=0.84),
            profile(8, world_discovery=0.90, challenge_seeking=0.78),
            profile(9, goal_progress=0.88, world_discovery=0.72, role_immersion=0.62),
            profile(10, **{dimension: 0.58 for dimension in DIMS}),
            profile(11, **{dimension: 0.52 for dimension in DIMS}),
            profile(12, **{dimension: 0.50 for dimension in DIMS}),
        ]
        targets = {
            "reframe": 3,
            "zoom_in": 2,
            "reveal": 2,
            "branch_out": 1,
            "follow": 1,
            "mixed": 1,
        }

        selected = select_cohorts(normalized=rows, target_cohorts=targets)

        self.assertEqual(10, len(selected))
        self.assertEqual(10, len({row["user_id"] for row in selected}))
        self.assertEqual(targets, dict(Counter(row["rare_intent_cohort"] for row in selected)))

    def test_synthetic_profile_is_realistic_audited_and_deterministic(self) -> None:
        source = profile(21, **{dimension: 0.35 for dimension in DIMS})
        source.update(
            {
                "rare_intent_cohort": "reframe",
                "natural_profile_evidence_passed": False,
                "profile_source": "normalized_original_profile",
            }
        )

        first = synthesize_profile_for_cohort(source, "reframe")
        second = synthesize_profile_for_cohort(source, "reframe")

        self.assertEqual(first["stable_profile"], second["stable_profile"])
        self.assertEqual(set(PROFILE_DIMS), set(first["stable_profile"]))
        self.assertTrue(all(0.32 <= value <= 0.84 for value in first["stable_profile"].values()))
        self.assertTrue(evidence_passes("reframe", first["stable_profile"]))
        self.assertEqual("synthetic_cohort_variant", first["profile_source"])
        self.assertTrue(first["profile_summary"])
        self.assertTrue(first["profile_summary_zh"])
        self.assertEqual(set(PROFILE_DIMS), set(first["dimension_rationales"]))
        self.assertFalse(first["synthetic_provenance"]["observed_ground_truth"])
        self.assertEqual(source["stable_profile"], first["synthetic_provenance"]["original_stable_profile"])

    def test_every_synthetic_cohort_passes_evidence(self) -> None:
        for index, cohort in enumerate(["reframe", "zoom_in", "reveal", "branch_out", "follow"]):
            with self.subTest(cohort=cohort):
                source = profile(100 + index, **{dimension: 0.30 for dimension in DIMS})
                result = synthesize_profile_for_cohort(source, cohort)
                self.assertTrue(evidence_passes(cohort, result["stable_profile"]))
                self.assertTrue(all(0.32 <= value <= 0.84 for value in result["stable_profile"].values()))

    def test_fallback_changes_only_failed_natural_evidence(self) -> None:
        natural = profile(31, role_immersion=0.80, aesthetic_customization=0.81)
        natural.update(
            {
                "rare_intent_cohort": "reframe",
                "natural_profile_evidence_passed": True,
                "profile_source": "normalized_original_profile",
            }
        )
        failed = profile(32, role_immersion=0.40, aesthetic_customization=0.42)
        failed.update(
            {
                "rare_intent_cohort": "reframe",
                "natural_profile_evidence_passed": False,
                "profile_source": "normalized_original_profile",
            }
        )

        result = apply_synthetic_fallback([natural, failed])

        self.assertEqual(natural["stable_profile"], result[0]["stable_profile"])
        self.assertFalse(result[0]["synthetic_profile"])
        self.assertTrue(result[1]["synthetic_profile"])
        self.assertNotEqual(failed["stable_profile"], result[1]["stable_profile"])


if __name__ == "__main__":
    unittest.main()
