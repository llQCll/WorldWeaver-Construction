from __future__ import annotations

import unittest

from dataset_tools.augment_high_density_mixed_branches import (
    MIXED_LABELS,
    select_mixed_canary,
)


class HighDensityMixedAugmentationTests(unittest.TestCase):
    def test_canary_contains_all_six_labels(self) -> None:
        rows = []
        for label in MIXED_LABELS:
            rows.extend(
                [
                    {
                        "augmentation_id": f"{label}_low",
                        "target_intent": label,
                        "profile_affordance_joint_score": 0.5,
                    },
                    {
                        "augmentation_id": f"{label}_high",
                        "target_intent": label,
                        "profile_affordance_joint_score": 0.9,
                    },
                ]
            )
        selected = select_mixed_canary(rows)
        self.assertEqual(6, len(selected))
        self.assertEqual(set(MIXED_LABELS), {row["target_intent"] for row in selected})
        self.assertTrue(all(row["augmentation_id"].endswith("_high") for row in selected))


if __name__ == "__main__":
    unittest.main()
