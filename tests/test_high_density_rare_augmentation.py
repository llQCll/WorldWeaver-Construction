from __future__ import annotations

import unittest

from dataset_tools.augment_high_density_rare_branches import (
    naturalness_prompt_contract,
    strict_visual_acceptance,
)


class HighDensityRareAugmentationTests(unittest.TestCase):
    def test_reframe_contract_locks_time_and_forbids_ui(self) -> None:
        prompt = naturalness_prompt_contract("reframe")
        self.assertIn("same moment", prompt)
        self.assertIn("do not advance time", prompt)
        self.assertIn("user profile", prompt)
        self.assertIn("UI", prompt)

    def test_profile_alignment_is_a_hard_acceptance_gate(self) -> None:
        calibration = {
            "calibration": {"intent_matches": True},
            "delta_alignment": {
                "image_intent_alignment": 0.93,
                "continuity_alignment": 0.92,
                "profile_alignment": 0.79,
            },
        }
        self.assertFalse(
            strict_visual_acceptance(
                calibration,
                min_visual_alignment=0.9,
                min_continuity_alignment=0.88,
                min_profile_alignment=0.8,
            )
        )
        calibration["delta_alignment"]["profile_alignment"] = 0.84
        self.assertTrue(
            strict_visual_acceptance(
                calibration,
                min_visual_alignment=0.9,
                min_continuity_alignment=0.88,
                min_profile_alignment=0.8,
            )
        )


if __name__ == "__main__":
    unittest.main()
