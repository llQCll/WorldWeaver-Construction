from __future__ import annotations

import unittest
from pathlib import Path

from dataset_tools.augment_existing_minority_branches import (
    accepted_visual_calibration,
    select_canary,
)
from dataset_tools.dataset_pipeline import DatasetPipeline, PipelineConfig


LABELS = ["zoom_in", "reveal", "branch_out", "reframe", "follow", "interact"]


class MinorityAugmentationTests(unittest.TestCase):
    def test_forced_intent_is_preserved_with_audit_metadata(self) -> None:
        pipeline = DatasetPipeline(
            config=PipelineConfig(intent_sampling_policy="hierarchical_natural"),
            output_dir=Path("/tmp/minority-augmentation-test"),
            dry_run=True,
        )
        profile = {
            "user_id": "91",
            "stable_profile": {
                "goal_progress": 0.4,
                "mastery_logic": 0.5,
                "challenge_seeking": 0.2,
                "social_attachment": 0.8,
                "cooperative_orientation": 0.6,
                "world_discovery": 0.4,
                "role_immersion": 0.8,
                "aesthetic_customization": 0.9,
            },
        }
        source_node = {
            "node_id": "n2",
            "user_id": "91",
            "depth": 2,
            "image": "/missing.png",
            "story_state": {},
            "stable_profile": profile["stable_profile"],
            "affective_state": {key: 0.5 for key in [
                "pleasure", "arousal", "dominance", "tension",
                "curiosity", "empathy", "cognitive_load",
            ]},
            "current_state": profile["stable_profile"],
        }
        assessment = {
            "topic_affinity": {label: 0.7 for label in LABELS},
            "node_affordance": {label: 0.7 for label in LABELS},
            "rationale": {label: "supported" for label in LABELS},
        }
        plan = pipeline.plan_branches(
            tree_id="tree_91",
            topic_spec={"topic_id": "courtroom", "world_bible": {}},
            profile=profile,
            source_node=source_node,
            target_ids=["aug_1", "aug_2"],
            forced_intents=["reframe", "zoom_in"],
            affordance_assessment=assessment,
        )
        self.assertEqual(
            ["reframe", "zoom_in"],
            [edge["closed_intent"] for edge in plan["edges"]],
        )
        for edge in plan["edges"]:
            self.assertEqual(
                "profile_grounded_augmentation",
                edge["annotation_metadata"]["assignment_policy"],
            )
            self.assertIn("node_affordance", edge["private_generation_metadata"])

    def test_forced_intent_length_must_match_targets(self) -> None:
        pipeline = DatasetPipeline(
            config=PipelineConfig(),
            output_dir=Path("/tmp/minority-augmentation-test"),
            dry_run=True,
        )
        with self.assertRaises(ValueError):
            pipeline.plan_branches(
                tree_id="tree",
                topic_spec={"topic_id": "topic", "world_bible": {}},
                profile={"user_id": "1", "stable_profile": {}},
                source_node={},
                target_ids=["a", "b"],
                forced_intents=["reframe"],
            )

    def test_canary_selects_one_candidate_per_non_interact_label(self) -> None:
        rows = []
        for label_index, label in enumerate(LABELS[:-1]):
            for rank in range(2):
                rows.append(
                    {
                        "augmentation_id": f"{label}_{rank}",
                        "target_intent": label,
                        "profile_affordance_joint_score": label_index + rank,
                    }
                )
        selected = select_canary(rows)
        self.assertEqual(5, len(selected))
        self.assertEqual(set(LABELS[:-1]), {row["target_intent"] for row in selected})
        self.assertTrue(all(row["augmentation_id"].endswith("_1") for row in selected))

    def test_visual_acceptance_requires_intent_and_both_thresholds(self) -> None:
        calibration = {
            "calibration": {"intent_matches": True},
            "delta_alignment": {
                "image_intent_alignment": 0.8,
                "continuity_alignment": 0.7,
            },
        }
        self.assertTrue(
            accepted_visual_calibration(
                calibration,
                min_visual_alignment=0.65,
                min_continuity_alignment=0.65,
            )
        )
        calibration["calibration"]["intent_matches"] = False
        self.assertFalse(
            accepted_visual_calibration(
                calibration,
                min_visual_alignment=0.65,
                min_continuity_alignment=0.65,
            )
        )


if __name__ == "__main__":
    unittest.main()
