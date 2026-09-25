import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from dataset_tools.dataset_pipeline import DatasetPipeline, PipelineConfig


class VisualCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pipeline = DatasetPipeline(
            config=PipelineConfig(), output_dir=self.root, dry_run=True
        )
        self.edge = {
            "source_node": "n0", "target_node": "n1", "closed_intent": "reveal",
            "expected_affective_delta": {"curiosity": 0.1},
            "expected_profile_delta": {"world_discovery": 0.1},
        }

    def assess(self, intent="reveal", score=0.9, flag=False):
        self.pipeline.dry_run = False
        self.pipeline.llm = Mock()
        self.pipeline.llm.chat_json.return_value = {
            "observed_closed_intent": intent,
            "observed_affective_delta": {"curiosity": 0.2},
            "observed_profile_delta": {"world_discovery": 0.2},
            "delta_alignment": {
                "profile_alignment": score, "affect_alignment": score,
                "image_intent_alignment": score, "continuity_alignment": score,
                "ending_diversity": score, "needs_revision": flag,
            },
        }
        result = self.pipeline.calibrate_edge(
            edge=self.edge, source_node={"image": "source.png"},
            target_node={"image": "target.png"},
        )
        self.assertEqual(
            self.pipeline.llm.chat_json.call_args.kwargs["images"],
            ["source.png", "target.png"],
        )
        return result

    def test_low_alignment_cannot_be_overridden(self):
        self.assertTrue(self.assess(score=0.2)["delta_alignment"]["needs_revision"])

    def test_intent_mismatch_cannot_be_overridden(self):
        self.assertTrue(self.assess(intent="follow")["delta_alignment"]["needs_revision"])

    def test_positive_model_review_flag_is_preserved(self):
        self.assertTrue(self.assess(flag=True)["delta_alignment"]["needs_revision"])

    def test_supported_transition_blends_observations(self):
        result = self.assess()
        self.assertFalse(result["delta_alignment"]["needs_revision"])
        self.assertAlmostEqual(result["final_affective_delta"]["curiosity"], 0.14)
        self.assertAlmostEqual(result["final_profile_delta"]["world_discovery"], 0.14)

    def test_missing_images_are_retried_and_completed_edges_skipped(self):
        tree = {"nodes": [
            {"node_id": "n0", "image": "source.png"},
            {"node_id": "n1", "image": "target.png"},
        ], "edges": [dict(self.edge)]}
        tree_path = self.root / "tree.json"
        tree_path.write_text(json.dumps(tree))
        (self.root / "manifest.jsonl").write_text(json.dumps({"tree_path": str(tree_path)}) + "\n")
        with patch("dataset_tools.dataset_pipeline.image_exists_for_stage", return_value=False):
            self.pipeline.calibrate_run(run_dir=self.root)
        self.assertEqual(json.loads(tree_path.read_text())["edges"][0]["calibration"]["status"], "missing_images")
        with patch("dataset_tools.dataset_pipeline.image_exists_for_stage", return_value=True):
            self.pipeline.calibrate_run(run_dir=self.root)
        self.assertEqual(json.loads(tree_path.read_text())["edges"][0]["calibration"]["status"], "calibrated")
        with patch.object(self.pipeline, "calibrate_edge") as assess:
            self.pipeline.calibrate_run(run_dir=self.root)
            assess.assert_not_called()


if __name__ == "__main__":
    unittest.main()
