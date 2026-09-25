from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from dataset_tools.dataset_pipeline import (
    DatasetPipeline,
    PipelineConfig,
    apply_image_rendering_constraints,
    mock_branch_plan,
    normalize_closed_intent,
    sample_hierarchical_intents,
    user_intent_prior,
    target_intent_for_edge,
    validate_branch_plan,
)


class IntentV2Tests(unittest.TestCase):
    def test_each_default_tree_is_nearly_balanced(self) -> None:
        intents = [
            target_intent_for_edge(tree_id="topic_user_42", target_id=f"n{index}")
            for index in range(1, 18)
        ]
        counts = Counter(intents)
        self.assertEqual(6, len(counts))
        self.assertEqual(2, min(counts.values()))
        self.assertEqual(3, max(counts.values()))

    def test_nineteen_node_tree_is_exactly_balanced(self) -> None:
        intents = [
            target_intent_for_edge(tree_id="topic_user_42", target_id=f"n{index}")
            for index in range(1, 19)
        ]
        self.assertEqual({3}, set(Counter(intents).values()))

    def test_image_rendering_constraints_are_idempotent(self) -> None:
        prompt = apply_image_rendering_constraints("Continue the scene.")
        self.assertIn("Do not include UI, HUD", prompt)
        self.assertEqual(prompt, apply_image_rendering_constraints(prompt))

    def test_mock_plan_passes_strict_validation(self) -> None:
        target_ids = ["n1", "n2", "n3"]
        required = [
            target_intent_for_edge(tree_id="topic_user_42", target_id=target_id)
            for target_id in target_ids
        ]
        plan = mock_branch_plan(
            target_ids=target_ids,
            source_node={"node_id": "n0"},
            required_intents=required,
        )
        self.assertEqual([], validate_branch_plan(plan["edges"], required_intents=required))
        self.assertTrue(plan["oos_region"]["decision"]["requires_confirmation"])

    def test_explicit_label_is_not_overwritten_by_keywords(self) -> None:
        edge = {
            "closed_intent": "reframe",
            "slots": {"action": "open the door and follow the guide"},
            "intent_ranking": [{"intent": "interact"}],
        }
        self.assertEqual("reframe", normalize_closed_intent(edge))

    def test_invalid_label_does_not_default_to_reframe(self) -> None:
        with self.assertRaises(ValueError):
            normalize_closed_intent({"closed_intent": "continue_story", "intent_ranking": []})

    def test_validator_rejects_top1_mismatch(self) -> None:
        plan = mock_branch_plan(
            target_ids=["n1"],
            source_node={"node_id": "n0"},
            required_intents=["reveal"],
        )
        plan["edges"][0]["intent_ranking"][0]["intent"] = "zoom_in"
        issues = validate_branch_plan(plan["edges"], required_intents=["reveal"])
        self.assertTrue(any("intent_ranking" in issue for issue in issues))

    def test_dry_run_calibration_preserves_visual_metrics(self) -> None:
        pipeline = DatasetPipeline(
            config=PipelineConfig(),
            output_dir=Path("/tmp/intent-v2-test"),
            dry_run=True,
        )
        result = pipeline.calibrate_edge(
            edge={
                "closed_intent": "reveal",
                "expected_affective_delta": {"curiosity": 0.1},
                "expected_profile_delta": {"mastery_logic": 0.05},
            },
            source_node={},
            target_node={},
        )
        self.assertEqual("reveal", result["calibration"]["observed_closed_intent"])
        self.assertTrue(result["calibration"]["intent_matches"])
        self.assertIn("image_intent_alignment", result["delta_alignment"])
        self.assertIn("continuity_alignment", result["delta_alignment"])
        self.assertIn("ending_diversity", result["delta_alignment"])

    def test_affordance_prompt_builds_with_real_client_path(self) -> None:
        class FakeClient:
            def chat_json(self, *, system_prompt, user_prompt, images=None):
                self.user_prompt = user_prompt
                return {
                    "topic_affinity": {label: 0.7 for label in ["zoom_in", "reveal", "branch_out", "reframe", "follow", "interact"]},
                    "node_affordance": {label: 0.6 for label in ["zoom_in", "reveal", "branch_out", "reframe", "follow", "interact"]},
                    "rationale": {},
                }

        pipeline = DatasetPipeline(config=PipelineConfig(), output_dir=Path("/tmp/intent-v2-test"))
        fake = FakeClient()
        pipeline.llm = fake
        result = pipeline.assess_intent_affordance(
            tree_id="courtroom_user_503",
            topic_spec={
                "topic_id": "courtroom",
                "topic": "Courtroom",
                "world_bible": {"premise": "A contested hearing."},
            },
            source_node={"node_id": "n0", "depth": 0, "story_state": {}, "image": "/missing.png"},
        )
        self.assertEqual(0.6, result["node_affordance"]["reframe"])
        self.assertIn("A contested hearing", fake.user_prompt)

    def test_parallel_image_generation_preserves_depth_dependencies(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            root_image = run_dir / "n0.png"
            root_image.write_bytes(b"root")
            n1 = run_dir / "n1.png"
            n2 = run_dir / "n2.png"
            n3 = run_dir / "n3.png"
            tree_path = run_dir / "tree.json"
            tree = {
                "nodes": [
                    {"node_id": "n0", "depth": 0, "image": str(root_image)},
                    {"node_id": "n1", "depth": 1, "image": str(n1)},
                    {"node_id": "n2", "depth": 2, "image": str(n2)},
                    {"node_id": "n3", "depth": 1, "image": str(n3)},
                ],
                "edges": [
                    {"source_node": "n0", "target_node": "n1", "generation_prompt": "first"},
                    {"source_node": "n1", "target_node": "n2", "generation_prompt": "second"},
                    {"source_node": "n0", "target_node": "n3", "generation_prompt": "sibling"},
                ],
            }
            tree_path.write_text(json.dumps(tree), encoding="utf-8")
            (run_dir / "manifest.jsonl").write_text(
                json.dumps({"tree_path": str(tree_path)}) + "\n", encoding="utf-8"
            )
            (run_dir / "topics.json").write_text("[]", encoding="utf-8")

            pipeline = DatasetPipeline(
                config=PipelineConfig(image_generation_workers=2),
                output_dir=run_dir,
            )
            calls: list[str] = []

            def generate(*, prompt, output_path, references):
                self.assertTrue(all(reference.exists() for reference in references))
                output_path.write_bytes(prompt.encode("utf-8"))
                calls.append(prompt)

            pipeline.maybe_generate_image = generate
            pipeline.generate_images_for_run(run_dir=run_dir)
            self.assertTrue(n1.exists())
            self.assertTrue(n2.exists())
            self.assertTrue(n3.exists())
            self.assertGreater(calls.index("second"), calls.index("first"))
            self.assertGreater(calls.index("second"), calls.index("sibling"))

    def test_hierarchical_sampling_is_reproducible_and_gates_low_affordance(self) -> None:
        stable_profile = {
            "goal_progress": 0.6,
            "mastery_logic": 0.7,
            "challenge_seeking": 0.4,
            "social_attachment": 0.5,
            "cooperative_orientation": 0.5,
            "world_discovery": 0.8,
            "role_immersion": 0.7,
            "aesthetic_customization": 0.8,
        }
        user_prior = user_intent_prior(user_id="503", stable_profile=stable_profile)
        kwargs = {
            "tree_id": "topic_user_503",
            "source_id": "n0",
            "target_ids": ["n1", "n2", "n3"],
            "user_prior": user_prior,
            "topic_affinity": {label: 0.7 for label in user_prior},
            "node_affordance": {
                "zoom_in": 0.8,
                "reveal": 0.7,
                "branch_out": 0.65,
                "reframe": 0.01,
                "follow": 0.75,
                "interact": 0.8,
            },
            "state_adjustment": {label: 1.0 for label in user_prior},
            "intent_counts": {label: 0 for label in user_prior},
            "recent_intents": [],
            "total_edges": 18,
            "seed": 20260804,
            "temperature": 0.85,
            "quota_strength": 0.7,
            "min_affordance": 0.2,
        }
        first, first_metadata = sample_hierarchical_intents(**kwargs)
        second, second_metadata = sample_hierarchical_intents(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first_metadata, second_metadata)
        self.assertNotIn("reframe", first)
        self.assertEqual(0.0, first_metadata[0]["final_distribution"]["reframe"])

    def test_hierarchical_sampling_is_aggregate_balanced_not_exact_per_tree(self) -> None:
        totals = Counter()
        histograms = set()
        for user_index in range(36):
            stable_profile = {
                dimension: 0.15 + ((user_index * 17 + dim_index * 11) % 70) / 100
                for dim_index, dimension in enumerate(
                    [
                        "goal_progress", "mastery_logic", "challenge_seeking", "social_attachment",
                        "cooperative_orientation", "world_discovery", "role_immersion", "aesthetic_customization",
                    ]
                )
            }
            prior = user_intent_prior(user_id=str(500 + user_index), stable_profile=stable_profile)
            local = {label: 0 for label in prior}
            recent: list[str] = []
            for source_index, branch_count in enumerate([3, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1]):
                affordance = {
                    label: 0.30 + ((user_index * 13 + source_index * 7 + label_index * 19) % 65) / 100
                    for label_index, label in enumerate(prior)
                }
                labels, _metadata = sample_hierarchical_intents(
                    tree_id=f"tree_{user_index}", source_id=f"n{source_index}",
                    target_ids=[f"x{source_index}_{offset}" for offset in range(branch_count)],
                    user_prior=prior, topic_affinity={label: 0.65 for label in prior},
                    node_affordance=affordance, state_adjustment={label: 1.0 for label in prior},
                    intent_counts=local, recent_intents=recent, total_edges=18, seed=20260804,
                    temperature=0.85, quota_strength=0.7, min_affordance=0.2,
                )
                for label in labels:
                    local[label] += 1
                    recent.append(label)
                    totals[label] += 1
            histograms.add(tuple(local[label] for label in prior))
        self.assertGreater(len(histograms), 20)
        self.assertLessEqual(max(totals.values()) - min(totals.values()), 20)



if __name__ == "__main__":
    unittest.main()
