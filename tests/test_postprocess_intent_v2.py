from __future__ import annotations

import unittest

from dataset_tools.postprocess_intent_v2 import (
    DEFAULT_CONFIG,
    annotation_issues,
    mock_annotation,
    normalize_annotation,
    normalize_box,
    public_config,
    text_prompt,
)


class PostprocessIntentV2Tests(unittest.TestCase):
    def test_gpt55_is_default_for_both_passes(self) -> None:
        self.assertEqual("gpt-5.5", DEFAULT_CONFIG["text_api"]["model"])
        self.assertEqual("gpt-5.5", DEFAULT_CONFIG["vision_api"]["model"])
        for section in ("text_api", "vision_api"):
            self.assertEqual("low", DEFAULT_CONFIG[section]["reasoning_effort"])
            self.assertEqual(2000, DEFAULT_CONFIG[section]["max_completion_tokens"])
            self.assertEqual(180, DEFAULT_CONFIG[section]["timeout_seconds"])

    def test_mock_annotation_obeys_strict_contract(self) -> None:
        edge = {
            "closed_intent": "reveal",
            "slots": {
                "action": "look behind the lifted panel",
                "target": "concealed compartment",
                "narrative_goal": "discover the hidden route",
                "mood": "curious",
                "continuity_constraint": "preserve the protagonist",
                "scope": "next_panel",
            },
            "profile_signal": {"mastery_logic": 0.5, "curiosity": 0.8},
            "expected_affective_delta": {"curiosity": 0.1},
            "expected_profile_delta": {"mastery_logic": 0.03},
        }
        annotation = normalize_annotation(mock_annotation(edge))
        self.assertEqual([], annotation_issues(annotation))
        self.assertNotIn("curiosity", annotation["profile_signal"])
        self.assertEqual("reveal", annotation["intent_ranking"][0]["intent"])

    def test_prompt_does_not_include_legacy_annotation_fields(self) -> None:
        evidence = {
            "action": "follow the courier",
            "target": "courier",
            "narrative_goal": "continue the current trail",
        }
        prompt = text_prompt(evidence, pass_index=1)
        self.assertNotIn("legacy_closed_intent", prompt)
        self.assertNotIn("generation_prompt", prompt)

    def test_box_normalization_repairs_inverted_coordinates(self) -> None:
        box, changed = normalize_box([1.2, 0.8, -0.2, 0.1])
        self.assertTrue(changed)
        self.assertLess(box[0], box[2])
        self.assertLess(box[1], box[3])
        self.assertTrue(all(0.0 <= value <= 1.0 for value in box))

    def test_public_config_removes_private_connection_fields(self) -> None:
        config = {
            "text_api": {"base_url": "private", "api_key": "secret", "model": "gpt-5.5"},
            "vision_api": {"base_url": "private", "api_key": "secret", "model": "gpt-5.5"},
            "processing": {},
        }
        cleaned = public_config(config)
        self.assertNotIn("base_url", cleaned["text_api"])
        self.assertNotIn("api_key", cleaned["text_api"])
        self.assertNotIn("base_url", cleaned["vision_api"])
        self.assertNotIn("api_key", cleaned["vision_api"])


if __name__ == "__main__":
    unittest.main()
