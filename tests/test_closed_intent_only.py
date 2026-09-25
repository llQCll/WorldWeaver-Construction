from __future__ import annotations

import copy
import unittest

from dataset_tools.postprocess_closed_intent_only import (
    assert_only_labels_changed,
    compact_evidence,
    label_issues,
    label_prompt,
)


class ClosedIntentOnlyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = {
            "tree_id": "tree_0",
            "world_bible": {"premise": "A sealed station"},
            "nodes": [
                {"node_id": "n0", "story_state": {"location": "deck"}},
                {"node_id": "n1", "story_state": {"location": "shaft"}},
            ],
            "edges": [
                {
                    "edge_id": "n0_to_n1",
                    "source_node": "n0",
                    "target_node": "n1",
                    "closed_intent": "follow",
                    "branch_label": "Enter the shaft",
                    "generation_prompt": "Show the protagonist entering the newly opened shaft route.",
                    "slots": {"action": "enter", "target": "shaft", "narrative_goal": "take a new route"},
                    "intent_ranking": [{"intent": "follow", "rank": 1, "score": 0.7}],
                    "profile_signal": {"world_discovery": 0.4},
                }
            ],
        }

    def test_allows_only_closed_intent_to_change(self) -> None:
        repaired = copy.deepcopy(self.tree)
        repaired["edges"][0]["closed_intent"] = "branch_out"
        assert_only_labels_changed(self.tree, repaired)
        repaired["edges"][0]["slots"]["action"] = "changed"
        with self.assertRaisesRegex(ValueError, "other than closed_intent"):
            assert_only_labels_changed(self.tree, repaired)

    def test_prompt_excludes_legacy_label_and_profile_fields(self) -> None:
        evidence = compact_evidence(self.tree, self.tree["edges"][0])
        prompt = label_prompt(evidence)
        self.assertIn("newly opened shaft route", prompt)
        self.assertNotIn('"closed_intent": "follow"', prompt)
        self.assertNotIn("profile_signal", prompt)
        self.assertNotIn("intent_ranking", prompt)

    def test_minimal_label_contract(self) -> None:
        self.assertEqual(
            [],
            label_issues(
                {"closed_intent": "branch_out", "confidence": 0.9, "rationale": "A new route is entered."}
            ),
        )


if __name__ == "__main__":
    unittest.main()
