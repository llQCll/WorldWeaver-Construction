from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset_tools.closed_intent_repair import load_repair_config, repair_closed_intents


class FakeClient:
    def chat_json(self, **_: object) -> dict[str, object]:
        return {
            "closed_intent": "branch_out",
            "confidence": 0.75,
            "rationale": "The next panel enters a distinct route.",
        }


class ClosedIntentRepairTests(unittest.TestCase):
    def test_generation_config_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "llm_base_url": "https://example.invalid/v1",
                        "llm_api_key": "private",
                        "llm_model": "gpt-5.5",
                        "timeout_seconds": 90,
                    }
                ),
                encoding="utf-8",
            )
            config = load_repair_config(path)
            self.assertEqual("https://example.invalid/v1", config["text_api"]["base_url"])
            self.assertEqual("private", config["text_api"]["api_key"])
            self.assertEqual("gpt-5.5", config["text_api"]["model"])
            self.assertEqual(90, config["text_api"]["timeout_seconds"])

    def test_repair_creates_overlay_and_preserves_source(self) -> None:
        tree = {
            "tree_id": "topic_user_1",
            "user_id": "1",
            "topic_id": "topic",
            "world_bible": {},
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
                    "intent_ranking": [{"rank": 1, "intent": "follow", "score": 0.9}],
                    "generation_prompt": "Enter a newly opened shaft route.",
                    "slots": {"action": "enter", "target": "shaft", "narrative_goal": "take a new route"},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "run/users/user_1/topic/tree.json"
            source.parent.mkdir(parents=True)
            source.write_text(json.dumps(tree), encoding="utf-8")
            source_before = source.read_bytes()
            output = root / "run/closed_intent_only_v1"
            summary = repair_closed_intents(
                source_root=root / "run",
                tree_paths=[source],
                output_dir=output,
                config={"text_api": {"model": "test"}},
                client=FakeClient(),
            )

            self.assertEqual(source_before, source.read_bytes())
            repaired = json.loads((output / "trees/user_1/topic/tree.json").read_text())
            self.assertEqual("branch_out", repaired["edges"][0]["closed_intent"])
            self.assertEqual(tree["edges"][0]["intent_ranking"], repaired["edges"][0]["intent_ranking"])
            self.assertEqual(1, summary["changed_label_count"])
            self.assertEqual(1, summary["review_queue_count"])
            self.assertEqual(1, summary["preserved_top1_mismatch_count"])
            self.assertFalse(summary["strict_intent_v2_compatible"])
            queue = (output / "label_review_queue.jsonl").read_text()
            self.assertIn("changed_label_below_0.80", queue)


if __name__ == "__main__":
    unittest.main()
