from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset_tools.merge_closed_intent_only_shards import merge_parts


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class MergeClosedIntentOnlyTests(unittest.TestCase):
    def test_merges_parts_and_preserves_label_only_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            batch = Path(temporary)
            for index, name in enumerate(("closed_intent_only_v1", "closed_intent_only_v1_part_1")):
                directory = batch / name
                tree_id = f"tree_{index}"
                output_tree = f"trees/user_{index}/topic/tree.json"
                write_json(
                    directory / "summary.json",
                    {
                        "tree_count": 1,
                        "model": "gpt-5.5",
                        "api_call_count": 1,
                        "reused_full_cache_count": 0,
                        "images_sent": 0,
                        "other_fields_modified": False,
                    },
                )
                write_json(directory / output_tree, {"tree_id": tree_id})
                write_jsonl(directory / "manifest.jsonl", [{"tree_id": tree_id, "output_tree": output_tree}])
                write_jsonl(
                    directory / "label_audit.jsonl",
                    [
                        {
                            "tree_id": tree_id,
                            "edge_id": "edge_0",
                            "legacy_closed_intent": "follow",
                            "repaired_closed_intent": "branch_out",
                            "label_changed": True,
                        }
                    ],
                )

            summary = merge_parts(batch, part_count=2, expected_trees_per_part=1)

            self.assertEqual(2, summary["tree_count"])
            self.assertEqual(2, summary["edge_count"])
            self.assertEqual(2, summary["api_call_count"])
            self.assertEqual(0, summary["images_sent"])
            self.assertFalse(summary["other_fields_modified"])
            self.assertTrue((batch / "closed_intent_only_v1/trees/user_1/topic/tree.json").is_file())


if __name__ == "__main__":
    unittest.main()
