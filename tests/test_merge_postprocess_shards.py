from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset_tools.merge_postprocess_intent_v2_shards import merge_shards


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class MergePostprocessShardsTests(unittest.TestCase):
    def test_merges_disjoint_shards_and_recomputes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            batch = Path(temporary)
            for index, directory_name in enumerate(
                ("postprocessed_intent_v2", "postprocessed_intent_v2_part_1")
            ):
                directory = batch / directory_name
                tree_id = f"tree_{index}"
                output_tree = f"trees/user_{index}/topic/tree.json"
                write_json(directory / "summary.json", {"dry_run": False, "tree_count": 1})
                write_json(directory / "config.public.json", {"text_api": {"model": "gpt-5.5"}})
                write_json(directory / output_tree, {"tree_id": tree_id})
                write_jsonl(
                    directory / "manifest.jsonl",
                    [{"tree_id": tree_id, "output_tree": output_tree}],
                )
                write_jsonl(
                    directory / "relabel_audit.jsonl",
                    [
                        {
                            "tree_id": tree_id,
                            "edge_id": "edge_0",
                            "legacy_closed_intent": "zoom_in",
                            "intended_closed_intent": "interact",
                            "observed_visual_intent": "interact",
                            "label_changed": True,
                        }
                    ],
                )
                write_jsonl(directory / "review_queue.jsonl", [] if index == 0 else [{"tree_id": tree_id}])

            summary = merge_shards(batch, shard_count=2, expected_trees_per_shard=1)

            self.assertEqual(2, summary["tree_count"])
            self.assertEqual(2, summary["edge_count"])
            self.assertEqual(1, summary["review_count"])
            self.assertEqual(2, summary["changed_label_count"])
            self.assertTrue(
                (batch / "postprocessed_intent_v2/trees/user_1/topic/tree.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
