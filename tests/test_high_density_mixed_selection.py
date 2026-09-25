from __future__ import annotations

import unittest

from dataset_tools.select_high_density_mixed_augmentations import (
    mixed_heuristic_affordance,
)


class HighDensityMixedSelectionTests(unittest.TestCase):
    def test_interact_affordance_uses_visible_action_evidence(self) -> None:
        tree = {
            "edges": [
                {
                    "source_node": "n1",
                    "closed_intent": "interact",
                    "slots": {"action": "repair and activate the visible mechanism"},
                    "grounding": {"target_box": [0.1, 0.1, 0.3, 0.3]},
                }
            ]
        }
        score = mixed_heuristic_affordance(
            label="interact",
            tree=tree,
            node={"node_id": "n1", "depth": 1, "story_state": {}},
        )
        self.assertGreater(score, 5.0)


if __name__ == "__main__":
    unittest.main()
