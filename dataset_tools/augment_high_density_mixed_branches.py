from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import augment_existing_minority_branches as base
import augment_high_density_rare_branches as strict
import select_high_density_mixed_augmentations as mixed


MIXED_LABELS = ("zoom_in", "reframe", "reveal", "interact", "branch_out", "follow")


def select_mixed_canary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for label in MIXED_LABELS:
        matching = [row for row in rows if str(row.get("target_intent")) == label]
        if not matching:
            continue
        selected.append(
            sorted(
                matching,
                key=lambda row: (
                    -float(row.get("profile_affordance_joint_score", 0.0)),
                    str(row["augmentation_id"]),
                ),
            )[0]
        )
    return selected


def main() -> None:
    base.select_canary = select_mixed_canary
    base.heuristic_affordance = mixed.mixed_heuristic_affordance
    strict.main()


if __name__ == "__main__":
    main()
