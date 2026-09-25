from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


CLOSED_INTENTS = {
    "zoom_in",
    "reveal",
    "branch_out",
    "reframe",
    "follow",
    "interact",
}

PROFILE_DIMS = {
    "goal_progress",
    "mastery_logic",
    "challenge_seeking",
    "social_attachment",
    "cooperative_orientation",
    "world_discovery",
    "role_immersion",
    "aesthetic_customization",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def active_tree_paths(batch_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for status_path in sorted(batch_dir.glob("user_*/job_status.json")):
        status = read_json(status_path)
        run_name = str(status.get("run_dir", "")).replace("\\", "/").rstrip("/").split("/")[-1]
        if not run_name:
            continue
        run_dir = status_path.parent / "runs" / run_name
        paths.extend(sorted(run_dir.glob("users/user_*/*/tree.json")))
    return paths


def audit(batch_dir: Path) -> dict[str, Any]:
    trees = active_tree_paths(batch_dir)
    class_counts: Counter[str] = Counter()
    ranking_lengths: Counter[int] = Counter()
    decision_types: Counter[str] = Counter()
    invalid_profile_signal_keys: Counter[str] = Counter()
    total_edges = 0
    missing_natural_language_intent = 0
    ranking_top1_mismatch = 0
    ranking_unsorted = 0
    ranking_duplicate_labels = 0
    calibrated_edges = 0

    for tree_path in trees:
        tree = read_json(tree_path)
        for edge in tree.get("edges", []):
            total_edges += 1
            closed_intent = str(edge.get("closed_intent", ""))
            class_counts[closed_intent] += 1
            decision_types[str(edge.get("decision", {}).get("type", "missing"))] += 1
            if not str(edge.get("natural_language_intent", "")).strip():
                missing_natural_language_intent += 1

            ranking = edge.get("intent_ranking", [])
            ranking_lengths[len(ranking)] += 1
            labels = [str(item.get("intent", "")).strip().lower() for item in ranking]
            if not labels or labels[0] != closed_intent:
                ranking_top1_mismatch += 1
            if len(labels) != len(set(labels)):
                ranking_duplicate_labels += 1
            try:
                scores = [float(item.get("score")) for item in ranking]
                if scores != sorted(scores, reverse=True):
                    ranking_unsorted += 1
            except (TypeError, ValueError):
                ranking_unsorted += 1

            for key in edge.get("profile_signal", {}):
                if key not in PROFILE_DIMS:
                    invalid_profile_signal_keys[key] += 1
            if edge.get("calibration", {}).get("status") == "calibrated":
                calibrated_edges += 1

    missing_classes = sorted(CLOSED_INTENTS - set(class_counts))
    return {
        "batch_dir": str(batch_dir),
        "active_tree_count": len(trees),
        "edge_count": total_edges,
        "class_counts": dict(class_counts),
        "missing_classes": missing_classes,
        "ranking_lengths": {str(key): value for key, value in sorted(ranking_lengths.items())},
        "ranking_top1_mismatch": ranking_top1_mismatch,
        "ranking_unsorted": ranking_unsorted,
        "ranking_duplicate_labels": ranking_duplicate_labels,
        "missing_natural_language_intent": missing_natural_language_intent,
        "invalid_profile_signal_keys": dict(invalid_profile_signal_keys),
        "decision_types": dict(decision_types),
        "calibrated_edges": calibrated_edges,
        "calibration_coverage": round(calibrated_edges / total_edges, 6) if total_edges else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit intent annotations from active runs in a batch.")
    parser.add_argument("--batch-dir", required=True, help="Batch directory containing user_*/job_status.json.")
    parser.add_argument("--output", default="", help="Optional JSON report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(Path(args.batch_dir))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
