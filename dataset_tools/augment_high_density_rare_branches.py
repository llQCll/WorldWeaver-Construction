from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import augment_existing_minority_branches as base  # noqa: E402
from dataset_pipeline import DatasetPipeline, write_json, write_jsonl  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def naturalness_prompt_contract(target_intent: str) -> str:
    intent_invariants = {
        "zoom_in": (
            "Change only viewing scale toward an already visible detail; "
            "do not add hidden knowledge or advance time."
        ),
        "reframe": (
            "Change only camera or character viewpoint at the same moment; "
            "preserve world state and do not advance time."
        ),
        "reveal": (
            "Reveal previously hidden or unknown information that is already motivated by the scene; "
            "do not substitute a simple close-up."
        ),
    }
    invariant = intent_invariants.get(target_intent, f"Express only {target_intent}.")
    return (
        "Use case: illustration-story.\n"
        "Input image role: continuity reference and source story state.\n"
        f"Single semantic transformation: {invariant}\n"
        "Continuity invariants: preserve established character identity, clothing, signature objects, location logic, "
        "visual style, and every story fact not changed by the target intent.\n"
        "Profile invariant: visual emphasis and emotional tone must remain supported by the supplied user profile "
        "and current state.\n"
        "Avoid: unrelated plot events, gratuitous spectacle, arbitrary new characters or props, UI, HUD, choice "
        "boxes, interaction frames, hotspots, cursors, click markers, buttons, labels, captions, dialogue balloons, "
        "readable text, borders, logos, and watermarks."
    )


class NaturalnessDatasetPipeline(DatasetPipeline):
    def plan_branches(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        result = super().plan_branches(*args, **kwargs)
        forced = list(kwargs.get("forced_intents") or [])
        for index, edge in enumerate(result.get("edges", [])):
            target_intent = forced[index] if index < len(forced) else str(edge["closed_intent"])
            edge["generation_prompt"] = (
                str(edge["generation_prompt"])
                + "\n\nDataset naturalness contract:\n"
                + naturalness_prompt_contract(target_intent)
            )
        return result


def strict_visual_acceptance(
    calibration: dict[str, Any],
    *,
    min_visual_alignment: float,
    min_continuity_alignment: float,
    min_profile_alignment: float,
) -> bool:
    return bool(
        base.accepted_visual_calibration(
            calibration,
            min_visual_alignment=min_visual_alignment,
            min_continuity_alignment=min_continuity_alignment,
        )
        and float(calibration["delta_alignment"].get("profile_alignment", 0.0))
        >= min_profile_alignment
    )


def install_profile_gate(min_profile_alignment: float) -> None:
    original = base.accepted_visual_calibration

    def gated(
        calibration: dict[str, Any],
        *,
        min_visual_alignment: float,
        min_continuity_alignment: float,
    ) -> bool:
        return bool(
            original(
                calibration,
                min_visual_alignment=min_visual_alignment,
                min_continuity_alignment=min_continuity_alignment,
            )
            and float(calibration["delta_alignment"].get("profile_alignment", 0.0))
            >= min_profile_alignment
        )

    base.accepted_visual_calibration = gated


def enrich_record(record: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    record["profile_percentile"] = candidate.get("profile_percentile")
    record["profile_evidence"] = candidate.get("profile_evidence", {})
    record["naturalness_constraints"] = candidate.get("naturalness_constraints", {})
    record["selection_story_state"] = candidate.get("story_state", {})
    return record


def enrich_outputs(
    *,
    plan_dir: Path,
    output_dir: Path,
    min_profile_alignment: float,
) -> None:
    candidates = {
        str(row["augmentation_id"]): row
        for row in read_jsonl(plan_dir / "candidate_branches.jsonl")
    }
    for record_path in sorted((output_dir / "records").glob("*.json")):
        record = read_json(record_path)
        candidate = candidates.get(str(record.get("augmentation_id")))
        if candidate:
            write_json(record_path, enrich_record(record, candidate))

    for filename in ("manifest.jsonl", "usable_branches.jsonl"):
        path = output_dir / filename
        if not path.is_file():
            continue
        rows = []
        for record in read_jsonl(path):
            candidate = candidates.get(str(record.get("augmentation_id")))
            rows.append(enrich_record(record, candidate) if candidate else record)
        write_jsonl(path, rows)

    run_config_path = output_dir / "run_config.json"
    run_config = read_json(run_config_path)
    run_config["thresholds"]["min_profile_alignment"] = min_profile_alignment
    run_config["naturalness_contract"] = {
        "single_semantic_transformation": True,
        "profile_grounded": True,
        "story_state_grounded": True,
        "continuity_locked": True,
        "ui_forbidden": True,
    }
    write_json(run_config_path, run_config)

    summary_path = output_dir / "summary.json"
    summary = read_json(summary_path)
    summary["min_profile_alignment"] = min_profile_alignment
    summary["naturalness_contract_applied"] = True
    write_json(summary_path, summary)


def parse_wrapper_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--min-profile-alignment", type=float, default=0.8)
    return parser.parse_known_args()


def main() -> None:
    wrapper_args, remaining = parse_wrapper_args()
    sys.argv = [sys.argv[0], *remaining]
    base_args = base.parse_args()
    base.DatasetPipeline = NaturalnessDatasetPipeline
    install_profile_gate(wrapper_args.min_profile_alignment)
    base.main()
    enrich_outputs(
        plan_dir=Path(base_args.plan_dir).resolve(),
        output_dir=Path(base_args.output_dir).resolve(),
        min_profile_alignment=wrapper_args.min_profile_alignment,
    )


if __name__ == "__main__":
    main()
