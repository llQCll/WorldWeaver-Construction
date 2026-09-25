from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dataset_pipeline import DatasetPipeline, PipelineConfig, read_profiles, write_jsonl


def profile_richness(row: dict[str, Any]) -> float:
    profile_text = str(row.get("profile", ""))
    positive_count = float(row.get("positive_count", 0) or 0)
    negative_count = float(row.get("negative_count", 0) or 0)
    # Favor profiles with enough evidence, including both positive and negative signals.
    return len(profile_text) / 1000.0 + positive_count * 1.0 + negative_count * 1.25


def select_profiles(rows: list[dict[str, Any]], *, limit: int, user_id: str | None) -> list[dict[str, Any]]:
    if user_id:
        selected = [row for row in rows if str(row.get("user_id")) == str(user_id)]
        if not selected:
            raise RuntimeError(f"Could not find user_id={user_id!r}.")
        return selected[:limit]
    ranked = sorted(rows, key=profile_richness, reverse=True)
    return ranked[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select informative raw user profiles and optionally map them into dataset dimensions."
    )
    parser.add_argument("--profiles", default="../data/user_profiles.jsonl", help="Input raw user profile JSONL.")
    parser.add_argument("--limit", type=int, default=5, help="Number of profiles to select.")
    parser.add_argument("--user-id", default="", help="Select a specific user_id instead of ranking by richness.")
    parser.add_argument("--output-dir", default="profile_selection", help="Output directory.")
    parser.add_argument("--normalize", action="store_true", help="Map selected profiles to stable/affective dimensions.")
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic mock normalized dimensions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    profile_path = Path(args.profiles)
    if not profile_path.is_absolute():
        profile_path = (base_dir / profile_path).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (base_dir / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_profiles(profile_path)
    selected = select_profiles(rows, limit=args.limit, user_id=args.user_id or None)
    selected_rows = [
        {
            **row,
            "selection_score": round(profile_richness(row), 4),
        }
        for row in selected
    ]
    write_jsonl(output_dir / "selected_profiles.jsonl", selected_rows)

    if args.normalize:
        pipeline = DatasetPipeline(
            config=PipelineConfig(),
            output_dir=output_dir,
            dry_run=args.dry_run,
        )
        normalized = [pipeline.normalize_profile(row) for row in selected]
        write_jsonl(output_dir / "selected_profiles_normalized.jsonl", normalized)

    summary = {
        "input": str(profile_path),
        "selected_count": len(selected_rows),
        "normalize": args.normalize,
        "dry_run": args.dry_run,
        "outputs": {
            "selected_profiles": str(output_dir / "selected_profiles.jsonl"),
            "selected_profiles_normalized": str(output_dir / "selected_profiles_normalized.jsonl")
            if args.normalize
            else None,
        },
    }
    (output_dir / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
