from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from dataset_pipeline import (
    DEFAULT_TOPICS,
    DatasetPipeline,
    PipelineConfig,
    public_config_snapshot,
    write_json,
)

try:
    from closed_intent_repair import SCHEMA_VERSION, load_repair_config, repair_closed_intents, run_tree_paths
except ModuleNotFoundError:
    from dataset_tools.closed_intent_repair import SCHEMA_VERSION, load_repair_config, repair_closed_intents, run_tree_paths


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve_under_base(base_dir: Path, path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_config_from_json(path: Path) -> PipelineConfig:
    data = read_json(path)
    allowed = set(PipelineConfig.__dataclass_fields__.keys())
    values = {key: value for key, value in data.items() if key in allowed}
    return PipelineConfig(**values)


def safe_config(config: PipelineConfig) -> dict[str, Any]:
    return public_config_snapshot(config)


def timed_stage(name: str, fn) -> float:
    start = time.perf_counter()
    print(f"[stage] {name} started")
    fn()
    elapsed = round(time.perf_counter() - start, 3)
    print(f"[stage] {name} finished in {elapsed}s")
    return elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run text planning, image generation, and VL calibration as one end-to-end dataset job."
    )
    parser.add_argument("--profiles", default="user_profiles(1).jsonl", help="Input JSONL profile file.")
    parser.add_argument("--output-dir", default="dataset_runs_e2e", help="Output directory for the new run.")
    parser.add_argument("--profile-limit", type=int, default=1, help="Number of users to process.")
    parser.add_argument("--profile-user-id", default="", help="Only process this user_id from the profile JSONL.")
    parser.add_argument("--topics", nargs="*", default=DEFAULT_TOPICS[:1], help="Topic names.")
    parser.add_argument("--topics-per-user", type=int, default=1, help="How many topics each user receives.")
    parser.add_argument("--root-assets-dir", default="", help="Optional curated topic root assets folder.")
    parser.add_argument(
        "--profile-overrides-json",
        default="",
        help="Optional controlled profile variants keyed by user_id.",
    )
    parser.add_argument("--branch-depth", type=int, default=5, help="Maximum tree depth, root depth is 0.")
    parser.add_argument("--tree-nodes", type=int, default=18, help="Maximum nodes per user-topic tree.")
    parser.add_argument("--image-size", default="1024x1024")
    parser.add_argument("--image-quality", default="high")
    parser.add_argument("--image-generation-workers", type=int, default=1)
    parser.add_argument(
        "--intent-sampling-policy",
        choices=["balanced_deterministic", "balanced_counterfactual", "hierarchical_natural"],
        default="",
        help="Override the config's closed-intent assignment policy.",
    )
    parser.add_argument("--intent-sampling-seed", type=int, default=None)
    parser.add_argument(
        "--config-json",
        default="",
        help="Optional local config.json to reuse model settings. Do not commit this file.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Use mock text/image/calibration outputs.")
    parser.add_argument("--skip-images", action="store_true", help="Only run text planning and calibration will be skipped.")
    parser.add_argument("--skip-calibration", action="store_true", help="Run text and images, but skip VL calibration.")
    parser.add_argument("--overwrite-images", action="store_true", help="Regenerate existing images if reusing a run manually.")
    parser.add_argument("--overwrite-calibration", action="store_true", help="Re-run calibration if records already exist.")
    parser.add_argument(
        "--postprocess-closed-intent",
        action="store_true",
        help="Append a text-only closed_intent repair overlay after generation.",
    )
    parser.add_argument(
        "--closed-intent-config",
        default="",
        help="Optional nested repair config; defaults to the generation model config.",
    )
    parser.add_argument(
        "--closed-intent-output-dir",
        default="",
        help="Repair output; defaults to <run>/closed_intent_only_v1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run and args.postprocess_closed_intent:
        raise ValueError("--postprocess-closed-intent cannot be combined with --dry-run")
    base_dir = Path(__file__).resolve().parent

    if args.config_json:
        config = load_config_from_json(resolve_under_base(base_dir, args.config_json))
        config.image_size = args.image_size
        config.image_quality = args.image_quality
    else:
        config = PipelineConfig(image_size=args.image_size, image_quality=args.image_quality)

    if args.root_assets_dir:
        config.root_assets_dir = args.root_assets_dir
    if args.profile_overrides_json:
        config.profile_overrides_path = str(resolve_under_base(base_dir, args.profile_overrides_json))
    config.branch_depth = args.branch_depth
    config.tree_nodes = args.tree_nodes
    config.valid_edges_per_tree = max(0, args.tree_nodes - 1)
    config.image_generation_workers = max(1, args.image_generation_workers)

    if args.intent_sampling_policy:
        config.intent_sampling_policy = args.intent_sampling_policy
    if args.intent_sampling_seed is not None:
        config.intent_sampling_seed = args.intent_sampling_seed
    pipeline = DatasetPipeline(
        config=config,
        output_dir=resolve_under_base(base_dir, args.output_dir),
        dry_run=args.dry_run,
    )

    run_dir_holder: dict[str, Path] = {}
    timings: dict[str, Any] = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "settings": {
            "profiles": str(resolve_under_base(base_dir, args.profiles)),
            "profile_limit": args.profile_limit,
            "profile_user_id": args.profile_user_id or None,
            "topics": args.topics,
            "topics_per_user": args.topics_per_user,
            "branch_depth": args.branch_depth,
            "tree_nodes": args.tree_nodes,
            "skip_images": args.skip_images,
            "skip_calibration": args.skip_calibration,
            "dry_run": args.dry_run,
            "intent_sampling_policy": config.intent_sampling_policy,
            "intent_sampling_seed": config.intent_sampling_seed,
            "postprocess_closed_intent": args.postprocess_closed_intent,
            "config": safe_config(config),
        },
        "stages": {},
    }
    total_start = time.perf_counter()

    def run_text() -> None:
        run_dir_holder["run_dir"] = pipeline.run(
            profiles_path=resolve_under_base(base_dir, args.profiles),
            profile_limit=args.profile_limit,
            profile_user_id=args.profile_user_id or None,
            topics=args.topics,
            topics_per_user=args.topics_per_user,
            generate_images=False,
        )

    timings["stages"]["text_generation_seconds"] = timed_stage("text generation", run_text)
    run_dir = run_dir_holder["run_dir"]

    if not args.skip_images:
        timings["stages"]["image_generation_seconds"] = timed_stage(
            "image generation",
            lambda: pipeline.generate_images_for_run(
                run_dir=run_dir,
                limit=None,
                overwrite=args.overwrite_images,
            ),
        )
    else:
        timings["stages"]["image_generation_seconds"] = None

    if not args.skip_images and not args.skip_calibration:
        timings["stages"]["calibration_seconds"] = timed_stage(
            "VL calibration",
            lambda: pipeline.calibrate_run(
                run_dir=run_dir,
                limit=None,
                overwrite=args.overwrite_calibration,
            ),
        )
    else:
        timings["stages"]["calibration_seconds"] = None

    repair_summary: dict[str, Any] | None = None
    repair_output_dir: Path | None = None
    if args.postprocess_closed_intent:
        if args.closed_intent_config:
            repair_config = load_repair_config(resolve_under_base(base_dir, args.closed_intent_config))
        else:
            repair_config = {
                "text_api": {
                    "base_url": config.llm_base_url,
                    "api_key": config.llm_api_key,
                    "model": config.llm_model,
                    "timeout_seconds": config.timeout_seconds,
                }
            }
        repair_output_dir = (
            resolve_under_base(base_dir, args.closed_intent_output_dir)
            if args.closed_intent_output_dir
            else run_dir / SCHEMA_VERSION
        )

        def run_label_repair() -> None:
            nonlocal repair_summary
            repair_summary = repair_closed_intents(
                source_root=run_dir,
                tree_paths=run_tree_paths(run_dir),
                output_dir=repair_output_dir,
                config=repair_config,
            )

        timings["stages"]["closed_intent_repair_seconds"] = timed_stage(
            "closed-intent repair", run_label_repair
        )
    else:
        timings["stages"]["closed_intent_repair_seconds"] = None

    timings["total_seconds"] = round(time.perf_counter() - total_start, 3)
    timings["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    timings["run_dir"] = str(run_dir)

    write_json(run_dir / "stage_timings.json", timings)
    write_json(
        run_dir / "end_to_end_summary.json",
        {
            "run_dir": str(run_dir),
            "manifest": str(run_dir / "manifest.jsonl"),
            "profiles_normalized": str(run_dir / "profiles_normalized.jsonl"),
            "topics": str(run_dir / "topics.json"),
            "stage_timings": str(run_dir / "stage_timings.json"),
            "closed_intent_repair": str(repair_output_dir) if repair_output_dir else None,
            "closed_intent_repair_summary": repair_summary,
            "completed_stages": [
                "text",
                *([] if args.skip_images else ["images"]),
                *([] if args.skip_images or args.skip_calibration else ["calibrate"]),
                *([] if not args.postprocess_closed_intent else ["closed_intent_repair"]),
            ],
        },
    )
    print(f"[done] run_dir={run_dir}")
    print(f"[done] timings={run_dir / 'stage_timings.json'}")


if __name__ == "__main__":
    main()
