from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from dataset_pipeline import DatasetPipeline, PipelineConfig, write_json


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_config_from_json(path: Path) -> PipelineConfig:
    data = read_json(path)
    allowed = set(PipelineConfig.__dataclass_fields__.keys())
    values = {key: value for key, value in data.items() if key in allowed}
    return PipelineConfig(**values)


def run_command(command: list[str], *, cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(command) + "\n\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
        return process.wait()


def latest_child_run(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    runs = sorted(path for path in output_dir.iterdir() if path.is_dir() and path.name.startswith("run_"))
    return runs[-1] if runs else None


def count_data_node_images(job_dir: Path) -> int:
    return sum(
        1
        for run_dir in (job_dir / "runs").glob("run_*")
        for path in run_dir.glob("users/user_*/*/images/n*.png")
        if path.is_file()
    )


def count_data_trees(job_dir: Path) -> int:
    return sum(
        1
        for run_dir in (job_dir / "runs").glob("run_*")
        for path in run_dir.glob("users/user_*/*/tree.json")
        if path.is_file()
    )


def build_web(*, python_exe: str, base_dir: Path, run_dir: Path, root_assets_dir: Path, log_path: Path) -> int:
    return run_command(
        [
            python_exe,
            "build_contrast_web.py",
            "--run-dir",
            str(run_dir),
            "--root-assets-dir",
            str(root_assets_dir),
            "--web-dir",
            str(run_dir / "web"),
        ],
        cwd=base_dir,
        log_path=log_path,
    )


def resume_existing_run(
    *,
    config: PipelineConfig,
    base_dir: Path,
    python_exe: str,
    job: dict[str, Any],
    root_assets_dir: Path,
    image_size: str,
    image_quality: str,
    image_generation_workers: int,
    skip_calibration: bool,
    build_web_page: bool,
) -> dict[str, Any]:
    user_id = str(job["user_id"])
    job_dir = Path(job["job_dir"])
    log_dir = job_dir / "logs"
    status_path = job_dir / "job_status.json"
    run_dir = latest_child_run(job_dir / "runs")
    started = time.perf_counter()
    if run_dir is None or not (run_dir / "manifest.jsonl").exists():
        raise RuntimeError("No resumable run found.")

    config.image_size = image_size
    config.image_quality = image_quality
    config.image_generation_workers = max(1, image_generation_workers)
    config.root_assets_dir = str(root_assets_dir)
    pipeline = DatasetPipeline(config=config, output_dir=job_dir / "runs", dry_run=False)

    active_stage = "resume_images"
    try:
        pipeline.generate_images_for_run(run_dir=run_dir, overwrite=False)
        if not skip_calibration:
            active_stage = "resume_calibration"
            pipeline.calibrate_run(run_dir=run_dir, overwrite=False)
    except Exception as exc:
        result = {
            "user_id": user_id,
            "topics": job["topics"],
            "status": "failed",
            "stage": active_stage,
            "exit_code": 1,
            "message": str(exc),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "job_dir": str(job_dir),
            "run_dir": str(run_dir),
        }
        write_json(status_path, result)
        raise

    web_code = 0
    if build_web_page:
        web_code = build_web(
            python_exe=python_exe,
            base_dir=base_dir,
            run_dir=run_dir,
            root_assets_dir=root_assets_dir,
            log_path=log_dir / "web.log",
        )

    result = {
        "user_id": user_id,
        "topics": job["topics"],
        "status": "done" if web_code == 0 else "failed",
        "stage": "complete" if web_code == 0 else "web",
        "exit_code": web_code,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "job_dir": str(job_dir),
        "run_dir": str(run_dir),
        "web_index": str(run_dir / "web" / "index.html") if build_web_page and web_code == 0 else None,
        "resumed_existing_run": True,
    }
    write_json(status_path, result)
    return result


def rerun_job(
    *,
    python_exe: str,
    base_dir: Path,
    job: dict[str, Any],
    args: argparse.Namespace,
    root_assets_dir: Path,
) -> dict[str, Any]:
    user_id = str(job["user_id"])
    job_dir = Path(job["job_dir"])
    log_dir = job_dir / "logs"
    status_path = job_dir / "job_status.json"
    started = time.perf_counter()
    command = [
        python_exe,
        "run_end_to_end_dataset.py",
        "--config-json",
        str(Path(args.config_json).resolve()),
        "--profiles",
        str(Path(args.profiles).resolve()),
        "--profile-user-id",
        user_id,
        "--profile-limit",
        "1",
        "--topics",
        *job["topics"],
        "--topics-per-user",
        str(len(job["topics"])),
        "--root-assets-dir",
        str(root_assets_dir),
        "--output-dir",
        str(job_dir / "runs"),
        "--branch-depth",
        str(args.branch_depth),
        "--tree-nodes",
        str(args.tree_nodes),
        "--image-size",
        args.image_size,
        "--image-generation-workers",
        str(args.image_generation_workers),
        "--intent-sampling-policy",
        args.intent_sampling_policy,
        "--intent-sampling-seed",
        str(args.intent_sampling_seed),
        "--image-quality",
        args.image_quality,
    ]
    if args.profile_overrides_json:
        command.extend([
            "--profile-overrides-json",
            str(Path(args.profile_overrides_json).resolve()),
        ])
    if args.skip_calibration:
        command.append("--skip-calibration")

    code = run_command(command, cwd=base_dir, log_path=log_dir / "pipeline.resume.log")
    run_dir = latest_child_run(job_dir / "runs")
    if code != 0 or run_dir is None:
        result = {
            "user_id": user_id,
            "topics": job["topics"],
            "status": "failed",
            "stage": "pipeline",
            "exit_code": code,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "job_dir": str(job_dir),
        }
        write_json(status_path, result)
        return result

    web_code = 0
    if args.build_web:
        web_code = build_web(
            python_exe=python_exe,
            base_dir=base_dir,
            run_dir=run_dir,
            root_assets_dir=root_assets_dir,
            log_path=log_dir / "web.log",
        )

    result = {
        "user_id": user_id,
        "topics": job["topics"],
        "status": "done" if web_code == 0 else "failed",
        "stage": "complete" if web_code == 0 else "web",
        "exit_code": web_code,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "job_dir": str(job_dir),
        "run_dir": str(run_dir),
        "web_index": str(run_dir / "web" / "index.html") if args.build_web and web_code == 0 else None,
        "rerun_after_failure": True,
    }
    write_json(status_path, result)
    return result


def process_job(job: dict[str, Any], *, args: argparse.Namespace, base_dir: Path, config: PipelineConfig) -> dict[str, Any]:
    python_exe = sys.executable
    root_assets_dir = Path(args.root_assets_dir).resolve()
    job_dir = Path(job["job_dir"])
    status_path = job_dir / "job_status.json"
    if status_path.exists():
        try:
            status = read_json(status_path)
        except (json.JSONDecodeError, OSError):
            status = {}
        if status.get("status") == "done":
            return status | {"skipped": True}

    run_dir = latest_child_run(job_dir / "runs")
    if run_dir is not None and (run_dir / "manifest.jsonl").exists():
        return resume_existing_run(
            config=config,
            base_dir=base_dir,
            python_exe=python_exe,
            image_generation_workers=args.image_generation_workers,
            skip_calibration=args.skip_calibration,
            job=job,
            root_assets_dir=root_assets_dir,
            image_size=args.image_size,
            image_quality=args.image_quality,
            build_web_page=args.build_web,
        )
    return rerun_job(
        python_exe=python_exe,
        base_dir=base_dir,
        job=job,
        args=args,
        root_assets_dir=root_assets_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume failed jobs from a batch_generate_users.py batch directory.")
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--profiles", default="../data/user_profiles.jsonl")
    parser.add_argument("--root-assets-dir", default="../assets/topic_roots")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--branch-depth", type=int, default=5)
    parser.add_argument("--tree-nodes", type=int, default=18)
    parser.add_argument("--image-size", default="1024x1024")
    parser.add_argument("--image-quality", default="high")
    parser.add_argument("--image-generation-workers", type=int, default=1)
    parser.add_argument("--skip-calibration", action="store_true")
    parser.add_argument("--intent-sampling-policy", choices=["balanced_deterministic", "balanced_counterfactual", "hierarchical_natural"], default="balanced_deterministic")
    parser.add_argument("--intent-sampling-seed", type=int, default=20260804)
    parser.add_argument("--profile-overrides-json", default="")
    parser.add_argument("--build-web", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    batch_dir = Path(args.batch_dir).resolve()
    plan = read_json(batch_dir / "batch_plan.json")
    config = load_config_from_json(Path(args.config_json).resolve())
    jobs = plan["jobs"]
    started = time.perf_counter()
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_job, job, args=args, base_dir=base_dir, config=config) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
            except Exception as exc:
                result = {"status": "failed", "stage": "resume", "message": str(exc)}
            results.append(result)
            with (batch_dir / "resume_status.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(json.dumps(result, ensure_ascii=False))

    summary = {
        "batch_root": str(batch_dir),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "done": sum(1 for result in results if result.get("status") == "done"),
        "failed": sum(1 for result in results if result.get("status") != "done"),
        "results": sorted(results, key=lambda item: str(item.get("user_id", ""))),
    }
    write_json(batch_dir / "resume_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
