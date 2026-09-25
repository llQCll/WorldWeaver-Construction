from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from dataset_pipeline import INTENT_ORDER, read_profiles, slugify, write_json, write_jsonl


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve_path(base_dir: Path, path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (base_dir / path).resolve()


def available_topics(root_assets_dir: Path) -> list[str]:
    topics = []
    for topic_dir in sorted(path for path in root_assets_dir.iterdir() if path.is_dir()):
        asset_path = topic_dir / "root_asset.json"
        if asset_path.exists():
            asset = read_json(asset_path)
            topics.append(str(asset.get("topic") or topic_dir.name.replace("_", " ").title()))
        else:
            topics.append(topic_dir.name.replace("_", " ").title())
    if not topics:
        raise RuntimeError(f"No topics found under {root_assets_dir}")
    return topics


def stable_topic_sample(*, user_id: str, topics: list[str], count: int, seed: int) -> list[str]:
    if count > len(topics):
        raise ValueError(f"topics_per_user={count} exceeds available topics={len(topics)}")
    rng = random.Random(f"{seed}:{user_id}")
    return rng.sample(topics, count)


def balanced_topic_assignments(
    *,
    user_ids: list[str],
    topics: list[str],
    count: int,
    seed: int,
) -> dict[str, list[str]]:
    if count > len(topics):
        raise ValueError(f"topics_per_user={count} exceeds available topics={len(topics)}")
    total_slots = len(user_ids) * count
    base_quota, remainder = divmod(total_slots, len(topics))
    quota_order = list(topics)
    random.Random(seed).shuffle(quota_order)
    remaining = {
        topic: base_quota + (1 if topic in set(quota_order[:remainder]) else 0)
        for topic in topics
    }

    assignments: dict[str, list[str]] = {}
    for user_id in user_ids:
        rng = random.Random(f"{seed}:{user_id}:balanced")
        tie_break = {topic: rng.random() for topic in topics}
        candidates = [topic for topic in topics if remaining[topic] > 0]
        selected = sorted(candidates, key=lambda topic: (-remaining[topic], tie_break[topic]))[:count]
        if len(selected) != count:
            raise RuntimeError(f"Unable to assign {count} unique topics to user {user_id}")
        assignments[user_id] = selected
        for topic in selected:
            remaining[topic] -= 1

    if any(remaining.values()):
        raise RuntimeError(f"Unfilled topic quotas: {remaining}")
    return assignments


def select_profiles(
    rows: list[dict[str, Any]],
    *,
    last_users: int,
    user_offset_from_end: int | None,
    user_count: int | None,
    user_start_index: int | None = None,
    user_end_index: int | None = None,
) -> list[dict[str, Any]]:
    if user_start_index is not None or user_end_index is not None:
        if user_offset_from_end is not None or user_count is not None:
            raise ValueError("Index range cannot be combined with offset-from-end selection")
        if user_start_index is None or user_end_index is None:
            raise ValueError("--user-start-index and --user-end-index must be provided together")
        if user_start_index < 1 or user_end_index < user_start_index:
            raise ValueError("User indices must satisfy 1 <= start <= end")
        if user_end_index > len(rows):
            raise ValueError(
                f"--user-end-index must be at most {len(rows)}; received {user_end_index}"
            )
        return rows[user_start_index - 1 : user_end_index]

    if user_offset_from_end is None:
        if user_count is not None:
            raise ValueError("--user-count requires --user-offset-from-end")
        if last_users < 1 or last_users > len(rows):
            raise ValueError(f"--last-users must be between 1 and {len(rows)}")
        return rows[-last_users:]

    if user_offset_from_end < 1 or user_offset_from_end > len(rows):
        raise ValueError(f"--user-offset-from-end must be between 1 and {len(rows)}")
    if user_count is None or user_count < 1:
        raise ValueError("--user-count must be at least 1 when --user-offset-from-end is used")

    start = len(rows) - user_offset_from_end
    end = start + user_count
    if end > len(rows):
        raise ValueError(
            f"Requested {user_count} users from offset {user_offset_from_end}, "
            f"but only {len(rows) - start} records are available"
        )
    return rows[start:end]


def latest_child_run(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    runs = sorted([path for path in output_dir.iterdir() if path.is_dir() and path.name.startswith("run_")])
    return runs[-1] if runs else None


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


def build_user_job(
    *,
    python_exe: str,
    base_dir: Path,
    job: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    user_id = str(job["user_id"])
    job_dir = Path(job["job_dir"])
    log_dir = job_dir / "logs"
    status_path = job_dir / "job_status.json"
    started = time.perf_counter()

    if status_path.exists() and not args.overwrite:
        existing = read_json(status_path)
        if existing.get("status") == "done":
            return existing | {"skipped": True}

    output_dir = job_dir / "runs"
    command = [
        python_exe,
        "run_end_to_end_dataset.py",
        "--config-json",
        str(resolve_path(base_dir, args.config_json)),
        "--profiles",
        str(resolve_path(base_dir, args.profiles)),
        "--profile-user-id",
        user_id,
        "--profile-limit",
        "1",
        "--topics",
        *job["topics"],
        "--topics-per-user",
        str(len(job["topics"])),
        "--root-assets-dir",
        str(resolve_path(base_dir, args.root_assets_dir)),
        "--output-dir",
        str(output_dir),
        "--branch-depth",
        str(args.branch_depth),
        "--tree-nodes",
        str(args.tree_nodes),
        "--image-size",
        args.image_size,
        "--image-quality",
        args.image_quality,
        "--image-generation-workers",
        str(args.image_generation_workers),
    ]
    command.extend(["--intent-sampling-policy", args.intent_sampling_policy])
    command.extend(["--intent-sampling-seed", str(args.intent_sampling_seed)])
    if args.profile_overrides_json:
        command.extend([
            "--profile-overrides-json",
            str(resolve_path(base_dir, args.profile_overrides_json)),
        ])
    if args.dry_run:
        command.append("--dry-run")
    if args.skip_images:
        command.append("--skip-images")
    if args.skip_calibration:
        command.append("--skip-calibration")

    text_image_code = run_command(command, cwd=base_dir, log_path=log_dir / "pipeline.log")
    run_dir = latest_child_run(output_dir)
    if text_image_code != 0 or run_dir is None:
        result = {
            "user_id": user_id,
            "topics": job["topics"],
            "status": "failed",
            "stage": "pipeline",
            "exit_code": text_image_code,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "job_dir": str(job_dir),
        }
        write_json(status_path, result)
        return result

    web_code = 0
    if args.build_web:
        web_command = [
            python_exe,
            "build_contrast_web.py",
            "--run-dir",
            str(run_dir),
            "--root-assets-dir",
            str(resolve_path(base_dir, args.root_assets_dir)),
            "--web-dir",
            str(run_dir / "web"),
        ]
        web_code = run_command(web_command, cwd=base_dir, log_path=log_dir / "web.log")

    result = {
        "user_id": user_id,
        "topics": job["topics"],
        "status": "done" if web_code == 0 else "failed",
        "stage": "web" if web_code != 0 else "complete",
        "exit_code": web_code,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "job_dir": str(job_dir),
        "run_dir": str(run_dir),
        "web_index": str(run_dir / "web" / "index.html") if args.build_web and web_code == 0 else None,
    }
    write_json(status_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-generate user-topic comic tree runs with bounded concurrency.")
    parser.add_argument("--profiles", default="../data/user_profiles.jsonl", help="Raw user profile JSONL.")
    parser.add_argument("--root-assets-dir", default="../assets/topic_roots", help="Curated topic root assets.")
    parser.add_argument("--config-json", required=True, help="Local config.json with private model service settings. Do not commit it.")
    parser.add_argument("--profile-overrides-json", default="", help="Controlled profile variants keyed by user_id.")
    parser.add_argument("--topic-assignments-json", default="", help="Optional explicit topic list for each selected user.")
    parser.add_argument("--output-dir", default="dataset_runs_batch", help="Batch output directory.")
    parser.add_argument("--batch-dir", default="", help="Existing or explicit batch directory. Enables resume with the same plan.")
    parser.add_argument("--last-users", type=int, default=36, help="Use the last N users from the profile file.")
    parser.add_argument(
        "--user-offset-from-end",
        type=int,
        default=None,
        help="Select a window beginning at the Nth profile from the end (inclusive).",
    )
    parser.add_argument("--user-count", type=int, default=None, help="Number of profiles in the offset window.")
    parser.add_argument("--topics-per-user", type=int, default=4, help="Random unique topics assigned to each user.")
    parser.add_argument(
        "--balance-topics",
        action="store_true",
        help="Assign global topic quotas so topic frequencies differ by at most one.",
    )
    parser.add_argument("--seed", type=int, default=20260721, help="Stable topic sampling seed.")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent user jobs.")
    parser.add_argument("--branch-depth", type=int, default=5)
    parser.add_argument("--tree-nodes", type=int, default=18, help="Number of nodes generated for each user-topic tree.")
    parser.add_argument("--image-size", default="1024x1024")
    parser.add_argument("--user-start-index", type=int, default=None, help="1-based inclusive profile index.")
    parser.add_argument("--user-end-index", type=int, default=None, help="1-based inclusive profile index.")
    parser.add_argument("--intent-sampling-policy", choices=["balanced_deterministic", "balanced_counterfactual", "hierarchical_natural"], default="balanced_deterministic")
    parser.add_argument("--intent-sampling-seed", type=int, default=20260804)
    parser.add_argument("--image-quality", default="high")
    parser.add_argument("--image-generation-workers", type=int, default=1, help="Concurrent image calls per user, synchronized by tree depth.")
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--skip-calibration", action="store_true", default=True)
    parser.add_argument("--with-calibration", dest="skip_calibration", action="store_false", help="Also run VL calibration.")
    parser.add_argument("--build-web", action="store_true", default=True)
    parser.add_argument("--no-web", dest="build_web", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="Write and validate batch_plan.json without launching jobs.")
    parser.add_argument(
        "--require-balanced-intents",
        action="store_true",
        help="Require the number of edges per tree to divide evenly across all closed-domain intents.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-run jobs even if job_status.json says done.")
    parser.add_argument("--postprocess-closed-intent", action="store_true", help="Repair closed_intent once after all user jobs finish.")
    parser.add_argument("--closed-intent-config", default="", help="Optional repair config; defaults to --config-json.")
    parser.add_argument("--closed-intent-output-dir", default="", help="Defaults to <batch>/closed_intent_only_v1.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run and args.postprocess_closed_intent:
        raise ValueError("--postprocess-closed-intent cannot be combined with --dry-run")
    edges_per_tree = max(0, args.tree_nodes - 1)
    if args.require_balanced_intents and (edges_per_tree == 0 or edges_per_tree % len(INTENT_ORDER) != 0):
        raise ValueError(
            f"--require-balanced-intents needs a positive edge count divisible by {len(INTENT_ORDER)}; "
            f"tree_nodes={args.tree_nodes} gives {edges_per_tree} edges"
        )
    base_dir = Path(__file__).resolve().parent
    profiles_path = resolve_path(base_dir, args.profiles)
    root_assets_dir = resolve_path(base_dir, args.root_assets_dir)
    batch_root = resolve_path(base_dir, args.batch_dir) if args.batch_dir else resolve_path(base_dir, args.output_dir) / time.strftime("batch_%Y%m%d_%H%M%S")
    batch_root.mkdir(parents=True, exist_ok=True)

    plan_path = batch_root / "batch_plan.json"
    if args.batch_dir and plan_path.exists():
        plan = read_json(plan_path)
        jobs = list(plan["jobs"])
    else:
        rows = read_profiles(profiles_path)
        selected = select_profiles(
            rows,
            last_users=args.last_users,
            user_offset_from_end=args.user_offset_from_end,
            user_count=args.user_count,
            user_start_index=args.user_start_index,
            user_end_index=args.user_end_index,
        )
        topics = available_topics(root_assets_dir)
        user_ids = [str(row["user_id"]) for row in selected]
        custom_assignments: dict[str, list[str]] = {}
        if args.topic_assignments_json:
            assignment_data = read_json(resolve_path(base_dir, args.topic_assignments_json))
            raw_assignments = assignment_data.get("assignments", assignment_data)
            if not isinstance(raw_assignments, dict):
                raise ValueError("Topic assignments must be an object keyed by user_id")
            for user_id in user_ids:
                assigned = raw_assignments.get(user_id)
                if not isinstance(assigned, list) or len(assigned) != args.topics_per_user:
                    raise ValueError(
                        f"Custom topic assignment for user {user_id} must contain "
                        f"exactly {args.topics_per_user} topics"
                    )
                if len(set(assigned)) != len(assigned) or any(topic not in topics for topic in assigned):
                    raise ValueError(f"Custom topic assignment for user {user_id} contains invalid topics")
                custom_assignments[user_id] = [str(topic) for topic in assigned]
        balanced_assignments = (
            balanced_topic_assignments(
                user_ids=user_ids,
                topics=topics,
                count=args.topics_per_user,
                seed=args.seed,
            )
            if args.balance_topics and not custom_assignments
            else {}
        )
        jobs = []
        for row in selected:
            user_id = str(row["user_id"])
            user_topics = (
                custom_assignments[user_id]
                if custom_assignments
                else balanced_assignments[user_id]
                if args.balance_topics
                else stable_topic_sample(
                    user_id=user_id,
                    topics=topics,
                    count=args.topics_per_user,
                    seed=args.seed,
                )
            )
            jobs.append(
                {
                    "user_id": user_id,
                    "topics": user_topics,
                    "topic_ids": [slugify(topic) for topic in user_topics],
                    "job_dir": str(batch_root / f"user_{user_id}"),
                }
            )

        plan = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "profiles": str(profiles_path),
            "root_assets_dir": str(root_assets_dir),
            "last_users": args.last_users,
            "user_offset_from_end": args.user_offset_from_end,
            "user_count": args.user_count,
            "user_start_index": args.user_start_index,
            "user_end_index": args.user_end_index,
            "selected_user_ids": [str(row["user_id"]) for row in selected],
            "topics_per_user": args.topics_per_user,
            "balance_topics": args.balance_topics,
            "intent_sampling_policy": args.intent_sampling_policy,
            "intent_sampling_seed": args.intent_sampling_seed,
            "topic_counts": {
                slugify(topic): sum(topic in job["topics"] for job in jobs)
                for topic in topics
            },
            "workers": args.workers,
            "branch_depth": args.branch_depth,
            "profile_overrides_json": args.profile_overrides_json or None,
            "topic_assignments_json": args.topic_assignments_json or None,
            "topic_assignment_policy": "custom" if custom_assignments else "balanced" if args.balance_topics else "stable_random",
            "tree_nodes": args.tree_nodes,
            "image_generation_workers": args.image_generation_workers,
            "intent_balance": {
                "required": args.require_balanced_intents,
                "labels": INTENT_ORDER,
                "edges_per_tree": edges_per_tree,
                "target_per_label_per_tree": (
                    edges_per_tree // len(INTENT_ORDER)
                    if edges_per_tree > 0 and edges_per_tree % len(INTENT_ORDER) == 0
                    else None
                ),
                "expected_total_per_label": (
                    len(jobs) * args.topics_per_user * edges_per_tree // len(INTENT_ORDER)
                    if edges_per_tree > 0 and edges_per_tree % len(INTENT_ORDER) == 0
                    else None
                ),
            },
            "skip_images": args.skip_images,
            "skip_calibration": args.skip_calibration,
            "build_web": args.build_web,
            "dry_run": args.dry_run,
            "postprocess_closed_intent": args.postprocess_closed_intent,
            "job_count": len(jobs),
            "tree_count": len(jobs) * args.topics_per_user,
            "estimated_branch_images": 0 if args.skip_images else len(jobs) * args.topics_per_user * max(0, args.tree_nodes - 1),
            "jobs": jobs,
        }
        write_json(batch_root / "batch_plan.json", plan)
        write_jsonl(batch_root / "batch_plan.jsonl", jobs)
    print(json.dumps({"batch_root": str(batch_root), "job_count": len(jobs), "tree_count": plan["tree_count"]}, ensure_ascii=False, indent=2))

    if args.plan_only:
        print(json.dumps({"status": "plan_only", "batch_plan": str(plan_path)}, ensure_ascii=False, indent=2))
        return

    started = time.perf_counter()
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                build_user_job,
                python_exe=sys.executable,
                base_dir=base_dir,
                job=job,
                args=args,
            )
            for job in jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            write_jsonl(batch_root / "batch_status.jsonl", results)
            print(json.dumps({"user_id": result["user_id"], "status": result["status"], "elapsed_seconds": result["elapsed_seconds"]}, ensure_ascii=False))

    failed_jobs = sum(1 for result in results if result.get("status") != "done")
    postprocess_status: dict[str, Any] = {"enabled": args.postprocess_closed_intent, "status": "not_requested"}
    if args.postprocess_closed_intent and failed_jobs == 0:
        repair_output = (
            resolve_path(base_dir, args.closed_intent_output_dir)
            if args.closed_intent_output_dir
            else batch_root / "closed_intent_only_v1"
        )
        repair_config = resolve_path(base_dir, args.closed_intent_config or args.config_json)
        repair_command = [
            sys.executable,
            "repair_closed_intent_dataset.py",
            "--batch-dir",
            str(batch_root),
            "--output-dir",
            str(repair_output),
            "--config",
            str(repair_config),
        ]
        repair_code = run_command(
            repair_command,
            cwd=base_dir,
            log_path=batch_root / "logs" / "closed_intent_repair.log",
        )
        postprocess_status = {
            "enabled": True,
            "status": "done" if repair_code == 0 else "failed",
            "exit_code": repair_code,
            "output_dir": str(repair_output),
            "summary": str(repair_output / "summary.json") if repair_code == 0 else None,
        }
    elif args.postprocess_closed_intent:
        postprocess_status = {
            "enabled": True,
            "status": "skipped_due_to_failed_generation_jobs",
        }

    summary = {
        "batch_root": str(batch_root),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "done": sum(1 for result in results if result.get("status") == "done"),
        "failed": failed_jobs,
        "closed_intent_postprocess": postprocess_status,
        "results": sorted(results, key=lambda item: str(item["user_id"])),
    }
    write_json(batch_root / "batch_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["failed"] or postprocess_status.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
