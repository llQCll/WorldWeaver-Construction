from __future__ import annotations

import copy
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from postprocess_closed_intent_only import (
        assert_only_labels_changed,
        call_label,
        compact_evidence,
        full_cache_index,
        make_client,
        reusable_full_label,
    )
    from postprocess_intent_v2 import (
        active_tree_paths,
        file_sha256,
        load_config,
        read_json,
        write_json,
        write_jsonl,
    )
except ModuleNotFoundError:
    from dataset_tools.postprocess_closed_intent_only import (
        assert_only_labels_changed,
        call_label,
        compact_evidence,
        full_cache_index,
        make_client,
        reusable_full_label,
    )
    from dataset_tools.postprocess_intent_v2 import (
        active_tree_paths,
        file_sha256,
        load_config,
        read_json,
        write_json,
        write_jsonl,
    )


SCHEMA_VERSION = "closed_intent_only_v1"


def load_repair_config(path: Path) -> dict[str, Any]:
    """Load either the nested postprocess config or the flat generation config."""
    config = load_config(path)
    supplied = read_json(path)
    flat_mapping = {
        "llm_base_url": "base_url",
        "llm_api_key": "api_key",
        "llm_model": "model",
        "timeout_seconds": "timeout_seconds",
    }
    for source_key, target_key in flat_mapping.items():
        if source_key in supplied and supplied[source_key] not in (None, ""):
            config["text_api"][target_key] = supplied[source_key]
    return config


def run_tree_paths(run_dir: Path) -> list[Path]:
    paths = sorted(run_dir.glob("users/user_*/*/tree.json"))
    if not paths:
        raise FileNotFoundError(f"No tree.json files found under run directory: {run_dir}")
    return paths


def source_tree_paths(*, batch_dir: Path | None = None, run_dir: Path | None = None) -> list[Path]:
    if (batch_dir is None) == (run_dir is None):
        raise ValueError("Provide exactly one of batch_dir or run_dir")
    return active_tree_paths(batch_dir) if batch_dir is not None else run_tree_paths(run_dir)  # type: ignore[arg-type]


def review_reasons(row: dict[str, Any], repaired_counts: Counter[str]) -> list[str]:
    reasons: list[str] = []
    confidence = float(row["confidence"])
    if row["label_changed"] and confidence < 0.80:
        reasons.append("changed_label_below_0.80")
    if confidence < 0.72:
        reasons.append("confidence_below_0.72")
    if row["legacy_closed_intent"] == "reframe":
        reasons.append("legacy_reframe_is_extremely_rare")
    if repaired_counts[str(row["repaired_closed_intent"])] < 5:
        reasons.append("repaired_rare_class")
    return reasons


def ranking_top1(edge: dict[str, Any]) -> str:
    ranking = edge.get("intent_ranking", [])
    if not isinstance(ranking, list) or not ranking:
        return ""
    first = ranking[0]
    return str(first.get("intent", "")).strip().lower() if isinstance(first, dict) else ""


def repair_closed_intents(
    *,
    source_root: Path,
    tree_paths: Iterable[Path],
    output_dir: Path,
    config: dict[str, Any],
    client: Any | None = None,
) -> dict[str, Any]:
    """Create a label-only overlay without mutating source trees or reading images."""
    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_paths = [path.resolve() for path in tree_paths]
    if not selected_paths:
        raise ValueError("No source trees selected for closed-intent repair")

    label_client = client or make_client(config)
    reusable = full_cache_index(source_root)
    audits: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    api_calls = 0
    reused_labels = 0
    preserved_top1_mismatch_count = 0

    for index, source_tree_path in enumerate(selected_paths, 1):
        original = read_json(source_tree_path)
        repaired = copy.deepcopy(original)
        original_edges = {str(edge["edge_id"]): edge for edge in original.get("edges", [])}
        tree_id = str(original["tree_id"])

        for edge in repaired.get("edges", []):
            edge_id = str(edge["edge_id"])
            legacy = str(edge.get("closed_intent", "")).strip().lower()
            cache_path = output_dir / "cache" / "labels" / tree_id / f"{edge_id}.json"
            if cache_path.is_file():
                result = read_json(cache_path)
                annotation_source = "label_only_cache"
            else:
                result = reusable_full_label(reusable.get((tree_id, edge_id)))
                if result is not None:
                    annotation_source = "reused_full_text_cache"
                    reused_labels += 1
                else:
                    result = call_label(label_client, compact_evidence(original, original_edges[edge_id]))
                    annotation_source = "label_only_api"
                    api_calls += 1
                write_json(cache_path, result)

            repaired_label = str(result["closed_intent"])
            edge["closed_intent"] = repaired_label
            if ranking_top1(original_edges[edge_id]) != repaired_label:
                preserved_top1_mismatch_count += 1
            audits.append(
                {
                    "tree_id": tree_id,
                    "edge_id": edge_id,
                    "legacy_closed_intent": legacy,
                    "repaired_closed_intent": repaired_label,
                    "label_changed": legacy != repaired_label,
                    "confidence": float(result["confidence"]),
                    "rationale": str(result["rationale"]),
                    "annotation_source": annotation_source,
                    "source_tree": str(source_tree_path),
                }
            )

        assert_only_labels_changed(original, repaired)
        relative_tree = Path("trees") / f"user_{repaired['user_id']}" / str(repaired["topic_id"]) / "tree.json"
        write_json(output_dir / relative_tree, repaired)
        manifest.append(
            {
                "tree_id": tree_id,
                "user_id": str(repaired["user_id"]),
                "topic_id": str(repaired["topic_id"]),
                "source_tree": str(source_tree_path),
                "source_tree_sha256": file_sha256(source_tree_path),
                "output_tree": relative_tree.as_posix(),
                "edge_count": len(repaired.get("edges", [])),
            }
        )
        print(f"[{index}/{len(selected_paths)}] {tree_id}: {len(repaired.get('edges', []))} labels", flush=True)

    legacy_counts = Counter(str(row["legacy_closed_intent"]) for row in audits)
    repaired_counts = Counter(str(row["repaired_closed_intent"]) for row in audits)
    review_queue: list[dict[str, Any]] = []
    for row in audits:
        reasons = review_reasons(row, repaired_counts)
        if reasons:
            review_queue.append({**row, "review_reasons": reasons})

    summary = {
        "schema_version": SCHEMA_VERSION,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "source_root": str(source_root),
        "output_dir": str(output_dir),
        "model": str(config["text_api"]["model"]),
        "tree_count": len(manifest),
        "edge_count": len(audits),
        "changed_label_count": sum(bool(row["label_changed"]) for row in audits),
        "legacy_class_counts": dict(legacy_counts),
        "repaired_class_counts": dict(repaired_counts),
        "api_call_count": api_calls,
        "reused_full_cache_count": reused_labels,
        "images_sent": 0,
        "images_modified": False,
        "other_fields_modified": False,
        "review_queue_count": len(review_queue),
        "preserved_top1_mismatch_count": preserved_top1_mismatch_count,
        "strict_intent_v2_compatible": False,
    }
    write_jsonl(output_dir / "manifest.jsonl", manifest)
    write_jsonl(output_dir / "label_audit.jsonl", audits)
    write_jsonl(output_dir / "label_review_queue.jsonl", review_queue)
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "compatibility_report.json",
        {
            "schema_version": SCHEMA_VERSION,
            "strict_intent_v2_compatible": False,
            "preserved_top1_mismatch_count": preserved_top1_mismatch_count,
            "explanation": "Only closed_intent was repaired. intent_ranking and every other source field were deliberately preserved.",
            "recommended_use": "Use closed_intent as the repaired target and ignore preserved intent_ranking until a later ranking-repair stage.",
        },
    )
    return summary
