from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from dataset_pipeline import CLOSED_INTENTS, INTENT_BOUNDARY_RULES, OpenAICompatibleClient
    from postprocess_intent_v2 import active_tree_paths, file_sha256, load_config, read_json, write_json, write_jsonl
except ModuleNotFoundError:
    from dataset_tools.dataset_pipeline import CLOSED_INTENTS, INTENT_BOUNDARY_RULES, OpenAICompatibleClient
    from dataset_tools.postprocess_intent_v2 import active_tree_paths, file_sha256, load_config, read_json, write_json, write_jsonl


SYSTEM_PROMPT = """You classify one interactive visual-narrative edge into exactly one closed-domain intent.
Use semantic action and next-panel transformation, not legacy heuristics. Return JSON only."""


def compact_evidence(tree: dict[str, Any], edge: dict[str, Any]) -> dict[str, Any]:
    nodes = {str(node.get("node_id")): node for node in tree.get("nodes", [])}
    source = nodes.get(str(edge.get("source_node")), {})
    target = nodes.get(str(edge.get("target_node")), {})
    slots = edge.get("slots", {})
    grounding = edge.get("grounding", {})
    return {
        "generation_prompt": edge.get("generation_prompt", ""),
        "branch_label": edge.get("branch_label", ""),
        "action": slots.get("action", ""),
        "target": slots.get("target", grounding.get("target_label", "")),
        "narrative_goal": slots.get("narrative_goal", ""),
        "grounded_target_caption": grounding.get("target_caption", ""),
        "source_story_state": source.get("story_state", {}),
        "target_story_state": target.get("story_state", edge.get("story_state_after", {})),
        "topic_world": tree.get("world_bible", {}),
    }


def label_prompt(evidence: dict[str, Any], issues: list[str] | None = None) -> str:
    correction = ""
    if issues:
        correction = "\nCorrect these failures: " + "; ".join(issues)
    return f"""Classify the intended NEXT-panel transformation.

Ontology:
{json.dumps(CLOSED_INTENTS, ensure_ascii=False)}

Boundary rules:
{json.dumps(INTENT_BOUNDARY_RULES, ensure_ascii=False)}

Evidence (legacy label is deliberately excluded):
{json.dumps(evidence, ensure_ascii=False)}

Return exactly:
{{"closed_intent":"one exact label","confidence":0.0,"rationale":"one brief boundary-based reason"}}
Confidence must be in [0,1].{correction}"""


def label_issues(data: Any) -> list[str]:
    if not isinstance(data, dict):
        return ["response is not an object"]
    issues: list[str] = []
    if str(data.get("closed_intent", "")).strip().lower() not in CLOSED_INTENTS:
        issues.append("closed_intent is invalid")
    try:
        if not 0.0 <= float(data.get("confidence")) <= 1.0:
            issues.append("confidence is outside [0,1]")
    except (TypeError, ValueError):
        issues.append("confidence is missing")
    if not str(data.get("rationale", "")).strip():
        issues.append("rationale is empty")
    return issues


def call_label(client: OpenAICompatibleClient, evidence: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    for _ in range(2):
        try:
            result = client.chat_json(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=label_prompt(evidence, issues),
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            issues = [f"response was not valid JSON: {type(exc).__name__}"]
            continue
        issues = label_issues(result)
        if not issues:
            return {
                "closed_intent": str(result["closed_intent"]).strip().lower(),
                "confidence": round(float(result["confidence"]), 6),
                "rationale": str(result["rationale"]).strip(),
            }
    raise ValueError("Label-only response failed validation: " + "; ".join(issues))


def full_cache_index(batch_dir: Path) -> dict[tuple[str, str], Path]:
    result: dict[tuple[str, str], Path] = {}
    for directory in sorted(batch_dir.glob("postprocessed_intent_v2*")):
        cache_root = directory / "cache" / "text"
        if not cache_root.is_dir():
            continue
        for path in cache_root.glob("*/*.json"):
            result.setdefault((path.parent.name, path.stem), path)
    return result


def reusable_full_label(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        selected = read_json(path).get("selected", {})
        label = str(selected.get("closed_intent", "")).strip().lower()
        confidence = float(selected.get("annotation_confidence"))
        rationale = str(selected.get("rationale", "")).strip()
    except (OSError, TypeError, ValueError):
        return None
    if label not in CLOSED_INTENTS or not 0.0 <= confidence <= 1.0 or not rationale:
        return None
    return {"closed_intent": label, "confidence": confidence, "rationale": rationale}


def assert_only_labels_changed(original: dict[str, Any], repaired: dict[str, Any]) -> None:
    expected = copy.deepcopy(original)
    repaired_edges = repaired.get("edges", [])
    if len(expected.get("edges", [])) != len(repaired_edges):
        raise ValueError("Edge count changed in label-only mode")
    for expected_edge, repaired_edge in zip(expected["edges"], repaired_edges):
        expected_edge["closed_intent"] = repaired_edge.get("closed_intent")
    if expected != repaired:
        raise ValueError("A field other than closed_intent changed in label-only mode")


def make_client(config: dict[str, Any]) -> OpenAICompatibleClient:
    service = config["text_api"]
    return OpenAICompatibleClient(
        base_url=str(service["base_url"]),
        api_key=str(service.get("api_key", "")),
        model=str(service.get("model", "gpt-5.5")),
        timeout_seconds=int(service.get("timeout_seconds", 180)),
        max_http_retries=5,
        reasoning_effort="low",
        max_completion_tokens=500,
    )


def main() -> None:
    try:
        from closed_intent_repair import load_repair_config, repair_closed_intents
    except ModuleNotFoundError:
        from dataset_tools.closed_intent_repair import load_repair_config, repair_closed_intents

    parser = argparse.ArgumentParser(description="Repair only edge closed_intent labels.")
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--output-dir", default="", help="Defaults to <batch>/closed_intent_only_v1.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tree-offset", type=int, default=0)
    parser.add_argument("--limit-trees", type=int, default=0)
    args = parser.parse_args()

    batch_dir = Path(args.batch_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else batch_dir / "closed_intent_only_v1"
    tree_paths = active_tree_paths(batch_dir)
    if args.tree_offset > 0:
        tree_paths = tree_paths[args.tree_offset :]
    if args.limit_trees > 0:
        tree_paths = tree_paths[: args.limit_trees]
    summary = repair_closed_intents(
        source_root=batch_dir,
        tree_paths=tree_paths,
        output_dir=output_dir,
        config=load_repair_config(Path(args.config).resolve()),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
