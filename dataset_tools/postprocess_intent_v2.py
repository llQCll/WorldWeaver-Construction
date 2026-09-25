from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Callable

try:
    from audit_intent_dataset import active_tree_paths
    from dataset_pipeline import (
        AFFECT_DIMS,
        CLOSED_INTENTS,
        INTENT_BOUNDARY_RULES,
        INTENT_CONFUSIONS,
        PERSONALITY_DIMS,
        REQUIRED_SLOT_KEYS,
        SCALE_METADATA,
        OpenAICompatibleClient,
        blend_deltas,
        clamp,
        compute_current_state,
        coerce_delta,
        update_affective_state,
        update_stable_profile,
    )
except ModuleNotFoundError:
    from dataset_tools.audit_intent_dataset import active_tree_paths
    from dataset_tools.dataset_pipeline import (
        AFFECT_DIMS,
        CLOSED_INTENTS,
        INTENT_BOUNDARY_RULES,
        INTENT_CONFUSIONS,
        PERSONALITY_DIMS,
        REQUIRED_SLOT_KEYS,
        SCALE_METADATA,
        OpenAICompatibleClient,
        blend_deltas,
        clamp,
        compute_current_state,
        coerce_delta,
        update_affective_state,
        update_stable_profile,
    )


SCHEMA_VERSION = "intent_v2_postprocessed"
DEFAULT_CONFIG: dict[str, Any] = {
    "text_api": {
        "base_url": "",
        "api_key": "",
        "model": "gpt-5.5",
        "timeout_seconds": 180,
        "reasoning_effort": "low",
        "max_completion_tokens": 2000,
    },
    "vision_api": {
        "base_url": "",
        "api_key": "",
        "model": "gpt-5.5",
        "timeout_seconds": 180,
        "reasoning_effort": "low",
        "max_completion_tokens": 2000,
    },
    "processing": {
        "text_passes": 2,
        "max_retries": 3,
        "confidence_threshold": 0.72,
        "visual_alignment_threshold": 0.60,
        "calibration_expected_weight": 0.60,
        "always_review_labels": ["reframe"],
    },
}

TEXT_SYSTEM_PROMPT = """You are a strict annotation adjudicator for an interactive visual narrative dataset.
Infer the user's INTENDED next-panel transformation from semantic evidence. The legacy class, legacy ranking,
and generation prompt are deliberately hidden. Use exactly one of the six ontology labels and return JSON only."""

VISION_SYSTEM_PROMPT = """You are a strict visual calibration annotator. The first image is the source panel and
the second is the generated target panel. Judge the transition actually visible in the images independently of the
planned label. Return JSON only."""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path | None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        supplied = read_json(path)
        for section in config:
            if isinstance(supplied.get(section), dict):
                config[section].update(supplied[section])
    env_map = {
        ("text_api", "base_url"): ["POSTPROCESS_TEXT_BASE_URL", "DATASET_LLM_BASE_URL"],
        ("text_api", "api_key"): ["POSTPROCESS_TEXT_API_KEY", "DATASET_LLM_API_KEY"],
        ("text_api", "model"): ["POSTPROCESS_TEXT_MODEL"],
        ("vision_api", "base_url"): ["POSTPROCESS_VISION_BASE_URL", "DATASET_LLM_BASE_URL"],
        ("vision_api", "api_key"): ["POSTPROCESS_VISION_API_KEY", "DATASET_LLM_API_KEY"],
        ("vision_api", "model"): ["POSTPROCESS_VISION_MODEL"],
    }
    for (section, key), names in env_map.items():
        for name in names:
            if os.getenv(name):
                config[section][key] = os.environ[name]
                break
    return config


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    clean = copy.deepcopy(config)
    for section in ["text_api", "vision_api"]:
        clean[section].pop("api_key", None)
        clean[section].pop("base_url", None)
    return clean


def make_client(config: dict[str, Any], section: str, *, dry_run: bool) -> OpenAICompatibleClient | None:
    if dry_run:
        return None
    service = config[section]
    if not str(service.get("base_url", "")).strip():
        raise ValueError(f"{section}.base_url is empty; fill the private config or use --dry-run")
    return OpenAICompatibleClient(
        base_url=str(service["base_url"]),
        api_key=str(service.get("api_key", "")),
        model=str(service.get("model", "gpt-5.5")),
        timeout_seconds=int(service.get("timeout_seconds", 180)),
        reasoning_effort=str(service.get("reasoning_effort", "low")),
        max_completion_tokens=int(service.get("max_completion_tokens", 2000)),
    )


def resolve_image_path(tree_path: Path, raw_path: Any, node_id: str) -> Path:
    raw = str(raw_path or "")
    candidates = [
        Path(raw),
        tree_path.parent / "images" / PureWindowsPath(raw).name,
        tree_path.parent / "images" / Path(raw).name,
        tree_path.parent / "images" / f"{node_id}.png",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[-1].resolve()


def relative_image_ref(batch_dir: Path, path: Path) -> str:
    try:
        return path.relative_to(batch_dir.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def normalize_box(value: Any) -> tuple[list[float], bool]:
    try:
        original = [float(item) for item in value]
    except (TypeError, ValueError):
        return [0.0, 0.0, 0.1, 0.1], True
    if len(original) != 4:
        return [0.0, 0.0, 0.1, 0.1], True
    x1, y1, x2, y2 = [clamp(item) for item in original]
    changed = [x1, y1, x2, y2] != original
    if x1 >= x2:
        x1, x2 = sorted([x1, x2])
        x2 = min(1.0, max(x2, x1 + 0.01))
        changed = True
    if y1 >= y2:
        y1, y2 = sorted([y1, y2])
        y2 = min(1.0, max(y2, y1 + 0.01))
        changed = True
    return [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)], changed


def edge_evidence(tree: dict[str, Any], edge: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = nodes.get(str(edge.get("source_node")), {})
    target = nodes.get(str(edge.get("target_node")), {})
    slots = edge.get("slots", {})
    grounding = edge.get("grounding", {})
    return {
        "branch_label": edge.get("branch_label", ""),
        "action": slots.get("action", ""),
        "target": slots.get("target", grounding.get("target_label", "")),
        "narrative_goal": slots.get("narrative_goal", ""),
        "mood": slots.get("mood", ""),
        "continuity_constraint": slots.get("continuity_constraint", ""),
        "grounded_target_label": grounding.get("target_label", ""),
        "grounded_target_caption": grounding.get("target_caption", ""),
        "source_story_state": source.get("story_state", {}),
        "target_story_state": target.get("story_state", edge.get("story_state_after", {})),
        "stable_profile_before": source.get("stable_profile", {}),
        "affective_state_before": source.get("affective_state", {}),
        "current_state_before": source.get("current_state", {}),
        "topic_world": tree.get("world_bible", {}),
    }


def text_prompt(
    evidence: dict[str, Any],
    *,
    pass_index: int,
    candidates: list[dict[str, Any]] | None = None,
) -> str:
    candidate_text = ""
    if candidates:
        candidate_text = (
            "\nIndependent annotations disagreed. Adjudicate from the evidence rather than majority voting:\n"
            + json.dumps(candidates, ensure_ascii=False, indent=2)
        )
    return f"""
Determine what the user intends the NEXT panel to do. Classify the intended transformation rather than the clicked
noun or an isolated verb. This is independent pass {pass_index}.

Closed-intent ontology:
{json.dumps(CLOSED_INTENTS, ensure_ascii=False, indent=2)}

Mandatory boundaries:
{json.dumps(INTENT_BOUNDARY_RULES, ensure_ascii=False, indent=2)}

Evidence; the legacy label, ranking, and generation prompt are excluded:
{json.dumps(evidence, ensure_ascii=False, indent=2)}
{candidate_text}

Return exactly:
{{
  "closed_intent": "one exact closed label",
  "natural_language_intent": "one specific sentence describing action, target, goal, and next-panel change",
  "intent_ranking": [
    {{"rank": 1, "intent": "closed label", "score": 0.0}},
    {{"rank": 2, "intent": "closed label", "score": 0.0}},
    {{"rank": 3, "intent": "closed label", "score": 0.0}}
  ],
  "slots": {{
    "action": "concrete free-form action",
    "target": "specific target",
    "narrative_goal": "desired result",
    "mood": "desired tone",
    "continuity_constraint": "what remains consistent",
    "scope": "next_panel"
  }},
  "profile_signal": {{"valid_8D_profile_dimension": 0.0}},
  "expected_affective_delta": {{"valid_7D_affect_dimension": 0.0}},
  "expected_profile_delta": {{"valid_8D_profile_dimension": 0.0}},
  "annotation_confidence": 0.0,
  "ambiguity_reason": "",
  "rationale": "brief boundary-based reason"
}}

Ranking contains exactly three unique closed labels; Top-1 equals closed_intent; scores strictly descend in [0,1].
Profile signals are in [-1,1]. Expected deltas are signed values in [-0.2,0.2].
Profile dimensions: {PERSONALITY_DIMS}
Affect dimensions: {AFFECT_DIMS}
"""


def annotation_issues(data: Any) -> list[str]:
    if not isinstance(data, dict):
        return ["annotation must be an object"]
    issues: list[str] = []
    closed = str(data.get("closed_intent", "")).strip().lower()
    if closed not in CLOSED_INTENTS:
        issues.append("closed_intent is invalid")
    if len(str(data.get("natural_language_intent", "")).strip()) < 20:
        issues.append("natural_language_intent is too short")
    ranking = data.get("intent_ranking", [])
    if not isinstance(ranking, list) or len(ranking) != 3:
        issues.append("intent_ranking must contain exactly 3 items")
    else:
        labels = [str(item.get("intent", "")).strip().lower() for item in ranking if isinstance(item, dict)]
        try:
            scores = [float(item.get("score")) for item in ranking if isinstance(item, dict)]
        except (TypeError, ValueError):
            scores = []
        if len(labels) != 3 or any(label not in CLOSED_INTENTS for label in labels):
            issues.append("ranking contains invalid labels")
        elif len(set(labels)) != 3:
            issues.append("ranking labels are not unique")
        elif labels[0] != closed:
            issues.append("ranking Top-1 does not match closed_intent")
        if len(scores) != 3 or not (1.0 >= scores[0] > scores[1] > scores[2] >= 0.0):
            issues.append("ranking scores are not strictly descending")
    slots = data.get("slots", {})
    if not isinstance(slots, dict) or set(slots) != REQUIRED_SLOT_KEYS:
        issues.append("slots do not match the six-key contract")
    elif any(not str(slots.get(key, "")).strip() for key in REQUIRED_SLOT_KEYS) or slots.get("scope") != "next_panel":
        issues.append("slots contain empty values or invalid scope")
    for field, dims, low, high in [
        ("profile_signal", set(PERSONALITY_DIMS), -1.0, 1.0),
        ("expected_profile_delta", set(PERSONALITY_DIMS), -0.2, 0.2),
        ("expected_affective_delta", set(AFFECT_DIMS), -0.2, 0.2),
    ]:
        values = data.get(field, {})
        if not isinstance(values, dict) or not set(values).issubset(dims):
            issues.append(f"{field} contains invalid dimensions")
            continue
        try:
            if any(not low <= float(value) <= high for value in values.values()):
                issues.append(f"{field} contains out-of-range values")
        except (TypeError, ValueError):
            issues.append(f"{field} contains non-numeric values")
    try:
        if not 0.0 <= float(data.get("annotation_confidence")) <= 1.0:
            issues.append("annotation_confidence is outside [0,1]")
    except (TypeError, ValueError):
        issues.append("annotation_confidence is missing")
    return issues


def normalize_annotation(data: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(data)
    result["closed_intent"] = str(data["closed_intent"]).strip().lower()
    result["natural_language_intent"] = str(data["natural_language_intent"]).strip()
    result["intent_ranking"] = [
        {
            "rank": index,
            "intent": str(item["intent"]).strip().lower(),
            "score": round(clamp(float(item["score"])), 6),
        }
        for index, item in enumerate(data["intent_ranking"], 1)
    ]
    result["slots"] = {key: str(data["slots"][key]).strip() for key in REQUIRED_SLOT_KEYS}
    result["profile_signal"] = {
        key: round(max(-1.0, min(1.0, float(value))), 4)
        for key, value in data.get("profile_signal", {}).items()
        if key in PERSONALITY_DIMS
    }
    result["expected_affective_delta"] = coerce_delta(data.get("expected_affective_delta", {}), AFFECT_DIMS)
    result["expected_profile_delta"] = coerce_delta(data.get("expected_profile_delta", {}), PERSONALITY_DIMS)
    result["annotation_confidence"] = round(clamp(float(data["annotation_confidence"])), 6)
    result["ambiguity_reason"] = str(data.get("ambiguity_reason", "")).strip()
    result["rationale"] = str(data.get("rationale", "")).strip()
    return result


def mock_annotation(edge: dict[str, Any]) -> dict[str, Any]:
    closed = str(edge.get("closed_intent", "")).strip().lower()
    if closed not in CLOSED_INTENTS:
        closed = "interact"
    slots = edge.get("slots", {})
    action = str(slots.get("action", "inspect the selected target")).strip()
    target = str(slots.get("target", edge.get("grounding", {}).get("target_label", "selected target"))).strip()
    goal = str(slots.get("narrative_goal", edge.get("branch_label", "continue the story"))).strip()
    alternatives = INTENT_CONFUSIONS[closed]
    return {
        "closed_intent": closed,
        "natural_language_intent": f"The user wants to {action} with {target} so the next panel can {goal}.",
        "intent_ranking": [
            {"rank": 1, "intent": closed, "score": 0.6},
            {"rank": 2, "intent": alternatives[0], "score": 0.25},
            {"rank": 3, "intent": alternatives[1], "score": 0.15},
        ],
        "slots": {
            "action": action,
            "target": target,
            "narrative_goal": goal,
            "mood": str(slots.get("mood", "consistent with the current scene")).strip(),
            "continuity_constraint": str(
                slots.get("continuity_constraint", "preserve characters, objects, and location logic")
            ).strip(),
            "scope": "next_panel",
        },
        "profile_signal": {
            key: value for key, value in edge.get("profile_signal", {}).items() if key in PERSONALITY_DIMS
        },
        "expected_affective_delta": coerce_delta(edge.get("expected_affective_delta", {}), AFFECT_DIMS),
        "expected_profile_delta": coerce_delta(edge.get("expected_profile_delta", {}), PERSONALITY_DIMS),
        "annotation_confidence": 0.5,
        "ambiguity_reason": "Dry-run preserves the legacy class and requires real-model adjudication.",
        "rationale": "Dry-run placeholder; no semantic relabeling was performed.",
    }


def call_validated(
    client: OpenAICompatibleClient,
    *,
    system_prompt: str,
    user_prompt: str,
    max_retries: int,
    validator: Callable[[Any], list[str]],
    images: list[str] | None = None,
) -> dict[str, Any]:
    issues: list[str] = []
    data: dict[str, Any] = {}
    for _attempt in range(max(1, max_retries)):
        correction = ""
        if issues:
            correction = "\n\nCorrect every validation failure and return the full JSON:\n- " + "\n- ".join(issues)
        data = client.chat_json(system_prompt=system_prompt, user_prompt=user_prompt + correction, images=images)
        issues = validator(data)
        if not issues:
            return data
    raise ValueError("Model output failed validation: " + "; ".join(issues))


def annotate_text(
    edge: dict[str, Any],
    evidence: dict[str, Any],
    *,
    client: OpenAICompatibleClient | None,
    dry_run: bool,
    text_passes: int,
    max_retries: int,
) -> dict[str, Any]:
    if dry_run:
        selected = normalize_annotation(mock_annotation(edge))
        return {"passes": [selected], "selected": selected, "annotator_disagreement": False}
    assert client is not None
    passes = []
    for index in range(max(1, text_passes)):
        raw = call_validated(
            client,
            system_prompt=TEXT_SYSTEM_PROMPT,
            user_prompt=text_prompt(evidence, pass_index=index + 1),
            max_retries=max_retries,
            validator=annotation_issues,
        )
        passes.append(normalize_annotation(raw))
    disagreement = len({item["closed_intent"] for item in passes}) > 1
    if disagreement:
        raw = call_validated(
            client,
            system_prompt=TEXT_SYSTEM_PROMPT,
            user_prompt=text_prompt(evidence, pass_index=len(passes) + 1, candidates=passes),
            max_retries=max_retries,
            validator=annotation_issues,
        )
        selected = normalize_annotation(raw)
    else:
        selected = max(passes, key=lambda item: item["annotation_confidence"])
    return {"passes": passes, "selected": selected, "annotator_disagreement": disagreement}


def vision_prompt(edge: dict[str, Any]) -> str:
    return f"""
The first image is the source panel; the second is the generated target panel.

Planned intended label: {edge["closed_intent"]}
Planned natural-language intent: {edge["natural_language_intent"]}
Planned definition: {CLOSED_INTENTS[edge["closed_intent"]]}
All definitions: {json.dumps(CLOSED_INTENTS, ensure_ascii=False)}
Boundary rules: {json.dumps(INTENT_BOUNDARY_RULES, ensure_ascii=False)}
Expected affective delta: {json.dumps(edge.get("expected_affective_delta", {}), ensure_ascii=False)}
Expected profile delta: {json.dumps(edge.get("expected_profile_delta", {}), ensure_ascii=False)}
Scale metadata: {json.dumps(SCALE_METADATA, ensure_ascii=False)}

Return exactly:
{{
  "observed_visual_intent": "one closed label or oos",
  "image_intent_alignment": 0.0,
  "continuity_alignment": 0.0,
  "observed_affective_delta": {{"valid affect dimensions": 0.0}},
  "observed_profile_delta": {{"valid profile dimensions": 0.0}},
  "visual_notes": "what the images actually show",
  "revision_suggestion": ""
}}
Alignment scores are in [0,1]. Deltas are signed values in [-0.2,0.2].
"""


def vision_issues(data: Any) -> list[str]:
    if not isinstance(data, dict):
        return ["vision result must be an object"]
    issues: list[str] = []
    if str(data.get("observed_visual_intent", "")).strip().lower() not in set(CLOSED_INTENTS) | {"oos"}:
        issues.append("observed_visual_intent is invalid")
    for field in ["image_intent_alignment", "continuity_alignment"]:
        try:
            if not 0.0 <= float(data.get(field)) <= 1.0:
                issues.append(f"{field} is outside [0,1]")
        except (TypeError, ValueError):
            issues.append(f"{field} is missing")
    for field, dims in [
        ("observed_affective_delta", set(AFFECT_DIMS)),
        ("observed_profile_delta", set(PERSONALITY_DIMS)),
    ]:
        values = data.get(field, {})
        if not isinstance(values, dict) or not set(values).issubset(dims):
            issues.append(f"{field} contains invalid dimensions")
            continue
        try:
            if any(not -0.2 <= float(value) <= 0.2 for value in values.values()):
                issues.append(f"{field} contains out-of-range values")
        except (TypeError, ValueError):
            issues.append(f"{field} contains non-numeric values")
    if not str(data.get("visual_notes", "")).strip():
        issues.append("visual_notes is required")
    return issues


def normalize_vision(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "observed_visual_intent": str(data["observed_visual_intent"]).strip().lower(),
        "image_intent_alignment": round(clamp(float(data["image_intent_alignment"])), 6),
        "continuity_alignment": round(clamp(float(data["continuity_alignment"])), 6),
        "observed_affective_delta": coerce_delta(data.get("observed_affective_delta", {}), AFFECT_DIMS),
        "observed_profile_delta": coerce_delta(data.get("observed_profile_delta", {}), PERSONALITY_DIMS),
        "visual_notes": str(data.get("visual_notes", "")).strip(),
        "revision_suggestion": str(data.get("revision_suggestion", "")).strip(),
    }


def calibrate_vision(
    edge: dict[str, Any],
    *,
    source_image: Path,
    target_image: Path,
    client: OpenAICompatibleClient | None,
    dry_run: bool,
    max_retries: int,
) -> dict[str, Any]:
    if not source_image.is_file() or not target_image.is_file():
        return {
            "observed_visual_intent": "oos",
            "image_intent_alignment": 0.0,
            "continuity_alignment": 0.0,
            "observed_affective_delta": {},
            "observed_profile_delta": {},
            "visual_notes": "Source or target image is missing.",
            "revision_suggestion": "Restore the image before visual calibration.",
            "missing_image": True,
        }
    if dry_run:
        return {
            "observed_visual_intent": edge["closed_intent"],
            "image_intent_alignment": 0.5,
            "continuity_alignment": 0.5,
            "observed_affective_delta": edge.get("expected_affective_delta", {}),
            "observed_profile_delta": edge.get("expected_profile_delta", {}),
            "visual_notes": "Dry-run placeholder; images were not evaluated by gpt-5.5.",
            "revision_suggestion": "Run with the configured API.",
        }
    assert client is not None
    raw = call_validated(
        client,
        system_prompt=VISION_SYSTEM_PROMPT,
        user_prompt=vision_prompt(edge),
        images=[str(source_image), str(target_image)],
        max_retries=max_retries,
        validator=vision_issues,
    )
    return normalize_vision(raw)


def cache_path(output_dir: Path, stage: str, tree_id: str, edge_id: str) -> Path:
    safe_tree = "".join(char if char.isalnum() or char in "_-" else "_" for char in tree_id)
    safe_edge = "".join(char if char.isalnum() or char in "_-" else "_" for char in edge_id)
    return output_dir / "cache" / stage / safe_tree / f"{safe_edge}.json"


def recompute_states(tree: dict[str, Any], *, expected_weight: float, use_visual: bool) -> None:
    nodes = {str(node["node_id"]): node for node in tree.get("nodes", [])}
    incoming = {str(edge["target_node"]): edge for edge in tree.get("edges", [])}
    ordered = sorted(tree.get("nodes", []), key=lambda node: (int(node.get("depth", 0)), str(node["node_id"])))
    for node in ordered:
        node_id = str(node["node_id"])
        if node_id == "n0" or node_id not in incoming:
            continue
        edge = incoming[node_id]
        parent = nodes[str(edge["source_node"])]
        affect_delta = edge.get("expected_affective_delta", {})
        profile_delta = edge.get("expected_profile_delta", {})
        visual = edge.get("visual_calibration")
        if use_visual and visual:
            affect_delta = blend_deltas(
                expected=affect_delta,
                observed=visual.get("observed_affective_delta", {}),
                dims=AFFECT_DIMS,
                expected_weight=expected_weight,
            )
            profile_delta = blend_deltas(
                expected=profile_delta,
                observed=visual.get("observed_profile_delta", {}),
                dims=PERSONALITY_DIMS,
                expected_weight=expected_weight,
            )
            edge["observed_affective_delta"] = visual.get("observed_affective_delta", {})
            edge["observed_profile_delta"] = visual.get("observed_profile_delta", {})
        edge["final_affective_delta"] = affect_delta
        edge["final_profile_delta"] = profile_delta
        node["affective_state"] = update_affective_state(
            before=parent.get("affective_state", {}), delta=affect_delta, decay=0.75
        )
        node["stable_profile"] = update_stable_profile(
            before=parent.get("stable_profile", {}), delta=profile_delta
        )
        node["current_state"] = compute_current_state(
            stable_profile=node["stable_profile"], affective_state=node["affective_state"], alpha=0.30
        )
        node["state_recomputed_by"] = SCHEMA_VERSION


def process_tree(
    source_tree_path: Path,
    *,
    batch_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    stage: str,
    dry_run: bool,
    overwrite_cache: bool,
    text_client: OpenAICompatibleClient | None,
    vision_client: OpenAICompatibleClient | None,
    remaining_edges: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], int]:
    original = read_json(source_tree_path)
    tree = copy.deepcopy(original)
    tree_id = str(tree["tree_id"])
    source_nodes = {str(node["node_id"]): node for node in original.get("nodes", [])}
    output_nodes = {str(node["node_id"]): node for node in tree.get("nodes", [])}
    settings = config["processing"]
    reviews: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    processed = 0

    for node in tree.get("nodes", []):
        resolved = resolve_image_path(source_tree_path, node.get("image"), str(node["node_id"]))
        node["image_ref"] = relative_image_ref(batch_dir, resolved)
        node["legacy_image_path"] = node.get("image", "")

    for edge in tree.get("edges", []):
        if remaining_edges is not None and processed >= remaining_edges:
            break
        edge_id = str(edge["edge_id"])
        legacy = copy.deepcopy(edge)
        evidence = edge_evidence(original, legacy, source_nodes)
        text_cache = cache_path(output_dir, "text", tree_id, edge_id)
        if text_cache.exists() and not overwrite_cache:
            text_result = read_json(text_cache)
        else:
            if stage == "vision":
                raise FileNotFoundError(f"Missing text cache for vision stage: {text_cache}")
            text_result = annotate_text(
                legacy,
                evidence,
                client=text_client,
                dry_run=dry_run,
                text_passes=int(settings["text_passes"]),
                max_retries=int(settings["max_retries"]),
            )
            write_json(text_cache, text_result)
        selected = text_result["selected"]

        edge["legacy_intent_annotation"] = {
            "closed_intent": legacy.get("closed_intent"),
            "intent_ranking": legacy.get("intent_ranking", []),
            "slots": legacy.get("slots", {}),
            "profile_signal": legacy.get("profile_signal", {}),
            "decision": legacy.get("decision", {}),
        }
        edge["closed_intent"] = selected["closed_intent"]
        edge["intended_closed_intent"] = selected["closed_intent"]
        edge["natural_language_intent"] = selected["natural_language_intent"]
        edge["intent_ranking"] = selected["intent_ranking"]
        edge["slots"] = selected["slots"]
        edge["profile_signal"] = selected["profile_signal"]
        edge["expected_affective_delta"] = selected["expected_affective_delta"]
        edge["expected_profile_delta"] = selected["expected_profile_delta"]
        edge["decision"] = {
            "type": "accept",
            "confidence": selected["annotation_confidence"],
            "requires_confirmation": False,
            "is_oos": False,
        }
        box, box_changed = normalize_box(edge.get("grounding", {}).get("target_box"))
        edge.setdefault("grounding", {})["target_box"] = box
        target_node = output_nodes.get(str(edge.get("target_node")), {})
        edge["target_image_ref"] = target_node.get("image_ref", "")
        edge["annotation_metadata"] = {
            "schema_version": SCHEMA_VERSION,
            "text_model": "dry-run" if dry_run else config["text_api"]["model"],
            "annotation_confidence": selected["annotation_confidence"],
            "annotator_disagreement": bool(text_result.get("annotator_disagreement")),
            "ambiguity_reason": selected.get("ambiguity_reason", ""),
            "rationale": selected.get("rationale", ""),
            "grounding_box_changed": box_changed,
            "needs_human_review": False,
        }

        if stage in {"vision", "all"}:
            source_node = output_nodes[str(edge["source_node"])]
            source_image = resolve_image_path(
                source_tree_path, source_node.get("legacy_image_path"), str(source_node["node_id"])
            )
            target_image = resolve_image_path(
                source_tree_path, target_node.get("legacy_image_path"), str(target_node.get("node_id", ""))
            )
            vision_cache = cache_path(output_dir, "vision", tree_id, edge_id)
            if vision_cache.exists() and not overwrite_cache:
                visual = read_json(vision_cache)
            else:
                visual = calibrate_vision(
                    edge,
                    source_image=source_image,
                    target_image=target_image,
                    client=vision_client,
                    dry_run=dry_run,
                    max_retries=int(settings["max_retries"]),
                )
                write_json(vision_cache, visual)
            edge["observed_visual_intent"] = visual["observed_visual_intent"]
            edge["visual_calibration"] = visual
            edge["delta_alignment"] = {
                "image_intent_alignment": visual["image_intent_alignment"],
                "continuity_alignment": visual["continuity_alignment"],
                "intent_matches": visual["observed_visual_intent"] == edge["closed_intent"],
                "needs_revision": (
                    visual["observed_visual_intent"] != edge["closed_intent"]
                    or visual["image_intent_alignment"] < float(settings["visual_alignment_threshold"])
                ),
            }

        reasons = []
        if selected["annotation_confidence"] < float(settings["confidence_threshold"]):
            reasons.append("low_text_confidence")
        if text_result.get("annotator_disagreement"):
            reasons.append("text_annotator_disagreement")
        if edge["closed_intent"] in set(settings.get("always_review_labels", [])):
            reasons.append("always_review_label")
        visual = edge.get("visual_calibration")
        if visual:
            if visual["observed_visual_intent"] != edge["closed_intent"]:
                reasons.append("text_visual_label_conflict")
            if visual["image_intent_alignment"] < float(settings["visual_alignment_threshold"]):
                reasons.append("low_visual_alignment")
            if visual.get("missing_image"):
                reasons.append("missing_image")
        edge["annotation_metadata"]["needs_human_review"] = bool(reasons)
        audit = {
            "tree_id": tree_id,
            "edge_id": edge_id,
            "source_tree": str(source_tree_path),
            "legacy_closed_intent": legacy.get("closed_intent"),
            "intended_closed_intent": edge["closed_intent"],
            "label_changed": legacy.get("closed_intent") != edge["closed_intent"],
            "annotation_confidence": selected["annotation_confidence"],
            "observed_visual_intent": edge.get("observed_visual_intent"),
            "image_intent_alignment": (visual or {}).get("image_intent_alignment"),
            "review_reasons": reasons,
        }
        audits.append(audit)
        if reasons:
            reviews.append(audit | {"evidence": evidence, "natural_language_intent": edge["natural_language_intent"]})
        processed += 1

    for region in tree.get("oos_regions", []):
        box, _changed = normalize_box(region.get("grounding", {}).get("target_box"))
        region.setdefault("grounding", {})["target_box"] = box
        region["natural_language_intent"] = (
            "The click is outside the supported intent space and needs free-text clarification."
        )
        region["decision"] = {
            "type": "oos",
            "confidence": 0.2,
            "requires_confirmation": True,
            "is_oos": True,
        }

    recompute_states(
        tree,
        expected_weight=float(settings["calibration_expected_weight"]),
        use_visual=stage in {"vision", "all"},
    )
    tree["postprocessing_metadata"] = {
        "schema_version": SCHEMA_VERSION,
        "source_tree": str(source_tree_path),
        "source_tree_sha256": file_sha256(source_tree_path),
        "source_batch": batch_dir.name,
        "stage": stage,
        "dry_run": dry_run,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "images_modified": False,
    }
    return tree, audits, reviews, processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Non-destructively relabel and calibrate legacy intent trees.")
    parser.add_argument("--batch-dir", required=True)
    parser.add_argument("--output-dir", default="", help="Defaults to <batch>/postprocessed_intent_v2.")
    parser.add_argument("--config", default="", help="Private JSON API config; keep it out of version control.")
    parser.add_argument("--stage", choices=["text", "vision", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true", help="No API calls; labels remain placeholders.")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--limit-trees", type=int, default=0)
    parser.add_argument("--tree-offset", type=int, default=0, help="Skip this many active trees before applying limits.")
    parser.add_argument("--limit-edges", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_dir = Path(args.batch_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else batch_dir / "postprocessed_intent_v2"
    config = load_config(Path(args.config).resolve() if args.config else None)
    text_client = make_client(config, "text_api", dry_run=args.dry_run) if args.stage in {"text", "all"} else None
    vision_client = make_client(config, "vision_api", dry_run=args.dry_run) if args.stage in {"vision", "all"} else None
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "config.public.json", public_config(config))

    tree_paths = active_tree_paths(batch_dir)
    if args.tree_offset > 0:
        tree_paths = tree_paths[args.tree_offset :]
    if args.limit_trees > 0:
        tree_paths = tree_paths[: args.limit_trees]
    audits: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    processed_edges = 0
    for index, source_tree_path in enumerate(tree_paths, 1):
        remaining = None if args.limit_edges <= 0 else max(0, args.limit_edges - processed_edges)
        if remaining == 0:
            break
        tree, tree_audits, tree_reviews, count = process_tree(
            source_tree_path,
            batch_dir=batch_dir,
            output_dir=output_dir,
            config=config,
            stage=args.stage,
            dry_run=args.dry_run,
            overwrite_cache=args.overwrite_cache,
            text_client=text_client,
            vision_client=vision_client,
            remaining_edges=remaining,
        )
        relative_tree = Path("trees") / f"user_{tree['user_id']}" / str(tree["topic_id"]) / "tree.json"
        write_json(output_dir / relative_tree, tree)
        audits.extend(tree_audits)
        reviews.extend(tree_reviews)
        processed_edges += count
        manifest.append(
            {
                "tree_id": tree["tree_id"],
                "user_id": tree["user_id"],
                "topic_id": tree["topic_id"],
                "source_tree": str(source_tree_path),
                "output_tree": relative_tree.as_posix(),
                "processed_edges": count,
            }
        )
        print(f"[{index}/{len(tree_paths)}] {tree['tree_id']}: {count} edges", flush=True)

    write_jsonl(output_dir / "manifest.jsonl", manifest)
    write_jsonl(output_dir / "relabel_audit.jsonl", audits)
    write_jsonl(output_dir / "review_queue.jsonl", reviews)
    legacy_counts = Counter(str(row["legacy_closed_intent"]) for row in audits)
    v2_counts = Counter(str(row["intended_closed_intent"]) for row in audits)
    observed_counts = Counter(str(row["observed_visual_intent"]) for row in audits if row["observed_visual_intent"])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "source_batch": str(batch_dir),
        "output_dir": str(output_dir),
        "stage": args.stage,
        "dry_run": args.dry_run,
        "tree_count": len(manifest),
        "edge_count": len(audits),
        "review_count": len(reviews),
        "changed_label_count": sum(bool(row["label_changed"]) for row in audits),
        "legacy_class_counts": dict(legacy_counts),
        "intended_v2_class_counts": dict(v2_counts),
        "observed_visual_class_counts": dict(observed_counts),
        "images_modified": False,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
