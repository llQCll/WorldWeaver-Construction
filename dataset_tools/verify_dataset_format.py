from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT_REQUIRED_KEYS = {
    "topic_id",
    "topic",
    "synopsis",
    "world_bible",
    "root_state",
    "root_prompt",
    "root_image",
}

TREE_REQUIRED_KEYS = {
    "tree_id",
    "topic_id",
    "user_id",
    "world_bible",
    "root_image",
    "nodes",
    "edges",
}

NODE_REQUIRED_KEYS = {
    "node_id",
    "depth",
    "image",
    "story_state",
    "stable_profile",
    "affective_state",
    "current_state",
}

EDGE_REQUIRED_KEYS = {
    "edge_id",
    "source_node",
    "target_node",
    "closed_intent",
    "grounding",
    "intent_ranking",
    "slots",
    "decision",
    "expected_affective_delta",
    "expected_profile_delta",
    "story_state_after",
    "generation_prompt",
}

PROFILE_REQUIRED_KEYS = {
    "user_id",
    "profile",
}

NORMALIZED_PROFILE_REQUIRED_KEYS = {
    "user_id",
    "raw_profile",
    "stable_profile",
    "affective_state_0",
    "profile_summary",
    "dimension_rationales",
}

STABLE_PROFILE_DIMS = {
    "goal_progress",
    "mastery_logic",
    "challenge_seeking",
    "social_attachment",
    "cooperative_orientation",
    "world_discovery",
    "role_immersion",
    "aesthetic_customization",
}

AFFECT_DIMS = {
    "pleasure",
    "arousal",
    "dominance",
    "tension",
    "curiosity",
    "empathy",
    "cognitive_load",
}

CLOSED_INTENTS = {
    "zoom_in",
    "reveal",
    "branch_out",
    "reframe",
    "follow",
    "interact",
}

REQUIRED_SLOT_KEYS = {
    "action",
    "target",
    "narrative_goal",
    "mood",
    "continuity_constraint",
    "scope",
}

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def fail(errors: list[str], path: Path, message: str) -> None:
    errors.append(f"{path}: {message}")


def has_absolute_or_private_path(value: Any) -> bool:
    if isinstance(value, dict):
        return any(has_absolute_or_private_path(item) for item in value.values())
    if isinstance(value, list):
        return any(has_absolute_or_private_path(item) for item in value)
    if not isinstance(value, str):
        return False
    normalized = value.replace("\\", "/").lower()
    return (
        ":/users/" in normalized
        or normalized.startswith("c:/")
        or "/desktop/" in normalized
        or "dataset_runs/" in normalized
        or "dataset_runs\\" in value.lower()
    )


def verify_root_assets(root_dir: Path) -> list[str]:
    errors: list[str] = []
    if not root_dir.exists():
        fail(errors, root_dir, "root asset directory does not exist")
        return errors

    topic_dirs = sorted(path for path in root_dir.iterdir() if path.is_dir())
    if not topic_dirs:
        fail(errors, root_dir, "no topic root directories found")
        return errors

    for topic_dir in topic_dirs:
        root_png = topic_dir / "root.png"
        root_asset = topic_dir / "root_asset.json"
        synopsis = topic_dir / "synopsis.txt"
        prompt = topic_dir / "root_prompt.txt"
        generation_meta = topic_dir / "generation_metadata.json"

        for required in [root_png, root_asset, synopsis, prompt, generation_meta]:
            if not required.exists():
                fail(errors, required, "missing required root asset file")

        if root_png.exists() and root_png.read_bytes()[:8] != PNG_SIGNATURE:
            fail(errors, root_png, "file is not a valid PNG")

        if synopsis.exists() and not synopsis.read_text(encoding="utf-8").strip():
            fail(errors, synopsis, "synopsis is empty")

        if prompt.exists() and not prompt.read_text(encoding="utf-8").strip():
            fail(errors, prompt, "root prompt is empty")

        if root_asset.exists():
            data = read_json(root_asset)
            missing = sorted(ROOT_REQUIRED_KEYS - set(data))
            if missing:
                fail(errors, root_asset, f"missing keys: {', '.join(missing)}")
            if data.get("topic_id") != topic_dir.name:
                fail(errors, root_asset, "topic_id must match directory name")
            if data.get("root_image") != "root.png":
                fail(errors, root_asset, "root_image must be the relative path root.png")
            if has_absolute_or_private_path(data):
                fail(errors, root_asset, "contains absolute/private local paths")

        if generation_meta.exists():
            data = read_json(generation_meta)
            if data.get("path") != "root.png":
                fail(errors, generation_meta, "path must be the relative path root.png")
            if has_absolute_or_private_path(data):
                fail(errors, generation_meta, "contains absolute/private local paths")

    return errors


def number_in_range(value: Any, low: float, high: float) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return low <= number <= high


def normalized_box(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    return 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0


def verify_intent_edge(errors: list[str], edge_path: Path, edge: dict[str, Any]) -> None:
    closed_intent = str(edge.get("closed_intent", "")).strip().lower()
    if closed_intent not in CLOSED_INTENTS:
        fail(errors, edge_path, f"closed_intent must be one of {sorted(CLOSED_INTENTS)}")

    natural_intent = str(edge.get("natural_language_intent", "")).strip()
    if len(natural_intent) < 20:
        fail(errors, edge_path, "natural_language_intent must be a specific sentence")

    ranking = edge.get("intent_ranking")
    if not isinstance(ranking, list) or len(ranking) != 3:
        fail(errors, edge_path, "intent_ranking must contain exactly 3 items")
    else:
        labels = [
            str(item.get("intent", "")).strip().lower()
            for item in ranking
            if isinstance(item, dict)
        ]
        scores = [item.get("score") for item in ranking if isinstance(item, dict)]
        if len(labels) != 3 or any(label not in CLOSED_INTENTS for label in labels):
            fail(errors, edge_path, "intent_ranking must use only the six closed labels")
        elif len(set(labels)) != 3:
            fail(errors, edge_path, "intent_ranking labels must be unique")
        elif labels[0] != closed_intent:
            fail(errors, edge_path, "intent_ranking rank 1 must equal closed_intent")
        if (
            len(scores) != 3
            or not all(number_in_range(score, 0.0, 1.0) for score in scores)
            or not float(scores[0]) > float(scores[1]) > float(scores[2])
        ):
            fail(errors, edge_path, "intent_ranking scores must be strictly descending in [0,1]")

    grounding = edge.get("grounding", {})
    if not isinstance(grounding, dict) or not normalized_box(grounding.get("target_box")):
        fail(errors, edge_path, "grounding.target_box must be a valid normalized box")
    if not str(grounding.get("target_label", "")).strip():
        fail(errors, edge_path, "grounding.target_label is required")

    slots = edge.get("slots", {})
    if not isinstance(slots, dict) or set(slots) != REQUIRED_SLOT_KEYS:
        fail(errors, edge_path, f"slots must contain exactly {sorted(REQUIRED_SLOT_KEYS)}")
    elif any(not str(slots.get(key, "")).strip() for key in REQUIRED_SLOT_KEYS):
        fail(errors, edge_path, "all slot values must be non-empty")
    elif slots.get("scope") != "next_panel":
        fail(errors, edge_path, "slots.scope must be next_panel")

    for field, allowed_dims, low, high in [
        ("profile_signal", STABLE_PROFILE_DIMS, -1.0, 1.0),
        ("expected_profile_delta", STABLE_PROFILE_DIMS, -0.2, 0.2),
        ("expected_affective_delta", AFFECT_DIMS, -0.2, 0.2),
    ]:
        values = edge.get(field, {})
        if not isinstance(values, dict) or not set(values).issubset(allowed_dims):
            fail(errors, edge_path, f"{field} contains invalid dimensions")
        elif not all(number_in_range(value, low, high) for value in values.values()):
            fail(errors, edge_path, f"{field} values must be in [{low},{high}]")

    decision = edge.get("decision", {})
    if (
        not isinstance(decision, dict)
        or decision.get("type") != "accept"
        or bool(decision.get("requires_confirmation"))
        or bool(decision.get("is_oos"))
    ):
        fail(errors, edge_path, "decision must be accepted, in-scope, and not require confirmation")


def verify_tree(tree_path: Path, *, strict_intent: bool = False) -> list[str]:
    errors: list[str] = []
    tree = read_json(tree_path)
    missing = sorted(TREE_REQUIRED_KEYS - set(tree))
    if missing:
        fail(errors, tree_path, f"missing tree keys: {', '.join(missing)}")
        return errors

    nodes = tree.get("nodes", [])
    edges = tree.get("edges", [])
    node_ids = {node.get("node_id") for node in nodes}
    if "n0" not in node_ids:
        fail(errors, tree_path, "missing root node n0")

    for node in nodes:
        node_path = tree_path.with_name(f"node:{node.get('node_id', '<missing>')}")
        missing = sorted(NODE_REQUIRED_KEYS - set(node))
        if missing:
            fail(errors, node_path, f"missing node keys: {', '.join(missing)}")

    for edge in edges:
        edge_path = tree_path.with_name(f"edge:{edge.get('edge_id', '<missing>')}")
        missing = sorted(EDGE_REQUIRED_KEYS - set(edge))
        if missing:
            fail(errors, edge_path, f"missing edge keys: {', '.join(missing)}")
        if edge.get("source_node") not in node_ids:
            fail(errors, edge_path, "source_node does not exist")
        if edge.get("target_node") not in node_ids:
            fail(errors, edge_path, "target_node does not exist")
        if strict_intent:
            verify_intent_edge(errors, edge_path, edge)

    if strict_intent:
        for index, region in enumerate(tree.get("oos_regions", [])):
            region_path = tree_path.with_name(f"oos:{index}")
            decision = region.get("decision", {})
            if (
                decision.get("type") != "oos"
                or not bool(decision.get("requires_confirmation"))
                or not bool(decision.get("is_oos"))
            ):
                fail(errors, region_path, "OOS decision must require free-text confirmation")
            if not normalized_box(region.get("grounding", {}).get("target_box")):
                fail(errors, region_path, "OOS grounding.target_box must be a valid normalized box")

    return errors


def verify_tree_manifest_profile_consistency(
    manifest_path: Path,
    *,
    tolerance: float = 1e-9,
) -> list[str]:
    errors: list[str] = []
    if not manifest_path.is_file():
        fail(errors, manifest_path, "tree manifest does not exist")
        return errors

    reference_by_user: dict[str, tuple[str, dict[str, float]]] = {}
    row_count = 0
    with manifest_path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row_count += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(errors, manifest_path, f"line {line_number} is not valid JSON: {exc}")
                continue
            tree_value = row.get("tree_path", row.get("output_tree", ""))
            tree_path = Path(str(tree_value))
            if not tree_path.is_absolute():
                tree_path = manifest_path.parent / tree_path
            if not tree_path.is_file():
                fail(errors, manifest_path, f"line {line_number} tree does not exist: {tree_path}")
                continue
            tree = read_json(tree_path)
            root = next(
                (node for node in tree.get("nodes", []) if str(node.get("node_id")) == "n0"),
                None,
            )
            if root is None:
                fail(errors, tree_path, "missing root node n0")
                continue
            profile = root.get("stable_profile", {})
            if set(profile) != STABLE_PROFILE_DIMS:
                fail(errors, tree_path, "root stable_profile dimensions mismatch")
                continue
            normalized = {dimension: float(profile[dimension]) for dimension in STABLE_PROFILE_DIMS}
            user_id = str(row.get("user_id", tree.get("user_id", "")))
            topic_id = str(row.get("topic_id", tree.get("topic_id", "")))
            previous = reference_by_user.get(user_id)
            if previous is None:
                reference_by_user[user_id] = topic_id, normalized
                continue
            previous_topic, reference = previous
            max_difference = max(abs(normalized[dim] - reference[dim]) for dim in STABLE_PROFILE_DIMS)
            if max_difference > tolerance:
                fail(
                    errors,
                    tree_path,
                    f"user_id={user_id} root profile differs from topic={previous_topic}; "
                    f"max_dimension_difference={max_difference:.6f}",
                )
    if row_count == 0:
        fail(errors, manifest_path, "tree manifest has no rows")
    return errors


def verify_profiles(profile_path: Path) -> list[str]:
    errors: list[str] = []
    if not profile_path.exists():
        fail(errors, profile_path, "profile file does not exist")
        return errors

    row_count = 0
    seen_ids: set[str] = set()
    with profile_path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(errors, profile_path, f"line {line_number} is not valid JSON: {exc}")
                continue
            row_count += 1
            missing = sorted(PROFILE_REQUIRED_KEYS - set(row))
            if missing:
                fail(errors, profile_path, f"line {line_number} missing keys: {', '.join(missing)}")
            user_id = str(row.get("user_id", "")).strip()
            if not user_id:
                fail(errors, profile_path, f"line {line_number} has empty user_id")
            if user_id in seen_ids:
                fail(errors, profile_path, f"line {line_number} duplicates user_id={user_id}")
            seen_ids.add(user_id)
            if not str(row.get("profile", "")).strip():
                fail(errors, profile_path, f"line {line_number} has empty profile")

    if row_count == 0:
        fail(errors, profile_path, "profile file has no rows")
    return errors


def verify_normalized_profiles(profile_path: Path) -> list[str]:
    errors: list[str] = []
    if not profile_path.exists():
        fail(errors, profile_path, "normalized profile file does not exist")
        return errors

    row_count = 0
    with profile_path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(errors, profile_path, f"line {line_number} is not valid JSON: {exc}")
                continue
            row_count += 1
            missing = sorted(NORMALIZED_PROFILE_REQUIRED_KEYS - set(row))
            if missing:
                fail(errors, profile_path, f"line {line_number} missing keys: {', '.join(missing)}")
            stable = row.get("stable_profile", {})
            affect = row.get("affective_state_0", {})
            if set(stable) != STABLE_PROFILE_DIMS:
                fail(errors, profile_path, f"line {line_number} stable_profile dimensions mismatch")
            if set(affect) != AFFECT_DIMS:
                fail(errors, profile_path, f"line {line_number} affective_state_0 dimensions mismatch")
            for dim, value in stable.items():
                if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
                    fail(errors, profile_path, f"line {line_number} stable_profile.{dim} must be in [0,1]")
            for dim, value in affect.items():
                if float(value) != 0.5:
                    fail(errors, profile_path, f"line {line_number} affective_state_0.{dim} must be exactly 0.5")
            if not str(row.get("profile_summary", "")).strip():
                fail(errors, profile_path, f"line {line_number} profile_summary is empty")

    if row_count == 0:
        fail(errors, profile_path, "normalized profile file has no rows")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify public root assets or generated tree JSON format.")
    parser.add_argument("--root-assets", default="", help="Path to public topic root assets.")
    parser.add_argument("--profiles", default="", help="Optional raw user profile JSONL to validate.")
    parser.add_argument("--normalized-profiles", default="", help="Optional normalized profile JSONL to validate.")
    parser.add_argument("--tree", default="", help="Optional tree.json to validate.")
    parser.add_argument("--tree-manifest", default="", help="Validate one fixed root profile per user across topics.")
    parser.add_argument("--strict-intent", action="store_true", help="Enforce the intent_v2 semantic contract.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    errors: list[str] = []
    if args.root_assets:
        errors.extend(verify_root_assets(Path(args.root_assets)))
    if args.profiles:
        errors.extend(verify_profiles(Path(args.profiles)))
    if args.normalized_profiles:
        errors.extend(verify_normalized_profiles(Path(args.normalized_profiles)))
    if args.tree:
        errors.extend(verify_tree(Path(args.tree), strict_intent=args.strict_intent))
    if args.tree_manifest:
        errors.extend(verify_tree_manifest_profile_consistency(Path(args.tree_manifest)))
    if not any((args.root_assets, args.profiles, args.normalized_profiles, args.tree, args.tree_manifest)):
        raise SystemExit("Provide --root-assets, --profiles, --normalized-profiles, --tree, and/or --tree-manifest.")

    if errors:
        print("Format verification failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("Format verification passed.")


if __name__ == "__main__":
    main()
