from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "behavior_propensity_v1"
PROFILE_DIMS = (
    "goal_progress",
    "mastery_logic",
    "challenge_seeking",
    "social_attachment",
    "cooperative_orientation",
    "world_discovery",
    "role_immersion",
    "aesthetic_customization",
)
INTENT_ORDER = (
    "zoom_in",
    "reveal",
    "branch_out",
    "reframe",
    "follow",
    "interact",
)
GLOBAL_INTENT_PRIOR = {
    "zoom_in": 1.00,
    "reveal": 1.05,
    "branch_out": 0.95,
    "reframe": 0.72,
    "follow": 1.00,
    "interact": 1.08,
}
USER_INTENT_WEIGHTS = {
    "zoom_in": {
        "mastery_logic": 0.42,
        "aesthetic_customization": 0.28,
        "goal_progress": 0.10,
    },
    "reveal": {
        "mastery_logic": 0.36,
        "world_discovery": 0.38,
        "goal_progress": 0.08,
    },
    "branch_out": {
        "world_discovery": 0.48,
        "challenge_seeking": 0.22,
        "role_immersion": 0.12,
    },
    "reframe": {
        "role_immersion": 0.38,
        "aesthetic_customization": 0.42,
        "social_attachment": 0.12,
    },
    "follow": {
        "goal_progress": 0.30,
        "world_discovery": 0.25,
        "challenge_seeking": 0.18,
        "role_immersion": 0.12,
    },
    "interact": {
        "goal_progress": 0.24,
        "cooperative_orientation": 0.28,
        "social_attachment": 0.20,
        "mastery_logic": 0.12,
    },
}
SCORE_WEIGHTS = {
    "profile_signal_affinity": 1.05,
    "intent_profile_affinity": 0.75,
    "current_signal_affinity": 0.45,
    "state_adjustment_factor": 0.35,
    "topic_affinity": 0.70,
    "node_affordance": 0.90,
    "visual_actionability": 0.55,
    "decision_confidence": 0.25,
    "global_intent_prior": 0.20,
    "narrative_novelty": 0.35,
}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def fixed_profile(value: Any, *, field: str) -> dict[str, float]:
    if not isinstance(value, dict) or set(value) != set(PROFILE_DIMS):
        raise ValueError(f"{field} must contain exactly the eight profile dimensions")
    profile = {dimension: float(value[dimension]) for dimension in PROFILE_DIMS}
    if any(not 0.0 <= score <= 1.0 for score in profile.values()):
        raise ValueError(f"{field} values must be in [0, 1]")
    return profile


def signed_signal_affinity(
    profile: dict[str, float],
    signal: Any,
) -> tuple[float, int]:
    if not isinstance(signal, dict):
        return 0.5, 0
    weighted_sum = 0.0
    total_weight = 0.0
    ignored = 0
    for dimension, raw_signal in signal.items():
        if dimension not in PROFILE_DIMS:
            ignored += 1
            continue
        score = float(raw_signal)
        if not math.isfinite(score):
            ignored += 1
            continue
        weight = abs(score)
        if weight <= 1e-12:
            continue
        compatibility = profile[dimension] if score >= 0.0 else 1.0 - profile[dimension]
        weighted_sum += weight * compatibility
        total_weight += weight
    if total_weight <= 1e-12:
        return 0.5, ignored
    return clamp(weighted_sum / total_weight), ignored


def intent_profile_affinity(profile: dict[str, float], intent: str) -> float:
    weights = USER_INTENT_WEIGHTS[intent]
    total = sum(weights.values())
    return clamp(
        sum(profile[dimension] * weight for dimension, weight in weights.items())
        / total
    )


def state_adjustment_factor(source_node: dict[str, Any], intent: str) -> float:
    affect = source_node.get("affective_state", {})
    current = source_node.get("current_state", {})

    def centered(values: Any, key: str) -> float:
        if not isinstance(values, dict):
            return 0.0
        return clamp(float(values.get(key, 0.5))) - 0.5

    factors = {
        "zoom_in": (
            1.0
            + 0.30 * centered(affect, "curiosity")
            + 0.22 * centered(current, "mastery_logic")
            + 0.12 * centered(affect, "cognitive_load")
        ),
        "reveal": (
            1.0
            + 0.38 * centered(affect, "curiosity")
            + 0.20 * centered(current, "world_discovery")
            - 0.12 * centered(affect, "cognitive_load")
        ),
        "branch_out": (
            1.0
            + 0.30 * centered(current, "world_discovery")
            + 0.18 * centered(affect, "arousal")
            - 0.18 * centered(affect, "cognitive_load")
        ),
        "reframe": (
            1.0
            + 0.26 * centered(current, "role_immersion")
            + 0.28 * centered(current, "aesthetic_customization")
            + 0.14 * centered(affect, "empathy")
        ),
        "follow": (
            1.0
            + 0.24 * centered(current, "goal_progress")
            + 0.20 * centered(affect, "arousal")
            + 0.12 * centered(affect, "tension")
        ),
        "interact": (
            1.0
            + 0.22 * centered(current, "cooperative_orientation")
            + 0.20 * centered(affect, "empathy")
            + 0.16 * centered(affect, "dominance")
        ),
    }
    return clamp(factors[intent], 0.65, 1.35)


def target_box_actionability(edge: dict[str, Any]) -> float:
    box = edge.get("grounding", {}).get("target_box")
    if not isinstance(box, list) or len(box) != 4:
        return 0.5
    x1, y1, x2, y2 = (float(value) for value in box)
    if not 0.0 <= x1 < x2 <= 1.0 or not 0.0 <= y1 < y2 <= 1.0:
        return 0.5
    area = (x2 - x1) * (y2 - y1)
    return clamp(math.sqrt(area) / 0.45, 0.15, 1.0)


def top_intent_score(edge: dict[str, Any]) -> float:
    intent = str(edge.get("closed_intent", ""))
    ranking = edge.get("intent_ranking", [])
    if isinstance(ranking, list):
        for item in ranking:
            if isinstance(item, dict) and str(item.get("intent", "")) == intent:
                return clamp(float(item.get("score", 0.5)))
    return 0.5


def visual_actionability(edge: dict[str, Any]) -> float:
    return clamp(
        0.72 * top_intent_score(edge) + 0.28 * target_box_actionability(edge)
    )


def narrative_novelty(history_intents: list[str], intent: str) -> float:
    if history_intents and history_intents[-1] == intent:
        return 0.82
    if intent in history_intents[-3:]:
        return 0.92
    return 1.0


def softmax_probabilities(logits: list[float], *, temperature: float) -> list[float]:
    if not logits:
        raise ValueError("Cannot normalize an empty sibling group")
    if len(logits) == 1:
        return [1.0]
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    maximum = max(logits)
    weights = [math.exp((value - maximum) / temperature) for value in logits]
    total = sum(weights)
    probabilities = [value / total for value in weights]
    rounded = [round(value, 8) for value in probabilities]
    residual = round(1.0 - sum(rounded), 8)
    largest = max(range(len(rounded)), key=rounded.__getitem__)
    rounded[largest] = round(rounded[largest] + residual, 8)
    return rounded


def topic_affinity_catalog(
    release_dir: Path,
    manifest: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in manifest:
        tree = read_json(release_dir / str(row["tree_path"]))
        topic_id = str(tree["topic_id"])
        for edge in tree.get("edges", []):
            metadata = edge.get("private_generation_metadata", {})
            affinity = metadata.get("topic_affinity", {})
            if not isinstance(affinity, dict):
                continue
            for intent in INTENT_ORDER:
                if intent in affinity:
                    values[topic_id][intent].append(clamp(float(affinity[intent])))

    global_values: dict[str, list[float]] = defaultdict(list)
    for by_intent in values.values():
        for intent, scores in by_intent.items():
            global_values[intent].extend(scores)
    global_median = {
        intent: (
            statistics.median(global_values[intent])
            if global_values[intent]
            else 0.75
        )
        for intent in INTENT_ORDER
    }
    topic_ids = {str(row["topic_id"]) for row in manifest}
    catalog = {
        topic_id: {
            intent: round(
                statistics.median(values[topic_id][intent])
                if values[topic_id][intent]
                else global_median[intent],
                6,
            )
            for intent in INTENT_ORDER
        }
        for topic_id in sorted(topic_ids)
    }
    counts = {
        topic_id: {
            intent: len(values[topic_id][intent]) for intent in INTENT_ORDER
        }
        for topic_id in sorted(topic_ids)
    }
    return catalog, counts


def edge_components(
    *,
    edge: dict[str, Any],
    source_node: dict[str, Any],
    fixed_user_profile: dict[str, float],
    topic_affinity: dict[str, float],
    history_intents: list[str],
) -> tuple[dict[str, float], dict[str, Any]]:
    intent = str(edge.get("closed_intent", ""))
    if intent not in INTENT_ORDER:
        raise ValueError(f"Unsupported closed intent: {intent!r}")
    current = {
        dimension: clamp(float(source_node.get("current_state", {}).get(dimension, 0.5)))
        for dimension in PROFILE_DIMS
    }
    profile_signal, ignored_profile_keys = signed_signal_affinity(
        fixed_user_profile,
        edge.get("profile_signal", {}),
    )
    current_signal, ignored_current_keys = signed_signal_affinity(
        current,
        edge.get("profile_signal", {}),
    )
    private = edge.get("private_generation_metadata", {})
    exact_topic = private.get("topic_affinity", {})
    exact_node = private.get("node_affordance", {})
    topic_score = clamp(
        float(exact_topic.get(intent, topic_affinity[intent]))
        if isinstance(exact_topic, dict)
        else topic_affinity[intent]
    )
    node_score = clamp(
        float(exact_node[intent])
        if isinstance(exact_node, dict) and intent in exact_node
        else 0.75
    )
    decision_score = clamp(float(edge.get("decision", {}).get("confidence", 0.5)))
    components = {
        "profile_signal_affinity": profile_signal,
        "intent_profile_affinity": intent_profile_affinity(
            fixed_user_profile, intent
        ),
        "current_signal_affinity": current_signal,
        "state_adjustment_factor": state_adjustment_factor(source_node, intent),
        "topic_affinity": topic_score,
        "node_affordance": node_score,
        "visual_actionability": visual_actionability(edge),
        "decision_confidence": decision_score,
        "global_intent_prior": GLOBAL_INTENT_PRIOR[intent],
        "narrative_novelty": narrative_novelty(history_intents, intent),
    }
    top_score = top_intent_score(edge)
    evidence_strength = clamp(
        0.42 * decision_score
        + 0.28 * top_score
        + 0.15 * (0.90 if exact_topic else 0.72)
        + 0.15 * (0.90 if exact_node else 0.50)
    )
    provenance = {
        "topic_affinity_source": (
            "edge_private_generation_metadata"
            if exact_topic
            else "topic_median_from_natural_sampling_metadata"
        ),
        "node_affordance_source": (
            "edge_private_generation_metadata"
            if exact_node
            else "neutral_missing"
        ),
        "ignored_non_profile_signal_keys": (
            ignored_profile_keys + ignored_current_keys
        ),
        "evidence_strength": round(evidence_strength, 6),
    }
    return components, provenance


def component_logit(components: dict[str, float]) -> float:
    total = 0.0
    for name, weight in SCORE_WEIGHTS.items():
        total += weight * math.log(max(0.05, float(components[name])))
    return total


def score_sibling_distribution(
    *,
    edges: list[dict[str, Any]],
    source_node: dict[str, Any],
    fixed_user_profile: dict[str, float],
    topic_affinity: dict[str, float],
    history_intents: list[str],
    temperature: float,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for edge in edges:
        components, provenance = edge_components(
            edge=edge,
            source_node=source_node,
            fixed_user_profile=fixed_user_profile,
            topic_affinity=topic_affinity,
            history_intents=history_intents,
        )
        scored.append(
            {
                "edge": edge,
                "components": components,
                "provenance": provenance,
                "logit": component_logit(components),
            }
        )
    probabilities = softmax_probabilities(
        [row["logit"] for row in scored],
        temperature=temperature,
    )
    output = []
    for row, probability in zip(scored, probabilities):
        edge = row["edge"]
        output.append(
            {
                "edge_id": str(edge["edge_id"]),
                "target_node": str(edge["target_node"]),
                "closed_intent": str(edge["closed_intent"]),
                "conditional_probability": probability,
                "logit": round(float(row["logit"]), 8),
                "components": {
                    key: round(float(value), 6)
                    for key, value in row["components"].items()
                },
                **row["provenance"],
            }
        )
    return output


def parent_history(tree: dict[str, Any]) -> dict[str, list[str]]:
    incoming: dict[str, dict[str, Any]] = {}
    for edge in tree.get("edges", []):
        target = str(edge["target_node"])
        if target in incoming:
            raise ValueError(f"Tree {tree['tree_id']} has multiple parents for {target}")
        incoming[target] = edge
    histories: dict[str, list[str]] = {}
    for node in tree.get("nodes", []):
        node_id = str(node["node_id"])
        history: list[str] = []
        current = node_id
        seen = {current}
        while current in incoming:
            edge = incoming[current]
            history.append(str(edge["closed_intent"]))
            current = str(edge["source_node"])
            if current in seen:
                raise ValueError(f"Cycle in tree {tree['tree_id']} at {current}")
            seen.add(current)
        histories[node_id] = list(reversed(history))
    return histories


def enumerate_paths(
    tree: dict[str, Any],
    edge_scores: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in tree.get("edges", []):
        outgoing[str(edge["source_node"])].append(edge)
    paths: list[dict[str, Any]] = []

    def visit(
        node_id: str,
        edges_so_far: list[dict[str, Any]],
        log_probability: float,
        evidence_logs: list[float],
        seen: set[str],
    ) -> None:
        children = outgoing.get(node_id, [])
        if not children:
            if not edges_so_far:
                raise ValueError(f"Tree {tree['tree_id']} has no interactions")
            length = len(edges_so_far)
            edge_ids = [str(edge["edge_id"]) for edge in edges_so_far]
            paths.append(
                {
                    "path_id": hashlib.sha256(
                        "/".join(edge_ids).encode("utf-8")
                    ).hexdigest()[:16],
                    "edge_ids": edge_ids,
                    "node_ids": [
                        "n0",
                        *[str(edge["target_node"]) for edge in edges_so_far],
                    ],
                    "intents": [
                        str(edge["closed_intent"]) for edge in edges_so_far
                    ],
                    "length": length,
                    "log_path_probability": round(log_probability, 10),
                    "path_probability": math.exp(log_probability),
                    "normalized_path_score": math.exp(
                        log_probability / length
                    ),
                    "path_evidence_strength": math.exp(
                        sum(evidence_logs) / length
                    ),
                }
            )
            return
        for edge in children:
            target = str(edge["target_node"])
            if target in seen:
                raise ValueError(f"Cycle in tree {tree['tree_id']} at {target}")
            score = edge_scores[str(edge["edge_id"])]
            probability = float(score["conditional_probability"])
            evidence = float(score["evidence_strength"])
            visit(
                target,
                [*edges_so_far, edge],
                log_probability + math.log(max(1e-12, probability)),
                [*evidence_logs, math.log(max(1e-12, evidence))],
                {*seen, target},
            )

    visit("n0", [], 0.0, [], {"n0"})
    total = sum(row["path_probability"] for row in paths)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Tree {tree['tree_id']} leaf path probabilities sum to {total}"
        )
    raw_order = sorted(
        range(len(paths)),
        key=lambda index: (-paths[index]["path_probability"], paths[index]["path_id"]),
    )
    normalized_order = sorted(
        range(len(paths)),
        key=lambda index: (
            -paths[index]["normalized_path_score"],
            paths[index]["path_id"],
        ),
    )
    raw_rank = {index: rank for rank, index in enumerate(raw_order, 1)}
    normalized_rank = {
        index: rank for rank, index in enumerate(normalized_order, 1)
    }
    count = len(paths)
    for index, row in enumerate(paths):
        rank = normalized_rank[index]
        if count == 1:
            band = "only"
        else:
            percentile = (rank - 1) / (count - 1)
            band = (
                "high"
                if percentile <= 0.25
                else "low"
                if percentile >= 0.75
                else "medium"
            )
        row["path_probability"] = round(row["path_probability"], 10)
        row["normalized_path_score"] = round(row["normalized_path_score"], 8)
        row["path_evidence_strength"] = round(row["path_evidence_strength"], 6)
        row["path_probability_rank"] = raw_rank[index]
        row["normalized_score_rank"] = rank
        row["likelihood_band"] = band
        row["is_map_path"] = raw_rank[index] == 1
    return sorted(paths, key=lambda row: row["path_probability_rank"])


def normalized_entropy(probabilities: list[float]) -> float:
    if len(probabilities) <= 1:
        return 0.0
    entropy = -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )
    return entropy / math.log(len(probabilities))


def build(args: argparse.Namespace) -> None:
    release_dir = Path(args.release).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not (release_dir / "tree_manifest.jsonl").is_file():
        raise FileNotFoundError(f"Missing tree manifest in {release_dir}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_jsonl(release_dir / "tree_manifest.jsonl")
    profiles = {
        str(row["user_id"]): fixed_profile(
            row["stable_profile"],
            field=f"canonical profile {row['user_id']}",
        )
        for row in read_jsonl(release_dir / "canonical_user_profiles.jsonl")
    }
    topic_catalog, topic_counts = topic_affinity_catalog(release_dir, manifest)
    node_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    group_sizes: Counter[int] = Counter()
    bands: Counter[str] = Counter()
    evidence_sources: Counter[str] = Counter()
    ignored_signal_keys = 0
    entropies: list[float] = []
    multi_entropies: list[float] = []
    multi_max_probabilities: list[float] = []

    for manifest_row in manifest:
        tree_path = release_dir / str(manifest_row["tree_path"])
        tree = read_json(tree_path)
        user_id = str(tree["user_id"])
        topic_id = str(tree["topic_id"])
        if user_id not in profiles:
            raise ValueError(f"Missing canonical profile for user {user_id}")
        nodes = {str(node["node_id"]): node for node in tree.get("nodes", [])}
        histories = parent_history(tree)
        outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for edge in tree.get("edges", []):
            outgoing[str(edge["source_node"])].append(edge)
        all_edge_scores: dict[str, dict[str, Any]] = {}

        for source_id, edges in outgoing.items():
            if source_id not in nodes:
                raise ValueError(f"Missing source node {source_id} in {tree['tree_id']}")
            scored = score_sibling_distribution(
                edges=edges,
                source_node=nodes[source_id],
                fixed_user_profile=profiles[user_id],
                topic_affinity=topic_catalog[topic_id],
                history_intents=histories[source_id],
                temperature=args.temperature,
            )
            probability_sum = sum(
                float(row["conditional_probability"]) for row in scored
            )
            if abs(probability_sum - 1.0) > 1e-9:
                raise ValueError(
                    f"Probabilities do not sum to one at {tree['tree_id']}:{source_id}"
                )
            if {row["edge_id"] for row in scored} != {
                str(edge["edge_id"]) for edge in edges
            }:
                raise ValueError(f"Sibling edge mismatch at {tree['tree_id']}:{source_id}")
            for row in scored:
                all_edge_scores[row["edge_id"]] = row
                ignored_signal_keys += int(row["ignored_non_profile_signal_keys"])
                evidence_sources[
                    f"topic:{row['topic_affinity_source']}"
                ] += 1
                evidence_sources[
                    f"node:{row['node_affordance_source']}"
                ] += 1
            probabilities = [
                float(row["conditional_probability"]) for row in scored
            ]
            entropy = normalized_entropy(probabilities)
            entropies.append(entropy)
            if len(edges) > 1:
                multi_entropies.append(entropy)
                multi_max_probabilities.append(max(probabilities))
            group_sizes[len(edges)] += 1
            node_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "record_type": "node_behavior_distribution",
                    "sample_id": f"{tree['tree_id']}:{source_id}",
                    "tree_id": str(tree["tree_id"]),
                    "tree_path": str(manifest_row["tree_path"]),
                    "user_id": user_id,
                    "topic_id": topic_id,
                    "source_node": source_id,
                    "source_depth": int(
                        nodes[source_id].get("depth", len(histories[source_id]))
                    ),
                    "history_intents": histories[source_id],
                    "candidate_count": len(edges),
                    "normalized_entropy": round(entropy, 6),
                    "probability_sum": round(probability_sum, 8),
                    "candidates": scored,
                    "estimator": {
                        "name": "metadata_grounded_profile_narrative_propensity",
                        "version": SCHEMA_VERSION,
                        "temperature": args.temperature,
                        "calibration_status": "synthetic_uncalibrated",
                        "fixed_profile_used_for_label_construction": True,
                        "future_target_state_used": False,
                        "future_target_image_used": False,
                        "label_quota_used": False,
                    },
                }
            )

        tree_paths = enumerate_paths(tree, all_edge_scores)
        for row in tree_paths:
            bands[row["likelihood_band"]] += 1
            path_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "record_type": "root_to_leaf_path_likelihood",
                    "tree_id": str(tree["tree_id"]),
                    "tree_path": str(manifest_row["tree_path"]),
                    "user_id": user_id,
                    "topic_id": topic_id,
                    **row,
                }
            )

    write_jsonl(output_dir / "node_distributions.jsonl", node_rows)
    write_jsonl(output_dir / "path_likelihoods.jsonl", path_rows)
    write_json(
        output_dir / "topic_affinity_catalog.json",
        {
            "schema_version": SCHEMA_VERSION,
            "method": "per-topic median of existing natural-sampling metadata",
            "catalog": topic_catalog,
            "source_edge_counts": topic_counts,
        },
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Post-hoc synthetic conditional interaction propensities. "
            "Original trees and images are unchanged."
        ),
        "source_release": release_dir.name,
        "tree_count": len(manifest),
        "user_count": len({str(row["user_id"]) for row in manifest}),
        "topic_count": len({str(row["topic_id"]) for row in manifest}),
        "node_distribution_count": len(node_rows),
        "edge_probability_count": sum(
            len(row["candidates"]) for row in node_rows
        ),
        "root_to_leaf_path_count": len(path_rows),
        "outgoing_group_sizes": {
            str(size): count for size, count in sorted(group_sizes.items())
        },
        "multi_candidate_node_count": sum(
            count for size, count in group_sizes.items() if size > 1
        ),
        "singleton_node_count": group_sizes[1],
        "mean_normalized_node_entropy": round(statistics.mean(entropies), 6),
        "mean_normalized_multi_candidate_entropy": round(
            statistics.mean(multi_entropies),
            6,
        ),
        "mean_multi_candidate_max_probability": round(
            statistics.mean(multi_max_probabilities),
            6,
        ),
        "multi_candidate_probability_range": [
            round(
                min(
                    float(candidate["conditional_probability"])
                    for row in node_rows
                    if row["candidate_count"] > 1
                    for candidate in row["candidates"]
                ),
                8,
            ),
            round(
                max(
                    float(candidate["conditional_probability"])
                    for row in node_rows
                    if row["candidate_count"] > 1
                    for candidate in row["candidates"]
                ),
                8,
            ),
        ],
        "evidence_source_edge_counts": dict(sorted(evidence_sources.items())),
        "path_likelihood_bands": dict(sorted(bands.items())),
        "ignored_non_profile_signal_key_occurrences": ignored_signal_keys,
        "temperature": args.temperature,
        "score_weights": SCORE_WEIGHTS,
        "calibration_status": "synthetic_uncalibrated",
        "probability_semantics": (
            "P(outgoing edge | fixed user profile, source-side history, "
            "current state, topic and node affordance, visible click target)"
        ),
        "leakage_policy": {
            "keep_sidecar_hidden_from_profile_prediction_model": True,
            "fixed_profile_is_used_only_to_construct_private_labels": True,
            "future_target_state_used": False,
            "future_target_image_used": False,
            "label_balancing_quota_used": False,
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add private node and path behavior propensities to a release."
    )
    parser.add_argument("--release", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--temperature", type=float, default=0.65)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
