from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dataset_pipeline import (
    DatasetPipeline,
    PipelineConfig,
    PERSONALITY_DIMS,
    public_config_snapshot,
    read_profiles,
    write_json,
    write_jsonl,
)
from select_and_normalize_profiles import profile_richness
from select_profile_grounded_augmentations import intent_scores


COHORT_ORDER = ["reframe", "zoom_in", "reveal", "branch_out", "follow"]

PROFILE_DIMS = list(PERSONALITY_DIMS)

# Moderate cohort centers encode a preference, not a deterministic intent label.
SYNTHETIC_COHORT_CENTERS = {
    "reframe": {
        "goal_progress": 0.54, "mastery_logic": 0.49, "challenge_seeking": 0.43,
        "social_attachment": 0.67, "cooperative_orientation": 0.53, "world_discovery": 0.55,
        "role_immersion": 0.77, "aesthetic_customization": 0.79,
    },
    "zoom_in": {
        "goal_progress": 0.61, "mastery_logic": 0.79, "challenge_seeking": 0.46,
        "social_attachment": 0.42, "cooperative_orientation": 0.45, "world_discovery": 0.57,
        "role_immersion": 0.52, "aesthetic_customization": 0.70,
    },
    "reveal": {
        "goal_progress": 0.59, "mastery_logic": 0.76, "challenge_seeking": 0.52,
        "social_attachment": 0.43, "cooperative_orientation": 0.46, "world_discovery": 0.79,
        "role_immersion": 0.57, "aesthetic_customization": 0.57,
    },
    "branch_out": {
        "goal_progress": 0.57, "mastery_logic": 0.47, "challenge_seeking": 0.68,
        "social_attachment": 0.40, "cooperative_orientation": 0.42, "world_discovery": 0.81,
        "role_immersion": 0.64, "aesthetic_customization": 0.53,
    },
    "follow": {
        "goal_progress": 0.76, "mastery_logic": 0.55, "challenge_seeking": 0.64,
        "social_attachment": 0.54, "cooperative_orientation": 0.49, "world_discovery": 0.69,
        "role_immersion": 0.65, "aesthetic_customization": 0.48,
    },
}

COHORT_SUMMARIES = {
    "reframe": (
        "This user is visually attentive and strongly role-immersed. They often want to revisit the same "
        "story moment from another character viewpoint or camera angle, especially when expression, "
        "composition, or emotional subtext could change its meaning. They do not seek viewpoint changes in every scene."
    ),
    "zoom_in": (
        "This user prefers careful inspection of visible clues, mechanisms, symbols, and design details. "
        "A closer view helps them understand the scene while they retain moderate interest in plot and atmosphere."
    ),
    "reveal": (
        "This user is driven by explanation and discovery. They tend to uncover concealed information, "
        "resolve mysteries, and connect clues, but do not treat every visible detail as a hidden secret."
    ),
    "branch_out": (
        "This user enjoys expanding the fictional world through new routes and locations and accepts some risk. "
        "They prefer story-plausible departures over arbitrary changes of setting."
    ),
    "follow": (
        "This user likes purposeful continuation: tracking moving characters, signals, or traces that advance "
        "the current narrative. Exploration and challenge usually serve a coherent goal."
    ),
}

COHORT_SUMMARIES_ZH = {
    "reframe": "该用户重视视觉构图与角色沉浸，常希望从另一人物视点或镜头角度重新观看同一叙事时刻，尤其关注表情、构图和情绪潜台词；但并非每个场景都会要求换视角。",
    "zoom_in": "该用户偏好细看已可见的线索、机关、符号和设计细节，倾向用近距离观察理解场景机制，同时仍保留适度的剧情推进与氛围兴趣。",
    "reveal": "该用户由解释欲和发现欲驱动，倾向揭开隐藏信息、解决谜团并串联线索，但不会把每个可见细节都误作秘密。",
    "branch_out": "该用户喜欢通过新路线和新地点扩展虚构世界，也能接受一定风险；其沉浸倾向使新分支需要符合剧情，而不是随意切换场景。",
    "follow": "该用户偏好有目的地延续行动，追踪移动人物、信号或痕迹来推进当前叙事；探索与挑战通常服务于连贯目标。",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def load_config(path: Path) -> PipelineConfig:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    allowed = set(PipelineConfig.__dataclass_fields__)
    return PipelineConfig(**{key: value for key, value in data.items() if key in allowed})


def entropy(scores: dict[str, float]) -> float:
    values = [max(1e-9, float(scores[label])) for label in COHORT_ORDER]
    total = sum(values)
    return -sum((value / total) * math.log(value / total) for value in values)


def evidence_passes(label: str, profile: dict[str, float]) -> bool:
    checks = {
        "reframe": profile["role_immersion"] >= 0.62 and profile["aesthetic_customization"] >= 0.65,
        "zoom_in": profile["mastery_logic"] >= 0.65 and profile["aesthetic_customization"] >= 0.52,
        "reveal": profile["mastery_logic"] >= 0.60 and profile["world_discovery"] >= 0.60,
        "branch_out": profile["world_discovery"] >= 0.65 and profile["challenge_seeking"] >= 0.52,
        "follow": profile["goal_progress"] >= 0.62
        and max(profile["world_discovery"], profile["challenge_seeking"], profile["role_immersion"]) >= 0.58,
    }
    return bool(checks[label])


DIMENSION_RATIONALE_STEMS = {
    "goal_progress": "Preference for explicit objectives and visible narrative progress",
    "mastery_logic": "Interest in clues, mechanisms, rules, and causal explanations",
    "challenge_seeking": "Willingness to approach risk, conflict, or difficult choices",
    "social_attachment": "Attention to characters, relationships, expressions, and emotional stakes",
    "cooperative_orientation": "Preference for helping, negotiating, protecting, and acting with others",
    "world_discovery": "Interest in new places, hidden regions, routes, and world expansion",
    "role_immersion": "Preference for in-character action, atmosphere, and narrative continuity",
    "aesthetic_customization": "Attention to composition, visual style, design detail, and creative variation",
}

DIMENSION_NAMES_ZH = {
    "goal_progress": "目标推进",
    "mastery_logic": "机制与逻辑掌握",
    "challenge_seeking": "挑战偏好",
    "social_attachment": "社会情感联结",
    "cooperative_orientation": "合作倾向",
    "world_discovery": "世界探索",
    "role_immersion": "角色沉浸",
    "aesthetic_customization": "审美与定制",
}

SYNTHETIC_EVIDENCE_FLOORS = {
    "reframe": {"role_immersion": 0.68, "aesthetic_customization": 0.71},
    "zoom_in": {"mastery_logic": 0.70, "aesthetic_customization": 0.58},
    "reveal": {"mastery_logic": 0.66, "world_discovery": 0.67},
    "branch_out": {"world_discovery": 0.71, "challenge_seeking": 0.58},
    "follow": {"goal_progress": 0.69, "world_discovery": 0.62},
}


def deterministic_jitter(*, user_id: str, cohort: str, dimension: str) -> float:
    digest = hashlib.sha256(f"{user_id}:{cohort}:{dimension}".encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) / 0xFFFFFFFF - 0.5) * 0.05


def value_band(value: float) -> str:
    if value >= 0.8:
        return "very high"
    if value >= 0.6:
        return "high"
    if value >= 0.4:
        return "moderate"
    if value >= 0.2:
        return "low"
    return "very low"


def synthesize_profile_for_cohort(row: dict[str, Any], cohort: str) -> dict[str, Any]:
    if cohort not in SYNTHETIC_COHORT_CENTERS:
        raise ValueError(f"Synthetic fallback is not defined for cohort {cohort!r}")
    user_id = str(row["user_id"])
    original = {dimension: float(row["stable_profile"].get(dimension, 0.5)) for dimension in PROFILE_DIMS}
    center = SYNTHETIC_COHORT_CENTERS[cohort]
    synthetic = {}
    for dimension in PROFILE_DIMS:
        blended = 0.35 * original[dimension] + 0.65 * center[dimension]
        blended += deterministic_jitter(user_id=user_id, cohort=cohort, dimension=dimension)
        synthetic[dimension] = round(max(0.32, min(0.84, blended)), 4)
    for dimension, minimum in SYNTHETIC_EVIDENCE_FLOORS[cohort].items():
        synthetic[dimension] = max(synthetic[dimension], minimum)

    rationales = {
        dimension: (
            f"{DIMENSION_RATIONALE_STEMS[dimension]}; the {synthetic[dimension]:.2f} ({value_band(synthetic[dimension])}) "
            f"value blends the original {original[dimension]:.2f} estimate with a moderate {cohort} cohort center. "
            "It is a controlled synthetic assumption, not an observed preference label."
        )
        for dimension in PROFILE_DIMS
    }
    top_dimensions = sorted(PROFILE_DIMS, key=lambda dimension: synthetic[dimension], reverse=True)[:3]
    top_dimensions_en = ", ".join(
        f"{dimension} ({synthetic[dimension]:.2f})" for dimension in top_dimensions
    )
    top_dimensions_zh = "、".join(
        f"{DIMENSION_NAMES_ZH[dimension]}（{synthetic[dimension]:.2f}）" for dimension in top_dimensions
    )
    result = {
        **row,
        "stable_profile": synthetic,
        "profile_summary": (
            f"{COHORT_SUMMARIES[cohort]} Its strongest modeled motivations are {top_dimensions_en}."
        ),
        "profile_summary_zh": (
            f"{COHORT_SUMMARIES_ZH[cohort]} 模拟画像中最突出的动机为{top_dimensions_zh}。"
        ),
        "dimension_rationales": rationales,
        "intent_propensities": intent_scores(synthetic),
        "profile_source": "synthetic_cohort_variant",
        "synthetic_profile": True,
        "synthetic_evidence_passed": evidence_passes(cohort, synthetic),
        "synthetic_provenance": {
            "schema_version": "controlled_synthetic_profile_v1",
            "source_user_id": user_id,
            "target_cohort": cohort,
            "construction_method": (
                "35% normalized original profile + 65% moderate cohort center + deterministic <=0.025 jitter; "
                "target evidence floors are then applied."
            ),
            "original_stable_profile": original,
            "original_profile_summary": str(row.get("profile_summary", "")),
            "observed_ground_truth": False,
            "evaluation_visibility": "stable_profile_only",
        },
    }
    if not result["synthetic_evidence_passed"]:
        raise ValueError(f"Synthetic profile for user {user_id} does not pass {cohort} evidence checks")
    return result


def apply_synthetic_fallback(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        synthesize_profile_for_cohort(row, str(row["rare_intent_cohort"]))
        if row["rare_intent_cohort"] != "mixed" and not row["natural_profile_evidence_passed"]
        else {**row, "synthetic_profile": False}
        for row in selected
    ]


def select_cohorts(
    *,
    normalized: list[dict[str, Any]],
    target_cohorts: dict[str, int],
) -> list[dict[str, Any]]:
    remaining = {str(row["user_id"]): row for row in normalized}
    selected: list[dict[str, Any]] = []
    for label in COHORT_ORDER:
        requested = int(target_cohorts.get(label, 0))
        ranked = sorted(
            remaining.values(),
            key=lambda row: (
                not evidence_passes(label, row["stable_profile"]),
                -intent_scores(row["stable_profile"])[label],
                -profile_richness({"profile": row.get("raw_profile", "")}),
                str(row["user_id"]),
            ),
        )
        if len(ranked) < requested:
            raise ValueError(f"Insufficient remaining users for cohort {label}")
        for row in ranked[:requested]:
            scores = intent_scores(row["stable_profile"])
            selected.append(
                {
                    **row,
                    "rare_intent_cohort": label,
                    "intent_propensities": scores,
                    "natural_profile_evidence_passed": evidence_passes(label, row["stable_profile"]),
                    "profile_source": "normalized_original_profile",
                }
            )
            remaining.pop(str(row["user_id"]))

    mixed_count = int(target_cohorts.get("mixed", 0))
    mixed_ranked = sorted(
        remaining.values(),
        key=lambda row: (
            -entropy(intent_scores(row["stable_profile"])),
            -profile_richness({"profile": row.get("raw_profile", "")}),
            str(row["user_id"]),
        ),
    )
    if len(mixed_ranked) < mixed_count:
        raise ValueError("Insufficient remaining users for mixed cohort")
    for row in mixed_ranked[:mixed_count]:
        selected.append(
            {
                **row,
                "rare_intent_cohort": "mixed",
                "intent_propensities": intent_scores(row["stable_profile"]),
                "natural_profile_evidence_passed": True,
                "profile_source": "normalized_original_profile",
            }
        )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize an ungenerated user window and select ten rare-intent cohorts."
    )
    parser.add_argument("--profiles", default="../data/user_profiles.jsonl")
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-json", default="")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--simulate-missing",
        action="store_true",
        help="Replace only selected profiles that fail natural cohort evidence with audited synthetic variants.",
    )
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    profiles_path = Path(args.profiles)
    if not profiles_path.is_absolute():
        profiles_path = (base_dir / profiles_path).resolve()
    candidate_path = Path(args.candidate_config)
    if not candidate_path.is_absolute():
        candidate_path = (base_dir / candidate_path).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (base_dir / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_config = json.loads(candidate_path.read_text(encoding="utf-8-sig"))
    candidate_ids = [str(user_id) for user_id in candidate_config["candidate_user_ids"]]
    target_cohorts = {str(key): int(value) for key, value in candidate_config["target_cohorts"].items()}
    raw_by_id = {str(row["user_id"]): row for row in read_profiles(profiles_path)}
    missing = [user_id for user_id in candidate_ids if user_id not in raw_by_id]
    if missing:
        raise ValueError(f"Candidate IDs missing from raw profiles: {missing}")
    candidates = [raw_by_id[user_id] for user_id in candidate_ids]
    write_jsonl(output_dir / "candidate_profiles.jsonl", candidates)

    config = PipelineConfig()
    if args.config_json:
        config_path = Path(args.config_json)
        if not config_path.is_absolute():
            config_path = (base_dir / config_path).resolve()
        config = load_config(config_path)

    normalized_path = output_dir / "normalized_candidates.jsonl"
    normalized_by_id = {str(row["user_id"]): row for row in read_jsonl(normalized_path)}
    if args.normalize:
        if not args.config_json and not args.dry_run:
            raise ValueError("--config-json is required for live normalization")

        def normalize(raw: dict[str, Any]) -> dict[str, Any]:
            pipeline = DatasetPipeline(config=config, output_dir=output_dir, dry_run=args.dry_run)
            return pipeline.normalize_profile(raw)

        pending = [row for row in candidates if str(row["user_id"]) not in normalized_by_id]
        workers = max(1, int(args.workers))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(normalize, row): str(row["user_id"]) for row in pending}
            for future in concurrent.futures.as_completed(futures):
                user_id = futures[future]
                normalized_by_id[user_id] = future.result()
                write_jsonl(
                    normalized_path,
                    [normalized_by_id[key] for key in candidate_ids if key in normalized_by_id],
                )

    complete = all(user_id in normalized_by_id for user_id in candidate_ids)
    selected: list[dict[str, Any]] = []
    if complete:
        selected = select_cohorts(
            normalized=[normalized_by_id[user_id] for user_id in candidate_ids],
            target_cohorts=target_cohorts,
        )
        if args.simulate_missing:
            selected = apply_synthetic_fallback(selected)
        selected_ids = [str(row["user_id"]) for row in selected]
        write_jsonl(output_dir / "selected_profiles_raw.jsonl", [raw_by_id[user_id] for user_id in selected_ids])
        write_jsonl(output_dir / "selected_profiles_normalized.jsonl", selected)
        write_json(
            output_dir / "profile_overrides.json",
            {
                "schema_version": "profile_cohort_overrides_v2",
                "description": (
                    "Original raw profiles normalized into the public 8D ontology. Values are frozen for "
                    "reproducible rare-intent cohort generation; they are not synthetic unless profile_source says so."
                ),
                "overrides": {
                    str(row["user_id"]): {
                        "variant_type": row["profile_source"],
                        "variant_name": (
                            f"synthetic_{row['rare_intent_cohort']}_cohort"
                            if row.get("synthetic_profile")
                            else f"natural_{row['rare_intent_cohort']}_cohort"
                        ),
                        "purpose": f"Profile-grounded {row['rare_intent_cohort']} rare-intent enrichment.",
                        "stable_profile": row["stable_profile"],
                        "profile_summary": row["profile_summary"],
                        "profile_summary_zh": row.get("profile_summary_zh", ""),
                        "dimension_rationales": row.get("dimension_rationales", {}),
                        "rare_intent_cohort": row["rare_intent_cohort"],
                        "synthetic_profile": bool(row.get("synthetic_profile", False)),
                        "synthetic_provenance": row.get("synthetic_provenance"),
                    }
                    for row in selected
                },
            },
        )

    evidence_failures = [
        str(row["user_id"])
        for row in selected
        if row["rare_intent_cohort"] != "mixed" and not row["natural_profile_evidence_passed"]
    ]
    summary = {
        "schema_version": "rare_user_cohort_preparation_v2",
        "profiles": str(profiles_path),
        "candidate_config": str(candidate_path),
        "candidate_count": len(candidates),
        "normalized_count": len(normalized_by_id),
        "normalization_complete": complete,
        "normalization_mode": (
            "dry_run_mock" if args.normalize and args.dry_run
            else "live_api" if args.normalize
            else "resume_only"
        ),
        "external_api_used": bool(args.normalize and not args.dry_run),
        "selected_count": len(selected),
        "selected_user_ids": [str(row["user_id"]) for row in selected],
        "cohort_counts": dict(sorted(Counter(row["rare_intent_cohort"] for row in selected).items())),
        "natural_evidence_failure_user_ids": evidence_failures,
        "simulated_profile_user_ids": [
            str(row["user_id"]) for row in selected if row.get("synthetic_profile")
        ],
        "simulation_requested": bool(args.simulate_missing),
        "simulation_applied": any(row.get("synthetic_profile") for row in selected),
        "simulation_required": bool(evidence_failures) and not args.simulate_missing,
        "models": public_config_snapshot(config) if args.normalize else None,
        "status": "cohorts_selected" if complete else "candidate_pool_only",
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
