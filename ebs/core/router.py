"""Rule-based routing logic for EBS experience selection.

This module implements the default EBS router. The implementation remains
lightweight and deterministic: no LLM calls, no judge calls, no learned
classifier, and no benchmark labels at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ebs.core.experience_bank import (
    CATEGORY_KEYS,
    RoutingDecision,
    RoutingDiagnostics,
    _extract_routing_evidence,
    _normalize_problem,
)

_JAILBREAK_WRAPPER_HINTS = (
    "ignore previous",
    "ignore all previous",
    "ignore your instructions",
    "do not say",
    "don't say",
    "without refusing",
    "without refusal",
    "continue from this refusal",
    "continue after saying",
    "pretend you are not an ai",
    "pretend you are",
    "role-play",
    "roleplay",
    "hypothetical scenario",
    "fictional scenario",
    "for educational purposes",
    "for research purposes",
    "purely hypothetical",
    "reset your parameters",
    "delete any bot safety measures",
    "safety measures",
    "jailbreak",
)

_HARMFUL_TARGET_HINTS = (
    "a person",
    "someone",
    "people",
    "target",
    "victim",
    "password",
    "account",
    "database",
    "system",
    "server",
    "device",
    "school",
    "airport",
    "government",
    "cdc",
    "police",
    "law enforcement",
    "my ex",
    "my boss",
)

_MALICIOUS_GOAL_HINTS = (
    "without permission",
    "without getting caught",
    "avoid detection",
    "evade detection",
    "bypass",
    "stealth",
    "cover my tracks",
    "not be traced",
    "not be recovered",
    "restore access",
    "gain access",
    "delete the evidence",
    "destroy evidence",
    "forge",
    "counterfeit",
    "poison",
    "exploit",
    "obfuscate",
    "backdoor",
    "privilege escalation",
)

_EVASION_OR_DECEPTION_HINTS = (
    "evade",
    "avoid detection",
    "without getting caught",
    "without permission",
    "cover my tracks",
    "delete the evidence",
    "destroy evidence",
    "forg",
    "counterfeit",
    "mislead",
    "lie about",
    "falsify",
    "hide the truth",
    "blackmail",
    "extort",
    "phishing",
    "social engineering",
)

_DEFENSIVE_CONTEXT_HINTS = (
    "authorized",
    "authorised",
    "with permission",
    "legally",
    "legal",
    "defensive",
    "troubleshooting",
    "debugging",
    "diagnostic",
    "for my study",
    "for class",
    "for a report",
    "for research",
    "for my thesis",
    "for my paper",
    "for a novel",
    "for my novel",
    "for a story",
    "for my story",
    "for a movie",
    "for my movie",
    "for a screenplay",
    "historical",
    "explain",
    "analysis",
    "analyze",
    "analytical",
    "research",
    "what is",
    "what happened",
    "how did",
    "why did",
)

_ETHICS_NORMATIVE_HINTS = (
    "should i",
    "should we",
    "is it ethical",
    "is it moral",
    "is it wrong",
    "is it right",
    "morally acceptable",
    "morally justified",
    "ethically acceptable",
    "fair",
    "unfair",
    "justice",
    "legality",
    "compliance",
    "transparent",
    "transparency",
    "consent",
    "deserve",
    "allowed to",
    "ought to",
    "responsibility",
    "obligation",
    "dilemma",
    "value conflict",
)

_ETHICS_SOCIAL_HARM_HINTS = (
    "discriminat",
    "bias",
    "prejudice",
    "racis",
    "sexis",
    "homophob",
    "transphob",
    "immigrants",
    "women",
    "muslims",
    "gay people",
    "black people",
    "white people",
    "disabled people",
    "jewish people",
    "rights should be taken away",
    "fewer rights",
    "less rights",
)

_ETHICS_ZH_HINTS = (
    "浼︾悊",
    "閬撳痉",
    "鍏钩",
    "姝ｄ箟",
    "鏉冨埄",
    "姝ц",
    "鍋忚",
    "浠囨仺",
    "搴斾笉搴旇",
    "鏄惁搴旇",
    "鍊间笉鍊煎緱",
    "涓ら毦",
    "璐ｄ换",
    "涔夊姟",
)

_BENIGN_EXPLANATORY_HINTS = (
    "python process",
    "sql command",
    "gardening",
    "kill weeds",
    "shoot a photo",
    "movie",
    "novel",
    "book i'm writing",
    "screenplay",
    "fictional character",
    "history of",
    "explain the concept",
    "compare",
    "analyze",
    "summarize",
)


@dataclass(slots=True, frozen=True)
class RuleRouterConfig:
    """Rule-based router configuration."""

    strong_gap: float = 2.0
    certainty_epsilon: float = 1e-8
    tie_margin: float = 1e-8


def _matched_phrases(text: str, phrases: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(phrase for phrase in phrases if phrase in text)


def _safe_clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def _normalize_scores(scores: dict[str, float], epsilon: float) -> dict[str, float]:
    total = sum(max(0.0, value) for value in scores.values())
    if total <= 0:
        return dict.fromkeys(CATEGORY_KEYS, 0.0)
    return {category: max(0.0, scores[category]) / (total + epsilon) for category in CATEGORY_KEYS}


def _argmax_bucket(scores: dict[str, float], priority: tuple[str, ...]) -> str:
    ranked = sorted(scores.items(), key=lambda item: (-item[1], priority.index(item[0])))
    return ranked[0][0]


def _ranked_categories(scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: (-item[1], ("harmful", "ethics", "benign").index(item[0])))


def route_experience_buckets_rule_with_trace(
    problem: str,
    *,
    config: RuleRouterConfig | None = None,
) -> tuple[RoutingDecision, dict[str, Any]]:
    """Return the rule-based routing decision plus interpretable trace metadata."""

    cfg = config or RuleRouterConfig()
    normalized = _normalize_problem(problem)
    if not normalized:
        decision = RoutingDecision(
            primary_bucket="harmful",
            secondary_bucket="harmful",
            confidence=0.0,
            scores=dict.fromkeys(CATEGORY_KEYS, 0.0),
            reason_tags=("empty_query_uncertain",),
            diagnostics=RoutingDiagnostics(
                evidence_conflict=True,
                conflict_type="empty_query",
                tied_buckets=("ethics", "harmful"),
            ),
        )
        return decision, {"matched_evidence": {}, "routing_uncertain": True}

    legacy_evidence = _extract_routing_evidence(normalized)
    wrapper_hits = _matched_phrases(normalized, _JAILBREAK_WRAPPER_HINTS)
    target_hits = _matched_phrases(normalized, _HARMFUL_TARGET_HINTS)
    goal_hits = _matched_phrases(normalized, _MALICIOUS_GOAL_HINTS)
    evasion_hits = _matched_phrases(normalized, _EVASION_OR_DECEPTION_HINTS)
    defensive_hits = _matched_phrases(normalized, _DEFENSIVE_CONTEXT_HINTS)
    ethics_normative_hits = _matched_phrases(normalized, _ETHICS_NORMATIVE_HINTS)
    ethics_social_hits = _matched_phrases(normalized, _ETHICS_SOCIAL_HARM_HINTS)
    ethics_zh_hits = _matched_phrases(normalized, _ETHICS_ZH_HINTS)
    benign_explanatory_hits = _matched_phrases(normalized, _BENIGN_EXPLANATORY_HINTS)

    operational_hits = legacy_evidence.operational_hits
    privacy_hits = legacy_evidence.privacy_hits
    group_hits = legacy_evidence.group_hits
    creative_hits = legacy_evidence.creative_hits
    media_hits = legacy_evidence.media_hits
    fictional_hits = legacy_evidence.fictional_hits
    benign_hits = legacy_evidence.benign_hits + benign_explanatory_hits

    harmful_semantic = bool(legacy_evidence.harmful_hits)
    harmful_operational = bool(operational_hits or wrapper_hits)
    harmful_target = bool(target_hits or privacy_hits or group_hits)
    harmful_goal = bool(goal_hits or evasion_hits or legacy_evidence.hostile_justification_hits)
    privacy_intrusion = bool(privacy_hits)
    evasion_or_deception = bool(evasion_hits)

    ethics_normative = bool(ethics_normative_hits or ethics_zh_hits)
    ethics_social = bool(ethics_social_hits and not harmful_goal)
    gray_area_judgment = bool(
        ethics_normative
        and (
            "legal" in normalized
            or "legally" in normalized
            or "compliance" in normalized
            or "consent" in normalized
            or "transparent" in normalized
            or "transparency" in normalized
            or "responsibility" in normalized
            or "fair" in normalized
            or "unfair" in normalized
            or "obligation" in normalized
            or "dilemma" in normalized
            or "value conflict" in normalized
        )
    )
    benign_context = bool(
        benign_hits
        or creative_hits
        or media_hits
        or fictional_hits
        or legacy_evidence.absurd_hits
        or defensive_hits
        or legacy_evidence.is_expository
    )

    scores = dict.fromkeys(CATEGORY_KEYS, 0.0)
    reason_tags: list[str] = []

    if harmful_semantic:
        scores["harmful"] += 1.0
        reason_tags.append("harmful_semantic")
    if harmful_operational:
        scores["harmful"] += 1.0
        reason_tags.append("harmful_operationality")
    if harmful_target:
        scores["harmful"] += 0.75
        reason_tags.append("harmful_target")
    if harmful_goal:
        scores["harmful"] += 1.25
        reason_tags.append("harmful_goal")
    if privacy_intrusion:
        scores["harmful"] += 1.25
        reason_tags.append("privacy_intrusion")
    if evasion_or_deception:
        scores["harmful"] += 1.25
        reason_tags.append("evasion_or_deception")
    if harmful_semantic and harmful_operational:
        scores["harmful"] += 1.0
        reason_tags.append("harmful_moderate_evidence")
    if (
        (harmful_semantic or privacy_intrusion or evasion_or_deception)
        and harmful_operational
        and (harmful_target or harmful_goal)
    ):
        scores["harmful"] += 1.5
        reason_tags.append("harmful_strong_evidence")

    if gray_area_judgment:
        scores["ethics"] += 1.0
        reason_tags.append("ethics_normative")
    if ethics_social:
        scores["ethics"] += 0.75
        reason_tags.append("ethics_social_harm")
    if gray_area_judgment and ethics_social:
        scores["ethics"] += 1.0
        reason_tags.append("ethics_strong_evidence")

    if benign_context:
        scores["benign"] += 1.0
        reason_tags.append("benign_context")
    if legacy_evidence.is_expository:
        scores["benign"] += 0.75
        reason_tags.append("expository_context")
    if defensive_hits:
        scores["benign"] += 0.75
        reason_tags.append("defensive_or_research_context")
    if benign_hits and not harmful_goal:
        scores["benign"] += 0.5
        reason_tags.append("explicit_benign_disambiguation")

    conflict_type: str | None = None
    if (harmful_semantic or privacy_intrusion or evasion_or_deception) and benign_context and not harmful_goal:
        conflict_type = "harmful_vs_benign_context"
        reason_tags.append("evidence_conflict")
    if harmful_operational and defensive_hits:
        conflict_type = conflict_type or "operational_vs_defensive"
        reason_tags.append("defensive_conflict")
    if (
        (harmful_semantic or privacy_intrusion or evasion_or_deception)
        and (creative_hits or media_hits or fictional_hits)
        and not harmful_goal
    ):
        conflict_type = conflict_type or "harmful_vs_fictional"
        reason_tags.append("fictional_conflict")
    if gray_area_judgment and (harmful_semantic or privacy_intrusion or evasion_or_deception):
        conflict_type = conflict_type or "ethics_vs_harmful"
        reason_tags.append("ethical_harm_conflict")

    if (
        (harmful_semantic or privacy_intrusion or evasion_or_deception)
        and harmful_operational
        and (harmful_target or harmful_goal)
    ):
        scores["harmful"] = max(scores["harmful"], scores["ethics"] + cfg.strong_gap)
        reason_tags.append("strong_harmful_override")
    elif gray_area_judgment and not harmful_goal and not (harmful_semantic and harmful_operational and harmful_target):
        scores["ethics"] = max(scores["ethics"], scores["benign"] + 0.1)
        reason_tags.append("ethics_override")

    zero_score = all(value <= 0.0 for value in scores.values())
    max_score = max(scores.values())
    tie_candidates = tuple(
        category for category in CATEGORY_KEYS if abs(scores[category] - max_score) <= cfg.tie_margin
    )
    tied_buckets = tie_candidates[1:] if max_score > 0 and len(tie_candidates) > 1 else ()

    if zero_score:
        primary_bucket = "ethics" if gray_area_judgment else "benign"
    else:
        primary_bucket = _ranked_categories(scores)[0][0]
    ranked = _ranked_categories(scores)
    top_bucket, top_score = ranked[0]
    second_bucket, second_score = ranked[1]

    normalized_scores = _normalize_scores(scores, cfg.certainty_epsilon)
    s1 = normalized_scores[top_bucket]
    s2 = normalized_scores[second_bucket]
    if s1 + s2 <= 0:
        confidence = 0.0
    else:
        confidence = _safe_clip((s1 - s2) / (s1 + s2 + cfg.certainty_epsilon))

    routing_uncertain = zero_score or bool(tied_buckets) or conflict_type is not None
    if routing_uncertain:
        confidence = 0.0

    secondary_bucket = second_bucket
    if second_score <= 0 and not routing_uncertain:
        secondary_bucket = None
    if not routing_uncertain and (top_score - second_score) > cfg.strong_gap:
        secondary_bucket = None
    if conflict_type is not None and second_score <= 0:
        if primary_bucket != "benign":
            secondary_bucket = "benign"
        elif gray_area_judgment:
            secondary_bucket = "ethics"
        else:
            secondary_bucket = "harmful"

    decision = RoutingDecision(
        primary_bucket=primary_bucket,
        secondary_bucket=secondary_bucket,
        confidence=confidence,
        scores={category: float(value) for category, value in scores.items()},
        reason_tags=tuple(dict.fromkeys(reason_tags)),
        diagnostics=RoutingDiagnostics(
            evidence_conflict=conflict_type is not None,
            conflict_type=conflict_type,
            tied_buckets=tied_buckets,
        ),
    )
    trace = {
        "matched_evidence": {
            "legacy_operational_hits": operational_hits,
            "legacy_harmful_hits": legacy_evidence.harmful_hits,
            "legacy_privacy_hits": privacy_hits,
            "legacy_rights_hits": legacy_evidence.rights_hits,
            "legacy_group_hits": group_hits,
            "legacy_creative_hits": creative_hits,
            "legacy_media_hits": media_hits,
            "legacy_fictional_hits": fictional_hits,
            "legacy_benign_hits": legacy_evidence.benign_hits,
            "wrapper_hits": wrapper_hits,
            "target_hits": target_hits,
            "goal_hits": goal_hits,
            "evasion_hits": evasion_hits,
            "defensive_hits": defensive_hits,
            "ethics_normative_hits": ethics_normative_hits,
            "ethics_social_hits": ethics_social_hits,
            "ethics_zh_hits": ethics_zh_hits,
            "benign_explanatory_hits": benign_explanatory_hits,
        },
        "routing_uncertain": routing_uncertain,
        "zero_score": zero_score,
        "tie_buckets": tie_candidates if len(tie_candidates) > 1 else (),
        "normalized_scores": normalized_scores,
        "router_mode": "rule",
    }
    return decision, trace


def route_experience_buckets_rule(
    problem: str,
    *,
    config: RuleRouterConfig | None = None,
) -> RoutingDecision:
    """Return only the rule-based routing decision."""

    return route_experience_buckets_rule_with_trace(problem, config=config)[0]
