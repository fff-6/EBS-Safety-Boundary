"""Helpers for categorized EBS experience banks."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

CATEGORY_KEYS = ("harmful", "benign", "ethics")
_CATEGORY_PREFIX = {"harmful": "H", "benign": "B", "ethics": "E"}

DEFAULT_MIX_CONFIDENCE_THRESHOLD = 0.35
DEFAULT_PRIMARY_MIX_K = 6
DEFAULT_SECONDARY_MIX_K = 2

_ETHICS_HINTS = (
    "ethic",
    "moral",
    "value conflict",
    "fairness",
    "rights",
    "discrimination",
    "should i",
)

_OPERATIONAL_REQUEST_HINTS = (
    "how do i",
    "how can i",
    "best way",
    "what steps",
    "what is the process",
    "where can i get",
    "without getting caught",
    "instructions for",
)

_HARMFUL_INTENT_HINTS = (
    "physical harm",
    "violent attack",
    "kill",
    "murder",
    "assault",
    "weapon",
    "explosive",
    "poison",
    "coercion",
    "steal",
    "fraud",
    "illegal access",
    "unauthorized access",
    "credential theft",
    "financial fraud",
    "illegal drugs",
    "self harm",
    "evade detection",
    "bypass security",
)

_PRIVACY_INTRUSION_HINTS = (
    "password",
    "login credential",
    "authentication code",
    "private records",
    "personal information",
    "personal data",
    "private communication",
    "home location",
    "identity document",
    "account access",
    "track a person",
    "without consent",
)

_RIGHTS_DEPRIVATION_HINTS = (
    "deny rights",
    "deprive rights",
    "discriminate",
    "exclude a group",
    "civil rights",
)

_HOSTILE_JUSTIFICATION_HINTS = (
    "morally justified",
    "justify discrimination",
    "justify exclusion",
    "deserve fewer rights",
    "inferior group",
    "superior group",
)

_GROUP_TARGETING_HINTS = (
    "protected group",
    "minority group",
    "religious group",
    "ethnic group",
    "racial group",
    "social group",
    "demographic group",
    "group of people",
)

_ATROCITY_OR_EXTREMISM_HINTS = (
    "extremist organization",
    "terrorist organization",
    "mass atrocity",
    "genocide",
)

_FICTIONAL_OR_LITERARY_CONTEXT_HINTS = (
    "fictional context",
    "fictional character",
    "fictional scenario",
    "literary context",
    "creative writing",
    "storytelling",
    "narrative analysis",
)

_MEDICAL_OR_EDUCATIONAL_CONTEXT_HINTS = (
    "medical context",
    "clinical context",
    "educational context",
    "academic context",
    "research context",
    "safety training",
    "risk awareness",
)

_TECHNICAL_OR_SYSTEM_CONTEXT_HINTS = (
    # Process / system administration
    "software process",
    "operating system",
    "computer program",
    "system administration",
    "authorized security testing",
    "defensive cybersecurity",
    "kill a process",
    "frozen process",
    "unresponsive process",
    "unresponsive program",
    "stop a process",
    "terminate a process",
    "terminate a program",
    "force quit",
    "task manager",
    "background process",
    "process id",
    "daemon process",
    # Operating systems / platforms
    "linux",
    "ubuntu",
    "debian",
    "centos",
    "fedora",
    "macos",
    "windows server",
    "unix",
    "bash",
    "shell",
    "command line",
    "terminal",
    "systemctl",
    "systemd",
    # Programming / development
    "python script",
    "python program",
    "python code",
    "code example",
    "programming language",
    "debug",
    "compiler error",
    "stack trace",
    "git command",
    "package manager",
    "error message",
    # Infrastructure / DevOps
    "docker",
    "kubernetes",
    "nginx",
    "apache",
    "ssh",
    "firewall",
    "log file",
    "database",
    "api endpoint",
)

# Harmful-intent hint strings that have legitimate meanings in a technical
# computing context.  When the *only* harmful signals belong to this set and
# a technical frame is detected, the router treats the query as benign.
_TECHNICALLY_AMBIGUOUS_HARMFUL_TOKENS: frozenset[str] = frozenset(
    {
        "kill",  # "kill a process", "kill -9"
        "physical harm",  # "physically harm the hardware"
        "evade detection",  # "how malware evades detection" (research)
    }
)

_BENIGN_EXPOSITORY_PREFIXES = (
    "what is",
    "what are",
    "what happened",
    "how did",
    "when was",
    "where is",
    "what was",
    "who is",
    "describe",
    "explain",
)

_TECHNICAL_SIGNAL_WORDS = (
    "linux",
    "ubuntu",
    "debian",
    "centos",
    "fedora",
    "macos",
    "unix",
    "windows",
    "bash",
    "shell",
    "terminal",
    "command",
    "docker",
    "kubernetes",
    "nginx",
    "apache",
    "ssh",
    "firewall",
    "python",
    "javascript",
    "java",
    "golang",
    "rust",
    "compiler",
    "debug",
    "runtime",
    "exception",
    "syntax",
    "git",
    "api",
    "database",
    "sql",
    "json",
    "xml",
    "config",
    "log",
    "server",
    "daemon",
    "kernel",
    "driver",
    "binary",
    "thread",
    "socket",
    "protocol",
    "port",
    "service",
    "package",
    "module",
    "library",
    "framework",
    "systemd",
    "process",
)


def _detect_technical_context(normalized_problem: str) -> bool:
    """Detect whether a request is framed in a technical/system-administration context.

    Returns True when 2+ technical signal words are found as whole words,
    indicating the request is likely a technical query rather than a harmful one.
    """
    words = set(normalized_problem.split())
    hits = sum(1 for word in _TECHNICAL_SIGNAL_WORDS if word in words)
    return hits >= 2


_COMPILED_MULTI_SPACE = re.compile(r"\s+")
DEFAULT_EXPERIENCE_TOP_K = 8
DEFAULT_EXPERIENCE_TOKEN_BUDGET = 0
_RETRIEVER_CACHE: dict[tuple[str, str, str | None, int, bool], ExperienceRetriever] = {}


@dataclass(slots=True, frozen=True)
class RoutingDecision:
    """Structured text-only routing output for experience-bucket selection."""

    primary_bucket: str
    secondary_bucket: str | None
    confidence: float
    scores: dict[str, float]
    reason_tags: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class ExperienceMatch:
    """Structured experience retrieval result."""

    experience_id: str
    text: str
    score: float
    bucket: str


@dataclass(frozen=True)
class ExperienceSelection:
    """Auditable result of routed experience retrieval."""

    routing: RoutingDecision
    selected_bucket: str
    experiences: dict[str, str]
    source_buckets: dict[str, str]
    similarity_scores: dict[str, float]
    mixed_retrieval: bool


def _safe_norm(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-8, a_max=None)
    return vectors / norms


class ExperienceTextEmbedder:
    """Embeds text with sentence-transformers when available, else a hash fallback."""

    def __init__(
        self,
        *,
        backend: str = "hash",
        model_name: str | None = None,
        dimension: int = 256,
        allow_fallback_hash: bool = True,
    ) -> None:
        self.backend = backend
        self.model_name = model_name
        self.dimension = dimension
        self.allow_fallback_hash = allow_fallback_hash
        self._encoder: Any | None = None
        self.resolved_backend: str | None = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        preferred = self.backend
        if preferred in {"auto", "sentence_transformer"}:
            try:
                return self._encode_sentence_transformer(texts)
            except Exception:
                if preferred == "sentence_transformer" and not self.allow_fallback_hash:
                    raise

        if preferred in {"auto", "hash"} or self.allow_fallback_hash:
            return self._encode_hash(texts)

        raise RuntimeError("No embedding backend available for experience retrieval.")

    def _encode_sentence_transformer(self, texts: list[str]) -> np.ndarray:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_name, local_files_only=True)
        embeddings = self._encoder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        self.resolved_backend = "sentence_transformer"
        return np.asarray(embeddings, dtype=np.float32)

    def _encode_hash(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            tokens = re.findall(r"[\w\u4e00-\u9fff]+", (text or "").lower())
            if not tokens:
                continue
            for token in tokens:
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                token_hash = int.from_bytes(digest, "little", signed=False)
                col = token_hash % self.dimension
                sign = 1.0 if (token_hash >> 1) % 2 == 0 else -1.0
                vectors[row_idx, col] += sign
            vectors[row_idx] /= float(len(tokens))
        self.resolved_backend = "hash"
        return _safe_norm(vectors).astype(np.float32)


def _normalize_problem(problem: str) -> str:
    normalized = _COMPILED_MULTI_SPACE.sub(" ", (problem or "").strip().lower())
    return normalized


def _count_experience_tokens(exp_id: str, text: str) -> int:
    from ebs.runtime.utils.token import TokenUtils

    return TokenUtils.count_tokens(f"[{exp_id}]. {text}")


def _has_token_budget(token_budget: int) -> bool:
    return token_budget > 0


class ExperienceRetriever:
    """Bucket-aware experience retriever with cached embeddings."""

    def __init__(
        self,
        experiences: Any,
        *,
        embedding_backend: str = "hash",
        embedding_model: str | None = None,
        embedding_dim: int = 256,
        allow_fallback_hash: bool = True,
    ) -> None:
        self._normalized = normalize_experience_bank(experiences)
        self._embedder = ExperienceTextEmbedder(
            backend=embedding_backend,
            model_name=embedding_model,
            dimension=embedding_dim,
            allow_fallback_hash=allow_fallback_hash,
        )
        self._items_by_bucket: dict[str, list[tuple[str, str]]] = {
            bucket: [(str(exp_id), str(text)) for exp_id, text in bucket_data.items()]
            for bucket, bucket_data in self._normalized.items()
        }
        self._matrix_by_bucket: dict[str, np.ndarray] = {}
        for bucket, items in self._items_by_bucket.items():
            self._matrix_by_bucket[bucket] = (
                self._embedder.encode([text for _, text in items]) if items else np.empty((0, 0), dtype=np.float32)
            )

    @property
    def resolved_backend(self) -> str:
        return self._embedder.resolved_backend or self._embedder.backend

    def select(
        self,
        query: str,
        *,
        bucket: str,
        max_items: int,
        token_budget: int,
        secondary_bucket: str | None = None,
        primary_boost: float = 0.05,
    ) -> dict[str, str]:
        if not query.strip():
            return {}
        query_vec = self._embedder.encode([query])[0]
        min_score = 0.2 if self.resolved_backend == "hash" else 0.1

        matches = self._rank_bucket(query_vec, bucket=bucket, min_score=min_score, score_boost=primary_boost)
        if secondary_bucket is not None:
            matches.extend(
                self._rank_bucket(
                    query_vec,
                    bucket=secondary_bucket,
                    min_score=min_score,
                    score_boost=0.0,
                )
            )
        ranked = sorted(matches, key=lambda item: (-item.score, item.experience_id))
        selected: dict[str, str] = {}
        tokens_used = 0
        for match in ranked:
            if len(selected) >= max_items:
                break
            item_tokens = _count_experience_tokens(match.experience_id, match.text)
            if _has_token_budget(token_budget) and tokens_used > 0 and tokens_used + item_tokens > token_budget:
                continue
            if _has_token_budget(token_budget) and not selected and item_tokens > token_budget:
                selected[match.experience_id] = match.text
                tokens_used = item_tokens
                break
            selected[match.experience_id] = match.text
            tokens_used += item_tokens
        if len(selected) < max_items:
            tokens_used = self._fill_from_bucket(
                selected=selected,
                bucket=bucket,
                max_items=max_items,
                token_budget=token_budget,
                tokens_used=tokens_used,
            )
        if secondary_bucket is not None and len(selected) < max_items:
            self._fill_from_bucket(
                selected=selected,
                bucket=secondary_bucket,
                max_items=max_items,
                token_budget=token_budget,
                tokens_used=tokens_used,
            )
        return selected

    def select_bucket_with_scores(
        self,
        query: str,
        *,
        bucket: str,
        max_items: int,
        token_budget: int,
        tokens_used: int = 0,
    ) -> tuple[dict[str, str], dict[str, float], int]:
        """Retrieve a fixed quota from one bucket, preserving ranked scores."""

        if not query.strip() or max_items <= 0:
            return {}, {}, tokens_used
        query_vec = self._embedder.encode([query])[0]
        min_score = 0.2 if self.resolved_backend == "hash" else 0.1
        ranked = self._rank_bucket(query_vec, bucket=bucket, min_score=min_score, score_boost=0.0)
        ranked_ids = {match.experience_id for match in ranked}
        for exp_id, text in self._items_by_bucket.get(bucket, []):
            if exp_id not in ranked_ids:
                ranked.append(ExperienceMatch(exp_id, text, 0.0, bucket))

        selected: dict[str, str] = {}
        scores: dict[str, float] = {}
        for match in ranked:
            if len(selected) >= max_items:
                break
            item_tokens = _count_experience_tokens(match.experience_id, match.text)
            if _has_token_budget(token_budget) and tokens_used > 0 and tokens_used + item_tokens > token_budget:
                continue
            if _has_token_budget(token_budget) and not selected and tokens_used == 0 and item_tokens > token_budget:
                continue
            selected[match.experience_id] = match.text
            scores[match.experience_id] = match.score
            tokens_used += item_tokens
        return selected, scores, tokens_used

    def _fill_from_bucket(
        self,
        *,
        selected: dict[str, str],
        bucket: str,
        max_items: int,
        token_budget: int,
        tokens_used: int,
    ) -> int:
        for exp_id, text in self._items_by_bucket.get(bucket, []):
            if len(selected) >= max_items:
                break
            if exp_id in selected:
                continue
            item_tokens = _count_experience_tokens(exp_id, text)
            if _has_token_budget(token_budget) and tokens_used > 0 and tokens_used + item_tokens > token_budget:
                continue
            selected[exp_id] = text
            tokens_used += item_tokens
        return tokens_used

    def _rank_bucket(
        self,
        query_vec: np.ndarray,
        *,
        bucket: str,
        min_score: float,
        score_boost: float,
    ) -> list[ExperienceMatch]:
        items = self._items_by_bucket.get(bucket, [])
        matrix = self._matrix_by_bucket.get(bucket)
        if not items or matrix is None or matrix.size == 0:
            return []
        scores = matrix @ query_vec
        ranked = np.argsort(-scores)
        matches: list[ExperienceMatch] = []
        for idx in ranked:
            score = float(scores[int(idx)])
            if score < min_score:
                continue
            exp_id, text = items[int(idx)]
            matches.append(
                ExperienceMatch(
                    experience_id=exp_id,
                    text=text,
                    score=score + score_boost,
                    bucket=bucket,
                )
            )
        return matches


def _get_cached_retriever(
    experiences: Any,
    *,
    embedding_backend: str,
    embedding_model: str | None,
    embedding_dim: int,
    allow_fallback_hash: bool,
) -> ExperienceRetriever:
    normalized = normalize_experience_bank(experiences)
    cache_key = (
        json.dumps(normalized, ensure_ascii=False, sort_keys=True),
        embedding_backend,
        embedding_model,
        embedding_dim,
        allow_fallback_hash,
    )
    retriever = _RETRIEVER_CACHE.get(cache_key)
    if retriever is None:
        retriever = ExperienceRetriever(
            normalized,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            allow_fallback_hash=allow_fallback_hash,
        )
        _RETRIEVER_CACHE[cache_key] = retriever
    return retriever


def _matched_phrases(problem: str, phrases: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(phrase for phrase in phrases if phrase in problem)


def _starts_with_any(problem: str, prefixes: tuple[str, ...]) -> bool:
    return any(problem.startswith(prefix) for prefix in prefixes)


def _confidence_from_scores(top_score: float, second_score: float) -> float:
    if top_score <= 0:
        return 0.0
    return max(0.0, min(1.0, (top_score - second_score) / (top_score + 1.0)))


def route_experience_buckets(problem: str) -> RoutingDecision:
    """Route a request into experience buckets using text-only intent heuristics."""

    normalized = _normalize_problem(problem)
    if not normalized:
        return RoutingDecision(
            primary_bucket="harmful",
            secondary_bucket="ethics",
            confidence=0.0,
            scores=dict.fromkeys(CATEGORY_KEYS, 0.0),
            reason_tags=("empty_query_fallback",),
        )

    scores = dict.fromkeys(CATEGORY_KEYS, 0.0)
    reason_tags: list[str] = []

    operational_hits = _matched_phrases(normalized, _OPERATIONAL_REQUEST_HINTS)
    harmful_hits = _matched_phrases(normalized, _HARMFUL_INTENT_HINTS)
    privacy_hits = _matched_phrases(normalized, _PRIVACY_INTRUSION_HINTS)
    ethics_hits = _matched_phrases(normalized, _ETHICS_HINTS)
    rights_hits = _matched_phrases(normalized, _RIGHTS_DEPRIVATION_HINTS)
    hostile_justification_hits = _matched_phrases(normalized, _HOSTILE_JUSTIFICATION_HINTS)
    group_hits = _matched_phrases(normalized, _GROUP_TARGETING_HINTS)
    atrocity_hits = _matched_phrases(normalized, _ATROCITY_OR_EXTREMISM_HINTS)
    fictional_hits = _matched_phrases(normalized, _FICTIONAL_OR_LITERARY_CONTEXT_HINTS)
    educational_hits = _matched_phrases(normalized, _MEDICAL_OR_EDUCATIONAL_CONTEXT_HINTS)
    technical_hits = _matched_phrases(normalized, _TECHNICAL_OR_SYSTEM_CONTEXT_HINTS)

    is_expository = _starts_with_any(normalized, _BENIGN_EXPOSITORY_PREFIXES)

    _has_strong_harmful = len(harmful_hits) >= 2 or bool(privacy_hits)
    _harmful_signal_is_ambiguous = all(
        h in _TECHNICALLY_AMBIGUOUS_HARMFUL_TOKENS for h in harmful_hits
    )

    if operational_hits:
        if _detect_technical_context(normalized) and not _has_strong_harmful and _harmful_signal_is_ambiguous:
            scores["benign"] += 1.5
            reason_tags.append("operational_technical_context")
        else:
            scores["harmful"] += 1.5
            reason_tags.append("operational_request")
    if harmful_hits:
        scores["harmful"] += min(6.0, 1.25 * len(harmful_hits))
        reason_tags.append("harmful_semantics")
    if privacy_hits:
        scores["harmful"] += 4.0
        reason_tags.append("privacy_intrusion")

    if ethics_hits:
        scores["ethics"] += min(4.0, 1.25 * len(ethics_hits))
        reason_tags.append("ethics_language")
    if rights_hits:
        scores["ethics"] += 4.0
        reason_tags.append("rights_or_discrimination_frame")
    if hostile_justification_hits:
        scores["harmful"] += 2.5
        scores["ethics"] += 1.5
        reason_tags.append("hostile_justification_frame")
    if group_hits and rights_hits:
        scores["ethics"] += 2.0
        scores["harmful"] += 1.5
        reason_tags.append("protected_group_target")
    if atrocity_hits and hostile_justification_hits:
        scores["harmful"] += 3.0
        reason_tags.append("atrocity_or_extremism_frame")

    if fictional_hits:
        scores["benign"] += 2.5
        reason_tags.append("fictional_or_literary_context")
    if educational_hits:
        scores["benign"] += 1.5
        reason_tags.append("medical_or_educational_context")
    if technical_hits:
        scores["benign"] += 3.0
        reason_tags.append("technical_or_system_context")

    if is_expository and not operational_hits and not rights_hits:
        scores["benign"] += 1.5
        reason_tags.append("expository_query")
    if is_expository and harmful_hits and not operational_hits and not privacy_hits:
        scores["benign"] += 1.5
        reason_tags.append("historical_or_definitional_context")

    if rights_hits and not group_hits:
        scores["benign"] += 1.5
        reason_tags.append("non_group_normative_frame")

    if fictional_hits and privacy_hits:
        scores["benign"] = max(scores["benign"], scores["harmful"] + 1.0)
        reason_tags.append("favor_benign_for_fictional_context")
    if technical_hits and harmful_hits and not privacy_hits:
        if _harmful_signal_is_ambiguous:
            scores["benign"] = max(scores["benign"], scores["harmful"] + 0.5)
            reason_tags.append("favor_benign_for_technical_context")
        else:
            reason_tags.append("retain_harmful_despite_technical_context")

    if (harmful_hits or privacy_hits) and (operational_hits or privacy_hits):
        if technical_hits and not privacy_hits and _harmful_signal_is_ambiguous:
            reason_tags.append("skip_harmful_override_due_to_benign_context")
        elif fictional_hits and privacy_hits:
            reason_tags.append("skip_harmful_override_due_to_benign_context")
        else:
            minimum_gap = 0.5 if fictional_hits else 2.5
            scores["harmful"] = max(scores["harmful"], scores["benign"] + minimum_gap)
            reason_tags.append("favor_harmful_for_operational_risk")

    if rights_hits and group_hits and hostile_justification_hits:
        scores["harmful"] = max(scores["harmful"], scores["ethics"] + 0.75)
        reason_tags.append("favor_harmful_for_hostile_group_justification")
    elif rights_hits and group_hits:
        scores["ethics"] = max(scores["ethics"], scores["harmful"] + 0.25)
        reason_tags.append("favor_ethics_for_rights_reasoning")

    if group_hits and hostile_justification_hits:
        scores["harmful"] += 1.5
        reason_tags.append("hostile_targeting")

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    primary_bucket, top_score = ranked[0]
    secondary_bucket, second_score = ranked[1]
    confidence = _confidence_from_scores(top_score, second_score)

    preserve_secondary_for_contextual_risk = (
        primary_bucket == "benign"
        and secondary_bucket == "harmful"
        and bool(harmful_hits or privacy_hits)
        and bool(fictional_hits or educational_hits or technical_hits)
    )
    if second_score <= 0 or ((top_score - second_score) > 2.5 and not preserve_secondary_for_contextual_risk):
        secondary_bucket = None

    return RoutingDecision(
        primary_bucket=primary_bucket,
        secondary_bucket=secondary_bucket,
        confidence=confidence,
        scores={category: float(score) for category, score in scores.items()},
        reason_tags=tuple(dict.fromkeys(reason_tags)),
    )


def is_categorized_experience_bank(data: Any) -> bool:
    """Return whether the payload already uses categorized experience storage."""

    return isinstance(data, Mapping) and any(key in data for key in CATEGORY_KEYS)


def normalize_experience_bank(data: Any) -> dict[str, dict[str, str]]:
    """Normalize flat or categorized experience payloads into categorized form."""

    normalized = {category: {} for category in CATEGORY_KEYS}
    if not data:
        return normalized

    if is_categorized_experience_bank(data):
        for category in CATEGORY_KEYS:
            bucket = data.get(category, {})
            if isinstance(bucket, Mapping):
                normalized[category] = {str(exp_id): str(text) for exp_id, text in bucket.items()}
        return normalized

    if isinstance(data, Mapping):
        flat = {str(exp_id): str(text) for exp_id, text in data.items()}
        for category in CATEGORY_KEYS:
            normalized[category] = dict(flat)
        return normalized

    raise ValueError("Experience payload must be a JSON object.")


def flatten_experience_bank(data: Any) -> dict[str, str]:
    """Flatten flat or categorized experiences into a single id-text mapping."""

    normalized = normalize_experience_bank(data)
    flattened: dict[str, str] = {}
    for category in CATEGORY_KEYS:
        for exp_id, text in normalized[category].items():
            key = exp_id if exp_id not in flattened else f"{category}:{exp_id}"
            flattened[key] = text
    return flattened


def has_any_experiences(data: Any) -> bool:
    """Return whether the bank contains at least one experience."""

    normalized = normalize_experience_bank(data)
    return any(bool(bucket) for bucket in normalized.values())


def infer_bucket_from_problem(problem: str, harmful_label: int | None = None) -> str:
    """Infer the primary experience bucket for a query from text alone.

    The ``harmful_label`` argument is kept only for backward compatibility with
    existing callers. Routing intentionally relies on the request text itself.
    """

    del harmful_label
    return route_experience_buckets(problem).primary_bucket


def get_problem_bucket(sample: Mapping[str, Any]) -> str:
    """Infer a sample bucket from problem text for retrieval-time consistency."""

    return infer_bucket_from_problem(sample.get("problem", ""))


def select_experiences(
    experiences: Any,
    *,
    problem: str,
    harmful_label: int | None = None,
    bucket: str | None = None,
    max_experiences: int = DEFAULT_EXPERIENCE_TOP_K,
    token_budget: int = DEFAULT_EXPERIENCE_TOKEN_BUDGET,
    embedding_backend: str = "hash",
    embedding_model: str | None = None,
    embedding_dim: int = 256,
    allow_fallback_hash: bool = True,
) -> tuple[str, dict[str, str]]:
    """Select the best-matching experiences with bucket routing and Top-K retrieval.

    The ``harmful_label`` argument is accepted for compatibility but is not used
    for routing. When text-only routing is uncertain, a few experiences from the
    secondary bucket are considered during retrieval to reduce hard routing mistakes.
    """

    selection = select_experiences_detailed(
        experiences,
        problem=problem,
        harmful_label=harmful_label,
        bucket=bucket,
        max_experiences=max_experiences,
        token_budget=token_budget,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        allow_fallback_hash=allow_fallback_hash,
    )
    return selection.selected_bucket, selection.experiences


def select_experiences_detailed(
    experiences: Any,
    *,
    problem: str,
    harmful_label: int | None = None,
    bucket: str | None = None,
    routing_decision: RoutingDecision | None = None,
    max_experiences: int = DEFAULT_EXPERIENCE_TOP_K,
    token_budget: int = DEFAULT_EXPERIENCE_TOKEN_BUDGET,
    mix_confidence_threshold: float = DEFAULT_MIX_CONFIDENCE_THRESHOLD,
    primary_mix_k: int = DEFAULT_PRIMARY_MIX_K,
    secondary_mix_k: int = DEFAULT_SECONDARY_MIX_K,
    embedding_backend: str = "hash",
    embedding_model: str | None = None,
    embedding_dim: int = 256,
    allow_fallback_hash: bool = True,
) -> ExperienceSelection:
    """Route once and retrieve with the fixed mixed-retrieval quotas."""

    del harmful_label
    normalized = normalize_experience_bank(experiences)
    decision = routing_decision or route_experience_buckets(problem)
    selected_bucket = str(bucket) if bucket in CATEGORY_KEYS else decision.primary_bucket
    if max_experiences <= 0:
        return ExperienceSelection(decision, selected_bucket, {}, {}, {}, False)

    mixed = (
        bucket not in CATEGORY_KEYS
        and decision.secondary_bucket is not None
        and decision.confidence < mix_confidence_threshold
        and bool(normalized.get(decision.secondary_bucket))
    )
    if mixed and primary_mix_k + secondary_mix_k != max_experiences:
        if max_experiences == DEFAULT_EXPERIENCE_TOP_K:
            raise ValueError("Mixed retrieval requires primary_mix_k + secondary_mix_k == K.")
        secondary_mix_k = min(secondary_mix_k, max(0, (max_experiences - 1) // 2))
        primary_mix_k = max_experiences - secondary_mix_k
        if secondary_mix_k == 0:
            mixed = False
    if mixed and primary_mix_k <= secondary_mix_k:
        raise ValueError("Mixed retrieval requires primary_mix_k > secondary_mix_k.")

    retriever = _get_cached_retriever(
        normalized,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        allow_fallback_hash=allow_fallback_hash,
    )
    primary_quota = primary_mix_k if mixed else max_experiences
    selected, scores, tokens_used = retriever.select_bucket_with_scores(
        problem, bucket=selected_bucket, max_items=primary_quota, token_budget=token_budget
    )
    sources = dict.fromkeys(selected, selected_bucket)
    if mixed and decision.secondary_bucket is not None:
        secondary, secondary_scores, _ = retriever.select_bucket_with_scores(
            problem,
            bucket=decision.secondary_bucket,
            max_items=secondary_mix_k,
            token_budget=token_budget,
            tokens_used=tokens_used,
        )
        selected.update(secondary)
        scores.update(secondary_scores)
        sources.update(dict.fromkeys(secondary, decision.secondary_bucket))
    if selected:
        return ExperienceSelection(decision, selected_bucket, selected, sources, scores, mixed)

    for fallback_bucket in CATEGORY_KEYS:
        if normalized[fallback_bucket]:
            selected, scores, _ = retriever.select_bucket_with_scores(
                problem, bucket=fallback_bucket, max_items=max_experiences, token_budget=token_budget
            )
            if selected:
                return ExperienceSelection(
                    decision,
                    fallback_bucket,
                    selected,
                    dict.fromkeys(selected, fallback_bucket),
                    scores,
                    False,
                )
    return ExperienceSelection(decision, selected_bucket, {}, {}, {}, mixed)


def format_experiences_for_prompt(experiences: Mapping[str, str] | None) -> str:
    """Render experiences into the numbered prompt block."""

    if not experiences:
        return "None"
    return "\n".join(f"[{exp_id}]. {text}" for exp_id, text in experiences.items()) or "None"


def get_next_experience_id(existing: Mapping[str, str], bucket: str) -> str:
    """Create the next category-aware experience id."""

    prefix = _CATEGORY_PREFIX.get(bucket, "G")
    max_index = -1
    for exp_id in existing:
        if str(exp_id).startswith(prefix):
            suffix = str(exp_id)[len(prefix) :]
            if suffix.isdigit():
                max_index = max(max_index, int(suffix))
    return f"{prefix}{max_index + 1}"
