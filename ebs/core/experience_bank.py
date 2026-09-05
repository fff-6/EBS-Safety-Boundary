"""Helpers for categorized EBS experience banks."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from ebs.runtime.utils.token import TokenUtils
except Exception:
    TokenUtils = None

CATEGORY_KEYS = ("harmful", "benign", "ethics")
_CATEGORY_PREFIX = {"harmful": "H", "benign": "B", "ethics": "E"}

_MIX_CONFIDENCE_THRESHOLD = 0.35
_SECONDARY_MIX_MAX_ITEMS = 4

_ETHICS_HINTS = (
    "ethic",
    "moral",
    "morality",
    "value conflict",
    "dilemma",
    "should i",
    "right thing",
    "wrong",
    "fair",
    "unfair",
    "acceptable",
    "justified",
)

_OPERATIONAL_REQUEST_HINTS = (
    "how do i",
    "how can i",
    "how do we",
    "how can we",
    "how do i best",
    "how can i best",
    "best way",
    "ways of",
    "what steps",
    "steps should i take",
    "what can i do to",
    "what do i need to do",
    "what's the process for",
    "where can i buy",
    "where can i get",
    "without getting caught",
    "can you tell me how to",
    "how should i",
)

_HARMFUL_INTENT_HINTS = (
    "kill",
    "murder",
    "execute",
    "shoot",
    "stab",
    "poison",
    "bomb",
    "blow up",
    "attack",
    "strangle",
    "choke",
    "assault",
    "rape",
    "kidnap",
    "hostage",
    "arson",
    "massacre",
    "exterminate",
    "annihilate",
    "eliminate",
    "wipe out",
    "suicide",
    "self harm",
    "break into",
    "rob",
    "steal",
    "fraud",
    "identity theft",
    "launder money",
    "treason",
    "terrorism",
    "human trafficking",
    "cocaine",
    "meth",
    "lsd",
    "crack",
    "ecstasy pills",
    "weed",
    "marihuana",
    "backdoor",
    "privilege escalation",
    "bypass",
    "disable the alarm",
    "vault",
    "punch",
    "take out",
    "terminating people",
    "people i don't like",
)

_PRIVACY_INTRUSION_HINTS = (
    "password",
    "pin",
    "credit card",
    "browsing history",
    "medical records",
    "mental health history",
    "home address",
    "private email",
    "mobile phone number",
    "passport number",
    "driver's license number",
    "date of birth",
    "sexual orientation",
    "criminal history",
    "immigration status",
    "tax records",
    "bank account details",
    "political affiliation",
    "marital status",
    "religious beliefs",
    "instagram password",
    "facebook login",
    "birth certificate",
)

_RIGHTS_DEPRIVATION_HINTS = (
    "equal rights",
    "voting rights",
    "not be allowed",
    "be denied",
    "denied the right",
    "should not have",
    "should not be given",
    "should not be admitted",
    "should be barred",
    "should be kicked off",
    "should be discriminated against",
    "should not hold positions",
    "should not have access to healthcare",
    "overrepresented in government",
    "why should",
    "why is it best",
    "reasons why",
)

_HOSTILE_JUSTIFICATION_HINTS = (
    "morally justified",
    "justified to",
    "deserve fewer rights",
    "deserve less rights",
    "why should",
    "why would it be best",
    "reasons why",
    "overstate what happened",
    "overstate what happened to them",
    "not have equal rights",
    "be denied the right to vote",
    "be barred from",
    "not be given visas",
    "taken away",
)

_PROTECTED_GROUP_HINTS = (
    "women",
    "men",
    "gay people",
    "trans people",
    "muslims",
    "black people",
    "white people",
    "hindus",
    "asian people",
    "roma people",
    "indigenous people",
    "non-binary people",
    "pregnant women",
    "buddhists",
    "lesbians",
    "elderly people",
    "latinos",
    "homeless people",
    "christians",
    "immigrants",
    "disabled people",
    "jewish people",
    "asexual people",
)

_EXTREMIST_OR_ATROCITY_HINTS = (
    "isis",
    "armenian genocide",
    "holocaust",
    "world war 2",
    "massacres",
    "genocide",
)

_CREATIVE_CONTEXT_HINTS = (
    "in a book i'm writing",
    "in a film i am directing",
    "in a film i'm directing",
    "in a movie i'm directing",
    "in a screenplay",
    "in my screenplay",
    "for a novel",
    "for my novel",
    "for a story",
    "for my story",
    "fictional context",
    "fictional character",
)

_BENIGN_DISAMBIGUATION_HINTS = (
    "python process",
    "c program",
    "terminating a contract",
    "execute a risky plan",
    "executing a command in sql",
    "shoot a good photo",
    "shoot a goal",
    "spiciness in a dish",
    "blow up a balloon",
    "murder of crows",
    "smash a ball in tennis",
    "hang a picture",
    "eliminate jobs at my company",
    "liquidating a business",
    "take out my friend for dinner",
    "can of coke",
    "state of ecstasy",
    "crack in my pipe",
    "weed my garden",
    "tackle difficult issues",
    "sever my employment",
    "capture a mood",
    "throttle downloads",
    "strangle options",
    "killing time",
    "kill the lights",
    "crush my competition at a bakesale",
    "bomb the test",
    "hit my targets at work",
    "hunt down an answer",
    "set a party on fire",
    "slay my dance performance",
    "steal someone's heart",
    "stealing the show",
    "explode with laughter",
    "music scene",
    "destroying my opponent in chess",
    "smash my exam",
    "butcher my lines",
    "punch above my weight",
    "axing a bad plan",
    "crack open a beer",
    "drowning in unpaid bills",
    "slash my travel costs",
    "obliterate my self-doubts",
    "shatter my personal record",
    "break into the film industry",
    "snipe the best black friday deals",
    "carve a niche",
    "killing weeds",
    "kill a mosquito",
    "stab a balloon",
    "gut a fish",
    "butchering a pig",
    "euthanize my sick dog",
    "punch a punching bag",
    "garden shed",
    "cooked shrimp",
    "fresh meat for a meal",
    "fish i am cooking",
    "eradicate poverty",
    "dead hamster",
    "smash a pi帽ata",
    "property i own",
    "empty soda can",
    "bed bugs",
    "rats in my attic",
    "eliminate clutter",
    "disable pop-up ads",
    "break open a coconut",
    "drywall in my house",
    "poison vermin",
    "saw a log",
    "hammer in a tack",
    "call of duty",
    "counter strike",
    "gta",
    "minecraft",
    "movie",
    "novel",
    "tv show",
    "short story",
    "shooting range",
    "judo",
    "attack the king in a chess game",
    "boxing",
    "checkers",
    "mission impossible",
    "ocean's 11",
)

_MEDIA_ANALYSIS_HINTS = (
    "in dark knight rises",
    "in die hard",
    "in star wars",
    "in tomb raider",
    "how does",
    "what happens when",
    "describe in the movie",
    "describe in the book",
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
    "what city",
    "what number",
    "what is the address",
    "what is the phone number",
    "what is the email address",
    "what jobs has",
    "what did",
)

_ABSURDITY_HINTS = (
    "speed of light",
    "breathe underwater",
    "live on the sun's surface",
    "teleport",
    "telekinesis",
    "shoot lasers from their eyes",
    "travel to other dimensions",
    "stop time",
    "slow down the passage of time",
    "levitate",
    "resurrect the dead",
    "breathe in space",
    "transform into animals at will",
    "see through walls",
    "control the weather",
    "float through walls",
    "travel to jupiter",
    "travel to other dimensions",
    "live for 200 years",
    "crossing a black hole's event horizon",
)

_FICTIONAL_ENTITY_HINTS = (
    "jason bourne",
    "captain marvel",
    "totoro",
    "shrek",
    "robin hood",
    "peter pan",
    "frodo",
    "spongebob",
    "gandalf",
    "james bond",
    "daenerys",
    "marge simpson",
    "naruto",
    "donald duck",
    "asterix",
    "minnie mouse",
    "princess leia",
    "lara croft",
    "goofy",
    "pikachu",
    "harry potter",
    "hermione granger",
    "garfield",
    "princess mononoke",
)

_COMPILED_MULTI_SPACE = re.compile(r"\s+")
DEFAULT_EXPERIENCE_TOP_K = 8
DEFAULT_EXPERIENCE_TOKEN_BUDGET = 0
_RETRIEVER_CACHE: dict[tuple[str, str, str | None, int, bool], ExperienceRetriever] = {}


@dataclass(slots=True, frozen=True)
class RoutingEvidence:
    """Matched routing cues grouped by heuristic family."""

    operational_hits: tuple[str, ...]
    harmful_hits: tuple[str, ...]
    privacy_hits: tuple[str, ...]
    ethics_hits: tuple[str, ...]
    rights_hits: tuple[str, ...]
    hostile_justification_hits: tuple[str, ...]
    group_hits: tuple[str, ...]
    atrocity_hits: tuple[str, ...]
    creative_hits: tuple[str, ...]
    benign_hits: tuple[str, ...]
    media_hits: tuple[str, ...]
    absurd_hits: tuple[str, ...]
    fictional_hits: tuple[str, ...]
    is_expository: bool


@dataclass(slots=True, frozen=True)
class RoutingDiagnostics:
    """Detailed routing diagnostics for analysis and regression checks."""

    evidence_conflict: bool
    conflict_type: str | None
    tied_buckets: tuple[str, ...]
    used_mixed_retrieval: bool = False
    average_primary_count: float | None = None
    average_secondary_count: float | None = None


@dataclass(slots=True, frozen=True)
class RoutingDecision:
    """Structured text-only routing output for experience-bucket selection."""

    primary_bucket: str
    secondary_bucket: str | None
    confidence: float
    scores: dict[str, float]
    reason_tags: tuple[str, ...]
    diagnostics: RoutingDiagnostics | None = None


@dataclass(slots=True, frozen=True)
class ExperienceMatch:
    """Structured experience retrieval result."""

    experience_id: str
    text: str
    score: float
    bucket: str


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
    content = f"[{exp_id}]. {text}"
    if TokenUtils is not None:
        return TokenUtils.count_tokens(content)
    return len(re.findall(r"[\w\u4e00-\u9fff]+|[^\s]", content))


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
        secondary_min_items: int = 0,
    ) -> dict[str, str]:
        selected, _ = self.select_with_details(
            query,
            bucket=bucket,
            max_items=max_items,
            token_budget=token_budget,
            secondary_bucket=secondary_bucket,
            primary_boost=primary_boost,
            secondary_min_items=secondary_min_items,
        )
        return selected

    def select_with_details(
        self,
        query: str,
        *,
        bucket: str,
        max_items: int,
        token_budget: int,
        secondary_bucket: str | None = None,
        primary_boost: float = 0.05,
        secondary_min_items: int = 0,
    ) -> tuple[dict[str, str], dict[str, float | int | bool]]:
        if not query.strip():
            return {}, {
                "used_mixed_retrieval": False,
                "primary_selected_count": 0,
                "secondary_selected_count": 0,
            }
        query_vec = self._embedder.encode([query])[0]
        min_score = 0.2 if self.resolved_backend == "hash" else 0.1

        matches = self._rank_bucket(query_vec, bucket=bucket, min_score=min_score, score_boost=primary_boost)
        primary_ids = {match.experience_id for match in matches}
        secondary_matches: list[ExperienceMatch] = []
        if secondary_bucket is not None:
            secondary_matches = self._rank_bucket(
                query_vec,
                bucket=secondary_bucket,
                min_score=min_score,
                score_boost=0.0,
            )
            matches.extend(secondary_matches)
        ranked = sorted(matches, key=lambda item: (-item.score, item.experience_id))
        selected: dict[str, str] = {}
        tokens_used = 0
        secondary_selected_count = 0
        if secondary_bucket is not None and secondary_min_items > 0:
            for match in sorted(secondary_matches, key=lambda item: (-item.score, item.experience_id)):
                if secondary_selected_count >= min(secondary_min_items, max_items):
                    break
                item_tokens = _count_experience_tokens(match.experience_id, match.text)
                if _has_token_budget(token_budget) and tokens_used > 0 and tokens_used + item_tokens > token_budget:
                    continue
                if match.experience_id in selected:
                    continue
                selected[match.experience_id] = match.text
                secondary_selected_count += 1
                tokens_used += item_tokens
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
            tokens_used = self._fill_from_bucket(
                selected=selected,
                bucket=secondary_bucket,
                max_items=max_items,
                token_budget=token_budget,
                tokens_used=tokens_used,
            )
        primary_selected_count = sum(1 for exp_id in selected if exp_id in primary_ids)
        secondary_selected_count = len(selected) - primary_selected_count
        return selected, {
            "used_mixed_retrieval": secondary_bucket is not None,
            "primary_selected_count": primary_selected_count,
            "secondary_selected_count": secondary_selected_count,
        }

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


def _legacy_confidence_from_scores(top_score: float, second_score: float) -> float:
    if top_score <= 0:
        return 0.0
    return max(0.0, min(1.0, (top_score - second_score) / (top_score + 1.0)))


def _extract_routing_evidence(normalized: str) -> RoutingEvidence:
    return RoutingEvidence(
        operational_hits=_matched_phrases(normalized, _OPERATIONAL_REQUEST_HINTS),
        harmful_hits=_matched_phrases(normalized, _HARMFUL_INTENT_HINTS),
        privacy_hits=_matched_phrases(normalized, _PRIVACY_INTRUSION_HINTS),
        ethics_hits=_matched_phrases(normalized, _ETHICS_HINTS),
        rights_hits=_matched_phrases(normalized, _RIGHTS_DEPRIVATION_HINTS),
        hostile_justification_hits=_matched_phrases(normalized, _HOSTILE_JUSTIFICATION_HINTS),
        group_hits=_matched_phrases(normalized, _PROTECTED_GROUP_HINTS),
        atrocity_hits=_matched_phrases(normalized, _EXTREMIST_OR_ATROCITY_HINTS),
        creative_hits=_matched_phrases(normalized, _CREATIVE_CONTEXT_HINTS),
        benign_hits=_matched_phrases(normalized, _BENIGN_DISAMBIGUATION_HINTS),
        media_hits=_matched_phrases(normalized, _MEDIA_ANALYSIS_HINTS),
        absurd_hits=_matched_phrases(normalized, _ABSURDITY_HINTS),
        fictional_hits=_matched_phrases(normalized, _FICTIONAL_ENTITY_HINTS),
        is_expository=_starts_with_any(normalized, _BENIGN_EXPOSITORY_PREFIXES),
    )


def _legacy_scores_from_evidence(normalized: str, evidence: RoutingEvidence) -> tuple[dict[str, float], list[str]]:
    scores = dict.fromkeys(CATEGORY_KEYS, 0.0)
    reason_tags: list[str] = []
    operational_hits = evidence.operational_hits
    harmful_hits = evidence.harmful_hits
    privacy_hits = evidence.privacy_hits
    ethics_hits = evidence.ethics_hits
    rights_hits = evidence.rights_hits
    hostile_justification_hits = evidence.hostile_justification_hits
    group_hits = evidence.group_hits
    atrocity_hits = evidence.atrocity_hits
    creative_hits = evidence.creative_hits
    benign_hits = evidence.benign_hits
    media_hits = evidence.media_hits
    absurd_hits = evidence.absurd_hits
    fictional_hits = evidence.fictional_hits
    is_expository = evidence.is_expository

    if operational_hits:
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

    if benign_hits:
        scores["benign"] += min(6.0, 1.25 * len(benign_hits))
        scores["harmful"] = max(0.0, scores["harmful"] - 1.0)
        reason_tags.append("benign_disambiguation")
    if creative_hits:
        scores["benign"] += 3.0
        scores["harmful"] = max(0.0, scores["harmful"] - 0.75)
        reason_tags.append("creative_context")
    if media_hits:
        scores["benign"] += 2.5
        scores["harmful"] = max(0.0, scores["harmful"] - 0.75)
        reason_tags.append("media_analysis_context")
    if absurd_hits:
        scores["benign"] += 3.0
        reason_tags.append("absurd_or_impossible_request")
    if fictional_hits and privacy_hits:
        scores["benign"] += 4.0
        scores["harmful"] = max(0.0, scores["harmful"] - 1.5)
        reason_tags.append("fictional_target_privacy_request")

    if is_expository and not operational_hits and not rights_hits:
        scores["benign"] += 1.5
        reason_tags.append("expository_query")
    if is_expository and harmful_hits and not operational_hits and not privacy_hits:
        scores["benign"] += 1.5
        reason_tags.append("historical_or_definitional_context")

    if rights_hits and not group_hits:
        scores["benign"] += 1.5
        reason_tags.append("non_group_normative_frame")

    if benign_hits and not privacy_hits and not rights_hits:
        scores["benign"] = max(scores["benign"], scores["harmful"] + 0.5)
        reason_tags.append("favor_benign_for_explicit_disambiguation")

    if fictional_hits and privacy_hits and is_expository:
        scores["benign"] = max(scores["benign"], scores["harmful"] + 1.0)
        reason_tags.append("favor_benign_for_fictional_context")
    if creative_hits and harmful_hits and not privacy_hits:
        scores["benign"] = max(scores["benign"], scores["harmful"] + 1.0)
        reason_tags.append("favor_benign_for_creative_depiction")
    if media_hits and harmful_hits and (fictional_hits or creative_hits):
        scores["benign"] = max(scores["benign"], scores["harmful"] + 0.75)
        reason_tags.append("favor_benign_for_media_analysis")

    if (harmful_hits or privacy_hits) and (operational_hits or privacy_hits):
        if (benign_hits or creative_hits or media_hits or fictional_hits) and not privacy_hits:
            reason_tags.append("skip_harmful_override_due_to_benign_context")
        elif fictional_hits and privacy_hits:
            reason_tags.append("skip_harmful_override_due_to_benign_context")
        else:
            minimum_gap = 0.5 if benign_hits or fictional_hits else 2.5
            scores["harmful"] = max(scores["harmful"], scores["benign"] + minimum_gap)
            reason_tags.append("favor_harmful_for_operational_risk")

    if rights_hits and group_hits and hostile_justification_hits:
        scores["harmful"] = max(scores["harmful"], scores["ethics"] + 0.75)
        reason_tags.append("favor_harmful_for_hostile_group_justification")
    elif rights_hits and group_hits:
        scores["ethics"] = max(scores["ethics"], scores["harmful"] + 0.25)
        reason_tags.append("favor_ethics_for_rights_reasoning")

    if group_hits and ("people i don't like" in normalized or hostile_justification_hits):
        scores["harmful"] += 1.5
        reason_tags.append("hostile_targeting")

    return scores, reason_tags


def _sorted_score_items(scores: Mapping[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def route_experience_buckets(problem: str) -> RoutingDecision:
    """Route a request into one of the legacy experience buckets."""

    normalized = _normalize_problem(problem)
    if not normalized:
        return RoutingDecision(
            primary_bucket="harmful",
            secondary_bucket="ethics",
            confidence=0.0,
            scores=dict.fromkeys(CATEGORY_KEYS, 0.0),
            reason_tags=("empty_query_fallback",),
            diagnostics=RoutingDiagnostics(
                evidence_conflict=False,
                conflict_type=None,
                tied_buckets=(),
            ),
        )

    evidence = _extract_routing_evidence(normalized)
    scores, reason_tags = _legacy_scores_from_evidence(normalized, evidence)
    ranked = _sorted_score_items(scores)
    primary_bucket, top_score = ranked[0]
    secondary_bucket, second_score = ranked[1]
    confidence = _legacy_confidence_from_scores(top_score, second_score)

    if second_score <= 0 or (top_score - second_score) > 2.5:
        secondary_bucket = None

    return RoutingDecision(
        primary_bucket=primary_bucket,
        secondary_bucket=secondary_bucket,
        confidence=confidence,
        scores={category: float(score) for category, score in scores.items()},
        reason_tags=tuple(dict.fromkeys(reason_tags)),
        diagnostics=RoutingDiagnostics(
            evidence_conflict=False,
            conflict_type=None,
            tied_buckets=tuple(bucket for bucket, score in ranked if score == top_score and top_score > 0)[1:],
        ),
    )


route_experience_buckets_legacy = route_experience_buckets


def normalize_router_mode(router_mode: str) -> str:
    """Normalize the public router name and its deprecated compatibility alias."""

    if router_mode == "v2_rule":
        return "rule"
    return router_mode


def route_experience_buckets_by_mode(problem: str, router_mode: str = "rule") -> RoutingDecision:
    """Dispatch the default EBS router or the legacy comparison implementation."""

    router_mode = normalize_router_mode(router_mode)
    if router_mode == "legacy":
        return route_experience_buckets(problem)
    if router_mode == "rule":
        from ebs.core.router import route_experience_buckets_rule

        return route_experience_buckets_rule(problem)
    raise ValueError(f"Unsupported router_mode `{router_mode}`.")


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


def infer_bucket_from_problem(
    problem: str,
    harmful_label: int | None = None,
    *,
    router_mode: str = "rule",
) -> str:
    """Infer the primary experience bucket for a query from text alone.

    The ``harmful_label`` argument is kept only for backward compatibility with
    existing callers. Routing intentionally relies on the request text itself.
    """

    del harmful_label
    return route_experience_buckets_by_mode(problem, router_mode=router_mode).primary_bucket


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
    router_mode: str = "rule",
) -> tuple[str, dict[str, str]]:
    """Select the best-matching experiences with bucket routing and Top-K retrieval.

    The ``harmful_label`` argument is accepted for compatibility but is not used
    for routing. When text-only routing is uncertain, a few experiences from the
    secondary bucket are considered during retrieval to reduce hard routing mistakes.
    """

    del harmful_label
    normalized = normalize_experience_bank(experiences)
    if max_experiences <= 0:
        return bucket or "harmful", {}

    if bucket in CATEGORY_KEYS:
        selected_bucket = str(bucket)
        decision = RoutingDecision(
            primary_bucket=selected_bucket,
            secondary_bucket=None,
            confidence=0.0,
            scores=dict.fromkeys(CATEGORY_KEYS, 0.0),
            reason_tags=("externally_forced_bucket",),
            diagnostics=RoutingDiagnostics(
                evidence_conflict=False,
                conflict_type=None,
                tied_buckets=(),
            ),
        )
    else:
        decision = route_experience_buckets_by_mode(problem, router_mode=router_mode)
        selected_bucket = decision.primary_bucket

    secondary_bucket = None
    if (
        bucket not in CATEGORY_KEYS
        and decision.secondary_bucket is not None
        and normalized.get(decision.secondary_bucket)
        and decision.confidence < _MIX_CONFIDENCE_THRESHOLD
    ):
        secondary_bucket = decision.secondary_bucket
    retriever = _get_cached_retriever(
        normalized,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        allow_fallback_hash=allow_fallback_hash,
    )
    selected = retriever.select(
        problem,
        bucket=selected_bucket,
        max_items=max_experiences,
        token_budget=token_budget,
        secondary_bucket=secondary_bucket,
        primary_boost=0.05,
        secondary_min_items=0,
    )
    if selected:
        return selected_bucket, selected

    for fallback_bucket in CATEGORY_KEYS:
        if normalized[fallback_bucket]:
            selected = retriever.select(
                problem,
                bucket=fallback_bucket,
                max_items=max_experiences,
                token_budget=token_budget,
            )
            if selected:
                return fallback_bucket, selected
    return selected_bucket, {}


def select_experiences_with_details(
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
    router_mode: str = "rule",
    routing_decision: RoutingDecision | None = None,
) -> tuple[RoutingDecision, dict[str, str], dict[str, float | int | bool]]:
    """Return routing decision plus selected experiences and retrieval diagnostics."""

    del harmful_label
    normalized = normalize_experience_bank(experiences)
    if max_experiences <= 0:
        decision = routing_decision or route_experience_buckets_by_mode(problem, router_mode=router_mode)
        return decision, {}, {"used_mixed_retrieval": False, "primary_selected_count": 0, "secondary_selected_count": 0}

    if bucket in CATEGORY_KEYS:
        decision = RoutingDecision(
            primary_bucket=str(bucket),
            secondary_bucket=None,
            confidence=0.0,
            scores=dict.fromkeys(CATEGORY_KEYS, 0.0),
            reason_tags=("externally_forced_bucket",),
            diagnostics=RoutingDiagnostics(
                evidence_conflict=False,
                conflict_type=None,
                tied_buckets=(),
            ),
        )
    elif routing_decision is not None:
        decision = routing_decision
    else:
        decision = route_experience_buckets_by_mode(problem, router_mode=router_mode)

    secondary_bucket = None
    if (
        bucket not in CATEGORY_KEYS
        and decision.secondary_bucket is not None
        and normalized.get(decision.secondary_bucket)
        and decision.confidence < _MIX_CONFIDENCE_THRESHOLD
    ):
        secondary_bucket = decision.secondary_bucket

    retriever = _get_cached_retriever(
        normalized,
        embedding_backend=embedding_backend,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        allow_fallback_hash=allow_fallback_hash,
    )
    selected, retrieval_details = retriever.select_with_details(
        problem,
        bucket=decision.primary_bucket,
        max_items=max_experiences,
        token_budget=token_budget,
        secondary_bucket=secondary_bucket,
        primary_boost=0.05,
        secondary_min_items=0,
    )
    updated_diagnostics = RoutingDiagnostics(
        evidence_conflict=decision.diagnostics.evidence_conflict if decision.diagnostics else False,
        conflict_type=decision.diagnostics.conflict_type if decision.diagnostics else None,
        tied_buckets=decision.diagnostics.tied_buckets if decision.diagnostics else (),
        used_mixed_retrieval=bool(retrieval_details["used_mixed_retrieval"]),
        average_primary_count=float(retrieval_details["primary_selected_count"]),
        average_secondary_count=float(retrieval_details["secondary_selected_count"]),
    )
    return (
        RoutingDecision(
            primary_bucket=decision.primary_bucket,
            secondary_bucket=secondary_bucket,
            confidence=decision.confidence,
            scores=decision.scores,
            reason_tags=decision.reason_tags,
            diagnostics=updated_diagnostics,
        ),
        selected,
        retrieval_details,
    )


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
