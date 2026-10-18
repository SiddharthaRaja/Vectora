"""
Centralised, environment-driven configuration.

Every tunable in Vectora lives here. Nothing else in the codebase reads
``os.environ`` or hardcodes a URL, model name or threshold, which is what
makes the same code run against a local Docker Qdrant and against Qdrant
Cloud without edits.

Precedence: explicit constructor argument > environment variable > ``.env``
file > the default below.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SearchMode(str, Enum):
    """How much work retrieval is allowed to do for one question."""

    FAST = "fast"
    BALANCED = "balanced"
    DEEP = "deep"


class QueryType(str, Enum):
    """What kind of question this is; selects retrieval params and prompt."""

    FACTUAL = "factual"
    COMPARATIVE = "comparative"
    METHODOLOGICAL = "methodological"
    DEFINITIONAL = "definitional"
    EXPLORATORY = "exploratory"
    CAUSAL = "causal"


class ModeProfile(BaseSettings):
    """The per-mode retrieval plan. Immutable once built."""

    model_config = SettingsConfigDict(frozen=True)

    dense_candidates: int
    sparse_candidates: int
    rewrite_variants: int
    use_hyde: bool
    use_reranker: bool
    use_mmr: bool
    check_faithfulness: bool
    rerank_pool_multiplier: int = 4


#: Mode profiles. ``fast`` makes exactly one LLM call (the answer itself).
MODE_PROFILES: dict[SearchMode, ModeProfile] = {
    SearchMode.FAST: ModeProfile(
        dense_candidates=24,
        sparse_candidates=24,
        rewrite_variants=0,
        use_hyde=False,
        use_reranker=False,
        use_mmr=False,
        check_faithfulness=False,
    ),
    SearchMode.BALANCED: ModeProfile(
        dense_candidates=40,
        sparse_candidates=40,
        rewrite_variants=2,
        use_hyde=True,
        use_reranker=True,
        use_mmr=True,
        check_faithfulness=False,
    ),
    SearchMode.DEEP: ModeProfile(
        dense_candidates=64,
        sparse_candidates=64,
        rewrite_variants=3,
        use_hyde=True,
        use_reranker=True,
        use_mmr=True,
        check_faithfulness=True,
    ),
}


class RouteProfile(BaseSettings):
    """Retrieval shape and prompt steer for one query type."""

    model_config = SettingsConfigDict(frozen=True)

    top_k: int
    mmr_lambda: float
    prompt_hint: str


#: ``mmr_lambda`` trades relevance (1.0) against diversity (0.0). A definition
#: wants the single best chunk; a comparison wants chunks from different papers.
ROUTE_PROFILES: dict[QueryType, RouteProfile] = {
    QueryType.FACTUAL: RouteProfile(
        top_k=4,
        mmr_lambda=0.85,
        prompt_hint=(
            "Answer precisely and concisely. Quote exact values, units and "
            "numbers as the sources state them. Do not round or convert."
        ),
    ),
    QueryType.DEFINITIONAL: RouteProfile(
        top_k=4,
        mmr_lambda=0.90,
        prompt_hint=(
            "Give a clear, precise definition. Include units, abbreviations "
            "and the context in which the term is used."
        ),
    ),
    QueryType.METHODOLOGICAL: RouteProfile(
        top_k=6,
        mmr_lambda=0.60,
        prompt_hint=(
            "Focus on experimental design, sample handling, instruments, "
            "statistical methods and validation or calibration procedures."
        ),
    ),
    QueryType.CAUSAL: RouteProfile(
        top_k=6,
        mmr_lambda=0.65,
        prompt_hint=(
            "Explain the mechanism or cause-and-effect relationship. State "
            "any confounders, limitations or alternative explanations the "
            "sources raise."
        ),
    ),
    QueryType.COMPARATIVE: RouteProfile(
        top_k=8,
        mmr_lambda=0.40,
        prompt_hint=(
            "Structure the answer as a direct comparison. Use a markdown "
            "table when there are three or more dimensions to compare."
        ),
    ),
    QueryType.EXPLORATORY: RouteProfile(
        top_k=8,
        mmr_lambda=0.35,
        prompt_hint=(
            "Summarise the main themes and findings across the sources. "
            "Note where sources disagree, and name open questions."
        ),
    ),
}


class Settings(BaseSettings):
    """Runtime configuration, assembled once per process."""

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        env_prefix="VECTORA_",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Environment ───────────────────────────────────────────────────────
    env: Literal["dev", "staging", "prod"] = "dev"
    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"
    debug: bool = False

    # ── OpenAI ────────────────────────────────────────────────────────────
    # Read without the VECTORA_ prefix so a standard OPENAI_API_KEY works.
    openai_api_key: SecretStr = Field(default=SecretStr(""), alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")

    answer_model: str = "gpt-4o-mini"
    utility_model: str = "gpt-4o-mini"
    """Model for routing, rewriting, HyDE and grounding — the cheap calls."""

    answer_max_tokens: int = 1500
    answer_temperature: float = 0.1
    utility_temperature: float = 0.3

    llm_timeout_seconds: float = 45.0
    llm_max_retries: int = 3
    llm_max_concurrency: int = 8
    """Cap on simultaneous OpenAI requests, so expansion fan-out cannot
    stampede the rate limit."""

    daily_cost_ceiling_usd: float = 5.0
    """Soft budget. When the running total crosses it, utility calls are
    skipped and only the answer call is made. 0 disables the ceiling."""

    # ── Embeddings ────────────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_batch_size: int = 64
    embedding_device: str | None = None
    """None means auto-detect (cuda if available, else cpu)."""

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_max_length: int = 512
    reranker_batch_size: int = 32

    # ── Vector store ──────────────────────────────────────────────────────
    qdrant_url: str | None = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_path: str | None = None
    """Set this instead of ``qdrant_url`` to use embedded on-disk Qdrant with
    no server. Mutually exclusive with ``qdrant_url``."""
    qdrant_collection: str | None = None
    """Override the collection name. Normally left unset: the active corpus
    derives it, so two corpora cannot collide in one collection."""
    qdrant_timeout: float = 30.0
    qdrant_upsert_batch: int = 256

    # ── Corpus ────────────────────────────────────────────────────────────
    corpus: str = "default"
    """Which corpus to use. Each one owns its own documents, chunk file,
    collection, lexical index and evaluation set, so switching corpora is
    this one value and rebuilding one cannot disturb another."""
    corpora_dir: Path = PROJECT_ROOT / "corpora"

    # ── Sparse index ──────────────────────────────────────────────────────
    bm25_index_path: Path | None = None
    """Override the lexical index location. Normally left unset and derived
    from the active corpus."""
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # ── Retrieval ─────────────────────────────────────────────────────────
    default_mode: SearchMode = SearchMode.BALANCED
    rrf_k: int = 60
    """Reciprocal-rank-fusion damping constant. 60 is the value from the
    original Cormack et al. paper and is a good default."""
    dense_weight: float = 1.0
    sparse_weight: float = 0.8
    near_duplicate_threshold: float = 0.93
    hyde_weight: float = 0.5
    """How much of the blended query vector comes from the hypothetical
    answer rather than the literal question."""

    # ── Answer guards ─────────────────────────────────────────────────────
    # Gating needs an ABSOLUTE relevance signal, and RRF is not one: it ranks,
    # so a chunk placed first in a hopeless pool scores exactly as well as one
    # placed first in a good pool. Both floors below are on scales that mean
    # the same thing from one query to the next. Tune them with `vectora eval`,
    # which reports how often off-topic questions are correctly refused.
    min_rerank_score: float = -4.0
    """Cross-encoder logit below which nothing retrieved is relevant. For
    ms-marco-MiniLM-L-6-v2, on-topic passages score above 0 and clearly
    irrelevant ones land between -8 and -11."""
    min_dense_score: float = 0.26
    """Cosine floor, used when the reranker did not run. all-MiniLM-L6-v2
    puts unrelated sentence pairs near 0.0-0.15 and genuinely related ones
    above 0.35; 0.26 sits in the gap."""
    min_fused_score: float = 0.012
    """Last-resort RRF floor, for when neither a rerank nor a dense score is
    available (lexical-only retrieval). It can only detect 'nothing was
    retrieved at all', which is why it is not the primary gate."""
    min_chunks_to_answer: int = 1
    max_context_tokens: int = 8000
    """Context is trimmed to this budget, lowest-scoring chunks dropped
    first, so a long retrieval can never blow the model's window."""

    # ── Conversation ──────────────────────────────────────────────────────
    history_turns: int = 12
    contextualise_below_words: int = 9
    """Questions shorter than this get rewritten against the conversation
    so follow-ups like 'and on CPU?' still retrieve sensibly."""

    # ── Storage ───────────────────────────────────────────────────────────
    database_url: str = Field(default="", alias="DATABASE_URL")
    sqlite_path: Path = PROJECT_ROOT / "data" / "chats.db"
    cache_path: Path = PROJECT_ROOT / "data" / "cache.db"
    cache_ttl_seconds: int = 60 * 60 * 24 * 14
    enable_cache: bool = True

    # These four are overrides; when unset they come from the active corpus.
    chunks_path: Path | None = None
    manifest_path: Path | None = None
    papers_dir: Path | None = None
    images_dir: Path | None = None

    legacy_chunks_path: Path = PROJECT_ROOT / "chunks.jsonl"
    """Where a pre-v3 single-corpus checkout keeps its chunks. Adopted into
    a corpus profile on first use rather than requiring a manual move."""

    # ── Ingestion ─────────────────────────────────────────────────────────
    semantic_similarity_threshold: float = 0.72
    max_chunk_tokens: int = 420
    min_chunk_tokens: int = 48
    chunk_overlap_sentences: int = 1
    """Sentences repeated from the previous chunk, so a fact split across a
    boundary is still retrievable from one chunk."""
    min_alpha_ratio: float = 0.62
    """Chunks whose characters are less than this fraction letters are
    discarded as OCR or layout debris."""
    image_min_width: int = 120
    image_min_height: int = 120
    caption_images: bool = False
    keep_uncaptioned_images: bool = False
    """An image chunk with no caption carries no retrievable text, so by
    default it is not emitted at all."""

    # ── API ───────────────────────────────────────────────────────────────
    host: str = "127.0.0.1"
    port: int = 5000
    # NoDecode: read the value as a plain comma-separated string, not JSON, so
    # VECTORA_CORS_ORIGINS=http://a,http://b works; _split_origins does the split.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5000"]
    )
    request_timeout_seconds: float = 120.0
    rate_limit_per_minute: int = 60
    enable_metrics: bool = True

    # ── Tracing ───────────────────────────────────────────────────────────
    enable_tracing: bool = False
    """Off by default: it needs a collector to be useful and a dependency to
    work. On, each question becomes a span tree showing where its time went."""
    otel_service_name: str = "vectora"
    otel_endpoint: str | None = Field(
        default=None, alias="OTEL_EXPORTER_OTLP_ENDPOINT"
    )
    """Read from the standard OTEL variable so this plugs into an existing
    collector with no Vectora-specific configuration."""

    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        level = v.upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"invalid log level: {v}")
        return level

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @model_validator(mode="after")
    def _check_store_target(self) -> Settings:
        if self.qdrant_path and self.qdrant_url:
            # An explicit path wins; this is the common case of someone
            # setting the path without clearing the default URL.
            object.__setattr__(self, "qdrant_url", None)
        if not self.qdrant_path and not self.qdrant_url:
            raise ValueError("set either VECTORA_QDRANT_URL or VECTORA_QDRANT_PATH")
        return self

    @model_validator(mode="after")
    def _make_dirs(self) -> Settings:
        """Create only what is unconditional. Corpus directories are created
        by the registry when a corpus is resolved, so an unset override does
        not silently make a stray folder."""
        for p in (
            self.sqlite_path.parent,
            self.cache_path.parent,
            self.corpora_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self

    # ── Derived ───────────────────────────────────────────────────────────

    @property
    def has_openai(self) -> bool:
        return bool(self.openai_api_key.get_secret_value())

    @property
    def resolved_device(self) -> str:
        if self.embedding_device:
            return self.embedding_device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:  # pragma: no cover - torch always present in practice
            pass
        return "cpu"

    # ── corpus resolution ─────────────────────────────────────────────────
    #
    # Everything that used to be a fixed path is now derived from the active
    # corpus, with the settings field acting as an override. The override
    # exists so a one-off run can point at a stray file without creating a
    # profile for it; normal operation leaves them unset.

    def resolve_corpus(self, name: str | None = None) -> Any:
        """Load the active corpus profile, adopting a legacy layout once.

        Returns a ``CorpusProfile``. Imported lazily because
        :mod:`vectora.corpus` imports the logger, which imports settings.
        """
        from vectora.corpus import CorpusRegistry, migrate_legacy_layout

        wanted = name or self.corpus
        registry = CorpusRegistry(self.corpora_dir)

        if not registry.exists(wanted):
            if wanted == "default":
                adopted = migrate_legacy_layout(
                    self.corpora_dir,
                    chunks_path=self.legacy_chunks_path,
                    papers_dir=PROJECT_ROOT / "papers",
                    collection_hint="research_papers",
                )
                if adopted is not None:
                    return adopted
                return registry.create(
                    "default",
                    description="Empty starter corpus.",
                    document_kind="documents",
                )
            registry.get(wanted)  # raises with the available names listed
        profile = registry.get(wanted)
        profile.ensure_dirs()
        return profile

    def paths_for(self, profile: Any) -> dict[str, Path]:
        """Concrete paths for one corpus, with overrides applied."""
        return {
            "chunks": self.chunks_path or profile.chunks_path,
            "manifest": self.manifest_path or profile.manifest_path,
            "papers": self.papers_dir or profile.papers_dir,
            "images": self.images_dir or profile.images_dir,
            "bm25": self.bm25_index_path or profile.bm25_path,
            "golden": profile.golden_path,
            "reports": profile.reports_dir,
        }

    def collection_for(self, profile: Any) -> str:
        return self.qdrant_collection or profile.collection

    def profile(self, mode: SearchMode) -> ModeProfile:
        return MODE_PROFILES[mode]

    def route(self, query_type: QueryType) -> RouteProfile:
        return ROUTE_PROFILES[query_type]

    def require_openai(self) -> str:
        key = self.openai_api_key.get_secret_value()
        if not key:
            raise RuntimeError(
                "No OpenAI API key. Put OPENAI_API_KEY=sk-... in your .env file "
                "or export it in your shell."
            )
        return key


@lru_cache(maxsize=1)
def get_settings(**overrides: object) -> Settings:
    """Process-wide settings singleton.

    Cached so that importing modules all observe the same object. Pass
    ``overrides`` only from tests, and call ``get_settings.cache_clear()``
    between tests that need different configuration.
    """
    return Settings(**overrides)  # type: ignore[arg-type]


def settings_for_test(**overrides: object) -> Settings:
    """Build a throwaway Settings that ignores the ambient .env file."""
    os.environ.setdefault("OPENAI_API_KEY", "sk-test")
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]
