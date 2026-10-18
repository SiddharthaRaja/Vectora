"""
Domain models.

These are the contracts between the four stages. Every stage boundary is a
validated pydantic model rather than a loose dict, so a malformed chunk or a
missing field fails at the boundary with a readable error instead of three
calls later as a ``KeyError``.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

from vectora.config import QueryType, SearchMode

#: Stable namespace for deriving Qdrant point UUIDs from chunk ids. Fixed for
#: the life of the project: changing it orphans every vector in the store.
POINT_ID_NAMESPACE = uuid.UUID("8f1b5d26-6c2f-4a3e-9d41-1f0a7c5b2e88")

ChunkType = Literal["text", "table", "image"]


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def content_hash(text: str) -> str:
    """Short stable digest of chunk content, used for identity and dedup."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def estimate_tokens(text: str) -> int:
    """Cheap token estimate.

    Roughly four characters per token for English prose, but scientific text
    is denser than that, so we also floor at the word count. Good enough for
    budgeting; we never bill against it.
    """
    return max(len(text) // 4, len(text.split()))


class Chunk(BaseModel):
    """One retrievable unit of a paper, as written to ``chunks.jsonl``."""

    model_config = ConfigDict(extra="ignore")

    chunk_id: str
    paper_id: str
    title: str
    authors: str = ""
    year: str = ""
    doi: str = ""
    section: str = ""
    page: int = 0
    chunk_type: ChunkType = "text"
    content: str
    image_path: str = ""
    token_estimate: int = 0
    quality: float = 1.0
    """0-1 heuristic confidence that this chunk is real prose rather than
    layout debris. Written at ingestion, used to filter and to explain."""

    @field_validator("page", mode="before")
    @classmethod
    def _coerce_page(cls, v: Any) -> int:
        try:
            return max(int(v), 0)
        except (TypeError, ValueError):
            return 0

    @field_validator("year", mode="before")
    @classmethod
    def _clean_year(cls, v: Any) -> str:
        """Reject partial years. The old pipeline stored ``"20"`` because
        ``re.findall`` with a capturing group returns the group, not the
        match; anything that is not four digits is dropped here."""
        s = str(v or "").strip()
        return s if re.fullmatch(r"(?:19|20)\d{2}", s) else ""

    @classmethod
    def build(
        cls,
        *,
        content: str,
        paper_id: str,
        title: str,
        chunk_type: ChunkType = "text",
        quality: float = 1.0,
        **rest: Any,
    ) -> Chunk:
        """Construct with ``chunk_id`` and ``token_estimate`` derived."""
        return cls(
            chunk_id=content_hash(content),
            paper_id=paper_id,
            title=title or paper_id,
            chunk_type=chunk_type,
            content=content,
            token_estimate=estimate_tokens(content),
            quality=quality,
            **rest,
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def point_id(self) -> str:
        """Deterministic Qdrant point id.

        Derived from ``paper_id`` and ``chunk_id`` so that re-running the
        embedder upserts the same vector in place instead of appending a
        duplicate or — as the previous ``enumerate``-based ids did —
        overwriting an unrelated paper's vectors.
        """
        return str(uuid.uuid5(POINT_ID_NAMESPACE, f"{self.paper_id}:{self.chunk_id}"))

    def payload(self) -> dict[str, Any]:
        """What gets stored alongside the vector in Qdrant."""
        return self.model_dump(exclude={"point_id"})

    def short_citation(self) -> str:
        return f"[{first_author(self.authors) or self.paper_id}, {self.year or 'n.d.'}]"


def first_author(authors: str) -> str:
    """Surname of the first author, best-effort.

    Handles ``"A.M. Wallace, S. Gibson"``, ``"Wallace AM, Gibson S"`` and
    ``"Wallace, A.M."`` without pretending to be a full name parser.
    """
    if not authors:
        return ""
    first = re.split(r"[,;&]| and ", authors.strip())[0].strip()
    first = re.sub(r"[\d\*†‡§¶]+", "", first).strip()
    parts = [p for p in first.split() if p]
    if not parts:
        return ""
    # Drop leading initials ("A.M. Wallace" -> "Wallace"); if everything
    # looks like an initial, keep the longest token.
    words = [p for p in parts if not re.fullmatch(r"(?:[A-Z]\.?){1,3}", p)]
    return (words[-1] if words else max(parts, key=len)).strip(".,")


class RetrievedChunk(BaseModel):
    """A chunk plus the scores that got it here.

    Every stage's score is kept rather than overwritten, so a bad answer can
    be traced to the stage that let the wrong chunk through.
    """

    model_config = ConfigDict(extra="ignore")

    chunk: Chunk
    dense_score: float | None = None
    sparse_score: float | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    matched_variants: list[str] = Field(default_factory=list)
    """Which query phrasings surfaced this chunk. A chunk found by several
    independent rewrites is usually genuinely on-topic."""

    @property
    def score(self) -> float:
        """The score the answer stage should trust."""
        return self.rerank_score if self.rerank_score is not None else self.fused_score

    @property
    def is_reranked(self) -> bool:
        return self.rerank_score is not None

    def to_context_block(self, index: int) -> str:
        c = self.chunk
        header = [
            f"SOURCE [{index}]",
            f"Title  : {c.title}",
            f"Authors: {c.authors or 'unknown'}",
            f"Year   : {c.year or 'n.d.'}",
            f"Section: {c.section or 'unknown'} | Page: {c.page or '?'} | Type: {c.chunk_type}",
        ]
        if c.doi:
            header.append(f"DOI    : {c.doi}")
        return "\n".join(header) + "\n---\n" + c.content


class Citation(BaseModel):
    """A source the answer actually used."""

    model_config = ConfigDict(extra="ignore")

    index: int
    """The ``SOURCE [n]`` marker the model was shown, so a reader can map an
    inline marker back to a reference."""
    chunk_id: str
    paper_id: str
    title: str
    authors: str = ""
    year: str = ""
    doi: str = ""
    section: str = ""
    page: int = 0
    quote: str = ""
    """Verbatim span from the chunk that supports the claim, when the model
    supplied one. Verified against the chunk before being stored."""

    @classmethod
    def from_retrieved(cls, rc: RetrievedChunk, index: int, quote: str = "") -> Citation:
        c = rc.chunk
        return cls(
            index=index,
            chunk_id=c.chunk_id,
            paper_id=c.paper_id,
            title=c.title,
            authors=c.authors,
            year=c.year,
            doi=c.doi,
            section=c.section,
            page=c.page,
            quote=quote,
        )

    def format(self) -> str:
        who = first_author(self.authors) or "Unknown"
        bits = [f"{who} et al. ({self.year or 'n.d.'})", self.title[:80]]
        loc = ", ".join(x for x in [self.section, f"p.{self.page}" if self.page else ""] if x)
        if loc:
            bits.append(f"[{loc}]")
        if self.doi:
            bits.append(f"doi:{self.doi}")
        return " — ".join(bits[:2]) + " " + " ".join(bits[2:])


class Grounding(BaseModel):
    """Result of checking the answer's claims against the retrieved text."""

    model_config = ConfigDict(extra="ignore")

    score: int = Field(ge=0, le=100)
    verdict: Literal["grounded", "mostly_grounded", "partly_grounded", "ungrounded"]
    supported_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    summary: str = ""

    @classmethod
    def from_score(cls, score: int, **rest: Any) -> Grounding:
        if score >= 90:
            verdict = "grounded"
        elif score >= 70:
            verdict = "mostly_grounded"
        elif score >= 40:
            verdict = "partly_grounded"
        else:
            verdict = "ungrounded"
        return cls(score=score, verdict=verdict, **rest)

    def badge(self) -> str:
        icon = {
            "grounded": "✓",
            "mostly_grounded": "~",
            "partly_grounded": "!",
            "ungrounded": "✗",
        }[self.verdict]
        return f"{icon} {self.score}% grounded"


class Fallback(BaseModel):
    """Returned instead of an answer when retrieval could not support one."""

    model_config = ConfigDict(extra="ignore")

    reason: Literal["no_results", "low_confidence", "no_corpus"]
    message: str
    suggestions: list[str] = Field(default_factory=list)
    related_topics: list[str] = Field(default_factory=list)

    @property
    def triggered(self) -> bool:
        """Always true; the field exists so the wire shape stays stable for
        clients that check ``fallback.triggered``."""
        return True


class RetrievalDiagnostics(BaseModel):
    """What retrieval did, for the UI and for debugging a bad answer."""

    mode: SearchMode
    query_type: QueryType
    variants: list[str] = Field(default_factory=list)
    hyde_used: bool = False
    dense_hits: int = 0
    sparse_hits: int = 0
    fused_pool: int = 0
    reranked: bool = False
    diversified: bool = False
    returned: int = 0
    top_score: float | None = None
    elapsed_ms: dict[str, float] = Field(default_factory=dict)
    degraded: list[str] = Field(default_factory=list)
    """Stages that were skipped because a dependency failed, so a fast answer
    is never silently mistaken for a thorough one."""


class Usage(BaseModel):
    """Token and cost accounting for one request."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0

    def add(self, prompt: int, completion: int, cost: float) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.cost_usd += cost
        self.calls += 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Answer(BaseModel):
    """The complete result of one question."""

    model_config = ConfigDict(extra="ignore")

    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    query_type: QueryType = QueryType.EXPLORATORY
    mode: SearchMode = SearchMode.BALANCED
    grounding: Grounding | None = None
    fallback: Fallback | None = None
    diagnostics: RetrievalDiagnostics | None = None
    usage: Usage = Field(default_factory=Usage)
    created_at: datetime = Field(default_factory=utcnow)

    def wire(self) -> dict[str, Any]:
        """Legacy-compatible response body.

        The original Flask API and the shipped frontend expect these exact
        keys, so they are preserved and new information is additive.
        """
        return {
            "answer": self.answer,
            "citations": [c.model_dump() for c in self.citations],
            "query_type": self.query_type.value,
            "mode": self.mode.value,
            "faithfulness": (
                {
                    "score": self.grounding.score,
                    "verdict": self.grounding.verdict.upper().replace(
                        "GROUNDED", "FAITHFUL"
                    ),
                    "supported_claims": self.grounding.supported_claims,
                    "unsupported_claims": self.grounding.unsupported_claims,
                    "summary": self.grounding.summary,
                }
                if self.grounding
                else None
            ),
            "fallback": (
                {
                    "triggered": True,
                    "reason": self.fallback.reason,
                    "message": self.fallback.message,
                    "suggestions": self.fallback.suggestions,
                    "related_topics": self.fallback.related_topics,
                }
                if self.fallback
                else None
            ),
            "diagnostics": (
                self.diagnostics.model_dump(mode="json") if self.diagnostics else None
            ),
            "usage": self.usage.model_dump(),
        }


class Message(BaseModel):
    """One turn stored in SQLite."""

    model_config = ConfigDict(extra="ignore")

    id: int | None = None
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    citations: list[Citation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)


class Session(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    title: str = "New chat"
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    message_count: int = 0
