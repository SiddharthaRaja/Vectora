"""
Answer orchestration.

One entry point, ``RAGEngine.answer``, used by the API, the CLI and the eval
harness. The previous code had two near-duplicate functions — ``generate``
and ``generate_with_history`` — which had drifted apart (different
faithfulness behaviour, different query handling), so a fix to one did not
reach the other. History is now just an argument.

Sequence:

    classify (rules, or one cheap call)
      → retrieve
      → guard: enough to answer?  → no: grounded fallback
      → build context, call the model
      → resolve citations from the markers
      → grounding check (deep mode)

Streaming shares the same preparation, so a streamed answer and a buffered
one differ only in delivery.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from vectora.config import QueryType, SearchMode, Settings
from vectora.generation import citations as cite
from vectora.generation import grounding as ground
from vectora.generation import guards
from vectora.generation.prompts import (
    ANSWER_SYSTEM,
    PROMPT_VERSION,
    answer_user_message,
)
from vectora.generation.router import classify
from vectora.llm import LLMClient, LLMError
from vectora.models import (
    Answer,
    RetrievalDiagnostics,
    RetrievedChunk,
    Usage,
)
from vectora.observability import QUESTIONS, Stopwatch, get_logger
from vectora.retrieval import Retriever
from vectora.tracing import question_span, set_attributes, span

log = get_logger(__name__)

#: Passage text included with each streamed source. Enough to read the
#: relevant part in the UI without making the sources frame large enough to
#: delay the first token.
_STREAM_TEXT_CHARS = 3000


def build_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Number the sources exactly as the citation parser expects: 1-indexed,
    in the order given."""
    return "\n\n".join(rc.to_context_block(i + 1) for i, rc in enumerate(chunks))


def clean_history(
    history: Sequence[dict[str, str]], *, limit: int
) -> list[dict[str, str]]:
    """Trim the transcript sent to the model.

    Assistant turns are stripped of anything that leaked in from a previous
    prompt — a trailing citation block from the old format, or a SOURCES
    section — so the context window is not spent re-sending chunks the model
    already answered from.
    """
    out: list[dict[str, str]] = []
    for turn in list(history)[-limit:]:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if not content or role not in {"user", "assistant"}:
            continue
        if role == "assistant":
            for marker in ("```json", "\nSOURCES:\n", "\nSOURCE [1]"):
                index = content.find(marker)
                if index > 0:
                    content = content[:index].strip()
        out.append({"role": role, "content": content})
    return out


@dataclass
class _Prepared:
    """Everything decided before the answer call."""

    query: str
    query_type: QueryType
    mode: SearchMode
    chunks: list[RetrievedChunk]
    diagnostics: RetrievalDiagnostics
    messages: list[dict[str, str]]
    usage: Usage
    fallback: Any | None = None


class RAGEngine:
    """Question in, grounded answer out."""

    def __init__(
        self,
        settings: Settings,
        *,
        retriever: Retriever,
        llm: LLMClient | None = None,
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.llm = llm

    # ── preparation ───────────────────────────────────────────────────────

    async def _prepare(
        self,
        query: str,
        *,
        mode: SearchMode,
        history: Sequence[dict[str, str]],
        filters: dict[str, Any] | None,
        top_k: int | None,
    ) -> _Prepared:
        usage = Usage()
        watch = Stopwatch()

        with watch.stage("route"), span("generation.route") as route_span:
            query_type, how = await classify(
                query, llm=self.llm, settings=self.settings, usage=usage
            )
            set_attributes(route_span, query_type=query_type.value, routed_by=how)

        chunks, diag = await self.retriever.retrieve(
            query,
            mode=mode,
            query_type=query_type,
            top_k=top_k,
            filters=filters,
            history=list(history),
            usage=usage,
        )
        diag.elapsed_ms = {**watch.timings, **diag.elapsed_ms}
        diag.elapsed_ms["routed_by"] = 0.0 if how == "rules" else 1.0

        ok, reason = guards.check(chunks, self.settings)
        if not ok:
            if not self.retriever.store.exists():
                reason = "no_corpus"
            with span("generation.fallback", reason=reason):
                fallback = await guards.build_fallback(
                    query, chunks, reason=reason, llm=self.llm, usage=usage
                )
            return _Prepared(
                query=query,
                query_type=query_type,
                mode=mode,
                chunks=list(chunks),
                diagnostics=diag,
                messages=[],
                usage=usage,
                fallback=fallback,
            )

        prior = clean_history(history, limit=self.settings.history_turns)
        messages = [
            {"role": "system", "content": ANSWER_SYSTEM},
            *prior,
            {
                "role": "user",
                "content": answer_user_message(
                    query=query,
                    context=build_context(chunks),
                    prompt_hint=self.settings.route(query_type).prompt_hint,
                    conversational=bool(prior),
                ),
            },
        ]
        return _Prepared(
            query=query,
            query_type=query_type,
            mode=mode,
            chunks=list(chunks),
            diagnostics=diag,
            messages=messages,
            usage=usage,
        )

    async def _finalise(
        self, prepared: _Prepared, raw_answer: str
    ) -> Answer:
        """Resolve citations and optionally grade grounding."""
        profile = self.settings.profile(prepared.mode)
        with span("generation.citations", sources=len(prepared.chunks)) as cite_span:
            answer_text, primary = cite.citations_from_markers(
                raw_answer, prepared.chunks
            )
            raw_markers = cite.parse_markers(raw_answer)
            invalid_markers = [
                n for n in raw_markers if not 1 <= n <= len(prepared.chunks)
            ]
            set_attributes(
                cite_span, cited=len(primary), invalid=len(invalid_markers)
            )

        tasks: dict[str, asyncio.Task] = {}
        if profile.check_faithfulness:
            tasks["grounding"] = asyncio.create_task(
                ground.check(
                    answer_text, prepared.chunks, llm=self.llm, usage=prepared.usage
                )
            )
            tasks["attribution"] = asyncio.create_task(
                cite.attribute(
                    answer_text, prepared.chunks, llm=self.llm, usage=prepared.usage
                )
            )

        grounding = None
        if tasks:
            with span("generation.verify", checks=len(tasks)):
                results = await asyncio.gather(
                    *tasks.values(), return_exceptions=True
                )
            named = dict(zip(tasks.keys(), results, strict=True))
            value = named.get("grounding")
            if value is not None and not isinstance(value, BaseException):
                grounding = value
            enriched = named.get("attribution")
            if isinstance(enriched, list) and enriched:
                primary = cite.merge_citations(primary, enriched)

        QUESTIONS.labels(
            mode=prepared.mode.value,
            query_type=prepared.query_type.value,
            outcome="answered",
        ).inc()

        return Answer(
            query=prepared.query,
            answer=answer_text,
            citations=cite.dedupe_by_paper(primary),
            query_type=prepared.query_type,
            mode=prepared.mode,
            grounding=grounding,
            diagnostics=prepared.diagnostics,
            usage=prepared.usage,
            raw_citation_markers=raw_markers,
            invalid_citation_markers=invalid_markers,
            source_chunk_ids=[rc.chunk.chunk_id for rc in prepared.chunks],
        )

    def _fallback_answer(self, prepared: _Prepared) -> Answer:
        QUESTIONS.labels(
            mode=prepared.mode.value,
            query_type=prepared.query_type.value,
            outcome=f"fallback_{prepared.fallback.reason}",
        ).inc()
        return Answer(
            query=prepared.query,
            answer=prepared.fallback.message,
            citations=[],
            query_type=prepared.query_type,
            mode=prepared.mode,
            fallback=prepared.fallback,
            diagnostics=prepared.diagnostics,
            usage=prepared.usage,
            source_chunk_ids=[rc.chunk.chunk_id for rc in prepared.chunks],
        )

    # ── public ────────────────────────────────────────────────────────────

    async def answer(
        self,
        query: str,
        *,
        mode: SearchMode | None = None,
        history: Sequence[dict[str, str]] | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> Answer:
        """Answer one question."""
        query = (query or "").strip()
        if not query:
            raise ValueError("empty query")
        mode = mode or self.settings.default_mode

        corpus_name = getattr(
            getattr(self.retriever, "hint", None), "document_kind", "documents"
        )
        with question_span(mode.value, "pending", corpus_name) as root:
            prepared = await self._prepare(
                query, mode=mode, history=history or [], filters=filters, top_k=top_k
            )
            set_attributes(
                root,
                query_type=prepared.query_type.value,
                chunks_retrieved=len(prepared.chunks),
                refused=prepared.fallback is not None,
            )
            if prepared.fallback is not None:
                return self._fallback_answer(prepared)
            if self.llm is None:
                raise LLMError(
                    "no OpenAI client configured; cannot generate an answer"
                )

            with span("generation.answer", model=self.settings.answer_model):
                raw = await self.llm.answer(
                    prepared.messages, usage=prepared.usage
                )
            answer = await self._finalise(prepared, raw)
            set_attributes(
                root,
                cost_usd=round(answer.usage.cost_usd, 6),
                llm_calls=answer.usage.calls,
                tokens=answer.usage.total_tokens,
                citations=len(answer.citations),
                grounding=answer.grounding.score if answer.grounding else None,
            )
            return answer

    async def stream(
        self,
        query: str,
        *,
        mode: SearchMode | None = None,
        history: Sequence[dict[str, str]] | None = None,
        filters: dict[str, Any] | None = None,
        top_k: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Answer with incremental delivery.

        Yields events: ``status`` while retrieving, ``sources`` once the
        chunks are known, ``token`` per delta, then exactly one ``done``
        carrying the full :class:`Answer` body — or one ``fallback``, or one
        ``error``. Callers can rely on a terminal event always arriving.

        Citations are resolved after the stream completes, because a marker
        can only be validated against the finished text.
        """
        query = (query or "").strip()
        mode = mode or self.settings.default_mode
        if not query:
            yield {"type": "error", "message": "Empty query."}
            return

        try:
            yield {"type": "status", "stage": "retrieving"}
            prepared = await self._prepare(
                query, mode=mode, history=history or [], filters=filters, top_k=top_k
            )
        except Exception as exc:
            log.exception("retrieval failed")
            yield {"type": "error", "message": f"Retrieval failed: {exc}"}
            return

        if prepared.fallback is not None:
            answer = self._fallback_answer(prepared)
            yield {"type": "fallback", **answer.wire()}
            return

        # The passage text and per-stage scores go out with the sources, so a
        # reader can inspect what the answer was built from while it is still
        # being written. Text is capped: this is a live stream, not a bulk
        # export, and a 20-chunk payload of full passages would stall the
        # first token behind several hundred kilobytes.
        yield {
            "type": "sources",
            "count": len(prepared.chunks),
            "sources": [
                {
                    "index": i + 1,
                    "paper_id": rc.chunk.paper_id,
                    "title": rc.chunk.title,
                    "authors": rc.chunk.authors,
                    "year": rc.chunk.year,
                    "doi": rc.chunk.doi,
                    "section": rc.chunk.section,
                    "page": rc.chunk.page,
                    "chunk_type": rc.chunk.chunk_type,
                    "quality": rc.chunk.quality,
                    "score": round(rc.score, 4),
                    "fused_score": rc.fused_score,
                    "dense_score": rc.dense_score,
                    "sparse_score": rc.sparse_score,
                    "rerank_score": rc.rerank_score,
                    "matched_variants": rc.matched_variants,
                    "text": rc.chunk.content[:_STREAM_TEXT_CHARS],
                    "truncated": len(rc.chunk.content) > _STREAM_TEXT_CHARS,
                }
                for i, rc in enumerate(prepared.chunks)
            ],
        }

        if self.llm is None:
            yield {"type": "error", "message": "No OpenAI client configured."}
            return

        yield {"type": "status", "stage": "generating"}
        pieces: list[str] = []
        try:
            async for token in self.llm.stream_answer(
                prepared.messages, usage=prepared.usage
            ):
                pieces.append(token)
                yield {"type": "token", "text": token}
        except Exception as exc:
            log.warning("stream failed, falling back to buffered", extra={"err": str(exc)[:140]})
            if pieces:
                # Already emitted tokens; cannot restart without duplicating.
                yield {"type": "error", "message": "The answer stream was interrupted."}
                return
            try:
                raw = await self.llm.answer(prepared.messages, usage=prepared.usage)
            except LLMError as exc2:
                yield {"type": "error", "message": str(exc2)}
                return
            pieces = [raw]
            yield {"type": "token", "text": raw}

        yield {"type": "status", "stage": "verifying"}
        answer = await self._finalise(prepared, "".join(pieces))
        yield {"type": "done", **answer.wire()}

    def info(self) -> dict[str, Any]:
        return {
            "prompt_version": PROMPT_VERSION,
            "default_mode": self.settings.default_mode.value,
            "answer_model": self.settings.answer_model,
            "utility_model": self.settings.utility_model,
            "llm_available": self.llm is not None,
        }
