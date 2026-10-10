"""
The retrieval pipeline.

Flow, with every stage gated by the mode profile:

    query
      → contextualise (follow-ups only)
      → expand: rewrite ∥ HyDE          [concurrent]
      → dense search per vector ∥ sparse search per query   [concurrent]
      → reciprocal rank fusion
      → cross-encoder rerank
      → MMR diversification
      → deduplicate, cap per paper, trim to context budget
      → chunks

What makes this different from the version it replaces: dense and sparse
retrieval are independent retrievers fused by rank, rather than BM25
re-scoring the dense pool; the stages are parameterised rather than
switching on module globals; the CPU-bound model calls run in a thread so
they do not block the event loop; and every stage reports what it did, so a
bad answer can be attributed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from vectora.config import QueryType, SearchMode, Settings
from vectora.embeddings import Encoder, Reranker
from vectora.llm import LLMClient
from vectora.models import RetrievalDiagnostics, RetrievedChunk, Usage, estimate_tokens
from vectora.observability import RETRIEVED, STAGE_SECONDS, Stopwatch, get_logger
from vectora.retrieval.diversify import cap_per_paper, deduplicate, mmr
from vectora.retrieval.expansion import Expansion, contextualise, expand
from vectora.retrieval.fusion import RankedList, reciprocal_rank_fusion
from vectora.store import QdrantStore, SparseIndex

log = get_logger(__name__)

#: No single paper may supply more than this fraction of the context, so an
#: "across the literature" question cannot be answered from one source.
_MAX_PER_PAPER_RATIO = 0.5


class Retriever:
    """Owns the retrieval stages and the models they need.

    One instance per process: the encoder and reranker are expensive to
    load and cheap to share.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        store: QdrantStore,
        sparse: SparseIndex,
        encoder: Encoder,
        reranker: Reranker | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.sparse = sparse
        self.encoder = encoder
        self.reranker = reranker or Reranker(settings)
        self.llm = llm

    # ── stages ────────────────────────────────────────────────────────────

    async def _dense(
        self,
        expansion: Expansion,
        *,
        limit: int,
        filters: dict[str, Any] | None,
    ) -> list[RankedList]:
        """One dense search per query vector, concurrently.

        Each vector's hits become their own ranked list rather than being
        merged by score, so fusion sees the agreement between independent
        phrasings instead of a single flattened ordering.
        """

        async def one(label: str, vector: Any) -> RankedList | None:
            try:
                hits = await asyncio.to_thread(
                    self.store.search, vector, limit=limit, filters=filters
                )
            except Exception as exc:
                log.warning("dense search failed", extra={"variant": label, "err": str(exc)[:140]})
                return None
            return RankedList(
                name="dense",
                chunks=[c for c, _ in hits],
                scores=[s for _, s in hits],
                weight=self.settings.dense_weight,
                variant=label,
            )

        results = await asyncio.gather(
            *(one(label, vector) for label, vector in expansion.dense_vectors)
        )
        return [r for r in results if r is not None and r.chunks]

    async def _sparse(
        self,
        expansion: Expansion,
        *,
        limit: int,
        filters: dict[str, Any] | None,
    ) -> list[RankedList]:
        """Lexical search over the whole corpus, once per query phrasing."""
        if not self.sparse.ready:
            return []

        async def one(query: str, label: str) -> RankedList | None:
            try:
                hits = await asyncio.to_thread(
                    self.sparse.search, query, limit=limit, filters=filters
                )
            except Exception as exc:
                log.warning("sparse search failed", extra={"variant": label, "err": str(exc)[:140]})
                return None
            return RankedList(
                name="sparse",
                chunks=[h.chunk for h in hits],
                scores=[h.score for h in hits],
                weight=self.settings.sparse_weight,
                variant=label,
            )

        queries = [(expansion.search_query, "original")] + [
            (v, f"variant{i + 1}") for i, v in enumerate(expansion.variants)
        ]
        results = await asyncio.gather(*(one(q, label) for q, label in queries))
        return [r for r in results if r is not None and r.chunks]

    async def _rerank(
        self, candidates: Sequence[RetrievedChunk], query: str, *, limit: int
    ) -> list[RetrievedChunk]:
        """Score (query, passage) pairs with the cross-encoder.

        Run in a thread: this is a synchronous forward pass that would
        otherwise block every other request on the event loop.
        """
        if not candidates:
            return []
        try:
            scores = await asyncio.to_thread(
                self.reranker.score, query, [rc.chunk.content for rc in candidates]
            )
        except Exception as exc:
            log.warning("rerank failed, keeping fused order", extra={"err": str(exc)[:140]})
            return list(candidates[:limit])

        for rc, score in zip(candidates, scores, strict=True):
            rc.rerank_score = score
        return sorted(candidates, key=lambda rc: rc.rerank_score or 0.0, reverse=True)[:limit]

    async def _diversify(
        self,
        candidates: Sequence[RetrievedChunk],
        query: str,
        *,
        top_k: int,
        lambda_: float,
    ) -> list[RetrievedChunk]:
        try:
            return await asyncio.to_thread(
                mmr,
                candidates,
                query,
                encode=lambda texts: self.encoder.encode(texts, progress=False),
                top_k=top_k,
                lambda_=lambda_,
            )
        except Exception as exc:
            log.warning("mmr failed", extra={"err": str(exc)[:140]})
            return list(candidates[:top_k])

    def _trim_to_budget(self, chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        """Drop the weakest chunks until the context fits the budget.

        Protects the answer call from a long retrieval: better to answer
        from five good chunks than to have the model silently truncate the
        prompt and cite a source it never saw.
        """
        budget = self.settings.max_context_tokens
        kept: list[RetrievedChunk] = []
        used = 0
        for rc in chunks:
            cost = rc.chunk.token_estimate or estimate_tokens(rc.chunk.content)
            cost += 60  # per-source header
            if kept and used + cost > budget:
                log.info(
                    "context budget reached",
                    extra={"kept": len(kept), "dropped": len(chunks) - len(kept)},
                )
                break
            kept.append(rc)
            used += cost
        return kept

    # ── entry point ───────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        *,
        mode: SearchMode | None = None,
        query_type: QueryType = QueryType.EXPLORATORY,
        top_k: int | None = None,
        mmr_lambda: float | None = None,
        filters: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
        usage: Usage | None = None,
    ) -> tuple[list[RetrievedChunk], RetrievalDiagnostics]:
        """Retrieve chunks for one question.

        Returns the chunks and a diagnostics record. Never raises for a
        retrieval-quality reason: an empty list is a valid result that the
        answer stage turns into a fallback.
        """
        mode = mode or self.settings.default_mode
        profile = self.settings.profile(mode)
        route = self.settings.route(query_type)
        effective_k = top_k or route.top_k
        lambda_ = route.mmr_lambda if mmr_lambda is None else mmr_lambda

        watch = Stopwatch()
        diag = RetrievalDiagnostics(mode=mode, query_type=query_type)
        if not self.sparse.ready:
            diag.degraded.append("sparse_index_unavailable")

        # 1. Resolve follow-ups.
        with watch.stage("contextualise"):
            search_query, rewritten = await contextualise(
                query,
                history or [],
                llm=self.llm,
                settings=self.settings,
                usage=usage,
            )

        # 2. Expand.
        with watch.stage("expand"):
            expansion = await expand(
                search_query,
                settings=self.settings,
                encoder=self.encoder,
                llm=self.llm,
                rewrite_variants=profile.rewrite_variants,
                use_hyde=profile.use_hyde,
                usage=usage,
            )
        diag.variants = expansion.variants
        diag.hyde_used = expansion.hyde_used
        diag.degraded.extend(expansion.degraded)
        if rewritten:
            diag.variants = [f"(standalone) {search_query}", *diag.variants]

        # 3. Dense and sparse, concurrently.
        with watch.stage("search"):
            dense_lists, sparse_lists = await asyncio.gather(
                self._dense(expansion, limit=profile.dense_candidates, filters=filters),
                self._sparse(expansion, limit=profile.sparse_candidates, filters=filters),
            )
        diag.dense_hits = sum(len(rl.chunks) for rl in dense_lists)
        diag.sparse_hits = sum(len(rl.chunks) for rl in sparse_lists)

        if not dense_lists and not sparse_lists:
            diag.elapsed_ms = watch.timings
            log.info("retrieval found nothing", extra={"query": search_query[:80]})
            return [], diag

        # 4. Fuse.
        with watch.stage("fuse"):
            fused = reciprocal_rank_fusion(
                [*dense_lists, *sparse_lists], k=self.settings.rrf_k
            )
        diag.fused_pool = len(fused)

        # 5. Rerank a pool large enough for MMR to have a real choice.
        pool_size = max(effective_k * profile.rerank_pool_multiplier, effective_k + 6)
        if profile.use_reranker:
            with watch.stage("rerank"):
                candidates = await self._rerank(
                    fused[: max(pool_size * 2, 24)], search_query, limit=pool_size
                )
            diag.reranked = any(rc.is_reranked for rc in candidates)
            if not diag.reranked:
                diag.degraded.append("reranker_unavailable")
        else:
            candidates = list(fused[:pool_size])

        # 6. Diversify.
        if profile.use_mmr and len(candidates) > effective_k:
            with watch.stage("mmr"):
                candidates = await self._diversify(
                    candidates, search_query, top_k=effective_k, lambda_=lambda_
                )
            diag.diversified = True
        else:
            candidates = candidates[:effective_k]

        # 7. Dedup, cap per paper, fit the budget.
        with watch.stage("postprocess"):
            candidates = deduplicate(
                candidates,
                encode=lambda texts: self.encoder.encode(texts, progress=False),
                threshold=self.settings.near_duplicate_threshold,
            )
            candidates = cap_per_paper(
                candidates,
                max_per_paper=max(2, int(effective_k * _MAX_PER_PAPER_RATIO)),
            )[:effective_k]
            candidates = self._trim_to_budget(candidates)

        diag.returned = len(candidates)
        diag.top_score = round(candidates[0].score, 5) if candidates else None
        diag.elapsed_ms = {**watch.timings, "total": watch.total_ms}

        for stage, ms in watch.timings.items():
            STAGE_SECONDS.labels(stage=stage).observe(ms / 1000)
        RETRIEVED.observe(len(candidates))

        log.info(
            "retrieved",
            extra={
                "mode": mode.value,
                "query_type": query_type.value,
                "dense": diag.dense_hits,
                "sparse": diag.sparse_hits,
                "fused": diag.fused_pool,
                "returned": diag.returned,
                "top_score": diag.top_score,
                "total_ms": watch.total_ms,
            },
        )
        return candidates, diag

    def warm(self) -> None:
        """Load the models so the first question is not the slow one."""
        self.encoder.warm()
        try:
            self.reranker.warm()
        except Exception as exc:
            log.warning("reranker warmup failed", extra={"err": str(exc)[:120]})
