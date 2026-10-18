"""
Application assembly.

One container holds the objects that are expensive to build — the models,
the store, the sparse index, the LLM client — and is created once at
startup. Routes reach it through FastAPI's dependency system, which keeps
them free of module-level globals and makes them trivially testable with a
stub container.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vectora.cache import Cache
from vectora.config import Settings, get_settings
from vectora.corpus import CorpusContext
from vectora.db import ChatRepository
from vectora.embeddings import Encoder, Reranker
from vectora.generation import RAGEngine
from vectora.llm import LLMClient
from vectora.observability import configure_logging, get_logger
from vectora.retrieval import Retriever
from vectora.store import QdrantStore, SparseIndex
from vectora.tracing import configure_tracing, shutdown_tracing

log = get_logger(__name__)


@dataclass
class Container:
    """Everything the app needs, built once."""

    settings: Settings
    corpus: CorpusContext
    cache: Cache
    store: QdrantStore
    sparse: SparseIndex
    encoder: Encoder
    reranker: Reranker
    repository: ChatRepository
    engine: RAGEngine
    llm: LLMClient | None
    startup_warnings: list[str]

    async def aclose(self) -> None:
        shutdown_tracing()
        if self.llm is not None:
            await self.llm.close()
        self.cache.close()
        self.repository.close()
        self.store.close()

    def readiness(self) -> dict[str, Any]:
        """Component-level health, for ``/ready``.

        Deliberately granular: the service can serve the UI and browse
        history with Qdrant down, so a single boolean would be misleading.
        """
        store_ok = self.store.healthy()
        indexed = 0
        if store_ok:
            try:
                indexed = self.store.stats().vectors
            except Exception:
                store_ok = False
        return {
            "corpus": {
                "name": self.corpus.name,
                "collection": self.corpus.collection,
                "documents": self.corpus.profile.documents,
                "kind": self.corpus.profile.document_kind,
            },
            "vector_store": {"ok": store_ok, "vectors": indexed},
            "sparse_index": {"ok": self.sparse.ready, "chunks": self.sparse.size},
            "llm": {"ok": self.llm is not None, "model": self.settings.answer_model},
            "database": {"ok": True, **self.repository.stats()},
            "can_answer": bool(store_ok and indexed > 0 and self.llm is not None),
            "warnings": self.startup_warnings,
        }


def build_container(
    settings: Settings | None = None,
    *,
    warm: bool = True,
    corpus: str | None = None,
) -> Container:
    """Construct the application graph.

    Startup is deliberately tolerant: a missing API key or an unreachable
    Qdrant produces a warning and a degraded service that can still serve
    the UI and explain the problem, rather than a process that will not
    boot. Anything genuinely unrecoverable is surfaced by ``/ready``.
    """
    settings = settings or get_settings()
    configure_logging(settings)
    if configure_tracing(settings):
        log.info("tracing active")
    warnings: list[str] = []

    # Resolve the corpus first: it decides the collection, the lexical index
    # and the chunk file, so nothing below can be built without it.
    try:
        ctx = CorpusContext.build(settings, corpus)
    except KeyError as exc:
        raise RuntimeError(str(exc).strip("\"'")) from exc
    log.info("corpus resolved", extra={"corpus": ctx.name, "collection": ctx.collection})

    cache = Cache(settings.cache_path, settings.cache_ttl_seconds, settings.enable_cache)
    encoder = Encoder(settings, cache=cache)
    reranker = Reranker(settings)
    store = QdrantStore(settings, collection=ctx.collection)
    sparse = SparseIndex(settings, path=ctx.bm25)
    repository = ChatRepository(settings.sqlite_path)

    llm: LLMClient | None = None
    if settings.has_openai:
        try:
            llm = LLMClient(settings, cache=cache)
        except Exception as exc:
            warnings.append(f"OpenAI client unavailable: {exc}")
            log.error("llm init failed", extra={"err": str(exc)})
    else:
        warnings.append(
            "No OPENAI_API_KEY set. Retrieval works; answers do not. "
            "Put OPENAI_API_KEY=sk-... in your .env file."
        )

    if not store.healthy():
        warnings.append(
            f"Cannot reach Qdrant at {settings.qdrant_url or settings.qdrant_path}. "
            "Start it, or set VECTORA_QDRANT_PATH to use the embedded engine."
        )
    elif not store.exists():
        warnings.append(
            f"Corpus '{ctx.name}' is not indexed yet (collection "
            f"'{ctx.collection}' does not exist). Run `vectora index`."
        )

    if not sparse.ensure(
        store=store if store.healthy() else None, chunks_path=ctx.chunks
    ):
        warnings.append(
            "Lexical (BM25) index unavailable, so retrieval is dense-only. "
            "Run `vectora index` to build it."
        )

    retriever = Retriever(
        settings,
        store=store,
        sparse=sparse,
        encoder=encoder,
        reranker=reranker,
        llm=llm,
        corpus=ctx.profile,
    )
    engine = RAGEngine(settings, retriever=retriever, llm=llm)

    if warm:
        try:
            retriever.warm()
        except Exception as exc:
            warnings.append(f"Model warmup failed: {exc}")
            log.warning("warmup failed", extra={"err": str(exc)[:160]})

    for warning in warnings:
        log.warning(warning)

    return Container(
        settings=settings,
        corpus=ctx,
        cache=cache,
        store=store,
        sparse=sparse,
        encoder=encoder,
        reranker=reranker,
        repository=repository,
        engine=engine,
        llm=llm,
        startup_warnings=warnings,
    )
