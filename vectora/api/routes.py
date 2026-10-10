"""
HTTP routes.

Route paths and response bodies match the original Flask app, so the
existing frontend works against this unchanged. Added on top: SSE
streaming at ``/sessions/{id}/chat/stream``, ``/ready`` with per-component
status, ``/metrics``, ``/stats``, and a search endpoint that returns
retrieval results without spending a generation call — which is the single
most useful thing to have when debugging answer quality.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from vectora.api.deps import Container
from vectora.api.schemas import (
    ChatRequest,
    MessageOut,
    RenameRequest,
    SessionOut,
)
from vectora.models import Session
from vectora.observability import get_logger, render_metrics, request_id_var

log = get_logger(__name__)

router = APIRouter()


def get_container(request: Request) -> Container:
    container = getattr(request.app.state, "container", None)
    if container is None:  # pragma: no cover - set during startup
        raise HTTPException(503, "Application is still starting up.")
    return container


def _session_out(session: Session) -> SessionOut:
    return SessionOut(
        id=session.id,
        title=session.title,
        created=session.created_at.isoformat(),
        updated=session.updated_at.isoformat(),
        message_count=session.message_count,
    )


# ── health and introspection ──────────────────────────────────────────────


@router.get("/health", tags=["ops"])
async def health(container: Container = Depends(get_container)) -> dict[str, Any]:
    """Liveness. Cheap, no dependencies touched — for the load balancer."""
    from vectora import __version__

    return {"status": "ok", "version": __version__, "env": container.settings.env}


@router.get("/ready", tags=["ops"])
async def ready(
    response: Response, container: Container = Depends(get_container)
) -> dict[str, Any]:
    """Readiness, component by component.

    Returns 503 only when the service cannot answer at all, so a degraded
    deployment (no lexical index, say) stays in rotation.
    """
    report = await asyncio.to_thread(container.readiness)
    if not report["can_answer"]:
        response.status_code = 503
    return report


@router.get("/stats", tags=["ops"])
async def stats(container: Container = Depends(get_container)) -> dict[str, Any]:
    """Corpus, cache and spend. Useful while tuning."""
    store_stats: dict[str, Any]
    try:
        store_stats = container.store.stats().__dict__
    except Exception as exc:
        store_stats = {"error": str(exc)}
    return {
        "corpus": store_stats,
        "sparse_index": container.sparse.stats(),
        "cache": container.cache.stats(),
        "database": container.repository.stats(),
        "llm": container.llm.stats() if container.llm else {"available": False},
        "engine": container.engine.info(),
        "embedding": {
            "model": container.settings.embedding_model,
            "dim": container.settings.embedding_dim,
            "device": container.settings.resolved_device,
            "reranker": container.settings.reranker_model,
        },
    }


@router.get("/metrics", tags=["ops"], include_in_schema=False)
async def metrics(container: Container = Depends(get_container)) -> Response:
    if not container.settings.enable_metrics:
        raise HTTPException(404, "Metrics are disabled.")
    return Response(content=render_metrics(), media_type="text/plain; version=0.0.4")


# ── sessions ──────────────────────────────────────────────────────────────


@router.get("/sessions", tags=["sessions"])
async def list_sessions(
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    container: Container = Depends(get_container),
) -> list[SessionOut]:
    sessions = await asyncio.to_thread(container.repository.list_sessions, limit, offset)
    return [_session_out(s) for s in sessions]


@router.post("/sessions", tags=["sessions"], status_code=201)
async def create_session(container: Container = Depends(get_container)) -> SessionOut:
    session = await asyncio.to_thread(container.repository.create_session)
    return _session_out(session)


@router.patch("/sessions/{session_id}", tags=["sessions"])
async def rename_session(
    session_id: str,
    body: RenameRequest,
    container: Container = Depends(get_container),
) -> dict[str, bool]:
    ok = await asyncio.to_thread(
        container.repository.rename_session, session_id, body.title
    )
    if not ok:
        raise HTTPException(404, "Session not found.")
    return {"ok": True}


@router.delete("/sessions/{session_id}", tags=["sessions"])
async def delete_session(
    session_id: str, container: Container = Depends(get_container)
) -> dict[str, bool]:
    await asyncio.to_thread(container.repository.delete_session, session_id)
    return {"ok": True}


@router.get("/sessions/{session_id}/messages", tags=["sessions"])
async def get_messages(
    session_id: str, container: Container = Depends(get_container)
) -> list[MessageOut]:
    if await asyncio.to_thread(container.repository.get_session, session_id) is None:
        raise HTTPException(404, "Session not found.")
    messages = await asyncio.to_thread(container.repository.get_messages, session_id)
    return [
        MessageOut(
            role=m.role,
            content=m.content,
            citations=[c.model_dump() for c in m.citations],  # type: ignore[misc]
            created=m.created_at.isoformat(),
            metadata=m.metadata,
        )
        for m in messages
    ]


# ── chat ──────────────────────────────────────────────────────────────────


async def _answer_and_store(
    container: Container, session_id: str, body: ChatRequest
) -> dict[str, Any]:
    history = await asyncio.to_thread(
        container.repository.get_history, session_id, container.settings.history_turns
    )
    answer = await container.engine.answer(
        body.query,
        mode=body.mode,
        history=history,
        filters=body.filters(),
        top_k=body.top_k,
    )
    await asyncio.to_thread(
        container.repository.add_exchange,
        session_id,
        body.query,
        answer.answer,
        citations=answer.citations,
        metadata={
            "mode": answer.mode.value,
            "query_type": answer.query_type.value,
            "grounding": answer.grounding.model_dump() if answer.grounding else None,
            "usage": answer.usage.model_dump(),
            "fallback": answer.fallback.reason if answer.fallback else None,
        },
    )
    return answer.wire()


@router.post("/sessions/{session_id}/chat", tags=["chat"])
async def chat(
    session_id: str,
    body: ChatRequest,
    container: Container = Depends(get_container),
) -> JSONResponse:
    """Answer a question in a session, buffered.

    Kept for clients that cannot consume SSE, and used by the existing
    frontend. The response shape is unchanged from the Flask version.
    """
    if await asyncio.to_thread(container.repository.get_session, session_id) is None:
        raise HTTPException(404, "Session not found.")
    try:
        payload = await asyncio.wait_for(
            _answer_and_store(container, session_id, body),
            timeout=container.settings.request_timeout_seconds,
        )
    except asyncio.TimeoutError:
        log.error("chat timed out", extra={"session_id": session_id})
        raise HTTPException(
            504, "The question took too long. Try `fast` mode, or a narrower question."
        ) from None
    return JSONResponse(payload)


@router.post("/sessions/{session_id}/chat/stream", tags=["chat"])
async def chat_stream(
    session_id: str,
    body: ChatRequest,
    container: Container = Depends(get_container),
) -> StreamingResponse:
    """Answer with server-sent events.

    Event stream: ``status``, ``sources``, many ``token``, then one of
    ``done`` / ``fallback`` / ``error``. The exchange is persisted when the
    answer completes, so an abandoned stream does not store a partial turn.
    """
    if await asyncio.to_thread(container.repository.get_session, session_id) is None:
        raise HTTPException(404, "Session not found.")

    request_id = request_id_var.get()

    async def events() -> AsyncIterator[bytes]:
        history = await asyncio.to_thread(
            container.repository.get_history,
            session_id,
            container.settings.history_turns,
        )
        final: dict[str, Any] | None = None
        try:
            async for event in container.engine.stream(
                body.query,
                mode=body.mode,
                history=history,
                filters=body.filters(),
                top_k=body.top_k,
            ):
                if event.get("type") in {"done", "fallback"}:
                    final = event
                yield f"data: {json.dumps(event, default=str)}\n\n".encode()
        except asyncio.CancelledError:
            log.info("client disconnected mid-stream", extra={"session_id": session_id})
            raise
        except Exception as exc:
            log.exception("stream failed")
            yield (
                "data: "
                + json.dumps(
                    {
                        "type": "error",
                        "message": str(exc),
                        "request_id": request_id,
                    }
                )
                + "\n\n"
            ).encode()
        else:
            if final is not None:
                await asyncio.to_thread(
                    container.repository.add_exchange,
                    session_id,
                    body.query,
                    final.get("answer", ""),
                    citations=None,
                    metadata={
                        "mode": final.get("mode"),
                        "query_type": final.get("query_type"),
                        "grounding": final.get("faithfulness"),
                        "usage": final.get("usage"),
                        "fallback": (final.get("fallback") or {}).get("reason"),
                        "citations": final.get("citations"),
                    },
                )
        yield b"data: [DONE]\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx must not buffer SSE
        },
    )


# ── retrieval only ────────────────────────────────────────────────────────


@router.post("/search", tags=["debug"])
async def search(
    body: ChatRequest, container: Container = Depends(get_container)
) -> dict[str, Any]:
    """Run retrieval and return the chunks, with no generation call.

    The fastest way to tell whether a bad answer is a retrieval problem or
    a prompting problem, and it costs nothing in ``fast`` mode.
    """
    from vectora.generation.router import classify

    query_type, routed_by = await classify(
        body.query, llm=container.llm, settings=container.settings
    )
    chunks, diag = await container.engine.retriever.retrieve(
        body.query,
        mode=body.mode,
        query_type=query_type,
        top_k=body.top_k,
        filters=body.filters(),
    )
    return {
        "query": body.query,
        "query_type": query_type.value,
        "routed_by": routed_by,
        "results": [
            {
                "rank": i + 1,
                "score": round(rc.score, 5),
                "fused_score": rc.fused_score,
                "dense_score": rc.dense_score,
                "sparse_score": rc.sparse_score,
                "rerank_score": rc.rerank_score,
                "dense_rank": rc.dense_rank,
                "sparse_rank": rc.sparse_rank,
                "matched_variants": rc.matched_variants,
                "quality": rc.chunk.quality,
                "paper_id": rc.chunk.paper_id,
                "title": rc.chunk.title,
                "authors": rc.chunk.authors,
                "year": rc.chunk.year,
                "section": rc.chunk.section,
                "page": rc.chunk.page,
                "chunk_type": rc.chunk.chunk_type,
                "content": rc.chunk.content,
            }
            for i, rc in enumerate(chunks)
        ],
        "diagnostics": diag.model_dump(mode="json"),
    }


@router.get("/papers", tags=["debug"])
async def papers(container: Container = Depends(get_container)) -> dict[str, Any]:
    """The papers in the index, for populating a filter dropdown."""
    try:
        ids = await asyncio.to_thread(container.store.indexed_paper_ids)
    except Exception as exc:
        raise HTTPException(503, f"Cannot read the index: {exc}") from exc
    return {"count": len(ids), "paper_ids": sorted(ids)}


# ── export ────────────────────────────────────────────────────────────────


@router.get("/sessions/{session_id}/export/pdf", tags=["export"])
async def export_pdf(
    session_id: str, container: Container = Depends(get_container)
) -> Response:
    """Render a conversation to PDF.

    weasyprint is an optional dependency with system libraries behind it, so
    a missing install returns a clear 501 rather than a stack trace.
    """
    from vectora.api.export import build_conversation_html, render_pdf

    session = await asyncio.to_thread(container.repository.get_session, session_id)
    if session is None:
        raise HTTPException(404, "Session not found.")
    messages = await asyncio.to_thread(container.repository.get_messages, session_id)
    if not messages:
        raise HTTPException(400, "This conversation has no messages to export.")

    html = build_conversation_html(session, messages)
    try:
        pdf = await asyncio.to_thread(render_pdf, html)
    except ImportError:
        raise HTTPException(
            501,
            "PDF export needs weasyprint: pip install 'vectora[pdf]'. "
            "On Windows it also needs the GTK runtime.",
        ) from None
    except Exception as exc:
        log.exception("pdf render failed")
        raise HTTPException(500, f"Could not render the PDF: {exc}") from exc

    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in session.title)[:50]
    filename = f"vectora_{safe.strip() or 'conversation'}_{session.created_at:%Y-%m-%d}.pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
