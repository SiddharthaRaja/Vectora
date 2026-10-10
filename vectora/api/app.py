"""
The FastAPI application.

Why FastAPI rather than the original Flask: streaming an answer token by
token needs an async server, the request/response bodies validate
themselves from the same pydantic models the rest of the code uses, and
there is an OpenAPI schema at ``/docs`` for free. The route paths and
response shapes are unchanged, so the existing frontend works against it
without edits.

What the middleware adds: a request id on every request and in every log
line and error response; a timing header; a simple per-client rate limit so
one browser tab cannot queue twenty expensive questions; and an exception
handler that returns a usable JSON error with that request id instead of an
HTML stack trace.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from vectora import __version__
from vectora.api.deps import build_container
from vectora.api.routes import router
from vectora.config import Settings, get_settings
from vectora.observability import (
    configure_logging,
    get_logger,
    request_id_var,
    session_id_var,
)

log = get_logger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"


def create_app(settings: Settings | None = None, *, warm: bool = True) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        log.info(
            "starting vectora",
            extra={"version": __version__, "env": settings.env, "port": settings.port},
        )
        # A container may already have been attached — tests inject a stub to
        # avoid loading models. Only what we build here is ours to close.
        owned = getattr(app.state, "container", None) is None
        if owned:
            # Built off the event loop: it loads two models and touches disk.
            app.state.container = await asyncio.to_thread(
                build_container, settings, warm=warm
            )
        container = app.state.container
        if container.startup_warnings:
            log.warning(
                "started with warnings", extra={"count": len(container.startup_warnings)}
            )
        else:
            log.info("ready")
        try:
            yield
        finally:
            log.info("shutting down")
            if owned:
                await container.aclose()

    app = FastAPI(
        title="Vectora",
        description=(
            "Retrieval-augmented question answering over a corpus of research papers. "
            "Hybrid dense + lexical retrieval, rank fusion, cross-encoder reranking, "
            "verified citations and a grounding check."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["*"],
    )

    _install_middleware(app, settings)
    _install_error_handlers(app)
    app.include_router(router)
    _install_static(app)
    return app


def _install_middleware(app: FastAPI, settings: Settings) -> None:
    @app.middleware("http")
    async def correlate(request: Request, call_next: Callable) -> Response:
        """Bind a request id and time the request."""
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming[:64] if incoming else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        session_token = session_id_var.set(
            str(request.path_params.get("session_id", "-")) or "-"
        )
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
            session_id_var.reset(session_token)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.1f}"
        if request.url.path not in {"/health", "/metrics"}:
            log.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "ms": round(elapsed_ms, 1),
                },
            )
        return response

    # Per-client sliding window, applied only to the expensive endpoints.
    # In-process, which is right for a single-instance deployment; behind
    # several replicas this belongs in a shared store.
    buckets: dict[str, deque[float]] = defaultdict(deque)
    guarded = ("/chat", "/search")
    lock = asyncio.Lock()

    @app.middleware("http")
    async def rate_limit(request: Request, call_next: Callable) -> Response:
        if settings.rate_limit_per_minute <= 0 or not any(
            part in request.url.path for part in guarded
        ):
            return await call_next(request)

        client = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (
            request.client.host if request.client else "unknown"
        )
        now = time.monotonic()
        async with lock:
            window = buckets[client]
            while window and now - window[0] > 60.0:
                window.popleft()
            if len(window) >= settings.rate_limit_per_minute:
                retry = int(60 - (now - window[0])) + 1
                log.warning("rate limited", extra={"client": client})
                return JSONResponse(
                    {
                        "error": "Too many requests",
                        "detail": f"Limit is {settings.rate_limit_per_minute} per minute.",
                        "request_id": request_id_var.get(),
                    },
                    status_code=429,
                    headers={"Retry-After": str(retry)},
                )
            window.append(now)
        return await call_next(request)


def _install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def on_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(p) for p in first.get("loc", ())[1:]) or "request"
        return JSONResponse(
            {
                "error": "Invalid request",
                "detail": f"{field}: {first.get('msg', 'is not valid')}",
                "request_id": request_id_var.get(),
            },
            status_code=422,
        )

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        return JSONResponse(
            {
                "error": str(exc.detail),
                "request_id": request_id_var.get(),
            },
            status_code=exc.status_code,
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def on_unhandled(request: Request, exc: Exception) -> JSONResponse:
        """Last resort: log with the traceback, return the request id.

        The message is deliberately generic — the detail is in the log,
        keyed by the same id the caller is shown.
        """
        log.exception("unhandled error", extra={"path": request.url.path})
        return JSONResponse(
            {
                "error": "Internal error",
                "detail": "Something failed on our side. Quote the request id.",
                "request_id": request_id_var.get(),
            },
            status_code=500,
        )


def _install_static(app: FastAPI) -> None:
    """Serve the single-page frontend, if it is present."""
    index = STATIC_DIR / "index.html"
    if not index.exists():
        log.warning("no static/index.html; serving the API only")
        return

    app.mount(
        "/static", StaticFiles(directory=str(STATIC_DIR), html=False), name="static"
    )

    @app.get("/", include_in_schema=False)
    async def root() -> FileResponse:
        # no-store: the shell is tiny and a stale copy against a new API is
        # a confusing failure to debug.
        return FileResponse(index, headers={"Cache-Control": "no-store"})


#: Module-level app for ``uvicorn vectora.api.app:app``.
app = create_app()
