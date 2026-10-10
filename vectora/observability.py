"""
Logging, request correlation and metrics.

One request id follows a question through routing, expansion, retrieval and
generation, so a slow or wrong answer can be reconstructed from the log
without guessing which lines belong together.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from vectora.config import Settings

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="-"
)
session_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "session_id", default="-"
)

_RESERVED = set(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | {"message", "asctime", "taskName"}


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        record.session_id = session_id_var.get()
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line, which is what log shippers want."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        session = getattr(record, "session_id", "-")
        if session != "-":
            payload["session_id"] = session
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload.setdefault(key, value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Readable during development, with the request id when there is one."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def __init__(self, *, colour: bool = True) -> None:
        super().__init__()
        self.colour = colour and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        tag = f"{self.COLORS.get(level, '')}{level:<7}{self.RESET}" if self.colour else f"{level:<7}"
        rid = getattr(record, "request_id", "-")
        prefix = f"{tag} " if rid == "-" else f"{tag} [{rid[:8]}] "
        line = f"{prefix}{record.name.removeprefix('vectora.'):<22} {record.getMessage()}"
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _RESERVED and not k.startswith("_") and k not in {"request_id", "session_id"}
        }
        if extras:
            line += "  " + " ".join(f"{k}={v}" for k, v in extras.items())
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


_configured = False


def configure_logging(settings: Settings) -> None:
    """Install handlers once per process. Safe to call repeatedly."""
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        JsonFormatter() if settings.log_format == "json" else ConsoleFormatter()
    )
    handler.addFilter(ContextFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level)

    # These libraries are chatty at INFO and say nothing we need.
    for noisy, level in {
        "httpx": "WARNING",
        "httpcore": "WARNING",
        "openai": "WARNING",
        "urllib3": "WARNING",
        "qdrant_client": "WARNING",
        "sentence_transformers": "WARNING",
        "transformers": "ERROR",
        "pdfminer": "ERROR",
        "PIL": "WARNING",
    }.items():
        logging.getLogger(noisy).setLevel(level)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@contextmanager
def request_context(request_id: str | None = None, session_id: str | None = None) -> Iterator[str]:
    """Bind a request id (and optionally a session) for the enclosed block."""
    rid = request_id or uuid.uuid4().hex
    rid_token = request_id_var.set(rid)
    sid_token = session_id_var.set(session_id or "-")
    try:
        yield rid
    finally:
        request_id_var.reset(rid_token)
        session_id_var.reset(sid_token)


class Stopwatch:
    """Accumulates per-stage timings for the diagnostics block."""

    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            self.timings[name] = round(self.timings.get(name, 0.0) + elapsed, 1)

    @property
    def total_ms(self) -> float:
        return round(sum(self.timings.values()), 1)


# ── Metrics ───────────────────────────────────────────────────────────────
# prometheus_client is optional: if it is not installed the counters become
# no-ops so the app still runs.

try:  # pragma: no cover
    from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest

    REGISTRY = CollectorRegistry()

    QUESTIONS = Counter(
        "vectora_questions_total",
        "Questions answered",
        ["mode", "query_type", "outcome"],
        registry=REGISTRY,
    )
    LLM_CALLS = Counter(
        "vectora_llm_calls_total",
        "OpenAI calls",
        ["purpose", "model", "outcome"],
        registry=REGISTRY,
    )
    LLM_COST = Counter(
        "vectora_llm_cost_usd_total",
        "Estimated OpenAI spend",
        ["purpose", "model"],
        registry=REGISTRY,
    )
    STAGE_SECONDS = Histogram(
        "vectora_stage_seconds",
        "Time per pipeline stage",
        ["stage"],
        buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32),
        registry=REGISTRY,
    )
    RETRIEVED = Histogram(
        "vectora_chunks_returned",
        "Chunks returned to the answer stage",
        buckets=(0, 1, 2, 3, 4, 6, 8, 12, 20),
        registry=REGISTRY,
    )
    METRICS_AVAILABLE = True

    def render_metrics() -> bytes:
        return generate_latest(REGISTRY)

except ImportError:  # pragma: no cover

    class _Noop:
        def labels(self, *_: object, **__: object) -> _Noop:
            return self

        def inc(self, *_: object, **__: object) -> None:
            return None

        def observe(self, *_: object, **__: object) -> None:
            return None

    QUESTIONS = LLM_CALLS = LLM_COST = STAGE_SECONDS = RETRIEVED = _Noop()  # type: ignore[assignment]
    METRICS_AVAILABLE = False

    def render_metrics() -> bytes:
        return b"# prometheus_client not installed\n"
