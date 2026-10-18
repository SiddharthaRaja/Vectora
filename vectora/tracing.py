"""
OpenTelemetry tracing.

The request id already ties a question's log lines together, which answers
"what happened". Spans answer "where did the time go", which is a different
question and the one you need when a p95 is bad: a flat 9-second answer
tells you nothing, while a trace showing 6 of those seconds in a serial
HyDE call tells you exactly what to fix.

This is the piece that makes the Vantage benchmark numbers *explainable*
rather than merely observable — a latency distribution says a question took
8 seconds, a trace says which stage owned it.

Design notes:

* **Optional, and genuinely optional.** If ``opentelemetry`` is not
  installed, or tracing is off, every function here becomes a no-op with no
  import error and no cost. Instrumentation that forces a dependency on
  everyone who does not want it is instrumentation people rip out.
* **One span per pipeline stage**, nested under one span per question, with
  the same ``request_id`` as an attribute. That makes a trace and the logs
  joinable.
* **Attributes are measurements, not payloads.** Candidate counts, scores,
  model names, token counts — never the question text or the retrieved
  passages. A trace backend is not where user content should end up, and a
  corpus of internal documents leaking into a vendor's tracing UI is a real
  way to cause harm with a convenience feature.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

from vectora.config import Settings
from vectora.observability import get_logger, request_id_var

log = get_logger(__name__)

_tracer: Any = None
_enabled = False
_configured = False

SPAN_PREFIX = "vectora"


def configure_tracing(settings: Settings) -> bool:
    """Set up the tracer once. Returns whether tracing is active.

    Honours the standard ``OTEL_EXPORTER_OTLP_ENDPOINT`` environment
    variable, so this plugs into an existing collector without
    Vectora-specific configuration. With tracing enabled but no endpoint
    set, spans go to the console exporter, which is what you want while
    developing and nowhere near what you want in production — so that case
    logs a warning.
    """
    global _tracer, _enabled, _configured
    if _configured:
        return _enabled
    _configured = True

    if not getattr(settings, "enable_tracing", False):
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        log.warning(
            "tracing is enabled but opentelemetry is not installed; "
            "pip install 'vectora[tracing]'"
        )
        return False

    from vectora import __version__

    resource = Resource.create(
        {
            "service.name": settings.otel_service_name,
            "service.version": __version__,
            "deployment.environment": settings.env,
        }
    )
    provider = TracerProvider(resource=resource)

    endpoint = settings.otel_endpoint
    exporter = None
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            exporter = OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")
            log.info("tracing to otlp", extra={"endpoint": endpoint})
        except ImportError:
            log.warning("otlp exporter not installed; falling back to console")

    if exporter is None:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        exporter = ConsoleSpanExporter()
        log.warning(
            "tracing to the console; set OTEL_EXPORTER_OTLP_ENDPOINT for a collector"
        )

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(SPAN_PREFIX, __version__)
    _enabled = True
    log.info("tracing enabled", extra={"service": settings.otel_service_name})
    return True


def is_enabled() -> bool:
    return _enabled and _tracer is not None


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """A span, or a no-op when tracing is off.

    Used as the single instrumentation primitive so call sites look the same
    whether or not tracing is configured::

        with span("retrieval.dense", candidates=40) as s:
            ...
            set_attributes(s, hits=len(results))
    """
    if not is_enabled():
        yield None
        return

    full_name = name if name.startswith(SPAN_PREFIX) else f"{SPAN_PREFIX}.{name}"
    with _tracer.start_as_current_span(full_name) as current:
        request_id = request_id_var.get()
        if request_id and request_id != "-":
            current.set_attribute("vectora.request_id", request_id)
        for key, value in attributes.items():
            _set(current, key, value)
        try:
            yield current
        except Exception as exc:
            # Record the failure on the span, then re-raise: swallowing here
            # would turn an instrumentation layer into a bug hider.
            current.record_exception(exc)
            try:
                from opentelemetry.trace import Status, StatusCode

                current.set_status(Status(StatusCode.ERROR, str(exc)[:200]))
            except Exception:
                pass
            raise


def set_attributes(current: Any, **attributes: Any) -> None:
    """Add attributes to a span, tolerating ``None`` when tracing is off."""
    if current is None:
        return
    for key, value in attributes.items():
        _set(current, key, value)


def _set(current: Any, key: str, value: Any) -> None:
    """Set one attribute, coercing to a type OTLP accepts.

    Silently skips anything unrepresentable rather than raising: a span
    attribute is never worth failing a request over.
    """
    if value is None:
        return
    name = key if key.startswith("vectora.") else f"vectora.{key}"
    try:
        if isinstance(value, bool | int | float | str):
            current.set_attribute(name, value)
        elif isinstance(value, list | tuple):
            current.set_attribute(name, [str(v) for v in value][:32])
        else:
            current.set_attribute(name, str(value)[:256])
    except Exception:
        pass


def question_span(mode: str, query_type: str, corpus: str) -> Any:
    """The root span for one question. Every stage nests under it."""
    if not is_enabled():
        return nullcontext(None)
    return span("question", mode=mode, query_type=query_type, corpus=corpus)


def shutdown_tracing() -> None:
    """Flush pending spans. Called on application shutdown.

    Without this a short-lived process — the CLI, or a container stopping —
    exits before the batch processor has sent anything, and the traces for
    exactly the run you were debugging are the ones that go missing.
    """
    if not _enabled:
        return
    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush(timeout_millis=4000)
        if hasattr(provider, "shutdown"):
            provider.shutdown()
        log.info("tracing flushed")
    except Exception as exc:
        log.warning("tracing shutdown failed", extra={"err": str(exc)[:120]})
