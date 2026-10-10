"""
Backwards-compatible entry point: ``python app.py`` still starts the server.

The real application lives in :mod:`vectora.api.app`; the CLI (``vectora
serve``) is the preferred way in, and in production uvicorn should be
pointed at ``vectora.api.app:app`` directly.
"""

from __future__ import annotations

if __name__ == "__main__":
    import uvicorn

    from vectora.config import get_settings
    from vectora.observability import configure_logging

    settings = get_settings()
    configure_logging(settings)
    print(f"\n  Vectora on http://{settings.host}:{settings.port}")
    print(f"  API docs at http://{settings.host}:{settings.port}/docs\n")
    uvicorn.run(
        "vectora.api.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_config=None,
        access_log=False,
    )
