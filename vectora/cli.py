"""
Vectora's command line.

One entry point for every operation, so there is no longer a separate
script per stage each with its own flags and its own hardcoded defaults:

    vectora doctor                 check the environment and say what is wrong
    vectora ingest                 PDFs  -> chunks.jsonl
    vectora repair                 re-clean an existing chunks.jsonl in place
    vectora index                  chunks.jsonl -> Qdrant + BM25
    vectora search "..."           retrieval only, with per-stage scores
    vectora ask "..."              one full answer
    vectora chat                   interactive session
    vectora serve                  run the web app
    vectora stats                  what is indexed, what it cost
    vectora eval                   score retrieval against the golden set
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import typer

from vectora import __version__
from vectora.config import SearchMode, get_settings
from vectora.observability import configure_logging, get_logger, request_context

app = typer.Typer(
    name="vectora",
    help="Retrieval-augmented question answering over research papers.",
    no_args_is_help=False,
    add_completion=False,
)
log = get_logger(__name__)


def _settings(log_level: str | None = None):
    settings = get_settings()
    if log_level:
        object.__setattr__(settings, "log_level", log_level.upper())
    configure_logging(settings)
    return settings


def _echo(text: str = "", **kwargs: object) -> None:
    typer.echo(text, **kwargs)  # type: ignore[arg-type]


def _ok(text: str) -> None:
    _echo(typer.style(f"  ok    {text}", fg=typer.colors.GREEN))


def _warn(text: str) -> None:
    _echo(typer.style(f"  warn  {text}", fg=typer.colors.YELLOW))


def _fail(text: str) -> None:
    _echo(typer.style(f"  fail  {text}", fg=typer.colors.RED))


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Print the version and exit."),
) -> None:
    """``--version`` needs ``invoke_without_command``, or Typer's
    ``no_args_is_help`` rejects it as a missing command."""
    if version:
        _echo(f"vectora {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        _echo(ctx.get_help())
        raise typer.Exit()


# ── doctor ────────────────────────────────────────────────────────────────


@app.command()
def doctor() -> None:
    """Check configuration and dependencies, and say how to fix what is wrong."""
    settings = _settings()
    _echo(f"\nVectora {__version__}  ({settings.env})\n")
    problems = 0

    _echo("Configuration")
    if settings.has_openai:
        _ok("OPENAI_API_KEY is set")
    else:
        _fail("OPENAI_API_KEY is missing — put OPENAI_API_KEY=sk-... in .env")
        problems += 1
    target = settings.qdrant_path or settings.qdrant_url
    _echo(f"  info  vector store: {target}")
    _echo(f"  info  collection:   {settings.qdrant_collection}")
    _echo(f"  info  embedding:    {settings.embedding_model} ({settings.embedding_dim}d)")
    _echo(f"  info  device:       {settings.resolved_device}")
    _echo(f"  info  answer model: {settings.answer_model}")

    _echo("\nPython packages")
    for module, hint in [
        ("fastapi", "pip install -e ."),
        ("qdrant_client", "pip install -e ."),
        ("sentence_transformers", "pip install -e ."),
        ("rank_bm25", "pip install -e ."),
        ("openai", "pip install -e ."),
        ("pdfplumber", "pip install -e '.[ingest]'"),
        ("fitz", "pip install -e '.[ingest]'"),
        ("weasyprint", "pip install -e '.[pdf]'  (optional, PDF export)"),
        ("prometheus_client", "pip install -e '.[metrics]'  (optional)"),
    ]:
        try:
            __import__(module)
            _ok(module)
        except ImportError:
            if "optional" in hint:
                _warn(f"{module} not installed — {hint}")
            else:
                _fail(f"{module} not installed — {hint}")
                problems += 1

    _echo("\nData")
    if settings.chunks_path.exists():
        with open(settings.chunks_path, encoding="utf-8") as fh:
            lines = sum(1 for _ in fh)
        _ok(f"chunks.jsonl present ({lines} chunks)")
    else:
        _warn(f"no chunk file at {settings.chunks_path} — run `vectora ingest`")
    if settings.papers_dir.exists():
        n = len(list(settings.papers_dir.rglob("*.pdf")))
        _ok(f"papers/ present ({n} PDFs)")
    else:
        _warn("no papers/ directory — needed only to re-ingest from source")

    _echo("\nServices")
    from vectora.store import QdrantStore, SparseIndex

    store = QdrantStore(settings)
    if store.healthy():
        _ok(f"Qdrant reachable at {target}")
        if store.exists():
            stats = store.stats()
            _ok(f"collection has {stats.vectors} vectors from {stats.papers} papers")
            if stats.dimensions != settings.embedding_dim:
                _fail(
                    f"dimension mismatch: collection {stats.dimensions}d, "
                    f"model {settings.embedding_dim}d — re-index with --recreate"
                )
                problems += 1
        else:
            _warn(f"collection '{settings.qdrant_collection}' not built — run `vectora index`")
    else:
        _fail(
            "Qdrant unreachable. Start it with "
            "`docker run -p 6333:6333 qdrant/qdrant`, or set "
            "VECTORA_QDRANT_PATH=./data/qdrant to use the embedded engine."
        )
        problems += 1

    sparse = SparseIndex(settings)
    if sparse.load():
        _ok(f"BM25 index present ({sparse.size} chunks)")
    else:
        _warn("no BM25 index — retrieval will be dense-only until `vectora index`")

    _echo()
    if problems:
        _echo(typer.style(f"{problems} problem(s) to fix before answering.", fg=typer.colors.RED))
        raise typer.Exit(1)
    _echo(typer.style("Ready.", fg=typer.colors.GREEN))


# ── ingest ────────────────────────────────────────────────────────────────


@app.command()
def ingest(
    papers: Path | None = typer.Option(None, "--papers", help="Folder of PDFs."),
    output: Path | None = typer.Option(None, "--output", help="Target JSONL."),
    force: bool = typer.Option(False, "--force", help="Re-process every paper."),
    no_semantic: bool = typer.Option(
        False, "--no-semantic", help="Skip embedding-based chunking (much faster)."
    ),
    limit: int | None = typer.Option(None, "--limit", help="Only the first N PDFs."),
) -> None:
    """Convert PDFs into chunks.jsonl. Incremental by default."""
    settings = _settings()
    from vectora.ingestion import run_ingestion

    try:
        report = run_ingestion(
            settings,
            papers_dir=papers,
            output=output,
            force=force,
            semantic=not no_semantic,
            limit=limit,
        )
    except FileNotFoundError as exc:
        _fail(str(exc))
        raise typer.Exit(1) from exc
    _echo(report.summary())
    if report.chunks_emitted:
        _echo("Next: vectora index\n")


@app.command()
def repair(
    source: Path | None = typer.Option(None, "--source", help="JSONL to repair."),
    output: Path | None = typer.Option(
        None, "--output", help="Where to write (defaults to in place)."
    ),
    backup: bool = typer.Option(True, "--backup/--no-backup", help="Keep a .bak copy."),
) -> None:
    """Re-clean an existing chunks.jsonl without the source PDFs.

    Use when the corpus was built by an older version of the pipeline and
    the PDFs are gone. Re-ingesting from source is strictly better.
    """
    settings = _settings()
    from vectora.ingestion import repair_chunks

    src = source or settings.chunks_path
    if backup and src.exists() and (output is None or output == src):
        backup_path = src.with_suffix(src.suffix + ".bak")
        backup_path.write_bytes(src.read_bytes())
        _echo(f"Backup written to {backup_path}")

    try:
        report = repair_chunks(settings, source=source, output=output)
    except FileNotFoundError as exc:
        _fail(str(exc))
        raise typer.Exit(1) from exc
    _echo(report.summary())
    _echo("Next: vectora index --recreate\n")


# ── index ─────────────────────────────────────────────────────────────────


@app.command()
def index(
    source: Path | None = typer.Option(None, "--source", help="chunks.jsonl to index."),
    recreate: bool = typer.Option(
        False, "--recreate", help="Drop and rebuild the collection."
    ),
    only_new: bool = typer.Option(
        False, "--only-new", help="Skip papers already in the collection."
    ),
    skip_sparse: bool = typer.Option(False, "--skip-sparse", help="Do not rebuild BM25."),
) -> None:
    """Embed chunks into Qdrant and build the BM25 index.

    Safe to re-run: point ids are derived from content, so an upsert
    updates in place rather than duplicating or overwriting.
    """
    settings = _settings()
    from vectora.embeddings import Encoder
    from vectora.ingestion import read_chunks
    from vectora.store import QdrantStore, embed_and_store

    path = source or settings.chunks_path
    if not path.exists():
        _fail(f"No chunk file at {path}. Run `vectora ingest` first.")
        raise typer.Exit(1)

    chunks = list(read_chunks(path))
    if not chunks:
        _fail("The chunk file is empty.")
        raise typer.Exit(1)
    _echo(f"Loaded {len(chunks)} chunks from {path}")

    store = QdrantStore(settings)
    if not store.healthy():
        _fail(
            f"Cannot reach Qdrant at {settings.qdrant_path or settings.qdrant_url}. "
            "Start it, or set VECTORA_QDRANT_PATH for the embedded engine."
        )
        raise typer.Exit(1)
    store.ensure_collection(recreate=recreate)

    if only_new and not recreate:
        already = store.indexed_paper_ids()
        before = len(chunks)
        chunks = [c for c in chunks if c.paper_id not in already]
        _echo(f"{before - len(chunks)} chunks belong to papers already indexed; skipping")
        if not chunks:
            _echo("Nothing new to embed.")
            if not skip_sparse:
                _rebuild_sparse(settings, store)
            raise typer.Exit()

    encoder = Encoder(settings)
    stored = embed_and_store(chunks, store, encoder, progress=True)
    stats = store.stats()
    _echo(f"\nUpserted {stored} points.  {stats.summary()}")

    if not skip_sparse:
        _rebuild_sparse(settings, store)
    _echo("\nNext: vectora serve\n")


def _rebuild_sparse(settings, store) -> None:
    from vectora.store import SparseIndex

    _echo("Building the BM25 index…")
    sparse = SparseIndex(settings)
    sparse.invalidate()
    count = sparse.build(store.iter_all())
    if count:
        sparse.save()
        _echo(f"BM25 index built over {count} chunks.")
    else:
        _warn("BM25 index came out empty; retrieval will be dense-only.")


# ── query ─────────────────────────────────────────────────────────────────


@app.command()
def search(
    query: str = typer.Argument(..., help="What to search for."),
    mode: SearchMode = typer.Option(SearchMode.BALANCED, "--mode", "-m"),
    top_k: int | None = typer.Option(None, "--top-k", "-k"),
    year: str | None = typer.Option(None, "--year"),
    paper_id: str | None = typer.Option(None, "--paper-id"),
    show_content: bool = typer.Option(True, "--content/--no-content"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Retrieve chunks without generating an answer.

    The first thing to reach for when an answer looks wrong: it separates a
    retrieval problem from a prompting one, and in `fast` mode costs nothing.
    """
    asyncio.run(_search(query, mode, top_k, year, paper_id, show_content, as_json))


async def _search(query, mode, top_k, year, paper_id, show_content, as_json) -> None:
    settings = _settings()
    from vectora.api.deps import build_container
    from vectora.generation.router import classify

    container = await asyncio.to_thread(build_container, settings, warm=False)
    try:
        with request_context():
            filters = {k: v for k, v in (("year", year), ("paper_id", paper_id)) if v}
            query_type, routed_by = await classify(
                query, llm=container.llm, settings=settings
            )
            chunks, diag = await container.engine.retriever.retrieve(
                query, mode=mode, query_type=query_type, top_k=top_k, filters=filters or None
            )

        if as_json:
            _echo(
                json.dumps(
                    {
                        "query": query,
                        "query_type": query_type.value,
                        "results": [
                            {
                                "score": rc.score,
                                "paper_id": rc.chunk.paper_id,
                                "title": rc.chunk.title,
                                "section": rc.chunk.section,
                                "page": rc.chunk.page,
                                "content": rc.chunk.content,
                            }
                            for rc in chunks
                        ],
                        "diagnostics": diag.model_dump(mode="json"),
                    },
                    indent=2,
                )
            )
            return

        _echo(f"\n{'=' * 74}")
        _echo(f"  {query}")
        _echo(
            f"  {mode.value} · {query_type.value} (by {routed_by}) · "
            f"{len(chunks)} results · {diag.elapsed_ms.get('total', 0):.0f}ms"
        )
        _echo("=" * 74)
        if diag.variants:
            _echo("\n  query variants:")
            for v in diag.variants:
                _echo(f"    · {v}")
        _echo(
            f"\n  dense {diag.dense_hits} · sparse {diag.sparse_hits} · "
            f"fused {diag.fused_pool} · reranked {diag.reranked} · mmr {diag.diversified}"
        )
        if diag.degraded:
            _warn("degraded: " + ", ".join(diag.degraded))

        for i, rc in enumerate(chunks, 1):
            c = rc.chunk
            _echo(f"\n  [{i}] {typer.style(c.title[:62] or c.paper_id, bold=True)}")
            _echo(
                f"      {c.section or '?'} · p.{c.page or '?'} · {c.chunk_type} · "
                f"{c.year or 'n.d.'} · quality {c.quality}"
            )
            parts = [f"final {rc.score:.4f}", f"rrf {rc.fused_score:.5f}"]
            if rc.dense_score is not None:
                parts.append(f"dense {rc.dense_score:.3f}@{rc.dense_rank}")
            if rc.sparse_score is not None:
                parts.append(f"bm25 {rc.sparse_score:.2f}@{rc.sparse_rank}")
            if rc.rerank_score is not None:
                parts.append(f"rerank {rc.rerank_score:+.2f}")
            _echo(f"      {' · '.join(parts)}")
            if rc.matched_variants:
                _echo(f"      found by: {', '.join(rc.matched_variants)}")
            if show_content:
                body = c.content[:320].replace("\n", " ")
                _echo(f"      {body}{'…' if len(c.content) > 320 else ''}")
        _echo()
    finally:
        await container.aclose()


@app.command()
def ask(
    query: str = typer.Argument(..., help="The question."),
    mode: SearchMode = typer.Option(SearchMode.BALANCED, "--mode", "-m"),
    year: str | None = typer.Option(None, "--year"),
    paper_id: str | None = typer.Option(None, "--paper-id"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Ask one question and print the cited answer."""
    asyncio.run(_ask(query, mode, year, paper_id, as_json))


async def _ask(query, mode, year, paper_id, as_json) -> None:
    settings = _settings()
    from vectora.api.deps import build_container

    container = await asyncio.to_thread(build_container, settings, warm=False)
    try:
        if container.llm is None:
            _fail("No OPENAI_API_KEY set, so no answer can be generated.")
            raise typer.Exit(1)
        filters = {k: v for k, v in (("year", year), ("paper_id", paper_id)) if v}
        with request_context():
            answer = await container.engine.answer(
                query, mode=mode, filters=filters or None
            )
        if as_json:
            _echo(json.dumps(answer.wire(), indent=2, default=str))
            return
        _render_answer(answer)
    finally:
        await container.aclose()


def _render_answer(answer) -> None:
    _echo(f"\n{'=' * 74}")
    _echo(f"  {answer.query}")
    _echo(f"  {answer.mode.value} · {answer.query_type.value}")
    _echo("=" * 74)
    _echo(f"\n{answer.answer}\n")

    if answer.fallback:
        if answer.fallback.suggestions:
            _echo("  Try asking instead:")
            for s in answer.fallback.suggestions:
                _echo(f"    · {s}")
        if answer.fallback.related_topics:
            _echo("\n  The collection does cover:")
            for t in answer.fallback.related_topics[:5]:
                _echo(f"    · {t}")
        _echo()
        return

    if answer.grounding:
        colour = (
            typer.colors.GREEN
            if answer.grounding.score >= 85
            else typer.colors.YELLOW
            if answer.grounding.score >= 60
            else typer.colors.RED
        )
        _echo(typer.style(f"  {answer.grounding.badge()}", fg=colour), nl=False)
        _echo(f"  {answer.grounding.summary}")
        for claim in answer.grounding.unsupported_claims:
            _echo(typer.style(f"    unsupported: {claim}", fg=typer.colors.RED))

    if answer.citations:
        _echo(f"\n  {'─' * 70}")
        _echo("  References")
        for c in answer.citations:
            _echo(f"    [{c.index}] {c.format()}")
            if c.quote:
                _echo(f"         “{c.quote}”")

    u = answer.usage
    _echo(
        f"\n  {u.calls} calls · {u.total_tokens} tokens · ${u.cost_usd:.5f}"
        + (
            f" · {answer.diagnostics.elapsed_ms.get('total', 0):.0f}ms retrieval"
            if answer.diagnostics
            else ""
        )
    )
    _echo()


@app.command()
def chat(
    mode: SearchMode = typer.Option(SearchMode.BALANCED, "--mode", "-m"),
) -> None:
    """Interactive session with conversation memory."""
    asyncio.run(_chat(mode))


async def _chat(mode: SearchMode) -> None:
    settings = _settings()
    from vectora.api.deps import build_container

    container = await asyncio.to_thread(build_container, settings, warm=True)
    try:
        if container.llm is None:
            _fail("No OPENAI_API_KEY set.")
            raise typer.Exit(1)
        _echo(
            f"\nVectora {__version__} — {mode.value} mode.\n"
            "Commands: /mode fast|balanced|deep, /clear, /stats, /quit\n"
        )
        history: list[dict[str, str]] = []
        while True:
            try:
                line = input(typer.style("you  ", fg=typer.colors.CYAN)).strip()
            except (EOFError, KeyboardInterrupt):
                _echo()
                break
            if not line:
                continue
            if line.lower() in {"/quit", "/exit", "/q"}:
                break
            if line.lower() == "/clear":
                history.clear()
                _echo("  (history cleared)\n")
                continue
            if line.lower() == "/stats":
                _echo(json.dumps(container.llm.stats(), indent=2) + "\n")
                continue
            if line.lower().startswith("/mode"):
                parts = line.split()
                if len(parts) > 1 and parts[1] in {m.value for m in SearchMode}:
                    mode = SearchMode(parts[1])
                    _echo(f"  (mode: {mode.value})\n")
                else:
                    _echo("  usage: /mode fast|balanced|deep\n")
                continue

            with request_context():
                try:
                    answer = await container.engine.answer(
                        line, mode=mode, history=history
                    )
                except Exception as exc:
                    _fail(str(exc))
                    continue
            _render_answer(answer)
            history.append({"role": "user", "content": line})
            history.append({"role": "assistant", "content": answer.answer})
    finally:
        await container.aclose()


# ── ops ───────────────────────────────────────────────────────────────────


@app.command()
def serve(
    host: str | None = typer.Option(None, "--host"),
    port: int | None = typer.Option(None, "--port"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes."),
    workers: int = typer.Option(1, "--workers"),
) -> None:
    """Run the web application."""
    settings = _settings()
    import uvicorn

    bind_host = host or settings.host
    bind_port = port or settings.port
    _echo(f"\n  Vectora {__version__} on http://{bind_host}:{bind_port}")
    _echo(f"  API docs at http://{bind_host}:{bind_port}/docs\n")
    uvicorn.run(
        "vectora.api.app:app",
        host=bind_host,
        port=bind_port,
        reload=reload,
        workers=workers if not reload else 1,
        log_config=None,  # our own logging is already configured
        access_log=False,  # the correlation middleware logs requests
    )


@app.command()
def stats() -> None:
    """What is indexed, what is cached, what it has cost."""
    settings = _settings()
    from vectora.api.deps import build_container

    container = build_container(settings, warm=False)
    try:
        payload = {
            "corpus": container.store.stats().__dict__ if container.store.healthy() else {},
            "sparse_index": container.sparse.stats(),
            "cache": container.cache.stats(),
            "database": container.repository.stats(),
            "llm": container.llm.stats() if container.llm else {"available": False},
        }
        _echo(json.dumps(payload, indent=2, default=str))
    finally:
        asyncio.run(container.aclose())


@app.command(name="eval")
def run_eval(
    dataset: Path | None = typer.Option(None, "--dataset", help="Golden set YAML/JSON."),
    mode: SearchMode | None = typer.Option(None, "--mode", help="Only this mode."),
    output: Path | None = typer.Option(None, "--output", help="Write results as JSON."),
) -> None:
    """Score retrieval against the golden set."""
    settings = _settings()
    from eval.run_eval import run

    asyncio.run(run(settings, dataset=dataset, only_mode=mode, output=output))


@app.command()
def calibrate(
    dataset: Path | None = typer.Option(None, "--dataset", help="Golden set YAML/JSON."),
    mode: SearchMode = typer.Option(SearchMode.BALANCED, "--mode", "-m"),
) -> None:
    """Measure the score distributions and suggest the refusal thresholds.

    The floors that decide when Vectora declines to answer depend on your
    corpus and your models, so they should be measured rather than guessed.
    Nothing is written; the suggested values are printed for .env.
    """
    settings = _settings()
    from eval.calibrate import run as run_calibration

    asyncio.run(run_calibration(settings, dataset=dataset, mode=mode))


@app.command()
def prune() -> None:
    """Housekeeping: drop empty sessions and expired cache entries."""
    settings = _settings()
    from vectora.cache import Cache
    from vectora.db import ChatRepository

    repo = ChatRepository(settings.sqlite_path)
    sessions = repo.prune_empty_sessions()
    repo.close()
    cache = Cache(settings.cache_path, settings.cache_ttl_seconds)
    entries = cache.purge_expired()
    cache.close()
    _echo(f"Removed {sessions} empty session(s) and {entries} expired cache entr(y/ies).")


def entrypoint() -> None:
    """Console-script target."""
    try:
        app()
    except KeyboardInterrupt:
        _echo("\nInterrupted.")
        sys.exit(130)


if __name__ == "__main__":
    entrypoint()
