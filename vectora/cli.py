"""
Vectora's command line.

One entry point for every operation. Grouped by what you are doing:

  Setting up a corpus
    vectora corpus list|create|show|delete
    vectora fetch arxiv --preset retrieval --limit 200
    vectora ingest                  documents -> chunks.jsonl
    vectora repair                  re-clean an existing chunks.jsonl
    vectora index                   chunks.jsonl -> Qdrant + BM25

  Using it
    vectora ask "..."               one cited answer
    vectora chat                    interactive, with memory
    vectora search "..."            retrieval only, all stage scores
    vectora serve                   the web app
    vectora mcp                     serve retrieval as MCP tools

  Measuring it
    vectora eval                    recall@k, MRR, nDCG, refusal rate
    vectora ablate                  is each stage earning its latency?
    vectora ablate --sweep          tune the fusion weights
    vectora answers                 citation and grounding quality
    vectora calibrate               measure the refusal thresholds
    vectora golden generate         draft an evaluation set

  Checking it
    vectora doctor                  what is wrong and how to fix it
    vectora stats                   what is indexed, what it cost
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
    help="Retrieval-augmented question answering over your own documents.",
    no_args_is_help=False,
    add_completion=False,
)
corpus_app = typer.Typer(help="Create and inspect corpora.", no_args_is_help=True)
fetch_app = typer.Typer(help="Download documents to build a corpus.", no_args_is_help=True)
golden_app = typer.Typer(help="Build and inspect evaluation sets.", no_args_is_help=True)
app.add_typer(corpus_app, name="corpus")
app.add_typer(fetch_app, name="fetch")
app.add_typer(golden_app, name="golden")

log = get_logger(__name__)

#: Set by the root callback so every command sees the same corpus.
_CORPUS: str | None = None


def _settings(log_level: str | None = None):
    settings = get_settings()
    if log_level:
        object.__setattr__(settings, "log_level", log_level.upper())
    if _CORPUS:
        object.__setattr__(settings, "corpus", _CORPUS)
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


def _ctx(settings, name: str | None = None):
    """Resolve the active corpus, reporting a missing one readably."""
    from vectora.corpus import CorpusContext

    try:
        return CorpusContext.build(settings, name)
    except KeyError as exc:
        _fail(str(exc).strip("\"'"))
        raise typer.Exit(1) from exc


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Print the version and exit."),
    corpus: str | None = typer.Option(
        None, "--corpus", "-c", help="Which corpus to act on. Default: the configured one."
    ),
) -> None:
    """``--version`` needs ``invoke_without_command``, or Typer treats it as
    a missing command."""
    global _CORPUS
    _CORPUS = corpus
    if version:
        _echo(f"vectora {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        _echo(ctx.get_help())
        raise typer.Exit()


# ══ corpus ════════════════════════════════════════════════════════════════


@corpus_app.command("list")
def corpus_list() -> None:
    """List the corpora and what each contains."""
    settings = _settings()
    from vectora.corpus import CorpusRegistry
    from vectora.store import QdrantStore

    registry = CorpusRegistry(settings.corpora_dir)
    profiles = registry.list()
    if not profiles:
        _echo("\nNo corpora yet.\n")
        _echo("  vectora corpus create mycorpus --kind 'research papers'")
        _echo("  vectora fetch arxiv --preset retrieval --limit 100 --corpus mycorpus\n")
        return

    _echo(f"\n  {'corpus':<18} {'docs':>6} {'chunks':>8} {'indexed':>8}  description")
    _echo("  " + "-" * 86)
    for profile in profiles:
        registry.refresh_counts(profile)
        indexed = "—"
        try:
            store = QdrantStore(settings, collection=profile.collection)
            if store.healthy() and store.exists():
                indexed = str(store.stats().vectors)
            store.close()
        except Exception:
            pass
        marker = "*" if profile.name == settings.corpus else " "
        _echo(
            f"{marker} {profile.name:<18} {profile.documents:>6} "
            f"{profile.chunks:>8} {indexed:>8}  {profile.description[:40]}"
        )
    _echo("\n  * is the active corpus. Change it with --corpus or VECTORA_CORPUS.\n")


@corpus_app.command("create")
def corpus_create(
    name: str = typer.Argument(..., help="Short name, e.g. 'ml-papers'."),
    description: str = typer.Option("", "--description", "-d"),
    kind: str = typer.Option(
        "documents", "--kind", help="What one item is: 'research papers', 'RFCs'…"
    ),
    terms: str | None = typer.Option(
        None, "--terms", help="Comma-separated domain vocabulary for query expansion."
    ),
) -> None:
    """Create an empty corpus.

    ``--kind`` and ``--terms`` are not decoration: they go into the query
    expansion prompts, which is how the same code serves papers, RFCs or
    runbooks without a prompt naming one field.
    """
    settings = _settings()
    from vectora.corpus import CorpusRegistry

    registry = CorpusRegistry(settings.corpora_dir)
    try:
        profile = registry.create(
            name,
            description=description,
            document_kind=kind,
            domain_terms=[t.strip() for t in (terms or "").split(",") if t.strip()],
        )
    except (FileExistsError, ValueError) as exc:
        _fail(str(exc))
        raise typer.Exit(1) from exc

    _echo(f"\nCreated corpus '{profile.name}'")
    _echo(f"  documents   {profile.papers_dir}")
    _echo(f"  chunks      {profile.chunks_path}")
    _echo(f"  collection  {profile.collection}")
    _echo(f"  golden set  {profile.golden_path}")
    _echo("\nNext: add documents to the documents folder, or")
    _echo(f"  vectora fetch arxiv --preset nlp --corpus {profile.name}\n")


@corpus_app.command("show")
def corpus_show(name: str | None = typer.Argument(None)) -> None:
    """Show one corpus in detail."""
    settings = _settings()
    ctx = _ctx(settings, name)
    profile = ctx.profile
    from vectora.corpus import CorpusRegistry

    CorpusRegistry(settings.corpora_dir).refresh_counts(profile)

    _echo(f"\n  {profile.name}")
    _echo(f"  {'-' * (len(profile.name))}")
    if profile.description:
        _echo(f"  {profile.description}")
    _echo(f"\n  kind         {profile.document_kind}")
    _echo(f"  documents    {profile.documents}")
    _echo(f"  chunks       {profile.chunks}")
    _echo(f"  collection   {ctx.collection}")
    _echo(f"  source       {profile.source or '—'}")
    _echo(f"  created      {profile.created_at[:10]}")
    if profile.domain_terms:
        _echo(f"  vocabulary   {', '.join(profile.domain_terms[:10])}")
    _echo(f"\n  documents at {ctx.papers}")
    _echo(f"  chunks at    {ctx.chunks}  {'✓' if profile.has_chunks else '(not built)'}")
    _echo(f"  lexical idx  {ctx.bm25}  {'✓' if ctx.bm25.exists() else '(not built)'}")
    _echo(f"  golden set   {ctx.golden}  {'✓' if profile.has_golden else '(none)'}")
    if profile.notes:
        _echo(f"\n  note: {profile.notes}")
    _echo()


@corpus_app.command("delete")
def corpus_delete(
    name: str = typer.Argument(...),
    everything: bool = typer.Option(
        False, "--everything", help="Also delete the source documents."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation."),
) -> None:
    """Delete a corpus's index and chunks. Keeps the documents by default."""
    settings = _settings()
    from vectora.corpus import CorpusRegistry
    from vectora.store import QdrantStore

    registry = CorpusRegistry(settings.corpora_dir)
    try:
        profile = registry.get(name)
    except KeyError as exc:
        _fail(str(exc).strip("\"'"))
        raise typer.Exit(1) from exc

    what = "the corpus and its documents" if everything else "its derived state"
    if not yes and not typer.confirm(f"Delete {what} for '{name}'?"):
        _echo("Cancelled.")
        raise typer.Exit()

    store = QdrantStore(settings, collection=profile.collection)
    if store.healthy() and store.exists():
        store.client.delete_collection(profile.collection)
        _echo(f"Dropped collection {profile.collection}")
    store.close()

    registry.delete(name, keep_documents=not everything)
    _echo(f"Deleted {what} for '{name}'.")
    if not everything:
        _echo(f"The documents are still at {profile.papers_dir}\n")


# ══ fetch ═════════════════════════════════════════════════════════════════


@fetch_app.command("presets")
def fetch_presets() -> None:
    """List the ready-made arXiv corpus definitions."""
    from vectora.fetch import PRESETS

    _echo("\n  Presets for `vectora fetch arxiv --preset <name>`:\n")
    for name, spec in PRESETS.items():
        _echo(f"  {typer.style(name, fg=typer.colors.CYAN)}")
        _echo(f"    {spec['description']}")
        _echo(f"    query: {spec['query']}\n")


@fetch_app.command("arxiv")
def fetch_arxiv(
    query: str | None = typer.Option(None, "--query", "-q", help="arXiv API query."),
    preset: str | None = typer.Option(
        None, "--preset", "-p", help="A ready-made subject. See `vectora fetch presets`."
    ),
    limit: int = typer.Option(50, "--limit", "-n", help="How many papers."),
    sort: str = typer.Option("relevance", "--sort", help="relevance | recent | updated."),
    create: bool = typer.Option(
        True, "--create/--no-create", help="Create the corpus if it does not exist."
    ),
) -> None:
    """Download papers from arXiv into a corpus.

    Needs network access to export.arxiv.org. Rate-limited to arXiv's stated
    guidance and resumable, so re-running only fetches what is missing.
    """
    settings = _settings()
    from vectora.corpus import CorpusRegistry
    from vectora.fetch import PRESETS, fetch_arxiv_corpus

    if not query and not preset:
        _fail("Pass --query or --preset. See `vectora fetch presets`.")
        raise typer.Exit(1)
    if preset and preset not in PRESETS:
        _fail(f"Unknown preset {preset!r}. Available: {', '.join(PRESETS)}")
        raise typer.Exit(1)

    spec = PRESETS.get(preset or "", {})
    effective_query = query or spec["query"]
    name = _CORPUS or (preset if preset else settings.corpus)

    registry = CorpusRegistry(settings.corpora_dir)
    if not registry.exists(name):
        if not create:
            _fail(f"No corpus {name!r}. Create it or pass --create.")
            raise typer.Exit(1)
        profile = registry.create(
            name,
            description=spec.get("description", f"Fetched from arXiv: {effective_query}"),
            document_kind=spec.get("document_kind", "research papers"),
            domain_terms=[t.strip() for t in spec.get("terms", "").split(",") if t.strip()],
            source=f"arxiv: {effective_query}",
        )
        _echo(f"Created corpus '{profile.name}'")
    else:
        profile = registry.get(name)

    profile.ensure_dirs()
    _echo(f"\nFetching up to {limit} papers into '{profile.name}'")
    _echo(f"  query: {effective_query}")
    _echo("  arXiv asks for one request every three seconds, so this is slow by design.\n")

    try:
        report = fetch_arxiv_corpus(
            effective_query, profile.papers_dir, limit=limit, sort=sort
        )
    except RuntimeError as exc:
        _fail(str(exc))
        raise typer.Exit(1) from exc

    _echo(report.summary())
    registry.refresh_counts(profile)
    if report.downloaded or report.skipped_existing:
        _echo(f"Next: vectora ingest --corpus {profile.name}\n")


# ══ ingestion ═════════════════════════════════════════════════════════════


@app.command()
def ingest(
    papers: Path | None = typer.Option(None, "--papers", help="Override the document folder."),
    output: Path | None = typer.Option(None, "--output", help="Override the target JSONL."),
    force: bool = typer.Option(False, "--force", help="Re-process every document."),
    no_semantic: bool = typer.Option(
        False, "--no-semantic", help="Skip embedding-based chunking (much faster)."
    ),
    limit: int | None = typer.Option(None, "--limit", help="Only the first N documents."),
) -> None:
    """Convert documents into chunks. Incremental by default.

    Reads PDFs, markdown, plain text and saved HTML. Where a document was
    fetched by Vectora, its metadata sidecar is used, so citations are exact
    rather than guessed from the first page.
    """
    settings = _settings()
    ctx = _ctx(settings)
    from vectora.corpus import CorpusRegistry
    from vectora.ingestion import run_ingestion

    _echo(f"\nIngesting corpus '{ctx.name}' from {papers or ctx.papers}\n")
    try:
        report = run_ingestion(
            settings,
            corpus=ctx,
            papers_dir=papers,
            output=output,
            force=force,
            semantic=not no_semantic,
            limit=limit,
        )
    except FileNotFoundError as exc:
        _fail(str(exc))
        _echo(
            f"\n  Put documents in {ctx.papers}, or fetch some:\n"
            f"    vectora fetch arxiv --preset nlp --corpus {ctx.name}\n"
        )
        raise typer.Exit(1) from exc

    _echo(report.summary())
    CorpusRegistry(settings.corpora_dir).refresh_counts(ctx.profile)
    if report.chunks_emitted:
        _echo(f"Next: vectora index --corpus {ctx.name}\n")


@app.command()
def repair(
    source: Path | None = typer.Option(None, "--source"),
    output: Path | None = typer.Option(None, "--output"),
    backup: bool = typer.Option(True, "--backup/--no-backup", help="Keep a .bak copy."),
) -> None:
    """Re-clean an existing chunk file without the source documents.

    For a corpus built by an older pipeline whose sources are gone.
    Re-ingesting is strictly better where the documents are available.
    """
    settings = _settings()
    ctx = _ctx(settings)
    from vectora.ingestion import repair_chunks

    src = source or ctx.chunks
    if backup and src.exists() and (output is None or output == src):
        backup_path = src.with_suffix(src.suffix + ".bak")
        backup_path.write_bytes(src.read_bytes())
        _echo(f"Backup written to {backup_path}")

    try:
        report = repair_chunks(settings, corpus=ctx, source=source, output=output)
    except FileNotFoundError as exc:
        _fail(str(exc))
        raise typer.Exit(1) from exc
    _echo(report.summary())
    _echo(f"Next: vectora index --corpus {ctx.name} --recreate\n")


@app.command()
def index(
    source: Path | None = typer.Option(None, "--source"),
    recreate: bool = typer.Option(False, "--recreate", help="Drop and rebuild."),
    only_new: bool = typer.Option(
        False, "--only-new", help="Skip documents already in the collection."
    ),
    skip_sparse: bool = typer.Option(False, "--skip-sparse", help="Do not rebuild BM25."),
) -> None:
    """Embed chunks into Qdrant and build the lexical index.

    Safe to re-run: point ids are derived from content, so an upsert updates
    in place rather than duplicating or overwriting.
    """
    settings = _settings()
    ctx = _ctx(settings)
    from vectora.embeddings import Encoder
    from vectora.ingestion import read_chunks
    from vectora.store import QdrantStore, embed_and_store

    path = source or ctx.chunks
    if not path.exists():
        _fail(f"No chunk file at {path}. Run `vectora ingest` first.")
        raise typer.Exit(1)

    chunks = list(read_chunks(path))
    if not chunks:
        _fail("The chunk file is empty.")
        raise typer.Exit(1)
    _echo(f"\nLoaded {len(chunks)} chunks for corpus '{ctx.name}' from {path}")

    store = QdrantStore(settings, collection=ctx.collection)
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
        _echo(f"{before - len(chunks)} chunks belong to documents already indexed; skipping")
        if not chunks:
            _echo("Nothing new to embed.")
            if not skip_sparse:
                _rebuild_sparse(settings, store, ctx)
            raise typer.Exit()

    encoder = Encoder(settings)
    stored = embed_and_store(chunks, store, encoder, progress=True)
    _echo(f"\nUpserted {stored} points.  {store.stats().summary()}")

    if not skip_sparse:
        _rebuild_sparse(settings, store, ctx)
    _echo(f"\nNext: vectora serve --corpus {ctx.name}\n")


def _rebuild_sparse(settings, store, ctx) -> None:
    from vectora.store import SparseIndex

    _echo("Building the lexical (BM25) index…")
    sparse = SparseIndex(settings, path=ctx.bm25)
    sparse.invalidate()
    count = sparse.build(store.iter_all())
    if count:
        sparse.save()
        _echo(f"Lexical index built over {count} chunks.")
    else:
        _warn("Lexical index came out empty; retrieval will be dense-only.")


# ══ query ═════════════════════════════════════════════════════════════════


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

    container = await asyncio.to_thread(
        build_container, settings, warm=False, corpus=_CORPUS
    )
    try:
        with request_context():
            filters = {k: v for k, v in (("year", year), ("paper_id", paper_id)) if v}
            query_type, routed_by = await classify(
                query, llm=container.llm, settings=settings
            )
            chunks, diag = await container.engine.retriever.retrieve(
                query, mode=mode, query_type=query_type, top_k=top_k,
                filters=filters or None,
            )

        if as_json:
            _echo(json.dumps({
                "corpus": container.corpus.name,
                "query": query,
                "query_type": query_type.value,
                "results": [
                    {
                        "score": rc.score, "paper_id": rc.chunk.paper_id,
                        "title": rc.chunk.title, "section": rc.chunk.section,
                        "page": rc.chunk.page, "content": rc.chunk.content,
                    }
                    for rc in chunks
                ],
                "diagnostics": diag.model_dump(mode="json"),
            }, indent=2))
            return

        _echo(f"\n{'=' * 74}")
        _echo(f"  {query}")
        _echo(
            f"  {container.corpus.name} · {mode.value} · {query_type.value} "
            f"(by {routed_by}) · {len(chunks)} results · "
            f"{diag.elapsed_ms.get('total', 0):.0f}ms"
        )
        _echo("=" * 74)
        if diag.variants:
            _echo("\n  query variants:")
            for v in diag.variants:
                _echo(f"    · {v}")
        _echo(
            f"\n  dense {diag.dense_hits} · lexical {diag.sparse_hits} · "
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

    container = await asyncio.to_thread(
        build_container, settings, warm=False, corpus=_CORPUS
    )
    try:
        if container.llm is None:
            _fail("No OPENAI_API_KEY set, so no answer can be generated.")
            _echo("  `vectora search` still works and needs no key.")
            raise typer.Exit(1)
        filters = {k: v for k, v in (("year", year), ("paper_id", paper_id)) if v}
        with request_context():
            answer = await container.engine.answer(
                query, mode=mode, filters=filters or None
            )
        if as_json:
            _echo(json.dumps(answer.wire(), indent=2, default=str))
            return
        _render_answer(answer, container.corpus.name)
    finally:
        await container.aclose()


def _render_answer(answer, corpus: str = "") -> None:
    _echo(f"\n{'=' * 74}")
    _echo(f"  {answer.query}")
    _echo(f"  {corpus} · {answer.mode.value} · {answer.query_type.value}")
    _echo("=" * 74)
    _echo(f"\n{answer.answer}\n")

    if answer.fallback:
        if answer.fallback.suggestions:
            _echo("  Try asking instead:")
            for s in answer.fallback.suggestions:
                _echo(f"    · {s}")
        if answer.fallback.related_topics:
            _echo("\n  The corpus does cover:")
            for t in answer.fallback.related_topics[:5]:
                _echo(f"    · {t}")
        _echo()
        return

    if answer.grounding:
        colour = (
            typer.colors.GREEN if answer.grounding.score >= 85
            else typer.colors.YELLOW if answer.grounding.score >= 60
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
            if answer.diagnostics else ""
        )
    )
    _echo()


@app.command()
def chat(mode: SearchMode = typer.Option(SearchMode.BALANCED, "--mode", "-m")) -> None:
    """Interactive session with conversation memory."""
    asyncio.run(_chat(mode))


async def _chat(mode: SearchMode) -> None:
    settings = _settings()
    from vectora.api.deps import build_container

    container = await asyncio.to_thread(
        build_container, settings, warm=True, corpus=_CORPUS
    )
    try:
        if container.llm is None:
            _fail("No OPENAI_API_KEY set.")
            raise typer.Exit(1)
        _echo(
            f"\nVectora {__version__} — corpus '{container.corpus.name}', "
            f"{mode.value} mode.\n"
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
                    answer = await container.engine.answer(line, mode=mode, history=history)
                except Exception as exc:
                    _fail(str(exc))
                    continue
            _render_answer(answer, container.corpus.name)
            history.append({"role": "user", "content": line})
            history.append({"role": "assistant", "content": answer.answer})
    finally:
        await container.aclose()


# ══ measurement ═══════════════════════════════════════════════════════════


@app.command(name="eval")
def run_eval(
    dataset: Path | None = typer.Option(None, "--dataset", help="Golden set override."),
    mode: SearchMode | None = typer.Option(None, "--mode", help="Only this mode."),
    output: Path | None = typer.Option(None, "--output", help="Write results as JSON."),
    no_llm: bool = typer.Option(
        False,
        "--no-llm",
        help="Retrieval only: never call the LLM. Rewrites/HyDE are skipped at once "
        "and routing uses keyword rules, so a slow or missing LLM cannot stall the run.",
    ),
) -> None:
    """Score retrieval against the golden set."""
    settings = _settings()
    if no_llm:
        from pydantic import SecretStr

        settings = settings.model_copy(update={"openai_api_key": SecretStr("")})
        _echo("--no-llm: LLM stages disabled (query expansion skipped, keyword routing).")
    from evaluation.run_eval import run

    asyncio.run(
        run(settings, dataset=dataset, only_mode=mode, output=output, corpus=_CORPUS)
    )


@app.command()
def ablate(
    mode: SearchMode = typer.Option(SearchMode.BALANCED, "--mode", "-m"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    metric: str = typer.Option("recall", "--metric", help="recall | mrr | ndcg | precision"),
    sweep: bool = typer.Option(
        False, "--sweep", help="Sweep the fusion weights instead of ablating stages."
    ),
    steps: int = typer.Option(5, "--steps", help="Sweep granularity."),
    output: Path | None = typer.Option(None, "--output"),
    no_llm: bool = typer.Option(
        False, "--no-llm",
        help="Never call the LLM; skips the expansion-only variants (CPU-friendly).",
    ),
) -> None:
    """Measure whether each retrieval stage earns its latency.

    Runs the golden set with one component removed at a time and reports the
    paired difference with a bootstrap confidence interval, so a difference
    inside the noise is labelled as such rather than quoted to three
    decimals.
    """
    settings = _settings()
    if no_llm:
        from pydantic import SecretStr

        settings = settings.model_copy(update={"openai_api_key": SecretStr("")})
        _echo("--no-llm: LLM stages disabled; expansion-only variants skipped.")
    from evaluation.ablate import run

    if metric not in {"recall", "mrr", "ndcg", "precision"}:
        _fail(f"Unknown metric {metric!r}.")
        raise typer.Exit(1)
    asyncio.run(
        run(settings, mode=mode, top_k=top_k, metric=metric, sweep=sweep,
            sweep_steps=steps, corpus=_CORPUS, output=output, no_llm=no_llm)
    )


@app.command()
def answers(
    mode: SearchMode = typer.Option(SearchMode.DEEP, "--mode", "-m"),
    limit: int | None = typer.Option(None, "--limit", help="Only the first N cases."),
    no_judge: bool = typer.Option(
        False, "--no-judge", help="Skip the model-judged scores; mechanical only."
    ),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Score the generated answers: citations, grounding, relevance.

    Spends a generation call per case, so use --limit while iterating.
    """
    settings = _settings()
    from evaluation.answers import run

    asyncio.run(
        run(settings, mode=mode, limit=limit, judge=not no_judge,
            corpus=_CORPUS, output=output)
    )


@app.command()
def calibrate(
    dataset: Path | None = typer.Option(None, "--dataset"),
    mode: SearchMode = typer.Option(SearchMode.BALANCED, "--mode", "-m"),
) -> None:
    """Measure the score distributions and suggest the refusal thresholds.

    The floors that decide when Vectora declines depend on your corpus and
    your models, so they should be measured rather than guessed. Nothing is
    written; the suggested values are printed for .env.
    """
    settings = _settings()
    from evaluation.calibrate import run

    asyncio.run(run(settings, dataset=dataset, mode=mode, corpus=_CORPUS))


@golden_app.command("generate")
def golden_generate(
    count: int = typer.Option(40, "--count", "-n", help="Questions to draft."),
    negatives: int = typer.Option(10, "--negatives", help="Off-corpus questions."),
    mode: SearchMode = typer.Option(SearchMode.BALANCED, "--mode", "-m"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Verification depth."),
    output: Path | None = typer.Option(None, "--output"),
    seed: int = typer.Option(7, "--seed"),
) -> None:
    """Draft an evaluation set from the indexed corpus, and verify it.

    Each question is generated from one passage and then checked by running
    it through real retrieval, so a case that cannot find its own source is
    flagged rather than quietly included. Review the output before trusting
    it.
    """
    settings = _settings()
    from evaluation.generate import generate

    asyncio.run(
        generate(settings, count=count, negatives=negatives, mode=mode,
                 top_k=top_k, output=output, corpus=_CORPUS, seed=seed)
    )


@golden_app.command("show")
def golden_show() -> None:
    """Summarise the active corpus's evaluation set."""
    settings = _settings()
    ctx = _ctx(settings)
    if not ctx.golden.exists():
        _warn(f"No golden set at {ctx.golden}")
        _echo("  Draft one:  vectora golden generate")
        _echo(f"  Or copy:    evaluation/golden.template.yaml -> {ctx.golden}\n")
        raise typer.Exit(1)

    from evaluation.run_eval import load_dataset

    cases, negatives = load_dataset(ctx.golden)
    _echo(f"\n  {ctx.golden}")
    _echo(f"  {len(cases)} cases, {len(negatives)} negatives\n")
    by_type: dict[str, int] = {}
    for case in cases:
        key = case.query_type.value if case.query_type else "(unset)"
        by_type[key] = by_type.get(key, 0) + 1
    for key, n in sorted(by_type.items(), key=lambda kv: -kv[1]):
        _echo(f"    {n:>3}  {key}")
    if len(cases) < 30:
        _echo(
            f"\n  {len(cases)} cases is on the small side. Differences under "
            f"~0.1 recall will not be distinguishable from noise."
        )
    _echo()


# ══ ops ═══════════════════════════════════════════════════════════════════


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
    if _CORPUS:
        # uvicorn re-imports the app in its own process, so the corpus has to
        # travel through the environment rather than this process's state.
        import os

        os.environ["VECTORA_CORPUS"] = _CORPUS
    _echo(f"\n  Vectora {__version__} on http://{bind_host}:{bind_port}")
    _echo(f"  corpus: {_CORPUS or settings.corpus}")
    _echo(f"  API docs at http://{bind_host}:{bind_port}/docs\n")
    uvicorn.run(
        "vectora.api.app:app",
        host=bind_host,
        port=bind_port,
        reload=reload,
        workers=workers if not reload else 1,
        log_config=None,
        access_log=False,
    )


@app.command()
def mcp() -> None:
    """Serve retrieval as MCP tools over stdio.

    Point an MCP client at this command. Protocol frames go to stdout, so
    logging is forced to stderr — do not add prints to this path.
    """
    settings = _settings()
    object.__setattr__(settings, "log_format", "console")
    configure_logging(settings)
    from vectora.mcp_server import serve_stdio

    try:
        asyncio.run(serve_stdio(settings))
    except KeyboardInterrupt:
        pass


@app.command()
def stats() -> None:
    """What is indexed, what is cached, what it has cost."""
    settings = _settings()
    from vectora.api.deps import build_container

    container = build_container(settings, warm=False, corpus=_CORPUS)
    try:
        payload = {
            "corpus": {
                "name": container.corpus.name,
                "collection": container.corpus.collection,
                "documents": container.corpus.profile.documents,
                "chunks": container.corpus.profile.chunks,
            },
            "index": container.store.stats().__dict__ if container.store.healthy() else {},
            "lexical_index": container.sparse.stats(),
            "cache": container.cache.stats(),
            "database": container.repository.stats(),
            "llm": container.llm.stats() if container.llm else {"available": False},
        }
        _echo(json.dumps(payload, indent=2, default=str))
    finally:
        asyncio.run(container.aclose())


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
        _echo("        (retrieval and `vectora search` work without it)")
        problems += 1
    target = settings.qdrant_path or settings.qdrant_url
    _echo(f"  info  vector store: {target}")
    _echo(f"  info  embedding:    {settings.embedding_model} ({settings.embedding_dim}d)")
    _echo(f"  info  device:       {settings.resolved_device}")
    _echo(f"  info  answer model: {settings.answer_model}")
    _echo(f"  info  tracing:      {'on' if settings.enable_tracing else 'off'}")

    _echo("\nPython packages")
    for module, hint in [
        ("fastapi", "pip install -e ."),
        ("qdrant_client", "pip install -e ."),
        ("sentence_transformers", "pip install -e ."),
        ("rank_bm25", "pip install -e ."),
        ("openai", "pip install -e ."),
        ("pdfplumber", "pip install -e '.[ingest]'"),
        ("fitz", "pip install -e '.[ingest]'"),
        ("yaml", "pip install -e '.[eval]'  (needed for the golden set)"),
        ("weasyprint", "pip install -e '.[pdf]'  (optional, PDF export)"),
        ("prometheus_client", "pip install -e '.[metrics]'  (optional)"),
        ("opentelemetry", "pip install -e '.[tracing]'  (optional)"),
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

    _echo("\nCorpora")
    from vectora.corpus import CorpusRegistry

    registry = CorpusRegistry(settings.corpora_dir)
    profiles = registry.list()
    if not profiles:
        _warn("no corpora yet — `vectora corpus create <name>`")
    for profile in profiles:
        registry.refresh_counts(profile)
        marker = "active" if profile.name == settings.corpus else "     "
        state = "ingested" if profile.has_chunks else "not ingested"
        _echo(
            f"  {marker} {profile.name:<16} {profile.documents:>4} docs  "
            f"{profile.chunks:>6} chunks  {state}"
        )

    ctx = None
    try:
        ctx = _ctx(settings)
    except typer.Exit:
        problems += 1

    if ctx is not None:
        _echo(f"\nActive corpus: {ctx.name}")
        if ctx.profile.has_chunks:
            _ok(f"chunks at {ctx.chunks}")
        else:
            _warn(f"no chunks at {ctx.chunks} — run `vectora ingest`")
        if ctx.papers.exists():
            from vectora.ingestion.sources import iter_documents

            _ok(f"{len(iter_documents(ctx.papers))} ingestible document(s) in {ctx.papers}")
        if ctx.golden.exists():
            _ok(f"golden set at {ctx.golden}")
        else:
            _warn("no golden set — `vectora golden generate` after indexing")

        _echo("\nServices")
        from vectora.store import QdrantStore, SparseIndex

        store = QdrantStore(settings, collection=ctx.collection)
        if store.healthy():
            _ok(f"Qdrant reachable at {target}")
            if store.exists():
                s = store.stats()
                _ok(f"collection '{ctx.collection}': {s.vectors} vectors, {s.papers} docs")
                if s.dimensions != settings.embedding_dim:
                    _fail(
                        f"dimension mismatch: collection {s.dimensions}d, "
                        f"model {settings.embedding_dim}d — `vectora index --recreate`"
                    )
                    problems += 1
            else:
                _warn(f"collection '{ctx.collection}' not built — `vectora index`")
        else:
            _fail(
                "Qdrant unreachable. Start it with "
                "`docker run -p 6333:6333 qdrant/qdrant`, or set "
                "VECTORA_QDRANT_PATH=./data/qdrant for the embedded engine."
            )
            problems += 1
        store.close()

        sparse = SparseIndex(settings, path=ctx.bm25)
        if sparse.load():
            _ok(f"lexical index present ({sparse.size} chunks)")
        else:
            _warn("no lexical index — retrieval is dense-only until `vectora index`")

    _echo()
    if problems:
        _echo(typer.style(f"{problems} problem(s) to fix.", fg=typer.colors.RED))
        raise typer.Exit(1)
    _echo(typer.style("Ready.", fg=typer.colors.GREEN))


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
