# Changelog

Notable changes per release. Dates are UTC.

## v3.0.0 — 2026-10-11

The corpus becomes a first-class object, and the system's claims become
measurements.

### Corpora are named and swappable

- A corpus is now a profile under `corpora/<name>/`, owning its documents,
  chunk file, lexical index, evaluation set and reports, plus its own Qdrant
  collection `vectora_<name>`. Switching is `--corpus`; rebuilding one
  cannot disturb another.
- The collection name is **derived** from the corpus rather than configured,
  so two corpora sharing one collection is now unrepresentable.
- `vectora corpus list|create|show|delete`.
- A pre-v3 checkout is adopted automatically: the top-level `chunks.jsonl`
  becomes the `default` corpus on first run, without moving anything.

### Nothing is hardcoded to one subject any more

- The query-expansion prompts previously said "biomedical and analytical
  chemistry papers". They now take the document kind and domain vocabulary
  from the corpus profile, so the same code serves papers, RFCs or runbooks
  without a prompt naming a field.
- The router's examples, the figure-captioning prompt and the comments are
  domain-neutral.
- The vitamin D golden set moved to `corpora/default/golden.yaml`, where it
  belongs; `evaluation/golden.template.yaml` is the starting point for a new one.

### Building a corpus

- `vectora fetch arxiv --preset retrieval --limit 200` downloads papers with
  their metadata. Rate-limited to arXiv's stated guidance and resumable.
  Six presets; `vectora fetch presets` lists them.
- Each fetched document gets a metadata sidecar which ingestion reads, so a
  fetched corpus **cites exactly** rather than guessing the title and
  authors from the first page.
- Markdown, plain text and HTML are ingestible, not only PDF. HTML metadata
  comes from `<title>` and the citation meta tags; scripts and styles never
  reach the corpus.

### Measurement

- `vectora ablate` runs the golden set with one stage removed at a time and
  reports the **paired difference with a bootstrap confidence interval**, so
  a difference inside the noise is labelled as such instead of being quoted
  to three decimals. Variants that change more than one thing are marked,
  because their effect cannot be attributed to a component.
- `vectora ablate --sweep` sweeps the fusion weights (roadmap v3.7).
- `vectora answers` scores the generated answer: citation coverage, citation
  validity, quote verifiability, anchor recall and refusal accuracy
  mechanically, plus model-judged groundedness and relevance reported
  separately — the mechanical scores are the ones to trust when they
  disagree.
- `vectora golden generate` drafts an evaluation set from the indexed corpus
  and **verifies each case against real retrieval**, flagging questions that
  cannot find their own source passage. Negatives are generated from
  adjacent-but-absent topics and checked against the guard.
- Reports are written to `corpora/<name>/reports/` as JSON and markdown.

### Integration and observability

- `vectora mcp` serves the corpus as MCP tools over stdio: `search_corpus`,
  `answer_question`, `list_corpora`, `corpus_info` (roadmap v3.5).
- OpenTelemetry spans per pipeline stage, nested under one span per
  question and carrying the same request id as the logs, so a slow p95
  becomes a flame graph (roadmap v3.6). Optional; a no-op when the
  dependency is absent.

### Frontend

- Consumes the SSE endpoint: tokens appear as they are generated.
- Retrieved sources are shown **while the answer is still being written**,
  each one openable to read the passage with its dense, lexical and rerank
  scores and its chunk quality.
- Inline `[n]` markers are clickable.
- A diagnostics panel with per-stage timings, dense and lexical hit counts,
  the query variants searched, and a warning when stages were skipped.
- Per-answer cost and token counts.

### Fixed

- **A table could be silently discarded with the reference list.** A page
  whose last lines were the references dragged its results table in with it,
  and the table was dropped as reference material. Tables now attach to the
  most recent indexable section on the page.
- **A results section containing a table could be dropped entirely.** The
  letter-ratio and digit-density rules judged the whole chunk by prose
  standards, so a grid of numbers failed the chunk and took the surrounding
  prose with it. Those rules now measure the prose part.
- HTML parsing stripped `<head>`, discarding the `<title>` and citation meta
  tags — the most reliable metadata an HTML document has.
- The ablation harness no longer needs to mutate shared configuration:
  `Retriever.retrieve` takes a `profile_override`.

### Still open

See `docs/ROADMAP.md`. The thresholds in `.env.example` remain reasoned
rather than measured; run `vectora calibrate` against a real index and
commit the measured values.

## v2.0.0 — 2026-10-10

Rebuilt as an installable package with a real retrieval pipeline.

- True hybrid retrieval: dense and lexical search run independently over the
  whole corpus and are fused by reciprocal rank fusion, replacing BM25
  re-scoring of the dense candidates — which by construction could not
  surface anything vector search missed.
- Deterministic Qdrant point ids, so incremental indexing is idempotent.
  Previously ids came from `enumerate` over the uploaded batch, so a second
  run restarted at 0 and overwrote the first run's vectors.
- Metadata extraction rewritten: valid publication years went from 0% to
  100% (the old year regex used a capturing group, so `re.findall` returned
  `"20"` for every paper), and junk titles from 23% to 6%.
- Chunk size floor enforced on every path; sub-floor chunks went from 2,189
  to 13.
- Citations by source number, resolved against the chunks actually in the
  prompt; invented markers are stripped. Quotes verified against source text.
- Guards gate on an absolute relevance score; grounded refusal instead of an
  unsupported answer.
- FastAPI with SSE streaming, request correlation, metrics and rate
  limiting, keeping the previous route contract.
- One CLI for every operation, including `doctor`, `repair`, `eval` and
  `calibrate`; an evaluation harness with a golden set and negative cases;
  Docker Compose.

## v1 — 2026-03

First working pipeline: ingestion, embedding, retrieval and generation as
four standalone scripts over a corpus of vitamin D measurement papers.
