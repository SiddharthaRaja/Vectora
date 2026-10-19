# Changelog

Notable changes per release. Dates are UTC.

## Unreleased — measuring what the milestones ask for

### Fixed: concurrent searches corrupted embedded Qdrant (50c59bc, now tested)

- **Symptom.** In run 2 (Colab T4) the ablation baseline scored 0.239 recall
  where the identical eval run scored 0.591, cosine scores of ~2.4e13 appeared,
  and numpy overflow warnings came from `qdrant_client/local`.
- **Cause.** qdrant-client's local (embedded) mode normalises the stored vector
  matrix *in place* during every cosine search. Balanced and deep modes run one
  dense search per query variant concurrently via `asyncio.to_thread`, so two
  threads wrote to the same matrix at once. Run 1 used fast mode (one search per
  query) and never hit it.
- **Fix.** `QdrantStore` serialises `query_points` and `upsert` with a lock in
  embedded mode only; a Qdrant server handles its own concurrency.
- **Regression tests** (`tests/test_qdrant_concurrency.py`): a fake client that
  makes overlap certain shows 8 concurrent entries without the lock and 1 with
  it; server mode stays parallel; real embedded Qdrant under 8 threads returns
  only valid cosines. The original corruption is timing-dependent and was only
  partly reproduced outside Colab (scores of 9 and 560 from concurrent
  normalisation of a non-unit matrix), so the test pins the invariant instead.
- **Recommendation.** Embedded mode is for laptops and notebooks. Anything
  serving concurrent users should use the Qdrant server (`docker compose up`),
  where the lock is not taken.
- **Consequence.** The run-2 ablation table is invalid and must be re-run.

### New evaluation metrics

- **Evidence-level retrieval recall.** Golden cases can carry
  `relevant_evidence` (QASPER's annotator-marked paragraphs). A chunk only
  counts if it contains one (6-word shingle overlap ≥ 0.5 of the smaller text),
  so "right paper, wrong section" is now a miss. `paper_recall` is reported
  alongside.
- **Answer token F1 and token recall** against every annotator's answer
  (`reference_answers`), SQuAD-normalised, best over references
  (`evaluation/textmatch.py`).
- **Pre-strip hallucinated-citation rate.** `Answer` now records the markers
  the model wrote (`raw_citation_markers`) and those pointing at no source
  (`invalid_citation_markers`). The old `citation_validity` was measured after
  stripping and so was 100% by construction; `raw_citation_validity` and
  `hallucinated_citation_rate` are the real numbers.
- **Judge agreement** (`evaluation/judge_agreement.py`): export ~30 judged
  answers with their cited sources to CSV, hand-label faithfulness, get
  agreement, Cohen's kappa and the confusion matrix.

### CI and the eval regression gate

- `.github/workflows/ci.yml`: lint and unit tests on every push and PR, then a
  retrieval eval gate. The gate builds a fixed 60-paper QASPER dev corpus
  (71 title-free questions, 8 negatives), indexes it with embedded Qdrant on
  CPU and runs `fast`-mode retrieval — no GPU, no LLM, no API key.
- `evaluation/gate.py` compares the report with `evaluation/ci_baseline.json`
  and fails on a drop of more than one case's worth (recall, paper recall,
  MRR, nDCG, negative rejection). The first run has no baseline, passes, and
  uploads `ci_baseline.suggested.json` as an artifact to commit.
- The CI corpus is chunked with `--no-semantic` to stay fast, so its numbers
  are a regression signal, not comparable with the semantic-chunked runs.
- Faithfulness is not gated in CI yet: it needs an LLM.

### QASPER builder

- Title-free queries by default (`--title-prefix` restores the old,
  recall-inflating behaviour); questions too generic to ask without the title
  are dropped. Evidence and all reference answers are written per case;
  placeholder tokens are cleaned from papers and evidence.
- New `corpora/qasper-300`: 300 papers from train+dev, 365 questions, 8
  off-corpus negatives.

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
