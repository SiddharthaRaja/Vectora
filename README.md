```
vectora fetch arxiv --preset retrieval    build a corpus from arXiv
vectora ingest                            documents    → chunks.jsonl
vectora index                             chunks.jsonl → Qdrant + BM25
vectora ask "..."                         question     → cited answer
vectora serve                             web UI on :5000
vectora ablate                            is each stage earning its latency?


corpora/<name>/
  corpus.json        what these documents are, and the vocabulary to expand with
  papers/            the source documents
  chunks.jsonl       ingestion output
  bm25.pkl           the lexical index
  golden.yaml        this corpus's evaluation set
  reports/           eval and ablation output
```

```bash
vectora corpus list                    # what exists, what is indexed
vectora corpus show mycorpus
vectora corpus delete mycorpus         # drops the index, keeps the documents
```

## How it works

Two independent paths that share exactly one thing — the index. The left path
runs when papers change; the right one runs on every question and never touches
a PDF.

```
PDFs ──► parse ──► sections ──► semantic chunks ──► quality gate ──► chunks.jsonl
                                                                          │
                                                                embed + index
                                                                          ▼
                                                        ┌─────────────────────┐
question ──► classify ──► expand ──► search ◄───────────┤ Qdrant  +  BM25     │
                                        │               └─────────────────────┘
                                        ▼
                      rank fusion ──► rerank ──► diversify ──► guard
                                                                 │
                                                    ┌────────────┴────────────┐
                                                    ▼                         ▼
                                            cited answer              grounded refusal
```

### Ingestion

PDF text, tables and figures are extracted separately and recombined into one
chunk stream. Text is split at topic boundaries — sentences are embedded and the
text is cut where adjacent-sentence similarity drops — then clamped to 48–420
tokens with a sentence of overlap, so a fact spanning a boundary stays
retrievable from one chunk. Tables become one chunk each, prefixed with their
paper and section so a retrieved grid is interpretable alone.

Then two filters, because answer quality is mostly corpus quality. Line-level
regexes drop running heads, page numbers, publisher furniture and reference
lists. A chunk-level **quality score** catches what survives: fragments,
low-letter-ratio OCR debris, repetition, and text where PDF extraction lost the
spaces between words. The score is kept on every chunk that passes, so
retrieval can prefer clean text and a bad citation can be traced back.

Incremental by file fingerprint: adding one paper to a corpus of sixty costs one
paper's work.

### Retrieval

Dense and lexical search run **independently over the whole corpus** and their
rankings are fused. This is the part that matters most: a vector search alone
misses exact rare strings — an assay name, `SRM 972a`, an instrument model — and
a lexical search alone misses paraphrase. Running BM25 only over the dense
candidates, which is the obvious shortcut, cannot help at all: it can only
reorder what the vector search already found.

Fusion is **reciprocal rank fusion**:

```
score(chunk) = Σ   weight / (k + rank_in_that_list)
             lists
```

Ranks rather than scores, because dense cosine sits in [0, 1] while BM25 is
unbounded and corpus-dependent, and normalising them against each other makes a
chunk's score depend on which other chunks happened to be in the batch. RRF is
comparable across queries. A chunk that several independent query phrasings and
both retrievers agree on accumulates score, which is the behaviour you want:
independent agreement is evidence.

Then a cross-encoder reranks a small pool, and MMR picks the final set balancing
relevance against redundancy — tight for a definition, diverse for a comparison.
No single paper may supply more than half the context, so an "across the
literature" question cannot be answered from one source while looking like a
survey.

### Generation

The question is classified into one of six types, which sets the number of
chunks, the MMR balance and a prompt instruction. Most questions are classified
by rule, with no network call.

The model cites by **source number** — `[1]`, `[2][4]` — which is parseable, and
every number is resolved against the exact chunks that were in the prompt. A
number the model invents is stripped from the answer rather than left looking
like a reference the reader could follow. In `deep` mode a second pass attaches
a verbatim supporting quote to each citation and **verifies it against the
source text**; a quote that is not really there is discarded.

### Refusing to answer

Two guards. Both matter more than they look.

The gate needs an **absolute** relevance signal, and RRF is not one — it ranks,
so a chunk placed first in a hopeless pool scores exactly as well as one placed
first in a good pool. So the guard uses the cross-encoder logit when the
reranker ran, and the raw dense cosine otherwise. Both mean the same thing from
one query to the next.

When retrieval comes up short the user gets a grounded refusal: what the corpus
does cover, and three rephrasings aimed at topics actually present, rather than
a fluent answer about something the papers never said.

Those floors are corpus-specific, so measure them instead of trusting the
defaults:

```bash
vectora calibrate     # prints the score distributions and suggested floors
```

### Search modes

| mode | candidates | HyDE | rewrites | rerank | MMR | grounding | LLM calls |
|------|-----------:|:----:|---------:|:------:|:---:|:---------:|----------:|
| `fast` | 24 | – | 0 | – | – | – | 1 |
| `balanced` | 40 | yes | 2 | yes | yes | – | 4 |
| `deep` | 64 | yes | 3 | yes | yes | yes | 6 |

`fast` makes exactly one OpenAI call — the answer. The expansion calls in the
other modes go out concurrently, so three rewrites plus a HyDE document cost
roughly one round-trip rather than four.

---

## Using it

### Web

`vectora serve`, then <http://localhost:5000>. Pick a mode, optionally filter by
year or paper. OpenAPI docs at `/docs`.

| endpoint | what it does |
|---|---|
| `POST /sessions/{id}/chat` | answer, buffered |
| `POST /sessions/{id}/chat/stream` | answer, server-sent events |
| `POST /search` | **retrieval only, no generation call** |
| `GET /ready` | per-component readiness |
| `GET /stats` | corpus, cache, spend |
| `GET /metrics` | Prometheus |
| `GET /sessions/{id}/export/pdf` | conversation as PDF |

### Command line

```bash
vectora ask "What defines vitamin D deficiency?"
vectora ask "..." --mode deep --year 2023
vectora chat                              # conversational, with memory
vectora search "LC-MS/MS calibration"     # retrieval only, all per-stage scores
vectora stats
vectora eval                              # score retrieval against the golden set
```

`vectora search` is the first thing to reach for when an answer looks wrong: it
shows what each stage contributed — dense rank, BM25 rank, fused score, rerank
logit, which query variants found each chunk — and separates a retrieval problem
from a prompting one. In `fast` mode it costs nothing.

### Measuring it

The point of this part is to stop guessing. Four commands, in the order you
would use them.

```bash
vectora golden generate     # draft an evaluation set from the indexed corpus
vectora eval                # recall@k, precision, MRR, nDCG, refusal rate
vectora ablate              # does each stage earn its latency?
vectora answers             # citation and grounding quality
vectora calibrate           # measure the refusal thresholds
```

**`golden generate`** drafts cases from the corpus and then *verifies each
one against real retrieval*, so a question that cannot find its own source
passage is flagged rather than quietly included. Negatives are generated
from adjacent-but-absent topics and checked against the guard. Review the
output — a generated set is a starting point, and the cases you write after
seeing the system get something wrong are worth more than ten generated
ones.

**`eval`** reports the ranking metrics per mode, and separately how often
off-topic questions are correctly refused. That last number matters at least
as much: recall and MRR cannot see whether the system declines, and a RAG
system that answers anything fluently is not trustworthy however good its
recall is.

**`ablate`** runs the golden set with one component removed at a time:

```
  variant               recall     MRR    nDCG   p50ms    neg         Δrecall  verdict
  hybrid+rerank+mmr      0.812   0.671   0.704    6100    89%       baseline
  no-rerank              0.770   0.588   0.631    1900    78%         -0.042   no sig. diff (CI -0.096..+0.013)
  dense-only             0.684   0.521   0.572    5800    89%         -0.128   worse (CI -0.201..-0.055)
```

Each comparison is a paired bootstrap confidence interval, and the report
says plainly when a difference is inside the noise. A table of three-decimal
numbers with no uncertainty invites exactly the overclaiming that makes
benchmark tables worthless. `--sweep` sweeps the fusion weights instead.

**`answers`** scores the generated answer, splitting mechanical measures
(citation coverage and validity, quote verifiability, anchor recall, refusal
accuracy) from model-judged ones (groundedness, relevance). The mechanical
ones are the ones to trust when they disagree — using a language model to
grade a language model shares the failure modes of both.
Measured, not assumed: against 30 answers labelled by hand claim by claim,
the qwen2.5:7b judge agreed 83% of the time but with Cohen's κ −0.06 (no
better than chance) and passed all 4 answers that contained an unsupported
claim ([run 3c v2](results/2026-10-27_run3c_v2_colab-t4/RUN.md#judge-agreement)).

Reports land in `corpora/<name>/reports/` as JSON and markdown.

---

## Configuration

Everything is set by environment variable or `.env`; nothing else in the
codebase reads the environment or hardcodes a URL, model or threshold. See
`.env.example` for the annotated list. The ones worth knowing:

| variable | default | notes |
|---|---|---|
| `OPENAI_API_KEY` | — | required to answer; retrieval works without it |
| `VECTORA_CORPUS` | `default` | which corpus to use; or pass `--corpus` |
| `VECTORA_QDRANT_URL` | `http://localhost:6333` | or `VECTORA_QDRANT_PATH` for embedded |
| `VECTORA_ANSWER_MODEL` | `gpt-4o-mini` | |
| `VECTORA_DAILY_COST_CEILING_USD` | `5.0` | soft cap; past it, only the answer call runs |
| `VECTORA_MIN_RERANK_SCORE` | `-4.0` | refusal floor — run `vectora calibrate` |
| `VECTORA_MIN_DENSE_SCORE` | `0.26` | refusal floor when no reranker ran |
| `VECTORA_RRF_K` | `60` | fusion damping |
| `VECTORA_DEFAULT_MODE` | `balanced` | |
| `VECTORA_ENABLE_TRACING` | `false` | OpenTelemetry spans per stage |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | — | standard OTEL variable; console exporter without it |

Changing the embedding model changes the vector dimension, so the collection has
to be rebuilt: `vectora index --recreate`. A mismatch is detected at startup and
refused rather than silently returning nonsense.

The collection name is derived from the corpus, not configured, so two
corpora cannot share one collection. `VECTORA_QDRANT_COLLECTION` exists as
an override for a one-off, and is normally left unset.

**Spend.** Every call's tokens and estimated cost are tracked per request and
against a daily ceiling. Crossing the ceiling disables the optional expansion
and grounding calls rather than the answer itself, so the service degrades
instead of stopping. `GET /stats` shows the running total. The utility calls are
cached, so re-asking a question does not re-pay for its routing and rewrites.

---

## Serving retrieval as MCP tools

```bash
vectora mcp
```

Speaks MCP over stdio, exposing four tools: `search_corpus` (passages with
citations, no generation call), `answer_question` (the full pipeline),
`list_corpora` and `corpus_info`. Point an MCP client at that command.

A client should normally prefer `search_corpus` and reason over the passages
with its own model, rather than putting a second model between itself and
the sources.

## Operating it

Structured logs (`VECTORA_LOG_FORMAT=json`) with a request id threaded through
every stage of a question, so one answer's routing, expansion, retrieval and
generation can be reconstructed from the log. Every error response carries the
same id.

`/health` is for the load balancer. `/ready` reports per component, and returns
503 only when the service genuinely cannot answer — a degraded deployment, such
as a missing lexical index, stays in rotation, and the degradation shows up in
each answer's `diagnostics.degraded` rather than being silent.

Docker: `docker compose up --build` brings up Qdrant and the app. The model
cache is a named volume, so the weights are not re-downloaded on every rebuild.

**Tracing.** Set `VECTORA_ENABLE_TRACING=true` and
`OTEL_EXPORTER_OTLP_ENDPOINT` to get an OpenTelemetry span per pipeline
stage, nested under one span per question and carrying the same request id
as the logs. The request id answers *what happened*; the spans answer *where
the time went*, which is the question you actually have when a p95 is bad.
Optional — a no-op if the dependency is absent.

---

## Layout

```
vectora/
  config.py          every tunable, env-driven, single source of truth
  corpus.py          corpora as named, swappable objects
  models.py          the contracts between stages (pydantic)
  llm.py             the only module that talks to OpenAI
  embeddings.py      encoder + cross-encoder, lazily loaded
  cache.py           SQLite KV cache for query vectors and utility calls
  observability.py   logging, request correlation, metrics
  tracing.py         OpenTelemetry spans (optional)
  fetch.py           corpus acquisition from arXiv
  mcp_server.py      retrieval as MCP tools
  ingestion/         parse · sources · metadata · sections · chunking · clean
  store/             qdrant_store (dense) · sparse (BM25)
  retrieval/         expansion · fusion · diversify · pipeline
  generation/        router · prompts · citations · guards · grounding · pipeline
  db/                conversation persistence
  api/               FastAPI app, routes, schemas, PDF export
  cli.py             every operation
corpora/<name>/      documents, chunks, index, golden set, reports
evaluation/                eval harness · ablations · answer metrics · calibration · generator
docs/DESIGN.md       why it is built this way — the long read
tests/               runs with no network, no models, no Qdrant
```

`pytest` runs the whole suite in a few seconds: everything that touches a model,
the network or a vector store is either a pure function or stubbed.

---

## Notes on the shipped corpus

The `default` corpus is the original material: research papers on vitamin D
measurement. Its `chunks.jsonl` was produced by the v1 pipeline and carries
one fault that cannot be fixed without the source PDFs:

**Spaces lost in parsing.** Around 1,500 chunks across ~30 papers have text
like `requiremoreexpensiveequipment` — the PDF extractor dropped the spaces
between words. Such text is unmatchable lexically and tokenises badly for
the embedder. These chunks are detected and dropped, and the affected papers
are named in the ingestion report. Re-ingesting fixes it; the current parser
uses a character tolerance that preserves the spacing.

To re-clean what is there without the PDFs:

```bash
vectora repair --corpus default        # keeps a .bak
vectora index  --corpus default --recreate
```

That takes 9,101 chunks to ~4,700, dropping reference lists, uncaptioned
figure placeholders, fragments and space-damaged text, and fixing every
chunk's publication year — the v1 pipeline stored `"20"` for all of them,
because its year regex used a capturing group and `re.findall` returns the
group rather than the match.

If you are starting fresh, ignore all of that and fetch a corpus:

```bash
vectora fetch arxiv --preset ml-systems --limit 150
```

## Reading the code

`docs/DESIGN.md` is the long-form explanation: what each decision was
between, what was chosen, and what it costs. It also lists what is still
unmeasured, which is the part worth knowing before trusting any number here.

Each module's docstring states the problem it solves and, where relevant,
the specific defect it replaces. The two most load-bearing ideas are in
`retrieval/fusion.py` (why ranks rather than scores) and
`generation/guards.py` (why refusing needs an absolute signal).

## License

MIT.
