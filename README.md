# Vectora

Retrieval-augmented question answering over a corpus of research papers. Ask a
question in natural language, get an answer whose every claim carries a
citation back to the passage it came from — and a refusal rather than a guess
when the papers do not support an answer.

The current corpus is ~60 papers on vitamin D measurement: assay methods,
LC-MS/MS, electrochemical and impedance biosensors, spectroscopy and nutrition.
Nothing in the code is specific to that subject; point it at a different folder
of PDFs and it works the same way.

```
vectora ingest      PDFs          → chunks.jsonl
vectora index       chunks.jsonl  → Qdrant (dense) + BM25 (lexical)
vectora ask "..."   question      → cited answer
vectora serve                     → web UI on :5000
```

---

## Quick start

```bash
python -m venv venv
venv\Scripts\activate            # Windows;  source venv/bin/activate elsewhere
pip install -e ".[ingest,metrics,eval]"

cp .env.example .env             # then put your OPENAI_API_KEY in it

docker run -p 6333:6333 -v "%cd%/data/qdrant:/qdrant/storage" qdrant/qdrant

vectora doctor                   # checks everything above and says what is wrong
vectora index                    # build the index from the chunks you already have
vectora serve                    # → http://localhost:5000
```

No Docker? Set `VECTORA_QDRANT_PATH=./data/qdrant` in `.env` and leave
`VECTORA_QDRANT_URL` blank — `qdrant-client` has an embedded on-disk engine and
no server is needed. One process can hold the directory at a time, so the CLI
and the server cannot both run against it simultaneously.

`vectora doctor` is the first thing to run when anything misbehaves. It checks
the API key, the vector store, the collection's dimension against the embedding
model, every optional dependency, and the data files, and prints the command
that fixes each problem.

---

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

### Evaluating changes

`eval/golden.yaml` holds questions with what a correct retrieval must surface,
plus **negative cases** the corpus genuinely cannot answer. `vectora eval`
reports recall@k, precision, MRR and nDCG per mode, and separately how often
off-topic questions are correctly refused — which the ranking metrics cannot
see, and which is half of whether this system is trustworthy.

Add a case every time you find a question it gets wrong. That is what turns "it
felt better after the change" into a number.

---

## Configuration

Everything is set by environment variable or `.env`; nothing else in the
codebase reads the environment or hardcodes a URL, model or threshold. See
`.env.example` for the annotated list. The ones worth knowing:

| variable | default | notes |
|---|---|---|
| `OPENAI_API_KEY` | — | required to answer; retrieval works without it |
| `VECTORA_QDRANT_URL` | `http://localhost:6333` | or `VECTORA_QDRANT_PATH` for embedded |
| `VECTORA_ANSWER_MODEL` | `gpt-4o-mini` | |
| `VECTORA_DAILY_COST_CEILING_USD` | `5.0` | soft cap; past it, only the answer call runs |
| `VECTORA_MIN_RERANK_SCORE` | `-4.0` | refusal floor — run `vectora calibrate` |
| `VECTORA_MIN_DENSE_SCORE` | `0.26` | refusal floor when no reranker ran |
| `VECTORA_RRF_K` | `60` | fusion damping |
| `VECTORA_DEFAULT_MODE` | `balanced` | |

Changing the embedding model changes the vector dimension, so the collection has
to be rebuilt: `vectora index --recreate`. A mismatch is detected at startup and
refused rather than silently returning nonsense.

**Spend.** Every call's tokens and estimated cost are tracked per request and
against a daily ceiling. Crossing the ceiling disables the optional expansion
and grounding calls rather than the answer itself, so the service degrades
instead of stopping. `GET /stats` shows the running total. The utility calls are
cached, so re-asking a question does not re-pay for its routing and rewrites.

---

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

---

## Layout

```
vectora/
  config.py          every tunable, env-driven, single source of truth
  models.py          the contracts between stages (pydantic)
  llm.py             the only module that talks to OpenAI
  embeddings.py      encoder + cross-encoder, lazily loaded
  cache.py           SQLite KV cache for query vectors and utility calls
  observability.py   logging, request correlation, metrics
  ingestion/         parse · metadata · sections · chunking · clean · pipeline
  store/             qdrant_store (dense) · sparse (BM25)
  retrieval/         expansion · fusion · diversify · pipeline
  generation/        router · prompts · citations · guards · grounding · pipeline
  db/                conversation persistence
  api/               FastAPI app, routes, schemas, PDF export
  cli.py             every operation
eval/                golden set, eval harness, threshold calibration
tests/               runs with no network, no models, no Qdrant
```

`pytest` runs the whole suite in a few seconds: everything that touches a model,
the network or a vector store is either a pure function or stubbed.

---

## Notes on the current corpus

`chunks.jsonl` was produced by an earlier version of the pipeline, and two of
its problems cannot be fixed without the source PDFs:

* **Spaces lost in parsing.** Around 1,500 chunks across ~30 papers have text
  like `requiremoreexpensiveequipment` — the PDF extractor dropped the spaces
  between words. Such text is unmatchable lexically and tokenises badly for the
  embedder. These chunks are now detected and dropped, and the papers are named
  in the ingestion report. The fix is to re-ingest from the PDFs; the parser now
  uses a character tolerance that preserves the spacing.
* **Titles.** Most papers' embedded titles were publisher furniture, so the
  filename is used instead. The filenames in this corpus are descriptive, so
  citations read acceptably, but re-ingesting recovers the real titles.

To re-clean what you have without the PDFs:

```bash
vectora repair          # keeps a .bak; re-applies all current cleaning rules
vectora index --recreate
```

On the shipped corpus that takes 9,101 chunks to ~4,700, dropping reference
lists, uncaptioned figure placeholders, fragments and space-damaged text, and
fixing every chunk's publication year — the old pipeline stored `"20"` for all
of them, because its year regex used a capturing group and `re.findall` returns
the group rather than the match.

Re-ingesting from the PDFs is strictly better. Put them in `papers/` and run
`vectora ingest`.

---

## License

MIT.
