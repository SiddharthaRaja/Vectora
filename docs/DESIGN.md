# Design notes

Why Vectora is built the way it is. Written to be read start to finish —
each section states a problem, the options, and what was chosen and why.
Where a choice has a real cost, that cost is named.

If you only read one section, read [Why RRF and not a weighted
score](#why-rrf-and-not-a-weighted-score) and [How the system decides not to
answer](#how-the-system-decides-not-to-answer). Those two contain most of
the reasoning the rest depends on.

---

## The shape of the problem

A RAG system is four decisions stacked on each other:

1. **What is a chunk?** Too small and a chunk has no context; too large and
   the relevant sentence is diluted by irrelevant ones.
2. **How do we find candidates?** Semantic similarity misses exact strings;
   exact matching misses paraphrase.
3. **Which candidates reach the model?** More context is not better: it
   costs money, dilutes attention, and gives the model more opportunities to
   cite the wrong thing.
4. **When should we refuse?** A fluent answer to a question the corpus
   cannot support is the worst output the system can produce, because it is
   indistinguishable from a good one.

Most of the interesting engineering is in 2 and 4.

---

## Chunking

### Fixed windows versus semantic boundaries

A fixed window (say 512 tokens with 50 of overlap) is simple and
predictable. Its flaw is that it cuts mid-argument: the sentence stating a
finding ends up in one chunk and the sentence qualifying it in the next, so
retrieving one without the other produces a confident, context-free quote.

Vectora splits on *topic* instead: sentences are embedded, and the text is
cut where the cosine similarity between adjacent sentences drops below a
threshold (0.72). A chunk is then a stretch of text about one thing.

The cost is honest: it needs an embedding pass at ingestion time, it is
slower, and the threshold is a magic number that suits some prose better
than others. `vectora ingest --no-semantic` falls back to size-based packing,
which is what the tests use.

### Why there is still a size clamp

Semantic chunking alone produces wildly uneven chunks — a tight paragraph
becomes 40 tokens, a rambling methods section becomes 1,500. So every path
ends at the same clamp: 48 to 420 tokens, with one sentence of overlap
between neighbours.

The overlap matters more than it looks. Without it, a fact stated across a
boundary — "…the limit of detection was" / "2.5 ng/mL in serum" — is
retrievable from neither side. One sentence of repetition costs a few per
cent of index size and removes that whole class of failure.

> **The bug this replaced.** The original `semantic_chunk` returned early
> when the text held a single sentence, skipping the clamp entirely. That is
> how 2,189 sub-floor chunks reached the index, including gems like
> `"journal homepage: www.elsevier.com/locate/steroids"`. Every path now
> terminates at `enforce_size`, and there is a test asserting the early
> return cannot come back.

### Corpus quality is most of answer quality

This is the least glamorous part of the system and the one with the largest
effect. A chunk of layout debris in the index is not neutral — it competes
for retrieval slots, and when it wins it gets cited.

Two filters in series:

- **Line-level regexes** remove running heads, page numbers, publisher
  furniture, licence boilerplate and reference lists.
- **A quality score** catches what the regexes miss, because no regex list
  is ever complete. It rejects fragments, low letter ratios, repetition and
  text where PDF extraction lost the spaces between words. The score is
  stored on every surviving chunk, so retrieval can prefer clean text and a
  bad citation can be traced back.

Sections that carry no retrievable claims — references, acknowledgements,
funding, conflicts of interest, data availability — are never indexed.
Nobody asks a question whose answer is a funding statement, and an author
list matches far too many queries lexically.

> **A subtlety worth knowing.** The character-ratio rules are applied to the
> *prose* part of a chunk, not the whole thing. A results section containing
> a table of numbers legitimately has a low letter ratio and a high digit
> ratio; judging it by prose standards failed the chunk and discarded the
> surrounding prose with it. `_without_table_rows` strips tabular spans
> before measuring — including inline ones, because by the time a chunk is
> scored, chunking has joined the table's rows into one long line.

---

## Retrieval

### Why hybrid at all

Dense retrieval embeds the query and the passages into the same vector space
and finds neighbours. It handles paraphrase well: a question about
"combining rankings" finds a passage about "fusing result lists" even with no
shared words.

It is reliably bad at exact strings. An assay name, a reference material
code (`SRM 972a`), an instrument model, a specific number — these are
tokenised into fragments and their meaning is not compositional. Ask for
`nDCG@20` and a dense retriever gives you passages about evaluation metrics
in general.

Lexical retrieval (BM25) is the mirror image: exact terms are its strength,
paraphrase its blind spot.

The two fail on *largely disjoint* queries, which is what makes combining
them worth more than improving either.

> **The bug this replaced, and it is the instructive one.** The original
> implementation ran BM25 **over the dense candidate pool**. That cannot
> help. If vector search did not return the chunk, BM25 never sees it, so
> the lexical retriever could only reorder what the dense retriever had
> already found — and the entire reason to run it is to catch what the dense
> retriever missed. The fix was not a better weighting; it was running BM25
> as an independent retriever over the whole corpus.

You can see this directly: `vectora search "nDCG@20"` finds the chunk whose
only mention of that term is inside a markdown table. No dense retriever
would have surfaced it.

### Why RRF and not a weighted score

Having two ranked lists, you have to combine them. The obvious approach is a
weighted sum of scores:

```
combined = α · dense_cosine + (1 − α) · normalised_bm25
```

This has two problems, and the second is the subtle one.

**Different scales.** Cosine similarity lives in roughly [0, 1]. BM25 is
unbounded and depends on corpus statistics. They are not comparable, so the
BM25 score has to be normalised first.

**Normalisation is batch-relative.** Min-max normalising BM25 scores over
the current candidate set means a chunk's score depends on *which other
chunks happened to be retrieved alongside it*. The same chunk, for the same
query, scores differently depending on the batch. That destroys any
possibility of a stable threshold anywhere downstream.

Reciprocal rank fusion discards the magnitudes and fuses *ranks*:

```
score(d) = Σ          weight_list / (k + rank_list(d))
         over lists
```

with `k = 60`, from Cormack, Clarke & Buettcher (2009). Properties that
matter here:

- **Scale-free.** No normalisation, so no batch dependence.
- **Comparable across queries.** A first-place hit always contributes
  `1/61`, whatever the corpus or query.
- **Rewards agreement.** A chunk that several independent query phrasings
  *and* both retrievers rank highly accumulates score. Independent agreement
  is evidence, and this is the mechanism that uses it.
- **`k` damps the top.** Without it, rank 1 would dominate rank 2 so heavily
  that one retriever's favourite would always win.

What RRF **cannot** tell you is whether anything was relevant. A chunk
ranked first in a hopeless pool scores exactly as well as one ranked first
in an excellent pool. That limitation drives the next section.

### Reranking, and why only on a small pool

A bi-encoder embeds the query and the passage separately. That is what makes
retrieval over a large corpus tractable — passage vectors are precomputed —
but it means the two texts never interact. The model never gets to notice
that the query's "latency" and the passage's "latency" refer to different
things.

A cross-encoder processes the concatenated pair with full attention between
them. Much more accurate, and linear in the number of candidates, so it is
only viable as a second stage: 24 to 64 candidates, never the collection.

This is the classic retrieve-then-rerank arrangement, and it is a real
quality/latency trade. `vectora ablate` measures whether it is worth it on
*your* corpus rather than asking you to take it on faith.

### MMR, and a bug worth understanding

Maximal marginal relevance selects the final set balancing relevance against
redundancy:

```
score(d) = λ · sim(d, query) − (1 − λ) · max sim(d, already selected)
```

λ near 1 maximises relevance — right for a definition, where you want the
single best passage. λ near 0 maximises coverage — right for a comparison or
an overview, where three near-identical chunks from one paper are a waste of
the context window. The value comes from the query router.

> **The bug.** MMR returned early whenever the candidate pool was no larger
> than `top_k`, and the stage before it handed over `min(top_k × 3,
> dense_k)`. For the `comparative` and `exploratory` routes — the ones that
> most need diversity — diversification almost never ran. The pool is now
> sized so there is a real choice to make.

### Capping one paper's share

No single document may supply more than half the context. Without this, one
long, lexically rich document fills the window and the answer silently
becomes a summary of one source while reading like a survey of many. That is
a *correctness* problem disguised as a ranking preference.

---

## Generation

### Routing, and why rules first

Six query types, each setting a `top_k`, an MMR λ and a prompt instruction. A
definitional question wants three chunks and λ=0.9; an exploratory one wants
eight and λ=0.35.

A rule pass runs **before** any model call, because most questions are
classifiable from their interrogative and a few cue words, and skipping a
network round-trip on roughly half of real traffic is free latency.

The rule ordering is the whole design and is worth reading in
`generation/router.py`. `DEFINITIONAL` is checked *last* among the "what
is…" patterns, because its shape also matches "what is the concentration
threshold?" (factual) and "what are the effects of X?" (causal), which are
narrower and more useful readings.

When nothing matches and the model is unavailable, the fallback is
`methodological`, not `exploratory`. Exploratory pulls the widest, most
diffuse candidate set, which is the worst thing to do when you have no idea
what was asked.

### Citations that can be verified

The hard part of citation is not producing markers; it is making them
*checkable*.

> **The bug this replaced.** The original asked the model to append a JSON
> block listing the paper titles it used, then matched those strings back to
> chunks with a 40-character substring comparison. It failed three ways: the
> block was sometimes missing or malformed; titles in that corpus were often
> wrong or duplicated across papers, so the match was ambiguous; and a
> hallucinated title produced *no citation at all* while the answer still
> read as cited.

What replaces it:

1. **The model cites by source number** — `[1]`, `[2][4]`. Unambiguous and
   parseable with a regex.
2. **Numbers resolve against the exact chunks that were in the prompt**, so
   a citation can only point at something the model was actually shown.
3. **Invented markers are stripped** from the answer text. A marker pointing
   nowhere reads as a reference the user could follow; removing it leaves
   the sentence merely uncited, which is honest.
4. **Quotes are verified.** In `deep` mode a second pass attaches a verbatim
   supporting quote to each citation, and that quote is checked against the
   chunk it claims to come from — exactly, then fuzzily, because PDF text
   carries whitespace and ligature variation a model will normalise when
   copying. A quote that is not really there is discarded and logged. A
   fabricated quote is worse than no quote.

### How the system decides not to answer

This is the part that determines whether the system can be trusted.

Two guards. The interesting question is *what signal they use*.

RRF scores are comparable across queries but, as established above, carry no
information about whether anything was relevant. So gating on them can only
detect "nothing was retrieved at all".

The gate therefore uses an **absolute** relevance signal, choosing by what is
available:

1. the **cross-encoder logit**, when the reranker ran — the best absolute
   judgement of (query, passage) relevance available. On
   `ms-marco-MiniLM-L-6-v2`, on-topic passages score above 0 and clearly
   irrelevant ones land between −8 and −11.
2. otherwise the **raw dense cosine**, which is absolute and comparable.
   `all-MiniLM-L6-v2` puts unrelated sentence pairs near 0.0–0.15 and
   genuinely related ones above 0.35.
3. only if neither exists (lexical-only retrieval), the RRF score.

> **The bug.** A cross-encoder logit floor of `−8.0` was compared against
> whatever `score` happened to hold — which in `fast` mode was a 0–1
> similarity. The guard therefore could never fire in the one mode where
> weak retrieval is most likely. Mixing score scales is an easy mistake to
> make and a hard one to notice, because the system keeps working; it just
> stops refusing.

When a guard trips, the user gets a **grounded refusal**: what the corpus
does cover, and three rephrasings aimed at topics actually present in the
retrieved-but-weak chunks. Not "I don't know", and not a fluent answer about
something the papers never said.

**The thresholds are corpus-specific.** The shipped values are reasoned from
how those two models score, not measured on your corpus. `vectora calibrate`
measures both distributions over your golden set's real and off-topic
questions and prints the floor that separates them. Until you run it, the
refusal behaviour is a hypothesis.

### Grounding

A separate call, at temperature 0, that grades the answer's claims against
the sources — and is deliberately **not shown the question**. Showing the
checker what the answer was supposed to say biases it toward accepting
plausible-sounding claims.

A mechanical pre-check runs first and is *additive*: any figure in the
answer that appears in no source caps the score regardless of the model's
verdict. A fabricated number is the most consequential and most detectable
kind of ungrounded claim, and finding it costs nothing.

---

## Evaluation

### Why the golden set lives with its corpus

An evaluation set written for one corpus measures nothing on another. So it
is `corpora/<name>/golden.yaml`, not a project-level file.

### Negative cases are half the evaluation

recall@k, MRR and nDCG all measure whether the right passage was found. None
of them can see whether the system correctly *declines*. A system that
answers anything fluently can score well on all three and be useless.

So the golden set has `negative_cases`, and `vectora eval` reports the
refusal rate separately. Three kinds are worth writing:

- plainly off-topic (a different field entirely)
- **adjacent but absent** — the hard one: neighbouring subject matter your
  corpus happens not to cover
- in-domain but unasked

The adjacent ones are where a RAG system actually fails, because retrieval
returns something superficially plausible and the guard has to be the thing
that catches it.

### Why the ablation reports confidence intervals

With 20 or 50 cases, a recall difference of 0.02 is noise. A table of
three-decimal numbers with no uncertainty invites exactly the overclaiming
that makes benchmark tables worthless.

Each variant is compared against the baseline with a **paired bootstrap**:

- *Paired*, because both configurations ran the same cases. Pairing removes
  case difficulty from the comparison, which makes a small real difference
  detectable with few cases.
- *Bootstrap* rather than a t-test, because recall per case is a 0/1
  variable and reciprocal rank is badly non-normal; the normality assumption
  does not hold at these sample sizes.

The report says "no sig. diff" when the interval includes zero. That is the
honest answer, and it is also actionable: it means *add more cases*, not
*ship the change*.

Variants that change more than one thing are marked, because their effect
cannot be attributed to a component.

### Answer metrics, and the limits of LLM-as-judge

`vectora answers` splits its measures deliberately:

**Mechanical** — computed from the text, no model involved, so they cannot
drift or flatter: citation coverage, citation validity, quote verifiability,
anchor recall, refusal accuracy.

**Judged** — one model call: groundedness and relevance.

Using a language model to grade a language model shares the failure modes of
both. The judge is systematically kinder to fluent, hedged, well-structured
answers than a careful human would be, and it cannot catch a confident error
on a subject it is also wrong about. They are reported separately, and **the
mechanical scores are the ones to trust when the two disagree.**

---

## Operational choices

### Configuration in exactly one place

`vectora/config.py` holds every tunable. Nothing else reads the environment
or hardcodes a URL, model name or threshold. The alternative — which is what
v1 did — scatters `localhost:6333` across four files, and then deploying
means finding all of them.

The collection name is *derived* from the corpus rather than configured. Two
corpora sharing one collection would silently mix their documents, so that
state is made unrepresentable rather than documented as a hazard.

### Degrading instead of failing

Startup is deliberately tolerant. No API key, unreachable Qdrant, missing
lexical index: each produces a warning and a degraded service that can still
explain the problem, rather than a process that will not boot.

The important half is that degradation is **reported**, not silent. Every
answer carries `diagnostics.degraded`, so a fast answer is never mistaken
for a thorough one. `/ready` reports per component and returns 503 only when
the service genuinely cannot answer, so a partially degraded deployment
stays in rotation.

### Spend

Every call's tokens and estimated cost are tracked per request and against a
daily ceiling. Crossing the ceiling disables the *optional* expansion and
grounding calls rather than the answer itself — the service degrades instead
of stopping. Utility calls are cached, so re-asking a question does not
re-pay for its routing and rewrites.

`fast` mode makes exactly one OpenAI call. `vectora search` makes none.

### Tracing

The request id ties a question's log lines together, which answers *what
happened*. Spans answer *where the time went*, which is the question you
actually have when a p95 is bad: a flat 9-second answer tells you nothing, a
trace showing 6 seconds in a serial HyDE call tells you what to fix.

Attributes are measurements only — candidate counts, scores, model names,
token counts. Never the question text or the retrieved passages. A trace
backend is not where user content should end up.

---

## Things that are still wrong, or unmeasured

Being specific about this is more useful than a confident summary.

- **The refusal thresholds are reasoned, not measured.** See above. This is
  the most consequential unmeasured number in the system.
- **`k = 60` is inherited, not tuned.** From the original RRF paper. It is
  probably fine; nobody has checked on this corpus.
- **The semantic chunking threshold (0.72) is a magic number.** It has not
  been swept. It likely suits some document types much better than others.
- **HyDE is expensive and unproven here.** It costs a generation call per
  question and the ablation has not been run on a real corpus yet, so
  whether it earns that is currently an assumption.
- **The quality score's weights are judgement.** The hard rejections are
  defensible; the soft penalties (−0.30 for no sentence punctuation, −0.20
  for digit density) are guesses that happen to behave sensibly on the
  corpora tried.
- **BM25 is in-process.** Right for tens of thousands of chunks, wrong past
  a few hundred thousand, where it should move to Qdrant's native sparse
  vectors or a search cluster. The interface is narrow enough that this is a
  contained change.
- **Lost inter-word spacing cannot be repaired.** Around 1,500 chunks in the
  original corpus have text like `requiremoreexpensiveequipment` because the
  first-generation parse dropped the spaces. They are detected and dropped,
  and the affected documents are named in the ingestion report, but only
  re-ingesting from the sources recovers that text.

---

## Where to read the code

In this order, it tells a story:

| File | What it decides |
|---|---|
| `vectora/config.py` | every tunable, in one place |
| `vectora/models.py` | the contracts between stages |
| `vectora/ingestion/clean.py` | what is allowed into the corpus |
| `vectora/ingestion/chunking.py` | what a chunk is |
| `vectora/retrieval/fusion.py` | how two retrievers are combined |
| `vectora/retrieval/diversify.py` | which candidates survive |
| `vectora/generation/guards.py` | when to refuse |
| `vectora/generation/citations.py` | how a citation is made checkable |
| `evaluation/ablate.py` | how a change is shown to be real |

Each module's docstring states the problem it solves and, where relevant,
the specific defect it replaces.
