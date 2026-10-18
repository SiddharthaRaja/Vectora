# Roadmap — v4

v2 was the rebuild. v3 made the corpus swappable and built the tools that
turn the system's claims into measurements. **v4 is about running them.**

Everything below needs something this project cannot supply for itself: a
real corpus, an API key, a machine with network access, or a decision only
you can make. That is the whole list — the tooling exists, so each item is
now a command plus a judgement rather than a build.

Ordered by value.

## v4.1 — choose and build the corpus

The `default` corpus is the original vitamin D material and you have decided
Vectora should not be a medical system. Nothing else can be measured until a
corpus is chosen, because every number below is corpus-specific.

```bash
vectora fetch presets
vectora fetch arxiv --preset retrieval --limit 200
vectora ingest --corpus retrieval
vectora index  --corpus retrieval
```

`retrieval` (cs.IR) is the natural pick: it makes the demo self-describing —
a retrieval system you can ask about retrieval — and the vocabulary in the
preset is already right. `ml-systems` pairs better with
[Vantage](https://github.com/SiddharthaRaja/vantage) if you would rather the
two projects share a subject.

Then decide what to do with `default`. Either keep it as evidence the system
is domain-independent, or `vectora corpus delete default --everything`.
Keeping it costs nothing and demonstrates something; just say in the README
which corpus the published numbers come from.

## v4.2 — draft, then earn, the golden set

```bash
vectora golden generate --corpus retrieval --count 50 --negatives 15
```

Then **read it**. The generator verifies each question against real
retrieval, so the cases it marks `weak` sit at the edge of what the system
finds — those are the informative ones. The commented-out `UNVERIFIED` block
at the end needs a human decision: a fair question that retrieval misses is
the most valuable case in the file.

Target ~50 cases with ~15 negatives. Below about 30, differences under 0.1
recall are not distinguishable from noise and the ablation will mostly say
so.

The negatives matter most, and the adjacent-but-absent ones most of all —
neighbouring subject matter the corpus happens not to cover is where RAG
systems actually fail, because retrieval returns something plausible and the
guard is the only thing standing between that and a confident wrong answer.

## v4.3 — calibrate the refusal thresholds

```bash
vectora calibrate --corpus retrieval
```

This is the most consequential unmeasured number in the system. The floors
in `.env.example` are reasoned from how the two models score, not measured
on any corpus. Until this runs, the refusal behaviour is a hypothesis.

Commit the measured values and put the score distribution in the commit
message. If the two distributions overlap, say so — that is a real finding
about the corpus, not a failure of the method.

## v4.4 — run the ablation and publish it

```bash
vectora ablate  --corpus retrieval
vectora ablate  --corpus retrieval --sweep
vectora answers --corpus retrieval --limit 20
```

The ablation table is the single most convincing artifact this repository
can produce. It is the difference between "uses hybrid retrieval with
reranking" and "hybrid retrieval improves recall@5 by 0.13 (95% CI
0.06–0.20) over dense-only on a 50-case set; reranking's effect is inside
the noise at this sample size".

Put the markdown report from `corpora/retrieval/reports/` in the README.
Report the intervals, and where a difference is not significant, say that —
it reads as more credible than a table of confident three-decimal numbers,
because it is.

If a stage turns out not to earn its latency, **delete it**. Finding one is
a better result than confirming everything helps.

## v4.5 — benchmark the serving layer with Vantage

Point [Vantage](https://github.com/SiddharthaRaja/vantage) at a running
`vectora serve`, using the golden set as the workload. Produces latency
distribution, throughput ceiling, SLO goodput and cost per question for this
system specifically, and cross-links the two projects into one story.

Turn tracing on first (`VECTORA_ENABLE_TRACING=true`): the spans make the
latency numbers *explainable* rather than merely observed. A p95 of 9
seconds is a fact; a span tree showing 6 of them in a serial expansion call
is a finding.

## v4.6 — a demo someone can click

A read-only deployment with the index baked in and generation
rate-limited. A reviewer who can try it forms an opinion in ten seconds that
no README buys.

Decisions to make: where it runs, whether generation is enabled at all (a
search-only demo needs no API key and cannot be abused), and what the rate
limit is. `VECTORA_RATE_LIMIT_PER_MINUTE` already exists; `/search` is the
endpoint to expose if you want zero marginal cost.

## v4.7 — re-ingest the original corpus, if you keep it

Only if `default` stays. Around 1,500 of its chunks lost their inter-word
spaces in the v1 parse and are now detected and dropped; the affected papers
are named in the ingestion report. Re-ingesting from the PDFs recovers that
text and the real titles. Needs the source PDFs, which are not in the repo.

## v4.8 — a sparse index that scales

BM25 is in-process, which is right for tens of thousands of chunks and wrong
past a few hundred thousand. At that point it should move to Qdrant's native
sparse vectors so one store serves both retrievers. The `SparseIndex`
interface is deliberately narrow enough that this is a contained change —
worth doing when a corpus actually demands it, not before.

## Housekeeping

- Tag `v3.0.0`. Keep `CHANGELOG.md` current.
- Open an issue per item above before starting it. The issue-then-commit
  loop is the clearest evidence a repository has been worked on rather than
  generated.
- CI runs tests, lint and a Docker build on every push. Keep it green; a red
  badge on a public repo costs more than the feature that broke it.
- The README's numbers should always say which corpus and which commit they
  came from. A benchmark without that is decoration.

---

## Deliberately not planned

Worth writing down so they are visibly choices rather than omissions.

- **A learned reranker fine-tuned on the corpus.** Would likely help, needs
  labelled data this project does not have, and the ablation cannot
  attribute a gain that comes with a different model.
- **Query-side caching of whole answers.** Tempting and wrong: answers
  depend on conversation history, and a stale answer presented as fresh is
  the kind of bug that is very hard to notice.
- **Multi-hop retrieval.** A real improvement for questions that need two
  facts combined, and a large amount of machinery. Not until the single-hop
  numbers are measured and good.
- **Agentic retrieval loops.** The same objection, more so. Measure first.
