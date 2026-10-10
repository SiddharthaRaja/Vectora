# Roadmap — v3

v2 is the rebuild. v3 is about turning its claims into measurements and fixing
the one thing v2 could not: the corpus itself.

Ordered by value.

## v3.1 — re-ingest from the source PDFs

Put the PDFs in `papers/` and run `vectora ingest`.

This is the item everything else depends on. The shipped `chunks.jsonl` came
from the v1 pipeline and carries two faults the `repair` path can only work
around:

- **~1,500 chunks across ~30 papers lost their spaces** in PDF extraction
  (`requiremoreexpensiveequipment`). Unmatchable lexically, badly tokenised for
  the embedder. Today they are detected and dropped; re-ingesting recovers them,
  because the current parser uses a character tolerance that preserves spacing.
- **Titles fall back to filenames**, because most embedded PDF titles were
  publisher furniture. The filenames are descriptive enough that citations read
  acceptably, but the real titles are better.

## v3.2 — calibrate the refusal thresholds

```bash
vectora calibrate
```

The floors in `.env.example` were reasoned, not measured, and the README says
so. Run the calibration against the real corpus, commit the measured values,
and put the score distribution in the commit message. Until this lands, the
refusal behaviour is a hypothesis rather than a result.

## v3.3 — expand the golden set and publish retrieval metrics

Take `eval/golden.yaml` to ~50 cases including negatives, then put recall@k,
MRR and nDCG per mode — plus the refusal rate on out-of-corpus questions — in
the README. The negative cases matter most: ranking metrics cannot see whether
the system correctly declines, and that is half of whether it can be trusted.

## v3.4 — benchmark the serving layer with Vantage

Point [Vantage](https://github.com/SiddharthaRaja/vantage) at a running
`vectora serve`, using Vectora's own questions as the workload. Produces
latency distribution, throughput ceiling, SLO goodput and cost per question for
this system specifically, and cross-links the two projects into one story.

## v3.5 — expose retrieval as an MCP server

Wrap the existing `/search` endpoint so an MCP client can query the corpus
directly as a tool. About a day. Shallow engineering, high current relevance.

## v3.6 — OpenTelemetry spans per retrieval stage

The request id is already threaded through routing, expansion, retrieval and
generation. Emitting spans turns that into a per-question flame graph and makes
the Vantage latency numbers explainable rather than merely observable.

## v3.7 — tune the fusion weights against the golden set

`VECTORA_DENSE_WEIGHT` and `VECTORA_SPARSE_WEIGHT` are 1.0 and 0.8 by
judgement. Sweep them, publish the curve, keep the winner. Small, honest, and
exactly the kind of experiment that reads as research maturity.

## v3.8 — a demo someone can click

A read-only deployment with the index baked in and generation rate-limited. A
reviewer who can try it forms an opinion in ten seconds that no README buys.

## Housekeeping

Tag `v2.0.0` on merge and `v3.0.0` after v3.1. Keep a `CHANGELOG.md`. Open an
issue for each item above before starting it — the issue-then-commit loop is
the clearest evidence a repository has been worked on rather than generated.
