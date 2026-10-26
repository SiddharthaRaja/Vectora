# Run 3c — deep mode on Colab T4 (2026-10-26)

Colab T4, Vectora `d44cc74`, qasper-300 with the current defaults (BGE-small,
MMR off, measured floors, reranker pool 8). Answers by gemma3:4b, utility and
grounding judge by qwen2.5:7b. `vectora answers --mode deep --limit 80`:
the first 80 questions plus 20 off-corpus negatives. 39 minutes, completed
(checkpointing in place; nothing lost this time).

## Results (`expD_deep.json`)

| metric | value |
|---|---|
| answered / refused | 77 / 23 (20 of 20 negatives refused, 3 of 80 real questions refused) |
| citation coverage | 96.0% |
| raw citation validity | 100% (0 invented source numbers of 293) |
| answer token recall / F1 | 0.363 / 0.132 |
| refusal accuracy | 97.0% |
| groundedness / relevance / completeness (LLM judge) | 87 / 63 / 64 |
| median latency | 18.8 s |

## Deep vs fast on the same 100 cases (paired)

Fast mode = run 3b's `expC_fast_gemma3-4b.json`, same answer model, same
corpus and defaults. 77 questions answered by both; 5,000 bootstrap resamples.

| | fast | deep | deep − fast (95% CI) |
|---|---|---|---|
| answer token recall | 0.329 | 0.363 | +0.034 [−0.034, +0.106] |
| answer token F1 | 0.112 | 0.132 | +0.021 [−0.004, +0.049] |
| citation coverage | 94.2% | 96.0% | +0.018 [−0.020, +0.055] |
| invented source numbers | 3 / 287 | 0 / 293 | too few to compare |
| real questions refused (of 80) | 1 | 3 | deep refused 2 that fast answered |
| median latency | 2.9 s | 18.8 s | 6.5× |

**Deep mode is not measurably better than fast mode at this sample size, and
costs 6.5× the latency.** Every interval spans zero. Comparing deep's 80
questions with fast's full 365 had suggested gains; paired on identical
questions they are within noise. The two extra refusals: "How is it
determined that a fact is easy-to-guess?" and "What is the average length of
dialog?".

## Judge export: not usable (bug found, fixed)

`judge_labels_UNUSABLE_unnumbered.csv` is kept as a record, not for labelling.
`vectora answers` stored only `cited_chunk_ids`, which come from
`dedupe_by_paper` (one citation per paper, for display). An answer citing
[1][2][4][5] from one paper therefore exported a single chunk, labelled by
chunk id rather than by [n]; a labeller could not see what [2] pointed at and
would mark faithful answers unfaithful. The labels would have measured the
export, not the judge.

Fixed: every `Answer` now carries `source_chunk_ids` (SOURCE [n] is element
n − 1), `vectora answers` stores it, and the export lists every cited source
under the number the answer uses (regression test included). Old reports get
a warning. The judge itself was unaffected: the grounding check always saw
all sources. Re-run deep mode to produce labellable answers.
