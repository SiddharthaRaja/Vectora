# 2026-10-22 — run 3b: the GPU remainder of run 3 (Colab T4)

**Code:** `56abab7`, i.e. the **new defaults** (bge-small-en-v1.5, MMR off, dense floor 0.707,
rerank floor 0.903, reranker pool 8) on qasper-300 with 365 questions and 50 off-corpus
negatives. Notebook `notebooks/vectora_run3b.ipynb`. Wall time ~2h20m.

## C2 — answers in fast mode, gemma3:4b

| | gemma3:4b (this run, new defaults) | qwen2.5:7b (run 3, old defaults, 8 negatives) |
|---|---|---|
| citation coverage | **94.1%** | 74.8% |
| raw citation validity (before stripping) | 99.8%: **4 invented markers out of 1,331** (2 answers) | 100%: 0 / 1,233 |
| answer token recall vs annotators | **0.326** | 0.302 |
| answer token F1 | 0.113 | 0.091 |
| refusal accuracy | 94.9%: 12 real questions refused, 41/50 negatives refused | 98.9% (8 negatives only) |
| latency p50 | 3.3 s | 4.8 s |

gemma cites almost every sentence and its answers contain slightly more of the gold answer, but it
does occasionally invent a source number (stripped before the user sees it), where qwen invented
none. Refusal numbers are not directly comparable: different retrieval defaults and 50 negatives
here vs 8 there.

## G — old defaults vs new defaults, **with** the LLM (balanced, query expansion on)

Same chunks; the old arm is a second corpus indexed with MiniLM.

| | old: MiniLM, MMR on, rerank floor −4.0 | **new: BGE-small, MMR off, rerank floor 0.903** |
|---|---|---|
| passage recall | 0.315 | **0.343** |
| paper recall | 0.800 | 0.800 |
| MRR / nDCG | 0.177 / 0.209 | **0.190 / 0.226** |
| off-corpus refused (50) | 78% | **96%** |
| p50 / p95 | 5.49 / 7.28 s | 5.41 / 7.41 s |

Paired over 365 questions: +0.027 passage recall (fixed 21, broke 11; 95% CI −0.003 to +0.058;
McNemar p = 0.11) — **not significant on its own**, but the same direction as every CPU test
(fast +0.041 significant; balanced-no-LLM +0.011 to +0.016). Refusals improve sharply at the same
latency.

## D — deep mode: failed

The job stopped ~15–20 minutes in with no error in its log (most likely the process was killed),
and because `vectora answers` wrote its report only at the end, nothing was kept. F (the 30-answer
export) depends on D, so it did not run. Fix: `vectora answers` now saves its report every 10
questions (`partial: true` until finished). D and F re-run in `notebooks/vectora_run3c.ipynb`
on 80 questions.
