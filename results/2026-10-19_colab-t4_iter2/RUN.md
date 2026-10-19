# 2026-10-19 — iteration 2: Colab T4, LLM on GPU, six experiments

**What this run is:** the follow-up to run 1. Each experiment targets one
problem run 1 exposed and changes one thing at a time. Notebook:
`vectora_iter2_overnight.ipynb`. Wall time 2h35m on a free T4.

## Setup

- Commit: `0f25c3b` (plus a cleaned corpus and title-free questions built in the notebook)
- Corpora: `qasper-sentiment` (raw, 3,185 chunks) and `qasper-sentiment-clean`
  (2,900 QASPER placeholder tokens — BIBREF, INLINEFORM, TABREF… — removed; years added)
- Golden set: 88 questions + 21 negatives, in two forms: **titled**
  (`In the paper "<title>": …`, as run 1) and **title-free**
- LLMs via Ollama on the GPU; `qwen2.5:7b` is the utility model and the judge throughout
- Embeddings all-MiniLM-L6-v2, reranker ms-marco-MiniLM-L-6-v2, floors dense 0.45 / rerank -4.0

## Headline numbers

| | run 1 | run 2 |
|---|---|---|
| LLM speed, qwen2.5:7b | 4.36 tok/s (CPU) | **29.6 tok/s** (GPU, 6.8x) |
| Prompt processing | 1.2 tok/s | **156 tok/s** |
| Median answer latency (fast) | 9.9 s | **2.0 s** (gemma3:4b) |
| Retrieval recall@5, fast, titled | 1.000 | 1.000 |
| Retrieval recall@5, fast, **title-free** | — | **0.545** |
| Citation validity / quote verifiability | 100% / 100% | 100% / 100% |
| Citation coverage | 74.8% | **95.2%** (gemma3:4b) |

## Experiment 0 — raw generation speed on the T4

| model | generation tok/s | prompt tok/s |
|---|---|---|
| qwen2.5:7b | 29.6 | 156 |
| llama3.1:8b | 27.7 | 120 |
| mistral:7b | 29.9 | 101 |
| gemma3:4b | 50.3 | 133 |
| qwen2.5:3b | 62.4 | 283 |

Run 1 had requested a GPU, but Ollama's installer could not detect it on Colab
(no `lspci`/`lshw`) and fell back to CPU. Installing `pciutils lshw` fixed it.

## Experiments 1–2 — what retrieval really scores

recall@5 / MRR, 88 questions:

| | fast | balanced | deep |
|---|---|---|---|
| A. raw corpus, titled questions | 1.000 / 0.987 | 0.989 / 0.955 | 0.989 / 0.956 |
| B. raw corpus, title-free | 0.523 / 0.446 | 0.602 / 0.487 | 0.580 / 0.468 |
| C. clean corpus, title-free | 0.545 / 0.450 | 0.591 / 0.469 | 0.568 / 0.470 |

- **The title prefix was doing most of the work.** Removing it halves recall.
  Part of this is fair: many QASPER questions ("What are the baselines?") do
  not identify a paper without its title. 0.55 is a lower bound and 1.00 an
  upper bound on realistic recall.
- **Cleaning the placeholders: +0.022 in fast mode** — within noise at n=88.
  Kept, because it also removes junk from answers.
- **Query expansion (balanced) adds +0.05 recall over fast**, at ~5 s of extra
  LLM latency per question.

## Experiment 3 — refusal threshold sweep (balanced, clean, title-free)

| rerank floor | refusal accuracy | citation coverage |
|---|---|---|
| -4.0 | 83.5% | 67.9% |
| -6.0 | 81.7% | 69.0% |
| -8.0 | 83.5% | 70.3% |

The reranker floor is **not** what causes over-refusal; moving it barely matters.

## Experiment 4 — answer model comparison (fast, clean, title-free; judge fixed)

| model | citation coverage | anchor recall | relevance | completeness | median latency |
|---|---|---|---|---|---|
| qwen2.5:7b | 74.7% | 100% | 59 | 60 | 3.8 s |
| llama3.1:8b | 70.4% | 100% | 60 | 64 | 4.8 s |
| mistral:7b | 72.1% | 94.7% | **68** | **68** | 6.4 s |
| **gemma3:4b** | **95.2%** | 98.9% | 65 | 66 | **2.0 s** |
| qwen2.5:3b | 84.2% | 100% | 56 | 54 | 1.2 s |

All five refused **the same 15 questions** (refusal accuracy 79.8% for every
model): refusals are made by the retrieval guard before the LLM is called.
Title-free questions score lower on dense similarity (p10 0.455), so the 0.45
dense floor now cuts real questions. Calibration suggests 0.40.

**Pick: gemma3:4b** — best citation coverage, second-fastest, half the size of qwen2.5:7b.
Groundedness reads 0/100 for every model: that judge only runs in deep mode, ignore it.

## Experiment 5 — ablation (balanced): invalid, see bug below

The ablation baseline ("hybrid+rerank+mmr") scored recall 0.239, while the
identical configuration in experiment 1 row C scored 0.591. The run printed
numpy overflow warnings from `qdrant_client/local/distances.py`, and
calibration reported a dense cosine of 2.4e13. These results are not used.

**Cause (fixed in the commit after this run):** embedded Qdrant's cosine search
normalises the stored vector matrix *in place*, and Vectora issues one dense
search per query variant concurrently from worker threads. In balanced/deep
mode those writes race; reproduced in isolation (concurrent cosine searches
over a non-unit matrix return "cosine" scores of 9 and 560). `fast` mode
(one dense search) is unaffected. Fix: serialise embedded-Qdrant calls with a
lock. To be confirmed in run 3 by re-running the ablation.

## Experiment 7 — retrieval tuning (fast, clean, title-free)

| variant | recall@5 | MRR | neg refused |
|---|---|---|---|
| baseline (MiniLM, max 420 tokens) | 0.545 | 0.450 | 33% |
| bge-small-en-v1.5 | 0.568 | 0.461 | 0% |
| bge-base-en-v1.5 | 0.534 | 0.476 | 0% |
| max chunk 250 | 0.545 | 0.450 | 33% |
| max chunk 600 | 0.545 | 0.450 | 33% |
| fusion sparse weight 0.0–1.6 | 0.523–0.545 | 0.402–0.460 | 33–38% |

- **Embedding model:** no significant gain at n=88. BGE refused 0% of
  negatives because its similarity scores sit higher overall — the 0.45 floor
  was calibrated for MiniLM. Refusal thresholds are model-specific.
- **Chunk size cap: no effect.** Semantic chunking splits QASPER's short
  paragraphs well below 250 tokens, so the cap rarely binds.
- **Fusion weights: flat.** No setting is distinguishable from the default.

## Latency breakdown (single CLI calls)

Single `vectora ask` calls were dominated by cold starts (model loading:
~10.6 s of the 10.8 s in fast mode). Warm figures come from the evals:
retrieval 30 ms p50 in fast mode; answers 1.2–6.4 s depending on model.

## Decisions for run 3

1. Default answer model → gemma3:4b.
2. Dense floor → 0.40 for title-free questions with MiniLM.
3. Re-run the ablation with the embedded-Qdrant lock; confirm baseline ≈ eval.
4. Build a title-free golden set whose questions identify their paper, so
   recall measures retrieval rather than question ambiguity.
