# 2026-10-20 — run 3: 300 papers, passage-level metrics (partial)

**What this run is:** the first run on `qasper-300` (300 QASPER papers, 23,448 chunks, 365 title-free
questions with annotator evidence and reference answers, 8 off-corpus negatives), and the first with
the new metrics: passage-level recall, answer token F1 / token recall, and the pre-strip
hallucinated-citation rate. Code: `9062e55` (plus `--no-llm` for the laptop parts).

Two places:
- **Colab T4** (notebook `vectora_run3.ipynb`): experiments A, B, C (qwen2.5:7b) finished; gemma3:4b
  stopped at 244/373; D (deep) started and was cut off when Colab's free GPU allowance ran out.
- **Laptop CPU, no LLM** (`vectora eval --no-llm`): fast and balanced retrieval.

## A — retrieval, 365 questions

| mode | where | passage recall@k | paper recall | MRR | nDCG | off-corpus refused | p50 / p95 |
|---|---|---|---|---|---|---|---|
| fast | T4 | 0.258 | 0.704 | 0.148 | 0.173 | 8/8 | 132 / 194 ms |
| balanced (rewrites + HyDE + rerank + MMR) | T4 | 0.293 | 0.800 | 0.173 | 0.202 | 6/8 | 5.8 / 7.8 s |
| deep | T4 | 0.290 | 0.800 | 0.173 | 0.201 | 6/8 | 3.2 / 4.3 s |
| fast | laptop CPU | 0.258 | 0.712 | 0.149 | 0.175 | 8/8 | 317 / 535 ms |
| balanced, `--no-llm` (rerank + MMR, no expansion) | laptop CPU | 0.299 | 0.819 | 0.171 | 0.201 | 6/8 | 2.1 / 4.0 s |

- **Passage vs paper:** the system finds the right paper 70–82% of the time but the passage holding
  the answer only 26–30%. The matcher was validated: the evidence is present in its paper's chunks
  for 357/365 questions.
- **The reranker helps, mostly at paper level:** fast → balanced-no-LLM (CPU, same machine) is
  +0.041 passage recall (39 questions fixed, 24 broken; paired 95% CI 0.000 to +0.082; McNemar
  p = 0.077) and +0.107 paper recall.
- **LLM query expansion adds nothing here:** balanced with rewrites + HyDE (0.293) is no better than
  balanced without them (0.299), and costs ~3.7 s of LLM time per question.
- Small T4-vs-CPU differences in paper recall (0.704 vs 0.712) come from different embedding
  hardware; compare modes within one machine.

## B — refusal calibration

| mode | signal | real questions | off-corpus | suggested floor | result |
|---|---|---|---|---|---|
| fast | dense cosine | min 0.349, p10 0.525 | max 0.345 | 0.347 | clean separation: 0 real refused, 8/8 caught |
| balanced | rerank logit | min −5.04, p10 +1.27 | max −2.40 | −0.26 | overlap: refuses 18/365 real to catch 8/8 |
| balanced | dense cosine | min 0.442 | max 0.468 | 0.529 | overlap: refuses 14/365 real to catch 8/8 |

The current rerank floor (−4.0) is why balanced answers 2 of 8 off-corpus questions. Only 8
negatives: these floors are indicative.

## C — answers in fast mode, qwen2.5:7b (365 questions + 8 negatives)

| | value |
|---|---|
| answered / refused | 361 / 12 (4 real questions refused, 8/8 negatives refused) |
| refusal accuracy | 98.9% |
| **raw citation validity (before stripping)** | **100%: 0 hallucinated markers out of 1,233** |
| citation coverage | 74.8% of substantive sentences cite a source (3 answers cite nothing) |
| answer token recall vs annotators | mean 0.302, median 0.214; 23.8% of answers contain ≥ half the reference |
| answer token F1 | mean 0.091 (low by construction: short span references vs ~480-character cited answers) |
| latency p50 / p95 | 4.8 / 8.3 s |

gemma3:4b (run 2's pick) reached 244/373 before the session moved on; 2 of its answers failed
with Ollama's "token repeat limit reached" (the model looped). Re-run in run 3b.

## Not done here
- D (deep, grounding judge) and F (30-answer export): cut off — `vectora_run3b.ipynb`.
- Ablation: moved to the laptop CPU, `vectora ablate --no-llm`.

## E — ablation (laptop CPU, `vectora ablate --no-llm`, balanced, top-5)

| variant | passage recall@5 | Δ vs baseline (95% CI) | off-corpus refused | p50 |
|---|---|---|---|---|
| no-mmr | 0.296 | +0.019 (−0.003 to +0.038) | 6/8 | 1.8 s |
| sparse-only | 0.288 | +0.011 (−0.019 to +0.038) | 7/8 | 2.6 s |
| **hybrid + rerank + MMR (baseline)** | 0.277 | — | 6/8 | 2.6 s |
| dense-only | 0.266 | −0.011 (−0.038 to +0.016) | 6/8 | 2.6 s |
| retrieval-only (no rerank, no MMR) | 0.255 | −0.022 (−0.063 to +0.019) | 8/8 | 0.3 s |
| no-rerank | 0.244 | −0.033 (−0.077 to +0.005) | 8/8 | 1.1 s |

- **The race is gone:** the ablation baseline (0.277 at top-5) now agrees with the eval (0.299 at
  top-6). In run 2 it was 0.239 vs 0.591.
- No single stage is significant at n = 365. The reranker is the largest and closest (−0.033 without
  it); MMR, if anything, hurts passage recall (+0.019 without it) — diversifying across papers
  pushes out a second chunk from the right paper. Dense and sparse are interchangeable here.
- Every reranked variant lets 2 off-corpus questions through; the rerank floor (−4.0) needs
  recalibrating (calibration suggests about −0.26).
- Candidate default for this corpus: rerank on, MMR off — as good or better, ~30% faster. To be
  confirmed as one combined change.

## F — MMR off + rerank floor −0.26 (laptop CPU, balanced, `--no-llm`)

`VECTORA_USE_MMR=false VECTORA_MIN_RERANK_SCORE=-0.26`, compared with the same run at the defaults:

| | defaults (MMR on, floor −4.0) | MMR off, floor −0.26 |
|---|---|---|
| passage recall | 0.299 | 0.304 (8 fixed, 6 broken: no real difference) |
| paper recall | 0.819 | 0.806 |
| off-corpus refused | 6/8 | **8/8** |
| p50 / p95 | 2.10 / 4.05 s | 2.01 / 3.44 s |

The floor fixes the off-corpus leak at no recall cost. Its price is not in this table: run 3's
calibration says −0.26 also refuses about 18 of 365 real questions (~5%). Turning MMR off is
neutral on recall and slightly faster.

## G — embedding models at passage level (Colab CPU, `vectora_embeddings_cpu.ipynb`)

Same chunks (MiniLM semantic chunking) re-indexed with each model; 365 questions; paired against
MiniLM on the same questions.

| model | fast passage | fast paper | balanced passage | balanced paper | fast p50 (Colab CPU) |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 (384-d) | 0.258 | 0.712 | 0.299 | 0.819 | 364 ms |
| **bge-small-en-v1.5 (384-d)** | **0.299** | 0.734 | 0.312 | 0.808 | 623 ms |
| bge-base-en-v1.5 (768-d) | 0.299 | 0.734 | 0.307 | 0.819 | 1,780 ms |

| paired vs MiniLM | Δ passage recall | fixed / broken | 95% CI |
|---|---|---|---|
| fast, bge-small | **+0.041** | 28 / 13 | **+0.008 to +0.074** (significant) |
| fast, bge-base | +0.041 | 28 / 13 | +0.008 to +0.074 (significant) |
| balanced, bge-small | +0.014 | 14 / 9 | −0.011 to +0.038 |
| balanced, bge-base | +0.008 | 13 / 10 | −0.016 to +0.033 |

- **BGE-small is a real first-stage improvement:** +0.041 passage recall in fast mode, significant,
  same as BGE-base at a third of the latency. Run 2's "no difference" was a paper-level, 88-question
  artefact.
- **The reranker absorbs most of it:** with reranking on, the gain shrinks to +0.014 (not
  significant). Fast mode with BGE-small matches balanced mode with MiniLM (0.299) at a fraction of
  the latency.
- **Refusal:** in fast mode both BGE models refused 0/8 off-corpus questions at the MiniLM floor
  (0.40): BGE similarities sit higher, so the dense floor must be recalibrated per model (see the
  calibrate output in the run logs).
