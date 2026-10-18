# 2026-10-19 — Colab, LLM on CPU, qwen2.5:7b

**What this run is:** the first full evaluation. The notebook requested a T4 GPU,
but Ollama's installer could not detect it (Colab lacks `lspci`/`lshw`), so the
LLM ran on CPU. Embeddings and the reranker did use the GPU. This is the
"before" for the GPU comparison.

## Setup

- Commit: `48506a0`
- Corpus: `qasper-sentiment` (40 papers, 3,185 chunks)
- LLM: `qwen2.5:7b` via Ollama (answer and utility model)
- Embeddings: all-MiniLM-L6-v2 · Reranker: ms-marco-MiniLM-L-6-v2
- Refusal floors: dense 0.45, rerank -4.0

## Headline numbers

| | value |
|---|---|
| LLM generation speed | 4.36 tok/s (prompt reading 1.2 tok/s) |
| Retrieval, fast | recall 1.000 · MRR 0.987 · nDCG 0.981 · p50 209 ms |
| Retrieval, balanced | recall 0.989 · MRR 0.955 · p50 240 ms |
| Retrieval, deep | recall 0.989 · MRR 0.956 · p50 3,598 ms |
| Citation validity | 100% |
| Quote verifiability | 100% |
| Citation coverage | 74.8% |
| Refusal accuracy | 76.1% |
| Median answer latency | 9.9 s |

## Findings

- LLM expansion and HyDE (balanced) and citation verification (deep) gave no
  recall gain over fast hybrid retrieval; deep is ~15x slower.
- Over-refusal is the main quality problem: several answerable questions were
  declined.
- Similarity floors catch off-topic questions but cannot catch in-domain
  "answer not in the paper" questions (29% negative accuracy at retrieval level).

## Files

Raw outputs copied from the Colab run: `eval.json`, `ablate.json`,
`one_answer_balanced.json`, `reports/`.
