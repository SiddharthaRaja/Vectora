# Experiment log

Every evaluation run gets its own folder, named `YYYY-MM-DD_<hardware>_<model>`,
holding the raw outputs it produced and a `RUN.md` with the exact setup and the
headline numbers. This table is the summary; the folders are the evidence.

Corpus for runs 1–2: `qasper-sentiment` — 40 QASPER papers (CC BY 4.0),
3,185 chunks; golden set of 88 answerable questions + 21 negatives. Run 3 onwards:
`qasper-300` — 300 papers, 23,448 chunks, 365 title-free questions with evidence and reference
answers + 8 off-corpus negatives. From run 3, recall is passage-level (paper-level shown beside it).

| run | hardware | LLM | retrieval recall (fast) | MRR | retrieval p50 | citation validity | refusal accuracy | LLM speed | notes |
|---|---|---|---|---|---|---|---|---|---|
| [2026-10-19](2026-10-19_colab-cpu_qwen2.5-7b/RUN.md) | Colab, LLM on CPU | qwen2.5:7b | 1.000 | 0.987 | 209 ms | 100% | 76.1% | 4.4 tok/s | Baseline. Ollama fell back to CPU. |
| [2026-10-19 #2](2026-10-19_colab-t4_iter2/RUN.md) | Colab T4, LLM on GPU | gemma3:4b (of 5 compared) | 1.000 titled · **0.545 title-free** | 0.987 · 0.450 | 30 ms | 100% | 79.8% | 29.6 tok/s (qwen2.5:7b) · 50.3 (gemma3:4b) | GPU fix; honest title-free recall; 5-model comparison; found embedded-Qdrant race |
| [2026-10-20 #3](2026-10-20_run3_colab-t4-and-laptop-cpu/RUN.md) | Colab T4 + laptop CPU | qwen2.5:7b (gemma3:4b pending) | **qasper-300, passage-level: 0.258** (paper 0.704) · balanced 0.293 | 0.148 · 0.173 | 132 ms | **raw (pre-strip) 100%: 0 / 1,233 hallucinated** | 98.9% | — | 300 papers; passage vs paper recall; reranker +0.04 passage (p=0.077), LLM expansion adds nothing; deep + gemma pending (run 3b) |
| [2026-10-22 #3b](2026-10-22_run3b_colab-t4/RUN.md) | Colab T4 | gemma3:4b (answers), qwen2.5:7b (utility) | new defaults (BGE-small, no MMR, pool 8): balanced **0.343** vs old 0.315 (with LLM) | 0.190 | — | raw 99.8% (4 / 1,331 invented) | 94.9% (50 negatives) | — | old vs new defaults with the LLM: +0.027 recall (n.s.), off-corpus refused 78% → 96%; gemma cites 94% of sentences; deep mode died (now checkpointed) → run 3c |

## Caveats that apply to every run

- Titled queries (`In the paper "<title>": …`) make retrieval far easier than
  real use. From run 2 on, the title-free number is the one to quote; it is a
  lower bound, since some QASPER questions cannot identify a paper on their own.
- LLM-judged scores (groundedness, relevance, completeness) come from a 7B
  judge (qwen2.5:7b), so treat them as indicative only.
- The run-1 retrieval p50 (209 ms) included CPU-side work that run 2 moved to
  the GPU; compare latencies within a run, not across them.
- The "cost" Vectora prints for local models is a gpt-4o-mini price estimate,
  not a real cost.
