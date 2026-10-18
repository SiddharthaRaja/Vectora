# Experiment log

Every evaluation run gets its own folder, named `YYYY-MM-DD_<hardware>_<model>`,
holding the raw outputs it produced and a `RUN.md` with the exact setup and the
headline numbers. This table is the summary; the folders are the evidence.

Corpus for all runs so far: `qasper-sentiment` — 40 QASPER papers (CC BY 4.0),
3,185 chunks; golden set of 88 answerable questions + 21 negatives.

| run | hardware | LLM | retrieval recall (fast) | MRR | retrieval p50 | citation validity | refusal accuracy | LLM speed | notes |
|---|---|---|---|---|---|---|---|---|---|
| [2026-10-19](2026-10-19_colab-cpu_qwen2.5-7b/RUN.md) | Colab, LLM on CPU | qwen2.5:7b | 1.000 | 0.987 | 209 ms | 100% | 76.1% | 4.4 tok/s | Baseline. Ollama fell back to CPU. |

## Caveats that apply to every run

- Golden-set queries include the paper title, which makes retrieval easier than
  real use. Recall here is an upper bound until a title-free set exists.
- LLM-judged scores (groundedness, relevance, completeness) use the same model
  as the generator, so treat them as indicative only.
- The "cost" Vectora prints for local models is a gpt-4o-mini price estimate,
  not a real cost.
