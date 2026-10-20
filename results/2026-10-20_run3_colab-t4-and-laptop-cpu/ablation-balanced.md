# Ablation — `qasper-300` corpus

2026-10-20 · balanced mode · 365 cases, 8 negatives

| variant | recall | MRR | nDCG | p50 ms | refused | Δrecall vs baseline |
|---|---:|---:|---:|---:|---:|---|
| no-mmr | 0.296 | 0.174 | 0.202 | 1819 | 75% | +0.019 (n.s.) |
| sparse-only | 0.288 | 0.169 | 0.197 | 2608 | 88% | +0.011 (n.s.) |
| hybrid+rerank+mmr | 0.277 | 0.168 | 0.193 | 2639 | 75% | _baseline_ |
| dense-only | 0.266 | 0.166 | 0.189 | 2598 | 75% | -0.011 (n.s.) |
| retrieval-only\* | 0.255 | 0.150 | 0.175 | 290 | 100% | -0.022 (n.s.) |
| no-rerank | 0.244 | 0.148 | 0.172 | 1071 | 100% | -0.033 (n.s.) |

Δ is the mean paired difference in recall against the shipped
configuration, with a 95% bootstrap confidence
interval. Bold means the interval excludes zero; `n.s.` means it does
not, so the difference is not distinguishable from noise at this
sample size. `*` marks variants that change more than one thing.

Reproduce with:

```bash
vectora ablate --corpus qasper-300 --mode balanced
```
