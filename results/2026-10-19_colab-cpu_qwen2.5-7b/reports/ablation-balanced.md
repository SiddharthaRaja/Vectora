# Ablation — `qasper-sentiment` corpus

2026-10-18 · balanced mode · 88 cases, 21 negatives

| variant | recall | MRR | nDCG | p50 ms | refused | Δrecall vs baseline |
|---|---:|---:|---:|---:|---:|---|
| no-mmr | 1.000 | 0.964 | 0.963 | 210 | 29% | **+0.045** (+0.011..+0.091) |
| no-hyde | 0.989 | 0.956 | 0.949 | 217 | 29% | +0.034 (n.s.) |
| no-expansion\* | 0.989 | 0.963 | 0.952 | 154 | 29% | +0.034 (n.s.) |
| sparse-only | 0.977 | 0.955 | 0.945 | 243 | 29% | +0.023 (n.s.) |
| hybrid+rerank+mmr | 0.955 | 0.925 | 0.920 | 247 | 29% | _baseline_ |
| no-rerank | 0.511 | 0.392 | 0.426 | 181 | 0% | **-0.443** (-0.557..-0.330) |
| no-rewrite | 0.159 | 0.127 | 0.132 | 186 | 76% | **-0.795** (-0.886..-0.705) |
| retrieval-only\* | 0.091 | 0.085 | 0.085 | 41 | 0% | **-0.864** (-0.932..-0.784) |
| dense-only | 0.080 | 0.074 | 0.074 | 250 | 90% | **-0.875** (-0.943..-0.795) |

Δ is the mean paired difference in recall against the shipped
configuration, with a 95% bootstrap confidence
interval. Bold means the interval excludes zero; `n.s.` means it does
not, so the difference is not distinguishable from noise at this
sample size. `*` marks variants that change more than one thing.

Reproduce with:

```bash
vectora ablate --corpus qasper-sentiment --mode balanced
```
