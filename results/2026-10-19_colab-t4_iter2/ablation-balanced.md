# Ablation — `qasper-sentiment-clean` corpus

2026-10-19 · balanced mode · 88 cases, 21 negatives

| variant | recall | MRR | nDCG | p50 ms | refused | Δrecall vs baseline |
|---|---:|---:|---:|---:|---:|---|
| sparse-only | 0.580 | 0.483 | 0.499 | 201 | 29% | **+0.341** (+0.227..+0.455) |
| no-expansion\* | 0.580 | 0.495 | 0.500 | 139 | 29% | **+0.341** (+0.227..+0.455) |
| no-hyde | 0.545 | 0.477 | 0.476 | 185 | 29% | **+0.307** (+0.205..+0.420) |
| hybrid+rerank+mmr | 0.239 | 0.205 | 0.210 | 217 | 52% | _baseline_ |
| no-mmr | 0.170 | 0.156 | 0.158 | 181 | 52% | -0.068 (n.s.) |
| no-rewrite | 0.091 | 0.078 | 0.080 | 178 | 71% | **-0.148** (-0.239..-0.057) |
| dense-only | 0.080 | 0.072 | 0.074 | 210 | 76% | **-0.159** (-0.239..-0.080) |
| no-rerank | 0.045 | 0.038 | 0.040 | 157 | 0% | **-0.193** (-0.284..-0.102) |
| retrieval-only\* | 0.045 | 0.045 | 0.045 | 30 | 0% | **-0.193** (-0.284..-0.102) |

Δ is the mean paired difference in recall against the shipped
configuration, with a 95% bootstrap confidence
interval. Bold means the interval excludes zero; `n.s.` means it does
not, so the difference is not distinguishable from noise at this
sample size. `*` marks variants that change more than one thing.

Reproduce with:

```bash
vectora ablate --corpus qasper-sentiment-clean --mode balanced
```
