# Run 3c v2 — deep mode again, with a labellable export (2026-10-27)

Colab T4, Vectora `cb60612` (the export fix from `bcafb2d`), notebook
`notebooks/vectora_run3c_v2.ipynb`. Same settings and the same 100 cases as
run 3c v1 (first 80 questions + 20 negatives; gemma3:4b answers, qwen2.5:7b
judge). 37 minutes.

## Results (`expD_deep.json`)

| metric | v2 (this run) | v1 (2026-10-26) |
|---|---|---|
| answered / refused | 77 / 23 | 77 / 23 |
| citation coverage | 94.8% | 96.0% |
| raw citation validity | 99.6% (3 invented of 306, all in one answer) | 100% (0 of 293) |
| answer token recall / F1 | 0.376 / 0.135 | 0.363 / 0.132 |
| refusal accuracy | 97.0% (same 3 fair questions refused) | 97.0% |
| groundedness / relevance / completeness (judge) | 85 / 67 / 68 | 87 / 63 / 64 |
| median latency | 17.3 s | 18.8 s |

## Run-to-run noise

Identical system, identical questions. Only 10 of the 77 answers came back
word for word the same; every refusal decision matched (100 / 100).

| paired, 5,000 resamples | difference (95% CI) |
|---|---|
| deep v2 − deep v1, token recall | +0.015 [−0.038, +0.070] |
| deep v2 − deep v1, citation coverage | −0.012 [−0.044, +0.025] |
| deep v2 − fast (run 3b), token recall | +0.050 [−0.006, +0.112] |
| deep v2 − fast (run 3b), citation coverage | +0.006 [−0.032, +0.042] |

Sampling alone moves answer metrics by one to two points between identical
runs, so a difference of that size between configurations is not a finding.
Deep mode versus fast mode stays not significant in both deep runs. "0 invented
citations" in v1 was partly luck: 3 in 599 across the two runs.

## Judge agreement

`judge_labels.csv` lists, for 30 answered cases, every source each answer
cites under the number the answer uses (checked: cited = shown in 30 / 30).
The judge passes 29 of 30 at the 70 threshold, so agreement will be high by
construction and kappa will rest on very few disagreements; the cases where
the human fails an answer the judge passed are the result to read.
Labels are being made one claim at a time; results go in this folder.
