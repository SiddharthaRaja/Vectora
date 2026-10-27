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
Each answer was labelled by hand one claim at a time (supported / not
supported / not a claim) with the judge's scores hidden; an answer counts as
faithful only if every claim is supported. Labels: `judge_labels_human.csv`
(answer level) and `human_claim_labels.json` (claim level: 133 units, 123
supported, 5 not supported, 5 not claims). Scores: `judge_agreement.json`.

| | human: faithful | human: not faithful |
|---|---|---|
| judge (qwen2.5:7b, score ≥ 70): faithful | 25 | **4** |
| judge: not faithful | 1 | 0 |

Agreement 83% (25 + 0 of 30), Cohen's κ −0.06: **the judge does no better
than chance.** The headline agreement comes entirely from both sides calling
most answers faithful. It passed all 4 answers the human failed, scoring them
75, 100, 85 and 80; its one failure (60) was an answer the human found
faithful. As a ranking it is weak too: the chance that a faithful answer
outscores an unfaithful one is 0.63 (0.5 = random). Raising the threshold to
80–95 does not rescue it (κ 0.13).

What the judge missed (one unsupported claim each, inside otherwise
supported answers):

- q-4db3c2ca6d: a dataset described as "blogs published from 2001 to 2018",
  attributed to a source that does not say so (score 100).
- q-31b20a4bab: a method's motivation attributed to source [7], which does
  not state it (85).
- q-93e8ce6236 and q-3b7798a6bc: statements about what the sources *do not*
  contain that misdescribe them (80, 75).

The judge grades the answer as a whole; one wrong sentence among many right
ones does not move its score. Claim-level checking is what catches it.

Two answers were marked faithful with a note that questions a citation
(q-7f452eb145: the source supports "data collection", not "data
validation"; q-968b7c3553: a stray [4]). Counting those as unfaithful
changes nothing: agreement 77%, κ −0.06, the judge passed all 6.

**Conclusion.** Judged groundedness numbers in this repo (e.g. 85 / 87 in
deep mode) are not evidence of faithfulness. Use claim-level human checks,
or a claim-level judge validated the same way, before reporting
faithfulness. 30 answers and 4–6 negatives are a small sample; the
direction (judge passes everything) is the finding, not the exact κ.
