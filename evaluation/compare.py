"""
Paired comparison of two `vectora eval` reports.

Two configurations scored on the same golden set can be compared question by
question, which removes question difficulty from the noise: only the questions
where they disagree carry information. This prints, per mode present in both
reports, how many questions B fixed and broke relative to A, the paired
bootstrap 95% CI on the recall difference, and an exact McNemar p-value.

    python -m evaluation.compare reports/eval-fast-minilm.json reports/eval-fast-bge.json
"""

from __future__ import annotations

import argparse
import json
import random
from math import comb
from pathlib import Path


def mcnemar_exact(fixed: int, broken: int) -> float:
    """Two-sided exact McNemar test on the discordant pairs."""
    n = fixed + broken
    if n == 0:
        return 1.0
    k = min(fixed, broken)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def paired(misses_a: set[str], misses_b: set[str], n: int, *, seed: int = 0, boots: int = 4000) -> dict:
    fixed, broken = len(misses_a - misses_b), len(misses_b - misses_a)
    diff = [1] * fixed + [-1] * broken + [0] * (n - fixed - broken)
    rng = random.Random(seed)
    samples = sorted(sum(rng.choice(diff) for _ in range(n)) / n for _ in range(boots))
    return {
        "fixed": fixed,
        "broken": broken,
        "delta": (fixed - broken) / n if n else 0.0,
        "ci_low": samples[int(boots * 0.025)],
        "ci_high": samples[int(boots * 0.975) - 1],
        "p_mcnemar": mcnemar_exact(fixed, broken),
    }


def compare(path_a: Path, path_b: Path) -> list[dict]:
    a = {r["mode"]: r for r in json.loads(path_a.read_text("utf-8"))["reports"]}
    b = {r["mode"]: r for r in json.loads(path_b.read_text("utf-8"))["reports"]}
    rows = []
    for mode in a.keys() & b.keys():
        ra, rb = a[mode], b[mode]
        if ra["cases"] != rb["cases"]:
            raise SystemExit(f"{mode}: different case counts ({ra['cases']} vs {rb['cases']})")
        row = {"mode": mode, "recall_a": ra["recall"], "recall_b": rb["recall"]}
        row.update(paired(set(ra["misses"]), set(rb["misses"]), ra["cases"]))
        for k in ("paper_recall", "negative_rejection_rate", "median_latency_ms"):
            row[f"{k}_a"], row[f"{k}_b"] = ra.get(k), rb.get(k)
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("a", type=Path, help="Baseline report")
    ap.add_argument("b", type=Path, help="Candidate report")
    args = ap.parse_args()
    for r in compare(args.a, args.b):
        verdict = "significant" if r["ci_low"] > 0 or r["ci_high"] < 0 else "not distinguishable"
        print(
            f"{r['mode']:<9} passage recall {r['recall_a']:.3f} -> {r['recall_b']:.3f} "
            f"({r['delta']:+.3f}; fixed {r['fixed']}, broke {r['broken']}; "
            f"95% CI {r['ci_low']:+.3f} to {r['ci_high']:+.3f}; McNemar p {r['p_mcnemar']:.3f}; {verdict})"
        )
        extras = []
        for k, label in (("paper_recall", "paper"), ("negative_rejection_rate", "neg ok"),
                         ("median_latency_ms", "p50 ms")):
            va, vb = r[f"{k}_a"], r[f"{k}_b"]
            if va is not None and vb is not None:
                extras.append(f"{label} {va:.3g} -> {vb:.3g}")
        print("          " + ";  ".join(extras))


if __name__ == "__main__":
    main()
