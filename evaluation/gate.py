"""
Eval regression gate: fail CI when retrieval quality drops.

Compares a fresh ``vectora eval --output`` report with a committed baseline
(``evaluation/ci_baseline.json``) and exits non-zero if any gated metric
fell by more than its tolerance. Improvements never fail; they are printed
so the baseline can be raised on purpose.

    python -m evaluation.gate check  --report ci-eval.json
    python -m evaluation.gate update --report ci-eval.json   # accept new numbers

With no baseline yet, ``check`` passes, prints the numbers it saw and writes
them to ``--suggest`` so the first CI run can bootstrap the baseline.

The tolerances are not arbitrary noise allowances: retrieval in fast mode is
deterministic on CPU, so on the same code two runs give the same numbers.
They exist because the 71-case CI set moves in steps of 1/71 ≈ 0.014: one
case flipping should not block a merge, two should.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_BASELINE = Path(__file__).resolve().parent / "ci_baseline.json"

#: metric -> largest allowed drop
TOLERANCE = {
    "recall": 0.02,
    "paper_recall": 0.02,
    "mrr": 0.04,
    "ndcg": 0.04,
    "negative_rejection_rate": 0.13,  # 8 negatives: one flip is 0.125
}


def metrics(report_path: Path) -> dict[str, dict[str, float]]:
    """{mode: {metric: value}} from an eval report."""
    report = json.loads(report_path.read_text("utf-8"))
    out: dict[str, dict[str, float]] = {}
    for r in report["reports"]:
        out[r["mode"]] = {k: r[k] for k in TOLERANCE if r.get(k) is not None}
        out[r["mode"]]["cases"] = r["cases"]
    return out


def compare(
    current: dict[str, dict[str, float]], baseline: dict[str, dict[str, float]]
) -> list[str]:
    failures: list[str] = []
    for mode, base in baseline.items():
        now = current.get(mode)
        if now is None:
            failures.append(f"{mode}: missing from the new report")
            continue
        if now.get("cases") != base.get("cases"):
            failures.append(
                f"{mode}: case count changed ({base.get('cases')} -> {now.get('cases')}); "
                "the eval set changed, so re-baseline deliberately with `update`"
            )
            continue
        for metric, tol in TOLERANCE.items():
            if metric not in base:
                continue
            old, new = base[metric], now.get(metric)
            if new is None:
                failures.append(f"{mode}.{metric}: missing")
                continue
            delta = new - old
            flag = "FAIL" if delta < -tol else ("up  " if delta > 1e-9 else "ok  ")
            print(f"  {flag} {mode:<9} {metric:<24} {old:.4f} -> {new:.4f} ({delta:+.4f}, tol {tol})")
            if delta < -tol:
                failures.append(f"{mode}.{metric} dropped {old:.4f} -> {new:.4f}")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["check", "update"])
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--suggest", type=Path, default=Path("ci_baseline.suggested.json"))
    args = ap.parse_args()

    current = metrics(args.report)
    if args.cmd == "update":
        args.baseline.write_text(json.dumps(current, indent=2) + "\n", "utf-8")
        print(f"Baseline written to {args.baseline}")
        return

    if not args.baseline.exists():
        args.suggest.write_text(json.dumps(current, indent=2) + "\n", "utf-8")
        print(f"No baseline at {args.baseline}; passing. Numbers seen:\n"
              f"{json.dumps(current, indent=2)}\n"
              f"Commit them as the baseline: copy {args.suggest} to {args.baseline}.")
        return

    print("Eval gate:")
    failures = compare(current, json.loads(args.baseline.read_text("utf-8")))
    if failures:
        print("\nRetrieval regressed:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("\nNo regression.")


if __name__ == "__main__":
    main()
