"""Paired comparison of eval reports."""

from __future__ import annotations

import json

from evaluation.compare import compare, mcnemar_exact, paired


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 0) < 0.01
    assert mcnemar_exact(5, 5) == 1.0


def test_paired_counts_and_interval():
    r = paired({"a", "b", "c"}, {"c", "d"}, n=100)
    assert (r["fixed"], r["broken"]) == (2, 1)
    assert r["delta"] == 0.01
    assert r["ci_low"] <= r["delta"] <= r["ci_high"]


def test_compare_reports(tmp_path):
    def report(name, misses, recall):
        p = tmp_path / name
        p.write_text(json.dumps({"reports": [{"mode": "fast", "cases": 4, "recall": recall,
                                             "misses": misses, "paper_recall": 0.5,
                                             "negative_rejection_rate": 1.0, "median_latency_ms": 10}]}))
        return p

    (row,) = compare(report("a.json", ["q1", "q2"], 0.5), report("b.json", ["q2"], 0.75))
    assert (row["fixed"], row["broken"], row["delta"]) == (1, 0, 0.25)
