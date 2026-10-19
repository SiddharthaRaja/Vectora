"""The CI eval gate fails on real drops, tolerates one flipped case, and bootstraps."""

from __future__ import annotations

import json

import pytest

from evaluation import gate


def _report(tmp_path, name, **overrides):
    row = {"mode": "fast", "cases": 71, "recall": 0.60, "paper_recall": 0.80,
           "mrr": 0.45, "ndcg": 0.50, "negative_rejection_rate": 0.875}
    row.update(overrides)
    path = tmp_path / name
    path.write_text(json.dumps({"reports": [row]}))
    return path


def test_unchanged_and_improved_pass(tmp_path):
    base = gate.metrics(_report(tmp_path, "b.json"))
    assert gate.compare(gate.metrics(_report(tmp_path, "n.json", recall=0.65)), base) == []


def test_one_flipped_case_is_tolerated_two_are_not(tmp_path):
    base = gate.metrics(_report(tmp_path, "b.json"))
    one = gate.metrics(_report(tmp_path, "one.json", recall=0.60 - 1 / 71))
    two = gate.metrics(_report(tmp_path, "two.json", recall=0.60 - 2 / 71))
    assert gate.compare(one, base) == []
    assert any("recall" in f for f in gate.compare(two, base))


def test_changed_eval_set_demands_a_deliberate_rebaseline(tmp_path):
    base = gate.metrics(_report(tmp_path, "b.json"))
    assert "case count changed" in gate.compare(gate.metrics(_report(tmp_path, "n.json", cases=70)), base)[0]


def test_check_without_baseline_passes_and_suggests(tmp_path, monkeypatch):
    report = _report(tmp_path, "r.json")
    suggest = tmp_path / "s.json"
    monkeypatch.setattr("sys.argv", ["gate", "check", "--report", str(report),
                                     "--baseline", str(tmp_path / "none.json"), "--suggest", str(suggest)])
    gate.main()
    assert json.loads(suggest.read_text())["fast"]["recall"] == 0.60


def test_check_exits_nonzero_on_regression(tmp_path, monkeypatch):
    baseline = tmp_path / "base.json"
    baseline.write_text(json.dumps(gate.metrics(_report(tmp_path, "b.json"))))
    report = _report(tmp_path, "r.json", mrr=0.30)
    monkeypatch.setattr("sys.argv", ["gate", "check", "--report", str(report), "--baseline", str(baseline)])
    with pytest.raises(SystemExit) as exc:
        gate.main()
    assert exc.value.code == 1
