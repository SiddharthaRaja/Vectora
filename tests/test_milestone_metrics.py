"""
Tests for the metrics added for the project milestones: answer token F1,
evidence-level retrieval relevance, the pre-strip citation hallucination
rate, judge agreement, and the QASPER golden-set builder's helpers.
"""

from __future__ import annotations

import asyncio
import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation import answers as ans
from evaluation.judge_agreement import cohen_kappa, export, score
from evaluation.run_eval import Case, load_dataset
from evaluation.textmatch import evidence_match, normalise, token_scores
from vectora.config import SearchMode
from vectora.generation.citations import parse_markers
from vectora.models import Answer, RetrievedChunk

EVIDENCE = (
    "We evaluate on the SemEval 2017 Task 4 dataset of English tweets, which "
    "contains about fifty thousand labelled examples across three classes."
)


# ── token F1 ─────────────────────────────────────────────────────────────


class TestTokenF1:
    def test_normalisation_drops_case_punctuation_articles_and_citations(self):
        assert normalise("The Model, [2] uses an LSTM!") == "model uses lstm"

    def test_exact_match_is_one(self):
        assert token_scores("BiLSTM with attention", ["bilstm with attention"]) == (1.0, 1.0)

    def test_best_reference_wins(self):
        f1, recall = token_scores("They use SemEval 2017", ["Twitter", "SemEval 2017"])
        assert recall == 1.0 and 0 < f1 < 1

    def test_long_answer_has_full_recall_but_low_f1(self):
        """Why token_recall is reported next to F1 for QASPER's short references."""
        long = "The authors evaluate on SemEval 2017 " + "and discuss many other things " * 5
        f1, recall = token_scores(long, ["SemEval 2017"])
        assert recall == 1.0 and f1 < 0.3

    def test_no_references_scores_zero(self):
        assert token_scores("anything", []) == (0.0, 0.0)


# ── evidence relevance ──────────────────────────────────────────────────


def _rc(chunk) -> RetrievedChunk:
    return RetrievedChunk(chunk=chunk)


class TestEvidenceRelevance:
    def test_chunk_containing_evidence_matches_despite_wrapping(self):
        chunk = "Setup.\n\nWe evaluate on the SemEval 2017 Task 4 dataset of English\ntweets, which contains about fifty thousand labelled examples across three classes. More."
        assert evidence_match(chunk, [EVIDENCE])

    def test_unrelated_chunk_does_not_match(self):
        assert not evidence_match("Our model is a transformer trained with Adam for ten epochs on GPUs.", [EVIDENCE])

    def test_evidence_beats_paper_match(self, chunk_factory):
        """Right paper, wrong passage: a miss once evidence is known."""
        case = Case(id="q", query="x", relevant_papers=["p1"], relevant_evidence=[EVIDENCE])
        wrong_section = _rc(chunk_factory("We thank the reviewers for their helpful comments on drafts.", paper_id="p1"))
        right = _rc(chunk_factory(EVIDENCE, paper_id="p1"))
        assert not case.is_relevant(wrong_section)
        assert case.paper_hit(wrong_section)
        assert case.is_relevant(right)

    def test_without_evidence_paper_match_still_counts(self, chunk_factory):
        case = Case(id="q", query="x", relevant_papers=["p1"])
        assert case.is_relevant(_rc(chunk_factory("anything at all here", paper_id="p1")))

    def test_dataset_loads_new_fields_and_keeps_chunk_ids_as_strings(self, tmp_path):
        path = tmp_path / "g.yaml"
        path.write_text(
            "cases:\n"
            "  - id: a\n    query: q\n    relevant_papers: [p1]\n"
            f"    relevant_evidence: [\"{EVIDENCE}\"]\n"
            "    reference_answers: [SemEval 2017, tweets]\n"
            "    relevant_chunks: [\"12e45678\"]\n"
            "negative_cases: []\n"
        )
        (case,), _ = load_dataset(path)
        assert case.relevant_evidence == [EVIDENCE]
        assert case.reference_answers == ["SemEval 2017", "tweets"]
        assert case.relevant_chunks == ["12e45678"]


# ── pre-strip citation hallucination ────────────────────────────────────


class _Engine:
    def __init__(self, result):
        self.result = result

    async def answer(self, query, mode):
        return self.result


def _score(result, references=None):
    container = SimpleNamespace(engine=_Engine(result), llm=None)
    return asyncio.run(
        ans.score_case(
            container, case_id="c", query="q", anchors=[], expected_refusal=False,
            mode=SearchMode.FAST, judge=False, references=references,
        )
    )


class TestPreStripCitations:
    def test_parse_markers_sees_out_of_range_numbers(self):
        assert parse_markers("A [1]. B [7][2]. C [1-3].") == [1, 7, 2, 3]

    def test_hallucinated_markers_are_counted_even_though_answer_is_clean(self):
        result = Answer(
            query="q", answer="Claim one [1]. Claim two.",
            raw_citation_markers=[1, 7], invalid_citation_markers=[7],
        )
        s = _score(result)
        assert s.citation_validity == 1.0  # post-strip: by construction
        assert s.raw_citation_validity == 0.5
        assert s.hallucinated_markers == 1

    def test_report_pools_the_rate_over_markers(self):
        good = ans.AnswerScore("a", "q", True, False, raw_markers=3, hallucinated_markers=0)
        bad = ans.AnswerScore("b", "q", True, False, raw_markers=1, hallucinated_markers=1)
        report = ans.AnswerReport(scores=[good, bad])
        assert report.hallucinated_rate == 0.25
        assert report.as_dict()["mechanical"]["answers_with_hallucinated_citation"] == 1

    def test_token_f1_is_scored_against_references(self):
        s = _score(Answer(query="q", answer="They use SemEval 2017 [1]."), ["SemEval 2017"])
        assert s.token_recall == 1.0 and s.token_f1 > 0.5
        assert s.answer_text.startswith("They use")


# ── judge agreement ─────────────────────────────────────────────────────


class TestJudgeAgreement:
    def test_kappa(self):
        assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
        assert cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == 0.0

    def test_export_then_score_round_trip(self, tmp_path):
        per_case = [
            {"case_id": f"c{i}", "query": "q", "refused": False, "groundedness": g,
             "answer_text": "answer [1]", "cited_chunk_ids": ["abc"]}
            for i, g in enumerate([95, 90, 40, 20])
        ]
        report = tmp_path / "answers-deep.json"
        report.write_text(json.dumps({"per_case": per_case}))
        chunks = tmp_path / "chunks.jsonl"
        chunks.write_text(json.dumps({"chunk_id": "abc", "title": "T", "content": "source text"}) + "\n")
        labels = tmp_path / "labels.csv"
        assert export(report, labels, 30, chunks, 70) == 4

        rows = list(csv.DictReader(labels.open(encoding="utf-8-sig")))
        assert "source text" in rows[0]["sources"]
        human = {"c0": "1", "c1": "0", "c2": "0", "c3": "0"}  # judge wrongly passed c1
        for r in rows:
            r["human_faithful"] = human[r["case_id"]]
        with labels.open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

        result = score(labels, 70)
        assert result["labelled"] == 4
        assert result["agreement"] == 0.75
        assert result["judge_passed_unfaithful"] == 1


# ── QASPER builder helpers ──────────────────────────────────────────────


@pytest.fixture(scope="module")
def qasper():
    path = Path(__file__).resolve().parent.parent / "scripts" / "qasper_corpus.py"
    spec = importlib.util.spec_from_file_location("qasper_corpus", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestQasperBuilder:
    QA = {
        "question": "Which sentiment lexicon do they use?",
        "answers": [
            {"answer": {"unanswerable": False, "extractive_spans": ["VADER lexicon"],
                        "evidence": ["We use the VADER lexicon BIBREF3 for scoring.",
                                     "FLOAT SELECTED: Table 2"]}},
            {"answer": {"unanswerable": False, "free_form_answer": "VADER",
                        "evidence": ["We use the VADER lexicon BIBREF3 for scoring."]}},
            {"answer": {"unanswerable": True, "evidence": []}},
        ],
    }

    def test_placeholders_are_cleaned_even_when_glued(self, qasper):
        assert qasper.clean("worldwideBIBREF0, BIBREF1 and TABREF2") == "worldwide and the table"

    def test_all_answers_and_evidence(self, qasper):
        assert qasper.all_answers(self.QA) == ["VADER lexicon", "VADER"]
        assert qasper.text_evidence(self.QA) == ["We use the VADER lexicon for scoring."]

    def test_generic_questions_are_detected(self, qasper):
        assert qasper.specific_words("What is the baseline?") < 3
        assert qasper.specific_words("Which sentiment lexicon do they use for tweets?") >= 3


class TestAnswerCheckpoints:
    def test_a_killed_run_keeps_what_it_scored(self, tmp_path, monkeypatch, settings):
        """Run 3b's deep-mode job died partway and lost everything, because the
        report was only written at the end. Now it is saved every N cases."""
        golden = tmp_path / "golden.yaml"
        golden.write_text(
            "cases:\n"
            + "".join(f"  - id: q{i}\n    query: question {i}?\n" for i in range(12))
            + "negative_cases: []\n"
        )
        ctx = SimpleNamespace(name="t", golden=golden, reports=tmp_path / "reports")
        container = SimpleNamespace(
            corpus=ctx, llm=object(), store=SimpleNamespace(exists=lambda: True)
        )

        async def aclose():
            return None

        container.aclose = aclose
        monkeypatch.setattr("vectora.api.deps.build_container", lambda *a, **k: container)

        calls = {"n": 0}

        async def fake_score(container, *, case_id, query, **_):
            calls["n"] += 1
            if calls["n"] == 8:
                raise KeyboardInterrupt("simulated kill")
            return ans.AnswerScore(case_id, query, True, False)

        monkeypatch.setattr(ans, "score_case", fake_score)
        out = tmp_path / "answers.json"
        with pytest.raises(KeyboardInterrupt):
            asyncio.run(ans.run(settings, output=out, judge=False, checkpoint_every=5))
        saved = json.loads(out.read_text())
        assert saved["partial"] is True
        assert len(saved["per_case"]) == 5
        assert saved["planned_cases"] == 12
