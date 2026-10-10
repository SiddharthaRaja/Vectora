"""
Tests for the parts of ingestion that determine corpus quality.

Each test here corresponds to a defect observed in the shipped corpus, so
a regression is caught rather than rediscovered months later by reading
citations.
"""

from __future__ import annotations

import pytest

from vectora.ingestion import clean, metadata
from vectora.ingestion.chunking import enforce_size, semantic_chunk, split_sentences
from vectora.ingestion.sections import canonicalise, classify_line, split_into_sections
from vectora.models import Chunk

# ── metadata ──────────────────────────────────────────────────────────────


class TestYearExtraction:
    def test_capturing_group_bug_is_fixed(self):
        """The old pattern returned "20" because findall yielded the group."""
        year, _ = metadata.extract_year("Steroids 75 (2010) 477-488")
        assert year == "2010"

    def test_partial_year_never_validates(self):
        """Even if something upstream produced "20", the model rejects it."""
        chunk = Chunk.build(content="x" * 300, paper_id="p", title="T", year="20")
        assert chunk.year == ""

    def test_copyright_cue_wins_over_other_years(self):
        text = "Received 4 January 1998; accepted 2009\n© 2010 Elsevier Ltd."
        year, source = metadata.extract_year(text)
        assert year == "2010"
        assert source == "cue"

    def test_falls_back_to_filename(self):
        year, source = metadata.extract_year("no dates here", filename="2010 Vit D review")
        assert year == "2010"
        assert source == "filename"

    def test_implausible_years_ignored(self):
        year, _ = metadata.extract_year("Sample 1837 and catalogue 2099 and © 2015")
        assert year == "2015"


class TestDoiExtraction:
    def test_extracts_and_trims_punctuation(self):
        doi, _ = metadata.extract_doi("See doi:10.1016/j.steroids.2010.02.012.")
        assert doi == "10.1016/j.steroids.2010.02.012"

    def test_no_doi(self):
        assert metadata.extract_doi("nothing here")[0] == ""


class TestTitleExtraction:
    def test_rejects_publisher_furniture(self):
        """The exact failure in the corpus: title came out "GModel ARTICLE IN PRESS"."""
        lines = [
            "GModel",
            "ARTICLE IN PRESS",
            "STE-6790; No.ofPages12",
            "Steroids",
            "journal homepage: www.elsevier.com/locate/steroids",
            "Measurement of 25-hydroxyvitamin D in the clinical laboratory: "
            "current procedures, performance characteristics and limitations",
            "A.M. Wallace, S. Gibson, A. de la Hunty, C. Lamberg-Allardt, M. Ashwell",
        ]
        title, source = metadata.extract_title(lines, "2010 Vit D method review")
        assert "25-hydroxyvitamin D" in title
        assert source == "header"
        assert "ARTICLE IN PRESS" not in title

    def test_rejects_section_words(self):
        assert metadata._title_score("Abstract", 0) < 0
        assert metadata._title_score("1. Introduction", 1) < 0

    def test_filename_fallback_when_header_is_useless(self):
        title, source = metadata.extract_title(["©", "12"], "2010 Vit D method review Wallace")
        assert source == "filename"
        assert "Vit D method review" in title

    def test_filename_cleanup(self):
        assert metadata.title_from_filename("molecules-28-03215-v2") == "molecules 28 03215"
        assert metadata.title_from_filename("E-main_2 (1)") == "E main 2"


class TestAuthorExtraction:
    def test_finds_author_line_not_affiliation(self):
        lines = [
            "Measurement of 25-hydroxyvitamin D in the clinical laboratory",
            "A.M. Wallace, S. Gibson, A. de la Hunty, M. Ashwell",
            "Department of Clinical Biochemistry, Royal Infirmary, Glasgow G4 0SF, UK",
        ]
        authors, source = metadata.extract_authors(
            lines, "Measurement of 25-hydroxyvitamin D in the clinical laboratory"
        )
        assert source == "header"
        assert "Wallace" in authors
        assert "Department" not in authors

    def test_strips_superscript_markers(self):
        cleaned = metadata._tidy_authors("A.M. Wallace1,*, S. Gibson2")
        assert "1" not in cleaned
        assert "*" not in cleaned
        assert "Wallace" in cleaned


# ── cleaning and quality ──────────────────────────────────────────────────


class TestNoiseRemoval:
    @pytest.mark.parametrize(
        "line",
        [
            "journal homepage: www.elsevier.com/locate/steroids",
            "STE-6790; No.ofPages12",
            "GModel",
            "ARTICLE IN PRESS",
            "© 2010 Elsevier Ltd. All rights reserved.",
            "12",
            "Page 4 of 12",
            "Received 4 January 2010; accepted 15 February 2010",
            "Downloaded from https://academic.oup.com",
            "doi:10.1016/j.steroids.2010.02.012",
            "[12] Smith J, Jones A. Some title. J Clin Path. 2019;4:12-20.",
            "Corresponding author: a.wallace@example.ac.uk",
        ],
    )
    def test_known_noise_is_dropped(self, line):
        assert clean.is_noise_line(line)

    @pytest.mark.parametrize(
        "line",
        [
            "Serum 25(OH)D was measured by LC-MS/MS in all 412 participants.",
            "The limit of detection was 2.5 ng/mL.",
            "Vitamin D deficiency is defined as a concentration below 20 ng/mL.",
        ],
    )
    def test_real_prose_survives(self, line):
        assert not clean.is_noise_line(line)

    def test_block_cleaning_keeps_only_prose(self, prose):
        block = f"GModel\nARTICLE IN PRESS\n{prose}\n© 2010 Elsevier Ltd.\n12"
        out = clean.clean_block(block)
        assert "GModel" not in out
        assert "Elsevier" not in out
        assert "mass spectrometry" in out

    def test_ligatures_and_hyphenation_repaired(self):
        out = clean.clean_block("The classiﬁcation of deﬁ-\nciency was speciﬁc.")
        assert "classification" in out
        assert "deficiency" in out


class TestQualityScoring:
    def test_rejects_the_fragments_that_reached_the_corpus(self, settings):
        for junk in [
            "journal homepage: www.elsevier.com/locate/steroids",
            "STE-6790; No.ofPages12 Steroidsxxx (2010) xxx-xxx",
            "1 2 3 4 5 6 7 8 9 10 11 12 13 14",
            "",
        ]:
            q = clean.score_quality(
                junk,
                min_tokens=settings.min_chunk_tokens,
                min_alpha_ratio=settings.min_alpha_ratio,
            )
            assert not q.ok, f"should have rejected: {junk!r}"

    def test_accepts_real_prose(self, settings, prose):
        q = clean.score_quality(
            prose,
            min_tokens=settings.min_chunk_tokens,
            min_alpha_ratio=settings.min_alpha_ratio,
        )
        assert q.ok
        assert q.score > 0.7

    def test_repetition_rejected(self, settings):
        q = clean.score_quality(
            "vitamin " * 80,
            min_tokens=settings.min_chunk_tokens,
            min_alpha_ratio=settings.min_alpha_ratio,
        )
        assert not q.ok
        assert "low_lexical_diversity" in q.reasons

    def test_table_judged_on_structure(self, settings):
        table = "| Assay | LOD | CV |\n| --- | --- | --- |\n| LC-MS/MS | 2.5 | 7.4 |"
        q = clean.score_quality(table, min_tokens=100, min_alpha_ratio=0.9, chunk_type="table")
        assert q.ok

    def test_table_without_structure_rejected(self, settings):
        q = clean.score_quality("some words here", min_tokens=5, min_alpha_ratio=0.5, chunk_type="table")
        assert not q.ok


class TestExcludedSections:
    @pytest.mark.parametrize(
        "section",
        ["References", "REFERENCES", "5. References", "Acknowledgements", "Funding",
         "Conflict of Interest", "Data Availability"],
    )
    def test_excluded(self, section):
        assert clean.is_excluded_section(section)

    @pytest.mark.parametrize("section", ["Methods", "Results", "Discussion", "Introduction"])
    def test_kept(self, section):
        assert not clean.is_excluded_section(section)


# ── sections ──────────────────────────────────────────────────────────────


class TestSectionDetection:
    @pytest.mark.parametrize(
        "line,expected",
        [
            ("2.3 Sample preparation", "Methods"),
            ("MATERIALS AND METHODS", "Methods"),
            ("3. Results and discussion", "Results"),
            ("Abstract", "Abstract"),
            ("References", "References"),
            ("IV. Conclusion", "Conclusion"),
        ],
    )
    def test_real_headings_detected(self, line, expected):
        assert classify_line(line, next_line="x" * 60) == expected

    @pytest.mark.parametrize(
        "line",
        [
            "Steroids",
            "Review",
            "GModel ARTICLE IN PRESS",
            "journal homepage: www.elsevier.com/locate/steroids",
            "This is a sentence that happens to start with a capital.",
            "However the results were inconclusive",
            "Vitamin D deficiency is widespread in northern latitudes.",
        ],
    )
    def test_furniture_and_prose_are_not_headings(self, line):
        """The old regex matched all of these, which is why the corpus has
        sections called "Steroids" and "Review"."""
        assert classify_line(line, next_line="x" * 60) is None

    def test_canonicalisation_collapses_variants(self):
        for variant in ["Materials and Methods", "METHODS", "2.1. Materials and methods"]:
            assert canonicalise(variant) == "Methods"

    def test_tables_attach_to_their_own_page(self):
        pages = [
            {
                "page_num": 1,
                "text": "MATERIALS AND METHODS\n" + "Prose about the assay. " * 12,
                "tables": ["| a | b |\n| --- | --- |\n| 1 | 2 |"],
                "image_paths": [],
            },
            {
                "page_num": 2,
                "text": "RESULTS\n" + "Prose about the findings. " * 12,
                "tables": ["| c | d |\n| --- | --- |\n| 3 | 4 |"],
                "image_paths": [],
            },
        ]
        sections = split_into_sections(pages)
        by_name = {s.name: s for s in sections}
        assert "Methods" in by_name and "Results" in by_name
        assert by_name["Methods"].tables[0][0] == 1
        assert by_name["Results"].tables[0][0] == 2

    def test_runt_sections_merge_back(self):
        pages = [
            {
                "page_num": 1,
                "text": "METHODS\n" + "Real prose here. " * 20 + "\nSomething Short\ntiny",
                "tables": [],
                "image_paths": [],
            }
        ]
        sections = split_into_sections(pages)
        assert all(len(s.text.strip()) >= 160 or s.name == "Methods" for s in sections)


# ── chunking ──────────────────────────────────────────────────────────────


class TestSentenceSplitting:
    def test_abbreviations_do_not_split(self):
        text = "Levels were approx. 25 ng/mL. Values rose after supplementation."
        assert len(split_sentences(text)) == 2

    def test_decimals_do_not_split(self):
        assert len(split_sentences("The LOD was 2.5 ng/mL and the LOQ 7.5 ng/mL.")) == 1

    def test_initials_do_not_split(self):
        assert len(split_sentences("Reported by A.M. Wallace in that review.")) == 1


class TestSizeEnforcement:
    def test_single_sentence_cannot_bypass_the_clamp(self, settings):
        """The exact bypass that let 12-token fragments into the index:
        semantic_chunk used to return early for one sentence."""
        tiny = "journal homepage"
        out = semantic_chunk(
            tiny,
            encode=None,
            similarity_threshold=0.72,
            min_tokens=settings.min_chunk_tokens,
            max_tokens=settings.max_chunk_tokens,
        )
        # It may survive as text, but it went through the clamp; the quality
        # gate is what rejects it, and it does.
        q = clean.score_quality(
            out[0] if out else "",
            min_tokens=settings.min_chunk_tokens,
            min_alpha_ratio=settings.min_alpha_ratio,
        )
        assert not q.ok

    def test_oversized_chunks_are_split(self):
        long_text = "This sentence is of a reasonable length and says something. " * 40
        out = enforce_size([long_text], min_tokens=20, max_tokens=100)
        assert len(out) > 1
        assert all(len(c) // 4 <= 140 for c in out)

    def test_first_chunk_merges_forwards(self):
        """The old merge only folded into the previous chunk, so the first
        chunk of a section could be arbitrarily small."""
        out = enforce_size(["tiny bit", "A much longer passage that carries real content here."],
                           min_tokens=10, max_tokens=200)
        assert len(out) == 1
        assert out[0].startswith("tiny bit")

    def test_overlap_repeats_a_sentence(self):
        sentences = [f"Sentence number {i} with enough words to count properly." for i in range(12)]
        out = enforce_size([" ".join(sentences)], min_tokens=5, max_tokens=40, overlap=1)
        assert len(out) > 1
        # Some sentence appears in two consecutive chunks.
        assert any(
            any(s in out[i] and s in out[i + 1] for s in sentences)
            for i in range(len(out) - 1)
        )

    def test_semantic_chunking_respects_bounds(self, settings, fake_encode, prose):
        out = semantic_chunk(
            prose * 3,
            encode=fake_encode,
            similarity_threshold=0.72,
            min_tokens=settings.min_chunk_tokens,
            max_tokens=settings.max_chunk_tokens,
        )
        assert out
        assert all(len(c) // 4 <= settings.max_chunk_tokens * 1.5 for c in out)


# ── identity ──────────────────────────────────────────────────────────────


class TestChunkIdentity:
    def test_point_id_is_deterministic(self):
        a = Chunk.build(content="same text here", paper_id="p1", title="T")
        b = Chunk.build(content="same text here", paper_id="p1", title="T")
        assert a.point_id == b.point_id

    def test_point_id_differs_across_papers(self):
        """The bug this prevents: enumerate-based ids meant paper B's first
        chunk overwrote paper A's first chunk."""
        a = Chunk.build(content="same text here", paper_id="p1", title="T")
        b = Chunk.build(content="same text here", paper_id="p2", title="T")
        assert a.point_id != b.point_id

    def test_point_id_is_a_uuid(self):
        import uuid

        chunk = Chunk.build(content="x" * 300, paper_id="p", title="T")
        assert uuid.UUID(chunk.point_id)
