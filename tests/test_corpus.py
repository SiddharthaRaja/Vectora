"""
Tests for corpora, document loading and corpus acquisition.

Everything here runs without network, models or a vector store.
"""

from __future__ import annotations

import json

import pytest

from vectora.corpus import (
    CorpusContext,
    CorpusProfile,
    CorpusRegistry,
    migrate_legacy_layout,
    slugify,
)
from vectora.fetch import (
    PRESETS,
    FetchedDocument,
    parse_arxiv_feed,
    read_sidecar,
    safe_filename,
)
from vectora.ingestion.sources import (
    SUPPORTED_SUFFIXES,
    is_supported,
    iter_documents,
    parse_document,
)
from vectora.retrieval.expansion import CorpusHint

ARXIV_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>2841</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2004.04906v3</id>
    <published>2020-04-10T07:04:11Z</published>
    <updated>2020-09-30T17:21:11Z</updated>
    <title>Dense Passage Retrieval for Open-Domain
  Question Answering</title>
    <summary>  Open-domain question answering relies on passage retrieval.
</summary>
    <author><name>Vladimir Karpukhin</name></author>
    <author><name>Barlas Oguz</name></author>
    <arxiv:doi>10.18653/v1/2020.emnlp-main.550</arxiv:doi>
    <link title="pdf" href="http://arxiv.org/pdf/2004.04906v3" rel="related"
          type="application/pdf"/>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2007.01282v2</id>
    <published>2020-07-02T18:30:00Z</published>
    <title>Leveraging Passage Retrieval with Generative Models</title>
    <summary>  Generative reader models.</summary>
    <author><name>Gautier Izacard</name></author>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>
"""


# ── profiles ──────────────────────────────────────────────────────────────


class TestCorpusProfile:
    def test_collection_is_derived_not_configured(self, tmp_path):
        """Two corpora sharing one collection is a bug worth making
        unrepresentable, so the name is derived from the corpus."""
        registry = CorpusRegistry(tmp_path)
        a = registry.create("alpha")
        b = registry.create("beta-two")
        assert a.collection == "vectora_alpha"
        assert b.collection == "vectora_beta_two"
        assert a.collection != b.collection

    def test_each_corpus_owns_its_paths(self, tmp_path):
        registry = CorpusRegistry(tmp_path)
        a = registry.create("alpha")
        b = registry.create("beta")
        assert a.chunks_path != b.chunks_path
        assert a.bm25_path != b.bm25_path
        assert a.golden_path != b.golden_path
        assert a.papers_dir != b.papers_dir

    def test_round_trips_through_disk(self, tmp_path):
        registry = CorpusRegistry(tmp_path)
        created = registry.create(
            "docs",
            description="Some documents",
            document_kind="RFCs",
            domain_terms=["MUST", "SHOULD", "transport"],
            source="rfc-editor",
        )
        loaded = registry.get("docs")
        assert loaded.description == "Some documents"
        assert loaded.document_kind == "RFCs"
        assert loaded.domain_terms == ["MUST", "SHOULD", "transport"]
        assert loaded.source == "rfc-editor"
        assert loaded.collection == created.collection

    @pytest.mark.parametrize("bad", ["", "a", "A-Caps", "has space", "-leading", "x" * 60])
    def test_invalid_names_rejected(self, bad):
        with pytest.raises(ValueError):
            CorpusProfile(name=bad)

    def test_missing_corpus_lists_the_alternatives(self, tmp_path):
        registry = CorpusRegistry(tmp_path)
        registry.create("alpha")
        registry.create("beta")
        with pytest.raises(KeyError) as exc:
            registry.get("gamma")
        message = str(exc.value)
        assert "alpha" in message and "beta" in message

    def test_duplicate_create_is_an_error_unless_allowed(self, tmp_path):
        registry = CorpusRegistry(tmp_path)
        registry.create("alpha")
        with pytest.raises(FileExistsError):
            registry.create("alpha")
        assert registry.create("alpha", exist_ok=True).name == "alpha"

    def test_delete_keeps_documents_by_default(self, tmp_path):
        registry = CorpusRegistry(tmp_path)
        profile = registry.create("alpha")
        (profile.papers_dir / "a.txt").write_text("content")
        profile.chunks_path.write_text('{"x": 1}\n')

        registry.delete("alpha")
        assert not profile.chunks_path.exists()
        assert (profile.papers_dir / "a.txt").exists(), "documents must survive"

        registry.delete("alpha", keep_documents=False)
        assert not profile.papers_dir.exists()

    def test_counts_are_recounted_from_disk(self, tmp_path):
        registry = CorpusRegistry(tmp_path)
        profile = registry.create("alpha")
        (profile.papers_dir / "a.txt").write_text("x")
        (profile.papers_dir / "b.md").write_text("y")
        profile.chunks_path.write_text('{"a":1}\n{"b":2}\n\n')
        registry.refresh_counts(profile)
        assert profile.documents == 2
        assert profile.chunks == 2

    @pytest.mark.parametrize(
        "raw,expected",
        [("ML Papers", "ml-papers"), ("cs.IR!!", "cs-ir"), ("", "corpus")],
    )
    def test_slugify(self, raw, expected):
        assert slugify(raw) == expected


class TestLegacyMigration:
    def test_adopts_a_pre_v3_chunk_file(self, tmp_path):
        legacy = tmp_path / "chunks.jsonl"
        legacy.write_text('{"chunk_id":"a","paper_id":"p","title":"T","content":"x"}\n')
        profile = migrate_legacy_layout(
            tmp_path / "corpora", chunks_path=legacy, papers_dir=tmp_path / "papers"
        )
        assert profile is not None
        assert profile.name == "default"
        assert profile.chunks_path.exists()
        assert profile.chunks == 1

    def test_returns_none_with_nothing_to_adopt(self, tmp_path):
        assert (
            migrate_legacy_layout(
                tmp_path / "corpora",
                chunks_path=tmp_path / "absent.jsonl",
                papers_dir=tmp_path / "papers",
            )
            is None
        )


class TestCorpusContext:
    def test_resolves_from_settings(self, settings):
        ctx = CorpusContext.build(settings)
        assert ctx.name == "default"
        assert ctx.collection == "vectora_default"
        assert ctx.chunks.name == "chunks.jsonl"

    def test_settings_override_wins_over_the_corpus(self, settings, tmp_path):
        """The override exists so a one-off run can point at a stray file."""
        elsewhere = tmp_path / "elsewhere.jsonl"
        tuned = settings.model_copy(update={"chunks_path": elsewhere})
        ctx = CorpusContext.build(tuned)
        assert ctx.chunks == elsewhere


# ── corpus hint reaching the prompts ──────────────────────────────────────


class TestCorpusHint:
    def test_prompt_names_the_corpus_not_a_fixed_field(self, tmp_path):
        """The old prompt hardcoded 'biomedical and analytical chemistry',
        which made every other corpus a worse fit than it needed to be."""
        registry = CorpusRegistry(tmp_path)
        profile = registry.create(
            "rfcs", document_kind="RFCs", domain_terms=["MUST", "transport"]
        )
        hint = CorpusHint.from_profile(profile)
        prompt = hint.rewrite_system()
        assert "RFCs" in prompt
        assert "MUST" in prompt and "transport" in prompt
        assert "biomedical" not in prompt.lower()
        assert "chemistry" not in prompt.lower()

    def test_defaults_are_neutral(self):
        hint = CorpusHint()
        assert "documents" in hint.rewrite_system()
        assert "{" not in hint.rewrite_system(), "template fully substituted"
        assert "{" not in hint.hyde_system()

    def test_no_terms_means_no_terms_clause(self):
        hint = CorpusHint(document_kind="papers")
        assert "Terminology that appears" not in hint.rewrite_system()

    def test_tolerates_a_profile_without_the_fields(self):
        assert CorpusHint.from_profile(object()).document_kind == "documents"
        assert CorpusHint.from_profile(None).document_kind == "documents"


# ── document loading ──────────────────────────────────────────────────────


class TestDocumentLoading:
    def test_supported_suffixes(self):
        assert ".pdf" in SUPPORTED_SUFFIXES
        assert ".md" in SUPPORTED_SUFFIXES
        assert ".html" in SUPPORTED_SUFFIXES
        assert ".txt" in SUPPORTED_SUFFIXES
        assert not is_supported(__import__("pathlib").Path("x.bin"))

    def test_discovery_skips_hidden_and_unsupported(self, tmp_path):
        (tmp_path / "a.md").write_text("content")
        (tmp_path / "b.bin").write_bytes(b"\x00")
        (tmp_path / ".hidden.md").write_text("skip me")
        nested = tmp_path / ".git"
        nested.mkdir()
        (nested / "c.md").write_text("skip me too")
        found = {p.name for p in iter_documents(tmp_path)}
        assert found == {"a.md"}

    def test_markdown_structure_becomes_plain_text(self, tmp_path):
        path = tmp_path / "doc.md"
        path.write_text(
            "---\ntitle: Hybrid Retrieval\nauthor: A. Author\n---\n\n"
            "# Hybrid Retrieval\n\n## Methods\n\n"
            "We use [rank fusion](https://example.com/x) over `BM25` and "
            "**dense** retrieval.\n\n"
            "```python\nscore = 1 / (60 + rank)\n```\n"
        )
        parsed = parse_document(path)
        text = "\n".join(p["text"] for p in parsed.pages)
        assert parsed.info["title"] == "Hybrid Retrieval"
        assert parsed.info["author"] == "A. Author"
        assert "# Hybrid" not in text, "heading markup stripped"
        assert "Methods" in text, "heading text kept so sections are detectable"
        assert "rank fusion" in text and "https://example.com" not in text
        assert "BM25" in text, "inline code content kept"
        assert "score = 1 / (60 + rank)" in text, "code fence kept verbatim"

    def test_html_metadata_survives_and_scripts_do_not(self, tmp_path):
        path = tmp_path / "page.html"
        path.write_text(
            "<html><head><title>Reranking</title>"
            '<meta name="citation_author" content="R. Nogueira">'
            '<meta content="S. Second" name="citation_author">'
            '<meta name="citation_date" content="2019-01-15">'
            "<script>var tracking = 1;</script><style>b{color:red}</style>"
            "</head><body><h1>Reranking</h1>"
            "<p>A cross-encoder scores query &amp; passage jointly.</p>"
            "<table><tr><th>M</th></tr><tr><td>0.37</td></tr></table>"
            "</body></html>"
        )
        parsed = parse_document(path)
        text = "\n".join(p["text"] for p in parsed.pages)
        # The head holds the only reliable metadata an HTML document has, so
        # stripping it (as an earlier version did) threw away the point.
        assert parsed.info["title"] == "Reranking"
        assert "R. Nogueira" in parsed.info["author"]
        assert "S. Second" in parsed.info["author"], "content-before-name order"
        assert "tracking" not in text
        assert "color:red" not in text
        assert "&amp;" not in text and "& passage" in text

    def test_text_gets_synthetic_pages(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("A line of notes.\n" * 100)
        parsed = parse_document(path)
        assert parsed.page_count > 1, "a long text file is paginated"
        assert all(p["page_num"] >= 1 for p in parsed.pages)

    def test_unsupported_type_raises(self, tmp_path):
        path = tmp_path / "x.bin"
        path.write_bytes(b"\x00")
        with pytest.raises(ValueError):
            parse_document(path)

    def test_unreadable_file_reports_rather_than_raises(self, tmp_path):
        parsed = parse_document(tmp_path / "missing.txt")
        assert parsed.errors, "a read failure is collected, not raised"


# ── arXiv fetching ────────────────────────────────────────────────────────


class TestArxivParsing:
    def test_parses_entries_and_total(self):
        docs, total = parse_arxiv_feed(ARXIV_FEED)
        assert total == 2841
        assert len(docs) == 2

    def test_multiline_title_is_collapsed(self):
        docs, _ = parse_arxiv_feed(ARXIV_FEED)
        assert "\n" not in docs[0].title
        assert docs[0].title == "Dense Passage Retrieval for Open-Domain Question Answering"

    def test_metadata_is_exact_not_inferred(self):
        docs, _ = parse_arxiv_feed(ARXIV_FEED)
        first = docs[0]
        assert first.doc_id == "2004.04906v3"
        assert first.year == "2020"
        assert first.doi == "10.18653/v1/2020.emnlp-main.550"
        assert first.author_string == "Vladimir Karpukhin, Barlas Oguz"

    def test_pdf_url_falls_back_when_absent(self):
        docs, _ = parse_arxiv_feed(ARXIV_FEED)
        assert docs[1].pdf_url.endswith("2007.01282v2")

    def test_filename_is_safe_and_identifiable(self):
        name = safe_filename("2004.04906v3", "Dense Passage Retrieval: a Study/Review")
        assert name.startswith("2004.04906v3__")
        assert name.endswith(".pdf")
        for bad in '/\\:*?"<>|':
            assert bad not in name

    def test_presets_are_well_formed(self):
        assert PRESETS
        for name, spec in PRESETS.items():
            assert {"query", "description", "document_kind", "terms"} <= set(spec)
            assert "cat:" in spec["query"], f"{name} should query an arXiv category"


class TestSidecar:
    def test_round_trip(self, tmp_path):
        doc = FetchedDocument(
            doc_id="2004.04906v3",
            title="Dense Passage Retrieval",
            authors=["V. Karpukhin", "B. Oguz"],
            published="2020-04-10T07:04:11Z",
            doi="10.18653/v1/x",
        )
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        (tmp_path / "paper.pdf.meta.json").write_text(json.dumps(doc.to_sidecar()))

        loaded = read_sidecar(pdf)
        assert loaded is not None
        assert loaded["title"] == "Dense Passage Retrieval"
        assert loaded["year"] == "2020"

    def test_absent_sidecar_is_none(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF")
        assert read_sidecar(pdf) is None

    def test_corrupt_sidecar_is_none_not_an_exception(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF")
        (tmp_path / "paper.pdf.meta.json").write_text("{not json")
        assert read_sidecar(pdf) is None
