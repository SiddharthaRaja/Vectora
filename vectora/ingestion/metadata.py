"""
Bibliographic metadata extraction.

Every citation Vectora shows is built from these four fields, so getting
them wrong is not cosmetic — it is the difference between a usable reference
and ``"GModel ARTICLE IN PRESS", authors: "", year: "20"``, which is what the
previous implementation produced for a large share of the corpus.

Three things were wrong before, and all three are fixed here:

1. ``YEAR_PATTERN.findall`` used a *capturing* group ``(19|20)``, so it
   returned the group — ``"20"`` — rather than the matched year. The groups
   are now non-capturing and the result is validated as four digits.
2. The title was "first non-empty line of page 1", which on a typical
   publisher PDF is the journal furniture, not the title. Titles are now
   scored against several signals and the best candidate wins.
3. Authors were "first line containing a capital letter after a comma",
   which matched affiliations, running heads and copyright lines.

The embedded PDF metadata dictionary is consulted first, since when a
publisher fills it in correctly it beats any heuristic. The filename is the
last resort, which is right for this corpus — the files are named after the
paper.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from vectora.ingestion.clean import is_noise_line, normalise_chars

#: DOIs: the registrant prefix is always ``10.NNNN``. The suffix runs until
#: whitespace; trailing sentence punctuation is trimmed after matching.
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9<>\[\]]+", re.IGNORECASE)
#: Non-capturing, so ``findall`` returns whole years.
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_ARXIV_RE = re.compile(r"\barxiv:\s*(\d{4})\.\d{4,5}", re.IGNORECASE)

_PUBLISHER_NOISE = re.compile(
    r"\b(elsevier|springer|wiley|mdpi|frontiers|taylor\s*&?\s*francis|sage|"
    r"oxford\s+university|cambridge\s+university|ieee|acs\s+publications|"
    r"royal\s+society\s+of\s+chemistry|hindawi|plos|bmc|nature\s+publishing|"
    r"sciencedirect|researchgate|pubmed|crossmark|journal\s+homepage|"
    r"contents\s+lists|article\s+in\s+press|g\s*model|all\s+rights\s+reserved|"
    r"open\s+access|creative\s+commons|received|accepted|revised|published|"
    r"doi|issn|isbn|volume|issue|pp\.|citation)\b",
    re.IGNORECASE,
)

_SECTION_WORD = re.compile(
    r"^\s*(abstract|introduction|background|keywords?|highlights|graphical\s+abstract|"
    r"summary|contents|review|article|research\s+article|original\s+article|"
    r"methods?|materials)\b",
    re.IGNORECASE,
)

_AFFILIATION_HINT = re.compile(
    r"\b(university|universit|institut|department|dept\.|college|hospital|clinic|"
    r"laborator|laboratoire|academy|school\s+of|centre|center|faculty|division|"
    r"p\.?o\.?\s*box|ltd\.?|inc\.?|gmbh|s\.?a\.?r\.?l)\b",
    re.IGNORECASE,
)

#: "A.M. Wallace", "S. Gibson", "Wallace AM", "Lamberg-Allardt C"
_AUTHOR_TOKEN = re.compile(
    r"(?:[A-Z]\.\s?){1,3}\s?[A-Z][a-zA-Z'’\-]+"          # A.M. Wallace
    r"|[A-Z][a-zA-Z'’\-]{1,}\s+(?:[A-Z]\.?\s?){1,3}"      # Wallace AM
    r"|[A-Z][a-zA-Z'’\-]{2,}\s+[A-Z][a-zA-Z'’\-]{2,}"      # Firstname Lastname
)
_SUPERSCRIPT = re.compile(r"[\d\*†‡§¶]+(?=\s|,|;|$)")
_FILENAME_JUNK = re.compile(
    r"[_\-]+|\(\d+\)|\bv\d+\b|\bfinal\b|\bcopy\b|\bdraft\b|\bpdf\b", re.IGNORECASE
)


@dataclass
class PaperMetadata:
    """What we know about a paper, plus how confident we are."""

    title: str = ""
    authors: str = ""
    year: str = ""
    doi: str = ""
    confidence: float = 0.0
    sources: dict[str, str] = field(default_factory=dict)
    """Which strategy produced each field — useful when auditing a bad
    citation, and surfaced by ``vectora ingest --report``."""

    def merged_with(self, other: PaperMetadata) -> PaperMetadata:
        """Fill blanks from ``other``; never overwrite what we already have."""
        out = PaperMetadata(
            title=self.title or other.title,
            authors=self.authors or other.authors,
            year=self.year or other.year,
            doi=self.doi or other.doi,
            sources={**other.sources, **self.sources},
        )
        filled = sum(bool(x) for x in (out.title, out.authors, out.year, out.doi))
        out.confidence = round(filled / 4, 2)
        return out


# ── field extractors ──────────────────────────────────────────────────────


def extract_doi(text: str) -> tuple[str, str]:
    m = DOI_RE.search(text)
    if not m:
        return "", ""
    # Strip punctuation a sentence may have appended, but keep balanced parens
    # that are genuinely part of some DOIs.
    doi = m.group(0).rstrip(".,;:")
    while doi.endswith(")") and doi.count("(") < doi.count(")"):
        doi = doi[:-1]
    return doi, "text"


def extract_year(text: str, doi: str = "", filename: str = "") -> tuple[str, str]:
    """Pick the publication year.

    Strategy: prefer a year that appears next to a copyright or publication
    cue, then the most common plausible year in the header, then the
    filename. Years outside 1950..next-year are ignored.
    """
    import datetime

    ceiling = datetime.date.today().year + 1

    def plausible(y: str) -> bool:
        return 1950 <= int(y) <= ceiling

    arxiv = _ARXIV_RE.search(text)
    if arxiv and plausible(arxiv.group(1)):
        return arxiv.group(1), "arxiv_id"

    for cue in (
        r"©\s*(?:\d{0,4}\s*)?((?:19|20)\d{2})",
        r"\(\s*((?:19|20)\d{2})\s*\)",
        r"\b(?:published|accepted|available\s+online)[^\n]{0,40}?((?:19|20)\d{2})",
        r"\b((?:19|20)\d{2})\s*;\s*\d+",            # "2010;75:477"
    ):
        m = re.search(cue, text, re.IGNORECASE)
        if m and plausible(m.group(1)):
            return m.group(1), "cue"

    years = [y for y in YEAR_RE.findall(text) if plausible(y)]
    if years:
        # Mode, tie-broken by latest: the publication year recurs in the
        # header (copyright, citation line, received date).
        best = max(set(years), key=lambda y: (years.count(y), int(y)))
        return best, "frequency"

    fname_years = [y for y in YEAR_RE.findall(filename) if plausible(y)]
    if fname_years:
        return fname_years[0], "filename"

    return "", ""


def _title_score(line: str, position: int) -> float:
    """How much a line looks like a paper title."""
    stripped = line.strip()
    n = len(stripped)
    if n < 18 or n > 260:
        return -10.0
    if _PUBLISHER_NOISE.search(stripped):
        return -10.0
    if _SECTION_WORD.match(stripped):
        return -10.0
    if _AFFILIATION_HINT.search(stripped):
        return -6.0
    if DOI_RE.search(stripped) or "@" in stripped:
        return -10.0

    words = stripped.split()
    if len(words) < 4:
        return -5.0

    score = 2.0
    score -= position * 0.22          # titles sit near the top
    score += min(len(words), 24) * 0.09

    letters = sum(c.isalpha() for c in stripped)
    if letters / n < 0.70:
        score -= 2.5
    if stripped.isupper():
        score -= 0.6                   # all-caps titles exist but are rarer
    if stripped.endswith((".", ";", ",")):
        score -= 0.8                   # titles usually lack terminal punctuation
    if re.search(r"\d{2,}", stripped):
        score -= 0.5
    # Title case or sentence case, both common; random capitals are not.
    capitalised = sum(1 for w in words if w[:1].isupper())
    ratio = capitalised / len(words)
    if 0.25 <= ratio <= 1.0:
        score += 0.6
    # Scientific titles are long and contain a colon or a hyphen often.
    if ":" in stripped and n > 40:
        score += 0.4
    return score


def extract_title(header_lines: list[str], filename_stem: str) -> tuple[str, str]:
    """Best title candidate from the first page, else the filename."""
    candidates: list[tuple[float, str]] = []
    for i, line in enumerate(header_lines[:28]):
        s = _title_score(line, i)
        if s > 0:
            candidates.append((s, line.strip()))

    if candidates:
        candidates.sort(key=lambda t: t[0], reverse=True)
        best_score, best = candidates[0]
        # A title wrapped onto a second line: stitch when the next line also
        # scores and the first does not end a sentence.
        idx = next(
            (i for i, ln in enumerate(header_lines[:28]) if ln.strip() == best), None
        )
        if idx is not None and idx + 1 < len(header_lines):
            nxt = header_lines[idx + 1].strip()
            if (
                (not best.endswith((".", "?", "!"))
                and 8 <= len(nxt) <= 140
                and _title_score(nxt, idx + 1) > -5
                and not _AUTHOR_TOKEN.fullmatch(nxt)
                and nxt[:1].islower()) or (len(best) < 70 and _title_score(nxt, idx + 1) > 0)
            ):
                best = f"{best} {nxt}".strip()
        if best_score >= 1.0:
            return re.sub(r"\s{2,}", " ", best), "header"

    return title_from_filename(filename_stem), "filename"


def title_from_filename(stem: str) -> str:
    """Turn ``"2010 Vit D method review Wallace et al"`` into a title.

    Keeps the words; only normalises separators and strips version junk.
    Filenames in this corpus are descriptive, so this is a decent fallback.
    """
    cleaned = _FILENAME_JUNK.sub(" ", stem)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" .-_")
    return cleaned or stem


def extract_authors(header_lines: list[str], title: str) -> tuple[str, str]:
    """Author block from the lines just below the title.

    Scored rather than first-match, because affiliations and running heads
    sit in the same region and look superficially similar.
    """
    start = 0
    for i, line in enumerate(header_lines[:28]):
        if line.strip() and line.strip()[:40] == title[:40]:
            start = i + 1
            break

    best: tuple[float, str] | None = None
    for offset, line in enumerate(header_lines[start : start + 8]):
        stripped = _SUPERSCRIPT.sub("", line.strip()).strip(" ,;·•*")
        if not stripped or len(stripped) > 320 or len(stripped) < 5:
            continue
        if _PUBLISHER_NOISE.search(stripped) or _SECTION_WORD.match(stripped):
            continue
        if "@" in stripped or DOI_RE.search(stripped):
            continue

        names = _AUTHOR_TOKEN.findall(stripped)
        if not names:
            continue

        score = len(names) * 1.0 - offset * 0.3
        if _AFFILIATION_HINT.search(stripped):
            score -= 3.0               # affiliation line, not authors
        if "," in stripped or " and " in stripped.lower():
            score += 0.8
        digits = sum(c.isdigit() for c in stripped)
        if digits > len(stripped) * 0.12:
            score -= 1.0
        # Author lines are mostly names; a long line with two names is prose.
        covered = sum(len(n) for n in names) / len(stripped)
        score += covered * 1.5

        if score > 0 and (best is None or score > best[0]):
            best = (score, _tidy_authors(stripped))

    if best:
        return best[1], "header"

    # Last resort: an "X et al" suffix in the filename still gives us one name.
    return "", ""


def _tidy_authors(raw: str) -> str:
    """Normalise separators and drop affiliation tails."""
    text = normalise_chars(raw)
    text = re.sub(r"\s+and\s+", ", ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*[;·•]\s*", ", ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = _SUPERSCRIPT.sub("", text)
    # Cut at the first affiliation cue, which often follows the author list.
    m = _AFFILIATION_HINT.search(text)
    if m and m.start() > 12:
        text = text[: m.start()]
    parts = [p.strip(" .,") for p in text.split(",")]
    parts = [p for p in parts if p and len(p) > 1 and any(c.isalpha() for c in p)]
    return ", ".join(parts[:14]).strip(" ,")


# ── orchestration ─────────────────────────────────────────────────────────


def from_pdf_dict(info: dict[str, object] | None) -> PaperMetadata:
    """Read the PDF's own metadata dictionary.

    Trusted when present and not obviously a tool artifact — publishers
    often write the real title here, and LaTeX writes the filename or
    ``"untitled"``.
    """
    meta = PaperMetadata()
    if not info:
        return meta

    raw_title = str(info.get("title") or info.get("Title") or "").strip()
    if (
        18 <= len(raw_title) <= 260
        and not _PUBLISHER_NOISE.search(raw_title)
        and not raw_title.lower().startswith(("untitled", "microsoft word", "doi:", "layout"))
        and len(raw_title.split()) >= 4
    ):
        meta.title = re.sub(r"\s{2,}", " ", normalise_chars(raw_title))
        meta.sources["title"] = "pdf_info"

    raw_authors = str(info.get("author") or info.get("Author") or "").strip()
    if raw_authors and 4 < len(raw_authors) < 320 and not _AFFILIATION_HINT.search(raw_authors):
        meta.authors = _tidy_authors(raw_authors)
        meta.sources["authors"] = "pdf_info"

    for key in ("creationDate", "CreationDate", "ModDate"):
        value = str(info.get(key) or "")
        m = YEAR_RE.search(value)
        if m:
            meta.year = m.group(0)
            meta.sources["year"] = "pdf_info"
            break

    return meta


def extract(
    header_text: str,
    filename_stem: str,
    pdf_info: dict[str, object] | None = None,
) -> PaperMetadata:
    """Extract metadata from the front matter of a paper.

    ``header_text`` should be the raw (uncleaned) text of roughly the first
    two pages: cleaning removes exactly the publisher lines that carry the
    year and the DOI.
    """
    text = normalise_chars(header_text)
    lines = [ln for ln in (raw.rstrip() for raw in text.split("\n")) if ln.strip()]
    # Keep noise lines out of title/author scoring but keep them in `text`
    # so the year and DOI cues remain visible.
    signal_lines = [ln for ln in lines if not is_noise_line(ln)]

    doi, doi_src = extract_doi(text)
    year, year_src = extract_year(text, doi, filename_stem)
    title, title_src = extract_title(signal_lines, filename_stem)
    authors, authors_src = extract_authors(signal_lines, title)

    heuristic = PaperMetadata(
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        sources={
            k: v
            for k, v in (
                ("title", title_src),
                ("authors", authors_src),
                ("year", year_src),
                ("doi", doi_src),
            )
            if v
        },
    )
    # PDF info wins on title and authors when it looks real; heuristics fill
    # the rest. DOI and year from the text are more reliable than the info
    # dictionary's dates, so heuristic goes second in the merge for those.
    embedded = from_pdf_dict(pdf_info)
    merged = embedded.merged_with(heuristic)
    if not merged.title:
        merged.title = title_from_filename(filename_stem)
        merged.sources["title"] = "filename"
    return merged
