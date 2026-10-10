"""
Section detection.

The old detector accepted any line matching ``^[A-Z][a-zA-Z\\s\\-:]{3,60}$``,
which is true of a very large fraction of ordinary lines in a paper. That is
why the corpus has sections called ``"Steroids"``, ``"Review"`` and
``"GModel ARTICLE IN PRESS"`` — the detector fired on journal furniture and
on the first words of wrapped sentences, and every chunk inherited the wrong
label.

The rule here is the opposite: a line is a heading only if it gives
*positive* evidence. Either it is numbered (``3.2 Sample preparation``), or
it names one of the sections scientific papers actually have, or it is a
short standalone line in a heading-shaped style that is followed by prose.
Everything else is body text, which is the safe default — a mislabelled
section degrades citations, a missed heading only means a slightly larger
chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Canonical section names and the variants papers use for them. Matching is
#: on the normalised key, so "MATERIALS AND METHODS" and "2.1. Materials and
#: methods" both land on "Methods" and filters stay predictable.
_CANONICAL: dict[str, tuple[str, ...]] = {
    "Abstract": ("abstract", "summary", "graphical abstract", "structured abstract"),
    "Keywords": ("keywords", "key words", "index terms"),
    "Highlights": ("highlights",),
    "Introduction": ("introduction", "background", "1 introduction", "intro"),
    "Related Work": (
        "related work", "literature review", "previous work", "prior art",
        "state of the art",
    ),
    "Methods": (
        "methods", "method", "materials and methods", "methods and materials",
        "materials & methods", "experimental", "experimental section",
        "experimental methods", "experimental procedures", "methodology",
        "materials", "study design", "patients and methods",
        "subjects and methods", "sample preparation", "instrumentation",
        "data collection", "statistical analysis", "statistical methods",
        "analytical methods", "assay", "measurements", "procedure",
        "fabrication", "synthesis", "characterization", "characterisation",
    ),
    "Results": (
        "results", "findings", "results and discussion", "observations",
        "experimental results",
    ),
    "Discussion": ("discussion", "general discussion", "interpretation"),
    "Conclusion": (
        "conclusion", "conclusions", "concluding remarks", "summary and conclusions",
        "conclusions and future work", "conclusions and perspectives",
        "future work", "outlook", "perspectives",
    ),
    "Limitations": ("limitations", "study limitations", "strengths and limitations"),
    "References": ("references", "reference", "bibliography", "works cited", "literature cited"),
    "Acknowledgements": (
        "acknowledgement", "acknowledgements", "acknowledgment", "acknowledgments",
    ),
    "Funding": ("funding", "funding statement", "financial support", "grant support"),
    "Author Contributions": ("author contributions", "authors contributions", "contributions"),
    "Conflict of Interest": (
        "conflict of interest", "conflicts of interest", "declaration of interest",
        "declaration of competing interest", "competing interests", "disclosure",
        "disclosures",
    ),
    "Data Availability": (
        "data availability", "data availability statement", "availability of data",
        "code availability",
    ),
    "Ethics": (
        "ethics", "ethics statement", "ethical approval",
        "institutional review board statement", "informed consent statement",
        "consent for publication",
    ),
    "Supplementary": (
        "supplementary material", "supplementary materials", "supporting information",
        "supplementary information", "appendix", "appendices", "annex",
    ),
    "Abbreviations": ("abbreviations", "nomenclature", "list of abbreviations", "glossary"),
}

_LOOKUP: dict[str, str] = {
    variant: canon for canon, variants in _CANONICAL.items() for variant in variants
}

#: ``3.`` / ``3.2`` / ``3.2.1`` / ``IV.`` followed by a short title.
_NUMBERED = re.compile(
    r"^\s*(?P<num>\d{1,2}(?:\.\d{1,2}){0,3}\.?|[IVXLC]{1,6}\.)\s+(?P<title>[A-Z][^\n]{2,72})\s*$"
)
#: ``METHODS`` / ``MATERIALS AND METHODS`` — all caps, no terminal punctuation.
_ALLCAPS = re.compile(r"^\s*(?P<title>[A-Z][A-Z0-9 &/\-'()]{2,58})\s*$")
#: ``Sample preparation`` on its own short line.
_SHORT_LINE = re.compile(r"^\s*(?P<title>[A-Z][A-Za-z0-9 &/\-',()]{2,58})\s*$")

_TRAILING_NUM = re.compile(r"^\s*\d{1,2}(?:\.\d{1,2}){0,3}\.?\s*")
_FURNITURE = re.compile(
    r"\b(elsevier|springer|wiley|mdpi|sciencedirect|journal|homepage|volume|issue|"
    r"article in press|g model|all rights reserved|open access|creative commons|"
    r"doi|issn|isbn|pmid|received|accepted|published|downloaded|licen[cs]e|"
    r"corresponding author|e-?mail|www|http)\b",
    re.IGNORECASE,
)
#: Words that make a short capitalised line a sentence fragment, not a heading.
_VERBY = re.compile(
    r"\b(is|are|was|were|be|been|being|has|have|had|do|does|did|can|could|"
    r"may|might|should|would|will|shall|we|our|this|these|those|which|that|"
    r"however|therefore|thus|although|because|while|using|used|shows|showed|"
    r"found|reported|suggests|indicates|demonstrated)\b",
    re.IGNORECASE,
)


def canonicalise(raw: str) -> str | None:
    """Map a heading to its canonical name, or ``None`` if unrecognised."""
    key = _TRAILING_NUM.sub("", raw.strip()).strip(" .:-–—")
    key = re.sub(r"\s{2,}", " ", key).lower()
    if not key:
        return None
    if key in _LOOKUP:
        return _LOOKUP[key]
    # "3.1 Materials and methods for serum" -> Methods
    for variant, canon in _LOOKUP.items():
        if len(variant) >= 7 and key.startswith(variant):
            return canon
    return None


def classify_line(line: str, *, next_line: str | None = None) -> str | None:
    """Return the section name if ``line`` is a heading, else ``None``.

    ``next_line`` is used to require that a candidate heading is actually
    followed by content — a short capitalised line followed by another short
    capitalised line is usually an author list or an address block.
    """
    stripped = line.strip()
    if not (3 <= len(stripped) <= 80):
        return None
    if _FURNITURE.search(stripped):
        return None
    if stripped.endswith((".", ",", ";", ":")) and not _NUMBERED.match(stripped):
        # A colon-terminated line can be a heading; a full stop cannot.
        if not stripped.endswith(":"):
            return None
    if sum(c.isdigit() for c in stripped) > len(stripped) * 0.3:
        return None

    # 1. Numbered headings are the strongest signal.
    m = _NUMBERED.match(stripped)
    if m:
        title = m.group("title").strip(" .:-")
        canon = canonicalise(title)
        return canon or _titlecase(title)

    # 2. A recognised section name, in any case.
    canon = canonicalise(stripped)
    if canon:
        return canon

    # 3. All caps, short, not a sentence.
    m = _ALLCAPS.match(stripped)
    if m and len(stripped.split()) <= 7 and not _VERBY.search(stripped):
        return _titlecase(m.group("title").strip(" .:-"))

    # 4. A short title-case line, only when it is clearly not prose and the
    #    following line looks like body text.
    m = _SHORT_LINE.match(stripped)
    if m and len(stripped.split()) <= 6 and not _VERBY.search(stripped):
        words = stripped.split()
        if len(words) < 2:
            return None
        if next_line is not None:
            nxt = next_line.strip()
            if not nxt or len(nxt) < 40:
                return None  # not followed by prose; probably a list or address
        lowered = sum(1 for w in words[1:] if w[:1].islower())
        if lowered >= max(1, (len(words) - 1) // 2):
            # "Sample preparation" — leading cap then lowercase words.
            return _titlecase(stripped.strip(" .:-"))
    return None


def _titlecase(text: str) -> str:
    """Normalise a shouted heading without mangling acronyms."""
    words = text.split()
    out: list[str] = []
    small = {"and", "or", "of", "the", "in", "for", "to", "a", "an", "with", "on", "by"}
    for i, w in enumerate(words):
        if len(w) <= 4 and w.isupper() and not w.isalpha():
            out.append(w)
        elif w.isupper() and len(w) <= 5 and w.isalpha() and w.lower() not in small:
            out.append(w)  # likely an acronym: LC, MS, HPLC, ELISA
        elif i > 0 and w.lower() in small:
            out.append(w.lower())
        else:
            out.append(w.capitalize() if w.isupper() else w)
    return " ".join(out)


@dataclass
class Section:
    """A heading and everything under it, up to the next heading."""

    name: str
    page: int
    text: str = ""
    tables: list[tuple[int, str]] = field(default_factory=list)
    """``(page, markdown)`` pairs, so a table keeps the page it came from."""
    images: list[tuple[int, str]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.text.strip() and not self.tables and not self.images


def split_into_sections(pages: list[dict]) -> list[Section]:
    """Group page text under detected headings.

    Unlike the previous implementation, tables and images are attached with
    their own page number and are flushed into the section that is current
    *at the point they occur*, so a table no longer migrates to whichever
    section happened to be open when the page ended.
    """
    sections: list[Section] = []
    current = Section(name="Front Matter", page=pages[0]["page_num"] if pages else 1)

    for page in pages:
        page_num = int(page.get("page_num") or 0)
        lines = (page.get("text") or "").split("\n")

        for i, line in enumerate(lines):
            nxt = lines[i + 1] if i + 1 < len(lines) else None
            heading = classify_line(line, next_line=nxt)
            if heading and heading != current.name:
                if not current.is_empty:
                    sections.append(current)
                current = Section(name=heading, page=page_num)
            else:
                current.text += line + "\n"

        for md in page.get("tables", []) or []:
            if md:
                current.tables.append((page_num, md))
        for path in page.get("image_paths", []) or []:
            current.images.append((page_num, path))

    if not current.is_empty:
        sections.append(current)

    return _merge_runts(sections)


def _merge_runts(sections: list[Section]) -> list[Section]:
    """Fold tiny sections into the previous one.

    A heading detected mid-sentence produces a section with a handful of
    characters. Rather than emit a chunk labelled with a bogus heading, the
    text is returned to where it came from.
    """
    out: list[Section] = []
    for s in sections:
        if (
            out
            and len(s.text.strip()) < 160
            and not s.tables
            and not s.images
            and s.name not in _CANONICAL
        ):
            out[-1].text += "\n" + s.text
            out[-1].tables.extend(s.tables)
            out[-1].images.extend(s.images)
        else:
            out.append(s)
    return out
