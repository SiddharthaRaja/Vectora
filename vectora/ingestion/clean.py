"""
Text cleaning, noise removal and chunk quality scoring.

This module is the fix for the single biggest corpus problem: the old
pipeline emitted chunks like ``"STE-6790; No.ofPages12"`` and
``"journal homepage: www.elsevier.com/locate/steroids"`` as retrievable
content, which then got cited as if they were findings.

Two defences, both needed:

1. **Line-level noise removal** strips running heads, page numbers,
   publisher furniture and licence boilerplate before chunking.
2. **A chunk-level quality score** catches whatever survives, because no
   regex list is ever complete. Chunks below the floor are dropped and the
   score is kept on the ones that pass, so retrieval can prefer clean text.

Everything here is pure: no I/O, no models. That makes it the easiest part
of the system to unit test, which matters because it is also the part that
quietly determines answer quality.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

#: Ligatures and typographic characters that PDF extraction leaves behind and
#: that break both lexical matching and the embedder's tokenizer.
_CHAR_FIXES = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
    "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
    "‘": "'", "’": "'", "‚": ",", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "—", "―": "—", "−": "-",
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
    "​": "", "‌": "", "‍": "", "﻿": "",
    "­": "",  # soft hyphen
}

#: Lines matching any of these are dropped outright. Anchored where possible
#: so they cannot eat real sentences that merely mention a DOI or a year.
_NOISE_LINE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^\s*$",
        r"^\s*[\d\s/|.-]{1,12}\s*$",                       # bare page numbers, "1 / 12"
        r"^\s*page\s+\d+\s*(of\s+\d+)?\s*$",
        r"^\s*https?://\S+\s*$",
        r"^\s*www\.\S+\s*$",
        r"^\s*(doi|isbn|issn|pmid|pmcid)\s*:?\s*\S*\s*$",
        r"^\s*©.*$",
        r"^\s*\(?c\)?\s*\d{4}\b.*(elsevier|springer|wiley|mdpi|taylor|sage|oxford|acs|rsc)",
        r"^\s*arxiv:\s*\d{4}\.\d+.*$",
        r"^\s*(received|accepted|published|revised|submitted|available online)\b.*\d.*$",
        r"^\s*journal\s+homepage\s*:.*$",
        r"^\s*contents\s+lists\s+available\s+at.*$",
        r"^\s*(sciencedirect|scienceDirect|pubmed|researchgate|crossmark)\s*$",
        r"^\s*downloaded\s+(from|by)\b.*$",
        r"^\s*this\s+article\s+is\s+(an\s+open\s+access|licensed|distributed)\b.*$",
        r"^\s*licensed\s+under\s+(a\s+)?creative\s+commons\b.*$",
        r"^\s*all\s+rights\s+reserved\.?\s*$",
        r"^\s*(corresponding\s+author|e-?mail\s+address(es)?)\s*:.*$",
        r"^\s*\*?\s*correspondence\s*:.*$",
        r"^\s*(table|figure|fig|scheme)\s+\d+\s*[.:]?\s*$",  # orphan caption stub
        r"^\s*g\s*model\s*$",                                # "GModel" artifact
        r"^\s*article\s+in\s+press\s*$",
        r"^\s*[A-Z]{2,6}-\d{3,6}\s*;?\s*(no\.?\s*of\s*pages\s*\d+)?\s*$",  # "STE-6790; No.ofPages12"
        r"^\s*no\.?\s*of\s*pages\s*\d+\s*$",
        r"^\s*\d+\s+[A-Z][a-z]+\s+et\s+al\.?\s*/?.*$",      # running head
        r"^\s*[A-Z][a-z]+\s+et\s+al\.?\s*/\s*.*\(\d{4}\)\s*.*$",
        r"^\s*(supplementary|supporting)\s+(material|information)\s*$",
        r"^\s*conflicts?\s+of\s+interest\s*:?\s*(none|the\s+authors?\s+declare).*$",
        r"^\s*\d+\s*of\s*\d+\s*$",
        r"^\s*[^\w\s]{1,6}\s*$",                             # stray punctuation lines
    )
)

#: Section names that carry no retrievable claims. Chunks from these are
#: dropped: nobody asks a question whose answer is a funding statement, and
#: author lists match far too many queries lexically.
EXCLUDED_SECTIONS: frozenset[str] = frozenset(
    {
        "references", "reference", "bibliography", "works cited",
        "acknowledgement", "acknowledgements", "acknowledgment", "acknowledgments",
        "funding", "funding statement", "author contributions",
        "conflict of interest", "conflicts of interest", "declaration of interest",
        "declaration of competing interest", "competing interests",
        "supplementary material", "supplementary materials",
        "supporting information", "data availability", "data availability statement",
        "ethics statement", "institutional review board statement",
        "informed consent statement", "abbreviations", "nomenclature",
        "copyright", "license", "publisher's note", "editorial board",
    }
)

#: A reference-list line: "[12] Smith J, Jones A. Title. Journal. 2019;4:12-20."
_REFERENCE_LINE = re.compile(
    r"^\s*(\[\d{1,3}\]|\(\d{1,3}\)|\d{1,3}\.)\s+\S+.*\b(19|20)\d{2}\b"
)

#: Publisher furniture that appears *mid-line*, not at the start of one.
#:
#: The line-anchored patterns above only work on text that still has its line
#: breaks. The existing corpus was produced by a pipeline that joined every
#: line with a space before writing the chunk, so a running head ends up
#: embedded in the middle of the prose ("…conclusions. STE-6790; No.of
#: Pages12 A.M.Wallaceetal./Steroidsxxx (2010) xxx-xxx 7 batch precision…").
#: These catch it wherever it sits.
_INLINE_FURNITURE: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"[A-Z]{2,6}-\d{3,6}\s*;?\s*No\.?\s*of\s*Pages\s*\d+",   # STE-6790; No.ofPages12
        r"\bNo\.?\s*of\s*Pages\s*\d+",
        r"\b[A-Z]\.?[A-Z]?\.?\s?[A-Z][a-zA-Z'\-]*\s?et\s?al\.?\s*/\s*\S+",  # Wallaceetal./Steroidsxxx
        r"\b[A-Za-z]+xxx\s*\(\d{4}\)\s*xxx\s*[-–]\s*xxx",        # Steroidsxxx (2010) xxx-xxx
        r"\bG\s?Model\b",
        r"\bARTICLE\s+IN\s+PRESS\b",
        r"\bjournal\s+homepage\s*:\s*\S+",
        r"\bContents\s+lists\s+available\s+at\s+\S+",
        r"\bDownloaded\s+(from|by)\s+\S+",
        r"\b(?:©|\(c\))\s*\d{4}\s+[A-Z][\w.]*(\s+[A-Z][\w.]*){0,3}(\s+(Ltd|Inc|B\.V|GmbH|AG))?\.?",
        r"\bAll\s+rights\s+reserved\.?",
        r"\bThis\s+article\s+is\s+(an\s+open\s+access|licensed|distributed)[^.]{0,80}\.",
        r"\bLicensed\s+under\s+(a\s+)?Creative\s+Commons[^.]{0,60}\.?",
        r"\bdoi\s*:\s*10\.\d{4,9}/\S+",
        r"\b(ISSN|ISBN|PMID|PMCID)\s*:?\s*[\d\-Xx]+",
    )
)

#: Section labels that are furniture rather than subject matter. Used when a
#: section name is inherited from an older corpus and cannot be re-derived.
_FURNITURE_SECTION = re.compile(
    r"article\s+in\s+press|g\s*model|journal|homepage|contents\s+lists|"
    r"elsevier|springer|wiley|mdpi|sciencedirect|crossmark|"
    r"all\s+rights\s+reserved|no\.?\s*of\s*pages|^[a-z]{2,6}-\d{3,6}$|"
    r"^(manuscript|untitled|accepted|received|preprint)$",
    re.IGNORECASE,
)

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
_MULTISPACE = re.compile(r"[ \t]{2,}")
_MULTINEWLINE = re.compile(r"\n{3,}")
#: "wordWord" run together across a line break in two-column layouts.
_GLUED_WORDS = re.compile(r"(?<=[a-z]{2})(?=[A-Z][a-z]{2})")


def normalise_chars(text: str) -> str:
    """Unicode-normalise and replace the characters PDFs get wrong."""
    text = unicodedata.normalize("NFKC", text)
    for bad, good in _CHAR_FIXES.items():
        if bad in text:
            text = text.replace(bad, good)
    return text


def repair_linebreaks(text: str) -> str:
    """Rejoin words split across a line break by hyphenation."""
    return _HYPHEN_BREAK.sub(r"\1\2", text)


def unglue(text: str, max_fixes: int = 40) -> str:
    """Insert spaces into runs like ``"methodsWere"``.

    PDF two-column extraction frequently drops the space between a word and
    the next capitalised word. Capped, because on text that is already clean
    this would otherwise split legitimate camel case and chemical names.
    """
    if " " in text and len(text) / max(text.count(" "), 1) < 9:
        return text  # spacing looks normal, leave it alone
    return _GLUED_WORDS.sub(" ", text, count=max_fixes)


def is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if _REFERENCE_LINE.match(stripped):
        return True
    return any(p.match(stripped) for p in _NOISE_LINE_PATTERNS)


def clean_block(text: str) -> str:
    """Full cleaning pass over a block of extracted page text.

    Order matters: repair hyphenation before dropping lines, or a word split
    across a dropped line is lost; normalise characters first, or the regexes
    miss lines containing ligatures.
    """
    if not text:
        return ""
    text = normalise_chars(text)
    text = repair_linebreaks(text)
    kept = [ln.strip() for ln in text.split("\n") if not is_noise_line(ln)]
    joined = "\n".join(kept)
    joined = _MULTISPACE.sub(" ", joined)
    joined = _MULTINEWLINE.sub("\n\n", joined)
    return joined.strip()


def strip_inline_noise(text: str) -> str:
    """Remove URLs, e-mail addresses and embedded publisher furniture.

    Unlike :func:`clean_block`, this does not need line structure, so it is
    what recovers already-flattened text from an older corpus. Applied to
    freshly parsed text too, since a running head can be glued onto a line
    of prose by the PDF extractor.
    """
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    for pattern in _INLINE_FURNITURE:
        text = pattern.sub(" ", text)
    # Page numbers stranded between the removed head and the prose.
    text = re.sub(r"\s+\d{1,3}\s+(?=[a-z])", " ", text)
    return _MULTISPACE.sub(" ", text).strip(" .;,-")


def is_furniture_section(section: str) -> bool:
    """True when a section label is publisher furniture, not subject matter."""
    return bool(section) and bool(_FURNITURE_SECTION.search(section.strip()))


@dataclass(frozen=True)
class Quality:
    """Why a chunk was kept or dropped."""

    score: float
    reasons: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.score > 0.0


def score_quality(
    text: str,
    *,
    min_tokens: int,
    min_alpha_ratio: float,
    chunk_type: str = "text",
) -> Quality:
    """Heuristic 0-1 confidence that ``text`` is retrievable prose.

    A score of exactly 0 means drop it. Tables are scored leniently because
    a markdown grid is legitimately low on letters and sentence punctuation.
    """
    reasons: list[str] = []
    stripped = text.strip()
    if not stripped:
        return Quality(0.0, ("empty",))

    tokens = max(len(stripped) // 4, len(stripped.split()))
    letters = sum(c.isalpha() for c in stripped)
    alpha_ratio = letters / len(stripped)
    words = stripped.split()
    word_count = len(words)

    if chunk_type == "table":
        # A table needs structure and a couple of real words, nothing more.
        if stripped.count("|") < 4:
            return Quality(0.0, ("table_without_structure",))
        if word_count < 6:
            return Quality(0.0, ("table_too_small",))
        return Quality(0.85, ())

    # ── hard rejections ───────────────────────────────────────────────────
    if tokens < min_tokens:
        return Quality(0.0, (f"too_short({tokens}<{min_tokens})",))
    if alpha_ratio < min_alpha_ratio:
        return Quality(0.0, (f"low_alpha({alpha_ratio:.2f})",))
    if word_count < 12:
        return Quality(0.0, ("too_few_words",))

    long_words = [w for w in words if len(w) > 2]
    if long_words and len(long_words) / word_count < 0.45:
        return Quality(0.0, ("mostly_fragments",))

    # A chunk that is nearly all one repeated token is extraction debris.
    distinct = len({w.lower() for w in words})
    if distinct / word_count < 0.35:
        return Quality(0.0, ("low_lexical_diversity",))

    if sum(1 for ln in stripped.split("\n") if _REFERENCE_LINE.match(ln.strip())) >= 3:
        return Quality(0.0, ("reference_list",))

    # ── soft penalties ────────────────────────────────────────────────────
    score = 1.0
    if not re.search(r"[.!?]", stripped):
        score -= 0.30
        reasons.append("no_sentence_punctuation")
    if alpha_ratio < min_alpha_ratio + 0.12:
        score -= 0.15
        reasons.append("letter_sparse")

    digits = sum(c.isdigit() for c in stripped) / len(stripped)
    if digits > 0.30:
        score -= 0.20
        reasons.append("digit_heavy")

    upper_words = sum(1 for w in words if len(w) > 3 and w.isupper())
    if word_count and upper_words / word_count > 0.45:
        score -= 0.20
        reasons.append("shouty")

    avg_word = sum(len(w) for w in words) / word_count
    if avg_word > 14 or avg_word < 2.6:
        score -= 0.15
        reasons.append("odd_word_length")

    glue = glued_ratio(stripped)
    if glue > 0.20:
        # Space loss this severe makes the chunk unmatchable lexically and
        # poorly tokenised for the embedder. Not recoverable by cleaning —
        # it has to be re-parsed from the PDF — so it is heavily penalised
        # and reported rather than silently indexed as normal prose.
        return Quality(0.0, ("spaces_lost_in_parsing",))
    if glue > 0.06:
        score -= 0.35
        reasons.append("partial_space_loss")

    return Quality(max(round(score, 3), 0.05), tuple(reasons))


#: A run of letters longer than any real English or scientific word, which is
#: what a lost space leaves behind ("requiremoreexpensiveequipment").
_GLUED_RUN = re.compile(r"[A-Za-z]{19,}")


def glued_ratio(text: str) -> float:
    """Fraction of characters sitting inside implausibly long letter runs.

    PDF text extraction with the wrong character tolerance drops the spaces
    between words, and the damage is invisible to every other check: the
    text is pure letters, has normal punctuation and scores well on letter
    ratio and diversity. This is what detects it.
    """
    if not text:
        return 0.0
    glued = sum(len(m.group(0)) for m in _GLUED_RUN.finditer(text))
    return glued / len(text)


def is_excluded_section(section: str) -> bool:
    """True for sections whose content should never be retrievable."""
    if not section:
        return False
    key = re.sub(r"^\d+(\.\d+)*\.?\s*", "", section.strip().lower()).strip(" .:-")
    if key in EXCLUDED_SECTIONS:
        return True
    return any(key.startswith(ex) for ex in ("reference", "acknowledg", "bibliograph"))
