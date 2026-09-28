"""
Text cleaning and normalisation — shared by every consumer of the pipeline.

The hard constraint here is that cleaning must never change legal meaning.
Whitespace, line-wrap artefacts and repeated page furniture are noise and get
removed. Dates, amounts, percentages, clause numbers, party names and
conditional words are signal and are left exactly as drafted.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Dict, List, Tuple

# Page furniture: standalone page numbers, "Page 3 of 12", rule lines.
_PAGE_ARTIFACT = re.compile(
    r"^\s*(?:"
    r"page\s+\d+(?:\s+of\s+\d+)?"
    r"|\d+\s*\|\s*page"
    r"|[-–—]\s*\d+\s*[-–—]"
    r"|\d{1,3}"
    r"|[-–—_=]{3,}"
    r")\s*$",
    re.IGNORECASE,
)

# Characters PDF producers emit that break downstream matching. Mapped to the
# plain ASCII equivalent — note the quote marks matter for defined-term
# detection, e.g. ("Provider").
_CHAR_FIXES = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "–": "-", "—": "-", "―": "-", "−": "-",
    "…": "...", " ": " ", "​": "", "‌": "",
    "‍": "", "﻿": "", "•": "- ", "­": "",
    "ﬁ": "fi", "ﬂ": "fl",
}

# A line that ends mid-sentence has been wrapped by the PDF, not by the author.
_WRAP_CONTINUES = re.compile(r"[a-z0-9,;:\-\(]$")
# ...unless the next line starts a new clause or a heading.
_NEW_BLOCK_STARTS = re.compile(
    r"^\s*(?:"
    r"\d{1,2}(?:\.\d{1,2})*\.?\s"          # 5.  or 5.4
    r"|\([a-z0-9ivx]{1,4}\)"               # (a) (iv) (2)
    r"|[-•*]\s"                            # bullets
    r"|(?:ARTICLE|SECTION|CLAUSE|SCHEDULE|EXHIBIT|ANNEXURE)\b"
    r"|[A-Z][A-Z\s]{4,}$"                  # ALL CAPS heading
    r")",
    re.IGNORECASE,
)


def normalize_characters(text: str) -> str:
    """Fix encoding artefacts without touching any meaningful character."""
    if not text:
        return ""
    # NFKC folds compatibility forms (ligatures, full-width digits) to canonical.
    text = unicodedata.normalize("NFKC", text)
    for bad, good in _CHAR_FIXES.items():
        text = text.replace(bad, good)
    # Strip control characters, keeping tab and newline.
    return "".join(
        ch for ch in text
        if ch in "\t\n" or not unicodedata.category(ch).startswith("C")
    )


def repair_line_wrapping(text: str) -> str:
    """
    Rejoin lines the PDF broke mid-sentence, and repair hyphenated splits.

    "indemni-\\nfication" becomes "indemnification"; a line ending in a lowercase
    word followed by a lowercase continuation is joined with a space. Lines that
    look like the start of a clause or heading are never joined onto the previous
    line, because that would destroy the structure we detect later.
    """
    if not text:
        return ""

    # Hyphen split across a line break.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    lines = text.split("\n")
    out: List[str] = []
    for line in lines:
        stripped = line.strip()
        if (
            out
            and stripped
            and out[-1]
            and _WRAP_CONTINUES.search(out[-1])
            and not _NEW_BLOCK_STARTS.match(stripped)
            and not stripped[0].isupper()
        ):
            out[-1] = f"{out[-1]} {stripped}"
        else:
            out.append(stripped)
    return "\n".join(out)


def find_repeated_furniture(page_texts: List[str], min_pages: int = 3) -> set:
    """
    Detect running headers and footers by looking for identical short lines that
    appear near the top or bottom of most pages.

    Done across the whole document rather than per page, because a single page
    cannot tell a running header from a genuine heading.
    """
    if len(page_texts) < min_pages:
        return set()

    candidates: Counter = Counter()
    for text in page_texts:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        # Only the first and last three lines can be furniture.
        for line in lines[:3] + lines[-3:]:
            if 3 <= len(line) <= 90:
                candidates[line] += 1

    threshold = max(min_pages, int(len(page_texts) * 0.6))
    repeated = {line for line, count in candidates.items() if count >= threshold}

    # A repeated line that carries a clause number is structure, not furniture.
    return {
        line for line in repeated
        if not re.match(r"^\s*\d{1,2}(?:\.\d{1,2})*\.?\s+\S", line)
    }


def clean_page_text(text: str, furniture: set = frozenset()) -> str:
    """
    Clean one page: normalise characters, repair wrapping, drop artefacts and
    furniture, collapse runs of whitespace while keeping paragraph breaks.
    """
    if not text:
        return ""

    text = normalize_characters(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = repair_line_wrapping(text)

    kept: List[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if _PAGE_ARTIFACT.match(stripped):
            continue
        if stripped in furniture:
            continue
        kept.append(stripped)

    text = "\n".join(kept)
    # Three or more newlines collapse to a paragraph break; two are preserved.
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def clean_pages(raw_pages: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    """
    Clean a whole document's pages, using cross-page analysis for furniture.

    Returns (page_number, cleaned_text) preserving page boundaries — page-level
    citation depends on this mapping staying intact.
    """
    normalized = [(num, normalize_characters(text or "")) for num, text in raw_pages]
    furniture = find_repeated_furniture([text for _, text in normalized])
    return [(num, clean_page_text(text, furniture)) for num, text in normalized]


# ── Meaning-preservation check (used by tests and by the summariser guard) ──
_PRESERVE_PATTERNS: Dict[str, re.Pattern] = {
    "money": re.compile(r"(?:[$₹£€]\s?[\d,]+(?:\.\d+)?|\b(?:USD|INR|EUR|GBP)\s?[\d,]+)"),
    "percent": re.compile(r"\b\d+(?:\.\d+)?\s?%"),
    # Contracts write periods as "thirty (30) calendar days", so the digits are
    # usually followed by a closing parenthesis rather than whitespace. Requiring
    # `\d+\s+days` made this check silently inert on real contract language.
    "days": re.compile(
        r"\b\d+\s*\)?\s*(?:calendar|business|working)?\s*"
        r"(?:days?|months?|years?|weeks?)\b",
        re.IGNORECASE,
    ),
    "dates": re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|October|"
        r"November|December)\s+\d{1,2},?\s+\d{4}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    ),
    "clause_refs": re.compile(r"\bSection\s+\d+(?:\.\d+)*\b", re.IGNORECASE),
}


def extract_preserved_values(text: str) -> Dict[str, List[str]]:
    """
    Pull out the values that must survive cleaning and summarisation.

    Used to assert that cleaning did not silently drop a monetary amount or a
    notice period, and to flag a summary that invented or lost a figure.
    """
    return {
        name: pattern.findall(text)
        for name, pattern in _PRESERVE_PATTERNS.items()
    }
