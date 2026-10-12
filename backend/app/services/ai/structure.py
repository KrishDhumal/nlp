"""
Legal structure detection — turn cleaned pages into sections.

Contracts are drafted in a small number of recognisable styles, so headings are
detectable with patterns. But no assumption holds for every document, so this
module reports how confident it is: when too few headings are found relative to
the document's length, the caller falls back to page/paragraph segmentation
rather than inventing a structure that isn't there.
"""

from __future__ import annotations

import re
import uuid
from typing import List, Optional, Tuple

from app.domain.document import Section

# ── Heading patterns, most specific first ───────────────────────────────────
# ARTICLE I / SECTION 5 / CLAUSE 3.2 / SCHEDULE A
_KEYWORD_HEADING = re.compile(
    r"^\s*(?P<ref>(?:ARTICLE|Article|SECTION|Section|CLAUSE|Clause|SCHEDULE|Schedule"
    r"|EXHIBIT|Exhibit|ANNEXURE|Annexure|APPENDIX|Appendix|PART|Part)"
    r"\s+(?:[IVXLC]+|\d{1,2}(?:\.\d{1,2})*|[A-Z]))\s*[:.\-–]?\s*(?P<title>[^\n]{0,90})\s*$"
)

# 1. Definitions / 5.4 Termination for Convenience
_NUMBERED_HEADING = re.compile(
    r"^\s*(?P<ref>\d{1,2}(?:\.\d{1,2}){0,3})[.)]?\s+(?P<title>[A-Z][^\n]{2,90})\s*$"
)

# TERMINATION AND RENEWAL — an all-caps title on its own line
_CAPS_HEADING = re.compile(r"^\s*(?P<title>[A-Z][A-Z0-9 ,&/'\-\(\)\.]{4,70})\s*$")

# Common legal section names, used to rescue documents with unnumbered headings.
_KNOWN_HEADINGS = {
    "definitions", "interpretation", "scope of services", "scope of work",
    "term", "term and renewal", "termination", "payment", "payment terms",
    "fees", "fees and payment", "compensation", "confidentiality",
    "non-disclosure", "intellectual property", "indemnity", "indemnification",
    "limitation of liability", "liability", "warranties", "representations",
    "governing law", "dispute resolution", "arbitration", "force majeure",
    "assignment", "notices", "miscellaneous", "entire agreement",
    "severability", "data protection", "privacy", "signatures",
}

_NOISE = re.compile(r"^\s*(?:page\s+\d+|confidential|draft|execution copy)\s*$", re.IGNORECASE)


# A heading is a noun phrase. These verbs only appear in operative language, so
# a candidate containing one is body text no matter how it is capitalised.
_OPERATIVE_VERB = re.compile(
    r"\b(?:shall|must|may|will|agrees?|undertakes?|warrants?|represents?|"
    r"acknowledges?|is|are|was|were|has|have|hereby)\b",
    re.IGNORECASE,
)


def _looks_like_prose(text: str) -> bool:
    """
    Reject heading candidates that are really sentences.

    Legal drafting sets liability and warranty disclaimers in ALL CAPS *inside
    the body*, so capitalisation alone cannot identify a heading. Commas, a
    terminal full stop, an operative verb, and long all-caps runs all mark prose.
    """
    if "," in text or ";" in text:
        return True
    if text.endswith("."):
        return True
    if _OPERATIVE_VERB.search(text):
        return True
    if text.isupper() and len(text.split()) > 6:
        return True
    return False


def match_heading(line: str) -> Optional[Tuple[str, str]]:
    """Return (number, title) when `line` is a section heading, else None."""
    stripped = line.strip()
    if not stripped or len(stripped) > 110 or _NOISE.match(stripped):
        return None
    if stripped.endswith((",", ";", "and", "or")):
        return None

    match = _KEYWORD_HEADING.match(stripped)
    if match:
        title = match.group("title").strip()
        if title and _looks_like_prose(title):
            return None
        return match.group("ref").strip(), title

    match = _NUMBERED_HEADING.match(stripped)
    if match:
        title = match.group("title").strip()
        if _looks_like_prose(title) or len(title.split()) > 8:
            return None
        return match.group("ref").strip(), title

    match = _CAPS_HEADING.match(stripped)
    if match:
        title = match.group("title").strip()
        if _looks_like_prose(title):
            return None
        if 1 <= len(title.split()) <= 8:
            return "", title.title()

    # An unnumbered, title-cased known legal heading, e.g. "Governing Law".
    if stripped.lower().rstrip(":") in _KNOWN_HEADINGS and len(stripped) <= 60:
        return "", stripped.rstrip(":").title()

    return None


def detect_sections(pages: List[Tuple[int, str]]) -> Tuple[List[Section], bool]:
    """
    Split cleaned pages into sections.

    Returns (sections, structure_detected). `structure_detected` is False when
    the document yielded too few headings to be trusted as a structure — the
    caller then treats pages as the organising unit instead.
    """
    sections: List[Section] = []
    current_number, current_title = "", "Preamble"
    current_lines: List[str] = []
    start_page = pages[0][0] if pages else 1
    end_page = start_page
    heading_count = 0

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if not body:
            return
        sections.append(Section(
            section_id=f"sec-{uuid.uuid4().hex[:10]}",
            section_number=current_number,
            section_title=current_title,
            page_start=start_page,
            page_end=end_page,
            text=body,
        ))

    for page_number, page_text in pages:
        if not page_text.strip():
            continue
        end_page = page_number
        for line in page_text.split("\n"):
            heading = match_heading(line)
            if heading is None:
                current_lines.append(line)
                continue

            flush()
            heading_count += 1
            current_number, current_title = heading
            # Keep the heading line inside its section — it is useful context
            # for both the summariser and the embedding.
            current_lines = [line.strip()]
            start_page = page_number
            end_page = page_number

    flush()

    # Confidence: a real contract has a heading every few hundred words. If we
    # found almost none, structure detection failed.
    total_chars = sum(len(text) for _, text in pages)
    expected = max(2, total_chars // 4000)
    structure_detected = heading_count >= expected and heading_count >= 2

    if not sections:
        return page_fallback_sections(pages), False

    return sections, structure_detected


def _extract_page_title(text: str, page_number: int) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines[:6]:
        if _NOISE.match(line):
            continue
        heading = match_heading(line)
        if heading:
            num, title = heading
            return f"{num} {title}".strip() if num else title
        clean = line.strip(" :.-–")
        words = clean.split()
        if 2 <= len(words) <= 10 and len(clean) <= 75:
            if not _looks_like_prose(clean) and (clean.isupper() or clean.istitle()):
                return clean.title() if clean.isupper() else clean

    return f"Page {page_number}"


def page_fallback_sections(pages: List[Tuple[int, str]]) -> List[Section]:
    """
    One section per page, for documents where no structure could be detected.

    Keeps the rest of the pipeline uniform: downstream code always sees
    sections, and page traceability still holds exactly.
    """
    fallback: List[Section] = []
    for page_number, text in pages:
        if not text.strip():
            continue
        title = _extract_page_title(text, page_number)
        fallback.append(Section(
            section_id=f"page-{page_number}",
            section_number="",
            section_title=title,
            page_start=page_number,
            page_end=page_number,
            text=text.strip(),
        ))
    return fallback
