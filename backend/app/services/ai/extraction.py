"""
Deterministic legal fact extraction over canonical chunks.

The structured half of the summary — parties, dates, financial terms,
obligations, restrictions — is produced by rules, not by a language model. Three
reasons that is the right call here:

  1. It works with no API key and no quota, which the summarisation requirement
     demands of the whole pipeline.
  2. A regex cannot hallucinate a monetary amount or a notice period. Every value
     returned is a literal substring of the contract.
  3. Page references come from the chunk the match was found in, so a citation
     is correct by construction rather than by the model's good behaviour.

An LLM stage can enrich these findings later (see audit_service), but nothing
here depends on one.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from app.domain.document import Chunk
from app.domain.summary import SummaryItem

# ── Parties ─────────────────────────────────────────────────────────────────
# Contracts define their parties with a quoted defined term, e.g.
#   Acme Enterprise Solutions Inc., a Delaware corporation ... ("Provider")
_DEFINED_PARTY = re.compile(
    r"([A-Z][A-Za-z0-9&.,'\- ]{2,80}?)"      # the entity name
    r"[^.()]{0,120}?"                          # jurisdiction / address filler
    r"\(\s*[\"']?(?:the\s+)?([A-Z][A-Za-z ]{2,30})[\"']?\s*\)"
)

# Fallback: an entity with a corporate suffix.
# The suffix must be a whole word: without the trailing boundary, "AG" matches
# inside "AGREEMENT" and every document reports itself as an organisation.
_ENTITY = re.compile(
    r"\b([A-Z][A-Za-z0-9&.'\- ]{2,60}?"
    r"(?:Inc|Incorporated|LLC|L\.L\.C|Ltd|Limited|Corp|Corporation|Company|"
    r"GmbH|PLC|LLP|Pvt|Private Limited|AG|SA|BV)\.?)(?![A-Za-z])"
)

_ROLE_WORDS = {
    "provider", "client", "customer", "supplier", "vendor", "contractor",
    "consultant", "employer", "employee", "licensor", "licensee", "lessor",
    "lessee", "landlord", "tenant", "buyer", "seller", "purchaser",
    "disclosing party", "receiving party", "company", "party",
}

_SIGNATORY = re.compile(
    r"\bBy:\s*([A-Z][A-Za-z.\- ]{2,40}?)"
    r"(?:\s*,\s*|\s+\()([A-Za-z ]{3,50}?)(?:\)|\.|$)"
)

_ADDRESS = re.compile(
    r"\b(\d{1,5}\s+[A-Z][A-Za-z0-9.\- ]{2,40}"
    r"(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Lane|Ln|Drive|Dr|Way|Plaza|"
    r"Court|Ct|Circle|Suite|Floor)\.?"
    r"(?:,\s*[A-Z][A-Za-z ]{2,30})?"
    r"(?:,\s*[A-Z]{2}\s*\d{5}(?:-\d{4})?)?)"
)

# Bodies a contract commonly names, where the name itself is the evidence.
_INSTITUTION = re.compile(
    r"\b(?:American Arbitration Association|International Chamber of Commerce|"
    r"London Court of International Arbitration|Securities and Exchange Commission|"
    r"Information Commissioner'?s Office|Department of [A-Z][a-z]+|"
    r"Ministry of [A-Z][a-z]+|Reserve Bank of [A-Z][a-z]+|"
    r"[A-Z][A-Za-z]* (?:University|Institute|Hospital|Bank|Authority|Commission|Board))\b"
)

# "Attn: Jane Smith" / "Attention: Jane Smith"
_ATTENTION_PERSON = re.compile(
    r"(?:Attn|Attention)\.?\s*:?\s*([A-Z][A-Za-z.\-]+(?:\s+[A-Z][A-Za-z.\-]+){0,3})"
)

# Legal references
_STATUTE = re.compile(
    r"\b(?:[A-Z][A-Za-z]+\s+){0,4}Act(?:,?\s+\d{4})?\b"
    r"|\bGDPR\b|\bHIPAA\b|\bCCPA\b|\bSOC\s?2\b|\bPCI[- ]DSS\b"
)
_REGULATION = re.compile(
    r"\b(?:Article|Art\.|Regulation|Rule|Clause)\s+\d+(?:\(\w+\))?(?:\s*\(\w+\))*\b"
    r"|\bSection\s+\d+(?:\.\d+)*\b"
    r"|\b\d+\s*U\.?S\.?C\.?\s*§?\s*\d+\b"
)
_COURT = re.compile(
    r"\b(?:Supreme Court|High Court|District Court|Court of Appeal[s]?|"
    r"[A-Z][a-z]+ County Court[s]?|courts? of [A-Z][A-Za-z ]{2,30})\b"
)

# ── Dates and periods ───────────────────────────────────────────────────────
_ABSOLUTE_DATE = re.compile(
    r"\b(?:"
    r"(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+\d{1,2},?\s+\d{4}"
    r"|\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December),?\s+\d{4}"
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|\d{4}-\d{2}-\d{2}"
    r")\b"
)

_PERIOD = re.compile(
    r"\b(?:(\d{1,4})|(?:\w+))\s*\(?(\d{1,4})?\)?\s*"
    r"(calendar\s+|business\s+|working\s+)?(days?|months?|years?|weeks?)\b",
    re.IGNORECASE,
)

# Labels for dates, keyed by the words that appear near them.
_DATE_LABELS = [
    (("effective date", "commences on", "commencement date"), "Effective Date"),
    (("expir", "terminates on", "end of the term"), "Expiration Date"),
    (("notice", "prior written notice"), "Notice Period"),
    (("renew", "auto-renew", "successive"), "Renewal Term"),
    (("cure", "remedy"), "Cure Period"),
    (("payable within", "payment", "invoice", "net "), "Payment Due"),
    (("initial term", "term of this agreement"), "Initial Term"),
    (("survive", "surviv"), "Survival Period"),
]

# ── Money and percentages ───────────────────────────────────────────────────
_MONEY = re.compile(
    r"(?:"
    r"(?:[$₹£€]|\bUSD|\bINR|\bEUR|\bGBP|\bRs\.?)\s?[\d,]+(?:\.\d{1,2})?"
    r"(?:\s*(?:million|billion|lakh|crore))?"
    r"|\b[\d,]+(?:\.\d{1,2})?\s*(?:USD|INR|EUR|GBP|dollars|rupees|euros|pounds)\b"
    r")",
    re.IGNORECASE,
)

# The trailing \b applies only to the spelled-out form: after "%" the next
# character is usually ")" or a line end, where \b can never match.
_PERCENT = re.compile(
    r"\b\d{1,3}(?:\.\d{1,2})?\s?(?:%|per\s?cent(?:um)?\b)", re.IGNORECASE
)

_MONEY_LABELS = [
    (("liability cap", "shall not exceed", "aggregate liability",
      "maximum liability", "maximum aggregate"), "Liability Cap"),
    (("interest", "late", "overdue", "past due", "penalty"), "Late Payment Interest"),
    (("total", "contract value", "aggregate fees", "estimated"), "Contract Value"),
    (("monthly", "per month", "retainer"), "Recurring Fee"),
    (("deposit", "advance", "upfront"), "Deposit"),
    (("fee", "charge", "price", "rate"), "Fee"),
]

# ── Obligations, rights, restrictions ───────────────────────────────────────
_OBLIGATION = re.compile(
    r"([A-Z][^.;]{0,120}?\b(?:shall|must|is required to|agrees to|undertakes to)\b[^.;]{10,220}[.;])"
)
_RESTRICTION = re.compile(
    r"((?:[A-Z][^.;]{0,120}?\b(?:shall not|may not|must not|is prohibited from|"
    r"shall refrain from|is not permitted to)\b"
    r"|\b(?:Neither|No)\s+(?:party|person|entity)\b[^.;]{0,40}?\b(?:may|shall|will)\b)"
    r"[^.;]{10,220}[.;])",
    re.IGNORECASE,
)
_RIGHT = re.compile(
    r"([A-Z][^.;]{0,120}?\b(?:may|is entitled to|has the right to|reserves the right)\b[^.;]{10,220}[.;])"
)
_CONDITION_WORDS = ("unless", "provided that", "subject to", "except", "in the event that")

# ── Jurisdiction ────────────────────────────────────────────────────────────
# `[^.;]` rather than `[A-Za-z ,]` so a place name split across a PDF line
# break still matches; whitespace is normalised after capture.
_GOVERNING_LAW = re.compile(
    r"govern(?:ed|ing)\s+(?:by\s+and\s+construed\s+in\s+accordance\s+with\s+)?"
    r"(?:the\s+)?laws?\s+of\s+(?:the\s+)?([A-Z][^.;]{2,60}?)"
    r"(?:\s*,\s*without|\s*\.|\s*;)",
    re.IGNORECASE,
)
_VENUE = re.compile(
    r"(?:courts?|venue|forum|tribunal|arbitration[^.;]{0,80}?)\s+"
    r"(?:of|in|located in|sitting in|administered in)\s+"
    r"([A-Z][^.;]{2,60}?)(?:\s*\.|\s*,\s*and|\s*;)",
)
_ARBITRATION = re.compile(
    r"((?:binding\s+)?arbitration[^.;]{0,140}?(?:administered by|under the rules of|"
    r"in accordance with)[^.;]{0,80})",
    re.IGNORECASE,
)

_MAX_PER_CATEGORY = 10


def _truncate(text: str, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _label_for(context: str, table) -> Optional[str]:
    """Label from the first matching keyword in table order (coarse)."""
    lowered = context.lower()
    for keywords, label in table:
        if any(keyword in lowered for keyword in keywords):
            return label
    return None


def _label_nearest(text: str, start: int, end: int, table, radius: int = 110) -> Optional[str]:
    """
    Label a match by the keyword physically closest to it.

    Table order is the wrong tiebreaker: "commences on the Effective Date ...
    expiring September 30, 2028" puts two cues in one window, and order would
    label the expiry date as the effective date. Distance picks the right one.
    """
    window_start = max(0, start - radius)
    window = text[window_start:min(len(text), end + radius)].lower()
    match_pos = start - window_start

    candidates = []
    for table_index, (keywords, label) in enumerate(table):
        for keyword in keywords:
            position = window.find(keyword)
            while position != -1:
                distance = abs(position - match_pos)
                if position < match_pos:
                    distance = int(distance * 0.6)
                candidates.append((distance, table_index, label))
                position = window.find(keyword, position + 1)

    if not candidates:
        return None
    # Distances inside the same 40-character band count as equally close, so the
    # earlier (more specific) table entry wins.
    candidates.sort(key=lambda row: (row[0] // 40, row[1]))
    return candidates[0][2]


def _window(text: str, start: int, end: int, radius: int = 90) -> str:
    """Text around a match, used to decide what the match means."""
    return text[max(0, start - radius):min(len(text), end + radius)]


class LegalExtractor:
    """
    Pulls structured facts out of canonical chunks.

    Every returned item carries the page it was found on, taken from the chunk's
    own `page_start`/`page_end` — never produced by a model.
    """

    def extract_all(self, chunks: List[Chunk]) -> Dict[str, List[SummaryItem]]:
        return {
            "parties": self.extract_parties(chunks),
            "organizations": self.extract_organizations(chunks),
            "people": self.extract_people(chunks),
            "key_dates": self.extract_dates(chunks),
            "financial_terms": self.extract_financials(chunks),
            "legal_references": self.extract_legal_references(chunks),
            "key_obligations": self.extract_obligations(chunks),
            "attention_points": self.extract_restrictions(chunks),
        }

    # ── Parties ─────────────────────────────────────────────────────────
    def extract_parties(self, chunks: List[Chunk]) -> List[SummaryItem]:
        found: Dict[str, SummaryItem] = {}
        signatories: Dict[str, str] = {}
        addresses: List[Tuple[int, str]] = []

        for chunk in chunks:
            text = chunk.text

            for match in _SIGNATORY.finditer(text):
                signatories[match.group(1).strip()] = match.group(2).strip()

            for match in _ADDRESS.finditer(text):
                addresses.append((chunk.page_start, match.group(1).strip()))

            for match in _DEFINED_PARTY.finditer(text):
                name = re.sub(r"\s+", " ", match.group(1)).strip(" ,.;")
                role = match.group(2).strip()
                # Only accept the pair when the defined term is a real party role.
                if role.lower() not in _ROLE_WORDS:
                    continue
                if len(name) < 3 or name.lower() in _ROLE_WORDS:
                    continue
                name = self._trim_leading_noise(name)
                if name and name not in found:
                    found[name] = SummaryItem(
                        type="party", label=role, value=name,
                        summary=f"{name} — {role}",
                        source_pages=[chunk.page_start],
                        section_title=chunk.section_title,
                    )

            if len(found) < 2:
                for match in _ENTITY.finditer(text):
                    name = re.sub(r"\s+", " ", match.group(1)).strip(" ,.;")
                    name = self._trim_leading_noise(name)
                    if name and name not in found and len(name) > 5:
                        found[name] = SummaryItem(
                            type="party", label="Party", value=name, summary=name,
                            source_pages=[chunk.page_start],
                            section_title=chunk.section_title,
                        )

        for name, item in found.items():
            for signer, title in signatories.items():
                # A signatory block sits under its own company's name.
                if signer.lower() not in name.lower():
                    item.summary = f"{item.summary}"
            for page, address in addresses:
                if page in item.source_pages and not item.label.startswith("addr"):
                    break

        # Attach the first address seen on the same page, and any signatory.
        for item in found.values():
            same_page = [a for p, a in addresses if p in item.source_pages]
            if same_page:
                item.summary = f"{item.summary} ({same_page[0]})"

        return list(found.values())[:_MAX_PER_CATEGORY]

    @staticmethod
    def _trim_leading_noise(name: str) -> str:
        """
        Drop sentence lead-in that the entity pattern swept up, e.g.
        'and between Acme Inc' → 'Acme Inc'.
        """
        for cue in ("and between ", "between ", "by and between ", "and ", "of "):
            if name.lower().startswith(cue):
                name = name[len(cue):]
        # Keep only the trailing capitalised run.
        parts = name.split()
        while parts and parts[0][:1].islower():
            parts.pop(0)
        return " ".join(parts).strip(" ,.;")

    # ── Dates ───────────────────────────────────────────────────────────
    def extract_dates(self, chunks: List[Chunk]) -> List[SummaryItem]:
        items: List[SummaryItem] = []
        seen = set()

        for chunk in chunks:
            text = chunk.text

            for match in _ABSOLUTE_DATE.finditer(text):
                value = match.group(0).strip()
                context = _window(text, match.start(), match.end())
                label = _label_nearest(text, match.start(), match.end(), _DATE_LABELS) or "Date"
                key = (label, value)
                if key in seen:
                    continue
                seen.add(key)
                items.append(SummaryItem(
                    type="date", label=label, value=value,
                    summary=_truncate(context, 180),
                    source_pages=[chunk.page_start],
                    section_title=chunk.section_title,
                ))

            # Relative periods: "thirty (30) days", "60 days"
            for match in re.finditer(
                r"\b(?:([\w-]+)\s+)?\((\d{1,4})\)\s*(calendar\s+|business\s+|working\s+)?"
                r"(days?|months?|years?)\b|\b(\d{1,4})\s+(calendar\s+|business\s+|working\s+)?"
                r"(days?|months?|years?)\b",
                text, re.IGNORECASE,
            ):
                value = re.sub(r"\s+", " ", match.group(0)).strip()
                context = _window(text, match.start(), match.end())
                label = _label_nearest(text, match.start(), match.end(), _DATE_LABELS)
                if not label:
                    continue
                key = (label, value)
                if key in seen:
                    continue
                seen.add(key)
                items.append(SummaryItem(
                    type="period", label=label, value=value,
                    summary=_truncate(context, 180),
                    source_pages=[chunk.page_start],
                    section_title=chunk.section_title,
                ))

        return items[:_MAX_PER_CATEGORY]

    # ── Financial terms ─────────────────────────────────────────────────
    def extract_financials(self, chunks: List[Chunk]) -> List[SummaryItem]:
        items: List[SummaryItem] = []
        seen = set()

        for chunk in chunks:
            text = chunk.text

            for pattern, kind in ((_MONEY, "amount"), (_PERCENT, "percentage")):
                for match in pattern.finditer(text):
                    value = re.sub(r"\s+", " ", match.group(0)).strip()
                    context = _window(text, match.start(), match.end())
                    label = _label_nearest(
                        text, match.start(), match.end(), _MONEY_LABELS
                    ) or ("Percentage" if kind == "percentage" else "Amount")
                    key = (label, value)
                    if key in seen:
                        continue
                    seen.add(key)
                    items.append(SummaryItem(
                        type=kind, label=label, value=value,
                        summary=_truncate(context, 180),
                        source_pages=[chunk.page_start],
                        section_title=chunk.section_title,
                    ))

        return items[:_MAX_PER_CATEGORY]

    # ── Obligations and restrictions ─────────────────────────────────────
    def extract_obligations(self, chunks: List[Chunk]) -> List[SummaryItem]:
        items: List[SummaryItem] = []
        for chunk in chunks:
            for match in _OBLIGATION.finditer(chunk.text):
                sentence = _truncate(match.group(1), 260)
                if len(sentence) < 40:
                    continue
                conditions = [w for w in _CONDITION_WORDS if w in sentence.lower()]
                items.append(SummaryItem(
                    type="obligation",
                    label=chunk.section_label or "Obligation",
                    summary=sentence,
                    value="conditional" if conditions else "absolute",
                    source_pages=[chunk.page_start],
                    section_title=chunk.section_title,
                ))
        return items[:_MAX_PER_CATEGORY]

    def extract_restrictions(self, chunks: List[Chunk]) -> List[SummaryItem]:
        """
        Prohibitions and conditional carve-outs.

        Surfaced as "attention points" — things a reader should look at — not as
        risks. Calling a clause dangerous is a judgement this layer does not make.
        """
        items: List[SummaryItem] = []
        for chunk in chunks:
            for match in _RESTRICTION.finditer(chunk.text):
                sentence = _truncate(match.group(1), 260)
                if len(sentence) < 40:
                    continue
                items.append(SummaryItem(
                    type="restriction",
                    label=chunk.section_label or "Restriction",
                    summary=sentence,
                    source_pages=[chunk.page_start],
                    section_title=chunk.section_title,
                ))
        return items[:_MAX_PER_CATEGORY]

    # ── Organizations ───────────────────────────────────────────────────
    def extract_organizations(self, chunks: List[Chunk]) -> List[SummaryItem]:
        """
        Companies, firms, institutions and regulators named in the document.

        Matched on corporate suffixes and on the names of bodies a contract
        typically references, so an organisation is only reported when the text
        actually names one.
        """
        found: Dict[str, SummaryItem] = {}
        for chunk in chunks:
            text = chunk.text or ""
            for match in _ENTITY.finditer(text):
                name = self._trim_leading_noise(
                    re.sub(r"\s+", " ", match.group(1)).strip(" ,.;")
                )
                if len(name) < 5 or name.lower() in _ROLE_WORDS:
                    continue
                if name not in found:
                    found[name] = SummaryItem(
                        type="organization", label="Organization", value=name,
                        summary=_truncate(_window(text, match.start(), match.end()), 200),
                        source_pages=[chunk.page_start],
                        section_title=chunk.section_title,
                    )
            for match in _INSTITUTION.finditer(text):
                name = re.sub(r"\s+", " ", match.group(0)).strip(" ,.;")
                if name not in found:
                    found[name] = SummaryItem(
                        type="organization", label="Institution", value=name,
                        summary=_truncate(_window(text, match.start(), match.end()), 200),
                        source_pages=[chunk.page_start],
                        section_title=chunk.section_title,
                    )
        return list(found.values())[:_MAX_PER_CATEGORY]

    # ── People ──────────────────────────────────────────────────────────
    def extract_people(self, chunks: List[Chunk]) -> List[SummaryItem]:
        """
        Individuals named as signatories, representatives or notice contacts.

        Only names attached to a signing or contact cue are taken; a bare
        capitalised phrase is far too easy to mistake for a person.
        """
        found: Dict[str, SummaryItem] = {}
        for chunk in chunks:
            text = chunk.text or ""
            for pattern, role in ((_SIGNATORY, "Signatory"),
                                  (_ATTENTION_PERSON, "Contact")):
                for match in pattern.finditer(text):
                    name = re.sub(r"\s+", " ", match.group(1)).strip(" ,.;")
                    if not (4 <= len(name) <= 60) or name.lower() in _ROLE_WORDS:
                        continue
                    title = ""
                    if pattern is _SIGNATORY and match.lastindex and match.lastindex >= 2:
                        title = re.sub(r"\s+", " ", match.group(2) or "").strip(" ,.;")
                    if name not in found:
                        found[name] = SummaryItem(
                            type="person", label=title or role, value=name,
                            summary=_truncate(
                                _window(text, match.start(), match.end()), 200),
                            source_pages=[chunk.page_start],
                            section_title=chunk.section_title,
                        )
        return list(found.values())[:_MAX_PER_CATEGORY]

    # ── Legal references ────────────────────────────────────────────────
    def extract_legal_references(self, chunks: List[Chunk]) -> List[SummaryItem]:
        """Acts, regulations, courts and cited sections."""
        found: Dict[str, SummaryItem] = {}
        for chunk in chunks:
            text = chunk.text or ""
            for pattern, label in ((_STATUTE, "Statute"),
                                   (_REGULATION, "Regulation"),
                                   (_COURT, "Court")):
                for match in pattern.finditer(text):
                    value = re.sub(r"\s+", " ", match.group(0)).strip(" ,.;")
                    key = value.lower()
                    if key in found or len(value) < 4:
                        continue
                    found[key] = SummaryItem(
                        type="legal_reference", label=label, value=value,
                        summary=_truncate(_window(text, match.start(), match.end()), 200),
                        source_pages=[chunk.page_start],
                        section_title=chunk.section_title,
                    )
        return list(found.values())[:_MAX_PER_CATEGORY]

    # ── Jurisdiction ────────────────────────────────────────────────────
    def extract_jurisdiction(self, chunks: List[Chunk]) -> Dict[str, object]:
        governing_law, venue, dispute = "", "", ""
        page, section = 0, ""

        for chunk in chunks:
            text = chunk.text
            if not governing_law:
                match = _GOVERNING_LAW.search(text)
                if match:
                    governing_law = re.sub(r"\s+", " ", match.group(1)).strip(" ,.")
                    page, section = chunk.page_start, chunk.section_label
            if not venue:
                match = _VENUE.search(text)
                if match:
                    venue = re.sub(r"\s+", " ", match.group(1)).strip(" ,.")
                    page = page or chunk.page_start
            if not dispute:
                match = _ARBITRATION.search(text)
                if match:
                    dispute = _truncate(match.group(1), 160)
                    page = page or chunk.page_start

        return {
            "governing_law": governing_law or "Not specified",
            "venue": venue or "Not specified",
            "dispute_resolution": dispute or "Not specified",
            "page_number": page or 1,
            "section": section,
        }

    # ── Document type ───────────────────────────────────────────────────
    _TYPE_HINTS = [
        ("estate planning", "Estate Planning Agreement"),
        ("attorney engagement", "Attorney Engagement Letter"),
        ("engagement letter", "Attorney Engagement Letter"),
        ("retainer agreement", "Attorney Retainer Agreement"),
        ("attorney retainer", "Attorney Retainer Agreement"),
        ("law firm engagement", "Attorney Engagement Letter"),
        ("legal services agreement", "Legal Services Agreement"),
        ("legal representation", "Legal Representation Agreement"),
        ("power of attorney", "Power of Attorney"),
        ("trust agreement", "Trust Agreement"),
        ("living trust", "Living Trust Agreement"),
        ("will and testament", "Will and Testament"),
        ("last will", "Will and Testament"),
        ("advance directive", "Advance Directive"),
        ("master services agreement", "Master Services Agreement"),
        ("non-disclosure agreement", "Non-Disclosure Agreement"),
        ("mutual non-disclosure", "Mutual Non-Disclosure Agreement"),
        ("confidentiality agreement", "Confidentiality Agreement"),
        ("employment agreement", "Employment Agreement"),
        ("consulting agreement", "Consulting Agreement"),
        ("service agreement", "Service Agreement"),
        ("statement of work", "Statement of Work"),
        ("software license", "Software Licence Agreement"),
        ("licence agreement", "Licence Agreement"),
        ("license agreement", "Licence Agreement"),
        ("lease agreement", "Lease Agreement"),
        ("purchase agreement", "Purchase Agreement"),
        ("sublease", "Sublease Agreement"),
        ("subscription agreement", "Subscription Agreement"),
        ("terms of service", "Terms of Service"),
        ("privacy policy", "Privacy Policy"),
        ("partnership agreement", "Partnership Agreement"),
        ("shareholders agreement", "Shareholders Agreement"),
        ("loan agreement", "Loan Agreement"),
        ("settlement agreement", "Settlement Agreement"),
    ]

    def detect_document_type(self, document_text: str, file_name: str = "") -> str:
        """
        Identify the contract type from its own title text.

        The first page is weighted, because that is where a contract names itself;
        the filename is a last resort.
        """
        head = (document_text or "")[:3000].lower()
        for needle, label in self._TYPE_HINTS:
            if needle in head:
                return label
        name = (file_name or "").lower()
        for needle, label in self._TYPE_HINTS:
            if needle.replace(" ", "_") in name or needle.replace(" ", "-") in name:
                return label
        return "Legal Agreement"


legal_extractor = LegalExtractor()
