"""
Deterministic clause detection over canonical chunks.

Every clause returned here is evidenced by text that actually appears in the
document: the detector matches drafting patterns, and the `source_text` it
reports is a literal span of the chunk it came from. Nothing is generated, so
nothing can be invented — the failure mode is a missed clause, never a fabricated
one.

This is the shared foundation for risk analysis, comparison and compliance, all
of which read the same `DetectedClause` objects rather than re-parsing the PDF
or re-prompting a model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from app.domain.document import Chunk

# ─────────────────────────────────────────────────────────────────────────────
# Clause taxonomy
# ─────────────────────────────────────────────────────────────────────────────
# Each clause type carries the phrases that identify it. `strong` phrases are
# near-conclusive on their own; `weak` phrases only count in support, so a
# passing mention of "terminate" inside a payment clause does not register as a
# termination clause.
@dataclass(frozen=True)
class ClauseSignature:
    clause_type: str
    description: str
    strong: Sequence[str]
    weak: Sequence[str] = ()
    heading_hints: Sequence[str] = ()


CLAUSE_SIGNATURES: List[ClauseSignature] = [
    ClauseSignature(
        "Confidentiality", "Obligations to keep information secret",
        strong=("confidential information", "strict confidence", "keep confidential",
                "proprietary information", "conflict of interest",
                "duty of confidentiality", "attorney-client privilege",
                "privileged communication"),
        weak=("confidential", "disclose", "privilege"),
        heading_hints=("confidentiality", "confidential", "conflict of interest",
                       "conflicts of interest", "attorney-client privilege"),
    ),
    ClauseSignature(
        "Non-Disclosure", "Restriction on disclosing information to third parties",
        strong=("non-disclosure", "nondisclosure", "shall not disclose",
                "may not disclose", "agrees not to disclose"),
        weak=("disclosure",),
        heading_hints=("non-disclosure", "nondisclosure"),
    ),
    ClauseSignature(
        "Termination", "How and when the agreement can be ended",
        strong=("terminate this agreement", "termination for convenience",
                "right to terminate", "may terminate", "terminated upon",
                "discharge of", "withdrawal of", "right to withdraw",
                "withdraws from representation", "discharged at any time",
                "termination of representation", "cease to represent",
                "end of engagement", "conclude the representation"),
        weak=("terminate", "termination", "discharge", "withdrawal", "withdraw"),
        heading_hints=("termination", "term and termination", "discharge",
                       "withdrawal", "discharge of attorney",
                       "discharge of law firm", "termination of engagement"),
    ),
    ClauseSignature(
        "Payment", "Fees, invoicing and payment timing",
        strong=("shall pay", "payable within", "invoice", "payment terms",
                "monthly retainer", "net thirty", "net 30",
                "initial deposit", "advance deposit", "retainer fee",
                "hourly rate", "flat fee", "contingency fee",
                "statements and payments", "billing statement",
                "legal fees", "attorney fees", "costs and expenses"),
        weak=("fee", "payment", "amount due", "deposit", "retainer",
              "billing", "hourly", "charges"),
        heading_hints=("payment", "fees", "compensation", "consideration",
                       "fees and deposit", "statements and payments",
                       "billing", "costs and fees", "attorney fees",
                       "fee arrangement", "retainer"),
    ),
    ClauseSignature(
        "Intellectual Property", "Ownership of IP and deliverables",
        strong=("intellectual property", "work product", "assigns to",
                "all right, title and interest", "copyright", "patent"),
        weak=("ownership", "licence", "license"),
        heading_hints=("intellectual property", "ip rights", "ownership"),
    ),
    ClauseSignature(
        "Liability", "Limits on what each party can be held liable for",
        strong=("limitation of liability", "aggregate liability", "shall not exceed",
                "liable for any indirect", "consequential damages"),
        weak=("liability", "liable", "damages"),
        heading_hints=("limitation of liability", "liability"),
    ),
    ClauseSignature(
        "Indemnification", "One party covering the other's losses",
        strong=("indemnify", "hold harmless", "defend and indemnify",
                "indemnification"),
        weak=("defend", "third-party claim"),
        heading_hints=("indemnification", "indemnity"),
    ),
    ClauseSignature(
        "Non-Compete", "Restriction on competing activity",
        strong=("non-compete", "noncompete", "shall not compete",
                "engage in any competing", "competing business"),
        weak=("compete",),
        heading_hints=("non-compete", "noncompetition"),
    ),
    ClauseSignature(
        "Non-Solicitation", "Restriction on poaching staff or customers",
        strong=("non-solicitation", "nonsolicitation", "shall not solicit",
                "not solicit any employee", "shall not induce"),
        weak=("solicit",),
        heading_hints=("non-solicitation", "nonsolicitation"),
    ),
    ClauseSignature(
        "Governing Law", "Which jurisdiction's law applies",
        strong=("governed by and construed", "governing law", "laws of the state",
                "laws of the republic", "conflict of laws"),
        weak=("jurisdiction",),
        heading_hints=("governing law", "applicable law"),
    ),
    ClauseSignature(
        "Dispute Resolution", "How disputes are settled",
        strong=("binding arbitration", "dispute resolution", "arbitration",
                "submit to the exclusive jurisdiction", "mediation"),
        weak=("dispute", "venue", "courts"),
        heading_hints=("dispute resolution", "arbitration", "disputes"),
    ),
    ClauseSignature(
        "Force Majeure", "Excused performance for events beyond control",
        strong=("force majeure", "acts of god", "beyond the reasonable control"),
        weak=("unforeseen",),
        heading_hints=("force majeure",),
    ),
    ClauseSignature(
        "Data Protection", "Handling of personal or regulated data",
        strong=("personal data", "data protection", "gdpr", "data processing",
                "data subject", "protected health information", "personally identifiable"),
        weak=("privacy", "data"),
        heading_hints=("data protection", "privacy", "data processing"),
    ),
    ClauseSignature(
        "Warranty", "Promises about quality or condition",
        strong=("warrants that", "represents and warrants", "warranty",
                "disclaims all warranties", "as is"),
        weak=("representation",),
        heading_hints=("warranty", "warranties", "representations"),
    ),
    ClauseSignature(
        "Audit Rights", "Right to inspect records or systems",
        strong=("right to audit", "audit rights", "inspect the records",
                "upon reasonable notice, audit"),
        weak=("audit", "inspection"),
        heading_hints=("audit", "audit rights", "records"),
    ),
    ClauseSignature(
        "Assignment", "Whether the contract can be transferred",
        strong=("may not assign", "shall not assign", "assignment of this agreement",
                "neither party may assign", "no party may assign",
                "without the prior written consent", "successors and assigns"),
        weak=("assign",),
        heading_hints=("assignment",),
    ),
    ClauseSignature(
        "Notice", "How formal notices must be delivered",
        strong=("written notice", "notices under this agreement", "shall be deemed given",
                "prior written notice"),
        weak=("notice",),
        heading_hints=("notice", "notices"),
    ),
    ClauseSignature(
        "Survival", "Which obligations continue after the contract ends",
        strong=("shall survive", "survive termination", "survival of"),
        weak=("survive",),
        heading_hints=("survival",),
    ),
]

CLAUSE_TYPES: List[str] = [sig.clause_type for sig in CLAUSE_SIGNATURES]

# Scoring. A strong phrase or a matching heading is enough on its own; weak
# phrases only accumulate.
_STRONG_WEIGHT = 4
_HEADING_WEIGHT = 5
_WEAK_WEIGHT = 1
_MIN_SCORE = 4


@dataclass
class DetectedClause:
    """One clause found in the document, with the evidence for it."""

    clause_type: str
    description: str
    page: int
    page_end: int
    section: str
    section_number: str
    source_text: str
    chunk_id: str
    risk_level: str = "Low"
    risk_reason: str = ""
    recommendation: str = ""
    confidence: float = 0.0
    matched_terms: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "clause_type": self.clause_type,
            "description": self.description,
            "page": self.page,
            "page_end": self.page_end,
            "section": self.section,
            "section_number": self.section_number,
            "source_text": self.source_text,
            "chunk_id": self.chunk_id,
            "risk_level": self.risk_level,
            "risk_reason": self.risk_reason,
            "recommendation": self.recommendation,
            "confidence": round(self.confidence, 2),
            "matched_terms": self.matched_terms[:6],
        }

    def to_ui_clause(self, index: int) -> dict:
        """The shape the existing Clauses tab renders."""
        return {
            "id": f"c{index}",
            "category": UI_CATEGORY.get(self.clause_type, self.clause_type),
            "title": self.clause_type,
            "page_number": self.page,
            "section": self.section_number or self.section,
            "risk_level": self.risk_level,
            "snippet": self.source_text,
            "explanation": self.description,
            "recommendation": self.recommendation,
            # The exact phrases that triggered detection, so the UI can mark the
            # operative language inside the quote rather than just showing it.
            "matched_terms": self.matched_terms[:6],
            "confidence": round(self.confidence, 2),
            "page_end": self.page_end,
        }


# The audit UI filters on a smaller set of chips; map the full taxonomy onto it
# so every detected clause is still reachable from the existing interface.
UI_CATEGORY: Dict[str, str] = {
    "Confidentiality": "Confidentiality",
    "Non-Disclosure": "Confidentiality",
    "Data Protection": "Confidentiality",
    "Termination": "Termination",
    "Survival": "Termination",
    "Payment": "Payment",
    "Liability": "Liability",
    "Warranty": "Liability",
    "Indemnification": "Indemnification",
    "Governing Law": "Governing Law",
    "Dispute Resolution": "Governing Law",
    "Intellectual Property": "IP Rights",
}


def _sentence_containing(text: str, term: str, limit: int = 320) -> str:
    """
    The sentence in `text` that contains `term`, as evidence.

    Returning the matched sentence rather than the whole chunk keeps the quoted
    evidence tight enough to read in the UI while staying verbatim.
    """
    lowered = text.lower()
    position = lowered.find(term.lower())
    if position == -1:
        return re.sub(r"\s+", " ", text[:limit]).strip()

    start = max(
        text.rfind(". ", 0, position),
        text.rfind("\n", 0, position),
    )
    start = 0 if start == -1 else start + 1
    end = len(text)
    for terminator in (". ", ".\n", "; "):
        found = text.find(terminator, position)
        if found != -1:
            end = min(end, found + 1)
    sentence = re.sub(r"\s+", " ", text[start:end]).strip()
    if len(sentence) > limit:
        sentence = sentence[:limit].rstrip() + "..."
    return sentence or re.sub(r"\s+", " ", text[:limit]).strip()


def detect_clauses(chunks: Sequence[Chunk]) -> List[DetectedClause]:
    """
    Find every clause type present in the document.

    One clause per type is returned — the best-evidenced occurrence — because the
    consumers (risk, comparison, compliance) ask "does this contract have a
    termination clause and what does it say", not "list every mention".
    """
    best: Dict[str, tuple] = {}

    for chunk in chunks:
        text = chunk.text or ""
        lowered = text.lower()
        heading = f"{chunk.section_number} {chunk.section_title}".lower()

        for signature in CLAUSE_SIGNATURES:
            score = 0
            matched: List[str] = []

            for phrase in signature.heading_hints:
                if phrase in heading:
                    score += _HEADING_WEIGHT
                    matched.append(phrase)
                    break

            for phrase in signature.strong:
                if phrase in lowered:
                    score += _STRONG_WEIGHT
                    matched.append(phrase)

            for phrase in signature.weak:
                if phrase in lowered:
                    score += _WEAK_WEIGHT
                    matched.append(phrase)

            if score < _MIN_SCORE:
                continue

            existing = best.get(signature.clause_type)
            if existing and existing[0] >= score:
                continue

            evidence_term = matched[0] if matched else signature.strong[0]
            best[signature.clause_type] = (
                score,
                DetectedClause(
                    clause_type=signature.clause_type,
                    description=signature.description,
                    page=chunk.page_start,
                    page_end=chunk.page_end,
                    section=chunk.section_title or "",
                    section_number=chunk.section_number or "",
                    source_text=_sentence_containing(text, evidence_term),
                    chunk_id=chunk.chunk_id,
                    confidence=min(1.0, score / 10),
                    matched_terms=sorted(set(matched)),
                ),
            )

    clauses = [entry[1] for entry in best.values()]
    clauses.sort(key=lambda c: (c.page, c.section_number or "", c.clause_type))
    return clauses


def clause_types_present(clauses: Sequence[DetectedClause]) -> set:
    return {clause.clause_type for clause in clauses}


# ─────────────────────────────────────────────────────────────────────────────
# Document-type families: which clause expectations apply to which kinds of
# document.  Non-commercial agreements (estate planning, attorney engagement)
# should not be penalised for missing IP, Force Majeure, etc.
# ─────────────────────────────────────────────────────────────────────────────
_COMMERCIAL_EXPECTED = (
    "Confidentiality", "Termination", "Payment", "Liability",
    "Indemnification", "Governing Law", "Dispute Resolution",
    "Intellectual Property", "Force Majeure", "Notice", "Assignment",
)

_LEGAL_SERVICES_EXPECTED = (
    "Confidentiality", "Termination", "Payment",
    "Governing Law", "Dispute Resolution", "Notice",
)

_ESTATE_PLANNING_EXPECTED = (
    "Confidentiality", "Termination", "Payment",
    "Governing Law", "Notice",
)

# Patterns that identify a document as belonging to a non-commercial family.
_LEGAL_SERVICES_TYPES = {
    "attorney engagement", "engagement letter", "legal services agreement",
    "attorney retainer", "retainer agreement", "law firm engagement",
    "legal representation", "attorney agreement", "legal agreement",
}

_ESTATE_PLANNING_TYPES = {
    "estate planning", "will and testament", "trust agreement",
    "power of attorney", "living trust", "estate plan",
    "testamentary trust", "advance directive",
}


def _expected_clauses_for_document_type(document_type: str) -> tuple:
    """Return the set of clause types one would *expect* for this document family."""
    dt = (document_type or "").lower()
    for pattern in _ESTATE_PLANNING_TYPES:
        if pattern in dt:
            return _ESTATE_PLANNING_EXPECTED
    for pattern in _LEGAL_SERVICES_TYPES:
        if pattern in dt:
            return _LEGAL_SERVICES_EXPECTED
    return _COMMERCIAL_EXPECTED


def missing_clause_types(clauses: Sequence[DetectedClause],
                         expected: Optional[Sequence[str]] = None,
                         document_type: str = "") -> List[str]:
    """
    Standard protections the document does not contain.

    When `document_type` is provided, the expected set is scoped to what that
    kind of document would normally carry.  Estate planning and attorney
    engagement letters, for example, are not penalised for missing IP, Force
    Majeure, or commercial Indemnification.
    """
    if expected is None:
        expected = _expected_clauses_for_document_type(document_type)
    present = clause_types_present(clauses)
    return [name for name in expected if name not in present]
