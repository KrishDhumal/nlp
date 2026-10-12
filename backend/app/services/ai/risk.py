"""
Deterministic legal risk analysis over detected clauses and canonical chunks.

Each finding is produced by a named rule that fires on text actually present in
the document, and carries the page and the sentence that triggered it. No model
decides what is risky, so a finding can always be traced to a rule and a quote.

Two design points worth stating, because they change how the output should be
read:

  * A score of 0 does not mean "safe". It means no rule fired, which for a
    document with no detectable clauses means *insufficient information*. The
    result carries an explicit `assessable` flag and a `risk_level` of
    "Unknown" in that case, rather than implying a clean bill of health.
  * Missing protections are reported as findings in their own right. The absence
    of a liability cap is a risk even though no clause says so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from app.domain.document import Chunk
from app.services.ai.clauses import (
    DetectedClause,
    missing_clause_types,
    _expected_clauses_for_document_type,
)

SEVERITIES = ("Low", "Medium", "High", "Critical")
_SEVERITY_WEIGHT = {"Critical": 25, "High": 15, "Medium": 8, "Low": 3}


@dataclass
class RiskFinding:
    title: str
    severity: str
    reason: str
    recommendation: str
    page: int = 1
    section: str = ""
    source_text: str = ""
    rule_id: str = ""

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "severity": self.severity,
            "reason": self.reason,
            # `description` is what the existing Risks tab renders.
            "description": self.reason,
            "recommendation": self.recommendation,
            "page_number": self.page,
            "page": self.page,
            "section": self.section,
            "snippet": self.source_text,
            "source_text": self.source_text,
            "rule_id": self.rule_id,
        }


@dataclass
class RiskAssessment:
    score: int = 0
    level: str = "Unknown"
    assessable: bool = False
    findings: List[RiskFinding] = field(default_factory=list)
    missing_protections: List[str] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "risk_score": self.score,
            "risk_level": self.level,
            "assessable": self.assessable,
            "risks": [f.to_dict() for f in self.findings],
            "missing_clauses": self.missing_protections,
            "risk_counts": self.counts,
            "risk_note": self.note,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Pattern rules, applied to the raw document text
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RiskRule:
    rule_id: str
    title: str
    severity: str
    pattern: re.Pattern
    reason: str
    recommendation: str
    # An optional pattern that, if also present, cancels the finding.
    mitigated_by: Optional[re.Pattern] = None


RISK_RULES: List[RiskRule] = [
    RiskRule(
        "unlimited-liability", "Liability is not capped", "Critical",
        re.compile(r"unlimited liability|without limitation of liability|"
                   r"no limit(?:ation)? on (?:its )?liability", re.I),
        "The agreement states liability is unlimited, so exposure is not bounded "
        "by the contract value.",
        "Negotiate an aggregate liability cap, typically tied to fees paid.",
    ),
    RiskRule(
        "uncapped-indemnity", "Indemnity sits outside the liability cap", "High",
        re.compile(r"indemnif\w*[^.;]{0,120}not subject to[^.;]{0,60}cap|"
                   r"cap[^.;]{0,60}shall not apply to[^.;]{0,60}indemnif", re.I),
        "Indemnification obligations are excluded from the liability cap, so that "
        "exposure is unbounded even though other liability is limited.",
        "Bring indemnity within the cap, or agree a separate ceiling for it.",
    ),
    RiskRule(
        "auto-renewal", "Automatic renewal", "Medium",
        re.compile(r"automatically renew|renews? automatically|"
                   r"successive[^.;]{0,40}(?:terms?|periods?)", re.I),
        "The agreement renews on its own, so missing the opt-out window extends "
        "the commitment.",
        "Diarise the opt-out deadline, or require affirmative renewal.",
    ),
    RiskRule(
        "unilateral-termination", "The other party may terminate at will", "Medium",
        re.compile(r"terminate[^.;]{0,60}for convenience|"
                   r"terminate[^.;]{0,40}at any time[^.;]{0,40}without cause", re.I),
        "Termination for convenience lets a party exit without breach, which "
        "makes revenue or supply less predictable.",
        "Confirm the notice period is long enough to transition.",
    ),
    RiskRule(
        "suspension-without-liability", "Services may be suspended without liability", "High",
        re.compile(r"suspend[^.;]{0,80}without liability|"
                   r"may suspend[^.;]{0,60}(?:immediately|forthwith)", re.I),
        "One party may suspend performance unilaterally and bears no liability "
        "for the consequences.",
        "Require notice and a cure period before suspension.",
    ),
    RiskRule(
        "long-notice", "Long notice period required", "Medium",
        re.compile(r"\b(?:ninety|120|180|one hundred (?:and )?eighty)\s*\(?\d*\)?\s*days?[^.;]{0,40}notice|"
                   r"\b(?:9[0-9]|1[0-9]{2,})\s*\)?\s*days?[^.;]{0,30}notice", re.I),
        "An unusually long notice period delays exit and extends obligations.",
        "Negotiate the notice period down to 30-60 days.",
    ),
    RiskRule(
        "perpetual-confidentiality", "Confidentiality runs indefinitely", "Medium",
        re.compile(r"confidential\w*[^.;]{0,80}(?:in perpetuity|perpetual|indefinitely|"
                   r"without limit of time)", re.I),
        "Confidentiality obligations never expire, which is difficult to manage "
        "and to prove compliance with.",
        "Agree a fixed survival period, commonly 3-5 years.",
    ),
    RiskRule(
        "broad-indemnity", "Broad indemnity obligation", "High",
        re.compile(r"indemnif\w*[^.;]{0,100}any and all (?:claims|losses|damages)|"
                   r"indemnif\w*[^.;]{0,80}regardless of (?:fault|negligence)", re.I),
        "The indemnity covers any and all claims regardless of fault, which is "
        "wider than the usual third-party-claim scope.",
        "Narrow the indemnity to third-party claims arising from breach.",
    ),
    RiskRule(
        "penalty-interest", "Penalty interest on late payment", "Low",
        re.compile(r"interest[^.;]{0,40}\d+(?:\.\d+)?\s*%[^.;]{0,30}per month|"
                   r"late (?:payment )?(?:charge|fee|penalty)", re.I),
        "Overdue amounts attract interest, which compounds if invoices are "
        "disputed or delayed.",
        "Confirm the rate is lawful and the dispute process pauses interest.",
    ),
    RiskRule(
        "jury-waiver", "Jury trial or class action waived", "Medium",
        re.compile(r"waive[^.;]{0,60}(?:jury trial|right to a jury)|"
                   r"waive[^.;]{0,60}class action|no class", re.I),
        "Procedural rights are waived, limiting the routes available if a dispute "
        "escalates.",
        "Confirm arbitration terms are acceptable before accepting the waiver.",
    ),
    RiskRule(
        "unilateral-amendment", "One party may change the terms unilaterally", "High",
        re.compile(r"may (?:modify|amend|change)[^.;]{0,60}at (?:its|their) "
                   r"(?:sole )?discretion|reserves the right to (?:modify|amend)", re.I),
        "One side can alter the agreement without consent, so the terms you "
        "reviewed may not be the terms you are held to.",
        "Require mutual written agreement for amendments.",
    ),
    RiskRule(
        "assignment-without-consent", "Assignment permitted without consent", "Medium",
        re.compile(r"may assign[^.;]{0,60}without[^.;]{0,30}consent", re.I),
        "The counterparty can transfer the contract to a third party you have not "
        "assessed.",
        "Require prior written consent for assignment, with a carve-out for "
        "group reorganisations.",
    ),
    RiskRule(
        "data-transfer", "Personal data may move across borders", "Medium",
        re.compile(r"transfer[^.;]{0,60}personal data[^.;]{0,60}outside|"
                   r"cross-border[^.;]{0,40}transfer", re.I),
        "Personal data crosses jurisdictions, which triggers additional "
        "data-protection obligations.",
        "Confirm an adequate transfer mechanism is in place.",
    ),
    RiskRule(
        "as-is-no-warranty", "All warranties disclaimed", "Medium",
        re.compile(r'"?as is"?[^.;]{0,60}without (?:any )?warrant|'
                   r"disclaims? all warranties", re.I),
        "The deliverable is provided with no quality assurances, so defects are "
        "your cost to bear.",
        "Negotiate at least a limited warranty for conformity to specification.",
    ),
]

# A missing protection is itself a finding.
_MISSING_SEVERITY = {
    "Liability": "High",
    "Indemnification": "High",
    "Governing Law": "Medium",
    "Dispute Resolution": "Medium",
    "Confidentiality": "Medium",
    "Termination": "High",
    "Payment": "Medium",
    "Intellectual Property": "Medium",
    "Force Majeure": "Low",
    "Notice": "Low",
    "Assignment": "Low",
}


def _evidence(text: str, match: re.Match, limit: int = 260) -> str:
    """The sentence around a rule match, as the quoted evidence."""
    start = max(text.rfind(". ", 0, match.start()), text.rfind("\n", 0, match.start()))
    start = 0 if start == -1 else start + 1
    end = text.find(". ", match.end())
    end = len(text) if end == -1 else end + 1
    sentence = re.sub(r"\s+", " ", text[start:end]).strip()
    return sentence[:limit] + ("..." if len(sentence) > limit else "")


def analyse_risk(
    chunks: Sequence[Chunk],
    clauses: Optional[Sequence[DetectedClause]] = None,
    document_type: str = "",
) -> RiskAssessment:
    """
    Score the document and list every risk finding with its evidence.

    The score starts at 100 and is reduced by severity-weighted findings. It is
    only meaningful when something was actually assessable — see `assessable`.

    When `document_type` is provided the missing-clause check uses the
    expectation set appropriate for that kind of agreement, so estate-planning
    or attorney-engagement documents are not penalised for missing commercial
    clauses like IP, Force Majeure, or Indemnification.
    """
    clauses = list(clauses or [])
    findings: List[RiskFinding] = []
    seen_rules: set = set()

    for chunk in chunks:
        text = chunk.text or ""
        for rule in RISK_RULES:
            if rule.rule_id in seen_rules:
                continue
            match = rule.pattern.search(text)
            if not match:
                continue
            if rule.mitigated_by and rule.mitigated_by.search(text):
                continue
            seen_rules.add(rule.rule_id)
            findings.append(RiskFinding(
                title=rule.title,
                severity=rule.severity,
                reason=rule.reason,
                recommendation=rule.recommendation,
                page=chunk.page_start,
                section=f"{chunk.section_number} {chunk.section_title}".strip(),
                source_text=_evidence(text, match),
                rule_id=rule.rule_id,
            ))

    missing = missing_clause_types(clauses, document_type=document_type) if clauses else []
    expected = set(_expected_clauses_for_document_type(document_type))
    for name in missing:
        # Only flag clauses that are actually expected for this document type.
        # Clauses outside the expected set are marked N/A and not penalised.
        if name not in expected:
            continue
        findings.append(RiskFinding(
            title=f"No {name.lower()} clause found",
            severity=_MISSING_SEVERITY.get(name, "Medium"),
            reason=f"The document does not appear to contain a {name.lower()} "
                   "clause, so the protection it normally provides is absent.",
            recommendation=f"Add a {name.lower()} clause.",
            page=0,
            section="",
            source_text="",
            rule_id=f"missing-{name.lower().replace(' ', '-')}",
        ))

    counts = {severity: 0 for severity in SEVERITIES}
    deduction = 0
    for finding in findings:
        counts[finding.severity] = counts.get(finding.severity, 0) + 1
        deduction += _SEVERITY_WEIGHT.get(finding.severity, 8)

    # "Assessable" means we actually had something to judge. Without clauses and
    # without rule hits, a score of 100 would be a false reassurance.
    assessable = bool(clauses) or bool(findings)
    if not assessable:
        return RiskAssessment(
            score=0, level="Unknown", assessable=False, findings=[],
            missing_protections=[], counts=counts,
            note="Not enough recognisable contract language was found to assess "
                 "risk. This is not a statement that the document is low risk.",
        )

    score = max(0, min(100, 100 - deduction))
    if counts.get("Critical"):
        level = "Critical"
    elif score >= 80:
        level = "Low"
    elif score >= 55:
        level = "Medium"
    elif score >= 30:
        level = "High"
    else:
        level = "Critical"

    severity_order = {s: i for i, s in enumerate(reversed(SEVERITIES))}
    findings.sort(key=lambda f: (severity_order.get(f.severity, 9), f.page))

    return RiskAssessment(
        score=score, level=level, assessable=True, findings=findings,
        missing_protections=missing, counts=counts,
        note="",
    )
