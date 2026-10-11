"""
ComplianceService — rule-based regulatory checks over canonical chunks.

Each framework is a list of requirements, and each requirement is a set of
phrases that would evidence it in a contract. A requirement is only marked PASS
when matching language is actually found, and the text that satisfied it is
returned with its page, so every PASS can be checked by a human in seconds.

This is intentionally deterministic rather than model-driven. A model asked
"is this GDPR compliant?" will produce a confident answer either way; a rule can
only say "I found this sentence on page 4, or I did not".

Scope statement, which the UI repeats: this identifies whether a document
*contains* language addressing a requirement. It is not a legal opinion and not
a certification. Absence of a finding is not assurance of compliance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from app.core.logging import logger
from app.domain.document import Chunk

STATUSES = ("PASS", "PARTIAL", "FAIL", "NOT_FOUND")
SEVERITIES = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

SUPPORTED_FRAMEWORKS = ("GDPR", "HIPAA", "SOC2")


@dataclass(frozen=True)
class Requirement:
    requirement_id: str
    title: str
    description: str
    severity: str
    # Any of these evidences the requirement.
    evidence: Sequence[str]
    # All-of groups: each inner tuple must have at least one hit for a full PASS.
    # Used where a requirement has several distinct parts.
    supporting: Sequence[Sequence[str]] = ()
    # Language that actively contradicts the requirement.
    contradicts: Sequence[str] = ()
    recommendation: str = ""


FRAMEWORKS: Dict[str, List[Requirement]] = {
    "GDPR": [
        Requirement(
            "gdpr-lawful-basis", "Lawful basis for processing",
            "The agreement identifies why personal data may be processed.",
            "HIGH",
            evidence=("lawful basis", "legitimate interest", "consent of the data subject",
                      "processing is necessary", "article 6"),
            recommendation="State the lawful basis relied on for each processing purpose.",
        ),
        Requirement(
            "gdpr-purpose-limitation", "Purpose limitation",
            "Processing is restricted to documented, specified purposes.",
            "MEDIUM",
            evidence=("only for the purposes", "documented instructions",
                      "specified purpose", "purpose limitation"),
            recommendation="Limit processing to the controller's documented instructions.",
        ),
        Requirement(
            "gdpr-subprocessor", "Sub-processor controls",
            "Engaging another processor requires authorisation and flow-down terms.",
            "HIGH",
            evidence=("sub-processor", "subprocessor", "sub processor",
                      "prior written authorisation", "prior written authorization"),
            recommendation="Require prior authorisation and equivalent terms for sub-processors.",
        ),
        Requirement(
            "gdpr-breach-notification", "Personal data breach notification",
            "A breach must be reported, within a stated timeframe.",
            "CRITICAL",
            evidence=("data breach", "personal data breach", "security incident",
                      "notify without undue delay", "breach notification"),
            supporting=(("without undue delay", "72 hours", "immediately", "promptly"),),
            recommendation="Require notification without undue delay, ideally within 72 hours.",
        ),
        Requirement(
            "gdpr-data-subject-rights", "Data subject rights",
            "The processor assists with access, erasure and portability requests.",
            "HIGH",
            evidence=("data subject rights", "right to erasure", "right to be forgotten",
                      "data portability", "right of access", "rectification"),
            recommendation="Add an obligation to assist with data subject requests.",
        ),
        Requirement(
            "gdpr-international-transfer", "International transfer safeguards",
            "Transfers outside the EEA rely on a recognised mechanism.",
            "HIGH",
            evidence=("standard contractual clauses", "adequacy decision",
                      "cross-border transfer", "transfer outside the eea",
                      "international transfer"),
            recommendation="Reference standard contractual clauses or another valid mechanism.",
        ),
        Requirement(
            "gdpr-security-measures", "Technical and organisational measures",
            "Appropriate security measures are required.",
            "HIGH",
            evidence=("technical and organisational measures", "appropriate security",
                      "encryption", "pseudonymisation", "security measures"),
            recommendation="Specify the technical and organisational measures required.",
        ),
        Requirement(
            "gdpr-deletion-return", "Deletion or return on termination",
            "Personal data is deleted or returned when the contract ends.",
            "MEDIUM",
            evidence=("delete or return", "return or destroy", "deletion of personal data",
                      "destroy all personal data"),
            recommendation="Require deletion or return of personal data on termination.",
        ),
        Requirement(
            "gdpr-audit", "Audit and inspection rights",
            "The controller can verify the processor's compliance.",
            "MEDIUM",
            evidence=("right to audit", "audit rights", "make available all information",
                      "inspection", "demonstrate compliance"),
            recommendation="Grant audit or evidence rights to the controller.",
        ),
    ],
    "HIPAA": [
        Requirement(
            "hipaa-baa", "Business Associate Agreement",
            "The arrangement is governed by a BAA.",
            "CRITICAL",
            evidence=("business associate", "business associate agreement", "baa",
                      "covered entity"),
            recommendation="Execute a Business Associate Agreement before any PHI is shared.",
        ),
        Requirement(
            "hipaa-phi-definition", "Protected Health Information defined",
            "PHI is identified as a distinct category of data.",
            "HIGH",
            evidence=("protected health information", "phi", "individually identifiable "
                      "health information", "electronic protected health information"),
            recommendation="Define PHI and distinguish it from other confidential data.",
        ),
        Requirement(
            "hipaa-permitted-uses", "Permitted uses and disclosures",
            "Uses of PHI are limited to those the contract permits.",
            "HIGH",
            evidence=("permitted use", "use or disclose", "minimum necessary",
                      "only as permitted"),
            recommendation="Restrict uses of PHI to those required to deliver the service.",
        ),
        Requirement(
            "hipaa-safeguards", "Administrative, physical and technical safeguards",
            "Safeguards protecting PHI are required.",
            "CRITICAL",
            evidence=("safeguards", "administrative, physical", "technical safeguards",
                      "security rule", "encryption"),
            recommendation="Require safeguards consistent with the HIPAA Security Rule.",
        ),
        Requirement(
            "hipaa-breach-notification", "Breach notification",
            "Breaches of unsecured PHI must be reported.",
            "CRITICAL",
            evidence=("breach notification", "unsecured protected health information",
                      "report any breach", "security incident"),
            supporting=(("without unreasonable delay", "60 days", "promptly",
                         "immediately", "without undue delay"),),
            recommendation="Require breach reporting without unreasonable delay.",
        ),
        Requirement(
            "hipaa-subcontractor", "Subcontractor flow-down",
            "Subcontractors handling PHI accept the same obligations.",
            "HIGH",
            evidence=("subcontractor", "sub-contractor", "agents and subcontractors",
                      "same restrictions and conditions"),
            recommendation="Flow HIPAA obligations down to every subcontractor.",
        ),
        Requirement(
            "hipaa-individual-access", "Individual access and amendment",
            "Individuals can access and amend their PHI.",
            "MEDIUM",
            evidence=("right of access", "access to phi", "amendment of phi",
                      "accounting of disclosures", "designated record set"),
            recommendation="Add obligations to support access, amendment and accounting.",
        ),
        Requirement(
            "hipaa-return-destruction", "Return or destruction of PHI",
            "PHI is returned or destroyed at termination.",
            "HIGH",
            evidence=("return or destroy", "destruction of phi", "return all protected health",
                      "destroy all protected health"),
            recommendation="Require return or destruction of PHI on termination.",
        ),
    ],
    "SOC2": [
        Requirement(
            "soc2-security-policy", "Information security programme",
            "A documented security programme is maintained.",
            "HIGH",
            evidence=("information security program", "security policy",
                      "information security policies", "security programme"),
            recommendation="Require a documented information security programme.",
        ),
        Requirement(
            "soc2-access-control", "Logical access controls",
            "Access to systems and data is restricted and reviewed.",
            "HIGH",
            evidence=("access control", "least privilege", "role-based access",
                      "authentication", "multi-factor"),
            recommendation="Require least-privilege access control and periodic review.",
        ),
        Requirement(
            "soc2-encryption", "Encryption of data",
            "Data is encrypted in transit and at rest.",
            "CRITICAL",
            evidence=("encryption", "encrypted at rest", "encrypted in transit", "tls", "aes"),
            supporting=(("at rest", "in transit"),),
            recommendation="Require encryption both in transit and at rest.",
        ),
        Requirement(
            "soc2-incident-response", "Incident response",
            "Security incidents are detected, handled and reported.",
            "CRITICAL",
            evidence=("incident response", "security incident", "incident management",
                      "notify of any security"),
            recommendation="Require a documented incident response and notification process.",
        ),
        Requirement(
            "soc2-availability", "Availability commitments",
            "Uptime or continuity obligations are stated.",
            "MEDIUM",
            evidence=("service level", "uptime", "availability", "business continuity",
                      "disaster recovery", "rto", "rpo"),
            recommendation="State availability targets and continuity obligations.",
        ),
        Requirement(
            "soc2-vendor-management", "Vendor and subservice management",
            "Third parties are assessed and monitored.",
            "MEDIUM",
            evidence=("subcontractor", "third-party vendor", "vendor management",
                      "subservice organization", "due diligence"),
            recommendation="Require assessment and monitoring of subservice providers.",
        ),
        Requirement(
            "soc2-audit-reporting", "Audit and reporting",
            "An independent report or audit right is provided.",
            "HIGH",
            evidence=("soc 2", "soc2", "independent audit", "audit report",
                      "right to audit", "third-party audit"),
            recommendation="Require an annual SOC 2 report or an equivalent audit right.",
        ),
        Requirement(
            "soc2-change-management", "Change management",
            "Changes to systems follow a controlled process.",
            "LOW",
            evidence=("change management", "change control", "release management"),
            recommendation="Require a documented change-management process.",
        ),
    ],
}


@dataclass
class RequirementResult:
    requirement_id: str
    requirement: str
    description: str
    status: str
    severity: str
    evidence: str = ""
    page: Optional[int] = None
    section: str = ""
    recommendation: str = ""
    matched_terms: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "requirement_id": self.requirement_id,
            "requirement": self.requirement,
            "description": self.description,
            "status": self.status,
            "severity": self.severity,
            "evidence": self.evidence,
            "page": self.page,
            "section": self.section,
            "recommendation": self.recommendation,
            "matched_terms": self.matched_terms[:5],
        }


@dataclass
class FrameworkResult:
    framework: str
    score: int
    results: List[RequirementResult] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "framework": self.framework,
            "score": self.score,
            "counts": self.counts,
            "requirements": [r.to_dict() for r in self.results],
        }


DISCLAIMER = (
    "This check identifies whether the document contains language addressing "
    "each requirement. It is not a legal opinion and not a certification of "
    "compliance; absence of a finding does not mean the obligation is unmet "
    "elsewhere."
)


def _sentence_for(text: str, term: str, limit: int = 260) -> str:
    lowered = text.lower()
    position = lowered.find(term.lower())
    if position == -1:
        return ""
    start = max(text.rfind(". ", 0, position), text.rfind("\n", 0, position))
    start = 0 if start == -1 else start + 1
    end = text.find(". ", position)
    end = len(text) if end == -1 else end + 1
    sentence = re.sub(r"\s+", " ", text[start:end]).strip()
    return sentence[:limit] + ("..." if len(sentence) > limit else "")


class ComplianceService:
    """Deterministic requirement checks against the canonical document."""

    def check(
        self,
        chunks: Sequence[Chunk],
        frameworks: Optional[Sequence[str]] = None,
    ) -> dict:
        selected = [f.upper().replace(" ", "").replace("-", "")
                    for f in (frameworks or SUPPORTED_FRAMEWORKS)]
        selected = [f for f in selected if f in FRAMEWORKS] or list(SUPPORTED_FRAMEWORKS)

        framework_results: List[FrameworkResult] = []
        for name in selected:
            framework_results.append(self._check_framework(name, chunks))

        overall = (
            round(sum(f.score for f in framework_results) / len(framework_results))
            if framework_results else 0
        )
        all_results = [r for f in framework_results for r in f.results]
        gaps = [r for r in all_results if r.status in ("FAIL", "NOT_FOUND")]
        severity_order = {s: i for i, s in enumerate(reversed(SEVERITIES))}
        gaps.sort(key=lambda r: severity_order.get(r.severity, 9))

        logger.info(
            f"[Compliance] frameworks={selected} overall={overall} "
            f"gaps={len(gaps)}/{len(all_results)}"
        )
        return {
            "compliance_score": overall,
            "overall_score": overall,
            "frameworks": [f.to_dict() for f in framework_results],
            "summary": self._summarise(framework_results, overall),
            "violations": [
                {
                    "regulation": r.requirement_id.split("-")[0].upper(),
                    "requirement": r.requirement,
                    "status": r.status,
                    "severity": r.severity,
                    "issue": r.description,
                    "evidence": r.evidence,
                    "page": r.page,
                    "suggestion": r.recommendation,
                }
                for r in gaps
            ],
            "action_items": [r.recommendation for r in gaps[:10] if r.recommendation],
            "disclaimer": DISCLAIMER,
            "assessable": bool(chunks),
        }

    def _check_framework(self, name: str, chunks: Sequence[Chunk]) -> FrameworkResult:
        requirements = FRAMEWORKS[name]
        results: List[RequirementResult] = []

        for requirement in requirements:
            results.append(self._check_requirement(requirement, chunks))

        counts = {status: sum(1 for r in results if r.status == status)
                  for status in STATUSES}
        # PASS counts fully, PARTIAL counts half. FAIL and NOT_FOUND count zero.
        earned = counts["PASS"] + 0.5 * counts["PARTIAL"]
        score = round(100 * earned / len(results)) if results else 0
        return FrameworkResult(framework=name, score=score, results=results, counts=counts)

    @staticmethod
    def _check_requirement(requirement: Requirement,
                           chunks: Sequence[Chunk]) -> RequirementResult:
        matched_terms: List[str] = []
        evidence = ""
        page: Optional[int] = None
        section = ""
        contradicted = False

        for chunk in chunks:
            text = chunk.text or ""
            lowered = text.lower()

            for term in requirement.contradicts:
                if term in lowered:
                    contradicted = True

            for term in requirement.evidence:
                if term in lowered:
                    matched_terms.append(term)
                    if not evidence:
                        evidence = _sentence_for(text, term)
                        page = chunk.page_start
                        section = f"{chunk.section_number} {chunk.section_title}".strip()

        if not matched_terms:
            status = "NOT_FOUND"
        elif contradicted:
            status = "FAIL"
        else:
            # Supporting groups turn a bare mention into a full PASS.
            status = "PASS"
            if requirement.supporting:
                joined = " ".join((c.text or "").lower() for c in chunks)
                for group in requirement.supporting:
                    if not any(term in joined for term in group):
                        status = "PARTIAL"
                        break

        return RequirementResult(
            requirement_id=requirement.requirement_id,
            requirement=requirement.title,
            description=requirement.description,
            status=status,
            severity=requirement.severity,
            evidence=evidence,
            page=page,
            section=section,
            recommendation=requirement.recommendation if status != "PASS" else "",
            matched_terms=sorted(set(matched_terms)),
        )

    @staticmethod
    def _summarise(framework_results: Sequence[FrameworkResult], overall: int) -> str:
        parts = [f"Overall coverage score {overall}/100 across "
                 f"{len(framework_results)} framework(s)."]
        for framework in framework_results:
            counts = framework.counts
            parts.append(
                f"{framework.framework}: {counts['PASS']} met, "
                f"{counts['PARTIAL']} partial, "
                f"{counts['FAIL'] + counts['NOT_FOUND']} not evidenced "
                f"({framework.score}/100)."
            )
        return " ".join(parts)


compliance_service = ComplianceService()
