"""
Tests for the deterministic legal analysis layer: clause detection, risk
analysis, version comparison and regulatory compliance.

None of these need a model, an API key, a database or the network. Every
assertion is about behaviour that must hold for a legal tool: findings are
evidenced by real text, nothing is invented, and absence of evidence is
reported as absence rather than as a pass.

Run with:  PYTHONPATH=. ./venv/bin/python tests/test_legal_analysis.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

from typing import List

from app.domain.document import Chunk
from app.services.ai.chunking import TokenAwareChunker, heuristic_token_count
from app.services.ai.clauses import (
    CLAUSE_TYPES,
    detect_clauses,
    missing_clause_types,
)
from app.services.ai.compare_service import compare_service
from app.services.ai.compliance_service import (
    FRAMEWORKS,
    STATUSES,
    SUPPORTED_FRAMEWORKS,
    compliance_service,
)
from app.services.ai.extraction import legal_extractor
from app.services.ai.processor import DocumentProcessor
from app.services.ai.risk import analyse_risk
from tests.fixtures import write_sample_contract

PASSED: List[str] = []
FAILED: List[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(label)
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))


def make_chunk(text: str, page: int = 1, number: str = "1",
               title: str = "Section", order: int = 0,
               document_id: str = "doc", chunk_id: str = "") -> Chunk:
    return Chunk(
        chunk_id=chunk_id or f"chk-{order}-{number}",
        document_id=document_id, user_id="u", text=text,
        token_count=heuristic_token_count(text),
        page_start=page, page_end=page,
        section_id=f"sec-{number}", section_number=number,
        section_title=title, order=order,
    )


def canonical_chunks(pdf_path: str) -> List[Chunk]:
    processor = DocumentProcessor(
        chunker=TokenAwareChunker(count_tokens=heuristic_token_count)
    )
    return processor.process(pdf_path, "doc", "u", "contract.pdf").chunks


# ═══════════════════════════════════════════════════════════════════════════
# 1. Clause detection
# ═══════════════════════════════════════════════════════════════════════════
def test_clause_detection(chunks: List[Chunk]) -> None:
    print("\n── 1. Clause detection ───────────────────────────────────")

    clauses = detect_clauses(chunks)
    found = {c.clause_type for c in clauses}

    for expected in ("Termination", "Payment", "Confidentiality", "Liability",
                     "Indemnification", "Governing Law", "Intellectual Property",
                     "Dispute Resolution", "Assignment"):
        check(f"detects {expected}", expected in found, str(sorted(found)))

    check("does not invent a force majeure clause the document lacks",
          "Force Majeure" not in found)
    check("reports force majeure as missing",
          "Force Majeure" in missing_clause_types(clauses))

    check("every clause cites a page inside the document",
          all(1 <= c.page <= 4 for c in clauses),
          str(sorted({c.page for c in clauses})))
    check("every clause carries verbatim source text",
          all(c.source_text for c in clauses))

    # The evidence must actually be in the document — this is what makes a
    # finding checkable rather than asserted.
    corpus = " ".join(c.text for c in chunks).lower()

    def normalise(text: str) -> str:
        return " ".join(text.lower().replace("...", "").split())

    grounded = sum(
        1 for c in clauses
        if normalise(c.source_text)[:60] in " ".join(corpus.split())
    )
    check("clause evidence is literal text from the document",
          grounded == len(clauses), f"{grounded}/{len(clauses)}")

    check("every clause has a risk level", all(c.risk_level in
          ("Low", "Medium", "High", "Critical") for c in clauses))
    check("confidence is reported", all(0 < c.confidence <= 1 for c in clauses))
    check("the taxonomy covers the 18 required clause types",
          len(CLAUSE_TYPES) >= 18, f"{len(CLAUSE_TYPES)} types")

    # An empty document must not produce clauses.
    check("no clauses are detected in an empty document", detect_clauses([]) == [])
    check("no clauses are detected in unrelated prose",
          detect_clauses([make_chunk("The weather today is pleasant and mild.")]) == [])


# ═══════════════════════════════════════════════════════════════════════════
# 2. Risk analysis
# ═══════════════════════════════════════════════════════════════════════════
def test_risk(chunks: List[Chunk]) -> None:
    print("\n── 2. Risk analysis ──────────────────────────────────────")

    clauses = detect_clauses(chunks)
    assessment = analyse_risk(chunks, clauses)

    check("produces an assessable result for a real contract", assessment.assessable)
    check("score is within 0-100", 0 <= assessment.score <= 100, str(assessment.score))
    check("risk level is reported", assessment.level in
          ("Low", "Medium", "High", "Critical"), assessment.level)
    check("findings are produced", len(assessment.findings) > 0,
          f"{len(assessment.findings)} findings")

    titles = " ".join(f.title.lower() for f in assessment.findings)
    check("detects suspension without liability", "suspend" in titles, titles[:90])
    check("detects the uncapped indemnity", "indemnit" in titles)
    check("detects automatic renewal", "renewal" in titles)

    evidenced = [f for f in assessment.findings if f.rule_id and
                 not f.rule_id.startswith("missing-")]
    check("every pattern finding quotes its evidence",
          all(f.source_text for f in evidenced))
    check("every pattern finding cites a page",
          all(f.page >= 1 for f in evidenced))
    check("every finding carries a recommendation",
          all(f.recommendation for f in assessment.findings))
    check("missing protections are reported as findings",
          any(f.rule_id.startswith("missing-") for f in assessment.findings))

    # The crucial one: a score of 0 must not read as "safe".
    empty = analyse_risk([], [])
    check("an unassessable document is not scored as safe",
          not empty.assessable and empty.level == "Unknown", empty.level)
    check("an unassessable document explains why", bool(empty.note))
    check("an unassessable document reports no fabricated findings",
          empty.findings == [])

    # A clean clause set should score better than a hostile one.
    hostile = [make_chunk(
        "Provider shall have unlimited liability. Client may not assign. "
        "The agreement shall automatically renew for successive terms. "
        "Provider may suspend services immediately without liability."
    )]
    hostile_assessment = analyse_risk(hostile, detect_clauses(hostile))
    check("a hostile document scores worse than the sample contract",
          hostile_assessment.score < assessment.score,
          f"{hostile_assessment.score} vs {assessment.score}")
    check("unlimited liability is rated Critical",
          any(f.severity == "Critical" for f in hostile_assessment.findings))


# ═══════════════════════════════════════════════════════════════════════════
# 3. Version comparison
# ═══════════════════════════════════════════════════════════════════════════
def test_comparison() -> None:
    print("\n── 3. Version comparison ─────────────────────────────────")

    version_a = [
        make_chunk("5. Termination\n5.1 Either party may terminate upon thirty (30) "
                   "days prior written notice.", page=1, number="5",
                   title="Termination", order=0),
        make_chunk("7. Limitation of Liability\n7.1 Aggregate liability shall not "
                   "exceed the fees paid in the preceding twelve (12) months.",
                   page=2, number="7", title="Limitation of Liability", order=1),
        make_chunk("3. Fees\n3.1 Client shall pay $50,000 per annum.",
                   page=1, number="3", title="Fees", order=2),
        make_chunk("9. Governing Law\n9.1 Governed by the laws of Maharashtra.",
                   page=3, number="9", title="Governing Law", order=3),
        make_chunk("11. Confidentiality\n11.1 Obligations survive for two (2) years.",
                   page=3, number="11", title="Confidentiality", order=4),
    ]
    version_b = [
        make_chunk("5. Termination\n5.1 Either party may terminate upon sixty (60) "
                   "days prior written notice.", page=1, number="5",
                   title="Termination", order=0),
        make_chunk("7. Limitation of Liability\n7.1 Provider shall have unlimited "
                   "liability under this Agreement.", page=2, number="7",
                   title="Limitation of Liability", order=1),
        make_chunk("3. Fees\n3.1 Client shall pay $75,000 per annum.",
                   page=1, number="3", title="Fees", order=2),
        make_chunk("9. Governing Law\n9.1 Governed by the laws of Karnataka.",
                   page=3, number="9", title="Governing Law", order=3),
        make_chunk("12. Force Majeure\n12.1 Neither party is liable for events "
                   "beyond the reasonable control of that party.",
                   page=4, number="12", title="Force Majeure", order=4),
    ]

    result = compare_service.compare(version_a, version_b, "v1.pdf", "v2.pdf")
    by_section = {c.section: c for c in result.changes}

    check("every change has a valid type",
          all(c.change_type in ("ADDED", "REMOVED", "MODIFIED", "UNCHANGED")
              for c in result.changes))
    check("every change has a valid severity",
          all(c.severity in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
              for c in result.changes))

    termination = next((c for c in result.changes if "Termination" in c.section), None)
    check("the termination clause is MODIFIED, not added and removed",
          termination is not None and termination.change_type == "MODIFIED",
          termination.change_type if termination else "not found")
    check("the notice-period change is detected as 30 -> 60",
          termination is not None and any(
              t["term"] == "Notice period" and "30" in str(t["old"]) and "60" in str(t["new"])
              for t in termination.term_changes),
          str(termination.term_changes) if termination else "")

    fees = next((c for c in result.changes if "Fees" in c.section), None)
    check("the fee change is detected as 50,000 -> 75,000",
          fees is not None and any(
              "50,000" in str(t["old"]) and "75,000" in str(t["new"])
              for t in fees.term_changes),
          str(fees.term_changes) if fees else "")

    liability = next((c for c in result.changes if "Liability" in c.section), None)
    check("capped becoming uncapped is detected",
          liability is not None and any(
              t["term"] == "Liability" and t["new"] == "Uncapped"
              for t in liability.term_changes),
          str(liability.term_changes) if liability else "")
    check("capped -> uncapped is rated CRITICAL",
          liability is not None and liability.severity == "CRITICAL",
          liability.severity if liability else "")

    law = next((c for c in result.changes if "Governing Law" in c.section), None)
    check("the governing-law change is detected",
          law is not None and any(
              t["term"] == "Governing law" and "Maharashtra" in str(t["old"])
              and "Karnataka" in str(t["new"]) for t in law.term_changes),
          str(law.term_changes) if law else "")

    check("a clause only in the new version is ADDED",
          any(c.change_type == "ADDED" and "Force Majeure" in c.section
              for c in result.changes))
    check("a clause only in the old version is REMOVED",
          any(c.change_type == "REMOVED" and "Confidentiality" in c.section
              for c in result.changes))
    check("gained protections are listed",
          "Force Majeure" in result.added_clauses, str(result.added_clauses))
    check("lost protections are listed",
          "Confidentiality" in result.removed_clauses, str(result.removed_clauses))

    check("a riskier new version produces a negative risk shift",
          result.risk_shift < 0, str(result.risk_shift))
    check("the risk shift is described in words", bool(result.risk_shift_label))
    check("the summary names real changes", "modified" in result.summary.lower())
    check("explanations are generated for every change",
          all(c.explanation for c in result.changes))

    # Identical documents must report no change rather than invented differences.
    same = compare_service.compare(version_a, list(version_a), "a.pdf", "a.pdf")
    check("an identical document reports everything UNCHANGED",
          all(c.change_type == "UNCHANGED" for c in same.changes),
          str({c.change_type for c in same.changes}))
    check("an identical document has no risk shift", same.risk_shift == 0,
          str(same.risk_shift))
    check("an identical document gains and loses nothing",
          not same.added_clauses and not same.removed_clauses)

    # Renumbering alone is not a change.
    renumbered = [make_chunk(
        "8. Termination\n8.1 Either party may terminate upon thirty (30) days "
        "prior written notice.", page=1, number="8", title="Termination", order=0)]
    original = [version_a[0]]
    moved = compare_service.compare(original, renumbered, "a", "b")
    check("a renumbered clause is matched, not reported as add+remove",
          all(c.change_type != "ADDED" for c in moved.changes),
          str([c.change_type for c in moved.changes]))


# ═══════════════════════════════════════════════════════════════════════════
# 4. Regulatory compliance
# ═══════════════════════════════════════════════════════════════════════════
def test_compliance(chunks: List[Chunk]) -> None:
    print("\n── 4. Regulatory compliance ──────────────────────────────")

    check("the three required frameworks are supported",
          set(SUPPORTED_FRAMEWORKS) == {"GDPR", "HIPAA", "SOC2"},
          str(SUPPORTED_FRAMEWORKS))
    check("every framework defines requirements",
          all(len(FRAMEWORKS[f]) >= 6 for f in SUPPORTED_FRAMEWORKS),
          str({f: len(FRAMEWORKS[f]) for f in SUPPORTED_FRAMEWORKS}))

    result = compliance_service.check(chunks, ["GDPR", "HIPAA", "SOC2"])
    all_requirements = [r for f in result["frameworks"] for r in f["requirements"]]

    check("every requirement has a valid status",
          all(r["status"] in STATUSES for r in all_requirements))
    check("every requirement carries a severity",
          all(r["severity"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
              for r in all_requirements))
    check("the overall score is within 0-100",
          0 <= result["compliance_score"] <= 100, str(result["compliance_score"]))

    # The property that matters: a PASS must be backed by text from the document.
    passes = [r for r in all_requirements if r["status"] == "PASS"]
    check("every PASS cites evidence from the document",
          all(r["evidence"] for r in passes), f"{len(passes)} passes")
    check("every PASS cites a page",
          all(r["page"] for r in passes))

    corpus = " ".join(" ".join(c.text.split()).lower() for c in chunks)
    grounded = sum(1 for r in passes
                   if " ".join(r["evidence"].replace("...", "").split()).lower()[:50] in corpus)
    check("PASS evidence is literal text from the document",
          grounded == len(passes), f"{grounded}/{len(passes)}")

    not_found = [r for r in all_requirements if r["status"] == "NOT_FOUND"]
    check("requirements with no supporting language are NOT_FOUND, not PASS",
          len(not_found) > 0, f"{len(not_found)} not found")
    check("a gap carries a recommendation",
          all(r["recommendation"] for r in not_found))

    # A commercial contract with no data-protection language must not pass HIPAA.
    hipaa = next(f for f in result["frameworks"] if f["framework"] == "HIPAA")
    check("a contract with no PHI language does not score full HIPAA marks",
          hipaa["score"] < 100, str(hipaa["score"]))

    # A document that genuinely addresses a requirement must pass it.
    gdpr_doc = [make_chunk(
        "12. Data Protection\n12.1 The processor shall notify the controller of any "
        "personal data breach without undue delay and in any event within 72 hours. "
        "12.2 The processor shall not engage any sub-processor without prior written "
        "authorisation. 12.3 Personal data shall be deleted or returned on termination."
    )]
    gdpr_result = compliance_service.check(gdpr_doc, ["GDPR"])
    gdpr_requirements = gdpr_result["frameworks"][0]["requirements"]
    breach = next(r for r in gdpr_requirements
                  if r["requirement_id"] == "gdpr-breach-notification")
    check("a document that does address breach notification passes it",
          breach["status"] == "PASS", breach["status"])
    check("the passing evidence quotes the document",
          "72 hours" in breach["evidence"] or "undue delay" in breach["evidence"],
          breach["evidence"][:70])

    check("the result states it is not a legal opinion",
          "not a legal opinion" in result["disclaimer"].lower())
    check("unsupported framework names fall back to the supported set",
          bool(compliance_service.check(chunks, ["NONSENSE"])["frameworks"]))


# ═══════════════════════════════════════════════════════════════════════════
# 5. Entity extraction
# ═══════════════════════════════════════════════════════════════════════════
def test_entities(chunks: List[Chunk]) -> None:
    print("\n── 5. Entity extraction ──────────────────────────────────")

    facts = legal_extractor.extract_all(chunks)
    for category in ("parties", "organizations", "people", "key_dates",
                     "financial_terms", "legal_references"):
        check(f"extracts {category}", category in facts, str(sorted(facts)))

    check("finds both contracting parties", len(facts["parties"]) == 2,
          str([p.value for p in facts["parties"]]))
    check("identifies party roles",
          {p.label for p in facts["parties"]} == {"Provider", "Client"},
          str({p.label for p in facts["parties"]}))
    check("finds the signatories", len(facts["people"]) >= 2,
          str([(p.value, p.label) for p in facts["people"]]))
    check("captures signatory titles",
          any("Officer" in p.label or "President" in p.label for p in facts["people"]),
          str([p.label for p in facts["people"]]))

    organizations = [o.value for o in facts["organizations"]]
    check("does not mistake the document title for an organisation",
          not any("AGREEMENT" in o.upper() for o in organizations), str(organizations))
    check("finds the arbitration body",
          any("Arbitration" in o for o in organizations), str(organizations))

    check("every entity cites a page",
          all(item.source_pages for group in facts.values() for item in group))
    check("entity pages are inside the document",
          all(1 <= page <= 4 for group in facts.values()
              for item in group for page in item.source_pages))

    corpus = " ".join(" ".join(c.text.split()) for c in chunks)
    valued = [i for g in ("parties", "organizations", "people", "financial_terms")
              for i in facts[g] if i.value]
    grounded = sum(1 for item in valued if item.value.split()[0] in corpus)
    check("extracted values are literal text from the document",
          grounded == len(valued), f"{grounded}/{len(valued)}")

    check("no entities are extracted from an empty document",
          all(not group for group in legal_extractor.extract_all([]).values()))


# ═══════════════════════════════════════════════════════════════════════════
def main_() -> int:
    print("=" * 74)
    print("Legal analysis: clauses, risk, comparison, compliance, entities")
    print("=" * 74)

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = write_sample_contract(os.path.join(tmp, "contract.pdf"))
        chunks = canonical_chunks(pdf_path)

        test_clause_detection(chunks)
        test_risk(chunks)
        test_comparison()
        test_compliance(chunks)
        test_entities(chunks)

    print("\n" + "=" * 74)
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    for label in FAILED:
        print(f"  FAILED: {label}")
    print("=" * 74)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main_())
