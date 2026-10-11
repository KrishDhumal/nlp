"""
Member 2 pipeline tests — PDF ingestion, chunking, vector store and audit API.

Runs without MongoDB, without Pinecone and without a Gemini key:
  * `get_db` / `get_current_user` are replaced with in-memory stubs.
  * `AuditService._invoke_json` is stubbed per stage, so the four-stage
    orchestration, normalisation and scoring are exercised for real while the
    network is not touched.
  * The vector store is pinned to its local index.

Everything else — validation, extraction, section detection, chunking,
embedding, the routes and the response schema — is the real code path.

Run with:  PYTHONPATH=. ./venv/bin/python tests/test_audit_pipeline.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

import jwt
from fastapi.testclient import TestClient

import main
from app.api.deps import get_current_user, get_db
from app.core.config import settings
from app.services.ai.audit_service import AuditService
from app.services.ai.chunking import TokenAwareChunker, heuristic_token_count
from app.services.ai.processor import DocumentProcessor, PDFValidationError, classify_clause_type
from app.services.ai.structure import detect_sections
from app.services.ai.vector_store import vector_store_service
from tests.fixtures import EXPECTED_SECTIONS, write_sample_contract

PASSED: list = []
FAILED: list = []


def check(label: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(label)
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))


# ═══════════════════════════════════════════════════════════════════════════
# 1. Upload validation
# ═══════════════════════════════════════════════════════════════════════════
def test_validation(pdf_bytes: bytes) -> None:
    print("\n── 1. Upload validation ──────────────────────────────────")
    processor = DocumentProcessor()

    try:
        processor.validate(pdf_bytes, "contract.pdf")
        check("accepts a genuine PDF", True, f"{len(pdf_bytes)} bytes")
    except PDFValidationError as e:
        check("accepts a genuine PDF", False, str(e))

    cases = [
        ("rejects a non-PDF extension", b"%PDF-1.4 real header", "notes.txt"),
        ("rejects bad magic bytes", b"this is definitely not a pdf", "fake.pdf"),
        ("rejects an empty file", b"", "empty.pdf"),
        ("rejects an oversized file",
         b"%PDF-" + b"x" * (settings.MAX_UPLOAD_MB * 1024 * 1024 + 1), "huge.pdf"),
    ]
    for label, blob, name in cases:
        try:
            processor.validate(blob, name)
            check(label, False, "validation wrongly passed")
        except PDFValidationError as e:
            check(label, True, str(e)[:60])


# ═══════════════════════════════════════════════════════════════════════════
# 2. Extraction, section detection and chunking
# ═══════════════════════════════════════════════════════════════════════════
def test_processing(pdf_path: str) -> None:
    print("\n── 2. Extraction, sections and chunking ──────────────────")
    # Heuristic token counting keeps this test free of any model download.
    processor = DocumentProcessor(
        chunker=TokenAwareChunker(count_tokens=heuristic_token_count)
    )
    pages = processor.extract_pages(pdf_path)

    check("extracts all 4 pages", len(pages) == 4, f"{len(pages)} pages")
    check("every page yields text", all(t.strip() for _, t in pages),
          f"chars={[len(t) for _, t in pages]}")

    document = processor.process(pdf_path, "doc-test", "user-test", "contract.pdf")
    sections = document.sections
    found = [(sec.section_number, sec.section_title, sec.page_start) for sec in sections]
    check("detects all 12 sections with correct refs and pages",
          found == EXPECTED_SECTIONS,
          f"{len(found)} found" if found == EXPECTED_SECTIONS else f"got {found}")

    # The regression this guards: ALL-CAPS liability *body* text used to be
    # mistaken for a section heading, splitting section 7 into three fragments.
    liability = [sec for sec in sections if sec.section_number == "7"]
    check("ALL-CAPS body text is not treated as a heading",
          len(liability) == 1 and len(liability[0].text) > 300,
          f"section 7 body = {len(liability[0].text) if liability else 0} chars")

    chunks = document.chunks
    check("produces chunks", len(chunks) > 0, f"{len(chunks)} chunks")
    check("no chunk exceeds the token budget",
          all(c.token_count <= settings.CHUNK_MAX_TOKENS for c in chunks),
          f"max={max(c.token_count for c in chunks)} of {settings.CHUNK_MAX_TOKENS}")
    check("every chunk carries citation metadata",
          all(c.page_start and c.section_id and c.chunk_id for c in chunks))
    check("chunks span every page",
          sorted({c.page_start for c in chunks}) == [1, 2, 3, 4])

    # Page attribution: the payment clause is on page 2 of the fixture.
    payment = [c for c in chunks if "Fees and Payment" in c.section_label]
    check("attributes the payment section to page 2",
          bool(payment) and payment[0].page_start == 2,
          f"page {payment[0].page_start}" if payment else "not found")

    print("\n  Clause-type tagging:")
    for chunk in chunks:
        print(f"    {chunk.section_label[:42]:44} -> "
              f"{classify_clause_type(chunk.text, chunk.section_title)}")

    expectations = {
        "5 Termination": "Termination",
        "3 Fees and Payment": "Payment",
        "7 Limitation of Liability": "Liability",
        "8 Indemnification": "Indemnification",
        "4 Confidentiality": "Confidentiality",
        "9 Governing Law and Dispute Resolution": "Governing Law",
        "6 Intellectual Property": "IP Rights",
    }
    correct = 0
    for chunk in chunks:
        want = expectations.get(chunk.section_label)
        if want and classify_clause_type(chunk.text, chunk.section_title) == want:
            correct += 1
    check("keyword clause-type tagging covers all 7 categories",
          correct == len(expectations), f"{correct}/{len(expectations)} correct")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Vector store: indexing, retrieval, isolation
# ═══════════════════════════════════════════════════════════════════════════
def test_vector_store(pdf_path: str) -> None:
    print("\n── 3. Vector store (local index) ─────────────────────────")
    processor = DocumentProcessor(
        chunker=TokenAwareChunker(count_tokens=heuristic_token_count)
    )
    vector_store_service._pinecone_ok = False  # never touch Pinecone in tests
    user, audit = "vec-user", "vec-audit"
    vector_store_service.delete_audit_vectors(user, audit)

    # The RAG index consumes exactly the same canonical chunks as summarisation.
    document = processor.process(pdf_path, audit, user, "contract.pdf")
    chunks = document.chunks
    result = vector_store_service.index_document(document)
    check("indexes every chunk", result["indexed"] == len(chunks), str(result))

    queries = {
        "What is the notice period for termination?": "Termination",
        "Is there a cap on liability?": "Limitation of Liability",
        "What are the payment terms?": "Fees and Payment",
        "Which law governs this contract?": "Governing Law",
        "Who owns the intellectual property?": "Intellectual Property",
    }
    hits_correct = 0
    print("\n  Semantic retrieval:")
    for question, expected_section in queries.items():
        hits = vector_store_service.search_similar(question, user, audit, k=1)
        top = hits[0].metadata["section_label"] if hits else "(none)"
        ok = expected_section in top
        hits_correct += ok
        print(f"    {'✓' if ok else '✗'} {question[:44]:46} -> {top}")
    check("top-1 retrieval finds the right section for every question",
          hits_correct == len(queries), f"{hits_correct}/{len(queries)}")

    sample = vector_store_service.search_similar("termination notice", user, audit, k=1)[0]
    required = {"doc_id", "audit_id", "user_id", "page_number", "clause_type", "raw_text"}
    check("vector metadata carries the required fields",
          required <= set(sample.metadata), f"missing={required - set(sample.metadata)}")
    check("clause_type is populated, not a placeholder",
          sample.metadata["clause_type"] != "General", sample.metadata["clause_type"])

    check("another user's namespace returns nothing",
          vector_store_service.search_similar("termination", "other-user", audit, k=5) == [])
    check("another document in the same namespace returns nothing",
          vector_store_service.search_similar("termination", user, "other-audit", k=5) == [])

    vector_store_service.delete_audit_vectors(user, audit)
    check("delete removes the audit's vectors",
          vector_store_service.search_similar("termination", user, audit, k=5) == [])


# ═══════════════════════════════════════════════════════════════════════════
# 4. Audit API — upload, retrieve, serve, with stubbed DB and LLM
# ═══════════════════════════════════════════════════════════════════════════
STAGES = {
    "overview": {
        "contract_type": "Master Services Agreement",
        "executive_summary": "This MSA governs consulting services between Acme and Global Logistics.",
        "key_takeaways": ["Net-30 payment terms.", "30 days notice to terminate.",
                          "Liability capped at 12 months of fees."],
        "risks": [
            {"title": "Unilateral suspension on late payment", "severity": "high",
             "description": "Provider may suspend without liability.", "page_number": 2,
             "snippet": "Provider may suspend services immediately"},
            {"title": "Uncapped indemnity", "severity": "CRITICAL",
             "description": "Indemnity sits outside the liability cap.", "page_number": 3,
             "snippet": "not subject to the cap"},
            {"title": "Class action waiver", "severity": "moderate",
             "description": "Waives jury trial and class participation.", "page_number": 4},
            {"title": "Unparseable severity", "severity": "banana",
             "description": "Should default to Medium.", "page_number": 1},
            {"description": "no title at all — must be dropped"},
        ],
        "missing_clauses": ["Force Majeure", "Data Protection / GDPR terms"],
    },
    "clauses": {"clauses": [
        {"category": "Termination & Renewal", "title": "Termination for Convenience",
         "page_number": 2, "section": "Sec 5.1", "risk_level": "Medium",
         "snippet": "thirty (30) calendar days prior written notice",
         "explanation": "Either side can walk with 30 days notice.",
         "recommendation": "Confirm 30 days is enough for transition."},
        {"category": "Limitation of Liability", "title": "Liability Cap",
         "page_number": 3, "section": "Sec 7.1", "risk_level": "High",
         "snippet": "SHALL NOT EXCEED THE TOTAL FEES PAID",
         "explanation": "Damages are capped at one year of fees.",
         "recommendation": "Push for a higher multiple."},
        {"category": "intellectual property", "title": "IP Assignment",
         "page_number": 3, "section": "Sec 6.2", "risk_level": "low",
         "snippet": "Provider assigns to Client all right, title",
         "explanation": "You own deliverables once paid.",
         "recommendation": "Accept as drafted."},
        {"category": "Totally Unknown Category", "title": "Odd clause",
         "page_number": 1, "risk_level": "Low", "snippet": "something",
         "explanation": "x", "recommendation": "y"},
        {"title": "", "snippet": ""},
    ]},
    "entities": {
        "parties": [
            {"name": "Acme Enterprise Solutions Inc.", "role": "Provider",
             "address": "100 Innovation Way, New York, NY 10001", "signatory": "John Doe (CEO)"},
            {"name": "Global Logistics Corp.", "role": "Client",
             "address": "500 Commerce Blvd, Chicago, IL 60601", "signatory": "Jane Smith (VP Ops)"},
            {"role": "party with no name — must be dropped"},
        ],
        "dates": [
            {"label": "Effective Date", "value": "October 1, 2026", "page_number": 1,
             "section": "Preamble", "note": "Commencement"},
            {"label": "Notice Period", "value": "30 days", "page_number": "2",
             "section": "Sec 5.1", "note": "For convenience"},
            {"label": "date with no value — must be dropped"},
        ],
        "financials": [
            {"label": "Contract Value", "amount": "$240,000 USD", "page_number": 2,
             "section": "Sec 3.1", "detail": "$10,000/month"},
            {"label": "Late Interest", "amount": "1.5% monthly", "page_number": 2,
             "section": "Sec 3.3", "detail": "After 30 days"},
        ],
        "jurisdiction": {"governing_law": "State of New York", "venue": "New York County",
                         "dispute_resolution": "AAA binding arbitration",
                         "page_number": 4, "section": "Sec 9.2"},
    },
    "sections": {"section_summaries": [
        {"title": "3. Fees and Payment", "page_number": 2, "section": "Sec 3.1",
         "text_snippet": "Invoices are payable within thirty (30) days",
         "key_points": ["Net-30", "1.5% monthly interest"]},
        {"title": "5. Termination", "page_number": 2, "section": "Sec 5.1",
         "text_snippet": "Either party may terminate for convenience",
         "key_points": ["30 days notice", "14-day cure period"]},
        {"title": "section with no page number", "key_points": []},
    ]},
}


class _Cursor:
    """Chainable stand-in for a Motor cursor: .sort().limit().to_list()."""

    def __init__(self, docs: list):
        self._docs = docs

    def sort(self, key, direction=1):
        self._docs = sorted(
            self._docs,
            key=lambda d: (d.get(key) is None, d.get(key)),
            reverse=direction < 0,
        )
        return self

    def limit(self, n):
        self._docs = self._docs[:n]
        return self

    async def to_list(self, length=None):
        return self._docs if length is None else self._docs[:length]


class _Collection:
    """Just enough of a Motor collection for the audit and dashboard routes."""

    def __init__(self):
        self.docs: dict = {}

    async def replace_one(self, flt, doc, upsert=False):
        self.docs[doc["_id"]] = doc
        return type("Result", (), {"upserted_id": doc["_id"]})()

    async def insert_one(self, doc):
        self.docs[doc["_id"]] = doc
        return type("Result", (), {"inserted_id": doc["_id"]})()

    async def update_one(self, flt, update, upsert=False):
        target = None
        for doc in self.docs.values():
            if all(doc.get(k) == v for k, v in flt.items()):
                target = doc
                break
        if target is None and upsert:
            target = dict(flt)
            self.docs[target.get("_id", f"gen-{len(self.docs)}")] = target
        if target is not None:
            target.update(update.get("$set", {}))
            for key, value in update.get("$setOnInsert", {}).items():
                target.setdefault(key, value)
        matched = 1 if target is not None else 0
        return type("Result", (), {"matched_count": matched, "modified_count": matched})()

    async def find_one(self, flt):
        for doc in self.docs.values():
            if all(doc.get(key) == value for key, value in flt.items()):
                return doc
        return None

    def find(self, flt=None):
        flt = flt or {}
        return _Cursor([
            doc for doc in self.docs.values()
            if all(doc.get(key) == value for key, value in flt.items())
        ])


class _DB:
    def __init__(self):
        self._collections = {"audits": _Collection(), "users": _Collection()}

    def __getitem__(self, name):
        return self._collections[name]


def test_api(pdf_bytes: bytes) -> None:
    print("\n── 4. Audit API (stubbed DB + stubbed LLM) ───────────────")

    db = _DB()
    user = {"_id": "u-123", "email": "tester@example.com", "name": "Tester"}
    db["users"].docs["u-123"] = user

    main.app.dependency_overrides[get_db] = lambda: db
    main.app.dependency_overrides[get_current_user] = lambda: user

    original_invoke = AuditService._invoke_json

    async def fake_invoke(self, prompt, stage):
        return STAGES.get(stage)

    AuditService._invoke_json = fake_invoke
    vector_store_service._pinecone_ok = False
    indexed_audits: list = []

    # The pipeline runs for real here (TestClient executes background tasks
    # synchronously), but with a stand-in summariser: no model download, and the
    # assertions stay deterministic.
    from app.services.ai.summarization.service import SummarizationService
    from app.services.pipeline import document_pipeline
    from tests.test_summarization_pipeline import FakeSummarizer

    original_summarizer = document_pipeline._summarizer
    document_pipeline._summarizer = SummarizationService(provider=FakeSummarizer())
    original_processor_chunker = document_pipeline.processor.chunker
    document_pipeline.processor.chunker = TokenAwareChunker(
        count_tokens=heuristic_token_count
    )

    try:
        client = TestClient(main.app)

        accepted = client.post(
            "/api/audits/upload",
            files={"file": ("contract.pdf", pdf_bytes, "application/pdf")},
        )
        check("POST /api/audits/upload returns 202 Accepted",
              accepted.status_code == 202,
              accepted.text[:160] if accepted.status_code != 202 else "")
        if accepted.status_code != 202:
            return

        audit_id = accepted.json()["id"]
        indexed_audits.append(audit_id)

        # The background task has already run under TestClient, so the finished
        # analysis is available immediately.
        response = client.get(f"/api/audits/{audit_id}")
        check("the processed analysis is retrievable", response.status_code == 200,
              response.text[:160] if response.status_code != 200 else "")
        if response.status_code != 200:
            return
        data = response.json()

        print(f"\n    contract_type : {data['contract_type']}")
        print(f"    risk_score    : {data['risk_score']}  counts={data['risk_counts']}")
        print(f"    pages/chunks  : {data['page_count']}/{data['chunk_count']}"
              f"  size={data['file_size']}B  vectors={data['vector_backend']}")
        print(f"    risks         : {[(r['title'][:24], r['severity']) for r in data['risks']]}")
        print(f"    clauses       : {[(c['category'], c['risk_level']) for c in data['clauses']]}")
        print(f"    entities      : {len(data['entities']['parties'])} parties, "
              f"{len(data['entities']['dates'])} dates, "
              f"{len(data['entities']['financials'])} financials")
        print(f"    suggestions   : {len(data['suggestions'])}\n")

        # Clauses and risks are now produced by the deterministic analysis layer
        # over the canonical chunks, not by the LLM. The LLM only enriches
        # explanations, so these assert the deterministic contract. (The LLM
        # normalisers are still unit-tested in section 5.)
        categories = [c["category"] for c in data["clauses"]]
        check("clauses are detected from the document itself",
              len(data["clauses"]) > 0, f"{len(data['clauses'])} clauses")
        check("clauses map onto the UI categories",
              {"Termination", "Liability", "IP Rights"} <= set(categories), str(categories))
        check("every clause quotes source text from the document",
              all(c["snippet"] for c in data["clauses"]))
        check("every clause cites a page inside the document",
              all(1 <= c["page_number"] <= data["page_count"] for c in data["clauses"]),
              str(sorted({c["page_number"] for c in data["clauses"]})))

        severities = {r["severity"] for r in data["risks"]}
        check("risk findings are produced from the document",
              len(data["risks"]) > 0, f"{len(data['risks'])} risks")
        check("risk severities are valid",
              severities <= {"Low", "Medium", "High", "Critical"}, str(severities))
        check("a risk score of 0 is not reported as assessable without findings",
              data["risk_score"] > 0 or not data["risks"],
              f"score={data['risk_score']}")

        check("drops a party with no name", len(data["entities"]["parties"]) == 2)

        # Entities now come from deterministic extraction over the canonical
        # chunks, not from the LLM payload — so these assert real extracted
        # values rather than the stub's. (The LLM entity normaliser is still
        # unit-tested in section 5.)
        dates = data["entities"]["dates"]
        check("dates are extracted deterministically from the document",
              len(dates) > 2, f"{len(dates)} dates")
        check("every date page number is an int inside the document",
              all(isinstance(d["page_number"], int) and 1 <= d["page_number"] <= data["page_count"]
                  for d in dates),
              str([d["page_number"] for d in dates][:6]))
        check("every date carries a label and a literal value",
              all(d["label"] and d["value"] for d in dates))
        financials = data["entities"]["financials"]
        check("financial values are extracted with pages",
              bool(financials) and all(f["amount"] and f["page_number"] for f in financials),
              f"{len(financials)} amounts")
        check("jurisdiction is populated",
              data["entities"]["jurisdiction"]["governing_law"] == "State of New York")

        check("risk score is within 0-100", 0 <= data["risk_score"] <= 100,
              str(data["risk_score"]))
        check("suggestions combine missing clauses and recommendations",
              len(data["suggestions"]) >= len(data["missing_clauses"]))
        check("summary aliases executive_summary for other modules",
              data["summary"] == data["executive_summary"])

        # Retrieval of the saved audit
        saved = client.get(f"/api/audits/{audit_id}")
        check("GET /api/audits/{id} returns 200", saved.status_code == 200)
        check("persisted audit matches the upload response",
              saved.status_code == 200
              and saved.json()["clauses"] == data["clauses"]
              and saved.json()["risk_score"] == data["risk_score"])
        check("GET /api/audits/{unknown} returns 404",
              client.get("/api/audits/no-such-audit").status_code == 404)

        # Route-order regression: /api/audits/recent must not be captured by
        # /api/audits/{audit_id}.
        check("/api/audits/recent is not swallowed by /api/audits/{id}",
              client.get("/api/audits/recent").status_code != 404,
              f"status {client.get('/api/audits/recent').status_code}")

        # Rejections through the real route
        for label, blob, name in [
            ("route rejects a .txt upload", b"%PDF-1.4", "notes.txt"),
            ("route rejects bad magic bytes", b"not a pdf", "fake.pdf"),
            ("route rejects an empty upload", b"", "empty.pdf"),
        ]:
            r = client.post("/api/audits/upload",
                            files={"file": (name, blob, "application/pdf")})
            check(label, r.status_code == 400, f"status {r.status_code}")

        # The uploaded document must be searchable for Member 3's RAG chat
        hits = vector_store_service.search_similar(
            "What is the notice period for termination?", "u-123", audit_id, k=2)
        check("uploaded document is immediately searchable", bool(hits),
              f"{len(hits)} hits -> {hits[0].metadata['section_label'] if hits else ''}")

        # PDF serving and ownership
        good = jwt.encode({"sub": user["email"]}, settings.SECRET_KEY,
                          algorithm=settings.ALGORITHM)
        other = jwt.encode({"sub": "someone-else@example.com"}, settings.SECRET_KEY,
                           algorithm=settings.ALGORITHM)
        served = client.get(f"/api/audits/file/{audit_id}?token={good}")
        check("owner can stream the original PDF",
              served.status_code == 200 and served.content[:5] == b"%PDF-",
              f"{len(served.content)} bytes")
        check("missing token is rejected",
              client.get(f"/api/audits/file/{audit_id}").status_code == 401)
        check("another user's token cannot read the PDF",
              client.get(f"/api/audits/file/{audit_id}?token={other}").status_code == 401)

    finally:
        AuditService._invoke_json = original_invoke
        document_pipeline._summarizer = original_summarizer
        document_pipeline.processor.chunker = original_processor_chunker
        main.app.dependency_overrides.clear()
        # Always drop what this test indexed, even if a check above raised —
        # the local index is persisted to disk and would otherwise accumulate.
        for stale in indexed_audits:
            vector_store_service.delete_audit_vectors("u-123", stale)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Scoring and normalisation units
# ═══════════════════════════════════════════════════════════════════════════
def test_scoring() -> None:
    print("\n── 5. Scoring and normalisation ──────────────────────────")
    service = AuditService()

    check("a clean document scores 100",
          service.compute_risk_score([], [], []) == 100)
    check("score is floored at 0, never negative",
          service.compute_risk_score([{"severity": "Critical"}] * 20, [], []) == 0)

    moderate = service.compute_risk_score(
        [{"severity": "High"}, {"severity": "Medium"}],
        [{"risk_level": "Low"}],
        ["Force Majeure"],
    )
    check("a mixed document lands between 0 and 100", 0 < moderate < 100, str(moderate))
    check("critical findings cost more than low ones",
          service.compute_risk_score([{"severity": "Critical"}], [], [])
          < service.compute_risk_score([{"severity": "Low"}], [], []))

    for raw, expected in [("high", "High"), ("CRITICAL", "Critical"), ("moderate", "Medium"),
                          ("minor", "Low"), ("High Risk", "High"), ("nonsense", "Medium"),
                          (None, "Medium")]:
        check(f"severity {raw!r} -> {expected}",
              service.normalize_severity(raw) == expected,
              service.normalize_severity(raw))

    for raw, expected in [("Termination & Renewal", "Termination"),
                          ("Limitation of Liability", "Liability"),
                          ("intellectual property", "IP Rights"),
                          ("Dispute Resolution", "Governing Law"),
                          ("Payment & Financial Terms", "Payment"),
                          ("Non-Disclosure", "Confidentiality"),
                          ("Indemnity", "Indemnification"),
                          ("Nonsense Category", "Other")]:
        check(f"category {raw!r} -> {expected}",
              service.normalize_category(raw) == expected,
              service.normalize_category(raw))

    # Malformed model output must never crash normalisation.
    check("tolerates a non-list risks payload", service._normalize_risks("garbage") == [])
    check("tolerates a non-dict entities payload",
          service._normalize_entities(None)["jurisdiction"]["governing_law"] == "Not specified")

    # JSON parsing tolerances
    parse = service._parse_json
    check("parses a fenced JSON block", parse('```json\n{"a": 1}\n```') == {"a": 1})
    check("parses JSON wrapped in prose",
          parse('Here you go:\n{"a": 1}\nHope that helps!') == {"a": 1})
    check("repairs trailing commas", parse('{"a": [1, 2,],}') == {"a": [1, 2]})
    check("returns None for unparseable output", parse("not json at all") is None)
    check("returns None for empty output", parse("") is None)

    # Retry classification — the bug where "generateContent" matched "rate".
    check("404 model-not-found is NOT retryable",
          not service._is_retryable(Exception(
              "404 NOT_FOUND models/x is not supported for generateContent")))
    check("429 quota errors are retryable",
          service._is_retryable(Exception("429 RESOURCE_EXHAUSTED quota exceeded")))
    check("503 overloaded is retryable",
          service._is_retryable(Exception("503 Service Unavailable, model overloaded")))
    check("401 unauthenticated is NOT retryable",
          not service._is_retryable(Exception("401 invalid api key")))

    # The per-day free-tier cap: retrying spends more of an empty bucket, so it
    # must be classified as non-retryable and flagged separately.
    daily = Exception(
        "429 RESOURCE_EXHAUSTED Quota exceeded for metric: "
        "generativelanguage.googleapis.com/generate_content_free_tier_requests, "
        "limit: 20 quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier. "
        "Please retry in 38.867487597s."
    )
    check("per-day quota exhaustion is NOT retryable", not service._is_retryable(daily))
    check("per-day quota exhaustion is detected", service._is_daily_quota_exhausted(daily))
    per_minute = Exception("429 rate limit exceeded, please retry in 12.5s")
    check("per-minute throttle stays retryable", service._is_retryable(per_minute))
    check("per-minute throttle is not flagged as daily",
          not service._is_daily_quota_exhausted(per_minute))
    check("honours the server's retry-after hint",
          service._retry_after(per_minute, 3) == 13.5,
          str(service._retry_after(per_minute, 3)))
    check("caps an absurd retry-after hint",
          service._retry_after(Exception("please retry in 9999s"), 3) == 45.0)
    check("falls back to the default delay with no hint",
          service._retry_after(Exception("503 overloaded"), 3) == 3)

    # Context selection for documents larger than the budget
    from langchain_core.documents import Document
    many = [Document(page_content=f"chunk {i}", metadata={"page_number": i}) for i in range(100)]
    selected = service._select_chunks(many, 28)
    check("caps context at the budget", len(selected) <= 28, f"{len(selected)} chunks")
    check("keeps the opening of the document", selected[0].page_content == "chunk 0")
    check("keeps the end of the document", selected[-1].page_content == "chunk 99")
    check("a short document is passed through untouched",
          service._select_chunks(many[:10], 28) == many[:10])


# ═══════════════════════════════════════════════════════════════════════════
# 6. Daily-quota short circuit
# ═══════════════════════════════════════════════════════════════════════════
def test_quota_short_circuit() -> None:
    """
    When the first stage hits the per-day cap, the remaining stages must not
    each grind through the whole model chain — that would spend four times the
    quota for nothing and stall the upload.
    """
    print("\n── 6. Daily-quota short circuit ──────────────────────────")
    import asyncio as _asyncio
    from langchain_core.documents import Document

    service = AuditService()
    service.api_key = "fake-key-for-test"
    calls: list = []

    daily_error = Exception(
        "429 RESOURCE_EXHAUSTED Quota exceeded for metric: "
        "generate_content_free_tier_requests, limit: 20, quotaId: "
        "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
    )

    class _Boom:
        def __init__(self, model):
            self.model = model

        async def ainvoke(self, prompt):
            calls.append(self.model)
            raise daily_error

    service._get_llm = lambda model_name: _Boom(model_name)

    chunks = [Document(page_content="Some contract text " * 20,
                       metadata={"page_number": 1, "section_label": "1 Term"})]

    try:
        _asyncio.run(service.generate_analysis(chunks))
        check("raises when every stage fails", False, "no exception raised")
    except RuntimeError as e:
        check("raises when every stage fails", True)
        check("error names the quota as the cause, not a config problem",
              "quota" in str(e).lower(), str(e)[:80])

    check("stops after the first quota failure instead of retrying the chain",
          len(calls) <= 4, f"{len(calls)} API calls made across 4 stages")
    print(f"    API calls attempted: {len(calls)} (models: {calls})")


# ═══════════════════════════════════════════════════════════════════════════
def main_() -> int:
    print("=" * 74)
    print("Member 2 pipeline tests — PDF ingestion, vectors, audit API")
    print("=" * 74)

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = write_sample_contract(os.path.join(tmp, "contract.pdf"))
        with open(pdf_path, "rb") as fh:
            pdf_bytes = fh.read()

        test_validation(pdf_bytes)
        test_processing(pdf_path)
        test_vector_store(pdf_path)
        test_api(pdf_bytes)
        test_scoring()
        test_quota_short_circuit()

    print("\n" + "=" * 74)
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        for label in FAILED:
            print(f"  FAILED: {label}")
    print("=" * 74)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main_())
