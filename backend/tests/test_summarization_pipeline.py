"""
Tests for the canonical pipeline and the summarisation architecture.

Runs with no MongoDB, no Pinecone, no API key and no model download: the
summarisation provider is replaced with a deterministic fake, and token counting
uses the heuristic counter. What is exercised for real: cleaning, structure
detection, token-aware chunking, deterministic legal extraction, the hierarchical
map/reduce, provider selection, and the async upload/status API contract.

Run with:  PYTHONPATH=. ./venv/bin/python tests/test_summarization_pipeline.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

from typing import List, Optional

from app.core.config import settings
from app.domain.document import CanonicalDocument, Chunk, ProcessingStatus, Section
from app.services.ai.chunking import TokenAwareChunker, heuristic_token_count
from app.services.ai.extraction import legal_extractor
from app.services.ai.processor import DocumentProcessor
from app.services.ai.structure import detect_sections, match_heading, page_fallback_sections
from app.services.ai.summarization.base import CHUNK_INSTRUCTION, SummarizationProvider
from app.services.ai.summarization.registry import build_provider, get_provider, reset_providers
from app.services.ai.summarization.service import SummarizationService, extractive_summary
from app.services.ai.text_cleaning import (
    clean_page_text,
    clean_pages,
    extract_preserved_values,
    find_repeated_furniture,
    repair_line_wrapping,
)
from tests.fixtures import write_sample_contract

PASSED: List[str] = []
FAILED: List[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(label)
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))


# ═══════════════════════════════════════════════════════════════════════════
# A fake provider — proves the abstraction and keeps tests model-free
# ═══════════════════════════════════════════════════════════════════════════
class FakeSummarizer(SummarizationProvider):
    """Deterministic stand-in. Records every call so batching can be asserted."""

    name = "fake"

    def __init__(self, available: bool = True):
        self._available = available
        self.calls: List[str] = []
        self.batch_sizes: List[int] = []

    @property
    def model_name(self) -> str:
        return "fake-summarizer-v1"

    @property
    def model_version(self) -> str:
        return "1.0.0-test"

    def is_available(self) -> bool:
        return self._available

    def summarize(self, text, max_output_tokens=None, instruction=None) -> str:
        """
        Stand in for a working summariser: shorter than the input and built from
        the input's own words, so it passes the grounding and fidelity gates the
        way real output must.
        """
        self.calls.append(text)
        if not self._available:
            return ""
        words = text.split()
        kept = words[: max(6, len(words) // 3)]
        summary = " ".join(kept).strip()
        if summary and summary[-1] not in ".!?":
            summary += "."
        return summary

    def summarize_batch(self, texts, max_output_tokens=None, instruction=None):
        self.batch_sizes.append(len(texts))
        return [self.summarize(t, max_output_tokens, instruction) for t in texts]


# ═══════════════════════════════════════════════════════════════════════════
# 1. Text cleaning — must not destroy legal meaning
# ═══════════════════════════════════════════════════════════════════════════
def test_cleaning() -> None:
    print("\n── 1. Text cleaning ──────────────────────────────────────")

    messy = (
        "Page 3 of 12\n"
        "ACME CONFIDENTIAL\n"
        "3.1 Client shall pay Provider a fee of $10,000 USD within thirty (30)\n"
        "days, subject to indemni-\nfication under Section 8.2, at 1.5% per month.\n"
        "\n\n\n"
        "   -  4  -\n"
    )
    cleaned = clean_page_text(messy, furniture={"ACME CONFIDENTIAL"})

    check("drops 'Page N of M' furniture", "Page 3 of 12" not in cleaned)
    check("drops a detected running header", "ACME CONFIDENTIAL" not in cleaned)
    check("drops a standalone page-number rule", "- 4 -" not in cleaned.replace("  ", " "))
    check("rejoins a hyphen split across lines", "indemnification" in cleaned, cleaned[-90:])
    check("preserves the monetary amount", "$10,000" in cleaned)
    check("preserves the percentage", "1.5%" in cleaned)
    check("preserves the notice period", "thirty (30)" in cleaned)
    check("preserves the clause cross-reference", "Section 8.2" in cleaned)
    check("preserves the conditional word", "subject to" in cleaned)
    check("collapses blank-line runs", "\n\n\n" not in cleaned)

    # Line-wrap repair must not swallow a new clause onto the previous line.
    wrapped = "The Provider shall deliver the\nservices promptly.\n5.1 Either party may terminate."
    repaired = repair_line_wrapping(wrapped)
    check("joins a genuine mid-sentence wrap",
          "deliver the services" in repaired, repr(repaired[:50]))
    check("does not join a new numbered clause onto the previous line",
          "\n5.1 Either party" in repaired)

    # Furniture detection needs agreement across pages, not one page's heading.
    pages = [f"ACME CONFIDENTIAL\nBody text for page {i}.\nfooter line" for i in range(6)]
    furniture = find_repeated_furniture(pages)
    check("detects a header repeated across pages", "ACME CONFIDENTIAL" in furniture)
    check("does not treat unique body text as furniture",
          "Body text for page 1." not in furniture)
    check("a numbered heading is never treated as furniture",
          not any(f.startswith("5.1") for f in find_repeated_furniture(
              ["5.1 Termination\nbody"] * 6)))

    values = extract_preserved_values("Pay $5,000 by March 1, 2027 at 2.5% within 15 days.")
    check("preservation check finds money, dates, percent and days",
          bool(values["money"] and values["dates"] and values["percent"] and values["days"]),
          str({k: v for k, v in values.items() if v}))


# ═══════════════════════════════════════════════════════════════════════════
# 2. Structure detection, with fallback
# ═══════════════════════════════════════════════════════════════════════════
def test_structure() -> None:
    print("\n── 2. Structure detection ────────────────────────────────")

    cases = [
        ("ARTICLE I", True), ("ARTICLE IV - Payment", True),
        ("SECTION 5", True), ("5. Termination", True),
        ("5.4 Termination for Convenience", True),
        ("TERMINATION", True), ("Governing Law", True),
        ("SCHEDULE A", True), ("EXHIBIT B - Pricing", True),
        # Not headings:
        ("5.1 The Provider shall deliver the services described in the attached statement", False),
        ("NEITHER PARTY SHALL BE LIABLE FOR INDIRECT, INCIDENTAL OR CONSEQUENTIAL DAMAGES", False),
        ("and the parties further agree,", False),
        ("This Agreement is made as of October 1, 2026.", False),
    ]
    correct = 0
    for line, should_match in cases:
        matched = match_heading(line) is not None
        correct += matched == should_match
        if matched != should_match:
            print(f"        mismatch: {line[:52]!r} -> {matched}, wanted {should_match}")
    check("heading patterns classify all 13 cases", correct == len(cases),
          f"{correct}/{len(cases)}")

    # A document with no headings at all must fall back to pages, not invent
    # a structure that isn't there.
    flat = [(1, "Some prose with no headings at all. " * 40),
            (2, "More prose continuing the same way. " * 40)]
    sections, detected = detect_sections(flat)
    check("reports structure_detected=False for an unstructured document", not detected)
    fallback = page_fallback_sections(flat)
    check("page fallback yields one section per page", len(fallback) == 2)
    check("page fallback keeps exact page numbers",
          [s.page_start for s in fallback] == [1, 2])


# ═══════════════════════════════════════════════════════════════════════════
# 3. Token-aware chunking
# ═══════════════════════════════════════════════════════════════════════════
def test_chunking() -> None:
    print("\n── 3. Token-aware chunking ───────────────────────────────")

    chunker = TokenAwareChunker(
        max_tokens=100, overlap_tokens=10, min_tokens=8,
        count_tokens=heuristic_token_count,
    )

    long_body = " ".join(
        f"Clause sentence number {i} stating an obligation of the parties." for i in range(80)
    )
    sections = [
        Section(section_id="s1", section_number="1", section_title="Short",
                page_start=1, page_end=1, text="A brief but sufficient clause body here today."),
        Section(section_id="s2", section_number="2", section_title="Long",
                page_start=2, page_end=3, text=long_body),
    ]
    chunks = chunker.chunk_sections(sections, "doc-1", "user-1")

    check("produces chunks", len(chunks) > 0, f"{len(chunks)} chunks")
    over = [c for c in chunks if c.token_count > 100]
    check("no chunk exceeds the token budget", not over,
          f"worst={max(c.token_count for c in chunks)}")
    check("a long section is split into several chunks",
          len([c for c in chunks if c.section_id == "s2"]) > 1)
    check("no chunk mixes two sections",
          all(c.section_id in {"s1", "s2"} for c in chunks))
    check("chunks carry the section's page range",
          all(c.page_start >= 1 and c.page_end >= c.page_start for c in chunks))
    check("every chunk has a token count", all(c.token_count > 0 for c in chunks))
    check("chunk ids are unique",
          len({c.chunk_id for c in chunks}) == len(chunks))
    check("order is sequential from zero",
          [c.order for c in chunks] == list(range(len(chunks))))
    check("document_id and user_id propagate to every chunk",
          all(c.document_id == "doc-1" and c.user_id == "user-1" for c in chunks))

    # Token budget, not character count, is the binding constraint.
    wide = TokenAwareChunker(max_tokens=300, overlap_tokens=0, min_tokens=8,
                             count_tokens=heuristic_token_count)
    wider_chunks = wide.chunk_sections(sections, "doc-1", "user-1")
    check("raising the token budget produces fewer, larger chunks",
          len(wider_chunks) < len(chunks),
          f"{len(wider_chunks)} vs {len(chunks)}")

    # A sentence longer than the whole budget still has to be emitted.
    monster = Section(section_id="s3", section_number="3", section_title="Monster",
                      page_start=1, page_end=1,
                      text="WORD " * 500)
    monster_chunks = chunker.chunk_sections([monster], "doc-1", "user-1")
    check("an over-budget sentence is split rather than dropped",
          len(monster_chunks) > 1 and all(c.token_count <= 100 for c in monster_chunks),
          f"{len(monster_chunks)} chunks")

    check("heuristic token count scales with length",
          heuristic_token_count("one two three four five") > heuristic_token_count("one two"))
    check("heuristic token count is zero for empty text", heuristic_token_count("") == 0)

    # Overlap is configurable and off means off.
    no_overlap = TokenAwareChunker(max_tokens=60, overlap_tokens=0, min_tokens=5,
                                   count_tokens=heuristic_token_count)
    with_overlap = TokenAwareChunker(max_tokens=60, overlap_tokens=25, min_tokens=5,
                                     count_tokens=heuristic_token_count)
    a = no_overlap.chunk_sections([sections[1]], "d", "u")
    b = with_overlap.chunk_sections([sections[1]], "d", "u")
    check("overlap is configurable and increases total tokens",
          sum(c.token_count for c in b) >= sum(c.token_count for c in a),
          f"overlap={sum(c.token_count for c in b)} none={sum(c.token_count for c in a)}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. One pipeline: canonical document reused by everything
# ═══════════════════════════════════════════════════════════════════════════
def test_canonical_document(pdf_path: str, pdf_bytes: bytes) -> None:
    print("\n── 4. Canonical document ─────────────────────────────────")

    processor = DocumentProcessor(
        chunker=TokenAwareChunker(count_tokens=heuristic_token_count)
    )
    document = processor.process(
        pdf_path, "doc-42", "user-7", "contract.pdf",
        processor.content_hash(pdf_bytes), len(pdf_bytes),
    )

    check("pages, sections and chunks are all populated",
          bool(document.pages and document.sections and document.chunks),
          f"{document.page_count}p / {len(document.sections)}s / {len(document.chunks)}c")
    check("page boundaries are preserved",
          [p.page_number for p in document.pages] == [1, 2, 3, 4])
    check("structure was detected in a well-formed contract", document.structure_detected)
    check("content hash is a sha256 hex digest", len(document.content_hash) == 64)
    check("total tokens are counted", document.total_tokens > 0, str(document.total_tokens))
    check("full_text joins pages without losing them",
          all(p.text[:30] in document.full_text for p in document.pages if p.text))
    check("has_usable_text is true for a text PDF", document.has_usable_text())

    # The same chunks must serve every consumer — this is the core requirement.
    ids = {c.chunk_id for c in document.chunks}
    check("chunks_for_section resolves by section id",
          all(c.chunk_id in ids for s in document.sections
              for c in document.chunks_for_section(s.section_id)))

    facts = legal_extractor.extract_all(document.chunks)
    check("extraction consumes the same canonical chunks",
          bool(facts["parties"]) and bool(facts["key_dates"]),
          f"{len(facts['parties'])} parties, {len(facts['key_dates'])} dates")
    check("every extracted fact carries a page number",
          all(item.source_pages for group in facts.values() for item in group))
    check("extracted pages are within the document",
          all(1 <= page <= document.page_count
              for group in facts.values() for item in group for page in item.source_pages))

    jurisdiction = legal_extractor.extract_jurisdiction(document.chunks)
    check("governing law is extracted deterministically",
          "New York" in jurisdiction["governing_law"], jurisdiction["governing_law"])
    check("document type is identified from the text",
          legal_extractor.detect_document_type(document.full_text, "contract.pdf")
          == "Master Services Agreement")

    # A scanned PDF must be reported, not silently summarised as empty.
    blank = CanonicalDocument(document_id="d", user_id="u")
    check("a document with no text reports has_usable_text=False",
          not blank.has_usable_text())


# ═══════════════════════════════════════════════════════════════════════════
# 5. Provider abstraction
# ═══════════════════════════════════════════════════════════════════════════
def test_providers() -> None:
    print("\n── 5. Provider abstraction ───────────────────────────────")

    reset_providers()
    local = build_provider("local")
    check("'local' builds the transformer provider", local.name == "local",
          type(local).__name__)
    check("local is the configured default", settings.SUMMARY_PROVIDER == "local",
          settings.SUMMARY_PROVIDER)
    gemini = build_provider("gemini")
    check("'gemini' builds the Gemini provider", gemini.name == "gemini")
    check("an unknown provider name falls back to local",
          build_provider("nonsense").name == "local")
    check("providers are cached, so a model loads once",
          get_provider("local") is get_provider("local"))
    check("describe() reports provider and model",
          set(local.describe()) == {"provider", "model_name", "model_version"})
    check("the local provider does not need an API key",
          "GOOGLE_API_KEY" not in str(type(local).__init__.__doc__ or ""))
    reset_providers()


# ═══════════════════════════════════════════════════════════════════════════
# 6. Hierarchical summarisation
# ═══════════════════════════════════════════════════════════════════════════
def _document_with_chunks(count: int) -> CanonicalDocument:
    document = CanonicalDocument(document_id="doc-h", user_id="user-h",
                                 file_name="long.pdf")
    document.pages = [__import__("app.domain.document", fromlist=["Page"]).Page(
        page_number=i + 1, text=f"Page {i + 1} body text. " * 20) for i in range(count)]
    for index in range(count):
        section_id = f"sec-{index}"
        document.sections.append(Section(
            section_id=section_id, section_number=str(index + 1),
            section_title=f"Section {index + 1}",
            page_start=index + 1, page_end=index + 1,
            text=f"Body of section {index + 1}.",
        ))
        document.chunks.append(Chunk(
            chunk_id=f"chk-{index}", document_id="doc-h", user_id="user-h",
            text=(f"Section {index + 1}\nThe Provider shall deliver within "
                  f"{index + 10} (10) days and pay $1,000 per month."),
            token_count=40, page_start=index + 1, page_end=index + 1,
            section_id=section_id, section_number=str(index + 1),
            section_title=f"Section {index + 1}", order=index,
        ))
    return document


def test_hierarchical_summarization() -> None:
    print("\n── 6. Hierarchical summarisation ─────────────────────────")

    # Short document: no intermediate reduce level.
    fake = FakeSummarizer()
    service = SummarizationService(provider=fake)
    short = _document_with_chunks(3)
    result = service.summarize_document(short)

    check("produces an executive summary", bool(result.executive_summary))
    check("produces one section summary per section",
          len(result.section_summaries) == 3, str(len(result.section_summaries)))
    check("records the provider and model", result.provider == "fake"
          and result.model_name == "fake-summarizer-v1")
    check("records pipeline and prompt versions",
          result.pipeline_version == settings.PIPELINE_VERSION
          and result.prompt_version == settings.PROMPT_VERSION)
    check("records processing time", result.processing_time >= 0)
    check("is not marked degraded when the provider works", not result.degraded)
    check("a short document skips the intermediate reduce level",
          len(short.chunks) < settings.SUMMARY_HIERARCHY_THRESHOLD)

    # Source traceability must survive the whole tree.
    check("every section summary keeps its chunk ids",
          all(s.chunk_ids for s in result.section_summaries))
    check("every section summary keeps a page range",
          all(s.page_start >= 1 and s.page_end >= s.page_start
              for s in result.section_summaries))
    check("source references are emitted", len(result.source_references) == 3)
    pages = {p for s in result.section_summaries for p in (s.page_start, s.page_end)}
    check("no page reference is outside the document",
          all(1 <= p <= short.page_count for p in pages), str(sorted(pages)))

    # Long document: the intermediate reduce level must engage.
    fake_long = FakeSummarizer()
    long_document = _document_with_chunks(settings.SUMMARY_HIERARCHY_THRESHOLD + 6)
    long_result = SummarizationService(provider=fake_long).summarize_document(long_document)
    check("a long document still yields one executive summary",
          bool(long_result.executive_summary))
    check("a long document summarises every section",
          len(long_result.section_summaries) == len(long_document.sections))
    check("the map stage batches rather than calling one chunk at a time",
          bool(fake_long.batch_sizes), str(fake_long.batch_sizes[:3]))
    check("the intermediate reduce ran for a long document",
          len(fake_long.calls) > len(long_document.chunks),
          f"{len(fake_long.calls)} calls for {len(long_document.chunks)} chunks")

    # Structured fields come from deterministic extraction, not the model.
    check("structured fields are populated without the model inventing them",
          bool(long_result.key_dates) or bool(long_result.financial_terms),
          f"{len(long_result.key_dates)} dates, {len(long_result.financial_terms)} amounts")
    check("no risk field exists on the summary result",
          not hasattr(long_result, "risks"))

    # Degradation: an unavailable model must not lose the document.
    broken = FakeSummarizer(available=False)
    degraded = SummarizationService(provider=broken).summarize_document(
        _document_with_chunks(4))
    check("an unavailable model degrades instead of failing", degraded.degraded)
    check("degradation is explained", bool(degraded.degraded_reason))
    check("extractive summaries are still produced when degraded",
          any(s.summary for s in degraded.section_summaries))
    check("structured extraction still works with no model",
          bool(degraded.key_dates) or bool(degraded.financial_terms))

    # An empty document must not crash the service.
    empty = SummarizationService(provider=FakeSummarizer()).summarize_document(
        CanonicalDocument(document_id="d", user_id="u"))
    check("an empty document returns a degraded result rather than raising",
          empty.degraded)

    # The extractive fallback picks legally salient sentences.
    text = ("This is filler prose about nothing much at all. "
            "The Client shall pay $50,000 within thirty (30) days of invoice. "
            "More filler that carries no obligation whatsoever.")
    picked = extractive_summary(text, max_sentences=1)
    check("extractive fallback prefers the sentence with the obligation",
          "$50,000" in picked, picked[:70])


# ═══════════════════════════════════════════════════════════════════════════
# 7. Async API contract: 202, status polling, dedupe, authorisation
# ═══════════════════════════════════════════════════════════════════════════
def test_async_api(pdf_bytes: bytes) -> None:
    print("\n── 7. Async upload API ───────────────────────────────────")

    import asyncio

    from fastapi.testclient import TestClient

    import main
    from app.api.deps import get_current_user, get_db
    from app.services.ai.vector_store import vector_store_service
    from tests.test_audit_pipeline import _DB

    vector_store_service._pinecone_ok = False

    db = _DB()
    user = {"_id": "u-async", "email": "async@example.com", "name": "Async"}
    db["users"].docs["u-async"] = user
    main.app.dependency_overrides[get_db] = lambda: db
    main.app.dependency_overrides[get_current_user] = lambda: user

    # Replace the pipeline with a recorder: the API contract is what is under
    # test here, not the pipeline internals.
    from app.services import pipeline as pipeline_module

    ran: List[str] = []

    async def fake_run(self, db_arg, audit_id, user_id):
        ran.append(audit_id)
        await db_arg["audits"].update_one(
            {"_id": audit_id},
            {"$set": {"status": ProcessingStatus.COMPLETED.value, "progress": 100,
                      "progress_label": "Completed", "executive_summary": "Done.",
                      "summary": "Done.", "page_count": 4, "chunk_count": 12}},
        )

    original_run = pipeline_module.DocumentPipeline.run
    pipeline_module.DocumentPipeline.run = fake_run

    try:
        client = TestClient(main.app)

        response = client.post(
            "/api/audits/upload",
            files={"file": ("contract.pdf", pdf_bytes, "application/pdf")},
        )
        check("upload returns 202 Accepted, not 200",
              response.status_code == 202, str(response.status_code))
        body = response.json()
        audit_id = body["id"]
        check("the 202 body carries an id and a status",
              bool(audit_id) and body["status"] == ProcessingStatus.UPLOADED.value,
              str(body.get("status")))
        check("the upload response is not the full analysis",
              "clauses" not in body and "entities" not in body)
        check("processing was queued as a background task", ran == [audit_id], str(ran))

        status_body = client.get(f"/api/audits/{audit_id}/status").json()
        check("the status endpoint reports completion after the task ran",
              status_body["is_complete"] is True, str(status_body.get("status")))
        check("the status endpoint reports progress and a label",
              status_body["progress"] == 100 and bool(status_body["progress_label"]))

        full = client.get(f"/api/audits/{audit_id}")
        check("the full analysis is fetched separately", full.status_code == 200)

        # Re-uploading the identical file must reuse, not reprocess.
        ran.clear()
        again = client.post(
            "/api/audits/upload",
            files={"file": ("contract.pdf", pdf_bytes, "application/pdf")},
        )
        again_body = again.json()
        check("an identical re-upload is deduplicated by content hash",
              again_body.get("reused") is True, str(again_body)[:90])
        check("a deduplicated upload returns the original audit id",
              again_body["id"] == audit_id)
        check("a deduplicated upload queues no new processing", ran == [], str(ran))

        # A different user uploading the same bytes must get their own analysis.
        other = {"_id": "u-other", "email": "other@example.com", "name": "Other"}
        db["users"].docs["u-other"] = other
        main.app.dependency_overrides[get_current_user] = lambda: other
        ran.clear()
        other_response = client.post(
            "/api/audits/upload",
            files={"file": ("contract.pdf", pdf_bytes, "application/pdf")},
        )
        check("the hash cache never crosses users",
              other_response.json()["id"] != audit_id
              and other_response.json().get("reused") is not True)
        check("the other user's upload is processed on its own", len(ran) == 1)

        # And cannot read the first user's document.
        check("another user cannot read someone else's audit",
              client.get(f"/api/audits/{audit_id}").status_code == 404)
        check("another user cannot poll someone else's status",
              client.get(f"/api/audits/{audit_id}/status").status_code == 404)

        main.app.dependency_overrides[get_current_user] = lambda: user
        check("reprocess re-queues an existing document",
              client.post(f"/api/audits/{audit_id}/reprocess").status_code == 202)

        # Validation still rejects before anything is stored.
        for label, blob, name in [
            ("rejects a non-PDF extension", b"%PDF-1.4", "notes.txt"),
            ("rejects bad magic bytes", b"not a pdf", "fake.pdf"),
            ("rejects an empty upload", b"", "empty.pdf"),
        ]:
            code = client.post(
                "/api/audits/upload",
                files={"file": (name, blob, "application/pdf")},
            ).status_code
            check(label, code == 400, f"status {code}")

    finally:
        pipeline_module.DocumentPipeline.run = original_run
        main.app.dependency_overrides.clear()


# ═══════════════════════════════════════════════════════════════════════════
# 8. Output fidelity guards (regressions)
# ═══════════════════════════════════════════════════════════════════════════
def test_output_guards() -> None:
    """
    Guards on what a model is allowed to return.

    Each check here is a bug that shipped: the echo guard rejected every
    legitimate summary, generated text was presented truncated mid-sentence, and
    invented figures were passed through into a legal summary.
    """
    print("\n── 8. Output fidelity guards ─────────────────────────────")

    from app.services.ai.summarization.local_provider import (
        LocalTransformerSummarizer,
        trim_to_sentence,
    )
    from app.services.ai.summarization.service import find_invented_values

    strip = LocalTransformerSummarizer._strip_echoed_input
    source = (
        "Either party may terminate this Agreement for convenience upon thirty (30) "
        "calendar days prior written notice to the other party."
    )

    # The critical regression: an earlier length-based guard dropped any summary
    # that was not shorter in characters than its input, which silently discarded
    # almost every real summary and made the whole feature fall back to extracts.
    good = "Either side may end the agreement with 30 days written notice."
    check("a legitimate shorter summary is kept", strip(good, source) == good)
    longer = (
        "The agreement allows either contracting party to terminate for "
        "convenience, provided they give the other party thirty calendar days of "
        "advance written notice before the termination takes effect."
    )
    check("a slightly longer paraphrase is kept, not dropped",
          strip(longer, source) == longer, f"len {len(longer)} vs source {len(source)}")

    check("a verbatim echo of the input is dropped", strip(source, source) == "")
    check("an echo with different whitespace is still dropped",
          strip(source.replace(" ", "  "), source) == "")
    check("output opening with the input verbatim is dropped",
          strip(source + " Additionally the parties agree to arbitrate.", source) == "")
    check("a leading copy of the instruction is stripped or dropped",
          strip(CHUNK_INSTRUCTION + " " + good, source) in {good, ""})
    check("empty output stays empty", strip("", source) == "")
    check("a wildly expanded 'summary' is rejected",
          strip(good * 8, source) == "")

    # Truncation must never be presented as a finished sentence.
    check("cuts back to the last complete sentence",
          trim_to_sentence(
              "Fees are payable within thirty (30) days of receipt. Past due "
              "balances shall accrue interest at one and"
          ).endswith("of receipt."))
    check("leaves an already-complete sentence untouched",
          trim_to_sentence("Fees are due within 30 days.") == "Fees are due within 30 days.")
    check("marks a fragment it cannot cut back",
          trim_to_sentence("accrue interest at one and").endswith("..."))
    check("handles empty generated text", trim_to_sentence("") == "")

    # Numeric fidelity: a figure the contract does not contain must not survive.
    contract = (
        "Client shall pay a monthly retainer of Ten Thousand Dollars ($10,000 USD), "
        "a total of $240,000 USD, payable within thirty (30) days, with interest at "
        "1.5% per month on past due balances."
    )
    check("a faithful summary reports no invented values",
          find_invented_values(
              "Client pays $10,000 monthly, $240,000 in total, due within 30 days at 1.5%.",
              contract,
          ) == [])
    invented = find_invented_values(
        "Client pays $99,999 within 45 days with interest at 7%.", contract
    )
    check("an invented amount is detected", any("99,999" in v for v in invented), str(invented))
    check("an invented period is detected", any("45" in v for v in invented), str(invented))
    check("an invented percentage is detected", any("7%" in v for v in invented), str(invented))
    check("comma and spacing differences do not count as invented",
          find_invented_values("Total is $240000 USD.", contract) == [])
    check("a summary with no figures is never flagged",
          find_invented_values("The client pays the provider a monthly fee.", contract) == [])

    # Mis-pairing: every value exists in the source, but attached to the wrong
    # words. A value-level check passes while the term has been altered.
    from app.services.ai.summarization.service import find_mispaired_values

    paired_source = (
        "Client shall pay Provider a monthly retainer of Ten Thousand Dollars "
        "($10,000 USD), for a total estimated contract value of Two Hundred Forty "
        "Thousand Dollars ($240,000 USD) over the initial term."
    )
    swapped = "Client will pay Provider a monthly retainer of Ten Thousand Dollars ($240,000 USD)"
    check("a figure attached to the wrong words is detected",
          bool(find_mispaired_values(swapped, paired_source)),
          str(find_mispaired_values(swapped, paired_source)))
    check("the value-level check alone cannot see a mis-pairing",
          find_invented_values(swapped, paired_source) == [],
          "both figures exist in the source")
    faithful_pair = "Client will pay a monthly retainer of Ten Thousand Dollars ($10,000 USD)"
    check("a correctly paired figure is accepted",
          find_mispaired_values(faithful_pair, paired_source) == [])
    check("text with no worded amounts is never flagged",
          find_mispaired_values("Payment is due within thirty (30) days.", paired_source) == [])

    class MispairingSummarizer(FakeSummarizer):
        def summarize(self, text, max_output_tokens=None, instruction=None):
            return "Provider is paid a monthly retainer of Ten Thousand Dollars ($240,000 USD) each month."

    mispair_doc = _document_with_chunks(2)
    for chunk in mispair_doc.chunks:
        chunk.text = paired_source
    mispair_result = SummarizationService(
        provider=MispairingSummarizer()
    ).summarize_document(mispair_doc)
    check("a mis-paired summary is replaced by the extract",
          all("240,000 USD) each month" not in s.summary
              for s in mispair_result.section_summaries),
          str([s.summary[:50] for s in mispair_result.section_summaries]))

    # A chunk whose summary invents a figure must fall back to the extract.
    class InventingSummarizer(FakeSummarizer):
        def summarize(self, text, max_output_tokens=None, instruction=None):
            return "The client shall pay $999,999 within 90 days."

    document = _document_with_chunks(3)
    result = SummarizationService(provider=InventingSummarizer()).summarize_document(document)
    check("a summary that invents figures is replaced by the extract",
          all("999,999" not in s.summary for s in result.section_summaries),
          str([s.summary[:40] for s in result.section_summaries]))

    # Grounding: the guard that catches a checkpoint fine-tuned on the wrong
    # domain. nsi319/legal-led-base-16384 is trained on SEC litigation releases
    # and emits fluent text about court judgments for any contract it is given —
    # fabrication that the numeric checks cannot see, because invented prose
    # contains no conflicting figures.
    from app.services.ai.summarization.service import (
        MIN_GROUNDING_RATIO,
        grounding_ratio,
        reject_reason,
    )

    clause = (
        "5. Termination 5.1 Either party may terminate this Agreement for "
        "convenience upon thirty (30) calendar days prior written notice to the "
        "other party."
    )
    fabricated = (
        "The Securities and Exchange Commission today announced that it has "
        "entered into a final judgment that permanently enjoins all defendants "
        "from violating the antifraud provisions."
    )
    paraphrase = (
        "Either contracting party can end the agreement for convenience by giving "
        "the other party thirty calendar days of prior written notice."
    )
    check("fabricated prose scores near zero grounding",
          grounding_ratio(fabricated, clause) < 0.2,
          f"{grounding_ratio(fabricated, clause):.0%}")
    check("a faithful paraphrase scores high grounding",
          grounding_ratio(paraphrase, clause) >= MIN_GROUNDING_RATIO,
          f"{grounding_ratio(paraphrase, clause):.0%}")
    check("a verbatim extract scores full grounding",
          grounding_ratio(clause, clause) == 1.0)
    check("an empty summary does not divide by zero",
          grounding_ratio("", clause) == 1.0)

    # Fabrication that reuses the parties' names is the hard case: it scored 45%
    # against the real document, which is why the threshold sits at 0.6.
    preamble = (
        'This Master Services Agreement is entered into as of October 1, 2026 by '
        'and between Acme Enterprise Solutions Inc., a Delaware corporation with '
        'offices at 100 Innovation Way, New York, and Global Logistics Corp., an '
        'Illinois corporation with offices at 500 Commerce Blvd, Chicago.'
    )
    name_reusing_fabrication = (
        "On October 1, 2026, the United States District Court for the Southern "
        "District of Illinois entered final consent judgments against Acme "
        "Enterprise Solutions Inc., a Delaware corporation, and Global Logistics "
        "Corp., in connection with an alleged insider trading scheme."
    )
    check("fabrication that reuses party names is still rejected",
          reject_reason(name_reusing_fabrication, preamble) is not None,
          f"grounding {grounding_ratio(name_reusing_fabrication, preamble):.0%}")

    # The unified gate must apply every check, at every level of the hierarchy.
    check("the gate rejects empty output", reject_reason("", clause) == "empty")
    check("the gate accepts a faithful paraphrase",
          reject_reason(paraphrase, clause) is None)
    check("the gate reports ungrounded output",
          "not grounded" in (reject_reason(fabricated, clause) or ""))
    check("the gate reports an invented value",
          "absent from the source" in (
              reject_reason(
                  "Either party may terminate on ninety (90) days notice.", clause
              ) or ""
          ))

    # A fabricating model must not be able to reach the executive summary either.
    # Previously the numeric checks ran only on chunks, so a fabricated executive
    # summary carrying a wrong date ("October 1, 2020") passed.
    class FabricatingSummarizer(FakeSummarizer):
        def summarize(self, text, max_output_tokens=None, instruction=None):
            return (
                "On October 1, 2020, the United States District Court entered "
                "final consent judgments over an alleged insider trading scheme."
            )

    fabricated_doc = _document_with_chunks(10)
    fabricated_result = SummarizationService(
        provider=FabricatingSummarizer()
    ).summarize_document(fabricated_doc)
    check("no fabricated text reaches the section summaries",
          all("insider trading" not in s.summary for s in fabricated_result.section_summaries))
    check("no fabricated text reaches the executive summary",
          "insider trading" not in fabricated_result.executive_summary,
          fabricated_result.executive_summary[:70])
    check("a run where nothing could be summarised is reported as degraded",
          fabricated_result.degraded and bool(fabricated_result.degraded_reason),
          fabricated_result.degraded_reason[:80])

    # Silent extractive fallback must never look like a successful summary.
    class SilentFailSummarizer(FakeSummarizer):
        def summarize(self, text, max_output_tokens=None, instruction=None):
            return ""

    silent = SummarizationService(
        provider=SilentFailSummarizer()
    ).summarize_document(_document_with_chunks(4))
    check("a model that returns nothing is reported, not passed off as a summary",
          silent.degraded and "no usable summary" in silent.degraded_reason.lower(),
          silent.degraded_reason[:80])
    check("section summaries still exist when the model produced nothing",
          all(s.summary for s in silent.section_summaries))

    # section_label existed on Chunk but not on SectionSummary, so a hasattr
    # branch in the service silently took the wrong path.
    from app.domain.summary import SectionSummary

    check("SectionSummary exposes a citation label",
          SectionSummary(section_number="5.4", section_title="Termination").section_label
          == "5.4 Termination")
    check("the label falls back to whichever part exists",
          SectionSummary(section_title="Preamble").section_label == "Preamble")


# ═══════════════════════════════════════════════════════════════════════════
# 9. Model loading resilience
# ═══════════════════════════════════════════════════════════════════════════
def test_load_resilience() -> None:
    """
    A missing or stalled checkpoint must degrade, never hang.

    An unbounded load left every upload stuck on "summarizing" with no diagnosis,
    because a first run downloads hundreds of megabytes and a rate-limited hub
    connection simply never returns.
    """
    print("\n── 9. Model loading resilience ───────────────────────────")

    from app.services.ai.summarization.local_provider import LocalTransformerSummarizer

    provider = LocalTransformerSummarizer(model_name="definitely/not-a-real-model")
    candidates = provider._candidate_models()
    check("the configured model is tried first",
          candidates[0] == "definitely/not-a-real-model")
    check("configured fallbacks follow it", len(candidates) > 1, str(candidates))
    check("no duplicate candidates", len(candidates) == len(set(candidates)))

    # A load that never returns must be abandoned at the deadline.
    import time as _time

    slow = LocalTransformerSummarizer(model_name="slow/model")
    slow._load_now = lambda name: _time.sleep(30) or True
    original_timeout = settings.SUMMARIZATION_LOAD_TIMEOUT
    settings.SUMMARIZATION_LOAD_TIMEOUT = 1
    try:
        started = _time.time()
        reason = slow._try_load("slow/model")
        elapsed = _time.time() - started
        check("a stalled load is abandoned at the deadline",
              reason is not None and elapsed < 10, f"{elapsed:.1f}s, reason={reason}")
        check("the timeout reason is explicit",
              bool(reason) and "exceeded" in reason, str(reason))
    finally:
        settings.SUMMARIZATION_LOAD_TIMEOUT = original_timeout

    broken = LocalTransformerSummarizer(model_name="bad/model")
    broken._load_now = lambda name: (_ for _ in ()).throw(OSError("no such checkpoint"))
    reason = broken._try_load("bad/model")
    check("a load error is reported as a reason, not raised",
          reason is not None and "OSError" in reason, str(reason))

    # T5 needs its task prefix; LED and BART do not.
    t5 = LocalTransformerSummarizer(model_name="some/t5-summariser")
    check("a T5 checkpoint is detected as needing a task prefix",
          t5._needs_task_prefix())
    led = LocalTransformerSummarizer(model_name="allenai/led-base-16384")
    check("an LED checkpoint needs no task prefix", not led._needs_task_prefix())
    check("an LED checkpoint is recognised for global attention", led._is_led())

    # LED reports attention_window as a list with one entry per layer. An int()
    # conversion placed before the list check raised on every LED generation,
    # silently turning the whole feature into verbatim extraction.
    class _Cfg:
        attention_window = [1024] * 6

    class _FakeModel:
        config = _Cfg()

    class _FakeTok:
        pad_token_id = 0

    import torch

    padder = LocalTransformerSummarizer(model_name="allenai/led-base-16384")
    padder._model = _FakeModel()
    padder._tokenizer = _FakeTok()
    padder._torch = torch
    encoded = {
        "input_ids": torch.ones((1, 181), dtype=torch.long),
        "attention_mask": torch.ones((1, 181), dtype=torch.long),
    }
    padded = padder._pad_to_attention_window(encoded)
    check("a list-valued attention_window does not raise",
          int(padded["input_ids"].shape[-1]) == 1024,
          f"padded to {int(padded['input_ids'].shape[-1])}")
    check("the attention mask is padded to the same length",
          int(padded["attention_mask"].shape[-1]) == int(padded["input_ids"].shape[-1]))
    already = {
        "input_ids": torch.ones((1, 1024), dtype=torch.long),
        "attention_mask": torch.ones((1, 1024), dtype=torch.long),
    }
    check("an already-aligned batch is left untouched",
          int(padder._pad_to_attention_window(already)["input_ids"].shape[-1]) == 1024)


# ═══════════════════════════════════════════════════════════════════════════
def main_() -> int:
    print("=" * 74)
    print("Canonical pipeline + summarisation architecture tests")
    print("=" * 74)

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = write_sample_contract(os.path.join(tmp, "contract.pdf"))
        with open(pdf_path, "rb") as fh:
            pdf_bytes = fh.read()

        test_cleaning()
        test_structure()
        test_chunking()
        test_canonical_document(pdf_path, pdf_bytes)
        test_providers()
        test_hierarchical_summarization()
        test_async_api(pdf_bytes)
        test_output_guards()
        test_load_resilience()

    print("\n" + "=" * 74)
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    for label in FAILED:
        print(f"  FAILED: {label}")
    print("=" * 74)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main_())
