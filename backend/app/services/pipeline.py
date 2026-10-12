"""
DocumentPipeline — the one orchestrator for processing an uploaded contract.

    validate → store → extract → clean → structure → chunk
                                            │
                        ┌───────────────────┼───────────────────┐
                        ▼                   ▼                   ▼
                  SUMMARIZATION        RAG INDEX        CLAUSE / RISK ANALYSIS
                        │                   │                   │
                        └────────── MongoDB persistence ────────┘

The canonical chunks are produced exactly once and handed to all three
consumers. Nothing here re-reads the PDF or re-chunks the text.

This runs in the background, not inside the upload request: summarising a long
contract takes far longer than an HTTP request should. The upload returns as soon
as the file is stored, and the pipeline reports progress through the document
record, which the frontend polls.
"""

from __future__ import annotations

import os
import tempfile
import time
from datetime import datetime
from typing import Dict, Optional

from app.core.config import settings
from app.core.logging import logger
from app.domain.document import STAGE_PROGRESS, CanonicalDocument, ProcessingStatus
from app.domain.summary import SummaryResult
from app.services.ai.extraction import legal_extractor
from app.services.ai.clauses import detect_clauses
from app.services.ai.processor import DocumentProcessor, classify_clause_type
from app.services.ai.risk import analyse_risk
from app.services.ai.summarization.service import SummarizationService
from app.services.ai.vector_store import vector_store_service
from app.services.storage_service import storage_service

# The optional LLM enrichment pass adds explanation text to clauses the
# deterministic analysis already found. It is off by default: the analysis is
# complete without it, and the Gemini SDK's call cannot be interrupted once it
# begins backing off against a rate limit, which would otherwise hold a document
# in "analyzing" long past any deadline set around it.
ENABLE_LLM_ENRICHMENT = os.getenv("ENABLE_LLM_ENRICHMENT", "false").lower() == "true"
LLM_ENRICHMENT_TIMEOUT = float(os.getenv("LLM_ENRICHMENT_TIMEOUT", "45"))


class DocumentPipeline:
    """
    Runs the full analysis for one document and keeps its status current.

    Every stage updates the stored record before it starts, so a client polling
    the status endpoint sees real progress and a crash leaves a `failed` record
    rather than one stuck on "processing" forever.
    """

    def __init__(
        self,
        processor: Optional[DocumentProcessor] = None,
        summarizer: Optional[SummarizationService] = None,
    ):
        self.processor = processor or DocumentProcessor()
        self._summarizer = summarizer

    @property
    def summarizer(self) -> SummarizationService:
        if self._summarizer is None:
            from app.services.ai.summarization.service import summarization_service

            self._summarizer = summarization_service
        return self._summarizer

    # ────────────────────────────────────────────────────────────────────
    async def run(self, db, audit_id: str, user_id: str) -> None:
        """
        Process a document that has already been validated and stored.

        Reads the record created by the upload route, runs every stage, and
        writes the results back. Exceptions are caught and recorded as a failed
        status — a background task that raises would otherwise vanish silently.
        """
        started = time.time()
        record = await db["audits"].find_one({"_id": audit_id, "user_id": user_id})
        if not record:
            logger.error(f"[Pipeline] {audit_id}: no record to process.")
            return

        storage_key = record.get("storage_key") or f"{audit_id}.pdf"
        file_name = record.get("file_name", "document.pdf")

        try:
            if not storage_service.exists(storage_key):
                await self._fail(db, audit_id, "The stored PDF could not be found.")
                return

            # ── Extract / clean / structure / chunk ──────────────────────
            await self._set_status(db, audit_id, ProcessingStatus.EXTRACTING)
            source_path = storage_service.get_absolute_path(storage_key)
            document = self.processor.process(
                file_path=source_path,
                document_id=audit_id,
                user_id=user_id,
                file_name=file_name,
                content_hash=record.get("content_hash", ""),
                file_size=record.get("file_size", 0),
            )

            # A scanned PDF is reported precisely, never as an empty success.
            if document.status == ProcessingStatus.OCR_REQUIRED:
                await self._set_status(
                    db, audit_id, ProcessingStatus.OCR_REQUIRED,
                    extra={
                        "page_count": document.page_count,
                        "error": (
                            "No readable text layer was found in this PDF. It appears "
                            "to be a scanned image; OCR is not yet supported."
                        ),
                    },
                )
                return

            if not document.chunks:
                await self._fail(db, audit_id, "The document contained too little text to analyse.")
                return

            await self._set_status(
                db, audit_id, ProcessingStatus.CHUNKING,
                extra={
                    "page_count": document.page_count,
                    "section_count": len(document.sections),
                    "chunk_count": len(document.chunks),
                    "token_count": document.total_tokens,
                    "structure_detected": document.structure_detected,
                },
            )

            # ── Clause typing: cheap, deterministic, shared downstream ───
            clause_types: Dict[str, str] = {
                chunk.chunk_id: classify_clause_type(chunk.text, chunk.section_title)
                for chunk in document.chunks
            }
            for chunk in document.chunks:
                chunk.clause_type = clause_types[chunk.chunk_id]

            # ── RAG index, from the same canonical chunks ────────────────
            await self._set_status(db, audit_id, ProcessingStatus.INDEXING)
            index_result = vector_store_service.index_document(document)

            # ── Summarisation (local model by default) ───────────────────
            await self._set_status(db, audit_id, ProcessingStatus.SUMMARIZING)
            summary = await self._summarize(db, audit_id, document, clause_types)

            # ── Legal analysis: deterministic first, LLM as enrichment ───
            await self._set_status(db, audit_id, ProcessingStatus.ANALYZING)
            detected_clauses = detect_clauses(document.chunks)
            risk = analyse_risk(
                document.chunks,
                detected_clauses,
                document_type=summary.document_type or "",
            )
            # The LLM pass is enrichment only. The deterministic analysis above
            # is already complete, so a rate-limited or slow API must never hold
            # the document in "analyzing" — it is bounded and then abandoned.
            analysis = await self._analyze_bounded(document)

            # ── Persist everything on the one canonical record ───────────
            await self._set_status(db, audit_id, ProcessingStatus.FINALIZING)
            await self._persist(
                db, audit_id, document, summary, analysis, index_result,
                detected_clauses=detected_clauses, risk=risk,
                elapsed=time.time() - started,
            )

            logger.info(
                f"[Pipeline] {audit_id} completed in {time.time() - started:.1f}s "
                f"({document.page_count}p / {len(document.chunks)} chunks / "
                f"provider={summary.provider}, degraded={summary.degraded})"
            )

        except Exception as e:
            logger.error(f"[Pipeline] {audit_id} failed: {e}", exc_info=True)
            await self._fail(db, audit_id, f"Processing failed: {e}")

    # ────────────────────────────────────────────────────────────────────
    async def _summarize(
        self, db, audit_id: str, document: CanonicalDocument,
        clause_types: Dict[str, str],
    ) -> SummaryResult:
        """
        Run hierarchical summarisation off the event loop.

        The local model is synchronous and CPU/GPU bound, so it goes to a worker
        thread; running it inline would block every other request in the process.
        """
        import asyncio

        progress_updates: list = []

        def on_progress(percent: int, label: str) -> None:
            progress_updates.append((percent, label))

        summary = await asyncio.to_thread(
            self.summarizer.summarize_document, document, on_progress, clause_types
        )

        # Surface the last stage label the summariser reported.
        if progress_updates:
            percent, label = progress_updates[-1]
            await db["audits"].update_one(
                {"_id": audit_id},
                {"$set": {"progress": percent, "progress_label": label}},
            )
        return summary

    async def _analyze_bounded(self, document: CanonicalDocument) -> dict:
        """
        Run the optional LLM enrichment, bounded and abandonable.

        The call runs as its own task so the deadline applies to *this* coroutine
        rather than to the SDK's internal retry loop. On timeout the task is
        cancelled and the pipeline moves on; the deterministic analysis is
        already complete, so nothing is lost but the extra prose.
        """
        import asyncio

        if not ENABLE_LLM_ENRICHMENT:
            logger.info(
                "[Pipeline] LLM enrichment disabled (ENABLE_LLM_ENRICHMENT=false) "
                "— using the deterministic analysis alone."
            )
            return {}

        task = asyncio.ensure_future(self._analyze(document))
        try:
            return await asyncio.wait_for(asyncio.shield(task),
                                          timeout=LLM_ENRICHMENT_TIMEOUT)
        except asyncio.TimeoutError:
            task.cancel()
            logger.warning(
                f"[Pipeline] LLM enrichment exceeded {LLM_ENRICHMENT_TIMEOUT}s "
                "— continuing with the deterministic analysis."
            )
            return {}
        except Exception as e:
            logger.warning(f"[Pipeline] LLM enrichment failed: {e}")
            return {}

    async def _analyze(self, document: CanonicalDocument) -> dict:
        """
        Optional LLM clause/risk analysis.

        Entirely best-effort: the summary and the RAG index do not depend on it,
        so a missing or rate-limited API key degrades this stage alone.
        """
        try:
            from langchain_core.documents import Document as LCDocument

            from app.services.ai.audit_service import AuditService

            service = AuditService()
            if not service.api_key:
                logger.info("[Pipeline] no GOOGLE_API_KEY — skipping LLM risk analysis.")
                return {}

            legacy_chunks = [
                LCDocument(
                    page_content=chunk.text,
                    metadata={
                        "page_number": chunk.page_start,
                        "section_label": chunk.section_label,
                        "section_title": chunk.section_title,
                    },
                )
                for chunk in document.chunks
            ]
            return await service.generate_analysis(legacy_chunks)
        except Exception as e:
            logger.warning(f"[Pipeline] LLM analysis unavailable: {e}")
            return {}

    # ────────────────────────────────────────────────────────────────────
    async def _persist(
        self, db, audit_id: str, document: CanonicalDocument,
        summary: SummaryResult, analysis: dict, index_result: dict,
        detected_clauses: list, risk, elapsed: float,
    ) -> None:
        """
        Write results onto the single `audits` record.

        One canonical place for a document's analysis: the summary, the clause
        findings, the entities and the chunk metadata all live on the same
        document rather than in parallel collections holding overlapping copies.
        """
        jurisdiction = legal_extractor.extract_jurisdiction(document.chunks)

        update = {
            "status": ProcessingStatus.COMPLETED.value,
            "progress": 100,
            "progress_label": "Completed",
            "updated_at": datetime.utcnow(),
            "processing_time": round(elapsed, 2),
            "error": None,

            # Canonical document facts
            "page_count": document.page_count,
            "section_count": len(document.sections),
            "chunk_count": len(document.chunks),
            "token_count": document.total_tokens,
            "structure_detected": document.structure_detected,
            "vector_backend": index_result.get("backend", "local"),

            # Structured summary
            "summary_result": summary.model_dump(mode="json"),
            "document_type": summary.document_type,
            "executive_summary": summary.executive_summary,
            "summary": summary.executive_summary,  # alias for dashboard/library
            "section_summaries": [
                {
                    **s.model_dump(mode="json"),
                    "title": s.section_title,
                    "page_number": s.page_start,
                    "section": s.section_title,
                    "text_snippet": s.summary[:200] if s.summary else "",
                }
                for s in summary.section_summaries
            ],
            "key_takeaways": [item.summary for item in summary.important_clauses[:6]],

            # Entities, in the shape the existing frontend grid expects
            "entities": {
                "parties": [
                    {
                        "name": p.value or p.summary,
                        "role": p.label,
                        "address": "",
                        "signatory": "",
                    }
                    for p in summary.parties
                ],
                "dates": [
                    {
                        "label": d.label, "value": d.value,
                        "page_number": d.source_pages[0] if d.source_pages else 1,
                        "section": d.section_title, "note": d.summary,
                    }
                    for d in summary.key_dates
                ],
                "financials": [
                    {
                        "label": f.label, "amount": f.value,
                        "page_number": f.source_pages[0] if f.source_pages else 1,
                        "section": f.section_title, "detail": f.summary,
                    }
                    for f in summary.financial_terms
                ],
                "jurisdiction": jurisdiction,
                "organizations": [
                    {
                        "name": o.value or o.summary, "role": o.label,
                        "page_number": o.source_pages[0] if o.source_pages else 1,
                        "section": o.section_title,
                    }
                    for o in summary.organizations
                ],
                "people": [
                    {
                        "name": p.value or p.summary, "role": p.label,
                        "page_number": p.source_pages[0] if p.source_pages else 1,
                        "section": p.section_title,
                    }
                    for p in summary.people
                ],
                "legal_references": [
                    {
                        "value": lr.value, "type": lr.label,
                        "page_number": lr.source_pages[0] if lr.source_pages else 1,
                        "section": lr.section_title,
                    }
                    for lr in summary.legal_references
                ],
            },

            # Chunk index, so chat and comparison can cite without re-parsing
            "chunks": [
                {
                    "chunk_id": c.chunk_id, "order": c.order,
                    "page_start": c.page_start, "page_end": c.page_end,
                    "section_id": c.section_id, "section_number": c.section_number,
                    "section_title": c.section_title, "clause_type": c.clause_type,
                    "token_count": c.token_count,
                }
                for c in document.chunks
            ],

            # Provenance
            "model_name": summary.model_name,
            "model_version": summary.model_version,
            "summary_provider": summary.provider,
            "pipeline_version": settings.PIPELINE_VERSION,
            "prompt_version": settings.PROMPT_VERSION,
            "degraded": summary.degraded,
            "degraded_reason": summary.degraded_reason,
        }

        # Deterministic analysis is the baseline — it always runs, needs no API
        # key, and every finding is evidenced by text in the document.
        risk_payload = risk.to_dict()
        update["clauses"] = [
            clause.to_ui_clause(index + 1)
            for index, clause in enumerate(detected_clauses)
        ]
        update["detected_clauses"] = [clause.to_dict() for clause in detected_clauses]
        update["risks"] = risk_payload["risks"]
        update["risk_score"] = risk_payload["risk_score"]
        update["risk_level"] = risk_payload["risk_level"]
        update["risk_counts"] = risk_payload["risk_counts"]
        update["risk_assessable"] = risk_payload["assessable"]
        update["risk_note"] = risk_payload["risk_note"]
        update["missing_clauses"] = risk_payload["missing_clauses"]
        update["suggestions"] = [
            finding.recommendation for finding in risk.findings if finding.recommendation
        ][:15]

        # The LLM pass, when a key is configured, only *adds* explanation to
        # clauses it recognises. It never overwrites a deterministic finding,
        # so an unavailable or rate-limited model cannot degrade the analysis.
        if analysis:
            llm_by_title = {
                str(c.get("title", "")).lower(): c for c in analysis.get("clauses", [])
            }
            for clause in update["clauses"]:
                enrichment = llm_by_title.get(clause["title"].lower())
                if enrichment:
                    clause["explanation"] = (
                        enrichment.get("explanation") or clause["explanation"]
                    )
                    clause["recommendation"] = (
                        enrichment.get("recommendation") or clause["recommendation"]
                    )
            if analysis.get("contract_type"):
                update["document_type"] = analysis["contract_type"]
            update["llm_enriched"] = True
        else:
            update["llm_enriched"] = False

        await db["audits"].update_one({"_id": audit_id}, {"$set": update})

    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    async def _set_status(
        db, audit_id: str, status: ProcessingStatus, extra: Optional[dict] = None,
    ) -> None:
        percent, label = STAGE_PROGRESS.get(status, (0, status.value))
        update = {
            "status": status.value,
            "progress": percent,
            "progress_label": label,
            "updated_at": datetime.utcnow(),
        }
        if extra:
            update.update(extra)
        await db["audits"].update_one({"_id": audit_id}, {"$set": update})

    @classmethod
    async def _fail(cls, db, audit_id: str, message: str) -> None:
        await cls._set_status(
            db, audit_id, ProcessingStatus.FAILED, extra={"error": message},
        )


document_pipeline = DocumentPipeline()
