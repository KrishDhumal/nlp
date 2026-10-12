"""
Audits API — upload, background processing, status polling, retrieval, file serving.

  POST /api/audits/upload         validate + store, queue processing, return at once
  GET  /api/audits/{id}/status    poll stage + progress while it runs
  GET  /api/audits/{id}           the finished analysis
  POST /api/audits/{id}/reprocess re-run the pipeline on an existing document
  GET  /api/audits/file/{id}      the original PDF (token via query, for iframes)

The upload no longer runs the analysis inline. Summarising a long contract takes
far longer than an HTTP request should, so the route returns as soon as the file
is safely stored and the work continues in a background task.

Every route resolves the document by `(_id, user_id)` taken from the verified
JWT. A user id is never read from the request body.
"""

import uuid
from datetime import datetime, timezone

import jwt
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse

from app.api.deps import get_current_user, get_db
from app.core.config import settings
from app.core.logging import logger
from app.domain.document import STAGE_PROGRESS, ProcessingStatus
from app.schemas.audit import AuditResponse, AuditStatusResponse, UploadAcceptedResponse
from app.services.ai.processor import DocumentProcessor, PDFValidationError
from app.services.pipeline import document_pipeline
from app.services.storage_service import storage_service

router = APIRouter()

processor = DocumentProcessor()

# Statuses that mean the pipeline is still working.
_IN_FLIGHT = {
    ProcessingStatus.UPLOADED.value,
    ProcessingStatus.EXTRACTING.value,
    ProcessingStatus.CLEANING.value,
    ProcessingStatus.CHUNKING.value,
    ProcessingStatus.INDEXING.value,
    ProcessingStatus.SUMMARIZING.value,
    ProcessingStatus.ANALYZING.value,
    ProcessingStatus.FINALIZING.value,
}


def _to_response(audit: dict) -> dict:
    """Project a stored record into the API response shape."""
    summary_result = audit.get("summary_result") or {}
    return AuditResponse(
        id=str(audit.get("_id", "")),
        file_name=audit.get("file_name", "document.pdf"),
        status=audit.get("status", ProcessingStatus.COMPLETED.value),
        progress=audit.get("progress", 100),
        progress_label=audit.get("progress_label", ""),
        error=audit.get("error"),

        document_type=audit.get("document_type") or summary_result.get("document_type", ""),
        contract_type=audit.get("document_type") or summary_result.get("document_type", ""),
        document_overview=audit.get("document_overview", ""),
        executive_summary=audit.get("executive_summary") or audit.get("summary", ""),
        summary=audit.get("summary") or audit.get("executive_summary", ""),
        key_takeaways=audit.get("key_takeaways", []),
        section_summaries=audit.get("section_summaries", []),
        summary_result=summary_result or None,

        clauses=audit.get("clauses", []),
        entities=audit.get("entities") or {},
        risks=audit.get("risks", []),
        missing_clauses=audit.get("missing_clauses", []),
        suggestions=audit.get("suggestions", []),
        risk_score=audit.get("risk_score", 0),
        risk_level=audit.get("risk_level", "Unknown"),
        risk_assessable=audit.get("risk_assessable", False),
        risk_note=audit.get("risk_note", ""),
        risk_counts=audit.get("risk_counts", {}),
        detected_clauses=audit.get("detected_clauses", []),
        compliance=audit.get("compliance"),
        llm_enriched=audit.get("llm_enriched", False),

        page_count=audit.get("page_count", 0),
        section_count=audit.get("section_count", 0),
        chunk_count=audit.get("chunk_count", 0),
        token_count=audit.get("token_count", 0),
        file_size=audit.get("file_size", 0),
        structure_detected=audit.get("structure_detected", False),
        vector_backend=audit.get("vector_backend", "local"),

        summary_provider=audit.get("summary_provider", ""),
        model_name=audit.get("model_name", ""),
        model_version=audit.get("model_version", ""),
        pipeline_version=audit.get("pipeline_version", ""),
        prompt_version=audit.get("prompt_version", ""),
        processing_time=audit.get("processing_time", 0.0),
        degraded=audit.get("degraded", False),
        degraded_reason=audit.get("degraded_reason", ""),

        created_at=audit.get("created_at").replace(tzinfo=timezone.utc) if isinstance(audit.get("created_at"), datetime) and audit.get("created_at").tzinfo is None else audit.get("created_at"),
        updated_at=audit.get("updated_at").replace(tzinfo=timezone.utc) if isinstance(audit.get("updated_at"), datetime) and audit.get("updated_at").tzinfo is None else audit.get("updated_at"),
    ).model_dump()


async def _load_owned(db, audit_id: str, user_id: str) -> dict:
    """Fetch a document the caller owns, or 404. The only lookup path used."""
    audit = await db["audits"].find_one({"_id": audit_id, "user_id": user_id})
    if not audit:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audit not found or you do not have access to it.",
        )
    return audit


@router.post(
    "/audits/upload",
    response_model=UploadAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_and_audit(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """
    Accept a PDF and queue it for processing.

    Returns 202 with an `id` immediately. The client then polls
    `GET /api/audits/{id}/status` and fetches the full analysis once complete.
    """
    user_id = str(current_user["_id"])
    content = await file.read()

    # ── Validate before anything is written ──────────────────────────────
    try:
        processor.validate(content, file.filename or "")
    except PDFValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    content_hash = processor.content_hash(content)

    # ── Reuse a previous analysis of the identical file ──────────────────
    # Scoped to this user: a hash match belonging to someone else is never
    # reused, so the dedupe cannot leak one user's document into another's.
    existing = await db["audits"].find_one({
        "user_id": user_id,
        "content_hash": content_hash,
        "status": ProcessingStatus.COMPLETED.value,
    })
    if existing:
        logger.info(
            f"[Audits] {content_hash[:12]} already analysed for user {user_id} "
            f"— returning audit {existing['_id']}"
        )
        return UploadAcceptedResponse(
            id=str(existing["_id"]),
            file_name=existing.get("file_name", file.filename or "document.pdf"),
            status=existing.get("status", ProcessingStatus.COMPLETED.value),
            progress=100,
            progress_label="Already analysed",
            reused=True,
        ).model_dump()

    audit_id = str(uuid.uuid4())
    storage_key = storage_service.save(content, audit_id, extension="pdf")
    now = datetime.now(timezone.utc)

    percent, label = STAGE_PROGRESS[ProcessingStatus.UPLOADED]
    await db["audits"].insert_one({
        "_id": audit_id,
        "user_id": user_id,
        "file_name": file.filename,
        "storage_key": storage_key,
        "content_hash": content_hash,
        "file_size": len(content),
        "status": ProcessingStatus.UPLOADED.value,
        "progress": percent,
        "progress_label": label,
        "created_at": now,
        "updated_at": now,
        "pipeline_version": settings.PIPELINE_VERSION,
    })

    # Runs after the response is sent.
    background_tasks.add_task(document_pipeline.run, db, audit_id, user_id)

    logger.info(f"[Audits] {audit_id} accepted for user {user_id}, processing queued.")
    return UploadAcceptedResponse(
        id=audit_id,
        file_name=file.filename or "document.pdf",
        status=ProcessingStatus.UPLOADED.value,
        progress=percent,
        progress_label=label,
    ).model_dump()


@router.get("/audits/{audit_id}/status", response_model=AuditStatusResponse)
async def get_audit_status(
    audit_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Lightweight poll target: stage, progress and whether the result is ready."""
    audit = await _load_owned(db, audit_id, str(current_user["_id"]))
    current = audit.get("status", ProcessingStatus.UPLOADED.value)
    return AuditStatusResponse(
        id=audit_id,
        status=current,
        progress=audit.get("progress", 0),
        progress_label=audit.get("progress_label", ""),
        error=audit.get("error"),
        is_complete=current == ProcessingStatus.COMPLETED.value,
        is_failed=current in {
            ProcessingStatus.FAILED.value, ProcessingStatus.OCR_REQUIRED.value
        },
        page_count=audit.get("page_count", 0),
        chunk_count=audit.get("chunk_count", 0),
    ).model_dump()


@router.post("/audits/{audit_id}/reprocess", status_code=status.HTTP_202_ACCEPTED)
async def reprocess_audit(
    audit_id: str,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """
    Re-run the pipeline on a stored document.

    Useful after a failure, or after the summarisation model changes — the stored
    `model_name` and `pipeline_version` say which engine produced a given result.
    """
    user_id = str(current_user["_id"])
    audit = await _load_owned(db, audit_id, user_id)

    if audit.get("status") in _IN_FLIGHT:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This document is already being processed.",
        )

    percent, label = STAGE_PROGRESS[ProcessingStatus.UPLOADED]
    await db["audits"].update_one(
        {"_id": audit_id},
        {"$set": {
            "status": ProcessingStatus.UPLOADED.value,
            "progress": percent,
            "progress_label": label,
            "error": None,
            "updated_at": datetime.utcnow(),
        }},
    )
    background_tasks.add_task(document_pipeline.run, db, audit_id, user_id)
    return {"id": audit_id, "status": ProcessingStatus.UPLOADED.value, "queued": True}


@router.get("/audits/{audit_id}", response_model=AuditResponse)
async def get_audit(
    audit_id: str,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """The full analysis for one document, scoped to the owning user."""
    audit = await _load_owned(db, audit_id, str(current_user["_id"]))
    return _to_response(audit)


@router.get("/audits/file/{audit_id}")
async def get_audit_file(
    audit_id: str,
    token: str = Query(None),
    db=Depends(get_db),
):
    """
    Stream the original PDF.

    The token arrives as a query parameter because an <iframe> src cannot carry
    an Authorization header; ownership is still verified against the record.
    """
    if not token:
        raise HTTPException(status_code=401, detail="Token required for document access")

    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email = payload.get("sub")
        if not email:
            raise ValueError("Missing subject in token")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired session token")

    user = await db["users"].find_one({"email": email})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session token")

    audit = await _load_owned(db, audit_id, str(user["_id"]))

    storage_key = audit.get("storage_key") or f"{audit_id}.pdf"
    if not storage_service.exists(storage_key):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="PDF file not found. It may have been removed from the server.",
        )

    return FileResponse(
        path=storage_service.get_absolute_path(storage_key),
        media_type="application/pdf",
        filename=audit.get("file_name") or f"{audit_id}.pdf",
        headers={"Content-Disposition": "inline"},
    )
