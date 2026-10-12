"""
Compare API — POST /api/compare, GET /api/compare/history

Compares two audited documents using the canonical structure both already
produced. The chunks are rebuilt from the stored PDFs through the same
DocumentProcessor the audit used, so there is one extraction implementation in
the codebase and the comparison sees exactly what the audit saw.

Both documents are resolved by (_id, user_id), so a user cannot compare against
a document they do not own by passing someone else's id.
"""

import uuid
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_current_user, get_db
from app.core.logging import logger
from app.domain.document import Chunk
from app.models.compare import CompareRequest
from app.services.ai.compare_service import compare_service
from app.services.ai.processor import DocumentProcessor
from app.services.storage_service import storage_service

router = APIRouter()
processor = DocumentProcessor()


async def _load_owned_audit(db, audit_id: str, user_id: str, label: str) -> dict:
    audit = await db["audits"].find_one({"_id": audit_id, "user_id": user_id})
    if not audit:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{label} not found or you do not have access to it.",
        )
    return audit


def _chunks_for(audit: dict, user_id: str) -> List[Chunk]:
    """
    Rebuild the canonical chunks for a stored document.

    Re-running the shared processor is deliberate: it guarantees the comparison
    uses the same extraction, cleaning, structure detection and chunking as the
    audit, rather than a second implementation that could drift.
    """
    storage_key = audit.get("storage_key") or f"{audit['_id']}.pdf"
    if not storage_service.exists(storage_key):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"The stored file for '{audit.get('file_name')}' is missing.",
        )
    document = processor.process(
        file_path=storage_service.get_absolute_path(storage_key),
        document_id=str(audit["_id"]),
        user_id=user_id,
        file_name=audit.get("file_name", "document.pdf"),
    )
    return document.chunks


@router.post("/compare")
async def compare_audits(
    request: CompareRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Compare two versions of a document and persist the result."""
    user_id = str(current_user["_id"])

    # If comparing the same document with itself, allow it — it will report identical match.
    audit_a = await _load_owned_audit(db, request.audit_id_a, user_id, "Document A")
    audit_b = await _load_owned_audit(db, request.audit_id_b, user_id, "Document B")

    name_a = audit_a.get("file_name", "Version A")
    name_b = audit_b.get("file_name", "Version B")
    if request.audit_id_a == request.audit_id_b:
        name_b = f"{name_a} (Copy)"

    try:
        chunks_a = _chunks_for(audit_a, user_id)
        chunks_b = _chunks_for(audit_b, user_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Compare] could not rebuild canonical chunks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not read one of the documents for comparison.",
        )

    if not chunks_a or not chunks_b:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="One of the documents has no extractable text to compare.",
        )

    result = compare_service.compare(chunks_a, chunks_b, name_a, name_b)
    payload = result.to_dict()

    # Format clauses list with citations and status for the UI
    status_map = {
        "MODIFIED": "modified",
        "ADDED": "missing_in_a",
        "REMOVED": "missing_in_b",
        "UNCHANGED": "matched",
    }
    formatted_clauses = []
    for c in payload.get("changes", []):
        formatted_clauses.append({
            "topic": c.get("section") or c.get("clause_type") or "Clause",
            "clause_type": c.get("clause_type", ""),
            "status": status_map.get(c.get("change_type"), "modified"),
            "change_type": c.get("change_type"),
            "doc_a_text": c.get("old_text") or "",
            "doc_b_text": c.get("new_text") or "",
            "doc_a_citation": {
                "page_number": c.get("page_old") or 1,
                "section": c.get("section"),
                "text": (c.get("old_text") or "")[:200],
            } if c.get("old_text") else None,
            "doc_b_citation": {
                "page_number": c.get("page_new") or 1,
                "section": c.get("section"),
                "text": (c.get("new_text") or "")[:200],
            } if c.get("new_text") else None,
            "risk_note": c.get("explanation") or "",
            "severity": c.get("severity", "LOW"),
            "similarity": c.get("similarity", 1.0),
            "term_changes": c.get("term_changes", []),
        })

    payload["clauses"] = formatted_clauses
    payload["missing_in_a"] = [c.get("section") for c in payload.get("changes", []) if c.get("change_type") == "ADDED"]
    payload["missing_in_b"] = [c.get("section") for c in payload.get("changes", []) if c.get("change_type") == "REMOVED"]

    payload.update({
        "audit_id_a": request.audit_id_a,
        "audit_id_b": request.audit_id_b,
        "doc_a_name": name_a,
        "doc_b_name": name_b,
    })

    # Persisted on the existing `comparisons` collection, scoped to the user.
    comparison_id = str(uuid.uuid4())
    await db["comparisons"].insert_one({
        "_id": comparison_id,
        "user_id": user_id,
        "audit_id_a": request.audit_id_a,
        "audit_id_b": request.audit_id_b,
        "doc_a_name": name_a,
        "doc_b_name": name_b,
        "summary": result.summary,
        "counts": result.counts,
        "risk_shift_score": result.risk_shift,
        "changes": payload["changes"],
        "clauses": formatted_clauses,
        "created_at": datetime.utcnow(),
    })
    payload["comparison_id"] = comparison_id
    return payload


@router.get("/compare/history")
async def comparison_history(
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Previous comparisons for the signed-in user, newest first."""
    user_id = str(current_user["_id"])
    cursor = db["comparisons"].find({"user_id": user_id}).sort("created_at", -1).limit(25)
    rows = await cursor.to_list(length=25)
    return [
        {
            "comparison_id": str(row["_id"]),
            "doc_a_name": row.get("doc_a_name", ""),
            "doc_b_name": row.get("doc_b_name", ""),
            "summary": row.get("summary", ""),
            "counts": row.get("counts", {}),
            "risk_shift_score": row.get("risk_shift_score", 0),
            "created_at": row.get("created_at"),
        }
        for row in rows
    ]
