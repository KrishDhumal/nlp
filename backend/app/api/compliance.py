"""
Compliance API — POST /api/compliance/check

Runs the rule-based framework checks against one audited document's canonical
chunks. The document is resolved by (_id, user_id), so another user's document
cannot be reached by changing the id in the request.
"""

from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_current_user, get_db
from app.core.logging import logger
from app.models.compliance import ComplianceCheckRequest
from app.services.ai.compliance_service import SUPPORTED_FRAMEWORKS, compliance_service
from app.services.ai.processor import DocumentProcessor
from app.services.storage_service import storage_service

router = APIRouter()
processor = DocumentProcessor()


@router.get("/compliance/frameworks")
async def list_frameworks():
    """Frameworks this build can check, for the selector UI."""
    return {
        "frameworks": [
            {"id": "GDPR", "name": "GDPR", "description": "EU General Data Protection Regulation"},
            {"id": "HIPAA", "name": "HIPAA", "description": "US health information privacy"},
            {"id": "SOC2", "name": "SOC 2", "description": "Trust services criteria"},
        ]
    }


@router.post("/compliance/check")
async def check_compliance(
    request: ComplianceCheckRequest,
    current_user: dict = Depends(get_current_user),
    db=Depends(get_db),
):
    """Check one document against the selected regulatory frameworks."""
    user_id = str(current_user["_id"])

    audit = await db["audits"].find_one({"_id": request.audit_id, "user_id": user_id})
    if not audit:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audit not found or you do not have access to it.",
        )

    storage_key = audit.get("storage_key") or f"{request.audit_id}.pdf"
    if not storage_service.exists(storage_key):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The stored PDF for this document is missing.",
        )

    try:
        # Same processor as the audit — one extraction implementation.
        document = processor.process(
            file_path=storage_service.get_absolute_path(storage_key),
            document_id=request.audit_id,
            user_id=user_id,
            file_name=audit.get("file_name", "document.pdf"),
        )
    except Exception as e:
        logger.error(f"[Compliance] could not read document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not read the document for compliance checking.",
        )

    if not document.chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This document has no extractable text to check.",
        )

    requested: List[str] = list(request.regulations or SUPPORTED_FRAMEWORKS)
    result = compliance_service.check(document.chunks, requested)
    result.update({
        "audit_id": request.audit_id,
        "file_name": audit.get("file_name", ""),
        "checked_at": datetime.utcnow().isoformat(),
    })

    await db["audits"].update_one(
        {"_id": request.audit_id},
        {"$set": {"compliance": result, "updated_at": datetime.utcnow()}},
    )
    return result
