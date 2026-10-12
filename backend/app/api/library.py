from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from bson import ObjectId
from ..models.library import LibraryDocument
from app.api.deps import get_current_user, get_db

router = APIRouter()

@router.get("/library", response_model=List[LibraryDocument])
async def get_library_documents(current_user: dict = Depends(get_current_user), db=Depends(get_db)):
    audits_collection = db["audits"]
    
    user_id_str = str(current_user["_id"])
    query = {"user_id": user_id_str}

    cursor = audits_collection.find(query).sort("created_at", -1)
    audits = await cursor.to_list(length=None)
    
    formatted_docs = []
    for a in audits:
        doc_id = str(a["_id"])
        file_name = a.get("file_name", "Unknown Document")
        
        file_size = a.get("file_size")
        if isinstance(file_size, int):
            if file_size > 1024 * 1024:
                doc_size = f"{file_size / (1024 * 1024):.1f} MB"
            elif file_size > 1024:
                doc_size = f"{file_size / 1024:.1f} KB"
            else:
                doc_size = f"{file_size} B"
        else:
            doc_size = str(file_size) if file_size else "1.2 MB"
            
        created_at = a.get("created_at")
        iso_created_at = None
        if created_at:
            if isinstance(created_at, datetime):
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                iso_created_at = created_at.isoformat()
                doc_date = created_at.strftime("%b %d, %Y")
            elif isinstance(created_at, str):
                iso_created_at = created_at if (created_at.endswith("Z") or "+" in created_at) else created_at + "Z"
                doc_date = created_at[:10]
        else:
            doc_date = "N/A"
            
        contract_type = a.get("document_type") or a.get("contract_type")
        if not contract_type:
            if "." in file_name:
                contract_type = file_name.split(".")[-1].upper() + " Document"
            else:
                contract_type = "Legal Agreement"

        risk_score = a.get("risk_score", 100)
        risk_level = a.get("risk_level")
        if not risk_level:
            if risk_score > 80:
                risk_level = "Safe"
            elif risk_score >= 50:
                risk_level = "Moderate"
            else:
                risk_level = "Critical"

        risks = a.get("risks", [])
        critical_count = sum(1 for r in risks if r.get("severity", "").lower() in ("high", "critical"))

        status_val = a.get("status", "Completed")
        summary_val = a.get("summary") or a.get("executive_summary") or ""
        
        formatted_docs.append(
            LibraryDocument(
                id=doc_id,
                name=file_name,
                type=contract_type,
                size=doc_size,
                date=doc_date,
                created_at=iso_created_at,
                contract_type=contract_type,
                risk_score=risk_score,
                risk_level=risk_level,
                status=status_val,
                summary=summary_val[:200] if summary_val else "",
                critical_risks=critical_count
            )
        )
        
    return formatted_docs

@router.delete("/library/{doc_id}")
async def delete_library_document(doc_id: str, current_user: dict = Depends(get_current_user), db=Depends(get_db)):
    audits_collection = db["audits"]
    user_id_str = str(current_user["_id"])
    
    # Check deletion by str _id or ObjectId
    result = await audits_collection.delete_one({"_id": doc_id, "user_id": user_id_str})
    if result.deleted_count == 0 and ObjectId.is_valid(doc_id):
        result = await audits_collection.delete_one({"_id": ObjectId(doc_id), "user_id": user_id_str})
        
    if result.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or access denied."
        )
        
    return {"message": "Document deleted successfully."}
