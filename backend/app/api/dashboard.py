from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any
from app.api.deps import get_db, get_current_user

router = APIRouter()


@router.get("/dashboard/stats")
async def get_dashboard_stats(current_user: dict = Depends(get_current_user), db=Depends(get_db)):
    audits_collection = db["audits"]
    user_id_str = str(current_user["_id"])
    
    query = {"user_id": user_id_str}
    audits = await audits_collection.find(query).to_list(length=None)
    
    total_audits = len(audits)
    if total_audits == 0:
        return {
            "Total Audits": 0,
            "Critical Risks": 0,
            "Average Compliance": 0
        }
        
    critical_risks = 0
    total_score = 0
    
    for audit in audits:
        total_score += audit.get("risk_score", 0)
        risks = audit.get("risks", [])
        if any(r.get("severity", "").lower() == "high" for r in risks):
            critical_risks += 1
            
    average_compliance = round(total_score / total_audits, 2)
    
    return {
        "Total Audits": total_audits,
        "Critical Risks": critical_risks,
        "Average Compliance": average_compliance
    }

@router.get("/audits/recent")
async def get_recent_audits(current_user: dict = Depends(get_current_user), db=Depends(get_db)):
    audits_collection = db["audits"]
    user_id_str = str(current_user["_id"])
    
    query = {"user_id": user_id_str}
    cursor = audits_collection.find(query).sort("created_at", -1).limit(3)
    audits = await cursor.to_list(length=3)
    
    formatted_audits = []
    for a in audits:
        a["_id"] = str(a["_id"])
        if "user_id" in a:
            a["user_id"] = str(a["user_id"])
        
        created_at = a.get("created_at")
        if created_at:
            if isinstance(created_at, datetime):
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                a["created_at"] = created_at.isoformat()
            elif isinstance(created_at, str):
                if not created_at.endswith("Z") and "+" not in created_at:
                    a["created_at"] = created_at + "Z"
                    
        formatted_audits.append(a)
        
    return formatted_audits

@router.get("/notifications")
async def get_notifications(current_user: dict = Depends(get_current_user), db=Depends(get_db)):
    """Returns the 10 most recent audit events as notifications."""
    audits_collection = db["audits"]
    user_id_str = str(current_user["_id"])
    
    cursor = audits_collection.find(
        {"user_id": user_id_str}
    ).sort("created_at", -1).limit(10)
    audits = await cursor.to_list(length=10)
    
    notifications = []
    for a in audits:
        risk_count = len(a.get("risks", []))
        has_critical = any(r.get("severity", "").lower() == "high" for r in a.get("risks", []))
        
        if has_critical:
            notif_type = "critical"
            message = f"Critical risks found in {a.get('file_name', 'document')}"
        elif risk_count > 0:
            notif_type = "warning"
            message = f"{risk_count} risk(s) detected in {a.get('file_name', 'document')}"
        else:
            notif_type = "success"
            message = f"Audit completed for {a.get('file_name', 'document')} — No risks found"
        
        created_at = a.get("created_at")
        iso_created_at = None
        if created_at:
            if isinstance(created_at, datetime):
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                iso_created_at = created_at.isoformat()
            elif isinstance(created_at, str):
                iso_created_at = created_at if (created_at.endswith("Z") or "+" in created_at) else created_at + "Z"

        notifications.append({
            "id": str(a["_id"]),
            "type": notif_type,
            "message": message,
            "file_name": a.get("file_name", "Unknown"),
            "risk_score": a.get("risk_score", 0),
            "created_at": iso_created_at
        })
    
    return notifications

