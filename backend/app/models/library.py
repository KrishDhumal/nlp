from pydantic import BaseModel
from typing import Optional, List, Any

class LibraryDocument(BaseModel):
    id: str
    name: str
    type: str
    size: str
    date: str
    created_at: Optional[str] = None
    contract_type: Optional[str] = None
    risk_score: Optional[int] = 100
    risk_level: Optional[str] = "Safe"
    status: Optional[str] = "Completed"
    summary: Optional[str] = ""
    critical_risks: Optional[int] = 0
