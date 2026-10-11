from pydantic import BaseModel, Field
from typing import List, Optional


class ComplianceCheckRequest(BaseModel):
    """Request body for POST /api/compliance/check."""
    audit_id: str = Field(..., description="The audit ID to run compliance check on")
    regulations: List[str] = Field(
        default=["GDPR", "HIPAA", "SOC2"],
        description="List of regulations to check against"
    )


class Violation(BaseModel):
    """A single compliance violation found in the document."""
    regulation: str = Field(..., description="e.g. GDPR, CCPA, AI_ACT")
    clause: str = Field(default="", description="The clause/section in the document")
    severity: str = Field(default="Medium", description="Low, Medium, High, Critical")
    description: str = Field(default="", description="What the violation is")
    suggestion: str = Field(default="", description="Specific redline/edit suggestion to fix it")
    article_reference: str = Field(default="", description="Relevant regulation article, e.g. 'GDPR Art. 17'")


class ComplianceCheckResponse(BaseModel):
    """Response body from compliance check."""
    compliance_score: int = Field(default=100, ge=0, le=100)
    total_violations: int = Field(default=0)
    violations: List[Violation] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list, description="Prioritized list of edits to make")
    summary: str = Field(default="", description="Overall compliance summary")
    audit_id: str = ""
    file_name: str = ""
