"""
Audit schemas — the response contract for the audit pipeline.

These mirror exactly what AuditNew.jsx renders, so the frontend never has to
guess at or backfill missing fields.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class Risk(BaseModel):
    title: str
    severity: str = "Medium"          # Low | Medium | High | Critical
    description: str = ""
    page_number: int = 1
    snippet: str = ""
    section: str = ""


class Clause(BaseModel):
    id: str
    category: str                     # one of CLAUSE_CATEGORIES, or "Other"
    title: str
    page_number: int = 1
    page_end: int = 1
    section: str = ""
    risk_level: str = "Low"           # Low | Medium | High | Critical
    snippet: str = ""
    explanation: str = ""
    recommendation: str = ""
    # Phrases that evidenced this clause, highlighted inside the snippet.
    matched_terms: List[str] = Field(default_factory=list)
    confidence: float = 0.0


class SectionSummary(BaseModel):
    title: str
    page_number: int = 1
    section: str = ""
    text_snippet: str = ""
    key_points: List[str] = Field(default_factory=list)


class Party(BaseModel):
    name: str
    role: str = "Party"
    address: str = ""
    signatory: str = ""


class KeyDate(BaseModel):
    label: str
    value: str
    page_number: int = 1
    section: str = ""
    note: str = ""


class FinancialValue(BaseModel):
    label: str
    amount: str
    page_number: int = 1
    section: str = ""
    detail: str = ""


class Jurisdiction(BaseModel):
    governing_law: str = "Not specified"
    venue: str = "Not specified"
    dispute_resolution: str = "Not specified"
    page_number: int = 1
    section: str = ""


class Entities(BaseModel):
    parties: List[Party] = Field(default_factory=list)
    dates: List[KeyDate] = Field(default_factory=list)
    financials: List[FinancialValue] = Field(default_factory=list)
    jurisdiction: Jurisdiction = Field(default_factory=Jurisdiction)


class UploadAcceptedResponse(BaseModel):
    """
    202 response from POST /api/audits/upload.

    The analysis has not run yet — the client polls the status endpoint. `reused`
    is true when an identical file had already been analysed for this user and no
    new processing was queued.
    """

    id: str
    file_name: str
    status: str
    progress: int = 0
    progress_label: str = ""
    reused: bool = False


class AuditStatusResponse(BaseModel):
    """Poll target while the pipeline runs."""

    id: str
    status: str
    progress: int = 0
    progress_label: str = ""
    error: Optional[str] = None
    is_complete: bool = False
    is_failed: bool = False
    page_count: int = 0
    chunk_count: int = 0


class AuditResponse(BaseModel):
    """Full audit payload returned by GET /api/audits/{audit_id}."""

    id: str
    file_name: str
    status: str = "completed"
    progress: int = 100
    progress_label: str = ""
    error: Optional[str] = None

    # `contract_type` is the older name the frontend reads; both are returned.
    document_type: str = ""
    contract_type: str = "Legal Document"
    document_overview: str = ""

    executive_summary: str = ""
    summary: str = ""
    key_takeaways: List[str] = Field(default_factory=list)
    section_summaries: List[dict] = Field(default_factory=list)
    # The complete structured SummaryResult, for clients that want every field.
    summary_result: Optional[dict] = None

    clauses: List[Clause] = Field(default_factory=list)
    entities: dict = Field(default_factory=dict)
    risks: List[Risk] = Field(default_factory=list)
    missing_clauses: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)

    risk_score: int = Field(default=0, ge=0, le=100)
    risk_level: str = "Unknown"
    # False when too little contract language was found to judge. A score of 0
    # with assessable=False means "unknown", not "safe".
    risk_assessable: bool = False
    risk_note: str = ""
    risk_counts: dict = Field(default_factory=dict)
    detected_clauses: List[dict] = Field(default_factory=list)
    compliance: Optional[dict] = None
    llm_enriched: bool = False

    page_count: int = 0
    section_count: int = 0
    chunk_count: int = 0
    token_count: int = 0
    file_size: int = 0
    structure_detected: bool = False
    vector_backend: str = "local"

    # Provenance — which engine and pipeline produced this result.
    summary_provider: str = ""
    model_name: str = ""
    model_version: str = ""
    pipeline_version: str = ""
    prompt_version: str = ""
    processing_time: float = 0.0
    degraded: bool = False
    degraded_reason: str = ""

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
