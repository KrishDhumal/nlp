"""
Canonical document representation — the single shared output of PDF processing.

Every downstream consumer (summarization, RAG embedding, clause detection,
entity extraction) reads from these objects. There is exactly one extraction
and one chunking implementation in the codebase; nothing re-parses the PDF.

    CanonicalDocument
        ├── pages[]     — page_number + text, page boundaries preserved
        ├── sections[]  — detected legal structure, may span pages
        └── chunks[]    — token-aware units carrying full source traceability

Page traceability is a product requirement, not a nicety: every chunk records
`page_start`/`page_end` and its section, so a summary sentence can always be
traced back to the page it came from.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class ProcessingStatus(str, Enum):
    """
    Lifecycle of a document through the pipeline.

    The frontend polls this and renders a human-readable label per stage, so the
    names are part of the API contract.
    """

    UPLOADED = "uploaded"
    EXTRACTING = "extracting"
    CLEANING = "cleaning"
    CHUNKING = "chunking"
    INDEXING = "indexing"
    SUMMARIZING = "summarizing"
    ANALYZING = "analyzing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    # A PDF with no usable text layer: a scanned image. Reported explicitly
    # rather than returning an empty summary that looks like a real result.
    OCR_REQUIRED = "extraction_failed_or_ocr_required"


# Percentage shown in the UI for each stage, and the label beside it.
STAGE_PROGRESS = {
    ProcessingStatus.UPLOADED: (5, "Upload received"),
    ProcessingStatus.EXTRACTING: (15, "Extracting document text..."),
    ProcessingStatus.CLEANING: (25, "Normalizing text..."),
    ProcessingStatus.CHUNKING: (35, "Analyzing sections..."),
    ProcessingStatus.INDEXING: (45, "Indexing for search..."),
    ProcessingStatus.SUMMARIZING: (65, "Generating summary..."),
    ProcessingStatus.ANALYZING: (85, "Detecting clauses and entities..."),
    ProcessingStatus.FINALIZING: (95, "Finalizing summary..."),
    ProcessingStatus.COMPLETED: (100, "Completed"),
    ProcessingStatus.FAILED: (100, "Processing failed"),
    ProcessingStatus.OCR_REQUIRED: (100, "No readable text — OCR required"),
}


class Page(BaseModel):
    """One page of the PDF, as extracted. `text` is cleaned, never raw."""

    page_number: int = Field(..., ge=1)
    text: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class Section(BaseModel):
    """
    A logical section of the contract, e.g. "5. Termination".

    A section may span pages, which is why it carries a start and an end.
    `number` is the drafted reference ("5.4", "ARTICLE VII") and is empty for
    unnumbered material such as a preamble.
    """

    section_id: str
    section_number: str = ""
    section_title: str = ""
    page_start: int = Field(default=1, ge=1)
    page_end: int = Field(default=1, ge=1)
    text: str = ""

    @property
    def label(self) -> str:
        """Citation label, e.g. '5.4 Termination'."""
        if self.section_number and self.section_title:
            return f"{self.section_number} {self.section_title}"
        return (self.section_number or self.section_title or "Unlabelled Section").strip()


class Chunk(BaseModel):
    """
    A token-bounded unit of text with full provenance.

    This is the object every consumer shares. `clause_type` is optional
    metadata — a chunk exists and is usable before any classification runs.
    """

    chunk_id: str
    document_id: str
    user_id: str
    text: str
    token_count: int = 0

    page_start: int = Field(default=1, ge=1)
    page_end: int = Field(default=1, ge=1)
    section_id: str = ""
    section_number: str = ""
    section_title: str = ""

    # Populated later by clause detection, when it runs at all.
    clause_type: Optional[str] = None

    # Position in document order, used to reassemble and to group for summaries.
    order: int = 0

    @property
    def page_label(self) -> str:
        """'Page 7' or 'Pages 7-8', for citations."""
        if self.page_start == self.page_end:
            return f"Page {self.page_start}"
        return f"Pages {self.page_start}-{self.page_end}"

    @property
    def section_label(self) -> str:
        if self.section_number and self.section_title:
            return f"{self.section_number} {self.section_title}"
        return (self.section_number or self.section_title or "").strip()


class CanonicalDocument(BaseModel):
    """
    The processed form of one uploaded PDF.

    Produced once by DocumentProcessor and consumed by every analysis stage.
    """

    document_id: str
    user_id: str
    file_name: str = "document.pdf"
    content_hash: str = ""
    file_size: int = 0

    pages: List[Page] = Field(default_factory=list)
    sections: List[Section] = Field(default_factory=list)
    chunks: List[Chunk] = Field(default_factory=list)

    # True when structure detection found real headings, False when the document
    # fell back to page/paragraph segmentation. Recorded because it changes how
    # much the section-level summaries can be trusted.
    structure_detected: bool = False

    # Detected after extraction, before summarisation. Only the summarisation
    # model depends on it; every other stage is language-agnostic.
    language: str = "en"
    language_confidence: float = 0.0

    status: ProcessingStatus = ProcessingStatus.UPLOADED
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def total_tokens(self) -> int:
        return sum(chunk.token_count for chunk in self.chunks)

    @property
    def full_text(self) -> str:
        """The whole document, page-ordered. Used by deterministic extraction."""
        return "\n\n".join(page.text for page in self.pages if page.text.strip())

    def has_usable_text(self, min_chars: int = 200) -> bool:
        """
        Whether enough text was extracted to analyse.

        A scanned PDF typically yields a handful of stray characters; treating
        that as a successful extraction would produce a confident empty summary.
        """
        return len(self.full_text.strip()) >= min_chars

    def chunks_for_section(self, section_id: str) -> List[Chunk]:
        return [chunk for chunk in self.chunks if chunk.section_id == section_id]
