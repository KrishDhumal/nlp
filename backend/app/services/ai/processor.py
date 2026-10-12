"""
DocumentProcessor — the single PDF processing pipeline.

    validate → extract (page-aware) → clean → detect structure → chunk

Its output is one CanonicalDocument, reused by summarisation, RAG embedding,
clause detection and entity extraction. There is deliberately no second
extraction or chunking path anywhere in the codebase: if this file is wrong,
every consumer is wrong in the same way, which is far easier to reason about
than four subtly different representations of the same contract.

Extraction backends, in order of preference:
  1. pdfplumber — best at preserving layout-driven line breaks, which is what
     makes heading detection work.
  2. PyMuPDF — faster and more tolerant of malformed files; used when pdfplumber
     cannot open the document or returns nothing.

OCR is deliberately not implemented yet, but the extraction layer reports
`OCR_REQUIRED` for documents with no usable text layer, so an OCR backend can
be added as a third step without touching anything downstream.
"""

from __future__ import annotations

import hashlib
import os
from typing import List, Optional, Tuple

from app.core.config import settings
from app.core.logging import logger
from app.domain.document import CanonicalDocument, Page, ProcessingStatus
from app.services.ai.chunking import TokenAwareChunker
from app.services.ai.structure import detect_sections, page_fallback_sections
from app.services.ai.text_cleaning import clean_pages

PDF_MAGIC = b"%PDF-"


class PDFValidationError(ValueError):
    """Raised when an uploaded file is not an acceptable PDF."""


class DocumentProcessor:
    def __init__(self, chunker: Optional[TokenAwareChunker] = None):
        self.chunker = chunker or TokenAwareChunker()

    # ────────────────────────────────────────────────────────────────────
    # 1. Validation
    # ────────────────────────────────────────────────────────────────────
    def validate(self, file_bytes: bytes, filename: str) -> None:
        """
        Validate an upload *before* it is written to storage or parsed.
        Raises PDFValidationError with a user-facing message on failure.
        """
        if not filename or not filename.lower().endswith(".pdf"):
            raise PDFValidationError(
                "Only PDF files are supported. Please upload a .pdf document."
            )

        if not file_bytes:
            raise PDFValidationError("The uploaded file is empty.")

        max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
        if len(file_bytes) > max_bytes:
            size_mb = len(file_bytes) / (1024 * 1024)
            raise PDFValidationError(
                f"File is {size_mb:.1f} MB, which exceeds the "
                f"{settings.MAX_UPLOAD_MB} MB limit."
            )

        # A .pdf extension proves nothing — check the actual header.
        if not file_bytes.lstrip()[:8].startswith(PDF_MAGIC):
            raise PDFValidationError(
                "This file is not a valid PDF. It may be renamed or corrupted."
            )

    @staticmethod
    def content_hash(file_bytes: bytes) -> str:
        """
        SHA-256 of the file contents, used to detect a re-upload of the same
        document and skip repeated summarisation. Never used to share results
        across users: the upload route scopes every hash lookup to the
        authenticated user, so a match under another account is ignored.
        """
        return hashlib.sha256(file_bytes).hexdigest()

    # ────────────────────────────────────────────────────────────────────
    # 2. Page-aware extraction
    # ────────────────────────────────────────────────────────────────────
    def extract_pages(self, file_path: str) -> List[Tuple[int, str]]:
        """
        Extract raw text per page as (page_number, text), 1-based.

        Page boundaries are never flattened: page-level citation is a core
        product requirement, so the mapping survives every later stage.
        """
        pages = self._extract_with_pdfplumber(file_path)
        if not any(text.strip() for _, text in pages):
            logger.warning("[Processor] pdfplumber found no text, trying PyMuPDF.")
            pages = self._extract_with_pymupdf(file_path)
        return pages

    def _extract_with_pdfplumber(self, file_path: str) -> List[Tuple[int, str]]:
        try:
            import pdfplumber
        except ImportError:
            logger.warning("[Processor] pdfplumber unavailable, using PyMuPDF.")
            return self._extract_with_pymupdf(file_path)

        pages: List[Tuple[int, str]] = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for index, page in enumerate(pdf.pages, start=1):
                    try:
                        text = page.extract_text(x_tolerance=1.5, y_tolerance=2.5) or ""
                    except Exception as e:
                        # One malformed page must not fail the whole upload.
                        logger.warning(f"[Processor] page {index} failed: {e}")
                        text = ""
                    pages.append((index, text))
        except Exception as e:
            logger.warning(f"[Processor] pdfplumber could not open the file: {e}")
            return self._extract_with_pymupdf(file_path)
        return pages

    def _extract_with_pymupdf(self, file_path: str) -> List[Tuple[int, str]]:
        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise PDFValidationError(
                "No PDF text extraction backend is installed."
            ) from e

        pages: List[Tuple[int, str]] = []
        doc = fitz.open(file_path)
        try:
            if doc.is_encrypted and not doc.authenticate(""):
                raise PDFValidationError(
                    "This PDF is password protected. Please upload an unlocked copy."
                )
            for index, page in enumerate(doc, start=1):
                pages.append((index, page.get_text("text") or ""))
        finally:
            doc.close()
        return pages

    def page_count(self, file_path: str) -> int:
        try:
            return len(self.extract_pages(file_path))
        except Exception:
            return 0

    # ────────────────────────────────────────────────────────────────────
    # 3. Full pipeline
    # ────────────────────────────────────────────────────────────────────
    def process(
        self,
        file_path: str,
        document_id: str,
        user_id: str,
        file_name: Optional[str] = None,
        content_hash: str = "",
        file_size: int = 0,
    ) -> CanonicalDocument:
        """
        Run extraction → cleaning → structure detection → chunking.

        Returns a CanonicalDocument. A scanned PDF comes back with
        `status = OCR_REQUIRED` and no chunks rather than raising, so the caller
        can record that state and tell the user precisely what went wrong.
        """
        name = file_name or os.path.basename(file_path)
        document = CanonicalDocument(
            document_id=document_id,
            user_id=user_id,
            file_name=name,
            content_hash=content_hash,
            file_size=file_size,
        )

        # Extract, then clean using cross-page analysis for running headers.
        raw_pages = self.extract_pages(file_path)
        cleaned = clean_pages(raw_pages)
        document.pages = [Page(page_number=num, text=text) for num, text in cleaned]

        if not document.has_usable_text(settings.MIN_TEXT_CHARS):
            document.status = ProcessingStatus.OCR_REQUIRED
            logger.warning(
                f"[Processor] {name}: only "
                f"{len(document.full_text.strip())} chars of text — OCR required."
            )
            return document

        # Structure, with a page-based fallback when no headings are found.
        sections, structure_detected = detect_sections(cleaned)
        if not structure_detected:
            logger.info(
                f"[Processor] {name}: no reliable structure detected "
                "— falling back to page segmentation."
            )
            sections = page_fallback_sections(cleaned)
        document.sections = sections
        document.structure_detected = structure_detected

        # Token-aware chunking over the detected structure.
        document.chunks = self.chunker.chunk_sections(sections, document_id, user_id)
        document.status = ProcessingStatus.CHUNKING

        logger.info(
            f"[Processor] {name}: {document.page_count} page(s), "
            f"{len(sections)} section(s), {len(document.chunks)} chunk(s), "
            f"{document.total_tokens} tokens, structure={structure_detected}"
        )
        return document


# ── Clause-type tagging (cheap, deterministic, no LLM) ──────────────────────
# Keyword signatures for the seven core categories. Used to tag every chunk at
# index time so vector metadata carries a real `clause_type`. The LLM analysis
# stage still produces the authoritative, explained classification.
_CLAUSE_SIGNATURES = {
    "Termination": (
        "terminat", "expiration", "renewal", "non-renewal", "notice of termination",
        "cure period", "material breach", "for convenience", "suspend services",
        "discharge of", "withdrawal of", "right to withdraw", "cease to represent",
        "end of engagement", "termination of representation",
    ),
    "Payment": (
        "payment", "invoice", "fee", "retainer", "compensation", "late charge",
        "interest at", "net thirty", "net 30", "reimburse", "price", "usd", "$",
        "deposit", "hourly rate", "flat fee", "contingency fee",
        "billing statement", "statements and payments", "legal fees",
        "attorney fees", "costs and expenses",
    ),
    "Liability": (
        "liability", "liable", "damages", "consequential", "indirect",
        "aggregate liability", "shall not exceed", "disclaimer", "warrant",
    ),
    "Indemnification": (
        "indemnif", "hold harmless", "defend", "third-party claim", "third party claim",
    ),
    "Confidentiality": (
        "confidential", "non-disclosure", "nondisclosure", "trade secret",
        "proprietary information", "personal data", "privacy",
        "conflict of interest", "attorney-client privilege",
        "duty of confidentiality", "privileged communication",
    ),
    "Governing Law": (
        "governing law", "governed by", "jurisdiction", "venue", "arbitration",
        "dispute", "conflict of laws", "jury trial", "class action",
    ),
    "IP Rights": (
        "intellectual property", "copyright", "patent", "trademark", "licence",
        "license", "work product", "deliverables", "assigns to", "moral rights",
    ),
}


def classify_clause_type(text: str, section_title: str = "") -> str:
    """
    Best-effort category for a chunk, from its section heading and body text.

    The heading is weighted far more heavily than the body, because a section
    titled "Termination" is about termination even when it mentions fees.
    Returns "General" when nothing scores.
    """
    haystack = (text or "").lower()
    heading = (section_title or "").lower()

    best_category, best_score = "General", 0
    for category, keywords in _CLAUSE_SIGNATURES.items():
        score = 0
        for keyword in keywords:
            if keyword in heading:
                score += 5
            if keyword in haystack:
                score += 1
        if score > best_score:
            best_category, best_score = category, score

    return best_category if best_score >= 2 else "General"


# Backwards-compatible alias: the class was called PDFProcessor before the
# canonical-document refactor, and other modules still import that name.
PDFProcessor = DocumentProcessor
