"""
SummarizationService — hierarchical map/reduce summarisation over canonical chunks.

A long contract cannot be sent to a model in one piece, so summarisation is a
tree:

    chunks → chunk summaries → group summaries → section summaries → executive

The depth adapts to the document. A four-page NDA does not need an intermediate
reduce level, and forcing one only blurs detail and wastes inference; documents
below `SUMMARY_HIERARCHY_THRESHOLD` chunks therefore skip straight to the final
summary. Long documents get the full tree.

Source traceability is carried structurally, not asked of the model: every
section summary keeps the chunk ids and page range it was built from, so any
statement can be traced to its pages. Page numbers are never generated.
"""

from __future__ import annotations

import os
import re
import time
from typing import Callable, Dict, List, Optional

from app.core.config import settings
from app.core.logging import logger
from app.domain.document import CanonicalDocument, Chunk
from app.services.ai.extraction import legal_extractor
from app.services.ai.text_cleaning import extract_preserved_values
from app.domain.summary import (
    SectionSummary,
    SourceReference,
    SummaryItem,
    SummaryResult,
)
from app.services.ai.summarization.base import (
    CHUNK_INSTRUCTION,
    FINAL_INSTRUCTION,
    GROUP_INSTRUCTION,
    SummarizationProvider,
)
from app.services.ai.summarization.registry import get_provider

ProgressHook = Optional[Callable[[int, str], None]]


class SummarizationService:
    """
    Orchestrates summarisation. Depends only on `SummarizationProvider`, so the
    engine behind it is a configuration choice.
    """

    def __init__(self, provider: Optional[SummarizationProvider] = None):
        self._provider = provider
        # How many chunks the model summarised on the last run, as opposed to
        # falling back to an extract.
        self._model_summary_count = 0

    @property
    def provider(self) -> SummarizationProvider:
        if self._provider is None:
            self._provider = get_provider()
        return self._provider

    # ────────────────────────────────────────────────────────────────────
    def summarize_document(
        self,
        document: CanonicalDocument,
        on_progress: ProgressHook = None,
        clause_types: Optional[Dict[str, str]] = None,
    ) -> SummaryResult:
        """
        Build the complete structured summary for one canonical document.

        Never raises for a model failure: if the provider is unavailable the
        result comes back `degraded=True` with extractive text, because a
        contract that was successfully parsed and indexed should still be usable.
        """
        started = time.time()
        provider = self.provider

        result = SummaryResult(
            document_id=document.document_id,
            provider=provider.name,
            pipeline_version=settings.PIPELINE_VERSION,
            prompt_version=settings.PROMPT_VERSION,
        )

        chunks = document.chunks
        if not chunks:
            result.degraded = True
            result.degraded_reason = "No chunks were produced for this document."
            result.processing_time = time.time() - started
            return result

        # ── Structured facts first: deterministic, no model needed ───────
        if on_progress:
            on_progress(55, "Extracting key entities...")
        facts = legal_extractor.extract_all(chunks)
        result.parties = facts["parties"]
        result.key_dates = facts["key_dates"]
        result.financial_terms = facts["financial_terms"]
        result.key_obligations = facts["key_obligations"]
        result.attention_points = facts["attention_points"]
        result.document_type = legal_extractor.detect_document_type(
            document.full_text, document.file_name
        )

        model_ready = provider.is_available()
        # Read the model identity only after the load: the provider may have
        # fallen through to a different checkpoint, and a stored result must name
        # the engine that actually produced it.
        result.model_name = provider.model_name
        result.model_version = provider.model_version
        if not model_ready:
            logger.warning(
                f"[Summarization] provider '{provider.name}' unavailable — "
                "falling back to extractive summaries."
            )
            result.degraded = True
            result.degraded_reason = (
                f"The {provider.name} summarisation model could not be loaded; "
                "summaries below are extracted directly from the document text."
            )

        # ── Map: one summary per chunk ───────────────────────────────────
        if on_progress:
            on_progress(60, "Summarizing sections...")
        self._model_summary_count = 0
        chunk_summaries = self._summarize_chunks(chunks, provider, model_ready)

        if model_ready and self._model_summary_count == 0:
            result.degraded = True
            result.degraded_reason = (
                f"The {provider.name} model ({provider.model_name}) loaded but "
                "produced no usable summary for any section, so the text below is "
                "extracted verbatim from the document rather than summarised. "
                "Check the server log for generation errors."
            )
            logger.error(f"[Summarization] {result.degraded_reason}")
        elif model_ready and self._model_summary_count < len(chunks) / 2:
            result.degraded = True
            result.degraded_reason = (
                f"Only {self._model_summary_count} of {len(chunks)} sections could "
                "be summarised by the model; the rest are extracted verbatim from "
                "the document."
            )
            logger.warning(f"[Summarization] {result.degraded_reason}")

        # ── Reduce: chunk summaries → section summaries ──────────────────
        if on_progress:
            on_progress(72, "Combining section summaries...")
        result.section_summaries = self._build_section_summaries(
            document, chunk_summaries, provider, model_ready
        )

        # ── Reduce again for long documents, then the executive summary ──
        if on_progress:
            on_progress(80, "Writing executive summary...")
        result.executive_summary = self._build_executive_summary(
            result.section_summaries, provider, model_ready
        )
        result.document_overview = self._build_overview(document, result)

        # ── Important clauses, using clause types when available ─────────
        result.important_clauses = self._build_important_clauses(
            result.section_summaries, chunks, clause_types
        )

        result.source_references = [
            SourceReference(
                chunk_ids=section.chunk_ids,
                page_start=section.page_start,
                page_end=section.page_end,
                section_title=section.section_title,
                section_number=section.section_number,
            )
            for section in result.section_summaries
        ]

        result.processing_time = round(time.time() - started, 2)
        logger.info(
            f"[Summarization] {document.file_name}: "
            f"{len(chunks)} chunk(s) -> {len(result.section_summaries)} section "
            f"summary/ies in {result.processing_time}s "
            f"(provider={provider.name}, degraded={result.degraded})"
        )
        return result

    # ────────────────────────────────────────────────────────────────────
    # Map stage
    # ────────────────────────────────────────────────────────────────────
    def _summarize_chunks(
        self,
        chunks: List[Chunk],
        provider: SummarizationProvider,
        model_ready: bool,
    ) -> Dict[str, str]:
        """
        Summarise each chunk, keyed by chunk_id.

        Every model summary is checked for numeric fidelity before it is accepted:
        a summary that introduces a monetary amount, percentage or period absent
        from its source has altered the contract's terms, and for a legal document
        that is worse than a blunter but faithful extract. Such a summary is
        replaced with the extractive one, which cannot invent anything.
        """
        if not model_ready:
            return {chunk.chunk_id: extractive_summary(chunk.text) for chunk in chunks}

        texts = [chunk.text for chunk in chunks]
        summaries = provider.summarize_batch(
            texts,
            max_output_tokens=settings.SUMMARIZATION_MAX_OUTPUT_TOKENS,
            instruction=CHUNK_INSTRUCTION,
        )

        accepted: Dict[str, str] = {}
        invented_count = 0
        for chunk, summary in zip(chunks, summaries):
            summary = (summary or "").strip()
            if not summary:
                accepted[chunk.chunk_id] = extractive_summary(chunk.text)
                continue
            reason = reject_reason(summary, chunk.text)
            if reason:
                invented_count += 1
                logger.warning(
                    f"[Summarization] {chunk.chunk_id} summary rejected — {reason}; "
                    "using the extractive summary instead."
                )
                accepted[chunk.chunk_id] = extractive_summary(chunk.text)
            else:
                accepted[chunk.chunk_id] = summary

        if invented_count:
            logger.warning(
                f"[Summarization] {invented_count}/{len(chunks)} chunk summaries "
                "failed the numeric fidelity check."
            )

        # Track how many chunks the model actually summarised. A run where the
        # model produced nothing usable returns verbatim source text, which looks
        # to a reader exactly like a broken summariser — so it must be reported
        # rather than passed off as a successful summary.
        self._model_summary_count = sum(
            1 for chunk, summary in zip(chunks, summaries)
            if (summary or "").strip() and accepted[chunk.chunk_id] == summary.strip()
        )
        return accepted

    # ────────────────────────────────────────────────────────────────────
    # Reduce stages
    # ────────────────────────────────────────────────────────────────────
    def _build_section_summaries(
        self,
        document: CanonicalDocument,
        chunk_summaries: Dict[str, str],
        provider: SummarizationProvider,
        model_ready: bool,
    ) -> List[SectionSummary]:
        """
        Group chunk summaries by section and reduce each group.

        A section with one chunk needs no further reduction — its chunk summary
        *is* the section summary, and re-summarising a summary only loses detail.
        """
        by_section: Dict[str, List[Chunk]] = {}
        for chunk in document.chunks:
            by_section.setdefault(chunk.section_id, []).append(chunk)

        # Preserve document order.
        ordered_ids = sorted(
            by_section, key=lambda sid: min(c.order for c in by_section[sid])
        )

        sections: List[SectionSummary] = []
        for section_id in ordered_ids:
            group = sorted(by_section[section_id], key=lambda c: c.order)
            first = group[0]
            parts = [chunk_summaries.get(c.chunk_id, "") for c in group]
            parts = [p for p in parts if p]
            if not parts:
                continue

            if len(parts) == 1:
                text = parts[0]
            elif model_ready:
                combined = "\n\n".join(parts)
                reduced = provider.summarize(
                    combined,
                    max_output_tokens=settings.SUMMARIZATION_MAX_OUTPUT_TOKENS,
                    instruction=GROUP_INSTRUCTION,
                )
                reason = reject_reason(reduced, combined)
                if reason:
                    logger.warning(
                        f"[Summarization] section reduce rejected — {reason}; "
                        "keeping the chunk summaries."
                    )
                    text = " ".join(parts)
                else:
                    text = reduced
            else:
                text = " ".join(parts)

            sections.append(SectionSummary(
                section_id=section_id,
                section_number=first.section_number,
                section_title=first.section_title or "Section",
                summary=tidy_summary(text, first.section_title, first.section_number),
                key_points=derive_key_points(text, group),
                page_start=min(c.page_start for c in group),
                page_end=max(c.page_end for c in group),
                chunk_ids=[c.chunk_id for c in group],
            ))

        return sections

    def _build_executive_summary(
        self,
        sections: List[SectionSummary],
        provider: SummarizationProvider,
        model_ready: bool,
    ) -> str:
        """
        Reduce section summaries to one executive summary.

        Long documents get an intermediate grouping pass so the final input stays
        inside the model's window without truncating away the later sections.
        """
        if not sections:
            return ""

        texts = [
            f"{s.section_number} {s.section_title}: {s.summary}".strip()
            for s in sections if s.summary
        ]
        if not texts:
            return ""

        if not model_ready:
            # Lead with the first sections, which in a contract carry the purpose
            # and the parties.
            return clip_to_sentence(" ".join(texts[:3]), 1200)

        if len(texts) > settings.SUMMARY_HIERARCHY_THRESHOLD:
            group_size = max(2, settings.SUMMARY_GROUP_SIZE)
            grouped: List[str] = []
            for start in range(0, len(texts), group_size):
                block = "\n\n".join(texts[start:start + group_size])
                summary = provider.summarize(
                    block,
                    max_output_tokens=settings.SUMMARIZATION_MAX_OUTPUT_TOKENS,
                    instruction=GROUP_INSTRUCTION,
                )
                reason = reject_reason(summary, block)
                if reason:
                    logger.warning(f"[Summarization] group reduce rejected — {reason}")
                    grouped.append(clip_to_sentence(block, 600))
                else:
                    grouped.append(summary)
            texts = grouped
            logger.info(
                f"[Summarization] intermediate reduce: "
                f"{len(sections)} sections -> {len(texts)} group(s)"
            )

        final = provider.summarize(
            "\n\n".join(texts),
            max_output_tokens=settings.SUMMARIZATION_MAX_OUTPUT_TOKENS * 2,
            instruction=FINAL_INSTRUCTION,
        )
        joined = "\n\n".join(texts)
        final_reason = reject_reason(final, joined)
        if not final_reason:
            return final
        logger.warning(f"[Summarization] executive summary rejected — {final_reason}")
        # The reduce produced nothing usable (a weak checkpoint often just echoes
        # its input, which the echo guard rejects). Fall back to the leading
        # section summaries, cut at a sentence boundary rather than mid-word.
        logger.info(
            "[Summarization] composing the executive summary from the section "
            "summaries instead."
        )
        return clip_to_sentence(" ".join(texts[:3]), 1200)

    @staticmethod
    def _build_overview(document: CanonicalDocument, result: SummaryResult) -> str:
        """
        A factual one-liner about the document itself.

        Assembled from counted facts, so there is nothing here a model could get
        wrong.
        """
        parts = [f"{result.document_type or 'Legal document'}"]
        if document.page_count:
            parts.append(f"{document.page_count} page{'s' if document.page_count != 1 else ''}")
        if result.parties:
            names = ", ".join(p.value for p in result.parties[:2] if p.value)
            if names:
                parts.append(f"between {names}")
        if document.sections:
            parts.append(f"{len(document.sections)} sections")
        return " · ".join(parts)

    @staticmethod
    def _build_important_clauses(
        sections: List[SectionSummary],
        chunks: List[Chunk],
        clause_types: Optional[Dict[str, str]] = None,
    ) -> List[SummaryItem]:
        """
        Surface the sections that matter commercially, with their pages.

        `clause_types` maps chunk_id → category when clause detection has run;
        otherwise the section title is used. No risk judgement is made here.
        """
        priority = {
            "Termination", "Liability", "Indemnification",
            "Payment", "Confidentiality", "IP Rights", "Governing Law",
        }
        type_by_section: Dict[str, str] = {}
        if clause_types:
            for chunk in chunks:
                category = clause_types.get(chunk.chunk_id)
                if category and category != "General":
                    type_by_section.setdefault(chunk.section_id, category)

        items: List[SummaryItem] = []
        for section in sections:
            category = type_by_section.get(section.section_id, "")
            if not category:
                from app.services.ai.processor import classify_clause_type

                category = classify_clause_type(section.summary, section.section_title)
            if category in priority:
                items.append(SummaryItem(
                    type=category,
                    label=section.section_label or category,
                    summary=section.summary,
                    source_pages=list(range(section.page_start, section.page_end + 1)),
                    section_title=section.section_title,
                ))
        return items[:12]


# ─────────────────────────────────────────────────────────────────────────────
# Extractive fallback — used when no model is available
# ─────────────────────────────────────────────────────────────────────────────
# Words that carry no topic signal, excluded when measuring grounding.
_STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to for from by
with without within into over under between among as is are was were be been being
shall will may must can could would should have has had do does did not no nor so
such any all each other another its it their his her our your there here which who
whom whose what when where why how each either neither both per upon
""".split())

# Minimum share of a summary's content words that must also appear in its source.
# Calibrated on real output: a faithful paraphrase of a clause scores 75-100%,
# while fabricated prose that merely reuses the parties' names scores around 45%.
# 0.6 separates them with margin on both sides.
MIN_GROUNDING_RATIO = float(os.getenv("SUMMARY_MIN_GROUNDING", "0.6"))


def content_words(text: str) -> set:
    """Lowercased content words of `text`, stopwords and short tokens removed."""
    import re

    words = re.findall(r"[a-z][a-z'\-]{2,}", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS}


def grounding_ratio(summary: str, source: str) -> float:
    """
    Share of the summary's content words that appear in the source.

    1.0 means every meaningful word came from the source; near 0 means the model
    wrote about something else. This is the guard that catches a checkpoint
    fine-tuned on the wrong domain — one that emits fluent, plausible text with
    no relationship to the document it was given. Numeric fidelity checks cannot
    see that, because fabricated prose contains no conflicting figures.
    """
    summary_words = content_words(summary)
    if not summary_words:
        return 1.0
    source_words = content_words(source)
    if not source_words:
        return 0.0
    return len(summary_words & source_words) / len(summary_words)


def reject_reason(candidate: str, source: str) -> Optional[str]:
    """
    Why `candidate` is not an acceptable summary of `source`, or None if it is.

    Applied at every level of the hierarchy — chunk, group, section and executive.
    An earlier version checked grounding on the reduce stages but only checked
    figures on the chunk stage, which let a fabricated executive summary through
    carrying a date the contract never contained.
    """
    text = (candidate or "").strip()
    if not text:
        return "empty"

    grounding = grounding_ratio(text, source)
    if grounding < MIN_GROUNDING_RATIO:
        return f"not grounded in the source ({grounding:.0%} of content words appear in it)"

    invented = find_invented_values(text, source)
    if invented:
        return f"introduces values absent from the source {invented}"

    mispaired = find_mispaired_values(text, source)
    if mispaired:
        return f"attaches figures to the wrong terms {mispaired}"

    return None


def clip_to_sentence(text: str, limit: int) -> str:
    """
    Cut `text` to at most `limit` characters, ending at a sentence boundary.

    A blind character slice ends mid-word, which looks like corrupted output.
    """
    import re

    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if len(cleaned) <= limit:
        return cleaned
    window = cleaned[:limit]
    cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if cut > limit // 3:
        return window[:cut + 1].strip()
    # No sentence boundary to use — cut on a word and mark the truncation.
    space = window.rfind(" ")
    return (window[:space] if space > 0 else window).rstrip(" ,;:-") + "..."


def tidy_summary(text: str, section_title: str = "", section_number: str = "") -> str:
    """
    Make a generated summary presentable.

    Extractive checkpoints tend to repeat the section heading they were given and
    keep the source's hard line breaks, so a summary arrives as
    "Confidentiality\n\n4.1 Each party shall...". The heading is already shown
    beside the summary in the UI, so repeating it wastes the reader's first line.
    """
    import re

    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if not cleaned:
        return ""

    # Drop a leading repeat of the heading, with or without its number.
    for prefix in filter(None, [
        f"{section_number} {section_title}".strip(),
        section_title,
        section_number,
    ]):
        if cleaned.lower().startswith(prefix.lower()):
            candidate = cleaned[len(prefix):].lstrip(" .:-\u2013\u2014")
            # Only if something substantial survives.
            if len(candidate) >= 40:
                cleaned = candidate
            break
    return cleaned


# "Ten Thousand Dollars ($10,000 USD)" — contracts state an amount in words and
# then in figures. A model that keeps the words but swaps the figure produces a
# sentence where every value exists in the source, so a value-level check passes
# while the meaning is wrong.
_WORDED_AMOUNT = re.compile(
    r"((?:[A-Z][a-z]+|and|[\w-]+)(?:\s+(?:[A-Z][a-z]+|and|[\w-]+)){0,5}?\s*"
    r"(?:Dollars?|Rupees?|Euros?|Pounds?|percent|per\s?cent))\s*"
    r"\(?\s*([$₹£€]?\s?[\d,]+(?:\.\d+)?\s?%?)",
    re.IGNORECASE,
)


def find_mispaired_values(summary: str, source: str) -> List[str]:
    """
    Figures that are paired with the wrong words.

    Catches the failure a value-level check cannot see: the summary keeps a
    spelled-out amount from the contract but attaches a different figure to it
    ("Ten Thousand Dollars ($240,000 USD)"). Both values exist in the source, so
    nothing was invented — but the term has been altered, which for a contract is
    the same kind of error.
    """
    def normalise_number(value: str) -> str:
        return re.sub(r"[^\d.]", "", value or "")

    source_pairs: dict = {}
    for words, figure in _WORDED_AMOUNT.findall(source):
        key = re.sub(r"\s+", " ", words).strip().lower()
        source_pairs.setdefault(key, set()).add(normalise_number(figure))

    mispaired: List[str] = []
    for words, figure in _WORDED_AMOUNT.findall(summary):
        key = re.sub(r"\s+", " ", words).strip().lower()
        expected = source_pairs.get(key)
        if not expected:
            continue
        if normalise_number(figure) not in expected:
            mispaired.append(f"{words.strip()} -> {figure.strip()}")
    return mispaired


def find_invented_values(summary: str, source: str) -> List[str]:
    """
    Values present in `summary` but absent from `source`.

    Compares only the categories where a wrong value changes the contract:
    monetary amounts, percentages, day/month/year periods and dates. Numbers are
    normalised (commas and spacing stripped) so "$10,000" and "$10000" match, and
    a value that appears anywhere in the source counts as supported.
    """
    import re

    source_values = extract_preserved_values(source)
    summary_values = extract_preserved_values(summary)

    def normalise(value: str) -> str:
        return re.sub(r"[\s,]", "", value).lower()

    supported = {
        normalise(value)
        for values in source_values.values()
        for value in values
    }
    # Bare digit runs in the source also support a figure in the summary, since a
    # contract writes "thirty (30) days" and a summary may render it "30 days".
    supported |= {normalise(n) for n in re.findall(r"\d[\d,.]*", source)}

    invented: List[str] = []
    for category in ("money", "percent", "days", "dates"):
        for value in summary_values.get(category, []):
            candidate = normalise(value)
            if candidate in supported:
                continue
            # A period like "30 days" is supported if its number appears at all.
            digits = re.sub(r"[^\d.]", "", candidate)
            if digits and digits in supported:
                continue
            invented.append(value)
    return invented


def extractive_summary(text: str, max_sentences: int = 3) -> str:
    """
    Pick the most informative sentences without any model.

    Scores sentences by the density of legally significant signals — obligation
    verbs, amounts, dates, conditionals. Crude, but it never invents anything,
    which is the property that matters for a fallback.
    """
    import re

    if not text or not text.strip():
        return ""

    sentences = [s.strip() for s in re.split(r"(?<=[.;])\s+", text) if len(s.strip()) > 30]
    if not sentences:
        return text.strip()[:300]

    signals = (
        "shall", "must", "may not", "shall not", "agrees", "terminate", "notice",
        "liability", "indemnif", "confidential", "pay", "fee", "interest",
        "unless", "provided that", "subject to", "except", "governing law",
    )

    scored = []
    for index, sentence in enumerate(sentences):
        lowered = sentence.lower()
        score = sum(2 for signal in signals if signal in lowered)
        score += len(re.findall(r"[$₹£€]\s?[\d,]+|\b\d+\s*%|\b\d+\s+days?\b", lowered))
        # Slight preference for earlier sentences, which usually state the rule.
        score += max(0, 3 - index)
        scored.append((score, index, sentence))

    scored.sort(key=lambda row: (-row[0], row[1]))
    chosen = sorted(scored[:max_sentences], key=lambda row: row[1])
    return " ".join(sentence for _, _, sentence in chosen)


def derive_key_points(summary_text: str, chunks: List[Chunk]) -> List[str]:
    """
    Two or three concrete bullets for a section.

    Taken from the source chunks rather than the summary, so the figures quoted
    are the contract's own.
    """
    import re

    points: List[str] = []
    seen = set()
    for chunk in chunks:
        for sentence in re.split(r"(?<=[.;])\s+", chunk.text):
            sentence = sentence.strip()
            if not (40 <= len(sentence) <= 200):
                continue
            has_value = re.search(
                r"[$₹£€]\s?[\d,]+|\b\d+\s*%|\b\d+\s+(?:calendar |business )?days?\b"
                r"|\b\d+\s+(?:months?|years?)\b",
                sentence, re.IGNORECASE,
            )
            has_duty = re.search(r"\b(?:shall|must|may not|shall not)\b", sentence, re.IGNORECASE)
            if not (has_value or has_duty):
                continue
            key = sentence[:60].lower()
            if key in seen:
                continue
            seen.add(key)
            points.append(re.sub(r"\s+", " ", sentence))
            if len(points) >= 3:
                return points
    return points


summarization_service = SummarizationService()
