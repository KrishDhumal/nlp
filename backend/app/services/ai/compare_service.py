"""
CompareService — version comparison over the canonical document structure.

A character diff of two contracts is almost useless: reformatting shows up as
change, and a renumbered clause reads as a deletion plus an insertion. This
compares *structure* instead:

    Document → sections → clauses → normalised text → aligned pairs

Alignment is three-pass. Sections are matched first on their number and title,
then on clause type, then on text similarity, so a clause that moved or was
renumbered is still recognised as the same clause. Only then is the text of each
aligned pair compared.

On top of the textual diff, the service extracts the *terms* that matter
commercially — amounts, periods, percentages, jurisdictions, liability posture —
and reports when one of them changed value. That is what turns
"the termination clause was modified" into "notice went from 30 days to 60 days".

Everything here is deterministic and reads the canonical chunks both documents
already produced. No PDF is re-parsed and no model is required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple

from app.core.logging import logger
from app.domain.document import Chunk
from app.services.ai.clauses import DetectedClause, detect_clauses

CHANGE_TYPES = ("ADDED", "REMOVED", "MODIFIED", "UNCHANGED")
SEVERITIES = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

# Two sections are the same section above this similarity.
_ALIGN_THRESHOLD = 0.45
# Aligned text above this is treated as unchanged (formatting noise only).
_UNCHANGED_THRESHOLD = 0.985


# ─────────────────────────────────────────────────────────────────────────────
# Commercial terms worth tracking across versions
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TermPattern:
    name: str
    pattern: re.Pattern
    severity: str
    # True when an increase is worse for the reader; False when a decrease is.
    increase_is_worse: bool = True


TERM_PATTERNS: List[TermPattern] = [
    TermPattern("Monetary amount",
                re.compile(r"(?:[$₹£€]|\bUSD|\bINR|\bEUR|\bGBP|\bRs\.?)\s?([\d,]+(?:\.\d{1,2})?)",
                           re.I),
                "HIGH"),
    TermPattern("Percentage",
                re.compile(r"\b(\d{1,3}(?:\.\d{1,2})?)\s?%"), "MEDIUM"),
    TermPattern("Notice period",
                re.compile(r"\b(\d{1,4})\s*\)?\s*(?:calendar |business |working )?days?\b", re.I),
                "HIGH"),
    TermPattern("Duration in months",
                re.compile(r"\b(\d{1,3})\s*\)?\s*months?\b", re.I), "MEDIUM"),
    TermPattern("Duration in years",
                re.compile(r"\b(\d{1,3})\s*\)?\s*years?\b", re.I), "MEDIUM"),
]

# Jurisdiction and liability posture are categorical, not numeric.
_GOVERNING_LAW = re.compile(
    r"laws?\s+of\s+(?:the\s+)?([A-Z][^.;,]{2,50})", re.I,
)
_LIABILITY_UNCAPPED = re.compile(
    r"unlimited liability|no limit(?:ation)? on (?:its )?liability|"
    r"without limitation of liability", re.I,
)
_LIABILITY_CAPPED = re.compile(
    r"shall not exceed|aggregate liability[^.;]{0,60}(?:cap|limited)|liability is capped", re.I,
)


@dataclass
class SectionChange:
    section: str
    change_type: str
    old_text: str = ""
    new_text: str = ""
    severity: str = "LOW"
    page_old: Optional[int] = None
    page_new: Optional[int] = None
    explanation: str = ""
    clause_type: str = ""
    similarity: float = 0.0
    term_changes: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "section": self.section,
            "change_type": self.change_type,
            "old_text": self.old_text,
            "new_text": self.new_text,
            "severity": self.severity,
            "page_old": self.page_old,
            "page_new": self.page_new,
            "explanation": self.explanation,
            "clause_type": self.clause_type,
            "similarity": round(self.similarity, 3),
            "term_changes": self.term_changes,
        }


@dataclass
class ComparisonResult:
    changes: List[SectionChange] = field(default_factory=list)
    summary: str = ""
    counts: Dict[str, int] = field(default_factory=dict)
    risk_shift: int = 0
    risk_shift_label: str = ""
    added_clauses: List[str] = field(default_factory=list)
    removed_clauses: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "difference_summary": self.summary,
            "summary": self.summary,
            "counts": self.counts,
            "risk_shift_score": self.risk_shift,
            "risk_shift_label": self.risk_shift_label,
            "added_clauses": self.added_clauses,
            "removed_clauses": self.removed_clauses,
            "changes": [c.to_dict() for c in self.changes],
        }


def _normalise(text: str) -> str:
    """Collapse formatting so only substantive wording differences remain."""
    text = re.sub(r"\s+", " ", (text or "")).strip().lower()
    # Clause numbering is renumbered between drafts; it is not a wording change.
    text = re.sub(r"\b\d{1,2}(?:\.\d{1,2}){0,3}[.)]\s*", "", text)
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"[“”]", '"', text)
    return text


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, _normalise(a), _normalise(b), autojunk=False).ratio()


@dataclass
class _Section:
    """A section of one document, assembled from its canonical chunks."""
    key: str
    number: str
    title: str
    text: str
    page_start: int
    page_end: int
    clause_type: str = ""


def _build_sections(chunks: Sequence[Chunk],
                    clauses: Sequence[DetectedClause]) -> List[_Section]:
    """Group canonical chunks back into sections, in document order."""
    clause_by_section: Dict[str, str] = {}
    for clause in clauses:
        key = clause.section_number or clause.section
        if key:
            clause_by_section.setdefault(key, clause.clause_type)

    grouped: Dict[str, List[Chunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.section_id, []).append(chunk)

    sections: List[_Section] = []
    for section_id, items in grouped.items():
        items.sort(key=lambda c: c.order)
        first = items[0]
        key = first.section_number or first.section_title or section_id
        sections.append(_Section(
            key=key,
            number=first.section_number or "",
            title=first.section_title or "",
            text=" ".join(c.text for c in items),
            page_start=min(c.page_start for c in items),
            page_end=max(c.page_end for c in items),
            clause_type=clause_by_section.get(key, ""),
        ))
    sections.sort(key=lambda s: (s.page_start, s.number))
    return sections


def _align(a_sections: List[_Section],
           b_sections: List[_Section]) -> Tuple[List[Tuple[_Section, _Section, float]],
                                                List[_Section], List[_Section]]:
    """
    Pair sections across the two versions.

    Pass 1 matches on section number and title, pass 2 on detected clause type,
    pass 3 on text similarity. A section that survives all three unmatched was
    genuinely added or removed.
    """
    pairs: List[Tuple[_Section, _Section, float]] = []
    remaining_a = list(a_sections)
    remaining_b = list(b_sections)

    def take(a: _Section, b: _Section) -> None:
        pairs.append((a, b, _similarity(a.text, b.text)))
        remaining_a.remove(a)
        remaining_b.remove(b)

    # Pass 1 — same number and comparable title.
    for a in list(remaining_a):
        for b in list(remaining_b):
            if a.number and a.number == b.number:
                if not a.title or not b.title or _similarity(a.title, b.title) > 0.5:
                    take(a, b)
                    break

    # Pass 2 — same detected clause type.
    for a in list(remaining_a):
        if not a.clause_type:
            continue
        for b in list(remaining_b):
            if b.clause_type == a.clause_type:
                take(a, b)
                break

    # Pass 3 — best remaining text similarity above the threshold.
    while remaining_a and remaining_b:
        best: Optional[Tuple[float, _Section, _Section]] = None
        for a in remaining_a:
            for b in remaining_b:
                score = _similarity(a.text, b.text)
                if best is None or score > best[0]:
                    best = (score, a, b)
        if best is None or best[0] < _ALIGN_THRESHOLD:
            break
        take(best[1], best[2])

    return pairs, remaining_a, remaining_b


def _numbers(pattern: re.Pattern, text: str) -> List[str]:
    return [m.group(1) for m in pattern.finditer(text or "")]


def _detect_term_changes(old: str, new: str) -> List[dict]:
    """
    Commercial values that changed between two versions of the same clause.

    Reported as old → new so the UI can render "30 days → 60 days" rather than a
    wall of highlighted text.
    """
    changes: List[dict] = []

    for term in TERM_PATTERNS:
        old_values = _numbers(term.pattern, old)
        new_values = _numbers(term.pattern, new)
        if not old_values and not new_values:
            continue
        if old_values == new_values:
            continue

        def as_float(value: str) -> Optional[float]:
            try:
                return float(value.replace(",", ""))
            except ValueError:
                return None

        old_first = as_float(old_values[0]) if old_values else None
        new_first = as_float(new_values[0]) if new_values else None

        direction = ""
        severity = term.severity
        if old_first is not None and new_first is not None and old_first != new_first:
            worse = (new_first > old_first) == term.increase_is_worse
            direction = "increased" if new_first > old_first else "decreased"
            if worse and abs(new_first - old_first) / max(old_first, 1) >= 0.5:
                severity = "CRITICAL" if term.severity == "HIGH" else "HIGH"
        elif not old_values or not new_values:
            direction = "added" if new_values else "removed"

        changes.append({
            "term": term.name,
            "old": ", ".join(old_values[:3]) or None,
            "new": ", ".join(new_values[:3]) or None,
            "direction": direction,
            "severity": severity,
        })

    # Governing law is categorical.
    old_law = _GOVERNING_LAW.search(old or "")
    new_law = _GOVERNING_LAW.search(new or "")
    if old_law and new_law:
        a_value = re.sub(r"\s+", " ", old_law.group(1)).strip(" ,.")
        b_value = re.sub(r"\s+", " ", new_law.group(1)).strip(" ,.")
        if a_value.lower() != b_value.lower():
            changes.append({
                "term": "Governing law", "old": a_value, "new": b_value,
                "direction": "changed", "severity": "HIGH",
            })

    # Liability posture: capped becoming uncapped is the single worst change.
    old_uncapped = bool(_LIABILITY_UNCAPPED.search(old or ""))
    new_uncapped = bool(_LIABILITY_UNCAPPED.search(new or ""))
    old_capped = bool(_LIABILITY_CAPPED.search(old or ""))
    new_capped = bool(_LIABILITY_CAPPED.search(new or ""))
    if (old_capped and new_uncapped) or (old_capped and not new_capped and not new_uncapped):
        changes.append({
            "term": "Liability", "old": "Capped", "new": "Uncapped",
            "direction": "worsened", "severity": "CRITICAL",
        })
    elif old_uncapped and new_capped:
        changes.append({
            "term": "Liability", "old": "Uncapped", "new": "Capped",
            "direction": "improved", "severity": "LOW",
        })

    return changes


def _severity_for(section: _Section, term_changes: List[dict],
                  similarity: float) -> str:
    """Severity of a modification, driven by what actually changed."""
    if term_changes:
        order = {s: i for i, s in enumerate(SEVERITIES)}
        return max((c["severity"] for c in term_changes), key=lambda s: order.get(s, 0))
    high_value_clause = section.clause_type in {
        "Liability", "Indemnification", "Termination", "Payment",
        "Intellectual Property", "Governing Law", "Data Protection",
    }
    if similarity < 0.6:
        return "HIGH" if high_value_clause else "MEDIUM"
    if similarity < 0.9:
        return "MEDIUM" if high_value_clause else "LOW"
    return "LOW"


def _explain(change_type: str, section: _Section, term_changes: List[dict]) -> str:
    """
    A factual sentence describing the change.

    Composed from what was measured, never generated — so it cannot describe a
    change that did not happen.
    """
    label = section.clause_type or section.title or "This section"
    if change_type == "ADDED":
        return f"{label} appears in the new version but not in the old one."
    if change_type == "REMOVED":
        return f"{label} was present in the old version and is absent from the new one."
    if change_type == "UNCHANGED":
        return f"{label} is unchanged apart from formatting."

    if term_changes:
        parts = []
        for change in term_changes[:3]:
            old_value, new_value = change.get("old"), change.get("new")
            if old_value and new_value:
                parts.append(f"{change['term'].lower()} {old_value} → {new_value}")
            elif new_value:
                parts.append(f"{change['term'].lower()} {new_value} added")
            elif old_value:
                parts.append(f"{change['term'].lower()} {old_value} removed")
        if parts:
            return f"{label}: " + "; ".join(parts) + "."
    return f"{label} was reworded between the two versions."


class CompareService:
    """Deterministic, structure-aware version comparison."""

    def compare(
        self,
        a_chunks: Sequence[Chunk],
        b_chunks: Sequence[Chunk],
        a_name: str = "Version A",
        b_name: str = "Version B",
    ) -> ComparisonResult:
        a_clauses = detect_clauses(a_chunks)
        b_clauses = detect_clauses(b_chunks)
        a_sections = _build_sections(a_chunks, a_clauses)
        b_sections = _build_sections(b_chunks, b_clauses)

        pairs, only_a, only_b = _align(a_sections, b_sections)
        changes: List[SectionChange] = []

        for a, b, similarity in pairs:
            term_changes = _detect_term_changes(a.text, b.text)
            if similarity >= _UNCHANGED_THRESHOLD and not term_changes:
                change_type = "UNCHANGED"
                severity = "LOW"
            else:
                change_type = "MODIFIED"
                severity = _severity_for(b, term_changes, similarity)
            changes.append(SectionChange(
                section=f"{b.number} {b.title}".strip() or a.key,
                change_type=change_type,
                old_text=re.sub(r"\s+", " ", a.text)[:1200],
                new_text=re.sub(r"\s+", " ", b.text)[:1200],
                severity=severity,
                page_old=a.page_start, page_new=b.page_start,
                explanation=_explain(change_type, b, term_changes),
                clause_type=b.clause_type or a.clause_type,
                similarity=similarity,
                term_changes=term_changes,
            ))

        for a in only_a:
            changes.append(SectionChange(
                section=f"{a.number} {a.title}".strip() or a.key,
                change_type="REMOVED",
                old_text=re.sub(r"\s+", " ", a.text)[:1200],
                severity="HIGH" if a.clause_type else "MEDIUM",
                page_old=a.page_start,
                explanation=_explain("REMOVED", a, []),
                clause_type=a.clause_type,
            ))

        for b in only_b:
            changes.append(SectionChange(
                section=f"{b.number} {b.title}".strip() or b.key,
                change_type="ADDED",
                new_text=re.sub(r"\s+", " ", b.text)[:1200],
                severity="MEDIUM" if b.clause_type else "LOW",
                page_new=b.page_start,
                explanation=_explain("ADDED", b, []),
                clause_type=b.clause_type,
            ))

        order = {t: i for i, t in enumerate(("MODIFIED", "REMOVED", "ADDED", "UNCHANGED"))}
        severity_order = {s: i for i, s in enumerate(reversed(SEVERITIES))}
        changes.sort(key=lambda c: (severity_order.get(c.severity, 9),
                                    order.get(c.change_type, 9)))

        counts = {t: sum(1 for c in changes if c.change_type == t) for t in CHANGE_TYPES}
        a_types = {c.clause_type for c in a_clauses}
        b_types = {c.clause_type for c in b_clauses}
        added_clauses = sorted(b_types - a_types)
        removed_clauses = sorted(a_types - b_types)

        risk_shift = self._risk_shift(changes, added_clauses, removed_clauses)
        result = ComparisonResult(
            changes=changes,
            counts=counts,
            risk_shift=risk_shift,
            risk_shift_label=self._shift_label(risk_shift),
            added_clauses=added_clauses,
            removed_clauses=removed_clauses,
        )
        result.summary = self._summarise(result, a_name, b_name)
        logger.info(
            f"[Compare] {a_name} vs {b_name}: {counts} risk_shift={risk_shift}"
        )
        return result

    @staticmethod
    def _risk_shift(changes: Sequence[SectionChange],
                    added_clauses: Sequence[str],
                    removed_clauses: Sequence[str]) -> int:
        """
        Negative means the new version is riskier for the reader.

        Weighted by severity of each substantive change, plus the protections
        gained or lost outright.
        """
        weights = {"CRITICAL": 30, "HIGH": 15, "MEDIUM": 7, "LOW": 2}
        shift = 0
        for change in changes:
            if change.change_type == "UNCHANGED":
                continue
            weight = weights.get(change.severity, 5)
            if change.change_type == "REMOVED":
                shift -= weight
            elif change.change_type == "ADDED":
                shift += weight // 2
            else:
                improved = any(c.get("direction") == "improved" for c in change.term_changes)
                shift += weight // 2 if improved else -weight
        shift += 10 * len(added_clauses)
        shift -= 12 * len(removed_clauses)
        return max(-100, min(100, shift))

    @staticmethod
    def _shift_label(shift: int) -> str:
        if shift <= -30:
            return f"{abs(shift)}% higher risk in the new version"
        if shift < 0:
            return f"Slightly higher risk in the new version ({abs(shift)}%)"
        if shift == 0:
            return "No material change in risk"
        if shift < 30:
            return f"Slightly lower risk in the new version ({shift}%)"
        return f"{shift}% lower risk in the new version"

    @staticmethod
    def _summarise(result: ComparisonResult, a_name: str, b_name: str) -> str:
        counts = result.counts
        parts = [
            f"Comparing {a_name} with {b_name}: "
            f"{counts.get('MODIFIED', 0)} section(s) modified, "
            f"{counts.get('ADDED', 0)} added, {counts.get('REMOVED', 0)} removed, "
            f"{counts.get('UNCHANGED', 0)} unchanged."
        ]
        notable = [
            c for c in result.changes
            if c.severity in ("CRITICAL", "HIGH") and c.change_type != "UNCHANGED"
        ][:3]
        for change in notable:
            parts.append(change.explanation)
        if result.removed_clauses:
            parts.append(
                "Protections no longer present: " + ", ".join(result.removed_clauses) + "."
            )
        parts.append(result.risk_shift_label + ".")
        return " ".join(parts)


compare_service = CompareService()
