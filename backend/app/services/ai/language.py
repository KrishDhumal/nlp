"""
Language detection for the canonical pipeline.

Detection runs *after* extraction and *before* summarisation, so the rest of the
pipeline — cleaning, structure detection, chunking, embedding, clause and entity
extraction — is shared across languages exactly as before. Only the choice of
summarisation model depends on the result.

Script detection is used rather than a statistical language-ID dependency:
Devanagari has its own Unicode block, so counting characters in it is both exact
and free. That distinguishes Hindi from English reliably, which is the decision
this pipeline actually needs to make.

Deliberately *not* done: translating Hindi to English, summarising, and
translating back. Each hop loses the exact figures and defined terms a contract
turns on, which is the opposite of what this pipeline is built to protect.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict

# Unicode blocks. Devanagari covers Hindi, Marathi, Nepali and Sanskrit.
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")

# Hindi sentences end with a danda, not a full stop.
DANDA = "।"
DOUBLE_DANDA = "॥"

SUPPORTED_LANGUAGES = ("en", "hi")
DEFAULT_LANGUAGE = "en"

# Share of script-bearing characters that must be Devanagari to call it Hindi.
_HINDI_THRESHOLD = 0.30
# Below this many script characters there is nothing to judge.
_MIN_CHARS = 30


@dataclass
class LanguageResult:
    language: str
    confidence: float
    script_counts: Dict[str, int]

    @property
    def is_hindi(self) -> bool:
        return self.language == "hi"

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "confidence": round(self.confidence, 3),
            "script_counts": self.script_counts,
        }


def detect_language(text: str) -> LanguageResult:
    """
    Identify the document's language from its script.

    Returns English for anything that is not predominantly Devanagari, including
    empty input — English is the safe default because the rest of the stack is
    tuned for it, and a wrong "Hindi" verdict would route the document to a model
    that cannot read it.
    """
    sample = (text or "")[:20000]
    devanagari = len(_DEVANAGARI.findall(sample))
    latin = len(_LATIN.findall(sample))
    total = devanagari + latin

    counts = {"devanagari": devanagari, "latin": latin}

    if total < _MIN_CHARS:
        return LanguageResult(DEFAULT_LANGUAGE, 0.0, counts)

    hindi_share = devanagari / total
    if hindi_share >= _HINDI_THRESHOLD:
        return LanguageResult("hi", hindi_share, counts)
    return LanguageResult("en", 1.0 - hindi_share, counts)


def split_sentences(text: str) -> list:
    """
    Split on sentence boundaries for either language.

    Hindi marks a sentence end with a danda; a contract in Hindi still uses
    Latin digits and may mix in English defined terms, so both terminators are
    honoured in one pass.
    """
    if not text:
        return []
    parts = re.split(rf"(?<=[.!?{DANDA}{DOUBLE_DANDA}])\s+", text)
    return [p.strip() for p in parts if p and p.strip()]


# Devanagari has no case, so a "content word" is simply a run of letters in
# either script. The English stopword list is extended with the Hindi function
# words that carry no topic signal.
_WORD = re.compile(r"[A-Za-zऀ-ॿ][A-Za-z'ऀ-ॿ\-]{1,}")

HINDI_STOPWORDS = frozenset("""
और या के की का को में से पर है हैं था थे थी हो होगा होगी कि यह वह जो इस उस
एक दो नहीं भी तो ही कोई कुछ सभी अपने अपनी द्वारा लिए साथ तक बाद पहले यदि
जब तब जैसे ऐसे वाले वाली रूप प्रकार अनुसार संबंधित गया गयी गए करने करना
""".split())


def content_words(text: str, stopwords: frozenset = frozenset()) -> set:
    """
    Content words of `text`, in either script.

    Used by the grounding check. An ASCII-only pattern would find no words at
    all in a Hindi summary, score its grounding as zero, and reject every
    legitimate Hindi summary produced — so this must stay script-aware.
    """
    words = {w.lower() for w in _WORD.findall(text or "")}
    return {w for w in words if w not in stopwords and w not in HINDI_STOPWORDS}
