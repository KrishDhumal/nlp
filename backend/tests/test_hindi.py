"""
Hindi / multilingual support.

The requirement these guard: the pipeline must handle a Hindi document in its
own language, must not destabilise English, and must never route Hindi through
an English translation hop — each hop loses the figures and defined terms a
contract turns on.

Run with:  PYTHONPATH=. ./venv/bin/python tests/test_hindi.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=True)

from app.core.config import settings
from app.services.ai.chunking import TokenAwareChunker, heuristic_token_count
from app.services.ai.extraction import legal_extractor
from app.services.ai.language import (
    DEFAULT_LANGUAGE,
    content_words,
    detect_language,
    split_sentences,
)
from app.services.ai.processor import DocumentProcessor
from app.services.ai.summarization.registry import get_provider_for_language
from app.services.ai.summarization.service import extractive_summary, grounding_ratio
from tests.fixtures import write_hindi_contract, write_sample_contract

PASSED, FAILED = [], []


def check(label, condition, detail=""):
    (PASSED if condition else FAILED).append(label)
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))


HINDI = ("यह सेवा अनुबंध दिनांक 1 अक्टूबर 2026 को प्रभावी होगा। कोई भी पक्ष साठ (60) "
         "दिनों की लिखित सूचना देकर इसे समाप्त कर सकता है।")
ENGLISH = ("This Agreement is effective October 1, 2026. Either party may terminate "
           "upon sixty (60) days written notice.")


def test_detection():
    print("\n── 1. Language detection ─────────────────────────────────")
    check("Hindi is detected", detect_language(HINDI).language == "hi")
    check("English is detected", detect_language(ENGLISH).language == "en")
    check("detection is confident on pure Hindi",
          detect_language(HINDI).confidence > 0.9,
          f"{detect_language(HINDI).confidence:.2f}")

    # A Hindi contract routinely contains Latin digits and English defined terms.
    mixed = "यह अनुबंध (the \"Agreement\") दिनांक 1 October 2026 को प्रभावी होगा।"
    check("mixed Hindi/English is still Hindi", detect_language(mixed).language == "hi",
          str(detect_language(mixed).script_counts))

    check("empty input defaults to English, never Hindi",
          detect_language("").language == DEFAULT_LANGUAGE)
    check("a short fragment defaults to English rather than guessing",
          detect_language("ok").language == DEFAULT_LANGUAGE)
    check("an English doc with one Hindi word stays English",
          detect_language(ENGLISH + " नोट").language == "en")


def test_text_handling():
    print("\n── 2. Script-aware text handling ─────────────────────────")
    # The danda ends a Hindi sentence; splitting on "." alone sees one sentence.
    check("Hindi sentences split on the danda", len(split_sentences(HINDI)) == 2,
          str(len(split_sentences(HINDI))))
    check("English sentences still split correctly",
          len(split_sentences(ENGLISH)) == 2, str(len(split_sentences(ENGLISH))))

    # This is the regression that would silently disable Hindi: an ASCII-only
    # word pattern finds nothing, scores grounding at zero, and the guard then
    # rejects every Hindi summary as ungrounded.
    check("Hindi content words are found", len(content_words(HINDI)) > 5,
          str(len(content_words(HINDI))))
    check("a Hindi extract is grounded in its source",
          grounding_ratio(HINDI, HINDI) == 1.0)
    check("unrelated Hindi text is not grounded",
          grounding_ratio("कंपनी ने नया कार्यालय खोला। बाजार में वृद्धि हुई।", HINDI) < 0.6,
          f"{grounding_ratio('कंपनी ने नया कार्यालय खोला। बाजार में वृद्धि हुई।', HINDI):.0%}")
    check("the Hindi extractive fallback returns Hindi",
          any("ऀ" <= ch <= "ॿ" for ch in extractive_summary(HINDI)))


def test_pipeline():
    print("\n── 3. Hindi document through the pipeline ────────────────")
    processor = DocumentProcessor(
        chunker=TokenAwareChunker(count_tokens=heuristic_token_count)
    )
    with tempfile.TemporaryDirectory() as tmp:
        hindi_doc = processor.process(
            write_hindi_contract(os.path.join(tmp, "hi.pdf")), "d", "u", "hi.pdf")
        english_doc = processor.process(
            write_sample_contract(os.path.join(tmp, "en.pdf")), "d2", "u", "en.pdf")

    check("the Hindi document is tagged hi", hindi_doc.language == "hi")
    check("the English document is still tagged en", english_doc.language == "en")
    check("Hindi text was extracted", hindi_doc.has_usable_text(),
          f"{len(hindi_doc.full_text)} chars")
    check("Hindi sections are detected", len(hindi_doc.sections) >= 5,
          f"{len(hindi_doc.sections)} sections")
    check("Hindi structure detection succeeds", hindi_doc.structure_detected)
    check("Hindi chunks are produced", len(hindi_doc.chunks) > 0,
          f"{len(hindi_doc.chunks)} chunks")
    check("Hindi chunks carry page numbers",
          all(c.page_start >= 1 for c in hindi_doc.chunks))

    # English must be untouched by all of this.
    check("English sections are unchanged", len(english_doc.sections) == 12,
          f"{len(english_doc.sections)} sections")
    check("English structure detection still succeeds", english_doc.structure_detected)

    # Figures survive: a translation round-trip is exactly what would lose these.
    text = hindi_doc.full_text
    for value in ("5,00,000", "30", "60", "1.5"):
        check(f"the figure {value} survives extraction", value in text)
    check("the Devanagari text is not transliterated",
          any("ऀ" <= ch <= "ॿ" for ch in text))

    # Deterministic extraction is language-agnostic for numeric facts.
    facts = legal_extractor.extract_all(hindi_doc.chunks)
    check("financial values are extracted from Hindi",
          len(facts["financial_terms"]) > 0,
          str([f.value for f in facts["financial_terms"]][:3]))


def test_routing():
    print("\n── 4. Model routing ──────────────────────────────────────")
    english = get_provider_for_language("en")
    hindi = get_provider_for_language("hi")
    check("English routes to the configured default",
          english.model_name == settings.SUMMARIZATION_MODEL, english.model_name)
    check("Hindi routes to the Hindi model",
          hindi.model_name == settings.HINDI_SUMMARIZATION_MODEL, hindi.model_name)
    check("the two providers are different objects", english is not hindi)
    check("providers are cached per language",
          get_provider_for_language("hi") is hindi)
    check("an unknown language falls back to the default",
          get_provider_for_language("fr").model_name == settings.SUMMARIZATION_MODEL)
    check("no translation step is configured anywhere",
          "translate" not in settings.HINDI_SUMMARIZATION_MODEL.lower())


def main_():
    print("=" * 74)
    print("Hindi / multilingual support")
    print("=" * 74)
    test_detection()
    test_text_handling()
    test_pipeline()
    test_routing()
    print("\n" + "=" * 74)
    print(f"{len(PASSED)} passed, {len(FAILED)} failed")
    for label in FAILED:
        print(f"  FAILED: {label}")
    print("=" * 74)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main_())
