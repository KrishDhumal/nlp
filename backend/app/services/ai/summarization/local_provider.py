"""
LocalTransformerSummarizer — the default summarisation engine.

Runs a Hugging Face seq2seq model in-process, so summarisation needs no API key,
has no quota, and costs nothing per document. This is what makes the platform
able to summarise long contracts repeatedly.

Model choice: `allenai/led-base-16384` (Longformer Encoder-Decoder). LED is
built for long documents — its sparse attention accepts up to 16k input tokens,
where BART and T5 stop at 1024 — which suits contracts. The base checkpoint is
~650 MB and ~162M parameters, comfortable on a 16 GB machine with no discrete
GPU. A legal-domain fine-tune (for example `nsi319/legal-led-base-16384`) is a
drop-in replacement via `SUMMARIZATION_MODEL`.

Deployment note: the model is held on the instance, so each worker process that
touches summarisation loads its own copy — roughly 1 GB resident for LED-base in
fp32. With N Uvicorn workers that is N copies. For more than two workers, run
summarisation in a dedicated single-worker service rather than in every web
worker.
"""

from __future__ import annotations

import os
import re
import threading
import time
from difflib import SequenceMatcher
from typing import List, Optional

from app.core.config import settings
from app.core.logging import logger
from app.services.ai.summarization.base import (
    CHUNK_INSTRUCTION,
    FINAL_INSTRUCTION,
    GROUP_INSTRUCTION,
    SummarizationProvider,
)


# Output length as a fraction of input length. Abstractive summaries of legal
# text land around a third to a half of the source; below that, obligations and
# figures start being dropped.
SUMMARY_LENGTH_RATIO = 0.45
MIN_TARGET_TOKENS = 32
# How much of the input prefix must be repeated verbatim to count as an echo.
PREFIX_ECHO_CHARS = 120
# Fraction of the output that may be one contiguous copy of the input.
VERBATIM_ECHO_RATIO = 0.8
# A summary longer than this multiple of its source has not summarised anything.
MAX_EXPANSION_RATIO = 1.6
# Shortest summary worth keeping when trimming back to a sentence boundary.
MIN_TRIMMED_CHARS = 40


def trim_to_sentence(text: str) -> str:
    """
    Cut a generated summary back to its last complete sentence.

    Generation stops at a token budget, not at a sentence boundary, so raw output
    routinely ends mid-clause ("...shall accrue interest at one and"). A dangling
    fragment reads as a truncated fact, which is worse than omitting it.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    # Already ends cleanly.
    if cleaned[-1] in ".!?":
        return cleaned
    # Cut at the last sentence terminator, as long as something survives.
    cut = max(cleaned.rfind("."), cleaned.rfind("!"), cleaned.rfind("?"))
    if cut >= MIN_TRIMMED_CHARS:
        return cleaned[:cut + 1].strip()
    # Nothing to cut back to: mark the truncation rather than implying it ended.
    return cleaned.rstrip(" ,;:-") + "..."


def resolve_device(preference: str = "auto") -> str:
    """
    Pick the best available torch device.

    "auto" prefers CUDA, then Apple Silicon's MPS, then CPU. An explicit value is
    honoured as given so a deployment can force CPU.
    """
    preference = (preference or "auto").lower()
    if preference != "auto":
        return preference
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class LocalTransformerSummarizer(SummarizationProvider):
    """
    Seq2seq summarisation with a locally held model.

    The model is loaded once, lazily, on first use and then reused for every
    chunk — never loaded and unloaded per chunk, which would dominate runtime.
    Loading is guarded by a lock so two concurrent requests cannot both pay the
    load cost.
    """

    name = "local"

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_input_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
    ):
        self._model_name = model_name or settings.SUMMARIZATION_MODEL
        self._device_preference = device or settings.SUMMARIZATION_DEVICE
        self.max_input_tokens = max_input_tokens or settings.SUMMARIZATION_MAX_INPUT_TOKENS
        self.max_output_tokens = max_output_tokens or settings.SUMMARIZATION_MAX_OUTPUT_TOKENS
        self.min_output_tokens = settings.SUMMARIZATION_MIN_OUTPUT_TOKENS

        self._model = None
        self._tokenizer = None
        self._device: Optional[str] = None
        self._load_lock = threading.Lock()
        # Only one thread may touch the model at a time: accelerator backends are
        # not thread safe, and the pipeline calls this from a worker thread.
        self._generate_lock = threading.RLock()
        self._on_device = False
        self._load_failed: Optional[str] = None

    # ── Identity ────────────────────────────────────────────────────────
    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        """Transformers version plus the resolved device — enough to reproduce."""
        try:
            import transformers

            return f"transformers-{transformers.__version__}/{self._device or self._device_preference}"
        except Exception:
            return "unknown"

    # ── Loading ─────────────────────────────────────────────────────────
    def _ensure_loaded(self) -> bool:
        """
        Load the model once. Returns False when it cannot be loaded.

        The load is bounded by SUMMARIZATION_LOAD_TIMEOUT: a first run downloads
        several hundred megabytes, and a stalled or rate-limited download would
        otherwise hang the pipeline forever with no diagnosis.
        """
        if self._model is not None:
            return True
        if self._load_failed is not None:
            return False

        with self._load_lock:
            # Another thread may have finished while we waited.
            if self._model is not None:
                return True
            if self._load_failed is not None:
                return False

            problems = []
            for candidate in self._candidate_models():
                outcome = self._try_load(candidate)
                if outcome is None:
                    self._model_name = candidate
                    return True
                problems.append(f"{candidate}: {outcome}")
                logger.warning(f"[LocalSummarizer] {candidate} unavailable — {outcome}")

            self._load_failed = "; ".join(problems) or "no candidate model configured"
            logger.error(
                f"[LocalSummarizer] no summarisation model could be loaded. {self._load_failed}"
            )
            return False

    def _candidate_models(self) -> List[str]:
        """The configured model first, then any configured fallbacks."""
        candidates = [self._model_name]
        for name in (settings.SUMMARIZATION_FALLBACK_MODELS or "").split(","):
            name = name.strip()
            if name and name not in candidates:
                candidates.append(name)
        return candidates

    def _try_load(self, model_name: str) -> Optional[str]:
        """
        Attempt one checkpoint under a deadline.

        Returns None on success, or a short reason string on failure. The load
        runs on a worker thread because a stalled download cannot be interrupted
        any other way, and an unbounded wait would leave every upload stuck on
        "summarizing" indefinitely.
        """
        result: dict = {}

        def worker() -> None:
            try:
                result["ok"] = self._load_now(model_name)
            except Exception as e:  # noqa: BLE001 - surfaced as a reason string
                result["error"] = e

        thread = threading.Thread(
            target=worker, name=f"summarizer-load-{model_name}", daemon=True
        )
        thread.start()
        thread.join(timeout=settings.SUMMARIZATION_LOAD_TIMEOUT)

        if thread.is_alive():
            # The thread is a daemon and will be abandoned; it cannot be killed.
            return (
                f"load exceeded {settings.SUMMARIZATION_LOAD_TIMEOUT}s "
                "(download stalled or rate limited)"
            )
        if "error" in result:
            return f"{type(result['error']).__name__}: {str(result['error'])[:120]}"
        if not result.get("ok"):
            return "load returned no model"
        return None

    def _load_now(self, model_name: str) -> bool:
        """Actually load tokenizer and model. Called on a worker thread."""
        if True:
            try:
                # Bound each HTTP request the hub makes, so a hanging file
                # download surfaces as an error instead of blocking forever.
                os.environ.setdefault(
                    "HF_HUB_DOWNLOAD_TIMEOUT", str(settings.HF_DOWNLOAD_TIMEOUT)
                )

                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

                self._device = resolve_device(self._device_preference)
                started = time.time()
                logger.info(
                    f"[LocalSummarizer] loading '{model_name}' "
                    f"on {self._device} (first use only)..."
                )

                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                model.eval()
                # Deliberately left on CPU here. This runs on the loader thread,
                # and moving the model to MPS on one thread then generating on
                # another aborts the process with an MTLCommandBuffer assertion.
                # The move happens lazily on the thread that first generates.
                self._model = model
                self._torch = torch
                self._on_device = False

                logger.info(
                    f"[LocalSummarizer] ready in {time.time() - started:.1f}s "
                    f"({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params)"
                )
                return True
            except Exception:
                # Reported by _ensure_loaded, which owns the failure state.
                raise

    def is_available(self) -> bool:
        """
        Whether the model can be loaded. This actually attempts the load, so the
        caller learns the truth rather than an optimistic guess.
        """
        return self._ensure_loaded()

    def warm_up(self) -> bool:
        """Pre-load the model, e.g. at application startup."""
        return self._ensure_loaded()

    # ── Summarisation ───────────────────────────────────────────────────
    def summarize(
        self,
        text: str,
        max_output_tokens: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> str:
        results = self.summarize_batch([text], max_output_tokens, instruction)
        return results[0] if results else ""

    def summarize_batch(
        self,
        texts: List[str],
        max_output_tokens: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> List[str]:
        """
        Summarise several texts, batching to keep the accelerator busy.

        A failure returns empty strings rather than raising: the pipeline above
        degrades to extractive output instead of losing the whole document.
        """
        usable = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        results = [""] * len(texts)
        if not usable:
            return results

        if not self._ensure_loaded():
            return results

        limit = max_output_tokens or self.max_output_tokens
        batch_size = max(1, settings.SUMMARIZATION_BATCH_SIZE)

        for start in range(0, len(usable), batch_size):
            batch = usable[start:start + batch_size]
            texts_in = [text for _, text in batch]
            try:
                with self._generate_lock:
                    self._ensure_on_device()
                    try:
                        summaries = self._generate(texts_in, limit, instruction)
                    except Exception as e:
                        # An accelerator fault is recoverable on CPU; anything
                        # else is re-raised to the outer handler.
                        if not self._fall_back_to_cpu(e):
                            raise
                        summaries = self._generate(texts_in, limit, instruction)
                for (index, _), summary in zip(batch, summaries):
                    results[index] = summary
            except Exception as e:
                logger.warning(f"[LocalSummarizer] batch failed: {e}")

        return results

    def _ensure_on_device(self) -> None:
        """
        Move the model onto the target device, on the calling thread.

        Done here rather than at load time so the thread that creates the
        accelerator context is the thread that uses it.
        """
        if self._on_device:
            return
        target = self._device or "cpu"
        if target == "cpu":
            self._on_device = True
            return
        try:
            self._model.to(target)
            self._on_device = True
            logger.info(f"[LocalSummarizer] model moved to {target}")
        except Exception as e:
            logger.warning(
                f"[LocalSummarizer] could not use {target} ({e}) — staying on CPU."
            )
            self._device = "cpu"
            self._on_device = True

    def _fall_back_to_cpu(self, error: Exception) -> bool:
        """
        Switch permanently to CPU after an accelerator failure.

        Returns True when the switch happened and the caller should retry. MPS in
        particular raises Metal assertions under threaded use; CPU is slower but
        always correct, and a working summary beats a crashed worker.
        """
        if self._device == "cpu":
            return False
        logger.warning(
            f"[LocalSummarizer] {self._device} generation failed ({str(error)[:120]}) "
            "— falling back to CPU for the rest of this process."
        )
        self._device = "cpu"
        try:
            self._model.to("cpu")
        except Exception:
            pass
        self._on_device = True
        return True

    def _is_led(self) -> bool:
        return "led" in (self._model_name or "").lower()

    def _needs_task_prefix(self) -> bool:
        """
        Whether this checkpoint is a T5-family model needing a task prefix.

        Detected from the loaded model's own config rather than its name, since
        fine-tune repositories rarely mention the base architecture.
        """
        model_type = ""
        try:
            model_type = (getattr(self._model.config, "model_type", "") or "").lower()
        except Exception:
            pass
        if model_type:
            return model_type in {"t5", "mt5", "longt5", "switch_transformers", "umt5"}
        return "t5" in (self._model_name or "").lower()

    def _pad_to_attention_window(self, encoded):
        """
        Pad a batch up to a multiple of LED's attention window.

        LED does this internally for `input_ids` but the caller's
        `global_attention_mask` is built at the original length, so padding here
        keeps every tensor aligned.
        """
        torch = self._torch
        # LED reports attention_window as a list with one entry per layer, so the
        # list check has to come before the int conversion.
        window = getattr(self._model.config, "attention_window", None) or 1024
        if isinstance(window, (list, tuple)):
            window = window[0] if window else 1024
        window = int(window)
        if window <= 0:
            return encoded
        length = int(encoded["input_ids"].shape[-1])
        remainder = length % window
        if remainder == 0:
            return encoded

        padding = window - remainder
        pad_id = self._tokenizer.pad_token_id or 0
        encoded["input_ids"] = torch.nn.functional.pad(
            encoded["input_ids"], (0, padding), value=pad_id
        )
        if "attention_mask" in encoded:
            encoded["attention_mask"] = torch.nn.functional.pad(
                encoded["attention_mask"], (0, padding), value=0
            )
        return encoded

    def _generate(
        self,
        texts: List[str],
        max_output_tokens: int,
        instruction: Optional[str],
    ) -> List[str]:
        """Tokenise, run the model, decode."""
        torch = self._torch
        # A fine-tuned summariser expects the raw document text. Prepending an
        # instruction makes it copy that instruction into the summary, so this is
        # off unless SUMMARIZATION_USE_INSTRUCTION is set for an
        # instruction-following checkpoint.
        if settings.SUMMARIZATION_USE_INSTRUCTION:
            prompt = instruction or CHUNK_INSTRUCTION
            prepared = [f"{prompt}\n\n{text}" for text in texts]
        else:
            # T5-family checkpoints are multi-task and need their task prefix;
            # without it a T5 summariser will often translate or copy instead.
            # BART and LED take the raw text.
            prefix = "summarize: " if self._needs_task_prefix() else ""
            prepared = [f"{prefix}{text}" for text in texts]

        encoded = self._tokenizer(
            prepared,
            max_length=self.max_input_tokens,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        # Scale the output budget to the actual input. Asking for a fixed 256
        # tokens from a 50-token clause forces the model to pad and ramble; a
        # summary should be a fraction of its source. The longest input in the
        # batch sets the budget, since generation is shared across the batch.
        input_tokens = int(encoded["input_ids"].shape[-1])
        target = int(input_tokens * SUMMARY_LENGTH_RATIO)
        target = max(MIN_TARGET_TOKENS, min(target, max_output_tokens))
        floor = max(8, min(self.min_output_tokens, target // 3))

        generate_kwargs = dict(
            max_new_tokens=target,
            min_new_tokens=floor,
            num_beams=4,
            length_penalty=1.0,
            # Legal text repeats phrases legitimately, so only block long repeats.
            no_repeat_ngram_size=4,
            early_stopping=True,
        )

        # LED needs to be told which tokens get global attention; the first token
        # is the convention for summarisation. LED also requires the sequence
        # length to be a multiple of its attention window and pads internally —
        # so pad here instead, keeping input_ids, attention_mask and
        # global_attention_mask the same length rather than relying on the model
        # to extend a mask it was handed.
        if self._is_led():
            encoded = self._pad_to_attention_window(encoded)
            global_attention_mask = torch.zeros_like(encoded["input_ids"])
            global_attention_mask[:, 0] = 1
            generate_kwargs["global_attention_mask"] = global_attention_mask

        with torch.no_grad():
            output = self._model.generate(**encoded, **generate_kwargs)

        decoded = [
            self._tokenizer.decode(sequence, skip_special_tokens=True).strip()
            for sequence in output
        ]
        return [
            self._strip_echoed_input(trim_to_sentence(summary), source)
            for summary, source in zip(decoded, texts)
        ]

    @staticmethod
    def _normalize(text: str) -> str:
        """Whitespace- and case-insensitive form, for comparing output to input."""
        return re.sub(r"\s+", " ", (text or "")).strip().lower()

    @classmethod
    def _strip_echoed_input(cls, summary: str, source: str) -> str:
        """
        Drop output that merely repeats the prompt or the source text.

        Detection is by *content*, not by length. An earlier version rejected any
        summary that was not shorter in characters than its input, which threw
        away almost every legitimate summary of a short clause — abstractive
        models are asked for a minimum number of tokens, so a 250-character
        clause can honestly summarise to a similar length. Only a genuine echo
        (the output opening with, or largely consisting of, the input) is dropped.
        """
        cleaned = (summary or "").strip()
        if not cleaned:
            return ""

        # 1. A leading copy of the instruction, for instruction-prefixed runs.
        for instruction in (CHUNK_INSTRUCTION, GROUP_INSTRUCTION, FINAL_INSTRUCTION):
            if cls._normalize(cleaned).startswith(cls._normalize(instruction)[:60]):
                cleaned = cleaned[len(instruction):].strip(" \n:-")

        if not cleaned:
            return ""

        normalized_out = cls._normalize(cleaned)
        normalized_in = cls._normalize(source)
        if not normalized_in:
            return cleaned

        # 2. The output opens with the input verbatim — the classic base-model echo.
        probe = normalized_in[:PREFIX_ECHO_CHARS]
        if len(probe) >= 40 and normalized_out.startswith(probe):
            return ""

        # 3. Most of the output is one contiguous verbatim run of the input.
        matcher = SequenceMatcher(None, normalized_out, normalized_in, autojunk=False)
        _, _, longest = matcher.find_longest_match(
            0, len(normalized_out), 0, len(normalized_in)
        )
        if longest and longest / len(normalized_out) >= VERBATIM_ECHO_RATIO:
            return ""

        # 4. A sanity ceiling: a "summary" far longer than its source is not one.
        if len(normalized_out) > len(normalized_in) * MAX_EXPANSION_RATIO:
            return ""

        return cleaned
