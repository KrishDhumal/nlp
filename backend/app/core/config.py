import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Auditor AI Backend"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "fallback_secret_key_if_not_in_env")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    DATABASE_NAME: str = os.getenv("DATABASE_NAME", "auditor_ai")
    CORS_ORIGIN: str = os.getenv("CORS_ORIGIN", "")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "uploads")
    GOOGLE_CLIENT_ID: str = os.getenv("GOOGLE_CLIENT_ID", "")

    # ── Embeddings / vector store ────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # ── PDF ingestion limits ─────────────────────────────────────────────
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "20"))
    # A PDF yielding less than this much text is treated as scanned, not empty.
    MIN_TEXT_CHARS: int = int(os.getenv("MIN_TEXT_CHARS", "200"))

    # ── Token-aware chunking (replaces character-based splitting) ────────
    # These are the primary constraint. Sized so a chunk fits comfortably inside
    # the summarisation model's input window with room for the instruction.
    CHUNK_MAX_TOKENS: int = int(os.getenv("CHUNK_MAX_TOKENS", "768"))
    CHUNK_OVERLAP_TOKENS: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "64"))
    CHUNK_MIN_TOKENS: int = int(os.getenv("CHUNK_MIN_TOKENS", "32"))

    # ── Summarisation ────────────────────────────────────────────────────
    # "local" (default, no API key needed) or "gemini".
    SUMMARY_PROVIDER: str = os.getenv("SUMMARY_PROVIDER", "local")
    # The checkpoint must be fine-tuned for summarisation *of documents like
    # these*, which is a stricter requirement than "legal" or "long context":
    #   - `allenai/led-base-16384` is only pretrained and echoes its input.
    #   - `nsi319/legal-led-base-16384` is fine-tuned on SEC litigation releases,
    #     and emits fluent press-release prose about court judgments no matter
    #     what contract it is given — fabrication, not summarisation.
    # This default is a small summarisation fine-tune that stays close to its
    # source, which is the property that matters for a legal document. It is
    # modest (60M parameters) and largely extractive; see the README for the
    # trade-off and how to substitute a larger model.
    SUMMARIZATION_MODEL: str = os.getenv(
        "SUMMARIZATION_MODEL", "Falconsai/text_summarization"
    )
    # Tried in order if SUMMARIZATION_MODEL cannot be loaded (a stalled or rate
    # limited first-run download is the common case). Comma-separated; each must
    # be fine-tuned for summarisation. Set to an empty string to disable.
    SUMMARIZATION_FALLBACK_MODELS: str = os.getenv(
        "SUMMARIZATION_FALLBACK_MODELS",
        "sshleifer/distilbart-cnn-12-6",
    )
    # Fine-tuned seq2seq summarisers are trained on raw article text, so an
    # instruction prefix pollutes the input and gets copied into the output.
    # Instruction-following providers (Gemini) set this True.
    SUMMARIZATION_USE_INSTRUCTION: bool = (
        os.getenv("SUMMARIZATION_USE_INSTRUCTION", "false").lower() == "true"
    )
    # "auto" resolves to cuda, then mps (Apple Silicon), then cpu.
    SUMMARIZATION_DEVICE: str = os.getenv("SUMMARIZATION_DEVICE", "auto")
    SUMMARIZATION_MAX_INPUT_TOKENS: int = int(os.getenv("SUMMARIZATION_MAX_INPUT_TOKENS", "4096"))
    SUMMARIZATION_MAX_OUTPUT_TOKENS: int = int(os.getenv("SUMMARIZATION_MAX_OUTPUT_TOKENS", "256"))
    SUMMARIZATION_MIN_OUTPUT_TOKENS: int = int(os.getenv("SUMMARIZATION_MIN_OUTPUT_TOKENS", "32"))
    SUMMARIZATION_BATCH_SIZE: int = int(os.getenv("SUMMARIZATION_BATCH_SIZE", "2"))
    # Seconds to wait for the model to load (including a first-run download)
    # before giving up and degrading to extractive summaries. Without a ceiling a
    # stalled download leaves every upload stuck on "summarizing" forever.
    SUMMARIZATION_LOAD_TIMEOUT: int = int(os.getenv("SUMMARIZATION_LOAD_TIMEOUT", "300"))
    # Per-file HTTP timeout for Hugging Face downloads.
    HF_DOWNLOAD_TIMEOUT: int = int(os.getenv("HF_HUB_DOWNLOAD_TIMEOUT", "30"))
    # Chunks per group at the reduce stage of hierarchical summarisation.
    SUMMARY_GROUP_SIZE: int = int(os.getenv("SUMMARY_GROUP_SIZE", "5"))
    # Documents with fewer chunks than this skip the intermediate reduce level.
    SUMMARY_HIERARCHY_THRESHOLD: int = int(os.getenv("SUMMARY_HIERARCHY_THRESHOLD", "8"))
    # Recorded with every stored summary so results stay attributable.
    PIPELINE_VERSION: str = "2.0.0"
    PROMPT_VERSION: str = "1.0.0"

    # ── Optional Gemini analysis (clauses / risk) ─────────────────────────
    # The audit degrades gracefully when this is absent; summarisation does not
    # depend on it at all.
    AUDIT_CONTEXT_CHUNKS: int = int(os.getenv("AUDIT_CONTEXT_CHUNKS", "28"))

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
