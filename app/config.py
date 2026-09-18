"""
Centralized application configuration.

All values are loaded from environment variables (via a .env file when
present) so the application never contains hardcoded paths, models or
secrets. Every setting has a sensible local-first default.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load a .env file if present (no-op in production/Docker where real
# environment variables are already set).
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


class Settings:
    """Application settings, all overridable via environment variables."""

    # --- Storage paths -----------------------------------------------
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
    UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", DATA_DIR / "uploads"))
    CHROMA_DIR: Path = Path(os.getenv("CHROMA_DIR", DATA_DIR / "chroma"))
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "rag_documents")

    # --- Embeddings ----------------------------------------------------
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # --- Chunking --------------------------------------------------------
    CHUNK_SIZE: int = _get_int("CHUNK_SIZE", 800)
    CHUNK_OVERLAP: int = _get_int("CHUNK_OVERLAP", 150)

    # --- Retrieval -------------------------------------------------------
    TOP_K: int = _get_int("TOP_K", 5)
    FETCH_K: int = _get_int("FETCH_K", 20)
    LAMBDA_MULT: float = _get_float("LAMBDA_MULT", 0.5)

    # --- LLM (Ollama) ------------------------------------------------------
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    OLLAMA_TIMEOUT: int = _get_int("OLLAMA_TIMEOUT", 120)
    LLM_TEMPERATURE: float = _get_float("LLM_TEMPERATURE", 0.1)

    # --- API ---------------------------------------------------------------
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = _get_int("API_PORT", 8000)

    # --- Misc ----------------------------------------------------------------
    MAX_UPLOAD_MB: int = _get_int("MAX_UPLOAD_MB", 50)
    ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

    @classmethod
    def ensure_dirs(cls) -> None:
        cls.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHROMA_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
