"""
config.py — AgriLearn AI
Handles API key loading and model configuration securely.

Supports:
  1. Streamlit Secrets (st.secrets["GEMINI_API_KEY"])
  2. Environment variable / .env file (GEMINI_API_KEY)

Never hardcode real API keys here.

FIX (2026-10): gemini-1.5-flash is shut down. Default updated to gemini-3.5-flash,
which is a stable model as of the current Gemini API. See:
https://ai.google.dev/gemini-api/docs/models
"""

import os
from dotenv import load_dotenv

# Load variables from .env file if present (local development only)
load_dotenv()


def get_api_key() -> str | None:
    """
    Return the Gemini API key from Streamlit Secrets or environment variables.

    Returns None (does NOT raise) so that calling code can display a user-friendly
    error message instead of crashing with an unhandled exception.
    """
    # 1. Try Streamlit Secrets first (production / Streamlit Cloud deployment)
    try:
        import streamlit as st
        key = st.secrets.get("GEMINI_API_KEY", None)
        if key:
            return key
    except Exception:
        # Streamlit is not running or secrets are not configured — fall through
        pass

    # 2. Fall back to environment variable / .env file
    key = os.environ.get("GEMINI_API_KEY", None)
    return key if key else None


def get_model_name() -> str:
    """
    Return the Gemini chat model name.

    Default: gemini-3.5-flash (stable as of October 2026).

    Override by setting GEMINI_MODEL in your .env file:
        GEMINI_MODEL=gemini-3.6-flash

    NOTE: gemini-1.5-flash was shut down and will return a 404 NotFound error.
    Current stable flash models: gemini-3.5-flash, gemini-3.6-flash, gemini-3.8-flash.
    """
    return os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")


def get_embedding_model_name() -> str:
    """
    Return the Gemini embedding model name for RAG.

    Default: gemini-embedding-001 (stable, supported until May 2028).
    text-embedding-004 was shut down on January 14, 2026.

    Override by setting GEMINI_EMBEDDING_MODEL in your .env file.
    """
    return os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")


def get_temperature() -> float:
    """
    Return the model temperature (creativity level).
    Lower values make responses more focused and factual.
    """
    try:
        return float(os.environ.get("GEMINI_TEMPERATURE", "0.3"))
    except ValueError:
        return 0.3


def get_rag_config() -> dict:
    """
    Return RAG (Retrieval-Augmented Generation) configuration.

    Returns a dict with:
      - pdf_dir:          folder where agricultural PDF documents are stored
      - vector_store_dir: folder where the FAISS index is persisted
      - chunk_size:       characters per text chunk
      - chunk_overlap:    characters of overlap between consecutive chunks
      - top_k:            number of chunks to retrieve per query
      - enabled:          whether RAG is active (requires PDFs to be indexed)
    """
    return {
        "pdf_dir": os.environ.get("AGRILEARN_PDF_DIR", "docs/pdfs"),
        "vector_store_dir": os.environ.get("AGRILEARN_VECTOR_STORE_DIR", "docs/vector_store"),
        "chunk_size": int(os.environ.get("AGRILEARN_CHUNK_SIZE", "800")),
        "chunk_overlap": int(os.environ.get("AGRILEARN_CHUNK_OVERLAP", "100")),
        "top_k": int(os.environ.get("AGRILEARN_TOP_K", "4")),
    }


# ── Convenience bundle ─────────────────────────────────────────────────────────
class AppConfig:
    """Single configuration object passed through the application."""

    def __init__(self):
        self.api_key: str | None = get_api_key()
        self.model_name: str = get_model_name()
        self.embedding_model_name: str = get_embedding_model_name()
        self.temperature: float = get_temperature()
        self.rag: dict = get_rag_config()
        self.app_title: str = "AgriLearn AI"
        self.app_subtitle: str = "Your Smart Agriculture Learning Assistant"
        self.app_description: str = (
            "Ask me anything about crop lifecycles, sowing, irrigation, "
            "harvesting, and general farming processes. I provide clear, "
            "beginner-friendly educational explanations."
        )

    @property
    def is_configured(self) -> bool:
        """True when a non-empty API key is available."""
        return bool(self.api_key)
