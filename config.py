"""
config.py — AgriLearn AI
Handles API key loading and model configuration securely.

Supports:
  1. Streamlit Secrets (st.secrets["GEMINI_API_KEY"])
  2. Environment variable / .env file (GEMINI_API_KEY)

Never hardcode real API keys here.
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
    Return the Gemini model name.

    Defaults to 'gemini-1.5-flash' if the GEMINI_MODEL environment variable
    is not set.  You can override this in your .env file:
        GEMINI_MODEL=gemini-1.5-flash-latest
    """
    return os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")


def get_temperature() -> float:
    """
    Return the model temperature (creativity level).
    Lower values make responses more focused and factual.
    """
    try:
        return float(os.environ.get("GEMINI_TEMPERATURE", "0.3"))
    except ValueError:
        return 0.3


# ── Convenience bundle ─────────────────────────────────────────────────────────
class AppConfig:
    """Single configuration object passed through the application."""

    def __init__(self):
        self.api_key: str | None = get_api_key()
        self.model_name: str = get_model_name()
        self.temperature: float = get_temperature()
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
