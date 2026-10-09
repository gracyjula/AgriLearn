"""
gemini_service.py — AgriLearn AI
LangChain + Gemini Flash integration for generating agricultural explanations.

Responsibilities:
  - Initialise ChatGoogleGenerativeAI with the configured model and API key.
  - Build the ChatPromptTemplate via prompts.py.
  - Invoke the chain with conversation history and optional RAG context.
  - Return a plain text response or a descriptive error message.
  - Handle API errors (NotFound, auth, quota, network) separately and clearly.
  - Never log or expose the API key.

FIX (2026-10): Separated error types so NotFound (wrong/shutdown model name),
authentication errors, and quota errors each show a distinct, actionable message.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, AIMessage

from config import AppConfig
from prompts import build_chat_prompt, build_rag_prompt

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


# ── History helpers ────────────────────────────────────────────────────────────

def build_history(chat_history: list[dict]) -> list["BaseMessage"]:
    """
    Convert a list of {'role': str, 'content': str} dicts (Streamlit session
    format) into LangChain message objects.

    Only the last N turns are kept to stay within context limits.
    """
    MAX_HISTORY_TURNS = 10  # keep last 10 exchanges (20 messages)
    messages: list[BaseMessage] = []

    # Limit history length
    recent = chat_history[-(MAX_HISTORY_TURNS * 2):]

    for msg in recent:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    return messages


def _classify_api_error(exc: Exception, model_name: str) -> str:
    """
    Return a clear, user-friendly error message for a Gemini API exception.

    Covers:
      - 404 NotFound     → wrong or shut-down model name
      - 401/403 auth     → invalid or missing API key
      - 429 quota        → rate limit or billing quota exceeded
      - network issues   → connectivity
      - everything else  → generic fallback

    Never exposes the API key value.
    """
    err_str = str(exc).lower()
    err_type = type(exc).__name__

    # 404 Not Found — model name is wrong or the model has been shut down
    if "404" in err_str or "not found" in err_str or "notfound" in err_type.lower():
        return (
            f"⚠️ **Model not found: `{model_name}`**\n\n"
            "The configured Gemini model does not exist or has been shut down. "
            "This is the most likely cause if you recently set up the project.\n\n"
            "**Fix:** Open your `.env` file and set:\n"
            "```\n"
            "GEMINI_MODEL=gemini-3.5-flash\n"
            "```\n"
            "Current stable flash models: `gemini-3.5-flash`, `gemini-3.6-flash`, `gemini-3.8-flash`.\n\n"
            "Note: `gemini-1.5-flash` has been shut down as of 2026."
        )

    # 401 / 403 / UNAUTHENTICATED — bad or missing API key
    if any(k in err_str for k in ["401", "403", "unauthenticated", "permission_denied", "invalid api key", "api_key"]):
        return (
            "⚠️ **Authentication error.**\n\n"
            "Your API key was rejected. Please check that:\n"
            "1. `GEMINI_API_KEY` in your `.env` file contains a valid key.\n"
            "2. The key has not expired or been revoked.\n"
            "3. Get a new key at: https://aistudio.google.com/app/apikey\n\n"
            "_The key value is never shown in this message for security._"
        )

    # 429 — rate limit or quota exceeded
    if any(k in err_str for k in ["429", "quota", "rate_limit", "resource_exhausted"]):
        return (
            "⚠️ **API quota or rate limit exceeded.**\n\n"
            "Your Gemini API key has hit its rate limit or monthly quota. "
            "Please wait a moment and try again, or check your quota at "
            "https://aistudio.google.com/app/apikey\n\n"
            "Free-tier keys have per-minute and per-day limits."
        )

    # Network / connection errors
    if any(k in err_str for k in ["connection", "timeout", "network", "ssl", "socket"]):
        return (
            "⚠️ **Network error.**\n\n"
            "Could not connect to the Gemini API. Please check your internet connection "
            "and try again."
        )

    # Generic fallback — include error type but not internal details
    return (
        f"⚠️ **An error occurred while generating a response.**\n\n"
        f"Error type: `{err_type}`\n\n"
        "Please check your API key and internet connection, then try again. "
        "If the problem persists, check the troubleshooting section in the README."
    )


# ── Core service ───────────────────────────────────────────────────────────────

class GeminiService:
    """
    Wraps LangChain + ChatGoogleGenerativeAI to generate agricultural explanations.

    Supports two modes:
      1. Direct chat mode    — uses conversation history + system prompt.
      2. RAG-grounded mode   — injects retrieved document context into the prompt.

    Usage:
        service = GeminiService(config)
        text, ok = service.generate(question="Explain crop growth stages.")
        text, ok = service.generate(question="...", rag_context="...", rag_sources=[...])
    """

    def __init__(self, config: AppConfig):
        self._config = config
        self._llm = None  # Lazy initialisation

    def _get_llm(self):
        """
        Lazily initialise the LangChain Gemini model.
        Returns None if initialisation fails (caller handles the error).
        """
        if self._llm is not None:
            return self._llm

        if not self._config.is_configured:
            return None

        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            self._llm = ChatGoogleGenerativeAI(
                model=self._config.model_name,
                google_api_key=self._config.api_key,
                temperature=self._config.temperature,
                # convert_system_message_to_human is deprecated in newer LangChain;
                # omit it so the system message is passed directly to the API.
            )
            return self._llm
        except ImportError:
            return None
        except Exception:
            return None

    def generate(
        self,
        question: str,
        chat_history: list[dict] | None = None,
        language: str = "English",
        rag_context: str | None = None,
        rag_sources: list[dict] | None = None,
    ) -> tuple[str, bool]:
        """
        Generate an agricultural explanation for the given question.

        Args:
            question:     The user's question text.
            chat_history: List of previous messages as {'role', 'content'} dicts.
            language:     Output language — 'English' or 'Telugu'.
            rag_context:  Optional retrieved document text for RAG mode.
            rag_sources:  Optional list of source metadata dicts for RAG mode.

        Returns:
            (response_text, success_flag)
            - On success: (AI response string, True)
            - On failure: (user-friendly error message, False)
        """
        if not question or not question.strip():
            return "Please enter a question.", False

        # Ensure API key is available
        if not self._config.is_configured:
            return (
                "⚠️ **API key not configured.**\n\n"
                "Please add your Gemini API key to the `.env` file or Streamlit Secrets "
                "before using the chatbot. See the README for setup instructions.",
                False,
            )

        llm = self._get_llm()
        if llm is None:
            return (
                "⚠️ **Could not initialise the AI model.**\n\n"
                "This may be caused by a missing `langchain-google-genai` package "
                "or an invalid API key format. Check your installation and API key, "
                "then restart the app.",
                False,
            )

        try:
            # Choose prompt template: RAG-grounded or plain chat
            if rag_context and rag_context.strip():
                prompt = build_rag_prompt(language=language)
                invoke_input = {
                    "history": build_history(chat_history or []),
                    "context": rag_context,
                    "question": question,
                }
            else:
                prompt = build_chat_prompt(language=language)
                invoke_input = {
                    "history": build_history(chat_history or []),
                    "question": question,
                }

            chain = prompt | llm
            response = chain.invoke(invoke_input)

            # Extract text content safely
            if hasattr(response, "content"):
                text = response.content
            else:
                text = str(response)

            if not text or not text.strip():
                return (
                    "The model returned an empty response. Please try rephrasing your question.",
                    False,
                )

            return text.strip(), True

        except Exception as exc:
            return _classify_api_error(exc, self._config.model_name), False
