"""
gemini_service.py — AgriLearn AI
LangChain + Gemini Flash integration for generating agricultural explanations.

Responsibilities:
  - Initialise ChatGoogleGenerativeAI with the configured model and API key.
  - Build the ChatPromptTemplate via prompts.py.
  - Invoke the chain and return a plain text response.
  - Handle API errors gracefully without crashing the Streamlit app.
  - Never log or expose the API key.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, AIMessage

from config import AppConfig
from prompts import build_chat_prompt

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


# ── Core service ───────────────────────────────────────────────────────────────

class GeminiService:
    """
    Wraps LangChain + ChatGoogleGenerativeAI to generate agricultural explanations.

    Usage:
        service = GeminiService(config)
        response = service.generate(question="Explain crop growth stages.")
    """

    def __init__(self, config: AppConfig):
        self._config = config
        self._llm = None  # Lazy initialisation

    def _get_llm(self):
        """
        Lazily initialise the LangChain Gemini model.
        Returns None and logs a descriptive error if initialisation fails.
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
                convert_system_message_to_human=False,
            )
            return self._llm
        except ImportError:
            # Package not installed
            return None
        except Exception:
            # Any other initialisation error (invalid key format, etc.)
            return None

    def generate(
        self,
        question: str,
        chat_history: list[dict] | None = None,
        language: str = "English",
    ) -> tuple[str, bool]:
        """
        Generate an agricultural explanation for the given question.

        Args:
            question:     The user's question text.
            chat_history: List of previous messages as {'role', 'content'} dicts.
            language:     Output language — 'English' or 'Telugu'.

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
                "or an invalid API key. Check your installation and API key, then restart the app.",
                False,
            )

        try:
            # Build prompt template for the selected language
            prompt = build_chat_prompt(language=language)

            # Build conversation history
            history = build_history(chat_history or [])

            # Construct and invoke the chain
            chain = prompt | llm

            response = chain.invoke(
                {
                    "history": history,
                    "question": question,
                }
            )

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
            error_type = type(exc).__name__
            # Generic error — do not expose API key or internal details
            return (
                f"⚠️ **An error occurred while generating a response.**\n\n"
                f"Error type: `{error_type}`\n\n"
                "Please check your API key and internet connection, then try again. "
                "If the problem persists, see the troubleshooting section in the README.",
                False,
            )
