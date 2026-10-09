"""
app.py — AgriLearn AI
Main Streamlit application — agriculture education chatbot interface.

Run with:
    streamlit run app.py
"""

import streamlit as st

from config import AppConfig
from gemini_service import GeminiService
from safety import check_input

# ── Page configuration (must be the very first Streamlit call) ─────────────────

st.set_page_config(
    page_title="AgriLearn AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "AgriLearn AI — Smart Agriculture Learning Assistant powered by Gemini Flash.",
    },
)

# ── Custom CSS — agriculture-inspired green/white theme ───────────────────────

st.markdown(
    """
    <style>
    /* ── Global background ── */
    .stApp {
        background-color: #f4f9f1;
    }

    /* ── Header banner ── */
    .agrilearn-header {
        background: linear-gradient(135deg, #2d6a4f 0%, #40916c 50%, #52b788 100%);
        color: white;
        padding: 2rem 2.5rem 1.5rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(45, 106, 79, 0.3);
    }
    .agrilearn-header h1 {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .agrilearn-header .subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    .agrilearn-header .description {
        font-size: 0.9rem;
        opacity: 0.8;
        line-height: 1.5;
    }

    /* ── Warning banner ── */
    .api-warning {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-left: 5px solid #ff9800;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
    }

    /* ── Example question buttons ── */
    .stButton > button {
        background-color: #d8f3dc;
        border: 1px solid #95d5b2;
        color: #1b4332;
        border-radius: 20px;
        padding: 0.4rem 1rem;
        font-size: 0.85rem;
        transition: all 0.2s;
        white-space: normal;
        height: auto;
    }
    .stButton > button:hover {
        background-color: #52b788;
        color: white;
        border-color: #52b788;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #1b4332;
        color: #d8f3dc;
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #d8f3dc;
    }
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #95d5b2;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] {
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }

    /* ── Footer ── */
    .agrilearn-footer {
        text-align: center;
        color: #6c757d;
        font-size: 0.78rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #dee2e6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state initialisation ───────────────────────────────────────────────

def init_session_state():
    """Initialise all required session state keys on first run."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "language" not in st.session_state:
        st.session_state.language = "English"
    if "config" not in st.session_state:
        st.session_state.config = AppConfig()
    if "service" not in st.session_state:
        st.session_state.service = GeminiService(st.session_state.config)
    if "example_question" not in st.session_state:
        st.session_state.example_question = None


init_session_state()
config: AppConfig = st.session_state.config
service: GeminiService = st.session_state.service

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🌾 AgriLearn AI")
    st.markdown("---")

    # Language selection
    st.markdown("### 🌐 Language / భాష")
    lang_choice = st.selectbox(
        "Select response language",
        options=["English", "Telugu"],
        index=0 if st.session_state.language == "English" else 1,
        label_visibility="collapsed",
    )
    if lang_choice != st.session_state.language:
        st.session_state.language = lang_choice

    st.markdown("---")

    # About
    st.markdown("### ℹ️ About")
    st.markdown(
        "AgriLearn AI is a **beginner-friendly agricultural education chatbot** "
        "that explains crop lifecycles and farming processes using Google Gemini Flash."
    )

    st.markdown("---")

    # How to use
    st.markdown("### 💬 How to Use")
    st.markdown(
        "1. Type your agricultural question below.\n"
        "2. Press **Enter** or click the send button.\n"
        "3. Read the clear, structured explanation.\n"
        "4. Ask follow-up questions naturally.\n"
        "5. Switch to **Telugu** for responses in Telugu."
    )

    st.markdown("---")

    # Supported topics
    st.markdown("### 🌱 Supported Topics")
    st.markdown(
        "- Sowing & seed germination\n"
        "- Crop lifecycle stages\n"
        "- Irrigation methods & cycles\n"
        "- Harvesting stages & processes\n"
        "- Grain drying & storage\n"
        "- General agricultural terms"
    )

    st.markdown("---")

    # Safety limitations
    st.markdown("### ⚠️ Limitations")
    st.markdown(
        "This bot **cannot** provide:\n"
        "- Fertilizer recommendations\n"
        "- Pesticide / chemical advice\n"
        "- Crop disease diagnosis\n"
        "- Yield predictions\n"
        "- Personalized farm prescriptions\n\n"
        "_For specific advice, contact your local agricultural extension officer._"
    )

    st.markdown("---")

    # Clear conversation button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.example_question = None
        st.rerun()

    st.markdown("---")

    # API key status indicator
    if config.is_configured:
        st.success("✅ API key configured")
    else:
        st.error("❌ API key not found")
        st.markdown(
            "Add `GEMINI_API_KEY=your_key` to a `.env` file in the project folder. "
            "See the README for instructions."
        )

# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <div class="agrilearn-header">
        <h1>🌾 {config.app_title}</h1>
        <div class="subtitle">🌱 {config.app_subtitle}</div>
        <div class="description">{config.app_description}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# API key warning banner
if not config.is_configured:
    st.markdown(
        """
        <div class="api-warning">
            ⚠️ <strong>API key not configured.</strong>
            Add your <code>GEMINI_API_KEY</code> to a <code>.env</code> file
            in the <code>agrilearn-ai/</code> folder, then restart the app.
            See the README for step-by-step instructions.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Example question buttons (shown only when chat is empty) ───────────────────

EXAMPLE_QUESTIONS = [
    "🌱 Explain crop growth stages.",
    "💧 What is an irrigation cycle?",
    "🌾 Explain the harvesting process.",
    "🏪 What are basic storage practices?",
]

if not st.session_state.messages:
    st.markdown("#### 👇 Try an example question:")
    cols = st.columns(len(EXAMPLE_QUESTIONS))
    for i, (col, question) in enumerate(zip(cols, EXAMPLE_QUESTIONS)):
        with col:
            if st.button(question, key=f"example_{i}", use_container_width=True):
                # Strip the emoji prefix before sending
                clean_q = question.split(" ", 1)[1] if " " in question else question
                st.session_state.example_question = clean_q

# ── Display existing conversation history ──────────────────────────────────────

for message in st.session_state.messages:
    avatar = "🧑‍🌾" if message["role"] == "user" else "🌾"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# ── Handle example question trigger ───────────────────────────────────────────

def process_user_input(user_text: str):
    """Validate, send to Gemini, and append both messages to session state."""
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(user_text)

    # Safety check
    safety = check_input(user_text)

    with st.chat_message("assistant", avatar="🌾"):
        if not safety.is_safe:
            # Show pre-canned safety refusal without calling the API
            st.markdown(safety.reason)
            st.session_state.messages.append(
                {"role": "assistant", "content": safety.reason}
            )
        else:
            # Call Gemini via LangChain
            with st.spinner("🌱 Thinking…"):
                # Pass history EXCLUDING the latest user message (already appended)
                history_for_llm = st.session_state.messages[:-1]
                response_text, success = service.generate(
                    question=user_text,
                    chat_history=history_for_llm,
                    language=st.session_state.language,
                )
            st.markdown(response_text)
            st.session_state.messages.append(
                {"role": "assistant", "content": response_text}
            )


if st.session_state.example_question:
    eq = st.session_state.example_question
    st.session_state.example_question = None
    process_user_input(eq)

# ── Chat input ─────────────────────────────────────────────────────────────────

placeholder_text = (
    "Ask about crop lifecycles, irrigation, harvesting, storage…"
    if st.session_state.language == "English"
    else "పంటల గురించి మీ ప్రశ్న అడగండి…"
)

if prompt := st.chat_input(placeholder_text):
    process_user_input(prompt)

# ── Footer ─────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="agrilearn-footer">
        AgriLearn AI &nbsp;•&nbsp; Powered by Google Gemini Flash &amp; LangChain &nbsp;•&nbsp;
        For educational purposes only &nbsp;•&nbsp;
        Not a substitute for professional agricultural advice.
    </div>
    """,
    unsafe_allow_html=True,
)
