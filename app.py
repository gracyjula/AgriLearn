"""
app.py — AgriLearn AI
Professional agriculture chatbot — Streamlit 1.35 compatible.

AVATAR STRATEGY (critical for alignment CSS):
  Streamlit 1.35 only attaches data-testid="chatAvatarIcon-user" and
  data-testid="chatAvatarIcon-assistant" when the avatar TYPE is ICON —
  i.e. when avatar="user" or avatar="assistant" (the built-in preset strings)
  are passed.  When a custom emoji string such as "🧑‍🌾" is passed, the avatar
  type is EMOJI and NO data-testid is added to the avatar element.

  Therefore this file passes avatar="user" and avatar="assistant" to
  st.chat_message() so that :has([data-testid="chatAvatarIcon-user"]) and
  :has([data-testid="chatAvatarIcon-assistant"]) reliably fire.

Run with:
    streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components

from config import AppConfig
from gemini_service import GeminiService
from rag_service import RAGService, check_rag_dependencies
from safety import check_input
from voice_component import (
    render_voice_input_component,
    render_tts_button,
    render_browser_support_banner,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgriLearn AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "AgriLearn AI — Smart Agriculture Learning Assistant."},
)

# ── CSS ────────────────────────────────────────────────────────────────────────
# Palette:
#   #123D2B  deep forest green  — sidebar, header
#   #238653  primary agri green — buttons, accents
#   #B9E769  leaf accent        — highlights, tagline
#   #F7F9F5  warm white         — page background
#   #FFFFFF  white              — message surfaces
#   #20352B  dark text
#   #637368  secondary text
#   #E2E9E1  border
#
# AVATAR NOTE: :has([data-testid="chatAvatarIcon-user"]) only fires when
# st.chat_message is called with avatar="user" (the ICON preset), NOT with
# a custom emoji string.  We rely on this by always using the preset strings.

st.markdown("""
<style>

/* ─── 1. Page & typography ─────────────────────────────────────────────────── */
html, body, .stApp {
    background-color: #F7F9F5 !important;
    color: #20352B !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Noto Sans Telugu", "Noto Serif Telugu", sans-serif !important;
}

/* Constrain main content width so long responses stay readable */
.block-container {
    max-width: 900px !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    padding-top: 0.75rem !important;
    padding-bottom: 1rem !important;
}

/* ─── 2. Markdown / text colours ──────────────────────────────────────────── */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] em,
[data-testid="stMarkdownContainer"] a,
.stMarkdown p, .stMarkdown li {
    color: #20352B !important;
}
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
    color: #123D2B !important;
}
[data-testid="stMarkdownContainer"] code {
    background: #EEF5EC !important;
    color: #123D2B !important;
    border-radius: 4px;
    padding: 1px 5px;
}

/* ─── 3. Compact page header ──────────────────────────────────────────────── */
.al-header {
    background: #123D2B;
    padding: 0.65rem 1.25rem;
    border-radius: 8px;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 0.6rem;
}
.al-header-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #F7F9F5 !important;
    margin: 0;
}
.al-header-tag {
    font-size: 0.74rem;
    color: #B9E769 !important;
    margin: 0;
}

/* ─── 4. System banners ───────────────────────────────────────────────────── */
.al-warn {
    background: #FFF8E1;
    border-left: 4px solid #F59E0B;
    border-radius: 6px;
    padding: 0.5rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.83rem;
    color: #78350F !important;
}
.al-warn strong, .al-warn code { color: #78350F !important; }

.al-rag {
    background: #ECFDF5;
    border-left: 4px solid #238653;
    border-radius: 6px;
    padding: 0.45rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.82rem;
    color: #123D2B !important;
}
.al-rag strong { color: #123D2B !important; }

/* ─── 5. Welcome state ────────────────────────────────────────────────────── */
.al-welcome {
    text-align: center;
    padding: 1.2rem 0 0.6rem;
}
.al-welcome h2 {
    font-size: 1.2rem;
    font-weight: 700;
    color: #123D2B !important;
    margin: 0 0 0.25rem;
}
.al-welcome p {
    font-size: 0.86rem;
    color: #637368 !important;
    margin: 0 0 0.6rem;
}

/* ─── 6. Suggestion chip buttons ─────────────────────────────────────────── */
.stButton > button {
    background: #F0F9F4 !important;
    border: 1.5px solid #238653 !important;
    color: #123D2B !important;
    border-radius: 20px !important;
    padding: 0.3rem 0.85rem !important;
    font-size: 0.81rem !important;
    font-weight: 600 !important;
    white-space: normal !important;
    height: auto !important;
    line-height: 1.4 !important;
    box-shadow: none !important;
    transition: background 0.12s ease, color 0.12s ease !important;
}
.stButton > button:hover, .stButton > button:focus {
    background: #238653 !important;
    color: #FFFFFF !important;
    outline: none !important;
}

/* ─── 7. Chat messages — user RIGHT, assistant LEFT ──────────────────────── */
/*
 * Streamlit 1.35 source (verified):
 *   - The outer wrapper gets class="stChatMessage" data-testid="stChatMessage"
 *   - When avatar="user" (ICON preset), the avatar div gets
 *     data-testid="chatAvatarIcon-user"
 *   - When avatar="assistant" (ICON preset), the avatar div gets
 *     data-testid="chatAvatarIcon-assistant"
 *   - Custom emoji avatars get NO data-testid — :has() won't work for them.
 *
 * We pass avatar="user" / avatar="assistant" (not emoji) so these testids fire.
 */

/* All chat message containers: clear default Streamlit background */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.2rem 0 !important;
    max-width: 100% !important;
}

/* USER: flip row direction so avatar is on the right */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse !important;
}

/* USER content bubble: green, right-aligned */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
    [data-testid="stChatMessageContent"] {
    background: #238653 !important;
    color: #FFFFFF !important;
    border-radius: 16px 2px 16px 16px !important;
    padding: 0.6rem 0.9rem !important;
    max-width: 74% !important;
    margin-left: auto !important;
    margin-right: 0 !important;
    box-shadow: 0 1px 2px rgba(18,61,43,0.14) !important;
    border: none !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
    [data-testid="stChatMessageContent"] p,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
    [data-testid="stChatMessageContent"] li,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
    [data-testid="stChatMessageContent"] span,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
    [data-testid="stChatMessageContent"] strong {
    color: #FFFFFF !important;
}

/* ASSISTANT content bubble: white, left-aligned */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
    [data-testid="stChatMessageContent"] {
    background: #FFFFFF !important;
    color: #20352B !important;
    border: 1px solid #E2E9E1 !important;
    border-radius: 2px 16px 16px 16px !important;
    padding: 0.6rem 0.9rem !important;
    max-width: 74% !important;
    margin-left: 0 !important;
    box-shadow: 0 1px 2px rgba(18,61,43,0.07) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
    [data-testid="stChatMessageContent"] p,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
    [data-testid="stChatMessageContent"] li,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
    [data-testid="stChatMessageContent"] span {
    color: #20352B !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
    [data-testid="stChatMessageContent"] strong,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
    [data-testid="stChatMessageContent"] h1,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
    [data-testid="stChatMessageContent"] h2,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
    [data-testid="stChatMessageContent"] h3 {
    color: #123D2B !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"])
    [data-testid="stChatMessageContent"] code {
    background: #EEF5EC !important;
    color: #123D2B !important;
}

/* ─── 8. Source citations ─────────────────────────────────────────────────── */
.al-sources {
    background: #EEF5EC;
    border: 1px solid #B9E769;
    border-radius: 6px;
    padding: 0.35rem 0.8rem;
    margin: 0.15rem 0 0.3rem 2.8rem;
    font-size: 0.76rem;
    color: #1D5238 !important;
    max-width: 72%;
}
.al-sources strong { color: #123D2B !important; }
.al-sources li { color: #1D5238 !important; list-style: none; padding: 0; margin: 0; }

/* ─── 9. TTS listen row ───────────────────────────────────────────────────── */
.al-tts-row {
    padding-left: 2.8rem;
    margin-top: 0;
    margin-bottom: 0.15rem;
}

/* ─── 10. Chat input bar ──────────────────────────────────────────────────── */
[data-testid="stChatInput"] textarea {
    background: #FFFFFF !important;
    color: #20352B !important;
    border: 1.5px solid #238653 !important;
    border-radius: 10px !important;
    font-size: 0.91rem !important;
    font-family: inherit !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #8FAF97 !important;
}
[data-testid="stBottom"] {
    background: #F7F9F5 !important;
    border-top: 1px solid #E2E9E1 !important;
    padding-top: 0.35rem !important;
    padding-bottom: 0.35rem !important;
}

/* ─── 11. Sidebar ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #123D2B !important;
    border-right: 1px solid #1A4D35 !important;
}
/* All sidebar text: light green */
[data-testid="stSidebar"] * {
    color: #C8E6C9 !important;
}
/* Headings: leaf accent */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: #B9E769 !important;
}
/* Paragraphs and list items */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li {
    color: #C8E6C9 !important;
    font-size: 0.83rem !important;
}
/* Captions */
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stCaption {
    color: #7DAE8A !important;
}
/* Selectbox labels */
[data-testid="stSidebar"] label {
    color: #B9E769 !important;
    font-size: 0.8rem !important;
}
/* Selectbox dropdown */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #1A4D35 !important;
    color: #C8E6C9 !important;
    border-color: #238653 !important;
    border-radius: 6px !important;
}
/* Sidebar buttons */
[data-testid="stSidebar"] .stButton > button {
    background: #1A4D35 !important;
    color: #C8E6C9 !important;
    border-color: #238653 !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #238653 !important;
    color: #FFFFFF !important;
}
/* Sidebar divider */
[data-testid="stSidebar"] hr {
    border-color: #1A4D35 !important;
    margin: 0.4rem 0 !important;
}
/* Sidebar expander header */
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    color: #C8E6C9 !important;
    font-size: 0.82rem !important;
}

/* ─── 12. Voice expander ──────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background: #EEF5EC !important;
    color: #123D2B !important;
    border: 1px solid #E2E9E1 !important;
    border-radius: 8px !important;
    font-size: 0.84rem !important;
}
.streamlit-expanderContent {
    background: #F7F9F5 !important;
    border: 1px solid #E2E9E1 !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}

/* ─── 13. Spinner ─────────────────────────────────────────────────────────── */
.stSpinner > div { color: #238653 !important; }

/* ─── 14. Footer ──────────────────────────────────────────────────────────── */
.al-footer {
    text-align: center;
    color: #637368 !important;
    font-size: 0.71rem;
    margin-top: 0.8rem;
    padding-top: 0.5rem;
    border-top: 1px solid #E2E9E1;
}

/* ─── 15. Misc chrome ─────────────────────────────────────────────────────── */
hr { border-color: #E2E9E1 !important; }
h1, h2, h3, h4, h5 { color: #123D2B !important; }

</style>
""", unsafe_allow_html=True)


# ── Helpers (defined before any use) ─────────────────────────────────────────

def _render_sources(sources: list[dict]) -> None:
    """Compact source-citation block beneath an assistant message."""
    if not sources:
        return
    unique: dict[str, dict] = {}
    for s in sources:
        k = f"{s['source_file']}:p{s['page_number']}"
        if k not in unique:
            unique[k] = s
    items = "".join(
        f'<li>📄 <strong>{s["source_file"]}</strong> — p.{s["page_number"]}</li>'
        for s in unique.values()
    )
    st.markdown(
        f'<div class="al-sources">📚 <strong>Sources:</strong>'
        f'<ul style="margin:0.15rem 0 0 0;padding:0">{items}</ul></div>',
        unsafe_allow_html=True,
    )


def _render_tts(message_text: str, button_id: str) -> None:
    """Small TTS listen button beneath an assistant bubble."""
    tts_html = render_tts_button(
        message_text=message_text,
        language=st.session_state.language,
        button_id=button_id,
    )
    st.markdown('<div class="al-tts-row">', unsafe_allow_html=True)
    components.html(tts_html, height=38, scrolling=False)
    st.markdown('</div>', unsafe_allow_html=True)


def _show_message(msg: dict, idx: int) -> None:
    """Replay one stored message with all decorations."""
    if msg["role"] == "user":
        with st.chat_message("user"):      # ICON preset → chatAvatarIcon-user
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"): # ICON preset → chatAvatarIcon-assistant
            st.markdown(msg["content"])
        if msg.get("sources"):
            _render_sources(msg["sources"])
        _render_tts(msg["content"], f"tts_{idx}")


# ── Session state ─────────────────────────────────────────────────────────────

def _init_session() -> None:
    import os
    for k, v in {
        "messages": [],
        "language": "English",
        "example_question": None,
        "voice_input_pending": "",
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if "config" not in st.session_state:
        st.session_state.config = AppConfig()
    if "service" not in st.session_state:
        st.session_state.service = GeminiService(st.session_state.config)
    if "rag_service" not in st.session_state:
        st.session_state.rag_service = RAGService(st.session_state.config)
    if "rag_enabled" not in st.session_state:
        deps_ok, _ = check_rag_dependencies()
        st.session_state.rag_enabled = (
            deps_ok and st.session_state.rag_service.is_index_ready()
        )
    cfg_rag = st.session_state.config.rag
    os.makedirs(cfg_rag["pdf_dir"], exist_ok=True)
    os.makedirs(cfg_rag["vector_store_dir"], exist_ok=True)


_init_session()

cfg: AppConfig     = st.session_state.config
svc: GeminiService = st.session_state.service
rag: RAGService    = st.session_state.rag_service


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    # Branding
    st.markdown(
        '<div style="padding:0.5rem 0 0.3rem;text-align:center;">'
        '<div style="font-size:1.8rem;line-height:1.1;">🌾</div>'
        '<div style="font-size:0.97rem;font-weight:700;color:#B9E769;margin-top:2px;">AgriLearn AI</div>'
        '<div style="font-size:0.68rem;color:#7DAE8A;">Your guide to smarter farming.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # Language
    st.markdown(
        '<div style="font-size:0.78rem;font-weight:600;color:#B9E769;margin-bottom:3px;">🌐 Language / భాష</div>',
        unsafe_allow_html=True,
    )
    lang_opts = ["English", "Telugu"]
    new_lang = st.selectbox(
        "language",
        lang_opts,
        index=lang_opts.index(st.session_state.language),
        label_visibility="collapsed",
    )
    if new_lang != st.session_state.language:
        st.session_state.language = new_lang
    st.divider()

    # Knowledge Base
    st.markdown(
        '<div style="font-size:0.78rem;font-weight:600;color:#B9E769;margin-bottom:3px;">📚 Knowledge Base</div>',
        unsafe_allow_html=True,
    )
    deps_ok, _ = check_rag_dependencies()
    if not deps_ok:
        st.caption("RAG unavailable. Install: `pip install pypdf faiss-cpu langchain-community`")
        st.session_state.rag_enabled = False
    else:
        pdf_count   = rag.get_indexed_pdf_count()
        index_ready = rag.is_index_ready()
        if index_ready:
            st.success(f"✅ Index ready · {pdf_count} PDF(s)")
            new_rag = st.toggle("Use document knowledge", value=st.session_state.rag_enabled, key="rag_toggle")
            if new_rag != st.session_state.rag_enabled:
                st.session_state.rag_enabled = new_rag
        elif pdf_count > 0:
            st.info(f"📄 {pdf_count} PDF(s) — index not built yet.")
            st.session_state.rag_enabled = False
        else:
            st.caption("Add agricultural PDFs to `docs/pdfs/` to enable document search.")
            st.session_state.rag_enabled = False
        if pdf_count > 0:
            if st.button("🔄 Build / Rebuild Index", use_container_width=True, key="rebuild_btn"):
                with st.spinner("Building index…"):
                    ok, msg = rag.build_index(force_rebuild=True)
                if ok:
                    st.session_state.rag_enabled = True
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
    st.divider()

    with st.expander("🌱 Topics I can explain"):
        st.markdown(
            "- Crop lifecycle & growth stages\n"
            "- Sowing & seed germination\n"
            "- Irrigation methods & cycles\n"
            "- Harvesting stages & processes\n"
            "- Grain drying & storage\n"
            "- General agricultural terms"
        )
    with st.expander("⚠️ What I cannot provide"):
        st.markdown(
            "- Fertilizer recommendations\n"
            "- Pesticide / chemical advice\n"
            "- Disease diagnosis\n"
            "- Yield predictions\n\n"
            "_Contact your local agricultural extension officer for specific advice._"
        )
    with st.expander("ℹ️ About"):
        st.markdown(
            "AgriLearn AI explains crop lifecycles and farming processes "
            "using Google Gemini Flash. Ask in English or Telugu."
        )
    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True, key="clear_btn"):
        st.session_state.messages = []
        st.session_state.example_question = None
        st.session_state.voice_input_pending = ""
        st.rerun()
    st.divider()

    if cfg.is_configured:
        st.success("✅ API key configured")
        st.caption(f"Model: `{cfg.model_name}`")
    else:
        st.error("❌ API key missing")
        st.caption("Add `GEMINI_API_KEY=your_key` to `.env` and restart.")


# ── Compact header ────────────────────────────────────────────────────────────

st.markdown(
    f'<div class="al-header">'
    f'<span style="font-size:1.6rem;line-height:1;">🌾</span>'
    f'<div>'
    f'<div class="al-header-title">{cfg.app_title}</div>'
    f'<div class="al-header-tag">Learn farming, one step at a time.</div>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# System banners
if not cfg.is_configured:
    st.markdown(
        '<div class="al-warn">⚠️ <strong>API key not configured.</strong> '
        'Add <code>GEMINI_API_KEY=your_key</code> to <code>.env</code> and restart.</div>',
        unsafe_allow_html=True,
    )

if st.session_state.rag_enabled and rag.is_index_ready():
    st.markdown(
        '<div class="al-rag">📚 <strong>Document mode active.</strong> '
        'Answers grounded in indexed agricultural documents.</div>',
        unsafe_allow_html=True,
    )


# ── Welcome state (only when no messages) ────────────────────────────────────

if not st.session_state.messages:
    st.markdown(
        '<div class="al-welcome">'
        '<div style="font-size:2.2rem;margin-bottom:0.2rem;">🌱</div>'
        '<h2>Learn farming, one step at a time.</h2>'
        '<p>Ask about crop lifecycles, irrigation, harvesting, or grain storage. '
        'Available in English and Telugu.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    SUGGESTIONS = [
        "Explain crop growth stages",
        "Learn irrigation basics",
        "Understand harvesting",
        "Explore crop storage",
    ]
    c1, c2, c3, c4 = st.columns(4)
    for col, sug in zip([c1, c2, c3, c4], SUGGESTIONS):
        with col:
            if st.button(sug, key=f"sug_{sug[:10]}", use_container_width=True):
                st.session_state.example_question = sug
    st.divider()


# ── Chat history display ──────────────────────────────────────────────────────

for idx, msg in enumerate(st.session_state.messages):
    _show_message(msg, idx)


# ── Message processor ─────────────────────────────────────────────────────────

def _process(user_text: str) -> None:
    """Safety → RAG → Gemini → persist."""
    user_text = user_text.strip()
    if not user_text:
        return

    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):           # ICON preset
        st.markdown(user_text)

    safety = check_input(user_text)
    if not safety.is_safe:
        with st.chat_message("assistant"):  # ICON preset
            st.markdown(safety.reason)
        st.session_state.messages.append(
            {"role": "assistant", "content": safety.reason, "sources": []}
        )
        return

    rag_context = ""
    sources: list[dict] = []
    if st.session_state.rag_enabled and rag.is_index_ready():
        retrieval = rag.retrieve(user_text)
        if retrieval.context_text:
            rag_context = retrieval.context_text
            sources = [
                {"source_file": c.source_file, "page_number": c.page_number}
                for c in retrieval.sources
            ]

    with st.spinner("🌱 Thinking…"):
        response_text, success = svc.generate(
            question=user_text,
            chat_history=st.session_state.messages[:-1],
            language=st.session_state.language,
            rag_context=rag_context if rag_context else None,
        )

    with st.chat_message("assistant"):      # ICON preset
        st.markdown(response_text)

    if sources and success:
        _render_sources(sources)

    _render_tts(response_text, f"tts_live_{len(st.session_state.messages)}")

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text, "sources": sources}
    )


# ── Suggestion chip handler ───────────────────────────────────────────────────

if st.session_state.example_question:
    eq = st.session_state.example_question
    st.session_state.example_question = None
    _process(eq)
    st.rerun()


# ── Voice input ───────────────────────────────────────────────────────────────

with st.expander("🎤 Voice Input", expanded=False):
    components.html(render_browser_support_banner(), height=26, scrolling=False)
    components.html(
        render_voice_input_component(language=st.session_state.language),
        height=110,
        scrolling=False,
    )
    st.caption(
        "Click the mic · your transcript appears in the box above · "
        "copy it into the chat input below and press Enter. "
        "Voice input requires Chrome or Edge."
    )


# ── Chat input ────────────────────────────────────────────────────────────────

placeholder = (
    "Ask about crop lifecycles, irrigation, harvesting, storage…"
    if st.session_state.language == "English"
    else "పంటల గురించి మీ ప్రశ్న అడగండి…"
)

if prompt := st.chat_input(placeholder):
    _process(prompt)
    st.rerun()


# ── Footer ────────────────────────────────────────────────────────────────────

mode = (
    "RAG + Gemini Flash"
    if (st.session_state.rag_enabled and rag.is_index_ready())
    else "Gemini Flash"
)
st.markdown(
    f'<div class="al-footer">'
    f'AgriLearn AI &nbsp;·&nbsp; {mode} + LangChain &nbsp;·&nbsp; '
    'Educational use only &nbsp;·&nbsp; Not a substitute for professional advice.'
    '</div>',
    unsafe_allow_html=True,
)
