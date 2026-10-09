"""
voice_component.py — AgriLearn AI
Provides browser-based voice input (Speech-to-Text) and voice output
(Text-to-Speech) using the Web Speech API via injected HTML/JavaScript.

These helpers return HTML strings that Streamlit renders via
st.components.v1.html().  Because they are pure browser-side, no Python
audio library is required.

Supported languages:
  - English  → en-US  (both recognition & synthesis)
  - Telugu   → te-IN  (synthesis; recognition support varies by browser)

Voice input flow (simplified — browser limitation aware):
  1. User clicks the 🎤 mic button inside the HTML component.
  2. Web Speech API starts, transcribes speech.
  3. Transcript is displayed inside the component for the user to see.
  4. User copies the transcript or reads it and types it into the Streamlit
     text_input below.
  NOTE: Cross-origin postMessage from a sandboxed iframe to Streamlit's parent
  Python process is NOT reliably interceptable.  The simplest, always-working
  pattern is to show the transcript in the component and let the user submit
  it via a normal Streamlit input.

Voice output flow:
  1. An HTML snippet with a 🔊 button is embedded alongside each assistant
     message using st.components.v1.html().
  2. Clicking the button calls window.speechSynthesis.speak() with the message
     text encoded in the HTML.
  3. No server round-trip required.
"""

from __future__ import annotations

import html as html_module


# ── Language code mapping ──────────────────────────────────────────────────────

LANGUAGE_CODES: dict[str, str] = {
    "English": "en-US",
    "Telugu": "te-IN",
}


def _lang_code(language: str) -> str:
    """Return the BCP-47 language tag for the selected UI language."""
    return LANGUAGE_CODES.get(language, "en-US")


# ── Voice Input Component ──────────────────────────────────────────────────────

def render_voice_input_component(language: str = "English") -> str:
    """
    Return an HTML/JS string that renders a mic button for speech-to-text input.

    The user clicks the 🎤 button; the browser asks for microphone permission and
    starts listening.  When speech ends, the transcript appears below the button
    in a read-only text field.  The user can then type (or copy-paste) the
    captured text into the Streamlit chat input below.

    Args:
        language: "English" or "Telugu"

    Returns:
        HTML string suitable for st.components.v1.html(html, height=120)
    """
    lang_code = _lang_code(language)
    placeholder = (
        "Transcript will appear here…"
        if language == "English"
        else "మీ మాటలు ఇక్కడ కనిపిస్తాయి…"
    )
    listen_label = "Click 🎤 to speak" if language == "English" else "🎤 నొక్కి మాట్లాడండి"
    listening_label = "Listening… 🔴" if language == "English" else "వింటున్నాను… 🔴"
    not_supported_msg = (
        "Speech recognition is not supported in this browser. Please use Chrome."
        if language == "English"
        else "ఈ బ్రౌజర్‌లో వాయిస్ గుర్తింపు లేదు. Chrome ఉపయోగించండి."
    )

    return f"""
<div style="font-family:sans-serif;padding:4px 0;">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
    <button id="micBtn"
      onclick="startListening()"
      title="Click to speak"
      style="background:#25D366;color:#fff;border:none;border-radius:50%;
             width:44px;height:44px;font-size:20px;cursor:pointer;
             box-shadow:0 2px 6px rgba(0,0,0,0.25);flex-shrink:0;">
      🎤
    </button>
    <span id="micStatus"
      style="color:#555;font-size:13px;font-style:italic;">
      {listen_label}
    </span>
  </div>
  <input id="transcriptBox" type="text" readonly
    placeholder="{placeholder}"
    style="width:100%;box-sizing:border-box;padding:7px 10px;
           border:1.5px solid #74c69d;border-radius:8px;
           font-size:13px;color:#1a2e1a;background:#f9fdf9;
           outline:none;">
</div>

<script>
(function() {{
  var recognition = null;

  window.startListening = function() {{
    var SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {{
      document.getElementById('micStatus').innerText =
        '{not_supported_msg}';
      document.getElementById('micStatus').style.color = '#c0392b';
      return;
    }}

    if (recognition) {{
      try {{ recognition.stop(); }} catch(e) {{}}
    }}

    recognition = new SpeechRecognition();
    recognition.lang = '{lang_code}';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.continuous = false;

    var btn = document.getElementById('micBtn');
    var status = document.getElementById('micStatus');
    var box = document.getElementById('transcriptBox');

    btn.style.background = '#e74c3c';
    btn.innerHTML = '⏹';
    btn.onclick = function() {{
      recognition.stop();
      btn.style.background = '#25D366';
      btn.innerHTML = '🎤';
      btn.onclick = startListening;
      status.innerText = '{listen_label}';
    }};
    status.innerText = '{listening_label}';
    status.style.color = '#c0392b';

    recognition.onresult = function(event) {{
      var transcript = event.results[0][0].transcript;
      box.value = transcript;
      status.innerText = '✅ ' + transcript;
      status.style.color = '#1b4332';
      btn.style.background = '#25D366';
      btn.innerHTML = '🎤';
      btn.onclick = startListening;
    }};

    recognition.onerror = function(event) {{
      var msg = event.error === 'not-allowed'
        ? '🔒 Microphone permission denied.'
        : '⚠️ Error: ' + event.error;
      status.innerText = msg;
      status.style.color = '#c0392b';
      btn.style.background = '#25D366';
      btn.innerHTML = '🎤';
      btn.onclick = startListening;
    }};

    recognition.onend = function() {{
      btn.style.background = '#25D366';
      btn.innerHTML = '🎤';
      btn.onclick = startListening;
    }};

    try {{
      recognition.start();
    }} catch(e) {{
      status.innerText = '⚠️ Could not start recognition: ' + e.message;
      status.style.color = '#c0392b';
    }}
  }};
}})();
</script>
"""


# ── Voice Output Component ─────────────────────────────────────────────────────

def render_tts_button(
    message_text: str,
    language: str = "English",
    button_id: str = "ttsBtn",
) -> str:
    """
    Return an HTML/JS snippet that renders a 🔊 speak button for a message.

    Clicking the button calls window.speechSynthesis.speak() with the message
    text.  A second click (while speaking) stops playback.

    Args:
        message_text: The assistant message text to speak.
        language:     "English" or "Telugu"
        button_id:    Unique ID for the button element (use message index).

    Returns:
        HTML string suitable for st.components.v1.html(html, height=46)
    """
    lang_code = _lang_code(language)
    # Escape the message text for safe embedding in JS string literals.
    safe_text = (
        html_module.escape(message_text)
        .replace("'", "\\'")
        .replace("\n", " ")
        .replace("\r", "")
    )

    speak_label = "🔊 Listen" if language == "English" else "🔊 వినండి"
    stop_label  = "⏹ Stop"   if language == "English" else "⏹ ఆపు"

    return f"""
<div style="margin-top:4px;">
  <button id="{button_id}"
    onclick="toggleSpeak_{button_id}()"
    style="background:transparent;border:1px solid #74c69d;border-radius:12px;
           padding:3px 10px;font-size:12px;color:#2d6a4f;cursor:pointer;
           font-family:sans-serif;">
    {speak_label}
  </button>
</div>

<script>
(function() {{
  var speaking_{button_id} = false;

  window.toggleSpeak_{button_id} = function() {{
    var btn = document.getElementById('{button_id}');
    if (speaking_{button_id}) {{
      window.speechSynthesis.cancel();
      speaking_{button_id} = false;
      btn.innerText = '{speak_label}';
      return;
    }}
    window.speechSynthesis.cancel();
    var msg = new SpeechSynthesisUtterance('{safe_text}');
    msg.lang = '{lang_code}';
    msg.rate = 0.9;
    msg.pitch = 1.0;
    msg.onstart = function() {{
      speaking_{button_id} = true;
      btn.innerText = '{stop_label}';
    }};
    msg.onend = function() {{
      speaking_{button_id} = false;
      btn.innerText = '{speak_label}';
    }};
    msg.onerror = function() {{
      speaking_{button_id} = false;
      btn.innerText = '{speak_label}';
    }};
    window.speechSynthesis.speak(msg);
  }};
}})();
</script>
"""


# ── Browser support check snippet ─────────────────────────────────────────────

def render_browser_support_banner() -> str:
    """
    Return a small HTML banner that checks for Web Speech API support and
    displays an info/warning message.  Intended to be shown once on page load.

    Returns:
        HTML string suitable for st.components.v1.html(html, height=36)
    """
    return """
<div id="speechSupportBanner" style="display:none;font-family:sans-serif;
  font-size:12px;padding:4px 10px;border-radius:6px;margin-bottom:4px;">
</div>
<script>
(function() {
  var banner = document.getElementById('speechSupportBanner');
  if (!('SpeechRecognition' in window) && !('webkitSpeechRecognition' in window)) {
    banner.style.display = 'block';
    banner.style.background = '#fff3cd';
    banner.style.border = '1px solid #ffc107';
    banner.style.color = '#856404';
    banner.innerText =
      '⚠️ Voice input requires Chrome or Edge. Text input always works.';
  }
  if (!('speechSynthesis' in window)) {
    banner.style.display = 'block';
    banner.style.background = '#fff3cd';
    banner.style.border = '1px solid #ffc107';
    banner.style.color = '#856404';
    banner.innerText =
      '⚠️ Text-to-speech is not supported in this browser.';
  }
})();
</script>
"""


# ── Python-testable helpers ────────────────────────────────────────────────────

def get_language_code(language: str) -> str:
    """
    Return the BCP-47 language tag for the given UI language string.

    >>> get_language_code("English")
    'en-US'
    >>> get_language_code("Telugu")
    'te-IN'
    >>> get_language_code("Unknown")
    'en-US'
    """
    return LANGUAGE_CODES.get(language, "en-US")


def is_html_string(value: str) -> bool:
    """Return True if value looks like an HTML string (starts with '<')."""
    return isinstance(value, str) and value.strip().startswith("<")
