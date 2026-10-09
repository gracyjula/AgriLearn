"""
safety.py — AgriLearn AI
Input validation and safety classification for user messages.

This module provides a first-pass heuristic check BEFORE sending a message to the
LLM.  It is intentionally conservative:

  - It does NOT rely solely on keyword matching.  Simple words like "fertilizer"
    in a clearly educational question ("what is fertilizer?") are allowed.
  - It flags messages that combine action verbs / request framing with prohibited
    topics (e.g. "recommend a fertilizer", "how much pesticide should I apply").

IMPORTANT: This module is a supplementary safeguard, not a guarantee.  The primary
safety mechanism is the system prompt inside the LLM.  Prompt-based safety controls
cannot be guaranteed to be 100% reliable.
"""

import re
from dataclasses import dataclass

# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class SafetyResult:
    """Result of a safety check on a user message."""
    is_safe: bool
    reason: str  # Human-readable explanation (shown to user when is_safe=False)
    category: str  # Internal category label


# ── Allowed topic patterns (educational questions) ──────────────────────────────

ALLOWED_TOPIC_PATTERNS: list[tuple[str, str]] = [
    # (regex pattern, label)
    (r"\b(crop|plant|seed|seedling)\s+(growth|lifecycle|stages?|cycle|germination)\b", "crop_lifecycle"),
    (r"\b(sow|sowing|germination|germinate|sprout)\b", "sowing"),
    (r"\b(irrigation|watering|water\s+cycle|drip\s+irrigation|flood\s+irrigation)\b", "irrigation"),
    (r"\b(harvest|harvesting|reaping|crop\s+cutting)\b", "harvesting"),
    (r"\b(storage|storing|grain\s+storage|silo|post.?harvest)\b", "storage"),
    (r"\b(drying|grain\s+dry|sun\s+dr[yi])\b", "drying"),
    (r"\b(farming\s+process|farming\s+practice|agricultural\s+method|crop\s+rotation)\b", "farming_process"),
    (r"\b(what\s+is|explain|describe|tell\s+me|how\s+does|what\s+are|define)\b", "educational_question"),
]

# ── Prohibited action patterns ─────────────────────────────────────────────────

# Combinations of REQUEST verbs + prohibited subjects are flagged.
REQUEST_VERBS = r"(recommend|suggest|tell\s+me\s+to\s+use|advise|prescribe|how\s+much|what\s+dose|apply|use|give\s+me|provide\s+(a|the|some)|what\s+should\s+i\s+(use|apply|do|buy|spray|add))"

PROHIBITED_PATTERNS: list[tuple[str, str]] = [
    # (regex pattern, refusal reason)
    (
        rf"{REQUEST_VERBS}.{{0,60}}\b(fertilizer|fertiliser|npk|urea|dap|compost\s+amount|manure\s+amount)\b",
        "fertilizer_recommendation",
    ),
    (
        rf"{REQUEST_VERBS}.{{0,60}}\b(pesticide|herbicide|fungicide|insecticide|chemical\s+spray|weedicide)\b",
        "pesticide_recommendation",
    ),
    (
        rf"{REQUEST_VERBS}.{{0,60}}\b(treat(ment)?|cure|fix|solve).{{0,40}}\b(disease|pest|blight|rot|wilt|fungus|infection)\b",
        "crop_treatment_advice",
    ),
    (
        r"\b(diagnos|identify\s+the\s+disease|what\s+disease|my\s+crop\s+is\s+(sick|dying|wilting|infected|affected))\b",
        "disease_diagnosis",
    ),
    (
        r"\b(yield\s+prediction|predict\s+.{0,20}yield|yield\s+forecast|how\s+many\s+(kg|tons?|quintals?).{0,30}harvest|forecast.{0,30}harvest|estimate.{0,30}yield|can\s+you\s+predict.{0,30}yield)\b",
        "yield_prediction",
    ),
    (
        r"\b(ignore\s+(your\s+)?(instructions?|rules?|prompt|guidelines?|restrictions?)|you\s+are\s+now\s+a|pretend\s+you\s+are|act\s+as\s+a\s+different|override\s+your|forget\s+your\s+(rules?|instructions?))\b",
        "prompt_injection",
    ),
]

# ── Refusal messages ───────────────────────────────────────────────────────────

REFUSAL_MESSAGES: dict[str, str] = {
    "fertilizer_recommendation": (
        "I can explain what fertilizers are and how they work in general, "
        "but I'm not able to recommend specific fertilizers, quantities, or "
        "application schedules. For personalized advice, please contact a "
        "qualified agricultural extension officer or your local Krishi Vigyan Kendra (KVK)."
    ),
    "pesticide_recommendation": (
        "I can explain what pesticides are in general educational terms, "
        "but I cannot recommend specific pesticide products, dosages, or "
        "application methods. Please consult a licensed agrochemical advisor "
        "or your local agricultural department."
    ),
    "crop_treatment_advice": (
        "I provide general agricultural education and cannot diagnose crop "
        "problems or recommend specific treatments. Please consult a qualified "
        "plant pathologist or agricultural extension officer for treatment advice."
    ),
    "disease_diagnosis": (
        "I'm an educational chatbot and I cannot diagnose crop diseases or "
        "pest infestations. For an accurate diagnosis, please visit your nearest "
        "agricultural extension office or contact a plant disease specialist."
    ),
    "yield_prediction": (
        "I cannot predict crop yields, as this depends on many local factors "
        "such as soil quality, weather, and farm management practices. "
        "For yield estimates, consult your local agricultural department or "
        "use an official crop modeling tool."
    ),
    "prompt_injection": (
        "I'm AgriLearn AI, an agricultural education assistant. I can only "
        "answer questions about crop lifecycles and general farming processes. "
        "I'm not able to change my role or override my guidelines."
    ),
}

DEFAULT_REFUSAL = (
    "I'm an educational assistant focused on general agricultural knowledge. "
    "I'm not able to provide specific recommendations or personalized advice. "
    "For guidance tailored to your situation, please contact a qualified "
    "agricultural extension officer."
)

# ── Public API ─────────────────────────────────────────────────────────────────

def check_input(user_message: str) -> SafetyResult:
    """
    Run a heuristic safety check on a user message.

    Returns a SafetyResult with:
      - is_safe=True  → pass the message to the LLM.
      - is_safe=False → show the refusal message directly to the user.

    NOTE: This does not replace the LLM's own safety policy.
    """
    if not user_message or not user_message.strip():
        return SafetyResult(
            is_safe=False,
            reason="Please type a question to get started.",
            category="empty_input",
        )

    text = user_message.lower().strip()

    # Check prohibited patterns
    for pattern, category in PROHIBITED_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            reason = REFUSAL_MESSAGES.get(category, DEFAULT_REFUSAL)
            return SafetyResult(is_safe=False, reason=reason, category=category)

    return SafetyResult(is_safe=True, reason="", category="allowed")


def get_refusal_message(category: str) -> str:
    """Return the user-facing refusal message for a given category."""
    return REFUSAL_MESSAGES.get(category, DEFAULT_REFUSAL)


def is_educational_question(user_message: str) -> bool:
    """
    Heuristic check: does this message look like a general educational question?
    Used to allow messages that contain prohibited keywords in a clearly
    educational framing (e.g., "what is a fertilizer?").
    """
    text = user_message.lower()
    for pattern, _ in ALLOWED_TOPIC_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False
