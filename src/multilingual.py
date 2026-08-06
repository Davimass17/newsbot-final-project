"""Multilingual tools for NewsBot."""

from langdetect import detect
from deep_translator import GoogleTranslator


def detect_language(text: str) -> str:
    """Detect the language of a text."""
    if not text.strip():
        return "unknown"

    try:
        return detect(text)
    except Exception:
        return "unknown"


def translate_text(
    text: str,
    target_language: str = "en",
) -> str:
    """Translate text into another language."""
    if not text.strip():
        return ""

    try:
        return GoogleTranslator(
            source="auto",
            target=target_language,
        ).translate(text)
    except Exception:
        return text
