"""Tiny language helpers: which script a text is written in, and which language to answer in.

Only Hindi (Devanagari) and English (Latin script) are distinguished. Roman-script Hindi (Hinglish) counts
as English here, so a Hinglish question is answered in English, which a Hinglish reader can read.
"""
import re

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")

LANGUAGE_NAMES = {"hi": "Hindi (Devanagari script)", "en": "English"}


def script_counts(text: str):
    return len(_DEVANAGARI.findall(text or "")), len(_LATIN.findall(text or ""))


def detect_language(text: str) -> str:
    """"hi" when Devanagari letters are at least as common as Latin ones, otherwise "en"."""
    dev, lat = script_counts(text)
    return "hi" if dev and dev >= lat else "en"


def other_language(lang: str) -> str:
    return "en" if lang == "hi" else "hi"


def is_cross_lingual(question: str, passages) -> bool:
    """True when the question is in one language and every cited passage is in the other."""
    langs = {detect_language(p) for p in passages if script_counts(p) != (0, 0)}
    return bool(langs) and detect_language(question) not in langs
