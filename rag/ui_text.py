"""Wording for the trust signals the UI shows. Plain functions, so the messages can be tested.

Every string returned here may contain text that came from a PDF or a model (a name, a font), so the UI must
pass it through rag.textsafe.escape_markdown before showing it.
"""
from typing import Dict, Optional, Tuple

from rag.lang import LANGUAGE_NAMES

Notice = Tuple[str, str]  # (kind, text) where kind is "warning" or "caption"

_LANG_SHORT = {"hi": "Hindi", "en": "English"}


def support_notice(support: Optional[Dict]) -> Optional[Notice]:
    """What the answer-support check found, or None when there is nothing worth saying."""
    if not support or support["level"] == "nothing_to_check":
        return None
    if support["level"] == "verified":
        return ("caption", "Checked: every number and name in this answer appears in the cited passage(s). "
                           "This does not prove the answer is right, so read the evidence.")
    bad = support["unsupported"]
    parts = []
    if bad["numbers"]:
        parts.append("numbers " + ", ".join(bad["numbers"]))
    if bad["names"]:
        parts.append("names " + ", ".join(bad["names"]))
    return ("warning", "Not found in the cited passages: " + "; ".join(parts) +
            ". The model may have added or changed them, so check the evidence before relying on "
            "this answer.")


def translation_notice(check: Optional[Dict]) -> Optional[Notice]:
    if not check:
        return None
    if check["level"] == "checked":
        if check["checked_numbers"]:
            return ("caption", "Checked: every number in the original appears in the translation.")
        return None
    parts = []
    if check["missing_numbers"]:
        parts.append("numbers missing from the translation: " + ", ".join(check["missing_numbers"]))
    if not check["length_ok"]:
        parts.append("the translation's length looks wrong for this passage")
    return ("warning", "Check this translation against the original: " + "; ".join(parts) + ".")


def cross_lingual_note(result: Dict) -> Optional[str]:
    """A badge when the question and the cited text are in different languages."""
    if not result.get("cross_lingual"):
        return None
    asked = result.get("question_lang", "en")
    source = "en" if asked == "hi" else "hi"
    return f"Asked in {_LANG_SHORT[asked]}, answered from {_LANG_SHORT[source]} text."


def quality_line(quality: Optional[Dict]) -> str:
    """One line about how trustworthy a page's extracted text is."""
    if not quality:
        return ""
    if quality["level"] == "ocr":
        return "Text quality of this page: read by OCR, so accuracy depends on the scan."
    score = f" ({quality['score']:.2f})" if quality["score"] is not None else ""
    line = f"Text quality of this page: {quality['level']}{score}."
    return line + (" " + "; ".join(quality["reasons"]) + "." if quality["reasons"] else "")


def quality_summary_line(summary: Optional[Dict]) -> str:
    if not summary:
        return ""
    order = ("good", "fair", "poor", "none", "ocr")
    names = {"good": "good", "fair": "fair", "poor": "poor", "none": "no text", "ocr": "read by OCR"}
    return "Text quality: " + ", ".join(f"{summary[k]} {names[k]}" for k in order if summary.get(k))


def translation_label(target: str) -> str:
    return f"Translate to {_LANG_SHORT[target]}"


def language_name(code: str) -> str:
    return LANGUAGE_NAMES[code]
