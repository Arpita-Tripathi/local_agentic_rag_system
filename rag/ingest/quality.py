"""How trustworthy is the text we extracted from a page? A score from 0 to 1, with reasons a person can read.

The main failure this guards against is Hindi PDFs set in legacy 8-bit fonts (Kruti Dev, DevLys, Chanakya...).
Those PDFs look fine on screen but extract as Latin-looking gibberish, which retrieval would index as if it
were real text. We detect them by the font names a page declares, which is reliable, and by structural
damage in Devanagari text (words that begin with a vowel sign or a virama), which is a hint, not proof.
Text that merely looks plausible and is wrong cannot be detected here.
"""
import re
from typing import Dict, List

_LEGACY_FONT_HINTS = ("kruti", "devlys", "chanakya", "shreedev", "shree-dev", "shree dev", "shusha")
# Dependent vowel signs, virama, nukta, anusvara etc. cannot start a word in correct Devanagari.
_NEVER_WORD_INITIAL = re.compile(r"^[ऀ-ःऺ-ॏ॑-ॗॢॣ]")
_HAS_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_JUNK = re.compile(r"[�-\x00-\x08\x0b\x0c\x0e-\x1f]|\(cid:\d+\)")


def page_fonts(page) -> List[str]:
    """Base font names declared by a pypdf page (empty on any problem)."""
    names: List[str] = []
    try:
        resources = page.get("/Resources")
        resources = resources.get_object() if resources is not None else {}
        fonts = resources.get("/Font")
        fonts = fonts.get_object() if fonts is not None else {}
        for ref in fonts.values():
            base = ref.get_object().get("/BaseFont")
            if base:
                names.append(str(base).lstrip("/"))
    except Exception:  # a malformed font table must not stop extraction
        return names
    return names


def legacy_font_names(fonts: List[str]) -> List[str]:
    return [f for f in fonts if any(h in f.lower() for h in _LEGACY_FONT_HINTS)]


def orphan_ratio(text: str) -> float:
    """Share of Devanagari words that start with a sign that can never start a word."""
    words = [w for w in text.split() if _HAS_DEVANAGARI.search(w)]
    if len(words) < 5:
        return 0.0
    return sum(1 for w in words if _NEVER_WORD_INITIAL.match(w)) / len(words)


def assess_page(text: str, fonts: List[str]) -> Dict:
    """{"score": 0..1, "level": good|fair|poor|none, "reasons": [...], "legacy_font": bool}."""
    reasons: List[str] = []
    legacy = legacy_font_names(fonts)
    if legacy:
        names = ", ".join(sorted(set(legacy)))
        reasons.append(f"legacy Hindi font ({names}): its text extracts as gibberish")
        return {"score": 0.05, "level": "poor", "reasons": reasons, "legacy_font": True}
    if not (text or "").strip():
        return {"score": 0.0, "level": "none", "legacy_font": False,
                "reasons": ["no text layer on this page"]}

    score = 1.0
    junk = len(_JUNK.findall(text)) / max(len(text), 1)
    if junk > 0.01:
        score -= min(0.9, junk * 8)
        reasons.append(f"{junk:.0%} of the characters are unreadable symbols")
    orphans = orphan_ratio(text)
    if orphans > 0.03:
        score -= min(0.6, orphans * 3)
        reasons.append(
            f"{orphans:.0%} of the Hindi words start with a vowel sign, a sign of broken extraction")
    if len(text.strip()) < 40:
        score -= 0.15
        reasons.append("very little text on this page")
    score = max(0.0, round(score, 2))
    level = "good" if score >= 0.85 else "fair" if score >= 0.6 else "poor"
    return {"score": score, "level": level, "reasons": reasons, "legacy_font": False}


def ocr_quality() -> Dict:
    return {"score": None, "level": "ocr", "legacy_font": False,
            "reasons": ["text was recognised by OCR; accuracy depends on the scan"]}


def summarize(qualities: List[Dict]) -> Dict[str, int]:
    """Counts per level, e.g. {"good": 12, "fair": 1, "poor": 1, "none": 0, "ocr": 2}."""
    counts = {"good": 0, "fair": 0, "poor": 0, "none": 0, "ocr": 0}
    for q in qualities:
        counts[q["level"]] = counts.get(q["level"], 0) + 1
    return counts
