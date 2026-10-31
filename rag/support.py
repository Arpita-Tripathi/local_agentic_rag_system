"""Deterministic checks that an answer (or a translation) does not contain facts its source lacks.

These are cheap string checks, not understanding. They catch the common, damaging failures: a number that
is not in the cited text, or a proper name the model made up. They cannot prove an answer is right, and they
say "nothing to check" for answers without numbers or names. The check is deliberately conservative: it flags,
it never rewrites, and it reads number words only on the source side so that "10" is found in "ten" but a
Hindi "एक" (the article "a") never becomes a number to verify.
"""
import re
from typing import Dict, Iterable, List, Set

from rag.lang import script_counts

_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_GROUPED = re.compile(r"(?<=\d),(?=\d)")  # 1,000 and Indian 1,00,000
_LIST_MARKER = re.compile(r"(?m)^\s*\d+[.)]\s")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
# Letters plus Devanagari vowel signs and marks (\w alone splits Hindi words at every matra),
# but not the danda (U+0964, U+0965) or Devanagari digits (U+0966 to U+096F).
_WORD = re.compile(r"(?:[^\W\d_]|[\u0900-\u0963\u0970-\u097f])+")

_EN_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
_EN_TENS = {w: 10 * i for i, w in enumerate(
    "x x twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "x"}
_HI_UNITS = {"शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5, "छह": 6, "छः": 6,
             "सात": 7, "आठ": 8, "नौ": 9, "दस": 10, "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14,
             "पंद्रह": 15, "पन्द्रह": 15, "सोलह": 16, "सत्रह": 17, "अठारह": 18, "उन्नीस": 19, "बीस": 20}

# Capitalised words that start sentences or are plain function words: not names to verify.
_NOT_NAMES = frozenset(
    "the this that these those there here it its in on at by of to for from with without and but or if as "
    "he she they we you i a an according however also yes no not none all some each both many most other "
    "page pages answer sources source based per".split())
_CAPITALISED = re.compile(r"(?<![A-Za-z])[A-Z][a-z][A-Za-z]+")


def normalize(text: str) -> str:
    """Devanagari digits to ASCII, thousands separators removed."""
    return _GROUPED.sub("", (text or "").translate(_DEVANAGARI_DIGITS))


# "one" and "एक" are also the article/pronoun ("one of the files", "एक गाँव" = "a village"), so a text that
# merely contains them does not necessarily state the number 1. They never create a number a translation
# must keep, but they may still satisfy a number that is being looked for.
_AMBIGUOUS_WORDS = frozenset({"one", "एक"})


def numbers_in(text: str, read_words: bool = False, include_ambiguous: bool = True) -> Set[str]:
    """Numbers as digits; with read_words=True also English and Hindi number words (source side only).

    include_ambiguous=False leaves out "one" and "एक", for building the set of numbers a text *requires*.
    """
    text = normalize(text)
    found = set(_NUMBER.findall(_LIST_MARKER.sub(" ", text)))
    found = {n.rstrip("0").rstrip(".") if "." in n else n for n in found}
    if read_words:
        words = [w.lower() for w in _WORD.findall(text)]
        for i, w in enumerate(words):
            if w in _AMBIGUOUS_WORDS and not include_ambiguous:
                continue
            if w in _EN_TENS:
                unit = _EN_UNITS.get(words[i + 1]) if i + 1 < len(words) else None
                found.add(str(_EN_TENS[w] + (unit if unit and 0 < unit < 10 else 0)))
            elif w in _EN_UNITS:
                found.add(str(_EN_UNITS[w]))
            elif w in _HI_UNITS:
                found.add(str(_HI_UNITS[w]))
    return found


def _names_in(text: str) -> List[str]:
    """Capitalised words that are not the first word of a sentence."""
    names = []
    for m in _CAPITALISED.finditer(text):
        before = text[: m.start()].rstrip()
        if not before or before[-1] in ".!?:\n":
            continue
        if m.group().lower() not in _NOT_NAMES:
            names.append(m.group())
    return names


def _tokens(text: str) -> Set[str]:
    return {w.lower() for w in _WORD.findall(text or "")}


def check_support(answer: str, passages: Iterable[str], question: str = "") -> Dict:
    """Which numbers and names in `answer` are missing from the cited `passages`.

    Numbers in the question are allowed (an answer may repeat them). Names are only checked when both the
    answer and every passage are in Latin script; across scripts only numbers are comparable.
    Returns {"level", "checked", "unsupported"}: level is "verified", "unverified" or "nothing_to_check".
    """
    passages = [p for p in passages if p]
    source_text = "\n".join(passages)
    allowed_numbers = numbers_in(source_text, read_words=True) | numbers_in(question)
    checked_numbers = sorted(numbers_in(answer), key=lambda n: (len(n), n))
    bad_numbers = [n for n in checked_numbers if n not in allowed_numbers]

    checked_names: List[str] = []
    bad_names: List[str] = []
    answer_dev, _ = script_counts(answer)
    source_dev, _ = script_counts(source_text)
    if passages and not answer_dev and not source_dev:
        known = _tokens(source_text) | _tokens(question)
        for name in dict.fromkeys(_names_in(answer)):
            checked_names.append(name)
            if name.lower() not in known:
                bad_names.append(name)

    unsupported = {"numbers": bad_numbers, "names": bad_names}
    checked = {"numbers": checked_numbers, "names": checked_names}
    if not checked_numbers and not checked_names:
        level = "nothing_to_check"
    else:
        level = "unverified" if bad_numbers or bad_names else "verified"
    return {"level": level, "checked": checked, "unsupported": unsupported}


def check_translation(source: str, translation: str) -> Dict:
    """Did the translation keep every number of the source? Also flags an implausible length ratio.

    Number words are read on both sides: the translation may write "ten" or "दस" where the source has "10".
    """
    wanted = numbers_in(source, read_words=True, include_ambiguous=False)
    have = numbers_in(translation, read_words=True)
    missing = sorted(wanted - have, key=lambda n: (len(n), n))
    ratio = len((translation or "").strip()) / max(len((source or "").strip()), 1)
    length_ok = 0.3 <= ratio <= 3.5
    return {"level": "warning" if missing or not length_ok else "checked",
            "missing_numbers": missing, "length_ok": length_ok, "checked_numbers": sorted(wanted)}
