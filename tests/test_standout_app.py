"""Trust features as the user sees them: support notice, cross-lingual badge, translation, page quality."""
import pathlib

import pytest
from streamlit.testing.v1 import AppTest

from rag import translate
from rag.providers import FakeProvider
from rag.ui_text import (
    cross_lingual_note,
    quality_line,
    quality_summary_line,
    support_notice,
    translation_notice,
)
from tests.test_member_a_phase4 import CHUNKS, _doc

APP = str(pathlib.Path(__file__).resolve().parent.parent / "app.py")
HINDI_CHUNK = {"id": "c0003", "doc_hash": "h", "page": 1, "type": "text", "parser": "pypdf@6",
               "text": "रामपुर में 12 वर्षामापी यंत्र लगाए गए थे।"}


# ---------- wording ----------
def test_support_notice_wording():
    assert support_notice(None) is None
    assert support_notice({"level": "nothing_to_check"}) is None
    kind, text = support_notice({"level": "verified"})
    assert kind == "caption" and "does not prove" in text
    unverified = {"level": "unverified", "unsupported": {"numbers": ["64"], "names": ["Berlin"]}}
    kind, text = support_notice(unverified)
    assert kind == "warning" and "numbers 64" in text and "names Berlin" in text


def test_translation_notice_wording():
    ok = {"level": "checked", "missing_numbers": [], "length_ok": True, "checked_numbers": ["10"]}
    assert translation_notice(ok)[0] == "caption"
    assert translation_notice({**ok, "checked_numbers": []}) is None
    bad = {"level": "warning", "missing_numbers": ["10"], "length_ok": False, "checked_numbers": ["10"]}
    kind, text = translation_notice(bad)
    assert kind == "warning" and "missing from the translation: 10" in text and "length" in text


def test_cross_lingual_note_wording():
    assert cross_lingual_note({"cross_lingual": False}) is None
    assert cross_lingual_note({"cross_lingual": True, "question_lang": "hi"}) == \
        "Asked in Hindi, answered from English text."
    assert cross_lingual_note({"cross_lingual": True, "question_lang": "en"}) == \
        "Asked in English, answered from Hindi text."


def test_quality_wording():
    assert "read by OCR" in quality_line({"level": "ocr", "score": None, "reasons": ["x"]})
    line = quality_line({"level": "poor", "score": 0.05, "reasons": ["legacy Hindi font (Kruti)"]})
    assert "poor (0.05)" in line and "legacy Hindi font" in line
    assert quality_line(None) == ""
    summary = {"good": 12, "fair": 0, "poor": 1, "none": 0, "ocr": 2}
    assert quality_summary_line(summary) == "Text quality: 12 good, 1 poor, 2 read by OCR"
    assert quality_summary_line(None) == ""


# ---------- app ----------
@pytest.fixture
def app(monkeypatch, tmp_path):
    provider = FakeProvider([])
    removed = []
    monkeypatch.setattr("rag.providers.get_provider", lambda mode=None, cloud_consent=False: provider)
    monkeypatch.setattr("rag.runlog.log_run", lambda *a, **k: None)
    monkeypatch.setattr("rag.runlog.purge_doc", lambda *a, **k: 0)
    monkeypatch.setattr("rag.summaries.clear_doc_summaries", lambda *a, **k: 0)
    monkeypatch.setattr("rag.translate.clear_doc_translations", lambda h, *a, **k: removed.append(h) or 0)
    real = translate.translate_chunk  # keep the disk cache out of ./index_cache
    monkeypatch.setattr("rag.translate.translate_chunk", lambda chunk, generate, target="": real(
        chunk, generate, target, cache_dir=str(tmp_path / "cache")))

    def make(doc=None):
        at = AppTest.from_file(APP, default_timeout=30)
        if doc is not None:
            at.session_state["doc"] = doc
            at.session_state["history"] = []
        return at.run()

    make.provider, make.removed = provider, removed
    return make


def _ask(app, reply, question="how many epochs of training", doc=None):
    app.provider.replies = [reply]
    at = app(doc or _doc())
    at.chat_input[0].set_value(question).run()
    assert not at.exception
    return at


def _texts(elements):
    """Raw rendered values: untrusted text must still be escaped here."""
    return [e.value for e in elements]


def _plain(elements):
    """The same values with markdown escapes removed, for reading what the user would see."""
    return [e.value.replace("\\", "") for e in elements]


def test_verified_answer_shows_the_check_caption(app):
    at = _ask(app, "Answer: 10 epochs.\nSources: [C1]")  # the passage says "ten": number words are read
    assert any("Checked: every number and name" in t for t in _plain(at.caption))
    assert not any("Not found in the cited passages" in t for t in _plain(at.warning))


def test_invented_number_shows_a_warning_but_keeps_the_answer(app):
    at = _ask(app, "Answer: 12 epochs.\nSources: [C1]")
    warning = " ".join(_plain(at.warning))
    assert "Not found in the cited passages" in warning and "numbers 12" in warning
    assert any("12 epochs" in t for t in _plain(at.markdown))  # still shown, flagged not hidden
    assert any(b.label == "Page 2" for b in at.button)


def test_hindi_question_over_english_passage_shows_the_cross_lingual_badge(app):
    doc = _doc()
    doc["search"] = lambda q, k: [dict(CHUNKS[1], score=0.9)]
    at = _ask(app, "Answer: १० युग।\nSources: [C1]", "प्रशिक्षण में कितने युग लगे?", doc)
    assert any("Asked in Hindi, answered from English text" in t for t in _plain(at.caption))
    assert any("Checked: every number" in t for t in _plain(at.caption))  # Devanagari digits match
    assert "Hindi (Devanagari script)" in app.provider.prompts[0]


def test_translate_button_shows_original_and_translation_side_by_side(app):
    at = _ask(app, "Answer: 10 epochs.\nSources: [C1]")
    assert "Translate to Hindi" in [b.label for b in at.button]
    app.provider.replies = ["प्रशिक्षण में एक GPU पर 10 युग लगे।"]
    next(b for b in at.button if b.label == "Translate to Hindi").click().run()
    assert not at.exception
    markdown = " ".join(_plain(at.markdown))
    assert "Training used ten epochs on one GPU" in markdown and "प्रशिक्षण में एक GPU पर 10 युग लगे" in markdown
    assert any("Original, page 2" in t for t in _plain(at.caption))
    assert any("Hindi (machine translation)" in t for t in _plain(at.caption))
    assert any("Checked: every number in the original" in t for t in _plain(at.caption))
    calls = len(app.provider.prompts)
    at.run()  # a rerun shows the stored result and never asks the model again
    assert len(app.provider.prompts) == calls


def test_translation_that_drops_a_number_is_flagged(app):
    at = _ask(app, "Answer: 10 epochs.\nSources: [C1]")
    app.provider.replies = ["प्रशिक्षण में कुछ युग लगे।"]
    next(b for b in at.button if b.label == "Translate to Hindi").click().run()
    assert any("Check this translation against the original" in t and "10" in t for t in _plain(at.warning))


def test_hindi_passage_gets_a_translate_to_english_button(app):
    doc = _doc()
    doc["chunks"] = CHUNKS + [HINDI_CHUNK]
    doc["search"] = lambda q, k: [dict(HINDI_CHUNK, score=0.9)]
    at = _ask(app, "Answer: 12.\nSources: [C1]", "how many gauges", doc)
    assert "Translate to English" in [b.label for b in at.button]


def test_translation_failure_is_shown_not_raised(monkeypatch, app):
    from rag.providers import ProviderError

    at = _ask(app, "Answer: 10 epochs.\nSources: [C1]")

    def fail(chunk, generate, target=""):
        raise ProviderError("Cloud provider rate limit reached (429). Wait a bit and try again.")

    monkeypatch.setattr("rag.translate.translate_chunk", fail)
    at = app()
    at.session_state["doc"], at.session_state["history"] = _doc(), []
    app.provider.replies = ["Answer: 10 epochs.\nSources: [C1]"]
    at.run()
    at.chat_input[0].set_value("how many epochs of training").run()
    next(b for b in at.button if b.label == "Translate to Hindi").click().run()
    assert not at.exception and any("429" in t for t in _texts(at.error))


def test_removing_a_document_clears_its_translations(app):
    at = app(_doc())
    next(b for b in at.button if b.label == "Remove document and index").click().run()
    assert app.removed == ["a" * 64]


def test_page_quality_is_shown_in_the_sidebar_and_preview(app):
    doc = _doc()
    doc["quality_summary"] = {"good": 1, "fair": 0, "poor": 1, "none": 0, "ocr": 0}
    doc["page_quality"] = [
        {"page": 1, "score": 1.0, "level": "good", "reasons": [], "legacy_font": False},
        {"page": 2, "score": 0.05, "level": "poor", "legacy_font": True,
         "reasons": ["legacy Hindi font (KrutiDev010): its text extracts as gibberish"]},
    ]
    at = app(doc)
    captions = _plain(at.caption)
    assert any("Text quality: 1 good, 1 poor" in t for t in captions)
    assert any(e.label == "Pages to double-check (1)" for e in at.expander)
    assert any("Page 2" in t and "legacy Hindi font" in t for t in _plain(at.markdown))
    assert any("Text quality of this page: good (1.00)" in t for t in captions)  # preview starts on page 1


def test_a_hostile_font_name_in_the_quality_reasons_is_escaped(app):
    doc = _doc()
    doc["page_quality"] = [{"page": 1, "score": 0.05, "level": "poor", "legacy_font": True,
                            "reasons": ["legacy Hindi font (![x](https://evil.example/a.png))"]}]
    doc["quality_summary"] = {"good": 0, "fair": 0, "poor": 1, "none": 0, "ocr": 0}
    at = app(doc)
    for value in _texts(at.markdown) + _texts(at.caption):
        assert "![x](" not in value and "https://evil" not in value
