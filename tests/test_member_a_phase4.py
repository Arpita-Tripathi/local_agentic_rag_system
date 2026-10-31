"""Phase 3/4 (Member A): run log, search seam, progress, and the Streamlit app flow."""
import json
import pathlib
import sys
import types

import pytest
from streamlit.testing.v1 import AppTest

from rag import search_backend
from rag.ingest import ingest_pdf
from rag.providers import FakeProvider, ProviderError
from rag.runlog import log_run
from rag.summaries import clear_doc_summaries, summarize_chunk
from tests.test_member_a import _text_pdf

APP = str(pathlib.Path(__file__).resolve().parent.parent / "app.py")

CHUNKS = [
    {"id": "c0000", "doc_hash": "h", "page": 1, "parser": "pypdf@6",
     "text": "The LSTM predicts the next token."},
    {"id": "c0001", "doc_hash": "h", "page": 2, "parser": "pypdf@6",
     "text": "Training used ten epochs on one GPU."},
]


# ---------- run log ----------
def _result(**kw):
    base = {"found": True, "reason": "ok", "retries": 0, "latency_s": 1.23456, "provider": "fake",
            "citations": [{"id": "c0000"}], "trace": [{"attempt": 0}]}
    return {**base, **kw}


def test_run_log_writes_one_json_line_without_pdf_text(tmp_path):
    log_run("What is an LSTM?", _result(), doc_hash="h", cache_dir=str(tmp_path))
    log_run("again", _result(found=False, reason="low_score"), doc_hash="h", cache_dir=str(tmp_path))
    lines = (tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    assert len(lines) == 2 and first["question"] == "What is an LSTM?" and first["latency_s"] == 1.235
    assert first["citations"] == ["c0000"] and first["trace"] == [{"attempt": 0}]
    assert "text" not in first and json.loads(lines[1])["reason"] == "low_score"


def test_run_log_can_hash_questions_only(tmp_path, monkeypatch):
    monkeypatch.setattr("rag.runlog.CONFIG", types.SimpleNamespace(log_questions=False))
    log_run("secret question", _result(), cache_dir=str(tmp_path))
    rec = json.loads((tmp_path / "runs.jsonl").read_text())
    assert rec["question"] is None and len(rec["question_sha256"]) == 16


def test_run_log_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    log_run("q", _result(), cache_dir=str(blocker / "sub"))  # parent is a file: must not raise


# ---------- search seam ----------
def test_keyword_search_ranks_and_scores():
    search = search_backend.keyword_search(CHUNKS)
    hits = search("how many epochs of training", 5)
    assert hits[0]["id"] == "c0001" and 0 < hits[0]["score"] <= 1
    assert search("zebra quantum", 5) == [] and search("", 5) == []
    assert search("lstm token epochs", 1)[0]["id"] in {"c0000", "c0001"} and len(search("lstm", 1)) == 1


def test_keyword_search_handles_hindi():
    chunks = [{"id": "c0", "page": 1, "text": "यह मॉडल दस युग तक प्रशिक्षित किया गया।"},
              {"id": "c1", "page": 2, "text": "Unrelated English text."}]
    assert search_backend.keyword_search(chunks)("मॉडल प्रशिक्षित", 3)[0]["id"] == "c0"


def test_build_search_falls_back_to_stub_when_retrieval_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "rag.retrieval", None)  # simulates "not installed"
    search, name = search_backend.build_search(CHUNKS, "h")
    assert name == search_backend.STUB_NAME and callable(search)


def test_build_search_uses_member_b_when_present(monkeypatch):
    mod = types.ModuleType("rag.retrieval")
    deleted = []

    def real(chunks, doc_hash):
        def s(q, k):
            return []
        s.name = "hybrid (MiniLM + BM25)"
        return s

    mod.build_search, mod.delete_index = real, deleted.append
    monkeypatch.setitem(sys.modules, "rag.retrieval", mod)
    _, name = search_backend.build_search(CHUNKS, "h")
    search_backend.delete_index("h")
    assert name == "hybrid (MiniLM + BM25)" and deleted == ["h"]


def test_missing_dependency_inside_retrieval_is_not_hidden(monkeypatch):
    class Broken(types.ModuleType):
        pass

    def boom(name, *a, **k):
        raise ModuleNotFoundError("No module named 'sentence_transformers'", name="sentence_transformers")

    monkeypatch.delitem(sys.modules, "rag.retrieval", raising=False)
    monkeypatch.setattr("builtins.__import__", lambda name, *a, **k: boom(name) if name == "rag.retrieval"
                        else __import__(name, *a, **k), raising=True)
    with pytest.raises(ModuleNotFoundError, match="sentence_transformers"):
        search_backend.build_search(CHUNKS, "h")


# ---------- ingest progress + summary cleanup ----------
def test_ingest_reports_progress_in_order(tmp_path):
    seen = []
    ingest_pdf(_text_pdf(tmp_path / "t.pdf", ["Enough real text lives on this page for a chunk."]),
               layout="off", progress=lambda msg, frac: seen.append(frac))
    assert seen == sorted(seen) and seen[0] < 0.1 and seen[-1] <= 0.9


def test_clear_doc_summaries_only_removes_that_document(tmp_path):
    a = {"id": "c1", "doc_hash": "docA", "page": 1, "type": "table", "text": "| a |"}
    b = {**a, "doc_hash": "docB"}
    summarize_chunk(a, FakeProvider(["sa"]), str(tmp_path))
    summarize_chunk(b, FakeProvider(["sb"]), str(tmp_path))
    assert clear_doc_summaries("docA", str(tmp_path)) == 1
    assert summarize_chunk(a, FakeProvider(["sa2"]), str(tmp_path))["cached"] is False
    assert summarize_chunk(b, FakeProvider([]), str(tmp_path))["cached"] is True
    assert clear_doc_summaries("docA", str(tmp_path / "nope")) == 0


# ---------- the app ----------
def _doc(parsers=("pypdf@6",), warnings=(), image_only=()):
    return {"doc_hash": "a" * 64, "n_pages": 2, "chunks": CHUNKS, "image_only_pages": list(image_only),
            "parsers": list(parsers), "warnings": list(warnings), "name": "paper.pdf", "bytes": b"%PDF-1.4",
            "file_id": "f1", "search": search_backend.keyword_search(CHUNKS),
            "backend": search_backend.STUB_NAME}


@pytest.fixture
def app(monkeypatch):
    runs, provider = [], FakeProvider([])
    monkeypatch.setattr("rag.providers.get_provider", lambda mode=None, cloud_consent=False: provider)
    monkeypatch.setattr("rag.runlog.log_run", lambda q, r, d="": runs.append((q, r["reason"])))
    monkeypatch.setattr("rag.summaries.clear_doc_summaries", lambda *a, **k: 0)
    monkeypatch.setattr("rag.runlog.purge_doc", lambda *a, **k: 0)

    def make(doc=None):
        at = AppTest.from_file(APP, default_timeout=30)
        if doc is not None:
            at.session_state["doc"] = doc
            at.session_state["history"] = []
        return at.run()

    make.provider, make.runs = provider, runs
    return make


def _texts(elements):
    return [e.value for e in elements]


def test_app_empty_state_prompts_for_upload(app):
    at = app()
    assert not at.exception and any("Upload a PDF" in t for t in _texts(at.info))
    assert any("Local only" in t for t in _texts(at.success))


def test_app_answer_shows_citation_button_evidence_and_provenance(app):
    app.provider.replies = ["Answer: Ten epochs.\nSources: [C1]"]
    at = app(_doc())
    at.chat_input[0].set_value("how many epochs of training").run()
    assert not at.exception
    assert any("Page 2" == b.label for b in at.button)
    page_btn = next(b for b in at.button if b.label == "Page 2")
    page_btn.click().run()
    assert at.session_state["preview_page"] == 2
    caption = " ".join(c.value for c in at.caption)
    assert "fake" in caption and "aaaaaaaa" in caption and "pypdf@6" in caption
    assert app.runs == [("how many epochs of training", "ok")]
    assert any("development mode" in t for t in _texts(at.info))


def test_app_not_found_shows_reason_and_retrieved_passages(app):
    app.provider.replies = ["NOT_FOUND", "rewrite", "NOT_FOUND"]
    at = app(_doc())
    at.chat_input[0].set_value("how many epochs of training").run()
    body = " ".join(m.value for m in at.markdown)
    assert "I could not find this in the PDF" in body  # shown with escaped punctuation
    assert any("Passages that were searched" in e.label for e in at.expander)


def test_app_low_relevance_question_abstains_without_model_call(app):
    app.provider.replies = ["rewritten but still unrelated"]
    at = app(_doc())
    at.chat_input[0].set_value("zebra quantum").run()
    assert "I could not find this in the PDF" in " ".join(m.value for m in at.markdown)
    assert len(app.provider.prompts) == 1 and "Excerpts:" not in app.provider.prompts[0]  # rewrite only


def test_app_provider_failure_shows_message_and_cloud_is_never_automatic(monkeypatch, app):
    def fail(mode=None, cloud_consent=False):
        if mode == "cloud":
            raise AssertionError("cloud provider must not be built without consent")
        raise ProviderError("Model file not found: ./models/x.litertlm. Download it first.")

    monkeypatch.setattr("rag.providers.get_provider", fail)
    at = app(_doc())
    assert any("Model file not found" in t for t in _texts(at.error))
    assert at.chat_input[0].disabled
    assert any("Use the cloud instead" in b.label for b in at.button)


def test_app_cloud_mode_requires_per_document_consent(monkeypatch, app):
    built = []

    def fake_get_provider(mode=None, cloud_consent=False):
        built.append((mode, cloud_consent))
        return app.provider

    monkeypatch.setattr("rag.providers.get_provider", fake_get_provider)
    at = app(_doc())
    at.radio[0].set_value("Send excerpts to cloud").run()
    assert any("Cloud mode" in t for t in _texts(at.warning))
    assert at.chat_input[0].disabled and ("cloud", True) not in built
    at.checkbox[0].check().run()
    assert not at.chat_input[0].disabled and built[-1] == ("cloud", True)
    at.radio[0].set_value("Local only (default)").run()
    at.radio[0].set_value("Send excerpts to cloud").run()
    assert at.checkbox[0].value is False and at.chat_input[0].disabled  # consent is asked again


def test_app_shows_ingest_warnings_and_image_only_pages(app):
    at = app(_doc(warnings=["Pages 2 have no readable text and need OCR. Tesseract is not installed."],
                  image_only=[2]))
    warnings = _texts(at.warning)
    assert any("Tesseract is not installed" in w for w in warnings)
    assert any("No text could be read from page(s) 2" in w for w in warnings)


def test_app_remove_document_clears_state(app):
    at = app(_doc())
    next(b for b in at.button if b.label == "Remove document and index").click().run()
    assert "doc" not in at.session_state and any("Upload a PDF" in t for t in _texts(at.info))


def test_app_uploads_a_real_pdf_end_to_end(tmp_path, app):
    pdf = _text_pdf(tmp_path / "t.pdf", ["The model was trained for ten epochs on a single GPU."])
    at = app()
    at.file_uploader[0].upload("t.pdf", open(pdf, "rb").read(), "application/pdf").run()
    assert not at.exception and at.session_state["doc"]["n_pages"] == 1
    assert at.session_state["doc"]["chunks"][0]["parser"].startswith("pypdf@")


def test_app_rejects_non_pdf_content_with_clear_error(app):
    at = app()
    at.file_uploader[0].upload("fake.pdf", b"not really a pdf", "application/pdf").run()
    assert any("not a PDF" in t for t in _texts(at.error)) and "doc" not in at.session_state
