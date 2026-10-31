"""Open plan items closed in the final pass: index provenance, table explain button, run-log purge."""
import json
import pathlib

import pytest
from streamlit.testing.v1 import AppTest

from rag import search_backend, summaries
from rag.citations import resolve
from rag.providers import FakeProvider, ProviderError
from rag.retrieval import build_search
from rag.runlog import purge_doc
from tests.fakes import FakeEmbedder
from tests.test_member_a_phase4 import CHUNKS, _doc

APP = str(pathlib.Path(__file__).resolve().parent.parent / "app.py")

TABLE = {
    "id": "c0002", "doc_hash": "h", "page": 3, "type": "table", "parser": "docling@2",
    "text": "| quarter | revenue |\n|---|---|\n| first | 100 |\n| second | 140 |",
}


# ---------- citations and index provenance ----------
def test_citations_carry_the_chunk_type():
    cites = resolve(["c0002", "c0000"], [TABLE, CHUNKS[0]])
    assert [(c["id"], c["type"]) for c in cites] == [("c0002", "table"), ("c0000", "text")]


def test_search_exposes_what_the_index_was_built_from(tmp_path):
    chunks = [{"id": "c0", "doc_hash": "h", "page": 1, "text": "alpha beta", "parser": "pypdf@6"}]
    search = build_search(chunks, "abc123", embedder=FakeEmbedder("my-embedder"), cache_dir=str(tmp_path))
    meta = search.index_meta
    assert meta["embedding_model"] == "my-embedder" and meta["n_chunks"] == 1
    assert {"chunk_size", "chunk_overlap", "chunks_sha256"} <= set(meta)


# ---------- run-log purge ----------
def _write_log(folder, records, extra_lines=()):
    lines = [json.dumps(r) for r in records] + list(extra_lines)
    (folder / "runs.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_purge_removes_only_that_documents_records(tmp_path):
    _write_log(tmp_path, [{"doc_hash": "A", "question": "qa1"}, {"doc_hash": "B", "question": "qb"},
                          {"doc_hash": "A", "question": "qa2"}], extra_lines=["not json at all"])
    assert purge_doc("A", cache_dir=str(tmp_path)) == 2
    left = (tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    assert left == [json.dumps({"doc_hash": "B", "question": "qb"}), "not json at all"]
    assert not list(tmp_path.glob("*.tmp"))


def test_purge_handles_missing_log_and_unknown_document(tmp_path):
    assert purge_doc("A", cache_dir=str(tmp_path)) == 0
    _write_log(tmp_path, [{"doc_hash": "B"}])
    before = (tmp_path / "runs.jsonl").read_text(encoding="utf-8")
    assert purge_doc("A", cache_dir=str(tmp_path)) == 0
    assert (tmp_path / "runs.jsonl").read_text(encoding="utf-8") == before


# ---------- app ----------
def _table_doc():
    doc = _doc()
    doc["chunks"] = CHUNKS + [TABLE]
    doc["search"] = search_backend.keyword_search(doc["chunks"])
    return doc


@pytest.fixture
def app(monkeypatch, tmp_path):
    provider = FakeProvider([])
    purged = []
    monkeypatch.setattr("rag.providers.get_provider", lambda mode=None, cloud_consent=False: provider)
    monkeypatch.setattr("rag.runlog.log_run", lambda *a, **k: None)
    monkeypatch.setattr("rag.runlog.purge_doc", lambda doc_hash, *a, **k: purged.append(doc_hash) or 0)
    monkeypatch.setattr("rag.summaries.clear_doc_summaries", lambda *a, **k: 0)
    real = summaries.summarize_chunk  # keep the disk cache out of ./index_cache
    monkeypatch.setattr("rag.summaries.summarize_chunk",
                        lambda chunk, generate: real(chunk, generate, cache_dir=str(tmp_path / "cache")))

    def make(doc=None):
        at = AppTest.from_file(APP, default_timeout=30)
        if doc is not None:
            at.session_state["doc"] = doc
            at.session_state["history"] = []
        return at.run()

    make.provider, make.purged = provider, purged
    return make


def _ask_about_the_table(app):
    app.provider.replies = ["Answer: 100.\nSources: [C1]"]
    at = app(_table_doc())
    at.chat_input[0].set_value("what was revenue for the first quarter").run()
    assert not at.exception
    return at


def _labels(at):
    return [b.label for b in at.button]


def test_explain_button_appears_only_for_table_citations(app):
    at = _ask_about_the_table(app)
    assert "Explain this table" in _labels(at)

    app.provider.replies = ["Answer: ok.\nSources: [C1]"]
    plain = app(_doc())
    plain.chat_input[0].set_value("how many epochs of training").run()
    assert "Explain this table" not in _labels(plain)


def test_explain_table_makes_one_call_then_remembers(app):
    at = _ask_about_the_table(app)
    calls_before = len(app.provider.prompts)
    app.provider.replies = ["A table of revenue by quarter."]
    next(b for b in at.button if b.label == "Explain this table").click().run()
    assert len(app.provider.prompts) == calls_before + 1
    assert "table" in app.provider.prompts[-1].lower() and "untrusted" in app.provider.prompts[-1]
    assert any("A table of revenue by quarter" in m.value for m in at.markdown)
    at.run()  # a later rerun shows the stored summary and never asks the model again
    assert len(app.provider.prompts) == calls_before + 1
    assert "Explain this table" not in _labels(at)


def test_explain_table_failure_is_shown_not_raised(monkeypatch, app):
    at = _ask_about_the_table(app)

    def fail(chunk, generate):
        raise ProviderError("Cloud provider rate limit reached (429). Wait a bit and try again.")

    monkeypatch.setattr("rag.summaries.summarize_chunk", fail)
    at = app()  # fresh run with the failing summariser
    at.session_state["doc"], at.session_state["history"] = _table_doc(), []
    app.provider.replies = ["Answer: 100.\nSources: [C1]"]
    at.run()
    at.chat_input[0].set_value("what was revenue for the first quarter").run()
    next(b for b in at.button if b.label == "Explain this table").click().run()
    assert not at.exception and any("429" in e.value for e in at.error)


def test_sidebar_shows_index_provenance(app):
    doc = _doc()
    search = search_backend.keyword_search(CHUNKS)
    search.index_meta = {"n_chunks": 2, "chunk_size": 1000, "chunk_overlap": 150, "embedding_model": "mx"}
    doc["search"] = search
    text = " ".join(c.value for c in app(doc).caption)
    assert "Index: 2 passages of up to 1000 characters" in text
    assert "overlap 150" in text and "embedding model mx" in text


def test_removing_a_document_purges_its_run_log_records(app):
    at = app(_doc())
    next(b for b in at.button if b.label == "Remove document and index").click().run()
    assert app.purged == ["a" * 64]
