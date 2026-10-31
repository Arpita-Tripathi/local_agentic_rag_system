"""Phase 1 gate (Member B): citations are verified end to end on real PDFs.

Every shown citation maps to a retrieved chunk and a real page; unknown IDs are rejected; an unanswerable
question abstains instead of inventing an answer.
"""

import pathlib

import pytest
from pypdf import PdfReader

from rag import search_backend
from rag.answer import NOT_FOUND
from rag.citations import resolve
from rag.ingest import ingest_pdf
from rag.pipeline import ask
from rag.providers import FakeProvider

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
EN, HI = str(DATA / "sample_text_en.pdf"), str(DATA / "sample_text_hi.pdf")


def _norm(text: str) -> str:
    return " ".join(text.split())


def _ingest_and_index(path):
    ing = ingest_pdf(path, layout="off", ocr=False)
    search, _ = search_backend.build_search(ing["chunks"], ing["doc_hash"])
    return ing, search


# ---------- a cited page really contains the cited text ----------
@pytest.mark.parametrize("path", [EN, HI], ids=["english", "hindi"])
def test_every_chunk_snippet_is_on_the_page_it_cites(path):
    ing = ingest_pdf(path, layout="off", ocr=False)
    reader = PdfReader(path)
    for chunk in ing["chunks"]:
        (citation,) = resolve([chunk["id"]], [chunk])
        page_text = _norm(reader.pages[citation["page"] - 1].extract_text())
        assert _norm(citation["snippet"]) in page_text, (chunk["id"], citation["page"])


def test_answer_cites_the_page_that_holds_the_fact():
    ing, search = _ingest_and_index(EN)
    provider = FakeProvider(["Answer: 91 percent.\nSources: [C1]"])
    res = ask("What recall did hybrid search reach at five passages?", search, provider)
    assert res["found"] and res["reason"] == "ok"
    assert [c["page"] for c in res["citations"]] == [3]
    page_text = _norm(PdfReader(EN).pages[2].extract_text())
    assert all(_norm(c["snippet"]) in page_text for c in res["citations"])
    assert "91 percent" in page_text


def test_hindi_answer_cites_the_right_hindi_page():
    ing, search = _ingest_and_index(HI)
    provider = FakeProvider(["Answer: जुलाई।\nSources: [C1]"])
    res = ask("सबसे अधिक वर्षा किस महीने में हुई?", search, provider)
    assert res["found"] and [c["page"] for c in res["citations"]] == [2]


# ---------- unknown source IDs are rejected ----------
def test_resolve_drops_ids_that_were_not_retrieved(caplog):
    retrieved = [{"id": "c0001", "page": 2, "text": "Training used ten epochs."}]
    out = resolve(["c0001", "c9999", "C7", ""], retrieved)
    assert [c["id"] for c in out] == ["c0001"] and out[0]["page"] == 2
    assert "c9999" in caplog.text  # dropped ids are logged, not silent


def test_a_model_that_cites_a_passage_it_was_not_shown_gets_no_answer():
    ing, search = _ingest_and_index(EN)
    bad = "Answer: 91 percent.\nSources: [C9]"  # only five (or fewer) passages were shown
    question = "What recall did hybrid search reach at five passages?"
    provider = FakeProvider(
        [bad, question, bad]
    )  # the "rewrite" repeats the question, so retrieval is stable
    res = ask(question, search, provider)
    assert res["found"] is False and res["citations"] == []
    assert res["answer"] == NOT_FOUND and res["reason"] == "no_valid_citation"
    assert res["retries"] == 1  # one query rewrite, never more


def test_an_answer_with_no_sources_is_not_trusted():
    ing, search = _ingest_and_index(EN)
    question = "What recall did hybrid search reach at five passages?"
    provider = FakeProvider(["Answer: 91 percent.", question, "Answer: 91 percent."])
    res = ask(question, search, provider)
    assert res["found"] is False and res["reason"] == "no_valid_citation"


# ---------- an unanswerable question abstains ----------
@pytest.mark.parametrize(
    "path,question",
    [
        (EN, "What is the capital of France?"),
        (HI, "भारत की राजधानी क्या है?"),
    ],
    ids=["english", "hindi"],
)
def test_off_topic_question_abstains_without_calling_the_answer_model(path, question):
    ing, search = _ingest_and_index(path)
    provider = FakeProvider(["zzzz qqqq"])  # a rewrite that matches nothing
    res = ask(question, search, provider)
    assert res["found"] is False and res["answer"] == NOT_FOUND and res["citations"] == []
    assert res["reason"] == "low_score"
    assert not any("Excerpts:" in p for p in provider.prompts)  # no evidence prompt was ever sent


def test_model_that_says_not_found_is_shown_as_not_found():
    ing, search = _ingest_and_index(EN)
    question = "How much funding did the study receive?"
    provider = FakeProvider(["NOT_FOUND", question, "NOT_FOUND"])
    res = ask(question, search, provider)
    assert res["found"] is False and res["answer"] == NOT_FOUND and res["citations"] == []
    assert len(provider.prompts) <= 3  # answer, one rewrite, one more answer: bounded
