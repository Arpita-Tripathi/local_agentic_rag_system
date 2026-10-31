"""Member B: the golden question set, the metrics, and the benchmark runner."""

import copy
import json
import pathlib

import pytest

from eval import run_eval
from eval.bench_parsers import evidence_kept, run_benchmark
from eval.run_eval import (
    check_ground_truth,
    evaluate,
    format_report,
    ingest_docs,
    load_questions,
    summarize,
    weak_cases,
)
from tests.fakes import FakeEmbedder

ROOT = pathlib.Path(__file__).resolve().parent.parent
QUESTIONS = load_questions()


# ---------- the golden set ----------
def test_golden_set_has_the_size_and_mix_the_plan_asks_for():
    assert 50 <= len(QUESTIONS) <= 100
    assert len({q["id"] for q in QUESTIONS}) == len(QUESTIONS)
    assert {q["language"] for q in QUESTIONS} == {"en", "hi", "hinglish", "mixed"}
    assert {q["case_type"] for q in QUESTIONS} == {"text", "table", "ocr"}
    assert any(q["answerable"] for q in QUESTIONS) and any(not q["answerable"] for q in QUESTIONS)
    assert {q.get("unanswerable_kind") for q in QUESTIONS if not q["answerable"]} == {
        "off_topic",
        "near_miss",
    }


def test_every_question_names_an_existing_pdf_and_has_the_required_fields():
    for q in QUESTIONS:
        assert (ROOT / q["doc"]).is_file(), q["id"]
        assert q["question"].strip() and isinstance(q["expected_pages"], list)
        if q["answerable"]:
            assert q["expected_answer"] and q["expected_pages"] and q["keywords"], q["id"]


def test_ground_truth_matches_the_pdfs():
    docs, skipped = ingest_docs(QUESTIONS, ocr=False)  # the scanned PDF needs Tesseract: checked when present
    assert set(skipped) <= {"data/sample_scanned.pdf"}
    assert check_ground_truth(QUESTIONS, docs, skipped) == []


def test_the_check_catches_a_wrong_gold_page_and_a_wrong_keyword():
    docs, skipped = ingest_docs(QUESTIONS, ocr=False)
    broken = copy.deepcopy(QUESTIONS)
    first = next(q for q in broken if q["id"] == "en07")
    first["expected_pages"] = [1]  # the 91 percent figure is on page 3
    second = next(q for q in broken if q["id"] == "en08")
    second["keywords"] = ["nonexistent value"]
    third = next(q for q in broken if q["id"] == "en01")
    third["expected_pages"] = [99]
    problems = check_ground_truth(broken, docs, skipped)
    assert any(p.startswith("en07") for p in problems)
    assert any(p.startswith("en08") for p in problems)
    assert any(p.startswith("en01") and "outside" in p for p in problems)


needs_ocr_marker = pytest.mark.skipif(
    not run_eval.ocr_available()[0], reason="Tesseract with Hindi+English data is not installed"
)


@needs_ocr_marker
def test_ground_truth_of_the_ocr_questions_when_tesseract_is_present():
    ocr_questions = [q for q in QUESTIONS if q["case_type"] == "ocr"]
    docs, skipped = ingest_docs(ocr_questions, ocr=True)
    assert check_ground_truth(ocr_questions, docs, skipped) == []


# ---------- metrics ----------
def _row(id, answerable=True, rank=1, kept=True, kind=None, score=0.5, language="en", case="text"):
    hit = {k: rank is not None and rank <= k for k in (1, 3, 5)}
    return {
        "id": id,
        "language": language,
        "case_type": case,
        "answerable": answerable,
        "kind": kind,
        "question": "q",
        "top": [],
        "gold_rank": rank,
        "hit": hit,
        "kept_gold": kept,
        "abstained": not kept if not answerable else False,
        "top_score": score,
        "latency_ms": 10.0,
    }


def test_summary_math():
    rows = [
        _row("a", rank=1),
        _row("b", rank=2),
        _row("c", rank=4),
        _row("d", rank=None, kept=False),
        _row("u1", answerable=False, rank=None, kept=False, kind="off_topic", score=0.05),
        _row("u2", answerable=False, rank=None, kept=True, kind="near_miss", score=0.6),
    ]
    s = summarize(rows)
    assert (s["n_answerable"], s["n_unanswerable"]) == (4, 2)
    assert s["recall@1"] == 0.25 and s["recall@3"] == 0.5 and s["recall@5"] == 0.75
    assert s["mrr@5"] == round((1 + 1 / 2 + 1 / 4) / 4, 4)
    assert s["recall@5_kept"] == 0.75
    assert s["abstain_off_topic"] == 1.0 and s["abstain_near_miss"] == 0.0 and s["abstain_retrieval"] == 0.5
    assert s["top_score_unanswerable_max"] == 0.6 and s["search_ms_median"] == 10.0


def test_summary_of_nothing_does_not_divide_by_zero():
    s = summarize([])
    assert s["recall@5"] is None and s["mrr@5"] is None and s["abstain_retrieval"] is None


def test_weak_cases_explain_each_failure():
    rows = [
        _row("ok"),
        _row("miss", rank=None, kept=False),
        _row("late", rank=3),
        _row("cut", rank=1, kept=False, score=0.1),
        _row("bad", answerable=False, rank=None, kept=True, kind="off_topic", score=0.6),
        _row(
            "near", answerable=False, rank=None, kept=True, kind="near_miss", score=0.6
        ),  # not retrieval's fault
        _row("fine", answerable=False, rank=None, kept=False, kind="off_topic", score=0.0),
    ]
    why = {w["id"]: w["why"] for w in weak_cases(rows)}
    assert set(why) == {"miss", "late", "cut", "bad"}
    assert "top 5" in why["miss"] and "ranked 3" in why["late"] and "threshold" in why["cut"]
    assert "off-topic" in why["bad"]


def test_answer_level_metrics_count_unsupported_claims():
    rows = [_row("a"), _row("u", answerable=False, rank=None, kept=True, kind="near_miss")]
    rows[0]["answer"] = {"found": True, "supported": True, "citation_ok": True, "latency_s": 2.0}
    rows[1]["answer"] = {"found": True, "supported": False, "citation_ok": False, "latency_s": 3.0}
    s = summarize(rows)
    assert s["answer_supported"] == 1.0 and s["citation_accuracy"] == 1.0
    assert s["unsupported_claims"] == 1 and s["abstain_answer"] == 0.0 and s["answer_s_median"] == 2.5
    assert any("model answered an unanswerable question" in w["why"] for w in weak_cases(rows))


# ---------- the runner ----------
def test_evaluate_runs_end_to_end_without_a_model():
    result = evaluate(QUESTIONS, FakeEmbedder(), ocr=False)
    meta = result["meta"]
    assert meta["n_questions"] == len(QUESTIONS) and meta["answers"] == "retrieval only"
    assert "data/sample_scanned.pdf" in meta["skipped_documents"]
    assert meta["n_run"] == len([q for q in QUESTIONS if q["case_type"] != "ocr"])
    assert result["overall"]["recall@5"] >= 0.9  # a bag-of-words fake embedder + BM25 still finds these
    assert set(result["by_language"]) == {"en", "hi", "hinglish", "mixed"}
    assert set(result["by_case_type"]) == {"text", "table"}
    assert [r["id"] for r in result["rows"]] == [q["id"] for q in QUESTIONS if q["case_type"] != "ocr"]
    assert "OVERALL" in format_report(result)
    json.dumps(result, ensure_ascii=False)  # the result must be serialisable as is


def test_library_mode_searches_all_documents_at_once():
    result = evaluate(QUESTIONS, FakeEmbedder(), library=True, ocr=False)
    assert result["meta"]["mode"].startswith("library")
    docs_seen = {top[0] for r in result["rows"] for top in r["top"]}
    assert len(docs_seen) > 1  # passages from different PDFs compete in one ranking


def test_evaluate_with_a_model_that_always_abstains():
    result = evaluate(QUESTIONS, FakeEmbedder(), generate=lambda prompt: "NOT_FOUND", ocr=False)
    o = result["overall"]
    assert result["meta"]["answers"] == "model run"
    assert o["answer_found"] == 0.0 and o["unsupported_claims"] == 0 and o["abstain_answer"] == 1.0


def test_evaluate_with_a_model_that_cites_the_first_passage():
    def generate(prompt):
        return "Answer: see the passage.\nSources: [C1]" if "Excerpts:" in prompt else "rewrite"

    result = evaluate(QUESTIONS, FakeEmbedder(), generate=generate, ocr=False)
    answered = [r for r in result["rows"] if r["answer"]["found"]]
    assert answered and all(r["answer"]["cited"] for r in answered)
    assert result["overall"]["citation_accuracy"] is not None


# ---------- saved results + the parser benchmark ----------
@pytest.mark.parametrize("path", sorted((ROOT / "eval" / "results").glob("*.json")) or [None])
def test_saved_result_files_have_the_shape_the_benchmark_page_reads(path):
    if path is None or path.name == "parsers.json":
        pytest.skip("no saved retrieval result to check here")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert {"meta", "overall", "by_language", "by_case_type", "weak_cases", "rows"} <= set(data)
    assert data["meta"]["embedding_model"] and data["overall"]["recall@5"] is not None


def test_parser_benchmark_reports_pypdf_and_never_fakes_docling():
    bench = run_benchmark(QUESTIONS)
    names = {r["parser"]: r for r in bench["results"]}
    assert names["pypdf"]["installed"] and names["pypdf"]["evidence_kept"] >= 0.85
    if not names["docling"]["installed"]:
        assert "not installed" in names["docling"]["note"] and "evidence_kept" not in names["docling"]


def test_evidence_kept_counts_only_the_gold_pages():
    qs = [
        {"doc": "d", "answerable": True, "expected_pages": [2], "keywords": ["alpha"]},
        {"doc": "d", "answerable": True, "expected_pages": [1], "keywords": ["beta"]},
        {"doc": "d", "answerable": False, "expected_pages": [], "keywords": []},
    ]
    assert evidence_kept(qs, "d", {1: "Alpha is here", 2: "nothing"}) == {"questions": 2, "kept": 0}
    assert evidence_kept(qs, "d", {1: "beta", 2: "ALPHA"}) == {"questions": 2, "kept": 2}


# ---------- the benchmark page ----------
def _page():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(ROOT / "pages" / "1_Benchmark.py"), default_timeout=30)


def test_benchmark_page_renders_saved_results():
    at = _page().run()
    assert not at.exception
    assert at.title[0].value == "Benchmark"
    labels = [m.label for m in at.metric]
    assert "Recall@5" in labels and "Off-topic abstained" in labels
    assert at.selectbox[0].options  # one entry per saved result set


def test_benchmark_page_says_when_there_are_no_results(monkeypatch, tmp_path):
    monkeypatch.setattr("pathlib.Path.glob", lambda self, pattern: iter([]))
    at = _page().run()
    assert not at.exception and any("No benchmark results yet" in i.value for i in at.info)
