"""Benchmark runner: retrieval quality always, answer quality when a provider is given.

    python -m eval.run_eval --check                      # prove every expected page holds its evidence
    python -m eval.run_eval                              # retrieval metrics, default embedding model
    python -m eval.run_eval --embed-model paraphrase-multilingual-MiniLM-L12-v2 \
        --out eval/results/multilingual.json
    python -m eval.run_eval --library                    # one index over every sample PDF (harder)
    python -m eval.run_eval --answers local              # also run the local model: support, citations

Retrieval metrics need no language model. Recall@k counts a question as found when a top-k chunk comes
from a gold page of the right document. Recall is measured with no score threshold, so it shows what the
ranker can do; the "kept" numbers apply the app's threshold, so they show what a user would see.
"""

import argparse
import datetime
import json
import logging
import os
import platform
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from rag.config import CONFIG
from rag.ingest import ingest_pdf
from rag.ingest.ocr import ocr_available
from rag.ingest.validate import PDFValidationError
from rag.pipeline import ask
from rag.retrieval.index import Embedder, build_or_load, get_embedder
from rag.retrieval.search import make_search

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "eval" / "questions.json"
RESULTS = ROOT / "eval" / "results"
KS = (1, 3, 5)


def load_questions(path: Path = QUESTIONS) -> List[Dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["questions"]


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _rate(flags: List[bool]) -> Optional[float]:
    return round(sum(flags) / len(flags), 4) if flags else None


def _median(values: List[float]) -> Optional[float]:
    return round(statistics.median(values), 3) if values else None


def _p95(values: List[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 3)


# ---------- documents ----------
def ingest_docs(questions: List[Dict], ocr: Optional[bool] = None) -> Tuple[Dict[str, Dict], Dict[str, str]]:
    """Ingest each distinct PDF once. Returns ({doc: ingest result}, {doc: reason it was skipped})."""
    if ocr is None:
        ocr = ocr_available()[0]
    docs, skipped = {}, {}
    for rel in sorted({q["doc"] for q in questions}):
        try:
            result = ingest_pdf(str(ROOT / rel), layout="off", ocr=ocr)
        except PDFValidationError as e:
            skipped[rel] = str(e)
            continue
        for c in result["chunks"]:
            c["doc"] = rel
        docs[rel] = result
    return docs, skipped


def check_ground_truth(questions: List[Dict], docs: Dict[str, Dict], skipped: Dict[str, str]) -> List[str]:
    """Problems in the golden set. Empty list means every expected page really holds its evidence."""
    problems = []
    seen = set()
    for q in questions:
        if q["id"] in seen:
            problems.append(f"{q['id']}: duplicate id")
        seen.add(q["id"])
        if q["doc"] in skipped:
            continue
        doc = docs.get(q["doc"])
        if doc is None:
            problems.append(f"{q['id']}: document {q['doc']} could not be read")
            continue
        if not q["answerable"]:
            if q["expected_pages"] or q["keywords"]:
                problems.append(f"{q['id']}: unanswerable question must not list pages or keywords")
            if q.get("unanswerable_kind") not in ("off_topic", "near_miss"):
                problems.append(f"{q['id']}: unanswerable_kind must be off_topic or near_miss")
            continue
        if not q["expected_pages"] or not q["keywords"]:
            problems.append(f"{q['id']}: answerable question needs expected_pages and keywords")
            continue
        if any(not 1 <= p <= doc["n_pages"] for p in q["expected_pages"]):
            problems.append(f"{q['id']}: expected page outside 1..{doc['n_pages']}")
            continue
        page_text = _norm(" ".join(c["text"] for c in doc["chunks"] if c["page"] in q["expected_pages"]))
        if not any(_norm(k) in page_text for k in q["keywords"]):
            problems.append(f"{q['id']}: none of {q['keywords']} found on page(s) {q['expected_pages']}")
    return problems


# ---------- evaluation ----------
def _gold(chunk: Dict, q: Dict) -> bool:
    return chunk.get("doc") == q["doc"] and chunk["page"] in q["expected_pages"]


def _groups(docs: Dict[str, Dict], library: bool) -> List[Tuple[str, List[Dict]]]:
    """(index key, chunks) groups. One per document, or a single library with unique ids."""
    if not library:
        return [(d["doc_hash"], d["chunks"]) for d in docs.values()]
    merged = []
    for rel, d in docs.items():
        stem = Path(rel).stem
        merged += [{**c, "id": f"{stem}:{c['id']}"} for c in d["chunks"]]
    return [("library", merged)]


def _row(
    q: Dict, full: Callable, prod: Callable, by_id: Dict[str, Dict], generate: Optional[Callable]
) -> Dict:
    start = time.perf_counter()
    ranked = full(q["question"], max(KS))
    latency_ms = (time.perf_counter() - start) * 1000
    kept = prod(q["question"], CONFIG.top_k)
    gold_rank = next((i for i, c in enumerate(ranked, start=1) if _gold(c, q)), None)
    row = {
        "id": q["id"],
        "language": q["language"],
        "case_type": q["case_type"],
        "answerable": q["answerable"],
        "kind": q.get("unanswerable_kind"),
        "question": q["question"],
        "top": [[c["doc"], c["page"], round(c["score"], 3)] for c in ranked],
        "gold_rank": gold_rank,
        "hit": {k: gold_rank is not None and gold_rank <= k for k in KS},
        "kept_gold": any(_gold(c, q) for c in kept),
        "abstained": not kept,
        "top_score": round(ranked[0]["score"], 3) if ranked else 0.0,
        "latency_ms": round(latency_ms, 1),
    }
    if generate is not None:
        res = ask(q["question"], prod, generate)
        cited = [by_id[c["id"]] for c in res["citations"] if c["id"] in by_id]
        row["answer"] = {
            "found": res["found"],
            "text": res["answer"],
            "reason": res["reason"],
            "retries": res["retries"],
            "latency_s": round(res["latency_s"], 2),
            "cited": [[c["doc"], c["page"]] for c in cited],
            "citation_ok": bool(cited) and all(_gold(c, q) for c in cited),
            "supported": any(_norm(k) in _norm(res["answer"]) for k in q["keywords"]),
        }
    return row


def evaluate(
    questions: List[Dict],
    embedder: Embedder,
    library: bool = False,
    generate: Optional[Callable[[str], str]] = None,
    ocr: Optional[bool] = None,
) -> Dict:
    docs, skipped = ingest_docs(questions, ocr)
    runnable = [q for q in questions if q["doc"] in docs]
    rows, build_s = [], 0.0
    with tempfile.TemporaryDirectory() as cache:
        for key, chunks in _groups(docs, library):
            start = time.perf_counter()
            index = build_or_load(chunks, key, embedder, cache)
            build_s += time.perf_counter() - start
            full = make_search(index, embedder, min_score=0.0)
            prod = make_search(index, embedder)
            by_id = {c["id"]: c for c in chunks}
            docs_here = {c["doc"] for c in chunks}
            rows += [_row(q, full, prod, by_id, generate) for q in runnable if q["doc"] in docs_here]
    order = {q["id"]: i for i, q in enumerate(questions)}
    rows.sort(key=lambda r: order[r["id"]])
    return {
        "meta": {
            "date": datetime.date.today().isoformat(),
            "embedding_model": getattr(embedder, "name", "unknown"),
            "mode": "library (one index over all PDFs)" if library else "per document",
            "answers": "model run" if generate is not None else "retrieval only",
            "top_k": CONFIG.top_k,
            "score_threshold": CONFIG.score_threshold,
            "chunk_size": CONFIG.chunk_size,
            "chunk_overlap": CONFIG.chunk_overlap,
            "parser": CONFIG.parser_version,
            "ocr_available": bool(ocr) if ocr is not None else ocr_available()[0],
            "documents": sorted(docs),
            "skipped_documents": skipped,
            "n_questions": len(questions),
            "n_run": len(rows),
            "index_build_s": round(build_s, 2),
            "hardware": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "cpus": os.cpu_count(),
                "python": platform.python_version(),
            },
        },
        "overall": summarize(rows),
        "by_language": {k: summarize(v) for k, v in _split(rows, "language").items()},
        "by_case_type": {k: summarize(v) for k, v in _split(rows, "case_type").items()},
        "weak_cases": weak_cases(rows),
        "rows": rows,
    }


def _split(rows: List[Dict], key: str) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for r in rows:
        out.setdefault(r[key], []).append(r)
    return dict(sorted(out.items()))


def summarize(rows: List[Dict]) -> Dict:
    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    off = [r for r in unans if r["kind"] == "off_topic"]
    near = [r for r in unans if r["kind"] == "near_miss"]
    out: Dict = {
        "n_answerable": len(ans),
        "n_unanswerable": len(unans),
        **{f"recall@{k}": _rate([r["hit"][k] for r in ans]) for k in KS},
        "mrr@5": round(sum(1 / r["gold_rank"] for r in ans if r["gold_rank"]) / len(ans), 4) if ans else None,
        "recall@5_kept": _rate([r["kept_gold"] for r in ans]),
        "abstain_retrieval": _rate([r["abstained"] for r in unans]),
        "abstain_off_topic": _rate([r["abstained"] for r in off]),
        "abstain_near_miss": _rate([r["abstained"] for r in near]),
        "top_score_answerable_median": _median([r["top_score"] for r in ans]),
        "top_score_unanswerable_max": max((r["top_score"] for r in unans), default=None),
        "search_ms_median": _median([r["latency_ms"] for r in rows]),
        "search_ms_p95": _p95([r["latency_ms"] for r in rows]),
    }
    if rows and "answer" in rows[0]:
        found = [r for r in ans if r["answer"]["found"]]
        out.update(
            {
                "answer_found": _rate([r["answer"]["found"] for r in ans]),
                "answer_supported": _rate([r["answer"]["supported"] for r in found]),
                "citation_accuracy": _rate([r["answer"]["citation_ok"] for r in found]),
                "unsupported_claims": sum(r["answer"]["found"] for r in unans),
                "abstain_answer": _rate([not r["answer"]["found"] for r in unans]),
                "answer_s_median": _median([r["answer"]["latency_s"] for r in rows]),
                "answer_s_p95": _p95([r["answer"]["latency_s"] for r in rows]),
            }
        )
    return out


def weak_cases(rows: List[Dict]) -> List[Dict]:
    weak = []
    for r in rows:
        if r["answerable"] and not r["hit"][5]:
            why = "gold page not in the top 5"
        elif r["answerable"] and not r["kept_gold"]:
            why = f"gold page dropped by the score threshold (top score {r['top_score']})"
        elif r["answerable"] and r["gold_rank"] != 1:
            why = f"gold page ranked {r['gold_rank']}, not 1"
        elif not r["answerable"] and "answer" in r and r["answer"]["found"]:
            why = f"model answered an unanswerable question ({r['kind']})"
        elif r["kind"] == "off_topic" and not r["abstained"]:
            # Retrieval can only be blamed for off-topic questions. A near-miss shares vocabulary with the
            # document, so keeping passages is expected and only the model can abstain.
            why = f"off-topic question but retrieval kept passages (top score {r['top_score']})"
        elif "answer" in r and r["answerable"] and not r["answer"]["supported"]:
            why = "answer does not contain the expected value" if r["answer"]["found"] else "model abstained"
        else:
            continue
        weak.append(
            {
                "id": r["id"],
                "language": r["language"],
                "case_type": r["case_type"],
                "question": r["question"],
                "why": why,
            }
        )
    return weak


# ---------- reporting ----------
def _pct(v: Optional[float]) -> str:
    return "  -  " if v is None else f"{v * 100:5.1f}"


def format_report(result: Dict) -> str:
    m = result["meta"]
    lines = [
        f"embedding model : {m['embedding_model']}   ({m['mode']}, {m['answers']})",
        f"questions run   : {m['n_run']} of {m['n_questions']}   "
        f"threshold {m['score_threshold']}   top_k {m['top_k']}",
    ]
    for doc, why in m["skipped_documents"].items():
        lines.append(f"skipped         : {doc} ({why[:90]})")
    has_answers = "answer_supported" in result["overall"]
    head = (
        f"{'split':<12}{'n ans':>6}{'n un':>6}{'R@1':>7}{'R@3':>7}{'R@5':>7}"
        f"{'R@5kept':>9}{'abst off':>9}{'abst near':>10}"
    )
    if has_answers:
        head += f"{'support':>9}{'cite':>7}{'unsupp':>8}"
    lines += ["", head]
    blocks = [("OVERALL", result["overall"])]
    blocks += [(f"lang:{k}", v) for k, v in result["by_language"].items()]
    blocks += [(f"case:{k}", v) for k, v in result["by_case_type"].items()]
    for name, s in blocks:
        line = (
            f"{name:<12}{s['n_answerable']:>6}{s['n_unanswerable']:>6}{_pct(s['recall@1']):>7}"
            f"{_pct(s['recall@3']):>7}{_pct(s['recall@5']):>7}{_pct(s['recall@5_kept']):>9}"
            f"{_pct(s['abstain_off_topic']):>9}{_pct(s['abstain_near_miss']):>10}"
        )
        if has_answers:
            line += (
                f"{_pct(s.get('answer_supported')):>9}{_pct(s.get('citation_accuracy')):>7}"
                f"{s.get('unsupported_claims', 0):>8}"
            )
        lines.append(line)
    o = result["overall"]
    lines += [
        "",
        f"search latency  : median {o['search_ms_median']} ms, p95 {o['search_ms_p95']} ms",
        f"top score       : answerable median {o['top_score_answerable_median']}, "
        f"unanswerable max {o['top_score_unanswerable_max']}  (threshold {m['score_threshold']})",
        f"weak cases      : {len(result['weak_cases'])}  "
        f"(ids: {', '.join(w['id'] for w in result['weak_cases'][:12])}"
        f"{' ...' if len(result['weak_cases']) > 12 else ''})",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only validate the golden set against the PDFs")
    ap.add_argument("--embed-model", default=None, help="sentence-transformers model name")
    ap.add_argument("--library", action="store_true", help="one index over all PDFs")
    ap.add_argument("--answers", choices=["none", "local"], default="none", help="also run the answer model")
    ap.add_argument("--questions", default=str(QUESTIONS))
    ap.add_argument("--out", default=None, help="write the full JSON result here")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.ERROR)

    questions = load_questions(Path(args.questions))
    if args.check:
        docs, skipped = ingest_docs(questions)
        problems = check_ground_truth(questions, docs, skipped)
        for doc, why in skipped.items():
            print(f"skipped {doc}: {why[:100]}")
        for p in problems:
            print("PROBLEM", p)
        print(f"{len(questions)} questions checked, {len(problems)} problem(s)")
        return 1 if problems else 0

    generate = None
    if args.answers == "local":
        from rag.providers import get_provider

        generate = get_provider("local")
    result = evaluate(questions, get_embedder(args.embed_model), library=args.library, generate=generate)
    print(format_report(result))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
