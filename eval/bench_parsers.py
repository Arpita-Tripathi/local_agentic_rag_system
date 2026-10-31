"""Benchmark the PDF parsers on the sample set: install size, runtime and extraction quality.

    python -m eval.bench_parsers                      # pypdf; Docling too if it is installed
    pip install docling==2.145.0 && python -m eval.bench_parsers --out eval/results/parsers.json

Quality = the share of answerable golden questions whose evidence value (a keyword) appears on the gold
page in the parser's own text. It measures whether the parser kept the facts, not how pretty the layout is.
Docling is optional and heavy, so a missing Docling is reported as "not installed", never faked.
"""

import argparse
import json
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Callable, Dict, List, Optional

from eval.run_eval import ROOT, _norm, load_questions
from rag.ingest.extract import extract_pages
from rag.ingest.layout import extract_layout_blocks, layout_available


def install_size_mb(dist_name: str) -> Optional[float]:
    """Size of the package's own files (not its dependencies). None if it is not installed."""
    try:
        files = metadata.distribution(dist_name).files or []
    except metadata.PackageNotFoundError:
        return None
    total = 0
    for f in files:
        try:
            total += f.locate().stat().st_size
        except OSError:
            continue
    return round(total / (1 << 20), 2)


def _pypdf_pages(path: str) -> Dict[int, str]:
    return {p["page"]: p["text"] for p in extract_pages(path)}


def _docling_pages(path: str) -> Dict[int, str]:
    blocks, _ = extract_layout_blocks(path)
    pages: Dict[int, str] = {}
    for b in blocks:
        pages[b["page"]] = pages.get(b["page"], "") + " " + b["text"]
    return pages


PARSERS: Dict[str, Dict] = {
    "pypdf": {"dist": "pypdf", "run": _pypdf_pages, "available": lambda: True},
    "docling": {"dist": "docling", "run": _docling_pages, "available": layout_available},
}


def evidence_kept(questions: List[Dict], doc: str, pages: Dict[int, str]) -> Dict[str, int]:
    answerable = [q for q in questions if q["doc"] == doc and q["answerable"]]
    kept = 0
    for q in answerable:
        text = _norm(" ".join(pages.get(p, "") for p in q["expected_pages"]))
        kept += any(_norm(k) in text for k in q["keywords"])
    return {"questions": len(answerable), "kept": kept}


def bench_parser(
    name: str, questions: List[Dict], docs: List[str], run: Callable[[str], Dict[int, str]]
) -> Dict:
    per_doc, total_q, total_kept, total_s = {}, 0, 0, 0.0
    for doc in docs:
        start = time.perf_counter()
        try:
            pages = run(str(ROOT / doc))
            error = None
        except Exception as e:  # a parser failing on a file is a result, not a crash
            pages, error = {}, str(e)[:160]
        seconds = time.perf_counter() - start
        ev = evidence_kept(questions, doc, pages)
        per_doc[doc] = {
            "seconds": round(seconds, 3),
            "chars": sum(len(t) for t in pages.values()),
            "pages_with_text": sum(bool(t.strip()) for t in pages.values()),
            "evidence_kept": f"{ev['kept']}/{ev['questions']}",
            "error": error,
        }
        total_q, total_kept, total_s = total_q + ev["questions"], total_kept + ev["kept"], total_s + seconds
    return {
        "parser": name,
        "installed": True,
        "install_mb": install_size_mb(PARSERS[name]["dist"]),
        "total_seconds": round(total_s, 3),
        "evidence_kept": round(total_kept / total_q, 4) if total_q else None,
        "per_doc": per_doc,
    }


def run_benchmark(questions: List[Dict]) -> Dict:
    docs = sorted({q["doc"] for q in questions})
    results = []
    for name, spec in PARSERS.items():
        if not spec["available"]():
            results.append(
                {
                    "parser": name,
                    "installed": False,
                    "install_mb": None,
                    "note": f"not installed (pip install {spec['dist']})",
                }
            )
            continue
        results.append(bench_parser(name, questions, docs, spec["run"]))
    return {"documents": docs, "results": results}


def format_report(bench: Dict) -> str:
    lines = [f"{'parser':<9}{'install MB':>11}{'seconds':>9}{'evidence kept':>15}"]
    for r in bench["results"]:
        if not r["installed"]:
            lines.append(f"{r['parser']:<9}{'-':>11}{'-':>9}{'-':>15}   {r['note']}")
            continue
        kept = "-" if r["evidence_kept"] is None else f"{r['evidence_kept'] * 100:.1f}%"
        lines.append(f"{r['parser']:<9}{r['install_mb']:>11}{r['total_seconds']:>9}{kept:>15}")
    for r in bench["results"]:
        if r["installed"]:
            lines.append(f"\n{r['parser']} per document:")
            for doc, d in r["per_doc"].items():
                lines.append(
                    f"  {Path(doc).name:<26} {d['seconds']:>7}s  {d['chars']:>6} chars  "
                    f"{d['evidence_kept']:>5} evidence  {d['error'] or ''}"
                )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    bench = run_benchmark(load_questions())
    print(format_report(bench))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(bench, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
