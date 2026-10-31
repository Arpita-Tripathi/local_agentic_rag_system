"""Benchmark page: what was measured, on what, with what result, and where it fails.

Reads eval/results/*.json (written by `python -m eval.run_eval --out ...` and
`python -m eval.bench_parsers --out eval/results/parsers.json`). Nothing here is computed on page load,
and nothing is claimed that a result file does not contain.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS = Path(__file__).resolve().parent.parent / "eval" / "results"

st.set_page_config(page_title="Benchmark", layout="wide")
st.title("Benchmark")
st.caption("Measured on a small, hand-written sample set. These numbers describe that set, not every PDF.")


def pct(v):
    return "-" if v is None else f"{v * 100:.1f}%"


def load():
    out = {}
    for p in sorted(RESULTS.glob("*.json")):
        if p.name == "parsers.json":
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if {"meta", "overall", "rows"} <= set(data):
                out[p.stem] = data
        except (OSError, ValueError):
            continue
    return out


results = load()
if not results:
    st.info("No benchmark results yet. Run `python -m eval.run_eval --out eval/results/minilm.json` first.")
    st.stop()

name = st.selectbox(
    "Result set",
    list(results),
    format_func=lambda n: (
        f"{n}  ({results[n]['meta']['embedding_model']}, {results[n]['meta']['mode']}, "
        f"{results[n]['meta']['answers']})"
    ),
)
res = results[name]
meta, overall = res["meta"], res["overall"]

st.subheader("Retrieval quality")
cols = st.columns(5)
cols[0].metric("Recall@1", pct(overall["recall@1"]))
cols[1].metric("Recall@3", pct(overall["recall@3"]))
cols[2].metric("Recall@5", pct(overall["recall@5"]))
cols[3].metric("Off-topic abstained", pct(overall["abstain_off_topic"]))
cols[4].metric("Search latency (median)", f"{overall['search_ms_median']} ms")
st.caption(
    f"Recall@k: a passage from the gold page is among the top k. Computed without the score threshold. "
    f"After the app's threshold ({meta['score_threshold']}) Recall@5 is {pct(overall['recall@5_kept'])}. "
    "Near-miss questions (on topic, answer absent) cannot be caught by retrieval; only the model can abstain."
)

if "answer_supported" in overall:
    st.subheader("Answer quality (model run)")
    c = st.columns(4)
    c[0].metric("Answer supported", pct(overall["answer_supported"]))
    c[1].metric("Citation accuracy", pct(overall["citation_accuracy"]))
    c[2].metric("Unsupported claims", overall["unsupported_claims"])
    c[3].metric("Answer time (median)", f"{overall['answer_s_median']} s")
else:
    st.info(
        "This result set is retrieval only. Answer support and citation accuracy need a model run: "
        "`python -m eval.run_eval --answers local`."
    )

COLUMNS = {
    "n_answerable": "answerable",
    "n_unanswerable": "unanswerable",
    "recall@1": "R@1",
    "recall@3": "R@3",
    "recall@5": "R@5",
    "recall@5_kept": "R@5 after threshold",
    "abstain_off_topic": "abstain off-topic",
    "abstain_near_miss": "abstain near-miss",
    "search_ms_median": "search ms",
}


def table(split):
    rows = {
        k: {COLUMNS[c]: (pct(v[c]) if c.startswith(("recall", "abstain")) else v[c]) for c in COLUMNS}
        for k, v in split.items()
    }
    return pd.DataFrame.from_dict(rows, orient="index")


left, right = st.columns(2)
with left:
    st.subheader("By language")
    st.dataframe(table(res["by_language"]), use_container_width=True)
with right:
    st.subheader("By case type")
    st.dataframe(table(res["by_case_type"]), use_container_width=True)

st.subheader("Weak cases")
if res["weak_cases"]:
    st.dataframe(pd.DataFrame(res["weak_cases"]), use_container_width=True, hide_index=True)
else:
    st.success("No weak cases in this result set.")

st.subheader("Test mix and hardware")
skipped = meta["skipped_documents"]
st.write(
    f"{meta['n_run']} of {meta['n_questions']} questions run "
    f"({overall['n_answerable']} answerable, {overall['n_unanswerable']} unanswerable) over "
    f"{len(meta['documents'])} PDF(s). Index build: {meta['index_build_s']} s. "
    f"Chunks of {meta['chunk_size']} "
    f"characters, overlap {meta['chunk_overlap']}, parser {meta['parser']}. Date: {meta['date']}."
)
hw = meta["hardware"]
st.write(
    f"Hardware: {hw['processor'] or 'unknown CPU'}, {hw['cpus']} logical CPUs, {hw['platform']}, "
    f"Python {hw['python']}."
)
if skipped:
    st.warning("Not run: " + "; ".join(f"{Path(d).name} ({why[:80]}...)" for d, why in skipped.items()))

parsers = RESULTS / "parsers.json"
if parsers.exists():
    st.subheader("PDF parsers")
    bench = json.loads(parsers.read_text(encoding="utf-8"))
    rows = [
        {
            "parser": r["parser"],
            "install MB": r.get("install_mb"),
            "seconds": r.get("total_seconds"),
            "evidence kept": pct(r.get("evidence_kept"))
            if r["installed"]
            else r.get("note", "not installed"),
        }
        for r in bench["results"]
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "Evidence kept: share of answerable questions whose answer value appears on the gold page in the "
        "parser's own text."
    )

st.divider()
st.caption(
    "Known limits: a small synthetic sample set; answers are not graded by a human; scanned-page "
    "results depend on Tesseract being installed."
)
