"""Streamlit UI: upload a PDF, ask questions, see exactly which pages back each answer.

Run:  streamlit run app.py
Local-only is the default. Nothing leaves this device unless the user switches to cloud mode and
agrees, for this document, to send retrieved excerpts.
"""
import base64
import logging
import os

import streamlit as st

from rag import search_backend
from rag.config import CONFIG
from rag.ingest import ingest_pdf
from rag.ingest.validate import PDFValidationError, save_upload
from rag.pipeline import ask, describe_reason
from rag.providers import ProviderError, get_provider
from rag.runlog import log_run, purge_doc
from rag.summaries import clear_doc_summaries, summarize_chunk
from rag.textsafe import escape_markdown as esc
from rag.translate import clear_doc_translations, target_language, translate_chunk
from rag.ui_text import (
    cross_lingual_note,
    quality_line,
    quality_summary_line,
    support_notice,
    translation_label,
    translation_notice,
)

log = logging.getLogger("app")

LOCAL, CLOUD = "Local only (default)", "Send excerpts to cloud"
MAX_QUESTION_CHARS = 2000
MAX_EMBED_BYTES = 8 << 20  # bigger PDFs skip the embedded viewer; evidence text is still shown

st.set_page_config(page_title="Ask your PDF", page_icon="📄", layout="wide")


# ---------- helpers ----------
def _viewer_src(data: bytes, page: int) -> str:
    """A data: URL the browser's built-in PDF viewer can open at a given page (no JavaScript involved).

    Deliberately not cached: a cache would keep a removed document's bytes in memory.
    """
    return "data:application/pdf;base64," + base64.b64encode(data).decode("ascii") + f"#page={page}"


def _go_to_page(page: int) -> None:
    st.session_state.preview_page = page


def _reset_consent(doc_hash: str) -> None:
    st.session_state[f"consent_{doc_hash}"] = False


def _remove_document() -> None:
    doc = st.session_state.get("doc")
    if doc:
        search_backend.delete_index(doc["doc_hash"])
        clear_doc_summaries(doc["doc_hash"])
        purge_doc(doc["doc_hash"])
        clear_doc_translations(doc["doc_hash"])
        st.session_state.pop(f"consent_{doc['doc_hash']}", None)
    for key in ("doc", "history", "preview_page", "provider_cloud", "failed_upload", "explained",
                "translated"):
        st.session_state.pop(key, None)
    st.session_state.uploader_n = st.session_state.get("uploader_n", 0) + 1


def _use_cloud_instead() -> None:
    st.session_state.mode_choice = CLOUD
    _reset_consent(st.session_state.doc["doc_hash"])


def _load_document(upload) -> str:
    """Load the upload into session state. Returns "" on success, or the user-readable failure message."""
    data = upload.getvalue()
    with st.status("Reading your PDF…", expanded=True) as status:
        bar = st.progress(0.0)

        def progress(msg: str, frac: float) -> None:
            bar.progress(frac, text=msg)

        try:
            path = save_upload(data)
            try:
                result = ingest_pdf(path, progress=progress)
            finally:
                os.unlink(path)
            progress("Building the search index", 0.95)
            search, backend = search_backend.build_search(result["chunks"], result["doc_hash"])
        except PDFValidationError as e:
            status.update(label="Could not read this PDF", state="error")
            return str(e)
        except Exception as e:  # index/model problems must show a message, not a stack trace
            log.exception("document load failed")
            status.update(label="Could not prepare this PDF", state="error")
            return f"Something went wrong while preparing the document: {e}"
        bar.progress(1.0, text="Done")
        status.update(label="Ready", state="complete", expanded=False)
    st.session_state.doc = {**result, "name": upload.name, "bytes": data, "file_id": upload.file_id,
                            "search": search, "backend": backend}
    st.session_state.history = []
    st.session_state.preview_page = 1
    st.session_state.pop("failed_upload", None)
    return ""


def _provider(doc_hash: str):
    """Return (provider, problem). Cloud is only built after consent for THIS document."""
    if st.session_state.get("mode_choice", LOCAL) == LOCAL:
        if "provider_local" not in st.session_state:
            try:
                st.session_state.provider_local = get_provider("local")
            except ProviderError as e:
                return None, str(e)
        return st.session_state.provider_local, None
    if not st.session_state.get(f"consent_{doc_hash}"):
        return None, "Tick the box above to allow sending excerpts to the cloud, or switch back to local."
    if "provider_cloud" not in st.session_state:
        try:
            st.session_state.provider_cloud = get_provider("cloud", cloud_consent=True)
        except ProviderError as e:
            return None, str(e)
    return st.session_state.provider_cloud, None


# ---------- sidebar ----------
def _sidebar(doc) -> None:
    with st.sidebar:
        st.header("Privacy and mode")
        st.radio(
            "Where are answers generated?", [LOCAL, CLOUD], key="mode_choice",
            on_change=_reset_consent if doc else None, args=(doc["doc_hash"],) if doc else (),
        )
        if st.session_state.get("mode_choice", LOCAL) == LOCAL:
            st.success("Local only: nothing leaves this device.")
        else:
            max_calls = 1 + 2 * CONFIG.max_retries
            st.warning(
                f"Cloud mode: each question and the passages retrieved for it are sent to "
                f"{esc(CONFIG.cloud_model or 'the configured model')}. Normally that is 1 call. If the first "
                f"answer is not found, a second search runs, for up to {max_calls} calls: a rewrite call "
                "that sends only your question, then another answer call with new passages. The whole PDF "
                "is never sent, but the provider may keep what it receives. Don't use this for private "
                "documents."
            )
            if doc:
                st.checkbox("I allow sending excerpts of this document to the cloud provider",
                            key=f"consent_{doc['doc_hash']}")
            usage = getattr(st.session_state.get("provider_cloud"), "usage", None)
            if usage:
                st.caption(f"Cloud usage this session: {usage['calls']} call(s), "
                           f"{usage['prompt_tokens']} prompt + {usage['completion_tokens']} reply tokens.")
        if doc:
            st.divider()
            st.subheader("Document")
            st.write(f"**{esc(doc['name'])}**")
            st.caption(f"{doc['n_pages']} pages · {len(doc['chunks'])} passages · "
                       f"parser: {', '.join(doc['parsers'])}")
            st.caption(f"Search: {doc['backend']}")
            summary_line = quality_summary_line(doc.get("quality_summary"))
            if summary_line:
                st.caption(summary_line)
            weak = [q for q in doc.get("page_quality", []) if q["level"] in ("fair", "poor", "none", "ocr")]
            if weak:
                with st.expander(f"Pages to double-check ({len(weak)})"):
                    for q in weak:
                        st.markdown(f"**Page {q['page']}**: " + esc(quality_line(q)))
            meta = getattr(doc["search"], "index_meta", None)
            if meta:
                st.caption(f"Index: {meta['n_chunks']} passages of up to {meta['chunk_size']} characters "
                           f"(overlap {meta['chunk_overlap']}), "
                           f"embedding model {esc(meta['embedding_model'])}")
            st.button("Remove document and index", on_click=_remove_document, use_container_width=True)


# ---------- answer rendering ----------
def _explain_table(i: int, cite: dict, doc: dict, provider) -> None:
    """On-demand table summary: one model call, cached on disk. Never runs unless the user clicks."""
    key = (doc["doc_hash"], cite["id"])
    explained = st.session_state.setdefault("explained", {})
    if key in explained:
        st.markdown(esc(explained[key]))
        return
    if not st.button("Explain this table", key=f"explain-{i}-{cite['id']}", disabled=provider is None,
                     help="Asks the selected model to summarise this table. One call; the result is cached."):
        return
    chunk = next((c for c in doc["chunks"] if c["id"] == cite["id"]), None)
    if chunk is None:
        return
    try:
        with st.spinner("Summarising the table…"):
            out = summarize_chunk(chunk, provider)
    except ProviderError as e:
        st.error(esc(e))
        return
    if out["summary"]:
        explained[key] = out["summary"]
        st.markdown(esc(out["summary"]))


def _show_notice(notice) -> None:
    if notice:
        kind, text = notice
        (st.warning if kind == "warning" else st.caption)(esc(text))


def _translate(i: int, cite: dict, doc: dict, provider) -> None:
    """Show a cited passage and its translation side by side. One model call per passage, then cached."""
    chunk = next((c for c in doc["chunks"] if c["id"] == cite["id"]), None)
    if chunk is None:
        return
    target = target_language(chunk)
    key = (doc["doc_hash"], cite["id"], target)
    done = st.session_state.setdefault("translated", {})
    if key not in done:
        if not st.button(translation_label(target), key=f"translate-{i}-{cite['id']}",
                         disabled=provider is None,
                         help="Shows the passage and its translation side by side. One call, then cached."):
            return
        try:
            with st.spinner("Translating…"):
                result = translate_chunk(chunk, provider, target)
        except ProviderError as e:
            st.error(esc(e))
            return
        if not result["translation"]:
            return
        done[key] = result
    result = done[key]
    left, right = st.columns(2)
    left.caption(f"Original, page {cite['page']}")
    left.markdown(esc(chunk["text"]))
    right.caption(f"{translation_label(target)[len('Translate to '):]} (machine translation)")
    right.markdown(esc(result["translation"]))
    _show_notice(translation_notice(result["check"]))


def _render_turn(i: int, turn: dict, doc: dict, provider=None) -> None:
    res = turn["result"]
    with st.chat_message("user"):
        st.markdown(esc(turn["question"]))
    with st.chat_message("assistant"):
        if res["found"]:
            st.markdown(esc(res["answer"]))
            note = cross_lingual_note(res)
            if note:
                st.caption(esc(note))
            _show_notice(support_notice(res.get("support")))
            st.caption("Sources (click to jump to the page):")
            cols = st.columns(min(len(res["citations"]), 4))
            for j, cite in enumerate(res["citations"]):
                cols[j % len(cols)].button(f"Page {cite['page']}", key=f"cite-{i}-{cite['id']}",
                                           on_click=_go_to_page, args=(cite["page"],))
            for cite in res["citations"]:
                with st.expander(f"Evidence · page {cite['page']} · {cite['parser'] or 'unknown parser'}"):
                    st.markdown(esc(cite["snippet"]))
                    if cite.get("type") == "table":
                        _explain_table(i, cite, doc, provider)
                    _translate(i, cite, doc, provider)
        else:
            st.markdown(esc(res["answer"]))
            why = describe_reason(res["reason"])
            if why:
                st.caption(why)
            if res["chunks"]:
                with st.expander(f"Passages that were searched ({len(res['chunks'])})"):
                    for c in res["chunks"]:
                        score = f" · score {c['score']:.2f}" if "score" in c else ""
                        st.markdown(f"**Page {c['page']}**{score}")
                        st.markdown(esc(c["text"][:400]))
        st.caption(f"{res['provider']} · {res['latency_s']:.1f}s · retries {res['retries']} · "
                   f"parser {', '.join(doc['parsers'])} · index {doc['doc_hash'][:8]} ({doc['backend']})")


def _preview(doc: dict) -> None:
    page = min(max(st.session_state.get("preview_page", 1), 1), doc["n_pages"])
    st.subheader(f"Page preview · page {page} of {doc['n_pages']}")
    nav = st.columns([1, 2, 1])
    nav[0].button("Previous", on_click=_go_to_page, args=(max(page - 1, 1),), disabled=page <= 1)
    nav[2].button("Next", on_click=_go_to_page, args=(min(page + 1, doc["n_pages"]),),
                  disabled=page >= doc["n_pages"])
    page_q = next((q for q in doc.get("page_quality", []) if q["page"] == page), None)
    if page_q:
        st.caption(esc(quality_line(page_q)))
    if len(doc["bytes"]) <= MAX_EMBED_BYTES:
        st.iframe(_viewer_src(doc["bytes"], page), height=640)
    else:
        st.info("This file is too large for the embedded viewer. The extracted text is shown below.")
    with st.expander(f"Text extracted from page {page}", expanded=len(doc["bytes"]) > MAX_EMBED_BYTES):
        texts = [c["text"] for c in doc["chunks"] if c["page"] == page]
        st.markdown(esc("\n\n".join(texts)) if texts else "No text was extracted from this page.")


# ---------- main ----------
def main() -> None:
    st.session_state.setdefault("mode_choice", CLOUD if CONFIG.mode == "cloud" else LOCAL)
    doc = st.session_state.get("doc")
    _sidebar(doc)

    st.title("Ask your PDF")
    st.caption("Answers come only from your document, with the page that backs each one. "
               "If the PDF doesn't say, you'll be told so.")

    upload = st.file_uploader(
        "Upload a PDF", type=["pdf"], key=f"uploader_{st.session_state.get('uploader_n', 0)}",
        help=f"Up to {CONFIG.max_pdf_mb} MB and {CONFIG.max_pages} pages.",
    )
    if upload is not None and (doc is None or doc["file_id"] != upload.file_id):
        failed = st.session_state.get("failed_upload")
        if not failed or failed["file_id"] != upload.file_id:  # never retry a file that already failed
            message = _load_document(upload)
            if message:
                st.session_state.failed_upload = {"file_id": upload.file_id, "message": message}
            doc = st.session_state.get("doc")
    failed = st.session_state.get("failed_upload")
    if upload is not None and failed and failed["file_id"] == upload.file_id:
        st.error(esc(failed["message"]))

    if doc is None:
        st.info("Upload a PDF to begin. Scanned pages need Tesseract (you'll be told if it's missing).")
        return

    for w in doc["warnings"]:
        st.warning(esc(w))
    if doc["image_only_pages"]:
        st.warning("No text could be read from page(s) "
                   + ", ".join(map(str, doc["image_only_pages"][:15])) + ". Questions can't use them.")
    if doc["backend"] == search_backend.STUB_NAME:
        st.info("Search is running in development mode (simple keyword matching). "
                "Answer quality will be limited until the full retrieval module is installed.")

    provider, problem = _provider(doc["doc_hash"])
    if problem:
        st.error(esc(problem))
        if st.session_state.get("mode_choice", LOCAL) == LOCAL:
            st.button("Use the cloud instead (asks for your consent)", on_click=_use_cloud_instead)

    question = st.chat_input("Ask a question about this PDF (English or Hindi)", disabled=provider is None)
    if question:
        question = question.strip()[:MAX_QUESTION_CHARS]
        try:
            with st.spinner("Searching the document and writing an answer…"):
                result = ask(question, doc["search"], provider)
            log_run(question, result, doc["doc_hash"])
            st.session_state.history.append({"question": question, "result": result})
            if result["citations"]:
                st.session_state.preview_page = result["citations"][0]["page"]
        except ProviderError as e:
            st.error(esc(e))
        except Exception as e:
            log.exception("question failed")
            st.error(esc(f"Something went wrong while answering: {e}"))

    chat, preview = st.columns([3, 2], gap="large")
    with chat:
        if not st.session_state.history:
            st.caption("No questions yet. Try asking about a specific fact in the document.")
        for i, turn in enumerate(st.session_state.history):
            _render_turn(i, turn, doc, provider)
    with preview:
        _preview(doc)


main()
