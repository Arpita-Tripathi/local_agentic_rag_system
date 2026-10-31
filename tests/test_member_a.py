import os

import pytest
from pypdf import PdfWriter

from rag.answer import NOT_FOUND, answer, build_prompt, parse_response
from rag.citations import resolve
from rag.config import CONFIG, Config
from rag.ingest import ingest_pdf
from rag.ingest.chunk import build_chunks
from rag.ingest.extract import extract_pages
from rag.ingest.validate import PDFValidationError, save_upload, validate_pdf
from rag.pipeline import ask
from rag.providers import (
    CloudProvider,
    ConsentRequired,
    FakeProvider,
    LocalProvider,
    ProviderError,
    get_provider,
)

CHUNKS = [
    {"id": "c0000", "page": 2, "text": "LSTM predicts the next token.", "parser": "pypdf"},
    {"id": "c0001", "page": 5, "text": "Training used 10 epochs.", "parser": "pypdf"},
]


# ---------- helpers ----------
def _blank_pdf(path, pages=1):
    w = PdfWriter()
    for _ in range(pages):
        w.add_blank_page(width=200, height=200)
    with open(path, "wb") as f:
        w.write(f)
    return str(path)


def _text_pdf(path, lines):
    """A minimal valid one-page PDF with real extractable text."""
    stream = "BT /F1 12 Tf 14 TL 40 700 Td " + " ".join(f"({ln}) Tj T*" for ln in lines) + " ET"
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    with open(path, "wb") as f:
        f.write(out)
    return str(path)


# ---------- validate / extract ----------
def test_validate_rejects_non_pdf(tmp_path):
    p = tmp_path / "x.pdf"
    p.write_bytes(b"hello")
    with pytest.raises(PDFValidationError, match="not a PDF"):
        validate_pdf(str(p))


def test_validate_rejects_empty_file(tmp_path):
    p = tmp_path / "e.pdf"
    p.write_bytes(b"")
    with pytest.raises(PDFValidationError, match="empty"):
        validate_pdf(str(p))


def test_validate_rejects_too_many_pages(tmp_path):
    with pytest.raises(PDFValidationError, match="limit"):
        validate_pdf(_blank_pdf(tmp_path / "a.pdf", pages=3), max_pages=2)


def test_validate_rejects_oversize(tmp_path):
    p = tmp_path / "big.pdf"
    p.write_bytes(b"%PDF-1.4\n" + b"0" * (2 << 20))
    with pytest.raises(PDFValidationError, match="limit"):
        validate_pdf(str(p), max_mb=1)


def test_validate_rejects_encrypted(tmp_path):
    w = PdfWriter()
    w.add_blank_page(width=100, height=100)
    w.encrypt("secret")
    p = tmp_path / "enc.pdf"
    with open(p, "wb") as f:
        w.write(f)
    with pytest.raises(PDFValidationError, match="password"):
        validate_pdf(str(p))


def test_validate_rejects_corrupt(tmp_path):
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"%PDF-1.4\nthis is garbage")
    with pytest.raises(PDFValidationError):
        validate_pdf(str(p))


def test_validate_accepts_good_pdf(tmp_path):
    assert validate_pdf(_blank_pdf(tmp_path / "ok.pdf", pages=2)) == 2


def test_save_upload_writes_private_temp_file_and_checks_size():
    path = save_upload(b"%PDF-1.4 data")
    try:
        assert os.path.isfile(path) and path.endswith(".pdf")
        assert open(path, "rb").read() == b"%PDF-1.4 data"
    finally:
        os.unlink(path)
    with pytest.raises(PDFValidationError, match="limit"):
        save_upload(b"x" * (2 << 20), max_mb=1)


def test_blank_page_is_flagged_image_only(tmp_path):
    pages = extract_pages(_blank_pdf(tmp_path / "b.pdf"))
    assert pages[0]["text"] == "" and pages[0]["image_only"] is True and pages[0]["garbled"] is False
    assert pages[0]["quality"]["level"] == "none"


def test_extract_reads_real_text(tmp_path):
    pages = extract_pages(_text_pdf(tmp_path / "t.pdf", ["Hello retrieval world, this is page one."]))
    assert pages[0]["image_only"] is False and "retrieval" in pages[0]["text"]


def test_extract_unreadable_file_raises_validation_error(tmp_path):
    p = tmp_path / "junk.pdf"
    p.write_bytes(b"not a pdf at all")
    with pytest.raises(PDFValidationError):
        extract_pages(str(p))


def test_ingest_pdf_end_to_end(tmp_path):
    out = ingest_pdf(_text_pdf(tmp_path / "t.pdf", ["The model was trained for ten epochs on one GPU."]))
    assert out["n_pages"] == 1 and out["image_only_pages"] == []
    assert out["chunks"] and out["chunks"][0]["doc_hash"] == out["doc_hash"]
    assert out["chunks"][0]["parser"].startswith("pypdf")


def test_ingest_pdf_with_no_text_raises_explicit_error(tmp_path):
    with pytest.raises(PDFValidationError, match="No text could be extracted"):
        ingest_pdf(_blank_pdf(tmp_path / "b.pdf", pages=2), ocr=False)


# ---------- chunking ----------
def test_chunks_carry_provenance_and_skip_image_only_pages():
    pages = [
        {"page": 1, "text": "1 Introduction\nThis is a sentence. Another one.", "image_only": False},
        {"page": 2, "text": "", "image_only": True},
        {"page": 3, "text": "More text on page three.", "image_only": False},
    ]
    chunks = build_chunks(pages, doc_hash="abc")
    assert [c["page"] for c in chunks] == [1, 3]
    assert len({c["id"] for c in chunks}) == len(chunks)
    assert chunks[0]["heading"] == "1 Introduction"
    assert all(c["doc_hash"] == "abc" and c["parser"] for c in chunks)
    assert all(c["type"] == "text" for c in chunks)


def test_chunk_ids_are_stable_across_runs():
    pages = [{"page": 1, "text": "One. Two. Three.", "image_only": False}]
    assert build_chunks(pages, "h") == build_chunks(pages, "h")


def test_long_text_is_split_within_size():
    pages = [{"page": 1, "text": "Word word word. " * 200, "image_only": False}]
    chunks = build_chunks(pages, "h", chunk_size=300, overlap=30)
    assert len(chunks) > 1
    assert all(len(c["text"]) <= 300 for c in chunks)


def test_single_giant_sentence_is_hard_split_within_size():
    pages = [{"page": 1, "text": "x" * 1000, "image_only": False}]
    chunks = build_chunks(pages, "h", chunk_size=300, overlap=30)
    assert len(chunks) > 1 and all(len(c["text"]) <= 300 for c in chunks)


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        build_chunks([], "h", chunk_size=100, overlap=100)


def test_heading_applies_per_section_not_per_page():
    pages = [{"page": 1, "text": "1 Intro\nAlpha text here.\n2 Method\nBeta text here.", "image_only": False}]
    chunks = build_chunks(pages, "h")
    assert [(c["heading"], "Alpha" in c["text"]) for c in chunks] == [("1 Intro", True), ("2 Method", False)]


def test_heading_carries_over_to_next_page():
    pages = [
        {"page": 1, "text": "3. Method\nWe did a thing.", "image_only": False},
        {"page": 2, "text": "And then another thing.", "image_only": False},
    ]
    assert [c["heading"] for c in build_chunks(pages, "h")] == ["3. Method", "3. Method"]


def test_numbered_sentence_is_not_a_heading():
    pages = [{"page": 1, "text": "10 epochs were used for training the model", "image_only": False}]
    assert build_chunks(pages, "h")[0]["heading"] == ""


def test_hindi_danda_splits_sentences():
    text = ("यह पहला वाक्य है। " * 40).strip()
    chunks = build_chunks([{"page": 1, "text": text, "image_only": False}], "h", chunk_size=120, overlap=0)
    assert len(chunks) > 1 and all(c["text"].endswith("।") for c in chunks)


# ---------- answer / citations ----------
def test_prompt_labels_and_pages():
    prompt = build_prompt("q?", CHUNKS)
    assert "[C1 | page 2]" in prompt and "[C2 | page 5]" in prompt


def test_prompt_tells_model_excerpts_are_untrusted():
    assert "untrusted" in build_prompt("q?", CHUNKS)


def test_parse_maps_labels_to_chunk_ids_and_keeps_unknown_raw():
    out = parse_response("Answer: It predicts tokens.\nSources: [C1], [C9]", CHUNKS)
    assert out["found"] and out["source_ids"] == ["c0000", "C9"]


def test_parse_handles_sources_on_same_line():
    out = parse_response("Answer: It predicts tokens. Sources: [C1]", CHUNKS)
    assert out["answer"] == "It predicts tokens." and out["source_ids"] == ["c0000"]


def test_parse_uses_last_sources_marker():
    out = parse_response("Answer: Sources: matter. Details here.\nSources: [C2]", CHUNKS)
    assert out["source_ids"] == ["c0001"] and out["answer"].startswith("Sources: matter")


def test_not_found_abstains():
    assert parse_response("NOT_FOUND", CHUNKS)["answer"] == NOT_FOUND
    assert parse_response("Answer: NOT_FOUND", CHUNKS)["found"] is False


def test_real_answer_mentioning_not_found_is_kept():
    out = parse_response("Answer: The log prints NOT_FOUND on error.\nSources: [C1]", CHUNKS)
    assert out["found"] and out["source_ids"] == ["c0000"]


def test_no_chunks_skips_model_call():
    fake = FakeProvider([])
    assert answer("q", [], fake)["found"] is False
    assert fake.prompts == []


def test_resolve_drops_unknown_ids_and_uses_stored_page():
    cites = resolve(["c0001", "C9", "zzz"], CHUNKS)
    assert [(c["id"], c["page"]) for c in cites] == [("c0001", 5)]


def test_resolve_snippet_comes_from_stored_text():
    cite = resolve(["c0000"], CHUNKS, snippet_chars=4)[0]
    assert cite["snippet"] == "LSTM"


# ---------- pipeline ----------
def test_ask_happy_path_one_call():
    gen = FakeProvider(["Answer: It predicts tokens.\nSources: [C1]"])
    out = ask("What does LSTM do?", lambda q, k: CHUNKS, gen)
    assert out["found"] and out["reason"] == "ok" and out["retries"] == 0 and len(gen.prompts) == 1
    assert out["citations"][0]["page"] == 2
    assert out["provider"] == "fake" and out["trace"][0]["outcome"] == "ok"


def test_ask_answer_without_valid_citation_is_not_trusted():
    gen = FakeProvider(["Answer: made up\nSources: [C9]", "rewritten", "Answer: made up\nSources: [C9]"])
    out = ask("q", lambda q, k: CHUNKS, gen)
    assert out["found"] is False and out["answer"] == NOT_FOUND and out["reason"] == "no_valid_citation"
    assert out["citations"] == []


def test_ask_retries_at_most_once():
    calls = []

    def search(q, k):
        calls.append(q)
        return []

    gen = FakeProvider(["a rewritten passage"])
    out = ask("q", search, gen, max_retries=1)
    assert out["found"] is False and len(calls) == 2 and out["retries"] == 1
    assert [t["attempt"] for t in out["trace"]] == [0, 1]
    assert calls[1] == "a rewritten passage"


def test_ask_model_not_found_triggers_one_rewrite_then_succeeds():
    gen = FakeProvider(["NOT_FOUND", "a passage about LSTM", "Answer: Yes.\nSources: [C1]"])
    out = ask("q", lambda q, k: CHUNKS, gen)
    assert out["found"] and out["retries"] == 1 and len(gen.prompts) == 3


def test_ask_drops_chunks_below_threshold_and_skips_model():
    weak = [dict(CHUNKS[0], score=0.01)]
    gen = FakeProvider(["rewrite"])
    out = ask("q", lambda q, k: weak, gen, max_retries=1, min_score=0.15)
    assert out["found"] is False and out["reason"] == "low_score"
    assert len(gen.prompts) == 1  # only the rewrite call; no answer call on weak evidence


def test_ask_keeps_strong_chunks_and_filters_weak_ones():
    mixed = [dict(CHUNKS[0], score=0.9), dict(CHUNKS[1], score=0.01)]
    gen = FakeProvider(["Answer: ok\nSources: [C1]"])
    out = ask("q", lambda q, k: mixed, gen, min_score=0.15)
    assert out["found"] and [c["id"] for c in out["chunks"]] == ["c0000"]
    assert "Training used 10 epochs" not in gen.prompts[0]


def test_ask_keeps_retrieved_chunks_on_failure_for_the_ui():
    gen = FakeProvider(["NOT_FOUND", "rewrite", "NOT_FOUND"])
    out = ask("q", lambda q, k: CHUNKS, gen)
    assert out["found"] is False and out["reason"] == "not_found" and len(out["chunks"]) == 2


def test_ask_failure_keeps_earlier_evidence_when_rewrite_finds_nothing():
    def search(q, k):
        return CHUNKS if q == "q" else []

    out = ask("q", search, FakeProvider(["NOT_FOUND", "unrelated rewrite"]))
    assert out["found"] is False and [c["id"] for c in out["chunks"]] == ["c0000", "c0001"]


def test_ask_empty_or_oversized_rewrite_is_sanitized():
    queries = []

    def search(q, k):
        queries.append(q)
        return []

    ask("original?", search, FakeProvider(["   "]), max_retries=1)
    assert queries[1] == "original?"
    queries.clear()
    ask("original?", search, FakeProvider(["w " * 1000]), max_retries=1)
    assert len(queries[1]) <= CONFIG.max_query_chars


# ---------- config ----------
def test_mode_defaults_to_local_and_reads_env(monkeypatch):
    monkeypatch.delenv("RAG_MODE", raising=False)
    assert Config().mode == "local"
    monkeypatch.setenv("RAG_MODE", "cloud")
    assert Config().mode == "cloud"


def test_invalid_mode_env_is_rejected(monkeypatch):
    monkeypatch.setenv("RAG_MODE", "bogus")
    with pytest.raises(ValueError, match="RAG_MODE"):
        Config()


def test_parser_version_records_pypdf_version():
    assert CONFIG.parser_version.startswith("pypdf@")


# ---------- providers ----------
def test_cloud_requires_consent():
    with pytest.raises(ConsentRequired):
        get_provider("cloud", cloud_consent=False)


def test_unknown_mode_is_rejected():
    with pytest.raises(ProviderError, match="Unknown mode"):
        get_provider("teleport")


def test_cloud_needs_key_and_model(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="OPENROUTER_API_KEY"):
        get_provider("cloud", cloud_consent=True)


def test_local_missing_model_file_gives_actionable_message(monkeypatch, tmp_path):
    monkeypatch.setattr("rag.providers.CONFIG", Config(model_dir=str(tmp_path), litert_model="nope"))
    with pytest.raises(ProviderError, match="Model file not found"):
        LocalProvider()


class _Resp:
    def __init__(self, status=200, body=None, text=""):
        self.status_code, self._body, self.text = status, body, text

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


@pytest.fixture
def cloud(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("rag.providers.CONFIG", Config(cloud_model="vendor/model"))
    return CloudProvider()


def _patch_post(monkeypatch, resp=None, exc=None):
    import requests

    sent = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.update(url=url, headers=headers, json=json, timeout=timeout)
        if exc:
            raise exc
        return resp

    monkeypatch.setattr(requests, "post", fake_post)
    return sent


def test_cloud_success_tracks_usage_and_caps_tokens(cloud, monkeypatch):
    body = {
        "choices": [{"message": {"content": " hi "}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }
    sent = _patch_post(monkeypatch, _Resp(200, body))
    assert cloud("prompt") == "hi"
    assert cloud.usage == {"calls": 1, "prompt_tokens": 10, "completion_tokens": 3}
    assert sent["json"]["max_tokens"] == CONFIG.max_output_tokens and sent["timeout"]
    assert cloud.name == "cloud:vendor/model"


def test_cloud_429_message(cloud, monkeypatch):
    _patch_post(monkeypatch, _Resp(429))
    with pytest.raises(ProviderError, match="429"):
        cloud("p")


def test_cloud_bad_key_message(cloud, monkeypatch):
    _patch_post(monkeypatch, _Resp(401))
    with pytest.raises(ProviderError, match="API key"):
        cloud("p")


def test_cloud_network_error_message(cloud, monkeypatch):
    import requests

    _patch_post(monkeypatch, exc=requests.ConnectionError("down"))
    with pytest.raises(ProviderError, match="Network error"):
        cloud("p")


def test_cloud_malformed_body_is_a_provider_error(cloud, monkeypatch):
    _patch_post(monkeypatch, _Resp(200, {"choices": []}))
    with pytest.raises(ProviderError, match="unexpected response"):
        cloud("p")
    _patch_post(monkeypatch, _Resp(200, None))
    with pytest.raises(ProviderError, match="unexpected response"):
        cloud("p")


def test_cloud_server_error_is_reported_without_crashing(cloud, monkeypatch):
    _patch_post(monkeypatch, _Resp(500, text="boom"))
    with pytest.raises(ProviderError, match="500"):
        cloud("p")
