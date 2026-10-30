import pytest
from pypdf import PdfWriter

from rag.answer import NOT_FOUND, answer, build_prompt, parse_response
from rag.citations import resolve
from rag.ingest.chunk import build_chunks
from rag.ingest.extract import extract_pages
from rag.ingest.validate import PDFValidationError, validate_pdf
from rag.pipeline import ask
from rag.providers import ConsentRequired, FakeProvider, get_provider

CHUNKS = [
    {"id": "c0000", "page": 2, "text": "LSTM predicts the next token.", "parser": "pypdf"},
    {"id": "c0001", "page": 5, "text": "Training used 10 epochs.", "parser": "pypdf"},
]


# ---------- validate / extract ----------
def _blank_pdf(path, pages=1):
    w = PdfWriter()
    for _ in range(pages):
        w.add_blank_page(width=200, height=200)
    with open(path, "wb") as f:
        w.write(f)
    return str(path)


def test_validate_rejects_non_pdf(tmp_path):
    p = tmp_path / "x.pdf"
    p.write_bytes(b"hello")
    with pytest.raises(PDFValidationError, match="not a PDF"):
        validate_pdf(str(p))


def test_validate_rejects_too_many_pages(tmp_path):
    with pytest.raises(PDFValidationError, match="limit"):
        validate_pdf(_blank_pdf(tmp_path / "a.pdf", pages=3), max_pages=2)


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


def test_blank_page_is_flagged_image_only(tmp_path):
    pages = extract_pages(_blank_pdf(tmp_path / "b.pdf"))
    assert pages == [{"page": 1, "text": "", "image_only": True}]


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


def test_long_text_is_split_within_size():
    pages = [{"page": 1, "text": "Word word word. " * 200, "image_only": False}]
    chunks = build_chunks(pages, "h", chunk_size=300, overlap=30)
    assert len(chunks) > 1
    assert all(len(c["text"]) <= 340 for c in chunks)


# ---------- answer / citations ----------
def test_prompt_labels_and_pages():
    prompt = build_prompt("q?", CHUNKS)
    assert "[C1 | page 2]" in prompt and "[C2 | page 5]" in prompt


def test_parse_maps_labels_to_chunk_ids_and_keeps_unknown_raw():
    out = parse_response("Answer: It predicts tokens.\nSources: [C1], [C9]", CHUNKS)
    assert out["found"] and out["source_ids"] == ["c0000", "C9"]


def test_not_found_abstains():
    assert parse_response("NOT_FOUND", CHUNKS)["answer"] == NOT_FOUND


def test_no_chunks_skips_model_call():
    fake = FakeProvider([])
    assert answer("q", [], fake)["found"] is False
    assert fake.prompts == []


def test_resolve_drops_unknown_ids_and_uses_stored_page():
    cites = resolve(["c0001", "C9", "zzz"], CHUNKS)
    assert [(c["id"], c["page"]) for c in cites] == [("c0001", 5)]


# ---------- pipeline ----------
def test_ask_happy_path_one_call():
    gen = FakeProvider(["Answer: It predicts tokens.\nSources: [C1]"])
    out = ask("What does LSTM do?", lambda q, k: CHUNKS, gen)
    assert out["found"] and out["retries"] == 0 and len(gen.prompts) == 1
    assert out["citations"][0]["page"] == 2


def test_ask_answer_without_valid_citation_is_not_trusted():
    gen = FakeProvider(["Answer: made up\nSources: [C9]"] * 2 + ["rewritten"])
    gen.replies = ["Answer: made up\nSources: [C9]", "rewritten", "Answer: made up\nSources: [C9]"]
    out = ask("q", lambda q, k: CHUNKS, gen)
    assert out["found"] is False and out["answer"] == NOT_FOUND


def test_ask_retries_at_most_once():
    calls = []

    def search(q, k):
        calls.append(q)
        return []

    gen = FakeProvider(["a rewritten passage"])
    out = ask("q", search, gen, max_retries=1)
    assert out["found"] is False and len(calls) == 2 and out["retries"] == 1


# ---------- providers ----------
def test_cloud_requires_consent():
    with pytest.raises(ConsentRequired):
        get_provider("cloud", cloud_consent=False)
