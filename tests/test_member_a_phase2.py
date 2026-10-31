"""Phase 2: OCR, garbled-text detection, Docling layout fallback, table chunks, cached summaries."""
import json
import sys
import types

import pytest
from PIL import Image

from rag.config import Config
from rag.ingest import ingest_pdf
from rag.ingest import layout as layout_mod
from rag.ingest.chunk import build_chunks
from rag.ingest.extract import extract_pages, looks_garbled
from rag.ingest.layout import LayoutUnavailable, extract_layout_blocks, looks_table_heavy, merge_layout
from rag.ingest.ocr import OCRUnavailable, ocr_available, ocr_pages
from rag.ingest.validate import PDFValidationError
from rag.providers import FakeProvider
from rag.summaries import summarize_chunk
from tests.test_member_a import _blank_pdf, _text_pdf

LONG = "This page was scanned and then read by optical character recognition software."


def _scanned_pdf(path, pages=1):
    imgs = [Image.new("RGB", (60, 60), "white") for _ in range(pages)]
    imgs[0].save(path, "PDF", save_all=True, append_images=imgs[1:])
    return str(path)


def _fake_tesseract(monkeypatch, text=LONG, langs=("eng", "hin"), version="5.3.0", fail_version=False):
    mod = types.ModuleType("pytesseract")
    mod.pytesseract = types.SimpleNamespace(tesseract_cmd="")
    calls = []

    def get_version():
        if fail_version:
            raise RuntimeError("tesseract is not installed")
        return version

    mod.get_tesseract_version = get_version
    mod.get_languages = lambda config="": list(langs)
    mod.image_to_string = lambda image, lang="": calls.append(lang) or text
    monkeypatch.setitem(sys.modules, "pytesseract", mod)
    return calls


# ---------- pypdf log noise ----------
def test_pypdf_font_map_warnings_are_not_shown():
    import logging

    assert logging.getLogger("pypdf").level == logging.ERROR
    assert logging.getLogger("pypdf._cmap").getEffectiveLevel() == logging.ERROR


# ---------- garbled detection ----------
def test_looks_garbled():
    assert looks_garbled("(cid:12)(cid:45)(cid:7) " * 5)
    assert looks_garbled(" ab")
    assert looks_garbled("���� text")
    assert not looks_garbled("यह एक सामान्य हिंदी वाक्य है। This is normal text.")
    assert not looks_garbled("")


def test_garbled_page_is_discarded_and_marked_for_ocr(monkeypatch):
    class Page:
        def extract_text(self):
            return "(cid:1)(cid:2)(cid:3)" * 20

    class Reader:
        def __init__(self, path):
            self.pages = [Page()]

    monkeypatch.setattr("rag.ingest.extract.PdfReader", Reader)
    page = extract_pages("x.pdf")[0]
    assert page == {"page": 1, "text": "", "image_only": True, "garbled": True}


# ---------- OCR ----------
def test_ocr_fills_image_only_pages(tmp_path, monkeypatch):
    calls = _fake_tesseract(monkeypatch)
    path = _scanned_pdf(tmp_path / "s.pdf")
    pages, warnings = ocr_pages(path, extract_pages(path))
    assert warnings == []
    assert pages[0]["text"] == LONG and pages[0]["image_only"] is False
    assert pages[0]["type"] == "ocr" and pages[0]["parser"] == "tesseract@5.3.0"
    assert calls == ["hin+eng"]


def test_ocr_skips_pages_that_already_have_text(tmp_path, monkeypatch):
    calls = _fake_tesseract(monkeypatch)
    path = _text_pdf(tmp_path / "t.pdf", ["Plenty of real text already lives on this page."])
    pages = extract_pages(path)
    out, warnings = ocr_pages(path, pages)
    assert out == pages and warnings == [] and calls == []


def test_ocr_empty_result_leaves_page_image_only_with_warning(tmp_path, monkeypatch):
    _fake_tesseract(monkeypatch, text="")
    path = _scanned_pdf(tmp_path / "s.pdf")
    pages, warnings = ocr_pages(path, extract_pages(path))
    assert pages[0]["image_only"] is True and "Page 1" in warnings[0]


def test_ocr_page_without_images_is_reported(tmp_path, monkeypatch):
    _fake_tesseract(monkeypatch)
    path = _blank_pdf(tmp_path / "b.pdf")
    pages, warnings = ocr_pages(path, extract_pages(path))
    assert pages[0]["image_only"] is True and warnings


def test_ocr_skips_oversized_image(tmp_path, monkeypatch):
    calls = _fake_tesseract(monkeypatch)
    monkeypatch.setattr("rag.ingest.ocr.CONFIG", Config(ocr_max_pixels=10))
    path = _scanned_pdf(tmp_path / "s.pdf")
    _, warnings = ocr_pages(path, extract_pages(path))
    assert calls == [] and any("too large" in w for w in warnings)


def test_ocr_unavailable_when_tesseract_missing(monkeypatch):
    _fake_tesseract(monkeypatch, fail_version=True)
    ok, msg = ocr_available()
    assert not ok and "Tesseract is not installed" in msg


def test_ocr_unavailable_when_hindi_data_missing(monkeypatch):
    _fake_tesseract(monkeypatch, langs=("eng",))
    ok, msg = ocr_available()
    assert not ok and "hin" in msg


def test_ocr_unavailable_when_pytesseract_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "pytesseract", None)
    ok, msg = ocr_available()
    assert not ok and "pytesseract" in msg


def test_ocr_pages_raises_up_front_when_unavailable(tmp_path, monkeypatch):
    _fake_tesseract(monkeypatch, fail_version=True)
    path = _scanned_pdf(tmp_path / "s.pdf")
    with pytest.raises(OCRUnavailable):
        ocr_pages(path, extract_pages(path))


# ---------- layout ----------
def test_table_heavy_heuristic():
    table = "\n".join(f"Item{i} 10 20 30 40 5.5" for i in range(20))
    assert looks_table_heavy([{"text": table, "image_only": False}])
    prose = "\n".join("This is an ordinary sentence of prose text." for _ in range(20))
    assert not looks_table_heavy([{"text": prose, "image_only": False}])
    assert not looks_table_heavy([{"text": table, "image_only": True}])


class TextItem:
    def __init__(self, text, page):
        self.text, self.prov = text, [types.SimpleNamespace(page_no=page)]


class SectionHeaderItem(TextItem):
    pass


class TableItem:
    def __init__(self, page):
        self.prov = [types.SimpleNamespace(page_no=page)]

    def export_to_markdown(self, doc=None):
        return "| a | b |\n|---|---|\n| 1 | 2 |"


def _fake_docling(monkeypatch, items):
    doc = types.SimpleNamespace(iterate_items=lambda: [(i, 0) for i in items])
    conv = types.SimpleNamespace(document=doc)
    mod = types.ModuleType("docling.document_converter")
    mod.DocumentConverter = lambda: types.SimpleNamespace(convert=lambda path: conv)
    monkeypatch.setitem(sys.modules, "docling", types.ModuleType("docling"))
    monkeypatch.setitem(sys.modules, "docling.document_converter", mod)
    monkeypatch.setattr(layout_mod, "_docling_version", lambda: "9.9.9")


def test_layout_unavailable_message(monkeypatch):
    monkeypatch.setattr(layout_mod, "_docling_version", lambda: None)
    with pytest.raises(LayoutUnavailable, match="pip install docling"):
        extract_layout_blocks("x.pdf")


def test_layout_blocks_split_text_tables_and_headings(monkeypatch):
    _fake_docling(monkeypatch, [
        SectionHeaderItem("2. Results", 1), TextItem("We measured things.", 1), TableItem(1),
        TextItem("Page two text.", 2),
    ])
    blocks, tag = extract_layout_blocks("x.pdf")
    assert tag == "docling@9.9.9"
    assert [(b["page"], b["type"]) for b in blocks] == [(1, "text"), (1, "table"), (2, "text")]
    assert blocks[0]["heading"] == "2. Results" and blocks[1]["text"].startswith("| a | b |")
    assert blocks[2]["heading"] == "2. Results" and all(b["parser"] == tag for b in blocks)


def test_layout_parse_failure_is_reported(monkeypatch):
    _fake_docling(monkeypatch, [])
    def explode():
        raise RuntimeError("boom")

    sys.modules["docling.document_converter"].DocumentConverter = explode
    with pytest.raises(LayoutUnavailable, match="boom"):
        extract_layout_blocks("x.pdf")


def test_merge_layout_keeps_pypdf_pages_docling_skipped():
    pages = [{"page": 1, "text": "a", "image_only": False}, {"page": 2, "text": "b", "image_only": False}]
    blocks = [{"page": 1, "type": "table", "text": "t", "image_only": False}]
    merged = merge_layout(pages, blocks)
    assert merged == [blocks[0], pages[1]]


# ---------- chunking of typed pages ----------
def test_table_chunks_repeat_header_and_respect_size():
    rows = "\n".join(f"| row{i} | {i * 11} |" for i in range(40))
    md = f"| name | value |\n|---|---|\n{rows}"
    page = {"page": 3, "type": "table", "text": md, "image_only": False, "parser": "docling@1"}
    chunks = build_chunks([page], "h", chunk_size=200, overlap=20)
    assert len(chunks) > 1
    assert all(c["type"] == "table" and c["page"] == 3 and c["parser"] == "docling@1" for c in chunks)
    assert all(len(c["text"]) <= 200 and c["text"].startswith("| name | value |") for c in chunks)
    assert sum(c["text"].count("| row") for c in chunks) == 40  # no row lost


def test_oversized_table_row_is_capped():
    md = "| a |\n|---|\n| " + "x" * 500 + " |"
    page = {"page": 1, "type": "table", "text": md, "image_only": False}
    chunks = build_chunks([page], "h", chunk_size=100, overlap=10)
    assert chunks and all(len(c["text"]) <= 100 for c in chunks)


def test_ocr_page_chunks_carry_type_and_parser():
    page = {"page": 4, "text": LONG, "image_only": False, "type": "ocr", "parser": "tesseract@5.3.0"}
    chunk = build_chunks([page], "h")[0]
    assert chunk["type"] == "ocr" and chunk["parser"] == "tesseract@5.3.0" and chunk["page"] == 4


def test_explicit_heading_from_parser_is_used():
    page = {"page": 1, "text": "Body text here.", "image_only": False, "heading": "2. Results"}
    assert build_chunks([page], "h")[0]["heading"] == "2. Results"


# ---------- ingest orchestration ----------
def test_ingest_ocr_path_records_parser(tmp_path, monkeypatch):
    _fake_tesseract(monkeypatch)
    out = ingest_pdf(_scanned_pdf(tmp_path / "s.pdf"), layout="off")
    assert out["chunks"][0]["type"] == "ocr" and out["parsers"] == ["tesseract@5.3.0"]
    assert out["image_only_pages"] == [] and out["warnings"] == []


def test_ingest_without_tesseract_raises_error_that_says_what_to_install(tmp_path, monkeypatch):
    _fake_tesseract(monkeypatch, fail_version=True)
    with pytest.raises(PDFValidationError, match="Tesseract is not installed"):
        ingest_pdf(_scanned_pdf(tmp_path / "s.pdf"), layout="off")


def test_ingest_mixed_pdf_keeps_text_pages_when_ocr_missing(tmp_path, monkeypatch):
    _fake_tesseract(monkeypatch, fail_version=True)
    text_pdf = _text_pdf(tmp_path / "t.pdf", ["Real extractable text lives on this first page."])
    out = ingest_pdf(text_pdf, layout="off")
    assert out["chunks"] and out["image_only_pages"] == []

    monkeypatch.setattr("rag.ingest.extract_pages", lambda p: [
        {"page": 1, "text": "Real extractable text lives on this first page.", "image_only": False,
         "garbled": False},
        {"page": 2, "text": "", "image_only": True, "garbled": False},
    ])
    out = ingest_pdf(text_pdf, layout="off")
    assert out["image_only_pages"] == [2] and "need OCR" in out["warnings"][0]


def test_ingest_ocr_disabled(tmp_path):
    with pytest.raises(PDFValidationError, match="OCR is turned off"):
        ingest_pdf(_scanned_pdf(tmp_path / "s.pdf"), layout="off", ocr=False)


def test_ingest_layout_on_without_docling_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(layout_mod, "_docling_version", lambda: None)
    path = _text_pdf(tmp_path / "t.pdf", ["Some ordinary text that is long enough to count."])
    with pytest.raises(PDFValidationError, match="Docling is not installed"):
        ingest_pdf(path, layout="on")


def test_ingest_auto_table_heavy_without_docling_warns_but_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(layout_mod, "_docling_version", lambda: None)
    monkeypatch.setattr("rag.ingest.looks_table_heavy", lambda pages: True)
    path = _text_pdf(tmp_path / "t.pdf", ["Some ordinary text that is long enough to count."])
    out = ingest_pdf(path, layout="auto")
    assert out["chunks"] and any("Docling" in w for w in out["warnings"])


def test_ingest_layout_on_uses_docling_blocks(tmp_path, monkeypatch):
    _fake_docling(monkeypatch, [
        SectionHeaderItem("1. Data", 1), TextItem("Docling saw this text.", 1), TableItem(1),
    ])
    path = _text_pdf(tmp_path / "t.pdf", ["Some ordinary text that is long enough to count."])
    out = ingest_pdf(path, layout="on")
    assert {c["type"] for c in out["chunks"]} == {"text", "table"}
    assert out["parsers"] == ["docling@9.9.9"]


def test_ingest_rejects_unknown_layout_mode(tmp_path):
    with pytest.raises(ValueError, match="layout"):
        ingest_pdf(_blank_pdf(tmp_path / "b.pdf"), layout="maybe")


# ---------- cached summaries ----------
TABLE_MD = "| a | b |\n|---|---|\n| 1 | 2 |"
CHUNK = {"id": "c0001", "doc_hash": "h", "page": 2, "type": "table", "text": TABLE_MD}


def test_summary_is_generated_once_then_cached(tmp_path):
    gen = FakeProvider(["A table of a and b."])
    first = summarize_chunk(CHUNK, gen, cache_dir=str(tmp_path))
    second = summarize_chunk(CHUNK, gen, cache_dir=str(tmp_path))
    assert first == {"summary": "A table of a and b.", "cached": False}
    assert second == {"summary": "A table of a and b.", "cached": True}
    assert len(gen.prompts) == 1 and "untrusted" in gen.prompts[0] and "table" in gen.prompts[0]


def test_summary_cache_is_keyed_on_content_and_provider(tmp_path):
    summarize_chunk(CHUNK, FakeProvider(["one"]), cache_dir=str(tmp_path))
    edited = {**CHUNK, "text": TABLE_MD + "\n| 3 | 4 |"}
    changed = summarize_chunk(edited, FakeProvider(["two"]), str(tmp_path))
    assert changed == {"summary": "two", "cached": False}

    class Other(FakeProvider):
        name = "other"

    assert summarize_chunk(CHUNK, Other(["three"]), cache_dir=str(tmp_path))["cached"] is False


def test_corrupt_summary_cache_is_regenerated(tmp_path):
    summarize_chunk(CHUNK, FakeProvider(["good"]), cache_dir=str(tmp_path))
    for f in (tmp_path / "summaries").iterdir():
        f.write_text("{not json")
    out = summarize_chunk(CHUNK, FakeProvider(["fresh"]), cache_dir=str(tmp_path))
    assert out == {"summary": "fresh", "cached": False}
    assert json.loads(next((tmp_path / "summaries").iterdir()).read_text())["summary"] == "fresh"


def test_empty_summary_is_not_cached(tmp_path):
    out = summarize_chunk(CHUNK, FakeProvider(["  "]), cache_dir=str(tmp_path))
    assert out == {"summary": "", "cached": False}
    assert not (tmp_path / "summaries").exists()
