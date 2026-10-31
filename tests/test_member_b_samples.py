"""Phase 2 gate (Member B): the mixed sample set. One test per file, each against its expected outcome."""

import json
import pathlib

import pytest
from pypdf import PdfReader

from eval import pdfgen
from eval.make_samples import OUTCOMES
from rag.ingest import ingest_pdf
from rag.ingest.ocr import OCRUnavailable, ocr_available
from rag.ingest.validate import PDFValidationError

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
SAMPLES = sorted(OUTCOMES)

has_ocr = ocr_available()[0]
needs_ocr = pytest.mark.skipif(not has_ocr, reason="Tesseract with Hindi+English data is not installed")


def _path(name: str) -> str:
    return str(DATA / name)


# ---------- the set itself ----------
def test_every_sample_has_a_file_and_an_expected_outcome():
    on_disk = {p.name for p in DATA.glob("sample_*.pdf")}
    assert on_disk == set(SAMPLES)
    saved = json.loads((DATA / "expected_outcomes.json").read_text(encoding="utf-8"))
    assert saved == OUTCOMES  # regenerate with: python -m eval.make_samples
    assert all(o["expected"] for o in OUTCOMES.values())


def test_the_set_covers_every_case_the_plan_asks_for():
    kinds = " ".join(o["kind"] for o in OUTCOMES.values())
    for needed in ("text", "scanned", "table", "corrupt", "empty"):
        assert needed in kinds


def test_pdf_builder_is_deterministic():
    pages = [["one line"], [], [["a", "b", "c"]]]
    assert pdfgen.build_pdf(pages) == pdfgen.build_pdf(pages)
    assert pdfgen.build_pdf([["नमस्ते"]], script="devanagari") == pdfgen.build_pdf(
        [["नमस्ते"]], script="devanagari"
    )


# ---------- text PDF ----------
def test_text_pdf_ingests_with_every_page_present():
    r = ingest_pdf(_path("sample_text_en.pdf"), layout="off", ocr=False)
    assert r["n_pages"] == 8 and r["image_only_pages"] == [] and r["warnings"] == []
    assert {c["page"] for c in r["chunks"]} == set(range(1, 9))
    assert all(c["type"] == "text" and c["parser"].startswith("pypdf@") for c in r["chunks"])
    assert any(c["heading"] == "3. Results" for c in r["chunks"])  # headings are detected


# ---------- Hindi text PDF ----------
def test_hindi_pdf_keeps_devanagari_text_intact():
    r = ingest_pdf(_path("sample_text_hi.pdf"), layout="off", ocr=False)
    text = " ".join(c["text"] for c in r["chunks"])
    assert r["n_pages"] == 3 and {c["page"] for c in r["chunks"]} == {1, 2, 3}
    for word in ("रामपुर", "वर्षामापी", "जुलाई", "मोहन लाल", "तालाब"):
        assert word in text, word
    assert "�" not in text and all(any("ऀ" <= ch <= "ॿ" for ch in c["text"]) for c in r["chunks"])


# ---------- table-heavy PDF ----------
def test_table_pdf_keeps_row_labels_and_numbers():
    r = ingest_pdf(_path("sample_tables.pdf"), layout="off", ocr=False)
    by_page = {c["page"]: c["text"] for c in r["chunks"]}
    assert all(x in by_page[1] for x in ("Keyword only", "Hybrid", "88", "0.9"))
    assert all(x in by_page[2] for x in ("Scanned PDF", "19", "Total", "38"))


def test_table_pdf_still_ingests_in_auto_layout_mode_without_docling():
    r = ingest_pdf(_path("sample_tables.pdf"), layout="auto", ocr=False)  # Docling absent or not needed
    assert r["chunks"] and r["image_only_pages"] == []


# ---------- scanned PDF ----------
def test_scanned_pdf_pages_are_pictures_with_no_text_layer():
    reader = PdfReader(_path("sample_scanned.pdf"))
    assert len(reader.pages) == 2
    for page in reader.pages:
        assert len(list(page.images)) == 1 and not (page.extract_text() or "").strip()


def test_scanned_pdf_with_ocr_off_is_rejected_with_a_clear_message():
    with pytest.raises(PDFValidationError, match="No text could be extracted") as e:
        ingest_pdf(_path("sample_scanned.pdf"), layout="off", ocr=False)
    assert "OCR" in str(e.value)


def test_scanned_pdf_without_tesseract_says_what_to_install(monkeypatch):
    def missing():
        raise OCRUnavailable("Tesseract is not installed. Install it or set RAG_TESSERACT_CMD.")

    monkeypatch.setattr("rag.ingest.ocr._tesseract", missing)
    with pytest.raises(PDFValidationError) as e:
        ingest_pdf(_path("sample_scanned.pdf"), layout="off", ocr=True)
    assert "Tesseract" in str(e.value) and "RAG_TESSERACT_CMD" in str(e.value)


@needs_ocr
def test_scanned_pdf_is_read_by_ocr():
    r = ingest_pdf(_path("sample_scanned.pdf"), layout="off", ocr=True)
    assert r["image_only_pages"] == [] and {c["page"] for c in r["chunks"]} == {1, 2}
    assert all(c["type"] == "ocr" and c["parser"] == "tesseract" for c in r["chunks"])
    text = " ".join(c["text"] for c in r["chunks"]).lower()
    assert "station" in text and "lakeside" in text


# ---------- corrupt PDF ----------
def test_corrupt_pdf_is_rejected_not_a_crash():
    data = (DATA / "sample_corrupt.pdf").read_bytes()
    assert data.startswith(b"%PDF-")  # it looks like a PDF, which is what makes it a good test
    with pytest.raises(PDFValidationError, match="corrupt"):
        ingest_pdf(_path("sample_corrupt.pdf"), layout="off", ocr=False)


# ---------- PDF with empty pages ----------
def test_empty_pages_are_reported_and_text_pages_survive():
    r = ingest_pdf(_path("sample_empty_pages.pdf"), layout="off", ocr=False)
    assert r["n_pages"] == 5 and r["image_only_pages"] == [2, 4]
    assert {c["page"] for c in r["chunks"]} == {1, 3, 5}
    assert any("2, 4" in w for w in r["warnings"])


def test_empty_pages_with_ocr_on_do_not_crash():
    # Blank pages hold no image, so there is nothing to OCR; the run must still finish with the text pages.
    r = ingest_pdf(_path("sample_empty_pages.pdf"), layout="off", ocr=has_ocr)
    assert {c["page"] for c in r["chunks"]} == {1, 3, 5}
