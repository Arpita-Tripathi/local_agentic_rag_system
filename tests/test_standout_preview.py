"""The page preview is an image drawn on the server, because browsers block embedded PDF data: URLs."""
import io
import pathlib

import pytest
from PIL import Image
from pypdf import PdfWriter
from streamlit.testing.v1 import AppTest

from rag.preview import MAX_SIDE_PX, render_page_png
from tests.test_member_a import _blank_pdf, _text_pdf
from tests.test_member_a_phase4 import _doc

APP = str(pathlib.Path(__file__).resolve().parent.parent / "app.py")
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _bytes(path):
    return pathlib.Path(path).read_bytes()


# ---------- rendering ----------
def test_a_page_is_rendered_as_a_png(tmp_path):
    png = render_page_png(_bytes(_text_pdf(tmp_path / "t.pdf", ["Hello preview world, page one."])), 1)
    assert png.startswith(PNG_MAGIC)
    width, height = Image.open(io.BytesIO(png)).size
    assert width > 300 and height > 300


def test_each_page_of_a_multi_page_pdf_renders(tmp_path):
    data = _bytes(_blank_pdf(tmp_path / "b.pdf", pages=3))
    assert all(render_page_png(data, n).startswith(PNG_MAGIC) for n in (1, 2, 3))


@pytest.mark.parametrize("page", [0, 4, -1])
def test_a_page_outside_the_document_gives_none(tmp_path, page):
    assert render_page_png(_bytes(_blank_pdf(tmp_path / "b.pdf", pages=3)), page) is None


def test_garbage_bytes_give_none_not_a_crash():
    assert render_page_png(b"not a pdf at all", 1) is None
    assert render_page_png(b"", 1) is None


def test_a_huge_page_is_scaled_down(tmp_path):
    writer = PdfWriter()
    writer.add_blank_page(width=14400, height=14400)
    path = tmp_path / "huge.pdf"
    with open(path, "wb") as f:
        writer.write(f)
    width, height = Image.open(io.BytesIO(render_page_png(_bytes(path), 1))).size
    assert max(width, height) <= MAX_SIDE_PX + 1


# ---------- app ----------
@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr("rag.providers.get_provider", lambda mode=None, cloud_consent=False: None)
    monkeypatch.setattr("rag.runlog.purge_doc", lambda *a, **k: 0)
    monkeypatch.setattr("rag.summaries.clear_doc_summaries", lambda *a, **k: 0)
    monkeypatch.setattr("rag.translate.clear_doc_translations", lambda *a, **k: 0)

    def make(doc):
        at = AppTest.from_file(APP, default_timeout=60)
        at.session_state["doc"], at.session_state["history"] = doc, []
        return at.run()

    return make


def _real_doc(data, n_pages):
    doc = _doc()
    doc["bytes"], doc["n_pages"] = data, n_pages
    return doc


def test_the_preview_shows_the_page_as_an_image(app, tmp_path):
    at = app(_real_doc(_bytes(_blank_pdf(tmp_path / "b.pdf", pages=2)), 2))
    assert not at.exception and len(at.image) == 1
    assert not any("page image is not available" in i.value for i in at.info)


def test_the_app_no_longer_embeds_a_pdf_data_url():
    source = (pathlib.Path(APP)).read_text(encoding="utf-8")
    assert "data:application/pdf" not in source and "st.iframe" not in source


def test_an_unrenderable_document_falls_back_to_text(app):
    at = app(_doc())  # its bytes are not a real PDF
    assert not at.exception and len(at.image) == 0
    assert any("page image is not available" in i.value for i in at.info)
    assert any(e.label.startswith("Text extracted from page 1") and e.proto.expanded for e in at.expander)


def test_next_and_previous_move_the_image_between_pages(app, tmp_path):
    at = app(_real_doc(_bytes(_blank_pdf(tmp_path / "b.pdf", pages=3)), 3))
    next(b for b in at.button if b.label == "Next").click().run()
    assert at.session_state["preview_page"] == 2 and len(at.image) == 1
    assert set(at.session_state["page_images"]) == {("a" * 64, 1), ("a" * 64, 2)}


def test_rendered_pages_are_capped_in_memory(app, tmp_path):
    at = app(_real_doc(_bytes(_blank_pdf(tmp_path / "b.pdf", pages=15)), 15))
    for _ in range(14):
        next(b for b in at.button if b.label == "Next").click().run()
    assert len(at.session_state["page_images"]) <= 12


def test_removing_the_document_drops_its_page_images(app, tmp_path):
    at = app(_real_doc(_bytes(_blank_pdf(tmp_path / "b.pdf", pages=2)), 2))
    assert at.session_state["page_images"]
    next(b for b in at.button if b.label == "Remove document and index").click().run()
    assert "page_images" not in at.session_state
