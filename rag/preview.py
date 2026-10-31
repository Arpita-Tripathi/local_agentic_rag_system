"""Render one PDF page to a PNG so the browser can show it.

Browsers refuse to show a PDF embedded as a data: URL, so the preview is drawn on the server instead and sent
as an image. That works in every browser. It uses PDFium through pypdfium2 (BSD-3-Clause / Apache-2.0, no
separate install). PDFium is native code: it only ever sees files that already passed rag.ingest.validate.
"""
import io
import logging
from typing import Optional

log = logging.getLogger(__name__)

MAX_SIDE_PX = 2400  # a huge page must not become a huge image


def render_page_png(data: bytes, page: int, scale: float = 1.6) -> Optional[bytes]:
    """PNG bytes of a 1-based page, or None when the page cannot be drawn (the caller shows text instead)."""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        log.warning("pypdfium2 is not installed, so there is no page image (pip install -r requirements.txt)")
        return None
    pdf = None
    try:
        pdf = pdfium.PdfDocument(data)
        if not 1 <= page <= len(pdf):
            return None
        pdf_page = pdf[page - 1]
        width, height = pdf_page.get_size()
        scale = min(scale, MAX_SIDE_PX / max(width, height, 1))
        image = pdf_page.render(scale=scale).to_pil()
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        return buffer.getvalue()
    except Exception as e:  # a page that PDFium cannot draw must not break the app
        log.warning("Could not render page %d: %s", page, e)
        return None
    finally:
        if pdf is not None:
            pdf.close()
