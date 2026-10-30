"""Page-wise text extraction with pypdf. Image-only pages are flagged, not OCR'd here."""
import logging
from typing import Dict, List

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from rag.config import CONFIG
from rag.ingest.validate import PDFValidationError

log = logging.getLogger(__name__)


def extract_pages(path: str, min_chars: int = CONFIG.min_page_chars) -> List[Dict]:
    """Return [{"page": 1-based, "text": str, "image_only": bool}, ...]."""
    try:
        reader = PdfReader(path)
        page_list = list(reader.pages)
    except (PdfReadError, ValueError, OSError) as e:
        raise PDFValidationError(f"The PDF could not be read: {e}") from e

    pages = []
    for i, page in enumerate(page_list, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception as e:  # one bad page must not kill the whole document
            log.warning("Could not extract text from page %d: %s", i, e)
            text = ""
        pages.append({"page": i, "text": text, "image_only": len(text) < min_chars})
    return pages
