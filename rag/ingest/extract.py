"""Page-wise text extraction with pypdf. Image-only pages are flagged, not OCR'd here."""
from typing import Dict, List

from pypdf import PdfReader

from rag.config import CONFIG


def extract_pages(path: str, min_chars: int = CONFIG.min_page_chars) -> List[Dict]:
    """Return [{"page": 1-based, "text": str, "image_only": bool}, ...]."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:  # one bad page must not kill the whole document
            text = ""
        pages.append({"page": i, "text": text, "image_only": len(text) < min_chars})
    return pages
