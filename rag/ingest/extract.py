"""Page-wise text extraction with pypdf. Image-only pages are flagged, not OCR'd here."""
import logging
import re
from typing import Dict, List

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from rag.config import CONFIG
from rag.ingest.validate import PDFValidationError

log = logging.getLogger(__name__)

# pypdf warns once per page about emoji/symbol fonts whose character maps it can't fully read
# ("Odd-length string", "Skipping broken line"). It skips those entries and still extracts the text,
# so only real errors are shown. Our own per-page failures and garbled-text warnings are logged below.
logging.getLogger("pypdf").setLevel(logging.ERROR)

_CID = re.compile(r"\(cid:\d+\)")
_JUNK_CHAR = re.compile(r"[�-\x00-\x08\x0b\x0c\x0e-\x1f]")


def looks_garbled(text: str, threshold: float = 0.1) -> bool:
    """True when extraction produced mostly junk (unmapped glyphs, private-use or replacement characters).

    This is typical of PDFs with custom fonts, including many Devanagari documents. It cannot catch
    text that decodes to plausible-looking but wrong letters, so it is a safety net, not a guarantee.
    """
    if not text:
        return False
    junk = len(_JUNK_CHAR.findall(text)) + sum(len(m) for m in _CID.findall(text))
    return junk / len(text) > threshold


def extract_pages(path: str, min_chars: int = CONFIG.min_page_chars) -> List[Dict]:
    """Return [{"page": 1-based, "text": str, "image_only": bool, "garbled": bool}, ...].

    "image_only" means "no usable text": the page is empty, nearly empty, or garbled. Such pages
    need OCR. Garbled text is discarded rather than indexed.
    """
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
        garbled = looks_garbled(text)
        if garbled:
            log.warning("Page %d text looks garbled; it will need OCR.", i)
            text = ""
        image_only = garbled or len(text) < min_chars
        pages.append({"page": i, "text": text, "image_only": image_only, "garbled": garbled})
    return pages
