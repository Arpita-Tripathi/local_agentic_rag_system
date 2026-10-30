"""Tesseract OCR, run only on pages that have no usable text.

Pages are not rendered (that would need PyMuPDF, which is AGPL, or poppler). Scanned PDFs store each
page as an embedded image, so we OCR those images. A page drawn with vector outlines and no image
cannot be OCR'd this way and is reported instead of being silently skipped.
"""
import logging
from typing import Dict, List, Tuple

from pypdf import PdfReader

from rag.config import CONFIG

log = logging.getLogger(__name__)


class OCRUnavailable(Exception):
    """Tesseract (or its language data) is missing. The message says what to install."""


def _tesseract():
    try:
        import pytesseract
    except ImportError as e:
        raise OCRUnavailable("pytesseract is not installed. Run: pip install -r requirements.txt") from e
    if CONFIG.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = CONFIG.tesseract_cmd
    try:
        version = str(pytesseract.get_tesseract_version())
        installed = set(pytesseract.get_languages(config=""))
    except Exception as e:
        raise OCRUnavailable(
            "Tesseract is not installed or not on PATH. Install it (Windows: UB Mannheim build; "
            "Linux: apt install tesseract-ocr tesseract-ocr-hin) or set RAG_TESSERACT_CMD."
        ) from e
    missing = [lang for lang in CONFIG.ocr_langs.split("+") if lang and lang not in installed]
    if missing:
        raise OCRUnavailable(
            f"Tesseract language data missing: {', '.join(missing)}. Install the '{missing[0]}' traineddata "
            "(Linux: apt install tesseract-ocr-hin) or change RAG_OCR_LANGS."
        )
    return pytesseract, version


def ocr_available() -> Tuple[bool, str]:
    """(ok, message). Message is empty when OCR can run."""
    try:
        _tesseract()
    except OCRUnavailable as e:
        return False, str(e)
    return True, ""


def _page_images(page) -> list:
    try:
        return list(page.images)
    except Exception as e:  # corrupt image streams, missing Pillow, unsupported filters
        log.warning("Could not read images from a page: %s", e)
        return []


def ocr_pages(
    path: str, pages: List[Dict], min_chars: int = CONFIG.min_page_chars
) -> Tuple[List[Dict], List[str]]:
    """OCR every image-only page. Returns (updated pages, warnings). Raises OCRUnavailable up front.

    OCR'd pages get type "ocr" and parser "tesseract@<version>". Pages that stay empty are left
    image-only and named in the warnings.
    """
    todo = [p["page"] for p in pages if p["image_only"]]
    if not todo:
        return pages, []
    pytesseract, version = _tesseract()
    reader = PdfReader(path)
    warnings: List[str] = []
    out = []
    for p in pages:
        if not p["image_only"]:
            out.append(p)
            continue
        texts = []
        for img in _page_images(reader.pages[p["page"] - 1]):
            image = img.image
            if image.width * image.height > CONFIG.ocr_max_pixels:
                warnings.append(f"Page {p['page']}: an embedded image is too large to OCR and was skipped.")
                continue
            try:
                texts.append(pytesseract.image_to_string(image, lang=CONFIG.ocr_langs).strip())
            except Exception as e:
                warnings.append(f"Page {p['page']}: OCR failed ({e}).")
        text = "\n".join(t for t in texts if t)
        if len(text) >= min_chars:
            out.append({**p, "text": text, "image_only": False, "type": "ocr",
                        "parser": f"tesseract@{version}"})
        else:
            warnings.append(f"Page {p['page']}: no readable text found, even with OCR.")
            out.append(p)
    return out, warnings
