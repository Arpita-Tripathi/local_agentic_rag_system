"""Ingest entry point: validate -> hash -> extract -> (layout) -> (OCR) -> chunk."""
import logging
from typing import Dict, List

from rag.config import CONFIG
from rag.ingest.chunk import build_chunks
from rag.ingest.extract import extract_pages
from rag.ingest.layout import (
    LayoutUnavailable,
    extract_layout_blocks,
    layout_available,
    looks_table_heavy,
    merge_layout,
)
from rag.ingest.ocr import OCRUnavailable, ocr_pages
from rag.ingest.validate import PDFValidationError, file_sha256, validate_pdf

log = logging.getLogger(__name__)

LAYOUT_MODES = ("off", "auto", "on")


def _fmt_pages(nums: List[int], limit: int = 10) -> str:
    shown = ", ".join(map(str, nums[:limit]))
    return shown + (f" and {len(nums) - limit} more" if len(nums) > limit else "")


def ingest_pdf(path: str, layout: str = "", ocr: bool = True) -> Dict:
    """Return {"doc_hash", "n_pages", "chunks", "image_only_pages", "parsers", "warnings"}.

    layout: "off" | "auto" (Docling only when pypdf output looks table-heavy and Docling is installed)
    | "on" (Docling required). Defaults to RAG_LAYOUT.
    Raises PDFValidationError on bad input, or when no page yields any text (message says why).
    "image_only_pages" are the pages that still have no text after every enabled step.
    """
    layout = layout or CONFIG.layout
    if layout not in LAYOUT_MODES:
        raise ValueError(f"layout must be one of {LAYOUT_MODES}, got {layout!r}")

    n_pages = validate_pdf(path)
    doc_hash = file_sha256(path)
    pages = extract_pages(path)
    warnings: List[str] = []

    if layout == "on" or (layout == "auto" and looks_table_heavy(pages) and layout_available()):
        try:
            blocks, _ = extract_layout_blocks(path)
            pages = merge_layout(pages, blocks)
        except LayoutUnavailable as e:
            if layout == "on":
                raise PDFValidationError(str(e)) from e
            warnings.append(f"Layout parser skipped: {e}")
    elif layout == "auto" and looks_table_heavy(pages):
        warnings.append(
            "This PDF looks table-heavy; install Docling (pip install docling) for better tables."
        )

    needs_ocr = [p["page"] for p in pages if p["image_only"]]
    if needs_ocr and ocr:
        try:
            pages, ocr_warnings = ocr_pages(path, pages)
            warnings.extend(ocr_warnings)
        except OCRUnavailable as e:
            warnings.append(f"Pages {_fmt_pages(needs_ocr)} have no readable text and need OCR. {e}")
    elif needs_ocr:
        warnings.append(f"Pages {_fmt_pages(needs_ocr)} have no readable text; OCR is turned off.")

    chunks = build_chunks(pages, doc_hash)
    image_only = sorted({p["page"] for p in pages if p["image_only"]})
    if not chunks:
        detail = " ".join(warnings) or "The PDF has no extractable text."
        raise PDFValidationError(f"No text could be extracted from this PDF. {detail}")
    for w in warnings:
        log.warning("ingest: %s", w)
    return {
        "doc_hash": doc_hash,
        "n_pages": n_pages,
        "chunks": chunks,
        "image_only_pages": image_only,
        "parsers": sorted({c["parser"] for c in chunks}),
        "warnings": warnings,
    }
