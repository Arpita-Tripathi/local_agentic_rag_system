"""Ingest entry point: validate -> hash -> extract -> (layout) -> (OCR) -> chunk."""
import logging
from typing import Callable, Dict, List, Optional

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
from rag.ingest.quality import assess_page, ocr_quality, summarize
from rag.ingest.validate import PDFValidationError, file_sha256, validate_pdf

log = logging.getLogger(__name__)

LAYOUT_MODES = ("off", "auto", "on")


def _fmt_pages(nums: List[int], limit: int = 10) -> str:
    shown = ", ".join(map(str, nums[:limit]))
    return shown + (f" and {len(nums) - limit} more" if len(nums) > limit else "")


def ingest_pdf(
    path: str, layout: str = "", ocr: bool = True, progress: Optional[Callable[[str, float], None]] = None
) -> Dict:
    """Return {"doc_hash", "n_pages", "chunks", "image_only_pages", "parsers", "warnings", "page_quality",
    "quality_summary"}. page_quality has one entry per page: score (0..1 or None for OCR), level, reasons.

    layout: "off" | "auto" (Docling only when pypdf output looks table-heavy and Docling is installed)
    | "on" (Docling required). Defaults to RAG_LAYOUT.
    progress(message, fraction) is called at each stage so a UI can show a progress bar.
    Raises PDFValidationError on bad input, or when no page yields any text (message says why).
    "image_only_pages" are the pages that still have no text after every enabled step.
    """
    layout = layout or CONFIG.layout
    if layout not in LAYOUT_MODES:
        raise ValueError(f"layout must be one of {LAYOUT_MODES}, got {layout!r}")

    def step(msg: str, frac: float) -> None:
        if progress:
            progress(msg, frac)

    step("Checking the file", 0.05)
    n_pages = validate_pdf(path)
    doc_hash = file_sha256(path)
    step(f"Reading text from {n_pages} page(s)", 0.2)
    pages = extract_pages(path)
    quality = {p["page"]: p.get("quality") or assess_page(p["text"], []) for p in pages}
    warnings: List[str] = []

    if layout == "on" or (layout == "auto" and looks_table_heavy(pages) and layout_available()):
        step("Analysing tables and layout", 0.4)
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
        step(f"Running OCR on {len(needs_ocr)} page(s)", 0.6)
        try:
            pages, ocr_warnings = ocr_pages(path, pages)
            warnings.extend(ocr_warnings)
        except OCRUnavailable as e:
            warnings.append(f"Pages {_fmt_pages(needs_ocr)} have no readable text and need OCR. {e}")
    elif needs_ocr:
        warnings.append(f"Pages {_fmt_pages(needs_ocr)} have no readable text; OCR is turned off.")

    for p in pages:
        if p.get("type") == "ocr":
            quality[p["page"]] = ocr_quality()
    page_quality = [{"page": n, **quality[n]} for n in sorted(quality)]
    legacy = [q["page"] for q in page_quality if q["legacy_font"]]
    if legacy:
        warnings.append(
            f"Pages {_fmt_pages(legacy)} use an old Hindi font (such as Kruti Dev) whose text cannot be read "
            "directly. They are read with OCR when it is available; otherwise they are skipped."
        )

    step("Splitting into passages", 0.9)
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
        "page_quality": page_quality,
        "quality_summary": summarize(page_quality),
    }
