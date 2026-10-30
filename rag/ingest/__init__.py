"""Ingest entry point: validate -> hash -> extract -> chunk."""
from typing import Dict

from rag.ingest.chunk import build_chunks
from rag.ingest.extract import extract_pages
from rag.ingest.validate import file_sha256, validate_pdf


def ingest_pdf(path: str) -> Dict:
    """Return {"doc_hash", "n_pages", "chunks", "image_only_pages"}. Raises PDFValidationError on bad input.

    image_only_pages lists pages with no usable text; they produce no chunks until OCR (Phase 2) exists,
    so the UI should tell the user about them instead of silently ignoring them.
    """
    n_pages = validate_pdf(path)
    doc_hash = file_sha256(path)
    pages = extract_pages(path)
    return {
        "doc_hash": doc_hash,
        "n_pages": n_pages,
        "chunks": build_chunks(pages, doc_hash),
        "image_only_pages": [p["page"] for p in pages if p["image_only"]],
    }
