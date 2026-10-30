"""Reject bad uploads before any expensive work happens."""
import hashlib
import os

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from rag.config import CONFIG


class PDFValidationError(Exception):
    """Raised with a user-readable message when a PDF cannot be processed."""


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_pdf(path: str, max_mb: int = CONFIG.max_pdf_mb, max_pages: int = CONFIG.max_pages) -> int:
    """Return the page count, or raise PDFValidationError."""
    if not os.path.isfile(path):
        raise PDFValidationError(f"File not found: {path}")

    size_mb = os.path.getsize(path) / (1 << 20)
    if size_mb > max_mb:
        raise PDFValidationError(f"File is {size_mb:.1f} MB; the limit is {max_mb} MB.")

    with open(path, "rb") as f:
        if f.read(5) != b"%PDF-":
            raise PDFValidationError("This is not a PDF file.")

    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise PDFValidationError("This PDF is password-protected. Remove the password and try again.")
        n_pages = len(reader.pages)
    except PDFValidationError:
        raise
    except (PdfReadError, ValueError, OSError) as e:
        raise PDFValidationError(f"The PDF appears to be corrupt: {e}") from e

    if n_pages == 0:
        raise PDFValidationError("The PDF has no pages.")
    if n_pages > max_pages:
        raise PDFValidationError(f"The PDF has {n_pages} pages; the limit is {max_pages}.")
    return n_pages
