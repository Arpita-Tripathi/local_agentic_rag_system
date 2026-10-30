"""Reject bad uploads before any expensive work happens."""
import hashlib
import os
import tempfile

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


def save_upload(data: bytes, max_mb: int = CONFIG.max_pdf_mb) -> str:
    """Write uploaded bytes to a private temp file and return its path. The caller deletes it.

    The size limit is checked before anything touches the disk. The name never comes from the user.
    """
    if len(data) > max_mb * (1 << 20):
        raise PDFValidationError(f"File is {len(data) / (1 << 20):.1f} MB; the limit is {max_mb} MB.")
    fd, path = tempfile.mkstemp(suffix=".pdf", prefix="rag_upload_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
    except Exception:
        os.unlink(path)
        raise
    return path


def validate_pdf(path: str, max_mb: int = CONFIG.max_pdf_mb, max_pages: int = CONFIG.max_pages) -> int:
    """Return the page count, or raise PDFValidationError."""
    if not os.path.isfile(path):
        raise PDFValidationError(f"File not found: {path}")

    size = os.path.getsize(path)
    if size == 0:
        raise PDFValidationError("The file is empty.")
    size_mb = size / (1 << 20)
    if size_mb > max_mb:
        raise PDFValidationError(f"File is {size_mb:.1f} MB; the limit is {max_mb} MB.")

    with open(path, "rb") as f:
        if b"%PDF-" not in f.read(1024):  # the PDF spec allows a little junk before the header
            raise PDFValidationError("This is not a PDF file.")

    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise PDFValidationError("This PDF is password-protected. Remove the password and try again.")
        n_pages = len(reader.pages)
    except PDFValidationError:
        raise
    except (PdfReadError, ValueError, OSError, KeyError, RecursionError) as e:
        raise PDFValidationError(f"The PDF appears to be corrupt: {e}") from e

    if n_pages == 0:
        raise PDFValidationError("The PDF has no pages.")
    if n_pages > max_pages:
        raise PDFValidationError(f"The PDF has {n_pages} pages; the limit is {max_pages}.")
    return n_pages
