"""Docling fallback for table- or layout-heavy PDFs. Optional install: pip install docling.

pypdf stays the default parser because Docling is a heavy install. This module only runs when the
layout mode asks for it (RAG_LAYOUT=on) or when the pypdf output looks table-heavy (RAG_LAYOUT=auto).
"""
import logging
import re
from importlib import metadata
from typing import Dict, List, Tuple

log = logging.getLogger(__name__)

_NUMERIC = re.compile(r"^[\(\-+$€₹%]*\d[\d,.\-/%)]*$")


class LayoutUnavailable(Exception):
    """Docling is not installed, or it failed on this document."""


def _docling_version():
    try:
        return metadata.version("docling")
    except metadata.PackageNotFoundError:
        return None


def layout_available() -> bool:
    return _docling_version() is not None


def looks_table_heavy(pages: List[Dict], min_lines: int = 10, min_share: float = 0.15) -> bool:
    """Heuristic on pypdf output: many lines made mostly of numbers suggests tables that pypdf flattens."""
    total = tabular = 0
    for p in pages:
        if p.get("image_only"):
            continue
        for line in p["text"].splitlines():
            tokens = line.split()
            if not tokens:
                continue
            total += 1
            if len(tokens) >= 4 and sum(bool(_NUMERIC.match(t)) for t in tokens) / len(tokens) >= 0.6:
                tabular += 1
    return tabular >= min_lines and total > 0 and tabular / total >= min_share


def _flush(blocks: List[Dict], page: int, heading: str, lines: List[str], tag: str) -> None:
    if lines:
        blocks.append({"page": page, "type": "text", "text": "\n".join(lines), "heading": heading,
                       "parser": tag, "image_only": False, "garbled": False})


def extract_layout_blocks(path: str) -> Tuple[List[Dict], str]:
    """Parse with Docling. Returns (blocks, parser_tag); a block is a page-dict with type text|table.

    Table blocks hold markdown. Raises LayoutUnavailable with a user-readable message on any failure.
    """
    if not layout_available():
        raise LayoutUnavailable("Docling is not installed. Run: pip install docling (large download).")
    tag = f"docling@{_docling_version()}"
    try:
        from docling.document_converter import DocumentConverter

        doc = DocumentConverter().convert(path).document
        items = list(doc.iterate_items())
    except Exception as e:
        raise LayoutUnavailable(f"Docling could not parse this PDF: {e}") from e

    blocks: List[Dict] = []
    heading, page, lines = "", None, []
    for item, _level in items:
        prov = getattr(item, "prov", None)
        if not prov:
            continue
        item_page = prov[0].page_no
        kind = type(item).__name__
        if page is not None and item_page != page:
            _flush(blocks, page, heading, lines, tag)
            lines = []
        page = item_page
        if kind == "TableItem":
            _flush(blocks, page, heading, lines, tag)
            lines = []
            md = item.export_to_markdown(doc=doc).strip()
            if md:
                blocks.append({"page": page, "type": "table", "text": md, "heading": heading,
                               "parser": tag, "image_only": False, "garbled": False})
        elif kind == "SectionHeaderItem":
            _flush(blocks, page, heading, lines, tag)
            heading, lines = item.text.strip(), [item.text.strip()]
        else:
            text = (getattr(item, "text", "") or "").strip()
            if text:
                lines.append(text)
    if page is not None:
        _flush(blocks, page, heading, lines, tag)
    return blocks, tag


def merge_layout(pages: List[Dict], blocks: List[Dict]) -> List[Dict]:
    """Use Docling blocks for the pages it produced content for; keep the pypdf page for the rest."""
    by_page: Dict[int, List[Dict]] = {}
    for b in blocks:
        by_page.setdefault(b["page"], []).append(b)
    merged: List[Dict] = []
    for p in pages:
        merged.extend(by_page.get(p["page"], [p]))
    return merged
