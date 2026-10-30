"""Turn extracted pages into chunk records that carry their provenance."""
import re
from typing import Dict, List, Tuple

from rag.config import CONFIG

# "3. Method" / "2.1 Setup" (must not read like a sentence), or a short ALL-CAPS line.
_HEADING = re.compile(r"^(\d+(\.\d+)*\.?\s+[A-Zऀ-ॿ][^.!?:;]{0,80}|[A-Z][A-Z \-]{3,60})$")
_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")  # includes the Hindi danda


def _is_heading(line: str) -> bool:
    return bool(line) and bool(_HEADING.match(line))


def _segments(text: str, current: str) -> Tuple[List[Tuple[str, str]], str]:
    """Split page text at heading lines -> [(heading, text)], so each chunk gets the heading it sits under."""
    segs: List[Tuple[str, List[str]]] = [(current, [])]
    for raw in text.splitlines():
        line = raw.strip()
        if _is_heading(line):
            current = line
            segs.append((current, [line]))
        elif line:
            segs[-1][1].append(line)
    return [(h, " ".join(lines)) for h, lines in segs if lines], current


def _tail(buf: str, overlap: int) -> str:
    """Last `overlap` chars of buf, starting at a word boundary."""
    if not overlap or len(buf) <= overlap:
        return buf if overlap else ""
    tail = buf[-overlap:]
    space = tail.find(" ")
    return tail[space + 1:] if space != -1 else ""


def _split(text: str, size: int, overlap: int) -> List[str]:
    """Sentence-aware split. Every returned chunk is at most `size` characters."""
    sentences = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
    chunks: List[str] = []
    buf = ""
    for s in sentences:
        while len(s) > size:  # a single very long sentence
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(s[:size])
            s = s[size - overlap:]
        if buf and len(buf) + len(s) + 1 > size:
            chunks.append(buf)
            tail = _tail(buf, overlap)
            buf = f"{tail} {s}" if tail and len(tail) + len(s) + 1 <= size else s
        else:
            buf = f"{buf} {s}".strip()
    if buf:
        chunks.append(buf)
    return chunks


def _split_table(md: str, size: int) -> List[str]:
    """Split a markdown table on row boundaries and repeat the header row so each piece stays readable."""
    rows = [r for r in md.splitlines() if r.strip()]
    header = rows[:2] if len(rows) > 2 and set(rows[1].replace("|", "").strip()) <= set("-: ") else []
    head = "\n".join(header)
    pieces: List[str] = []
    buf = head
    for row in rows[len(header):]:
        row = row[:size]  # a single absurdly wide row must not break the size guarantee
        if buf != head and len(buf) + len(row) + 1 > size:
            pieces.append(buf)
            buf = head
        if len(buf) + len(row) + 1 > size:  # header + one row is already too big: drop the header copy
            buf = ""
        buf = f"{buf}\n{row}" if buf else row
    if buf and buf != head:
        pieces.append(buf)
    return pieces


def build_chunks(
    pages: List[Dict],
    doc_hash: str,
    chunk_size: int = CONFIG.chunk_size,
    overlap: int = CONFIG.chunk_overlap,
    parser: str = CONFIG.parser_version,
) -> List[Dict]:
    """Chunk record schema is the shared contract with retrieval (see implementation.md).

    A page dict may carry its own "type" (text|ocr|table), "parser" and "heading"; otherwise the
    defaults are text, the `parser` argument, and headings detected from the text.
    """
    if overlap >= chunk_size:
        raise ValueError("chunk overlap must be smaller than chunk size")
    chunks: List[Dict] = []
    heading = ""

    def add(page: dict, kind: str, seg_heading: str, text: str) -> None:
        chunks.append({
            "id": f"c{len(chunks):04d}",
            "doc_hash": doc_hash,
            "page": page["page"],
            "heading": seg_heading,
            "type": kind,
            "text": text,
            "parser": page.get("parser", parser),
        })

    for p in pages:
        if p["image_only"]:
            continue  # no usable text, and OCR was unavailable or found nothing
        kind = p.get("type", "text")
        if kind == "table":
            for part in _split_table(p["text"], chunk_size):
                add(p, "table", p.get("heading", heading), part)
            continue
        if "heading" in p:  # parser already knows the section (Docling)
            heading = p["heading"]
            for part in _split(" ".join(p["text"].split()), chunk_size, overlap):
                add(p, kind, heading, part)
            continue
        segments, heading = _segments(p["text"], heading)
        for seg_heading, seg_text in segments:
            for part in _split(seg_text, chunk_size, overlap):
                add(p, kind, seg_heading, part)
    return chunks
