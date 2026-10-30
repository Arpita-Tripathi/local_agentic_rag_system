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


def build_chunks(
    pages: List[Dict],
    doc_hash: str,
    chunk_size: int = CONFIG.chunk_size,
    overlap: int = CONFIG.chunk_overlap,
    parser: str = CONFIG.parser_version,
) -> List[Dict]:
    """Chunk record schema is the shared contract with retrieval (see implementation.md)."""
    if overlap >= chunk_size:
        raise ValueError("chunk overlap must be smaller than chunk size")
    chunks: List[Dict] = []
    heading = ""
    for p in pages:
        if p["image_only"]:
            continue  # OCR (Phase 2) will fill these in
        segments, heading = _segments(p["text"], heading)
        for seg_heading, seg_text in segments:
            for part in _split(seg_text, chunk_size, overlap):
                chunks.append({
                    "id": f"c{len(chunks):04d}",
                    "doc_hash": doc_hash,
                    "page": p["page"],
                    "heading": seg_heading,
                    "type": "text",
                    "text": part,
                    "parser": parser,
                })
    return chunks
