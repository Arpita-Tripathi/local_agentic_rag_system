"""Turn extracted pages into chunk records that carry their provenance."""
import re
from typing import Dict, List

from rag.config import CONFIG

_HEADING = re.compile(r"^(\d+(\.\d+)*\.?\s+\S.{0,80}|[A-Z][A-Z \-]{3,60})$")
_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")  # includes the Hindi danda


def _find_heading(text: str, current: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line and _HEADING.match(line):
            current = line
    return current


def _split(text: str, size: int, overlap: int) -> List[str]:
    sentences = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
    chunks, buf = [], ""
    for s in sentences:
        while len(s) > size:  # a single very long sentence
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(s[:size])
            s = s[size - overlap:]
        if buf and len(buf) + len(s) + 1 > size:
            chunks.append(buf)
            buf = buf[-overlap:] + " " + s if overlap else s
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
    chunks: List[Dict] = []
    heading = ""
    for p in pages:
        if p["image_only"]:
            continue  # OCR (Phase 2) will fill these in
        heading = _find_heading(p["text"], heading)
        for part in _split(p["text"], chunk_size, overlap):
            chunks.append({
                "id": f"c{len(chunks):04d}",
                "doc_hash": doc_hash,
                "page": p["page"],
                "heading": heading,
                "type": "text",
                "text": part,
                "parser": parser,
            })
    return chunks
