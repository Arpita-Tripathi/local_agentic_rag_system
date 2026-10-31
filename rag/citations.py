"""Validate model-returned source IDs against what retrieval actually returned."""
import logging
from typing import Dict, List

log = logging.getLogger(__name__)


def resolve(source_ids: List[str], retrieved: List[Dict], snippet_chars: int = 240) -> List[Dict]:
    """Keep only IDs that were really retrieved; page numbers come from stored metadata, never the model."""
    by_id = {c["id"]: c for c in retrieved}
    out = []
    for sid in source_ids:
        chunk = by_id.get(sid)
        if chunk is None:
            log.warning("Dropping unknown citation id %r", sid)
            continue
        out.append({
            "id": sid,
            "page": chunk["page"],
            "snippet": chunk["text"][:snippet_chars],
            "parser": chunk.get("parser", ""),
            "type": chunk.get("type", "text"),
        })
    return out
