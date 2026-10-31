"""Member B: hybrid retrieval. rag/search_backend.py imports build_search and delete_index from here."""

from typing import Dict, List, Optional

from rag.retrieval.index import Embedder, build_or_load, get_embedder
from rag.retrieval.index import delete_index as _delete_index
from rag.retrieval.search import Search, make_search


def build_search(
    chunks: List[Dict],
    doc_hash: str,
    embedder: Optional[Embedder] = None,
    cache_dir: Optional[str] = None,
) -> Search:
    """Build (or load the saved) index for one document; return search(query, top_k) -> chunks + "score"."""
    embedder = embedder or get_embedder()
    index = build_or_load(chunks, doc_hash, embedder, cache_dir)
    search = make_search(index, embedder)
    search.name = f"hybrid ({getattr(embedder, 'name', 'embedder')} + BM25)"  # type: ignore[attr-defined]
    return search


def delete_index(doc_hash: str, cache_dir: Optional[str] = None) -> None:
    _delete_index(doc_hash, cache_dir)


__all__ = ["build_search", "delete_index", "Search"]
