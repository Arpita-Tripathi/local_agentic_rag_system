"""Connects the app to retrieval without depending on Member B's code being finished.

CONTRACT (agree with Member B): rag/retrieval/__init__.py exposes
    build_search(chunks: list[dict], doc_hash: str) -> Search
where Search is `search(query: str, top_k: int) -> list[dict]` (chunk + "score"). The returned callable may
carry a `.name` attribute (shown in the UI) and the module may expose `delete_index(doc_hash)`.

Until that exists, the app uses the keyword search below. It is a DEVELOPMENT STUB: it has no embeddings,
no Hindi/English semantics and no benchmark. The UI says so out loud whenever it is in use.
"""
import re
import sys
from typing import Callable, Dict, List, Tuple

Search = Callable[[str, int], List[Dict]]

STUB_NAME = "development keyword search (rag.retrieval not installed)"

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if len(t) > 1]


def keyword_search(chunks: List[Dict]) -> Search:
    """Score = share of distinct query words found in the chunk (0..1). Deterministic, no downloads."""
    indexed = [(c, set(_tokens(c["text"]))) for c in chunks]

    def search(query: str, top_k: int = 5) -> List[Dict]:
        words = set(_tokens(query))
        if not words:
            return []
        scored = [(len(words & toks) / len(words), c) for c, toks in indexed]
        scored = [(s, c) for s, c in scored if s > 0]
        scored.sort(key=lambda sc: (-sc[0], sc[1]["id"]))
        return [{**c, "score": s} for s, c in scored[:top_k]]

    search.name = STUB_NAME  # type: ignore[attr-defined]
    return search


def _retrieval_module():
    try:
        import rag.retrieval  # noqa: F401  (runs the import; the module is read from sys.modules below)
    except ModuleNotFoundError as e:
        if e.name and e.name.startswith("rag.retrieval"):
            return None
        raise  # a real missing dependency inside Member B's code must not be hidden
    # Not `import rag.retrieval as mod`: that reads the attribute on the rag package, which can be stale.
    return sys.modules["rag.retrieval"]


def build_search(chunks: List[Dict], doc_hash: str) -> Tuple[Search, str]:
    """Return (search, backend_name). Uses Member B's retrieval when it is installed."""
    mod = _retrieval_module()
    if mod is not None and hasattr(mod, "build_search"):
        search = mod.build_search(chunks, doc_hash)
        return search, getattr(search, "name", "hybrid retrieval")
    search = keyword_search(chunks)
    return search, STUB_NAME


def delete_index(doc_hash: str) -> None:
    """Ask Member B's retrieval to drop the saved index for a document, if it can."""
    mod = _retrieval_module()
    if mod is not None and hasattr(mod, "delete_index"):
        mod.delete_index(doc_hash)
