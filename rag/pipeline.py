"""Question time: search -> one answer call -> validate citations. At most one query rewrite."""
import logging
import time
from typing import Callable, Dict, List

from rag.answer import NOT_FOUND, answer
from rag.citations import resolve
from rag.config import CONFIG
from rag.providers import Generate

log = logging.getLogger(__name__)

Search = Callable[[str, int], List[Dict]]  # provided by retrieval (Member B)

_REWRITE = (
    "Write a short passage (2-3 sentences) that would directly answer the question, as if it came "
    "from the source document. Use natural sentences, not keywords.\n\nQuestion: {question}"
)


def ask(question: str, search: Search, generate: Generate, max_retries: int = CONFIG.max_retries) -> Dict:
    """Return {"answer", "citations", "found", "retries", "latency_s", "chunks"}."""
    start = time.time()
    query = question
    for attempt in range(max_retries + 1):
        chunks = search(query, CONFIG.top_k)
        result = answer(question, chunks, generate)
        log.info("attempt=%d retrieved=%d found=%s", attempt, len(chunks), result["found"])
        if result["found"]:
            citations = resolve(result["source_ids"], chunks)
            if citations:  # an answer with no valid citation is not trusted
                return {"answer": result["answer"], "citations": citations, "found": True,
                        "retries": attempt, "latency_s": time.time() - start, "chunks": chunks}
        if attempt < max_retries:
            query = generate(_REWRITE.format(question=question))
    return {"answer": NOT_FOUND, "citations": [], "found": False,
            "retries": max_retries, "latency_s": time.time() - start, "chunks": []}
