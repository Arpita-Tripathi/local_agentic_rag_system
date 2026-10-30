"""Question time: search -> one answer call -> validate citations. At most one query rewrite."""
import logging
import time
from typing import Callable, Dict, List, Optional

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


def _above_threshold(chunks: List[Dict], min_score: float) -> List[Dict]:
    """Drop weak hits. Chunks without a "score" are kept (retrieval already thresholded them)."""
    return [c for c in chunks if c.get("score") is None or c["score"] >= min_score]


def _clean_query(rewritten: str, fallback: str, max_chars: int) -> str:
    rewritten = " ".join((rewritten or "").split())
    return rewritten[:max_chars] if rewritten else fallback


def ask(
    question: str,
    search: Search,
    generate: Generate,
    max_retries: int = CONFIG.max_retries,
    min_score: float = CONFIG.score_threshold,
) -> Dict:
    """Return {"answer", "citations", "found", "reason", "retries", "latency_s", "chunks", "trace",
    "provider"}.

    "chunks" is what retrieval returned (the newest non-empty attempt), so the UI can show the evidence
    even when the answer is "not found". "reason" is one of ok / low_score / not_found / no_valid_citation.
    """
    start = time.time()
    provider = getattr(generate, "name", "unknown")
    query = question
    trace: List[Dict] = []
    chunks: List[Dict] = []
    shown: List[Dict] = []  # newest non-empty retrieval, so a failed rewrite doesn't hide earlier evidence
    reason = "not_found"

    for attempt in range(max_retries + 1):
        chunks = search(query, CONFIG.top_k)
        shown = chunks or shown
        kept = _above_threshold(chunks, min_score)
        step = {"attempt": attempt, "query": query, "retrieved": len(chunks), "kept": len(kept)}

        if not kept:
            reason = "low_score"
        else:
            result = answer(question, kept, generate)
            if not result["found"]:
                reason = "not_found"
            else:
                citations = resolve(result["source_ids"], kept)
                if citations:  # an answer with no valid citation is not trusted
                    step["outcome"] = "ok"
                    trace.append(step)
                    log.info("ask %s", step)
                    return {"answer": result["answer"], "citations": citations, "found": True, "reason": "ok",
                            "retries": attempt, "latency_s": time.time() - start, "chunks": kept,
                            "trace": trace, "provider": provider}
                reason = "no_valid_citation"

        step["outcome"] = reason
        trace.append(step)
        log.info("ask %s", step)
        if attempt < max_retries:
            rewritten = generate(_REWRITE.format(question=question))
            query = _clean_query(rewritten, question, CONFIG.max_query_chars)
            log.info("retry %d: rewritten query=%r", attempt + 1, query)

    return {"answer": NOT_FOUND, "citations": [], "found": False, "reason": reason,
            "retries": max_retries, "latency_s": time.time() - start, "chunks": shown,
            "trace": trace, "provider": provider}


def describe_reason(reason: str) -> Optional[str]:
    """Short UI text for why an answer was withheld."""
    return {
        "low_score": "Nothing in the PDF was similar enough to the question.",
        "not_found": "The model found no supporting text in the retrieved passages.",
        "no_valid_citation": "The model cited no retrieved passage, so its answer was withheld.",
    }.get(reason)
