"""Append-only run log: one JSON line per question, so a failure can be reproduced.

Stays on this machine (under the cache dir). It records the question, the retrieval trace, the
outcome and the provider, never the PDF text. Set RAG_LOG_QUESTIONS=0 to store only a question hash.
"""
import hashlib
import json
import logging
import os
import time
from typing import Dict

from rag.config import CONFIG

log = logging.getLogger(__name__)


def log_run(question: str, result: Dict, doc_hash: str = "", cache_dir: str = CONFIG.cache_dir) -> None:
    """Never raises: a logging problem must not break the answer the user is waiting for."""
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "doc_hash": doc_hash,
        "question": question if CONFIG.log_questions else None,
        "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest()[:16],
        "found": result["found"],
        "reason": result["reason"],
        "retries": result["retries"],
        "latency_s": round(result["latency_s"], 3),
        "provider": result["provider"],
        "citations": [c["id"] for c in result["citations"]],
        "trace": result["trace"],
    }
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, "runs.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        log.warning("Could not write the run log: %s", e)
