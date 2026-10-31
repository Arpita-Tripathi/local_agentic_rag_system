"""Hybrid search: dense (cosine) and BM25 rankings fused with reciprocal rank fusion.

Order comes from RRF. The reported "score" is the dense cosine similarity (0..1), not the RRF value:
RRF with k=60 tops out near 0.03, which would fall below the pipeline's 0.15 threshold and make every
question abstain. Cosine is on the scale that threshold was written for, and it is what rejects
questions the document does not cover, even when a stray keyword matches.
"""

import re
from typing import Callable, Dict, List

import numpy as np
from rank_bm25 import BM25Okapi

from rag.config import CONFIG
from rag.retrieval.index import Embedder, Index

RRF_K = 60
CANDIDATES = 20  # how deep each ranking contributes to the fusion

Search = Callable[[str, int], List[Dict]]

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if len(t) > 1]


def rrf_fuse(rankings: List[List[int]], k: int = RRF_K) -> Dict[int, float]:
    """Reciprocal rank fusion: each ranking gives 1 / (k + rank) to the items it lists (rank starts at 1)."""
    fused: Dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return fused


def make_search(index: Index, embedder: Embedder, min_score: float = CONFIG.score_threshold) -> Search:
    chunks = index.chunks
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks]) if chunks else None

    def search(query: str, top_k: int = CONFIG.top_k) -> List[Dict]:
        if not chunks or top_k <= 0 or not (query or "").strip():
            return []
        pool = max(CANDIDATES, top_k)

        cosine = index.embeddings @ embedder([query])[0]
        dense_rank = np.argsort(-cosine, kind="stable")[:pool].tolist()

        words = tokenize(query)
        keyword_rank: List[int] = []
        if words:
            bm = bm25.get_scores(words)
            keyword_rank = [i for i in np.argsort(-bm, kind="stable")[:pool].tolist() if bm[i] > 0]

        fused = rrf_fuse([dense_rank, keyword_rank])
        ordered = sorted(fused, key=lambda i: (-fused[i], i))  # ties break by chunk position: deterministic
        hits = []
        for i in ordered:
            score = float(min(max(cosine[i], 0.0), 1.0))
            if score >= min_score:
                hits.append({**chunks[i], "score": score, "rrf_score": fused[i]})
            if len(hits) == top_k:
                break
        return hits

    return search
