"""Persistent embedding index. The saved index records what it was built from, so a stale one is rebuilt.

Only the embeddings are stored (cache_dir/retrieval/<doc_hash>/). Chunk text always comes from the caller,
and a digest of it is part of the index key, so the saved vectors can never be matched to different text.
"""

import hashlib
import json
import logging
import os
import re
import shutil
import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from rag.config import CONFIG

log = logging.getLogger(__name__)

INDEX_FORMAT = 1
DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"

# Maps a list of texts to an (n, dim) float32 array of L2-normalised vectors. Carries a `.name`.
Embedder = Callable[[List[str]], np.ndarray]

_SAFE_HASH = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_embedders: Dict[str, "SentenceTransformerEmbedder"] = {}
_embedders_lock = threading.Lock()


class SentenceTransformerEmbedder:
    """Loads the model on first use. One instance per model name is shared (a UI rerun must not reload it)."""

    def __init__(self, model_name: str):
        self.name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer  # ModuleNotFoundError is left visible

                try:
                    self._model = SentenceTransformer(self.name)
                except Exception as e:
                    raise RuntimeError(
                        f"Could not load the embedding model {self.name!r}. The first run needs internet "
                        f"to download it; after that it is cached. Set RAG_EMBED_MODEL to change it. ({e})"
                    ) from e
            return self._model

    def __call__(self, texts: List[str]) -> np.ndarray:
        vectors = self._load().encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32
        )
        return np.asarray(vectors, dtype=np.float32)


def get_embedder(model_name: Optional[str] = None) -> SentenceTransformerEmbedder:
    name = model_name or os.environ.get("RAG_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    with _embedders_lock:
        return _embedders.setdefault(name, SentenceTransformerEmbedder(name))


@dataclass
class Index:
    chunks: List[Dict]
    embeddings: np.ndarray  # (n_chunks, dim), rows L2-normalised, so a dot product is cosine similarity
    meta: Dict


def index_dir(doc_hash: str, cache_dir: Optional[str] = None) -> Optional[str]:
    """Folder for one document's index, or None if doc_hash is not a plain token (never path-join it)."""
    if not _SAFE_HASH.match(doc_hash or ""):
        return None
    return os.path.join(cache_dir or CONFIG.cache_dir, "retrieval", doc_hash)


def index_key(chunks: List[Dict], doc_hash: str, embedder_name: str) -> Dict:
    """Everything the saved vectors depend on. If any of it differs, the index is rebuilt."""
    digest = hashlib.sha256()
    for c in chunks:
        digest.update(f"{c['id']}\x1f{c['text']}\x1e".encode("utf-8"))
    return {
        "format": INDEX_FORMAT,
        "doc_hash": doc_hash,
        "embedding_model": embedder_name,
        "chunk_size": CONFIG.chunk_size,
        "chunk_overlap": CONFIG.chunk_overlap,
        "parsers": sorted({c.get("parser", "") for c in chunks}),
        "n_chunks": len(chunks),
        "chunks_sha256": digest.hexdigest(),
    }


def _load(path: str, key: Dict) -> Optional[np.ndarray]:
    try:
        with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
            if json.load(f) != key:
                log.info("index for %s is stale (settings changed), rebuilding", key["doc_hash"][:8])
                return None
        vectors = np.load(os.path.join(path, "embeddings.npy"), allow_pickle=False)
    except FileNotFoundError:
        return None
    except Exception as e:  # a corrupt cache must cost a rebuild, never a crash
        log.warning("index for %s is unreadable (%s), rebuilding", key["doc_hash"][:8], e)
        return None
    return vectors if vectors.ndim == 2 and vectors.shape[0] == key["n_chunks"] else None


def _save(path: str, vectors: np.ndarray, key: Dict) -> None:
    """Write to a temp folder, then swap it in, so an interrupted save never leaves a half-written index."""
    tmp = f"{path}.tmp-{os.getpid()}-{threading.get_ident()}"
    try:
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        np.save(os.path.join(tmp, "embeddings.npy"), vectors, allow_pickle=False)
        with open(os.path.join(tmp, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(key, f, indent=2)
        shutil.rmtree(path, ignore_errors=True)
        os.replace(tmp, path)
    except OSError as e:  # saving is an optimisation: search still works from memory
        log.warning("could not save the index (%s); it will be rebuilt next time", e)
        shutil.rmtree(tmp, ignore_errors=True)


def build_or_load(
    chunks: List[Dict], doc_hash: str, embedder: Embedder, cache_dir: Optional[str] = None
) -> Index:
    key = index_key(chunks, doc_hash, getattr(embedder, "name", "unknown"))
    path = index_dir(doc_hash, cache_dir)
    vectors = _load(path, key) if path else None
    if vectors is None:
        vectors = embedder([c["text"] for c in chunks]) if chunks else np.zeros((0, 0), dtype=np.float32)
        if path:
            _save(path, vectors, key)
        log.info("built index for %s: %d chunks", doc_hash[:8], len(chunks))
    else:
        log.info("loaded saved index for %s: %d chunks", doc_hash[:8], len(chunks))
    return Index(chunks=chunks, embeddings=vectors, meta=key)


def delete_index(doc_hash: str, cache_dir: Optional[str] = None) -> None:
    path = index_dir(doc_hash, cache_dir)
    if path:
        shutil.rmtree(path, ignore_errors=True)
