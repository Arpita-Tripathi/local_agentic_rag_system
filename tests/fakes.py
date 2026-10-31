"""Test doubles shared by the test files."""

import hashlib

import numpy as np

STOPWORDS = frozenset(
    "a an the of is are was were what which who how when do does did to in on at for with by and or "
    "की का के को से ने में है हैं था थी थे क्या कौन कब कितने कितनी कितना".split()
)


class FakeEmbedder:
    """Bag-of-words (minus stopwords) hashed into 4096 dims, L2-normalised. Deterministic. Counts texts."""

    def __init__(self, name="fake-embedder"):
        self.name = name
        self.embedded = 0

    def __call__(self, texts):
        self.embedded += len(texts)
        out = np.zeros((len(texts), 4096), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                word = word.strip(".,!?।")
                if word in STOPWORDS:
                    continue
                out[row, int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % 4096] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)
