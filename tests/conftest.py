import hashlib

import numpy as np

from src.rag.index import normalize


class FakeEmbedder:
    """Deterministic stand-in for the embedding model: a vector from each text's hash."""

    name = "fake-embedder"

    def embed_documents(self, texts):
        return normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text):
        return normalize(self._vec(text))

    @staticmethod
    def _vec(text):
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        return np.random.default_rng(seed).normal(size=16)
