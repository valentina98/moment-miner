import hashlib

import numpy as np

from .base import EmbeddingBackend


class MockBackend(EmbeddingBackend):
    """Deterministic hash-seeded vectors for tests. No semantic meaning."""

    name = "mock"
    dim = 64

    def _vec(self, key: bytes) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha1(key).digest()[:8], "little")
        v = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def embed_text(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t.encode()) for t in texts])

    def embed_frames(self, frames: np.ndarray) -> np.ndarray:
        return np.stack([self._vec(f.tobytes()) for f in frames])
