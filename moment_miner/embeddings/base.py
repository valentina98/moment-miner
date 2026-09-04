from abc import ABC, abstractmethod

import numpy as np


class EmbeddingBackend(ABC):
    """Maps text and video segments into one shared vector space.

    Implementations must L2-normalize outputs so L2 ANN ranking equals
    cosine ranking.
    """

    name: str
    dim: int

    @abstractmethod
    def embed_text(self, texts: list[str]) -> np.ndarray:
        """(len(texts), dim) float32."""

    @abstractmethod
    def embed_segment(self, frames: np.ndarray) -> np.ndarray:
        """(k, h, w, 3) uint8 frames of one segment → (dim,) float32."""
