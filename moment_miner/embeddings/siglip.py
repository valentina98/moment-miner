import numpy as np

from .base import EmbeddingBackend

DEFAULT_MODEL = "google/siglip2-base-patch16-256"


class SiglipBackend(EmbeddingBackend):
    """Frame-level SigLIP2 baseline: mean-pooled frame embeddings.

    Zero-shot text↔video retrieval baseline that runs on CPU or GPU.
    Weak on motion-defined actions — upgrade path is a native video
    encoder (PE-AV / InternVideo2) behind this same interface.
    """

    name = "siglip"
    dim = 768

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None):
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_id).eval().to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_id)

    def embed_text(self, texts: list[str]) -> np.ndarray:
        inputs = self.processor(
            text=texts, padding="max_length", truncation=True, return_tensors="pt"
        ).to(self.device)
        with self._torch.no_grad():
            feats = self.model.get_text_features(**inputs)
        return self._normalize(feats)

    def embed_segment(self, frames: np.ndarray) -> np.ndarray:
        from PIL import Image

        images = [Image.fromarray(f) for f in frames]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with self._torch.no_grad():
            feats = self.model.get_image_features(**inputs)
        pooled = self._normalize(feats).mean(axis=0)
        return pooled / np.linalg.norm(pooled)

    def _normalize(self, feats) -> np.ndarray:
        # get_text_features returns a ModelOutput in some transformers
        # versions while get_image_features returns a tensor.
        if hasattr(feats, "pooler_output"):
            feats = feats.pooler_output
        arr = feats.cpu().float().numpy()
        return arr / np.linalg.norm(arr, axis=-1, keepdims=True)
