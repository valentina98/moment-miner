from .base import EmbeddingBackend


def get_backend(name: str, **kwargs) -> EmbeddingBackend:
    if name == "mock":
        from .mock import MockBackend
        return MockBackend()
    if name == "siglip":
        from .siglip import SiglipBackend
        return SiglipBackend(**kwargs)
    raise ValueError(f"unknown embedding backend: {name!r} (available: mock, siglip)")
