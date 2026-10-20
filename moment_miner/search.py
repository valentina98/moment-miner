from .embeddings.base import EmbeddingBackend
from .motion import STATIC_MAX
from .store import SegmentStore


def merge_overlapping(hits: list[dict]) -> list[dict]:
    """NMS over time: absorb lower-ranked hits overlapping a kept hit."""
    kept: list[dict] = []
    for h in hits:
        for k_ in kept:
            if k_["video_id"] == h["video_id"] and not (
                h["t1"] <= k_["t0"] or h["t0"] >= k_["t1"]
            ):
                k_["t0"] = min(k_["t0"], h["t0"])
                k_["t1"] = max(k_["t1"], h["t1"])
                break
        else:
            kept.append(dict(h))
    return kept


def search(
    query: str,
    backend: EmbeddingBackend,
    store: SegmentStore,
    k: int = 10,
    candidates: int = 50,
    static: bool | None = None,
    static_max: float = STATIC_MAX,
) -> list[dict]:
    where = None
    if static is not None:
        # Filtered before ranking, not after, so k results still come back.
        # The threshold is applied here rather than at index time so it can be
        # retuned without a re-index. A segment with no motion value matches
        # neither side rather than being guessed at.
        where = f"motion {'<' if static else '>='} {float(static_max)}"
    vec = backend.embed_text([query])[0]
    return merge_overlapping(store.search(vec, query, k=candidates, where=where))[:k]
