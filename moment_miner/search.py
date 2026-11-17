from .embeddings.base import EmbeddingBackend
from .motion import STATIC_MAX
from .store import SegmentStore


def merge_overlapping(hits: list[dict]) -> list[dict]:
    """NMS over time: absorb lower-ranked hits overlapping a kept hit.

    A kept hit lists the caption of every segment it absorbed, best text match
    first: the moment a merged span covers can sit in a segment the fused
    ranking put below the one it kept."""
    kept: list[dict] = []
    for h in hits:
        for k_ in kept:
            if k_["video_id"] == h["video_id"] and not (
                h["t1"] <= k_["t0"] or h["t0"] >= k_["t1"]
            ):
                k_["t0"] = min(k_["t0"], h["t0"])
                k_["t1"] = max(k_["t1"], h["t1"])
                k_["merged"].append(h)
                break
        else:
            kept.append(dict(h, merged=[h]))
    for k_ in kept:
        best_first = sorted(k_.pop("merged"), key=lambda h: -h.get("text_score", 0.0))
        k_["captions"], seen = [], set()
        for h in best_first:
            # Neighbouring segments often share a caption word for word.
            if h.get("text") and h["text"] not in seen:
                seen.add(h["text"])
                k_["captions"].append({"t0": h["t0"], "t1": h["t1"], "text": h["text"]})
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
