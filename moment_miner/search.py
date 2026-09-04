from .embeddings.base import EmbeddingBackend
from .store import SegmentStore


def rrf_fuse(rank_lists: list[list[dict]], k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    by_id: dict[str, dict] = {}
    for ranking in rank_lists:
        for rank, row in enumerate(ranking):
            scores[row["id"]] = scores.get(row["id"], 0.0) + 1.0 / (k + rank + 1)
            by_id.setdefault(row["id"], row)
    fused = [dict(by_id[i], score=s) for i, s in scores.items()]
    return sorted(fused, key=lambda r: r["score"], reverse=True)


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
) -> list[dict]:
    vec = backend.embed_text([query])[0]
    fused = rrf_fuse([
        store.vector_search(vec, k=candidates),
        store.text_search(query, k=candidates),
    ])
    return merge_overlapping(fused)[:k]
