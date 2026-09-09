from .embeddings.base import EmbeddingBackend
from .motion import STATIC_MAX
from .store import MotionStore, SegmentStore


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
    static: bool | None = None,
    motion_store: MotionStore | None = None,
    static_max: float = STATIC_MAX,
) -> list[dict]:
    vec = backend.embed_text([query])[0]
    ranked = [
        store.vector_search(vec, k=candidates),
        store.text_search(query, k=candidates),
    ]
    if static is not None:
        # Filtered before fusing, not after, so k results still come back.
        # The threshold is applied here rather than at index time so it can be
        # retuned without a re-index. A segment with no motion row carries no
        # value and is dropped rather than guessed at.
        ids = [r["id"] for lst in ranked for r in lst]
        motion = motion_store.get(ids) if motion_store is not None else {}
        ranked = [
            [r for r in lst
             if r["id"] in motion and (motion[r["id"]] < static_max) is static]
            for lst in ranked
        ]
    return merge_overlapping(rrf_fuse(ranked))[:k]
