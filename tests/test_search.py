import numpy as np

from moment_miner.embeddings.mock import MockBackend
from moment_miner.search import merge_overlapping, rrf_fuse, search
from moment_miner.store import SegmentStore


def _rows(backend, docs):
    return [
        {"id": f"v1:{i}", "video_id": "v1", "path": "/x/v1.mp4",
         "t0": float(i * 4), "t1": float(i * 4 + 8), "text": text,
         "vector": backend.embed_text([text])[0]}
        for i, text in enumerate(docs)
    ]


def test_vector_and_fts_search(tmp_path):
    be = MockBackend()
    store = SegmentStore(tmp_path, be.name)
    store.add(_rows(be, ["athlete performs kong vault", "crowd waiting", "backflip off wall"]))
    store.rebuild_fts()

    hits = search("athlete performs kong vault", be, store, k=2)
    assert hits[0]["id"].startswith("v1:")
    assert hits[0]["text"] == "athlete performs kong vault"

    fts = store.text_search("backflip", k=5)
    assert fts and fts[0]["text"] == "backflip off wall"


def test_rrf_prefers_agreement():
    a = [{"id": "x"}, {"id": "y"}]
    b = [{"id": "y"}, {"id": "z"}]
    fused = rrf_fuse([a, b])
    assert fused[0]["id"] == "y"


def test_merge_overlapping():
    hits = [
        {"id": "1", "video_id": "v", "t0": 4.0, "t1": 12.0, "score": 0.9},
        {"id": "2", "video_id": "v", "t0": 8.0, "t1": 16.0, "score": 0.5},
        {"id": "3", "video_id": "v", "t0": 40.0, "t1": 48.0, "score": 0.4},
        {"id": "4", "video_id": "w", "t0": 8.0, "t1": 16.0, "score": 0.3},
    ]
    merged = merge_overlapping(hits)
    assert len(merged) == 3
    assert merged[0]["t0"] == 4.0 and merged[0]["t1"] == 16.0


def test_mock_backend_deterministic():
    be = MockBackend()
    v1 = be.embed_text(["hello"])[0]
    v2 = be.embed_text(["hello"])[0]
    assert np.allclose(v1, v2)
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-5
