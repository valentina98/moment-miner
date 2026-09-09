import numpy as np

from moment_miner.embeddings.mock import MockBackend
from moment_miner.search import merge_overlapping, rrf_fuse, search
from moment_miner.store import MotionStore, SegmentStore


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


def test_second_mount_root_upserts_rather_than_duplicating(tmp_path):
    be = MockBackend()
    store = SegmentStore(tmp_path, be.name)
    docs = ["athlete performs kong vault", "crowd waiting"]
    store.add(_rows(be, docs))
    again = _rows(be, docs)
    for r in again:
        r["path"] = "/other/root/v1.mp4"
    store.add(again)

    assert store.count() == len(docs)
    assert {r["path"] for r in store.vector_search(be.embed_text(docs)[0])} == {
        "/other/root/v1.mp4"
    }


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


def test_static_filter_uses_the_motion_table(tmp_path):
    be = MockBackend()
    store = SegmentStore(tmp_path, be.name)
    docs = ["athlete performs kong vault", "crowd waiting"]
    store.add(_rows(be, docs))
    store.rebuild_fts()
    motion = MotionStore(tmp_path)
    motion.add([
        {"id": "v1:0", "path": "/x/v1.mp4", "t0": 0.0, "t1": 8.0, "motion": 0.05},
        {"id": "v1:1", "path": "/x/v1.mp4", "t0": 4.0, "t1": 12.0, "motion": 0.90},
    ])

    q = "athlete performs kong vault"
    still = search(q, be, store, k=5, static=True, motion_store=motion)
    moving = search(q, be, store, k=5, static=False, motion_store=motion)
    assert [h["id"] for h in still] == ["v1:0"]
    assert [h["id"] for h in moving] == ["v1:1"]


def test_the_threshold_is_applied_at_search_time_not_stored(tmp_path):
    """The whole point of keeping the float: retuning must not need a re-index."""
    be = MockBackend()
    store = SegmentStore(tmp_path, be.name)
    store.add(_rows(be, ["athlete performs kong vault"]))
    store.rebuild_fts()
    motion = MotionStore(tmp_path)
    motion.add([{"id": "v1:0", "path": "/x/v1.mp4", "t0": 0.0, "t1": 8.0,
                 "motion": 0.40}])

    q = "athlete performs kong vault"
    # 0.40 is moving under the 0.35 default and static once the cut moves up.
    assert search(q, be, store, k=5, static=True, motion_store=motion) == []
    assert [h["id"] for h in search(q, be, store, k=5, static=True,
                                    motion_store=motion, static_max=0.5)] == ["v1:0"]


def test_segments_with_no_motion_row_drop_out_of_a_filtered_search(tmp_path):
    """Guessing a value would silently put moving shots into a --static result."""
    be = MockBackend()
    store = SegmentStore(tmp_path, be.name)
    store.add(_rows(be, ["athlete performs kong vault"]))
    store.rebuild_fts()
    motion = MotionStore(tmp_path)

    q = "athlete performs kong vault"
    assert search(q, be, store, k=5, static=True, motion_store=motion) == []
    assert search(q, be, store, k=5, static=False, motion_store=motion) == []
    # Unfiltered search is unaffected by the motion table being empty.
    assert len(search(q, be, store, k=5)) == 1


def test_motion_is_shared_across_backends(tmp_path):
    """It is a property of the frames, not the embedding, so one table serves
    every backend rather than being recomputed per segments_<backend>."""
    motion = MotionStore(tmp_path)
    motion.add([{"id": "v1:0", "path": "/x/v1.mp4", "t0": 0.0, "t1": 8.0,
                 "motion": 0.05}])
    assert MotionStore(tmp_path).get(["v1:0"]) == {"v1:0": 0.05}


def test_deleting_a_path_drops_its_motion_rows(tmp_path):
    motion = MotionStore(tmp_path)
    motion.add([
        {"id": "v1:0", "path": "/x/v1.mp4", "t0": 0.0, "t1": 8.0, "motion": 0.05},
        {"id": "v2:0", "path": "/x/v2.mp4", "t0": 0.0, "t1": 8.0, "motion": 0.90},
    ])
    motion.delete_path("/x/v1.mp4")
    assert motion.get(["v1:0", "v2:0"]) == {"v2:0": 0.90}
