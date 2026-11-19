

def test_axis_store_round_trips_and_replaces_by_path(tmp_path):
    from moment_miner.store import AxisStore

    a = AxisStore(tmp_path)
    assert a.get(["v:0.0"]) == {}
    a.add([{"id": "v:0.0", "path": "/x.mp4", "t0": 0.0, "t1": 8.0,
            "action": "vault", "who": "one person", "scene": "a park",
            "light": "sunny"}])
    assert a.get(["v:0.0"])["v:0.0"]["action"] == "vault"
    # A re-caption of the same file replaces its rows rather than doubling them.
    a.delete_path("/x.mp4")
    a.add([{"id": "v:0.0", "path": "/x.mp4", "t0": 0.0, "t1": 8.0,
            "action": "pull-up", "who": "one person", "scene": "a park",
            "light": "sunny"}])
    assert len(a.all()) == 1
    assert a.get(["v:0.0"])["v:0.0"]["action"] == "pull-up"


def test_tied_scores_rank_by_segment_id_whatever_the_row_order(tmp_path):
    import numpy as np

    from moment_miner.store import SegmentStore

    vec = np.ones(8, dtype=np.float32) / np.sqrt(8)

    def rows(path, starts):
        return [{"id": f"{path}:{t0}", "video_id": path, "path": f"/{path}.mp4",
                 "t0": t0, "t1": t0 + 8.0, "text": "kong vault over a rail",
                 "vector": vec, "motion": 1.0} for t0 in starts]

    def ranked(store):
        return [h["id"] for h in store.search(vec, "kong vault", k=20)]

    store = SegmentStore(tmp_path, "mock")
    store.add(rows("a", [0.0, 4.0, 8.0]) + rows("b", [0.0, 4.0, 8.0]))
    store.rebuild_fts()
    expected = sorted(r["id"] for r in store.segments())
    assert ranked(store) == expected

    # What a re-index does to one video: its rows leave and come back.
    store.delete_path("/a.mp4")
    store.add(rows("a", [8.0, 4.0, 0.0]))
    store.rebuild_fts()
    assert ranked(store) == expected
