

def test_axis_store_round_trips_and_replaces_by_path(tmp_path):
    from moment_miner.store import AxisStore

    a = AxisStore(tmp_path)
    assert a.get(["v:0.0"]) == {}
    a.add([{"id": "v:0.0", "path": "/x.mp4", "t0": 0.0, "t1": 8.0,
            "action": "vault", "who": "one person", "scene": "a park",
            "light": "sunny", "things": "rail", "move": ""}])
    assert a.get(["v:0.0"])["v:0.0"]["action"] == "vault"
    # A re-caption of the same file replaces its rows rather than doubling them.
    a.delete_path("/x.mp4")
    a.add([{"id": "v:0.0", "path": "/x.mp4", "t0": 0.0, "t1": 8.0,
            "action": "pull-up", "who": "one person", "scene": "a park",
            "light": "sunny", "things": "rail", "move": ""}])
    assert len(a.all()) == 1
    assert a.get(["v:0.0"])["v:0.0"]["action"] == "pull-up"


def test_tied_scores_rank_in_segment_order_whatever_the_row_order(tmp_path):
    import numpy as np

    from moment_miner.store import SegmentStore

    query = np.eye(8, dtype=np.float32)[0]
    tied = (query + np.eye(8, dtype=np.float32)[1]) / np.sqrt(2)

    def rows(path, starts, text="kong vault over a rail", vec=tied):
        return [{"id": f"{path}:{t0}", "video_id": path, "path": f"/{path}.mp4",
                 "t0": t0, "t1": t0 + 8.0, "text": text, "vector": vec,
                 "motion": 1.0} for t0 in starts]

    def ranked(store):
        return [h["id"] for h in store.search(query, "kong vault", k=20)]

    store = SegmentStore(tmp_path, "mock")
    store.add(rows("a", [8.0, 100.0, 12.0]) + rows("b", [0.0, 4.0])
              + rows("z", [0.0], text="kong vault kong vault", vec=query))
    store.rebuild_fts()
    # The better match leads; the tied rows follow by video, then by time.
    expected = ["z:0.0", "a:8.0", "a:12.0", "a:100.0", "b:0.0", "b:4.0"]
    assert ranked(store) == expected

    # What a re-index does to one video: its rows leave and come back.
    store.delete_path("/a.mp4")
    store.add(rows("a", [100.0, 12.0, 8.0]))
    store.rebuild_fts()
    assert ranked(store) == expected


def test_a_tied_group_larger_than_k_is_cut_in_segment_order(tmp_path):
    import numpy as np

    from moment_miner.store import SegmentStore

    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(6, 8)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    def rows(order):
        return [{"id": f"a:{float(i)}", "video_id": "a", "path": "/a.mp4",
                 "t0": float(i), "t1": i + 8.0, "text": "kong vault over a rail",
                 "vector": vecs[i], "motion": 1.0} for i in order]

    def ranked(store):
        return [h["id"] for h in store.search(vecs[5], "kong vault", k=3)]

    store = SegmentStore(tmp_path, "mock")
    store.add(rows(range(6)))
    store.rebuild_fts()
    first = ranked(store)
    assert len(first) == 3

    store.delete_path("/a.mp4")
    store.add(rows([5, 3, 1, 4, 2, 0]))
    store.rebuild_fts()
    assert ranked(store) == first
