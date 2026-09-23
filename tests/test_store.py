

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
