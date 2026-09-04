import json
import os

from moment_miner.export import write_llc_projects


def test_write_llc_default_subfolder(tmp_path):
    video = tmp_path / "clips" / "a.mp4"
    video.parent.mkdir()
    video.write_bytes(b"x")
    hits = [
        {"path": str(video), "t0": 12.0, "t1": 20.0},
        {"path": str(video), "t0": 4.0, "t1": 8.0},
    ]
    written = write_llc_projects(hits, label="kong vault")
    assert written == [str(tmp_path / "clips" / "llc" / "a.llc")]

    a = json.loads((tmp_path / "clips" / "llc" / "a.llc").read_text())
    assert a["version"] == 1
    assert a["mediaFileName"] == "../a.mp4"
    assert [s["start"] for s in a["cutSegments"]] == [4.0, 12.0]
    assert a["cutSegments"][0]["name"] == "kong vault"
    # LosslessCut resolves media relative to the .llc location
    resolved = (tmp_path / "clips" / "llc" / a["mediaFileName"]).resolve()
    assert resolved == video.resolve()


def test_write_llc_custom_dir_relative_media(tmp_path):
    video = tmp_path / "archive" / "sub" / "b.mov"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")
    out_dir = tmp_path / "results"
    written = write_llc_projects([{"path": str(video), "t0": 1.0, "t1": 3.0}], out_dir)
    b = json.loads((out_dir / "b.llc").read_text())
    assert b["mediaFileName"] == os.path.join("..", "archive", "sub", "b.mov")
    assert (out_dir / b["mediaFileName"]).resolve() == video.resolve()
    assert written == [str(out_dir / "b.llc")]


def test_write_llc_dir_stem_collision(tmp_path):
    hits = [
        {"path": "/x/a.mp4", "t0": 1.0, "t1": 2.0},
        {"path": "/y/a.mp4", "t0": 3.0, "t1": 4.0},
    ]
    written = write_llc_projects(hits, tmp_path, label="q")
    assert len(set(written)) == 2
