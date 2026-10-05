import json
import os

from moment_miner.export import write_llc_projects


def test_write_llc_one_project_per_video_hits_in_time_order(tmp_path):
    video = tmp_path / "clips" / "a.mp4"
    video.parent.mkdir()
    video.write_bytes(b"x")
    hits = [
        {"path": str(video), "t0": 12.0, "t1": 20.0},
        {"path": str(video), "t0": 4.0, "t1": 8.0},
    ]
    written = write_llc_projects(hits, tmp_path / "cuts", label="kong vault")
    assert written == [str(tmp_path / "cuts" / "a.llc")]

    a = json.loads((tmp_path / "cuts" / "a.llc").read_text())
    assert a["version"] == 1
    assert [s["start"] for s in a["cutSegments"]] == [4.0, 12.0]
    assert [s["name"] for s in a["cutSegments"]] == ["kong vault? #2", "kong vault? #1"]


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


def test_write_llc_in_a_container_names_the_video_by_host_paths(tmp_path, monkeypatch):
    videos, data = tmp_path / "videos", tmp_path / "data"
    video = videos / "src" / "a.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")
    monkeypatch.setenv("MM_HOST_PATHS", f"{videos}=/proj/footage;{data}=/home/me/.mm_data")
    out_dir = data / "cuts" / "run"
    write_llc_projects([{"path": str(video), "t0": 1.0, "t1": 3.0}], out_dir)
    a = json.loads((out_dir / "a.llc").read_text())
    on_host = os.path.normpath(os.path.join("/home/me/.mm_data/cuts/run", a["mediaFileName"]))
    assert on_host == "/proj/footage/src/a.mp4"
