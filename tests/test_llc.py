import json
import os

import pytest

from moment_miner import export
from moment_miner.export import NameTaken, cut_project, read_project, write_llc_projects
from moment_miner.hostpaths import in_container


def test_write_llc_one_project_per_video_hits_in_time_order(tmp_path):
    video = tmp_path / "clips" / "a.mp4"
    video.parent.mkdir()
    video.write_bytes(b"x")
    hits = [
        {"path": str(video), "t0": 12.0, "t1": 20.0},
        {"path": str(video), "t0": 4.0, "t1": 8.0},
    ]
    written, _ = write_llc_projects(hits, tmp_path / "cuts", "kong vault")
    assert written == [str(tmp_path / "cuts" / "a-kong-vault.llc")]

    a = json.loads((tmp_path / "cuts" / "a-kong-vault.llc").read_text())
    assert a["version"] == 2
    assert [s["start"] for s in a["cutSegments"]] == [4.0, 12.0]
    assert [s["name"] for s in a["cutSegments"]] == ["kong vault? #2", "kong vault? #1"]


def test_write_llc_custom_dir_relative_media(tmp_path):
    video = tmp_path / "archive" / "sub" / "b.mov"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")
    out_dir = tmp_path / "results"
    written, _ = write_llc_projects([{"path": str(video), "t0": 1.0, "t1": 3.0}], out_dir, "q")
    b = json.loads((out_dir / "b-q.llc").read_text())
    assert b["mediaFileName"] == os.path.join("..", "archive", "sub", "b.mov")
    assert (out_dir / b["mediaFileName"]).resolve() == video.resolve()
    assert written == [str(out_dir / "b-q.llc")]


def test_write_llc_dir_stem_collision(tmp_path):
    hits = [
        {"path": "/x/a.mp4", "t0": 1.0, "t1": 2.0},
        {"path": "/y/a.mp4", "t0": 3.0, "t1": 4.0},
    ]
    written, _ = write_llc_projects(hits, tmp_path, "q")
    assert written == [str(tmp_path / "a-q.llc"), str(tmp_path / "a-q_2.llc")]


def test_different_queries_never_share_a_project(tmp_path):
    hits = [{"path": "/x/a.mp4", "t0": 1.0, "t1": 2.0}]
    first, _ = write_llc_projects(hits, tmp_path, "kong vault")
    second, taken = write_llc_projects(hits, tmp_path, "Back flip?")
    assert (first, second, taken) == ([str(tmp_path / "a-kong-vault.llc")],
                                      [str(tmp_path / "a-back-flip.llc")], [])


def test_write_llc_in_a_container_names_the_video_by_host_paths(tmp_path, monkeypatch):
    videos, data = tmp_path / "videos", tmp_path / "data"
    video = videos / "src" / "a.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")
    monkeypatch.setenv("MM_HOST_PATHS", f"{videos}=/proj/footage;{data}=/home/me/.mm_data")
    out_dir = data / "cuts" / "run"
    write_llc_projects([{"path": str(video), "t0": 1.0, "t1": 3.0}], out_dir, "q")
    a = json.loads((out_dir / "a-q.llc").read_text())
    on_host = os.path.normpath(os.path.join("/home/me/.mm_data/cuts/run", a["mediaFileName"]))
    assert on_host == "/proj/footage/src/a.mp4"


def test_write_llc_never_replaces_an_existing_project(tmp_path):
    hits = [{"path": "/x/a.mp4", "t0": 1.0, "t1": 2.0}]
    (tmp_path / "a-q.llc").write_text("mine")
    written, taken = write_llc_projects(hits, tmp_path, "q")
    assert (written, taken) == ([], [str(tmp_path / "a-q.llc")])
    assert (tmp_path / "a-q.llc").read_text() == "mine"


# The way LosslessCut saves a project: JSON5 with single quotes and trailing
# commas, an end left out when the segment runs to the end of the video.
LOSSLESSCUT_PROJECT = """{
  version: 2,
  mediaFileName: 'a.mp4',
  cutSegments: [
    {
      start: 1.5,
      end: 2.5,
      name: 'kong vault? #2',
    },
    {
      start: 3,
      end: 4,
      name: 'not exported',
      selected: false,
    },
    {
      start: 7.25,
      name: '',
    },
  ],
}
"""


def test_a_losslesscut_project_reads_as_its_video_and_selected_segments(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"x")
    project = tmp_path / "a-proj.llc"
    project.write_text(LOSSLESSCUT_PROJECT)
    video, segments = read_project(project)
    assert video == str(tmp_path / "a.mp4")
    assert segments == [(1.5, 2.5, "kong vault? #2"), (7.25, None, "")]


def test_a_project_naming_no_video_is_refused(tmp_path):
    project = tmp_path / "x.llc"
    project.write_text("{version: 2, cutSegments: []}")
    with pytest.raises(Exception, match="names no video"):
        read_project(project)


def _fake_cuts(monkeypatch, fail_at=()):
    """export_clip that records its calls and raises for starts in `fail_at`,
    so the loop around it is tested without ffmpeg."""
    calls = []

    def fake(path, start, end, out, smart=True, prefix="", encode=None):
        calls.append((start, end, prefix))
        if start in fail_at:
            raise RuntimeError("ffmpeg export failed: boom")
        return os.path.join(out, f"{prefix}clip-{start}"), start, end

    monkeypatch.setattr(export, "probe", lambda p: {"duration": 10.0})
    monkeypatch.setattr(export, "export_clip", fake)
    return calls


def test_cut_puts_the_rank_in_front_and_runs_an_open_end_to_the_video_end(tmp_path, monkeypatch):
    calls = _fake_cuts(monkeypatch)
    (tmp_path / "a.mp4").write_bytes(b"x")
    project = tmp_path / "a-proj.llc"
    project.write_text(LOSSLESSCUT_PROJECT)
    written, taken, failed = cut_project(project, tmp_path / "out")
    assert calls == [(1.5, 2.5, "02_"), (7.25, 10.0, "")]
    assert (len(written), taken, failed) == (2, [], [])


def test_cut_continues_past_a_failed_segment(tmp_path, monkeypatch):
    """#38: one failed export used to end the run, losing every clip after it."""
    calls = _fake_cuts(monkeypatch, fail_at={1.5})
    (tmp_path / "a.mp4").write_bytes(b"x")
    project = tmp_path / "a-proj.llc"
    project.write_text(LOSSLESSCUT_PROJECT)
    written, taken, failed = cut_project(project, tmp_path / "out")
    assert [c[0] for c in calls] == [1.5, 7.25]
    assert [w[2] for w in written] == [7.25]
    assert len(failed) == 1 and "1.500-2.500" in failed[0] and "boom" in failed[0]


def test_cut_records_a_taken_name_and_goes_on(tmp_path, monkeypatch):
    _fake_cuts(monkeypatch)

    def taken_first(path, start, end, out, smart=True, prefix="", encode=None):
        if start == 1.5:
            raise NameTaken(os.path.join(out, "02_taken.mp4"))
        return os.path.join(out, "x"), start, end

    monkeypatch.setattr(export, "export_clip", taken_first)
    (tmp_path / "a.mp4").write_bytes(b"x")
    project = tmp_path / "a-proj.llc"
    project.write_text(LOSSLESSCUT_PROJECT)
    written, taken, failed = cut_project(project, tmp_path / "out")
    assert (len(written), taken, failed) == (1, [str(tmp_path / "out" / "02_taken.mp4")], [])


def test_cut_refuses_a_project_whose_video_is_gone(tmp_path):
    project = tmp_path / "a-proj.llc"
    project.write_text(LOSSLESSCUT_PROJECT)
    with pytest.raises(Exception, match="is not there"):
        cut_project(project, tmp_path / "out")


def test_a_host_path_maps_into_the_deepest_mount(monkeypatch):
    """An output folder inside the footage is its own writable mount: mapping
    it into the read-only footage mount would refuse every write."""
    monkeypatch.setenv("MM_HOST_PATHS", "/videos=/mnt/f/Parkour;/out=/mnt/f/Parkour/day1")
    assert in_container("/mnt/f/Parkour/day1/a.llc") == "/out/a.llc"
    assert in_container("/mnt/f/Parkour/b.mp4") == "/videos/b.mp4"
    assert in_container("/mnt/f/Parkour2/b.mp4") == "/mnt/f/Parkour2/b.mp4"
    assert in_container("/videos/b.mp4") == "/videos/b.mp4"
