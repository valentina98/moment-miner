import csv

from click.testing import CliRunner

from moment_miner.cli import main

from .conftest import requires_ffmpeg


def test_annotate_from_template_and_append(tmp_path):
    folder = tmp_path / "footage"
    folder.mkdir()
    (folder / "run1.mp4").write_bytes(b"x")
    (folder / "run2.mp4").write_bytes(b"x")

    data = tmp_path / "data"
    result = CliRunner().invoke(main, [
        "--data-dir", str(data), "annotate", str(folder), "--template", "parkour",
    ], input="kong vault over an obstacle\n2\n0:04\n0:09\n\n")
    assert result.exit_code == 0, result.output
    assert "created" in result.output and "1 label(s) appended" in result.output
    assert sorted(p.name for p in folder.iterdir()) == ["run1.mp4", "run2.mp4"]

    rows = list(csv.DictReader((data / "labels.csv").open()))
    filled = [r for r in rows if r["video"]]
    # category comes back empty rather than holding the video name: a positional append
    # against this 5-column header would shift every field one place.
    assert filled == [{"query": "kong vault over an obstacle", "category": "",
                       "video": "run2.mp4", "start": "4.0", "end": "9.0"}]
    # template queries are present as unfilled rows, carrying their category
    assert any(r["query"] == "backflip" and r["category"] == "flips" for r in rows)


def test_annotate_appends_to_a_labels_file_without_a_category_column(tmp_path):
    """footage/labels.csv predates the column, so the four-field header must still work."""
    folder = tmp_path / "footage"
    folder.mkdir()
    (folder / "run1.mp4").write_bytes(b"x")
    labels = tmp_path / "labels.csv"
    labels.write_text("query,video,start,end\n")

    result = CliRunner().invoke(main, [
        "annotate", str(folder), "--labels", str(labels),
    ], input="backflip\n1\n0:02\n0:05\n\n")
    assert result.exit_code == 0, result.output

    rows = list(csv.DictReader(labels.open()))
    assert rows == [{"query": "backflip", "video": "run1.mp4",
                     "start": "2.0", "end": "5.0"}]


def test_help_command():
    runner = CliRunner()
    top = runner.invoke(main, ["help"])
    assert top.exit_code == 0 and "mine" in top.output and "annotate" in top.output

    sub = runner.invoke(main, ["help", "mine"])
    assert sub.exit_code == 0 and "mm mine" in sub.output and "-k" in sub.output

    bad = runner.invoke(main, ["help", "nope"])
    assert bad.exit_code != 0 and "unknown command" in bad.output

    ver = runner.invoke(main, ["--version"])
    assert ver.exit_code == 0 and "mm" in ver.output


def test_annotate_help_lists_templates():
    result = CliRunner().invoke(main, ["annotate", "--help"])
    assert result.exit_code == 0
    assert "parkour" in result.output and "wedding" in result.output


def test_annotate_unknown_template(tmp_path):
    result = CliRunner().invoke(main, ["annotate", str(tmp_path), "--template", "nope"])
    assert result.exit_code != 0
    assert "unknown template" in result.output


@requires_ffmpeg
def test_mine_end_to_end(tmp_path, synthetic_video):
    data_dir = tmp_path / "mm_data"
    out_dir = tmp_path / "clips"
    result = CliRunner().invoke(main, [
        "--data-dir", str(data_dir),
        "mine", "test pattern", str(synthetic_video.parent),
        "-o", str(out_dir), "-k", "2", "--backend", "mock", "--no-asr",
    ])
    assert result.exit_code == 0, result.output
    assert "indexing 1 new/changed video(s)" in result.output
    clips = list(out_dir.glob("*.mp4"))
    assert len(clips) >= 1
    assert clips[0].stat().st_size > 1024

    # second run: nothing re-indexed, still exports
    result = CliRunner().invoke(main, [
        "--data-dir", str(data_dir),
        "mine", "test pattern", str(synthetic_video.parent),
        "-o", str(out_dir), "-k", "1", "--backend", "mock", "--no-asr",
    ])
    assert result.exit_code == 0, result.output
    assert "indexing" not in result.output


@requires_ffmpeg
def test_mine_reindex_keeps_the_stored_captions(tmp_path, synthetic_video):
    from moment_miner.manifest import Manifest
    from moment_miner.store import AxisStore, SegmentStore

    data_dir = tmp_path / "mm_data"
    args = ["--data-dir", str(data_dir), "mine", "test pattern", str(synthetic_video.parent),
            "-o", str(tmp_path / "clips"), "-k", "1", "--backend", "mock", "--no-asr"]
    assert CliRunner().invoke(main, args).exit_code == 0
    axes = {"action": "kong vault", "who": "one person", "scene": "park", "light": "sunny"}
    AxisStore(data_dir).add([{"id": s["id"], "path": s["path"], "t0": s["t0"], "t1": s["t1"], **axes}
                             for s in SegmentStore(data_dir, "mock").segments()])
    Manifest(data_dir / "manifest.db").reset(synthetic_video.parent)

    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert "indexing 1 new/changed video(s)" in result.output
    texts = {s["text"] for s in SegmentStore(data_dir, "mock").segments()}
    assert texts == {"kong vault, one person, park, sunny"}


@requires_ffmpeg
def test_mine_default_output_is_mined_sibling(tmp_path, synthetic_video):
    result = CliRunner().invoke(main, [
        "--data-dir", str(tmp_path / "mm_data"),
        "mine", "test pattern", str(synthetic_video.parent),
        "-k", "1", "--backend", "mock", "--no-asr",
    ])
    assert result.exit_code == 0, result.output
    mined = synthetic_video.parent.parent / f"{synthetic_video.parent.name}_mined"
    runs = list(mined.iterdir())
    assert len(runs) == 1 and runs[0].name.startswith("test-pattern_")
    assert list(runs[0].glob("*.mp4"))


def _two_indexed(tmp_path):
    from moment_miner.manifest import Manifest
    folder = tmp_path / "footage"
    folder.mkdir()
    for name in ("a.mp4", "b.mp4"):
        (folder / name).write_bytes(b"x" * 64)
    m = Manifest(tmp_path / "data" / "manifest.db")
    m.scan(folder)
    for row in m.pending():
        m.mark_done(row["path"], 10.0, {"win": 8.0, "stride": 4.0, "frames_per_window": 8,
                                        "frame_w": 456, "frame_h": 256, "backend": "mock"})
    return m, folder


def test_a_deleted_video_is_forgotten_and_the_rest_stay(tmp_path):
    from moment_miner.cli import _forget_missing
    m, folder = _two_indexed(tmp_path)
    (folder / "a.mp4").unlink()
    counts = m.scan(folder)
    _forget_missing(m, counts, tmp_path / "data", "mock", folder)
    assert counts["removed"] == 1
    assert [r["path"] for r in m.conn.execute("SELECT path FROM videos")] == [
        str((folder / "b.mp4").resolve())]


def test_every_video_missing_looks_like_a_new_mount_and_removes_nothing(tmp_path):
    from moment_miner.cli import _forget_missing
    m, folder = _two_indexed(tmp_path)
    for p in folder.iterdir():
        p.unlink()
    counts = m.scan(folder)
    _forget_missing(m, counts, tmp_path / "data", "mock", folder)
    assert counts["removed"] == 0
    assert m.conn.execute("SELECT count(*) FROM videos").fetchone()[0] == 2


def test_a_local_mm_data_left_behind_is_named(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mm_data").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    runner = CliRunner()
    warned = runner.invoke(main, ["status"])
    assert "./mm_data is not read" in warned.output
    chosen = runner.invoke(main, ["--data-dir", "mm_data", "status"])
    assert "./mm_data is not read" not in chosen.output
