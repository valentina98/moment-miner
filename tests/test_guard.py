import sqlite3

import pytest
from click.testing import CliRunner

from moment_miner import guard
from moment_miner.cli import main

from .conftest import requires_ffmpeg


@pytest.fixture
def armed(tmp_path):
    guard.arm(tmp_path / "data")
    yield
    guard.disarm()


def _tree(folder):
    return {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in [folder, *folder.rglob("*")]}


def _mm(tmp_path, *args, **kwargs):
    return CliRunner().invoke(main, ["--data-dir", str(tmp_path / "data"), *args], **kwargs)


def test_an_unarmed_guard_lets_library_callers_write_anywhere(tmp_path):
    assert guard.writable(tmp_path / "x") == tmp_path / "x"


def test_writes_go_to_the_data_folder_or_a_named_output(tmp_path, armed):
    guard.output(tmp_path / "clips")
    guard.writable(tmp_path / "data" / "cuts" / "a.mp4")
    guard.writable(tmp_path / "clips" / "a.mp4")
    with pytest.raises(guard.WriteRefused, match="outside the data folder"):
        guard.writable(tmp_path / "elsewhere" / "a.mp4")


def test_a_source_folder_is_refused_even_when_named_as_output(tmp_path, armed):
    guard.source(tmp_path / "footage")
    guard.output(tmp_path / "footage" / "clips")
    with pytest.raises(guard.WriteRefused, match="source folder"):
        guard.writable(tmp_path / "footage" / "clips" / "a.mp4")


def test_folders_from_earlier_scans_stay_sources(tmp_path, armed):
    (tmp_path / "data").mkdir()
    conn = sqlite3.connect(tmp_path / "data" / "manifest.db")
    conn.execute("CREATE TABLE videos (path TEXT)")
    conn.execute("INSERT INTO videos VALUES (?)", (str(tmp_path / "root" / "day1" / "a.mp4"),))
    conn.execute("CREATE TABLE folders (path TEXT)")
    conn.execute("INSERT INTO folders VALUES (?)", (str(tmp_path / "root"),))
    conn.commit()
    conn.close()
    guard.output(tmp_path)
    for target in ("root/day1/x.mp4", "root/x.mp4"):
        with pytest.raises(guard.WriteRefused, match="source folder"):
            guard.writable(tmp_path / target)
    guard.writable(tmp_path / "elsewhere" / "x.mp4")


@requires_ffmpeg
def test_no_command_writes_inside_the_source_folder(tmp_path, synthetic_video):
    src = synthetic_video.parent
    commands = [
        ["calibrate", str(synthetic_video), "--backend", "mock"],
        ["index", str(src), "--backend", "mock", "--no-asr", "--no-estimate",
         "--caption", "mock"],
        ["caption", str(src), "--model", "mock", "--backend", "mock", "--force"],
        ["search", "test pattern", "--backend", "mock", "--llc"],
        ["export", str(synthetic_video), "0:01", "0:03"],
        ["mine", "test pattern", str(src), "--backend", "mock", "--no-asr", "-k", "1"],
        ["locate", "-q", "test pattern", "-v", str(synthetic_video), "--backend", "mock",
         "-o", str(tmp_path / "locate.html")],
        ["annotate", str(src)],
    ]
    before = _tree(src)
    for args in commands:
        result = _mm(tmp_path, *args, input="\n")
        assert result.exit_code == 0, (args, result.output)
        assert _tree(src) == before, args

    [captions] = (tmp_path / "data" / "captions").glob("*.csv")
    result = _mm(tmp_path, "caption-compare", str(captions), "--videos", str(src))
    assert result.exit_code == 0, result.output
    assert _tree(src) == before
    assert (tmp_path / "data" / "caption-comparison.html").is_file()

    cuts = tmp_path / "data" / "cuts"
    assert len(list(cuts.glob("*.mp4"))) == 1
    assert len(list(cuts.glob("test-pattern_*/*.llc"))) == 1
    assert len(list(cuts.glob("test-pattern_*/*.mp4"))) == 1


@requires_ffmpeg
@pytest.mark.parametrize("args", [
    ["mine", "test pattern", "{src}", "--backend", "mock", "--no-asr", "-o", "{src}/clips"],
    ["export", "{video}", "0:01", "0:03", "-o", "{src}/clip.mp4"],
    ["index", "{src}", "--backend", "mock", "--no-asr", "--no-estimate",
     "--caption", "mock", "--caption-dir", "{src}/_captions"],
])
def test_an_output_inside_the_source_folder_is_refused(tmp_path, synthetic_video, args):
    src = synthetic_video.parent
    before = _tree(src)
    result = _mm(tmp_path, *(a.format(src=src, video=synthetic_video) for a in args))
    assert result.exit_code != 0
    assert "refusing to write" in result.output
    assert _tree(src) == before


@requires_ffmpeg
def test_a_data_folder_inside_the_source_folder_is_refused(tmp_path, synthetic_video):
    src = synthetic_video.parent
    before = _tree(src)
    result = CliRunner().invoke(main, ["--data-dir", str(src / "mm_data"), "index", str(src),
                                       "--backend", "mock", "--no-asr", "--no-estimate"])
    assert result.exit_code != 0
    assert "refusing to write" in result.output
    assert _tree(src) == before


@requires_ffmpeg
def test_a_folder_indexed_earlier_is_refused_to_later_commands(tmp_path, synthetic_video):
    root = tmp_path / "footage"
    sub = root / "day1"
    sub.mkdir(parents=True)
    video = sub / "clip.mp4"
    synthetic_video.rename(video)
    assert _mm(tmp_path, "index", str(root), "--backend", "mock", "--no-asr",
               "--no-estimate").exit_code == 0
    result = _mm(tmp_path, "export", str(video), "0:01", "0:03", "-o", str(root / "clip.mp4"))
    assert result.exit_code != 0
    assert "refusing to write" in result.output
