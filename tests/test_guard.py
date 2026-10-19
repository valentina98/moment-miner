import hashlib
import re
import sqlite3

import pytest
from click.testing import CliRunner

from moment_miner import guard
from moment_miner.cli import main
from moment_miner.export import clip_name

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


def test_an_existing_video_is_never_replaced(tmp_path, armed):
    (tmp_path / "data" / "cuts").mkdir(parents=True)
    (tmp_path / "data" / "cuts" / "a.MP4").write_bytes(b"x")
    with pytest.raises(guard.WriteRefused, match="never replaced"):
        guard.writable(tmp_path / "data" / "cuts" / "a.MP4")
    guard.writable(tmp_path / "data" / "cuts" / "b.MP4")


def test_clips_are_named_the_losslesscut_way_with_exact_times():
    assert (clip_name("/f/2M4A8050.MP4", 72.3449, 80.678, "01_")
            == "01_2M4A8050-00.01.12.345-00.01.20.678.MP4")
    assert clip_name("a.mp4", 3725.0, 3726.5) == "a-01.02.05.000-01.02.06.500.mp4"


def _sha1(path):
    return hashlib.sha1(path.read_bytes()).hexdigest()


@requires_ffmpeg
def test_clips_go_into_the_footage_folder_and_a_taken_name_is_skipped(tmp_path, synthetic_video):
    src = synthetic_video.parent
    original = (_sha1(synthetic_video), synthetic_video.stat().st_mtime_ns)
    mine = ["mine", "test pattern", str(src), "--backend", "mock", "--no-asr", "-k", "1",
            "-o", str(src)]
    first = _mm(tmp_path, *mine)
    assert first.exit_code == 0, first.output
    [clip] = [p for p in src.iterdir() if p != synthetic_video]
    assert re.fullmatch(r"01_test_video-\d\d\.\d\d\.\d\d\.\d{3}-\d\d\.\d\d\.\d\d\.\d{3}\.mp4",
                        clip.name), clip.name
    clip_before = (_sha1(clip), clip.stat().st_mtime_ns)

    again = _mm(tmp_path, *mine)
    assert again.exit_code == 1
    assert "1 clip(s) not written, the name was taken" in again.output
    assert str(clip) in again.output
    assert (_sha1(clip), clip.stat().st_mtime_ns) == clip_before
    assert (_sha1(synthetic_video), synthetic_video.stat().st_mtime_ns) == original


@requires_ffmpeg
def test_an_index_never_picks_up_the_tools_own_clips(tmp_path, synthetic_video):
    src = synthetic_video.parent
    data = src / "mm_data"
    for args in (["mine", "test pattern", str(src), "--backend", "mock", "--no-asr",
                  "-k", "1", "-o", str(src)],
                 ["export", str(synthetic_video), "0:01", "0:03"],
                 ["index", str(src), "--backend", "mock", "--no-asr", "--no-estimate"]):
        result = CliRunner().invoke(main, ["--data-dir", str(data), *args])
        assert result.exit_code == 0, (args, result.output)
    assert len(list(src.rglob("*.mp4"))) == 3
    conn = sqlite3.connect(data / "manifest.db")
    assert [r[0] for r in conn.execute("SELECT path FROM videos")] == [str(synthetic_video)]
