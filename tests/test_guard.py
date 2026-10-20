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
    cuts = tmp_path / "data" / "cuts"
    commands = [
        lambda: ["calibrate", str(synthetic_video), "--backend", "mock"],
        lambda: ["index", str(src), "--backend", "mock", "--no-asr", "--no-estimate"],
        lambda: ["caption", str(src), "--model", "mock", "--backend", "mock", "--force"],
        lambda: ["search", "test pattern", "--backend", "mock", "-k", "1"],
        lambda: ["cut", *map(str, cuts.glob("test-pattern_*/*.llc"))],
        lambda: ["locate", "-q", "test pattern", "-v", str(synthetic_video),
                 "--backend", "mock", "-o", str(tmp_path / "locate")],
        lambda: ["annotate", str(src)],
        lambda: ["caption-compare", *map(str, (tmp_path / "data" / "captions").glob("*.csv")),
                 "--videos", str(src)],
    ]
    before = _tree(src)
    for args in commands:
        args = args()
        result = _mm(tmp_path, *args, input="\n")
        assert result.exit_code == 0, (args, result.output)
        assert _tree(src) == before, args

    assert len(list(cuts.glob("test-pattern_*/*.llc"))) == 1
    assert len(list(cuts.glob("test_video-test-pattern_*/01_*.mp4"))) == 1
    assert len(list(cuts.glob("caption-compare_*/caption-comparison.html"))) == 1
    assert (tmp_path / "locate" / "locate.html").is_file()


@requires_ffmpeg
def test_a_second_run_into_the_same_folder_replaces_nothing(tmp_path, synthetic_video):
    """cli spec rule 5, for every command that writes results."""
    src, out = synthetic_video.parent, tmp_path / "out"
    for args in (["index", str(src), "--backend", "mock", "--no-asr", "--no-estimate"],
                 ["caption", str(src), "--model", "mock", "--backend", "mock"]):
        assert _mm(tmp_path, *args).exit_code == 0
    [captions] = (tmp_path / "data" / "captions").glob("*.csv")
    project = out / "search" / "test_video-test-pattern.llc"
    commands = {
        "search": ["search", "test pattern", "--backend", "mock", "-k", "1",
                   "-o", str(out / "search")],
        "cut": ["cut", str(project), "-o", str(out / "cut")],
        "locate": ["locate", "-q", "test pattern", "-v", str(synthetic_video),
                   "--backend", "mock", "-o", str(out / "locate")],
        "caption-compare": ["caption-compare", str(captions), "-o", str(out / "caption-compare")],
    }
    for name, args in commands.items():
        first = _mm(tmp_path, *args)
        assert first.exit_code == 0, (name, first.output)
        written = _tree(out / name)
        again = _mm(tmp_path, *args)
        assert again.exit_code == 1, (name, again.output)
        assert "the name was taken" in again.output, (name, again.output)
        assert _tree(out / name) == written, name


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
    for args in (["index", str(src), "--backend", "mock", "--no-asr", "--no-estimate"],
                 ["search", "test pattern", "--backend", "mock", "-k", "1", "-o", str(src)]):
        assert _mm(tmp_path, *args).exit_code == 0
    cut = ["cut", str(src / "test_video-test-pattern.llc"), "-o", str(src)]
    first = _mm(tmp_path, *cut)
    assert first.exit_code == 0, first.output
    [clip] = src.glob("01_*")
    assert re.fullmatch(r"01_test_video-\d\d\.\d\d\.\d\d\.\d{3}-\d\d\.\d\d\.\d\d\.\d{3}\.mp4",
                        clip.name), clip.name
    clip_before = (_sha1(clip), clip.stat().st_mtime_ns)

    again = _mm(tmp_path, *cut)
    assert again.exit_code == 1
    assert f"not written, the name was taken: {clip}" in again.output
    assert (_sha1(clip), clip.stat().st_mtime_ns) == clip_before
    assert (_sha1(synthetic_video), synthetic_video.stat().st_mtime_ns) == original


@requires_ffmpeg
def test_an_index_never_picks_up_the_tools_own_clips(tmp_path, synthetic_video):
    src = synthetic_video.parent
    data = src / "mm_data"
    index = ["index", str(src), "--backend", "mock", "--no-asr", "--no-estimate"]
    for args in (index,
                 ["search", "test pattern", "--backend", "mock", "-k", "1", "-o", str(src)],
                 ["cut", str(src / "test_video-test-pattern.llc"), "-o", str(src)],
                 index):
        result = CliRunner().invoke(main, ["--data-dir", str(data), *args])
        assert result.exit_code == 0, (args, result.output)
    assert len(list(src.rglob("*.mp4"))) == 2
    conn = sqlite3.connect(data / "manifest.db")
    assert [r[0] for r in conn.execute("SELECT path FROM videos")] == [str(synthetic_video)]
