"""index-export spec 5b, the safety test: after index, caption, cut, prune and
an interrupted run, every source file keeps its checksum and modification
time, and nothing outside the data and output folders was created, changed or
deleted. "Outside" is everything under the test's root, the home and working
folders the commands run with included."""
import hashlib
import subprocess

from click.testing import CliRunner

from moment_miner import indexer
from moment_miner.cli import main

from .conftest import requires_ffmpeg


def _video(path, source):
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    f"{source}=duration=10:size=320x240:rate=15", "-c:v", "libx264",
                    "-preset", "ultrafast", "-g", "30", str(path)], check=True)


def _world(root, *written):
    """Every path under `root` but the `written` folders, by content and mtime."""
    state = {}
    for p in [root, *root.rglob("*")]:
        if any(p.is_relative_to(w) for w in written):
            continue
        content = hashlib.sha1(p.read_bytes()).hexdigest() if p.is_file() else None
        state[p] = (content, p.stat().st_mtime_ns)
    return state


@requires_ffmpeg
def test_no_command_changes_anything_but_the_data_and_output_folders(
        tmp_path, synthetic_video, monkeypatch):
    src = synthetic_video.parent
    _video(src / "second.mp4", "testsrc2")
    _video(src / "gone.mp4", "smptebars")
    (src / "notes.txt").write_text("not a video")
    data, out = tmp_path / "data", tmp_path / "out"
    for folder in (data, out, tmp_path / "home", tmp_path / "cwd"):
        folder.mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path / "cwd")

    def mm(*args):
        return CliRunner().invoke(main, ["--data-dir", str(data), *args])

    def unchanged(since, *args):
        result = mm(*args)
        assert result.exit_code == 0, (args, result.output)
        assert _world(tmp_path, data, out) == since, args
        return result

    index = ["index", str(src), "--backend", "mock", "--no-asr"]
    before = _world(tmp_path, data, out)
    unchanged(before, *index)
    unchanged(before, "caption", str(src), "--model", "mock", "--backend", "mock")

    (src / "gone.mp4").unlink()  # the user deletes a video
    before = _world(tmp_path, data, out)
    assert "removed: " in unchanged(before, *index).output

    unchanged(before, "search", "test pattern", "--backend", "mock", "-k", "2",
              "-o", str(out))
    unchanged(before, "cut", *map(str, out.glob("*.llc")), "-o", str(out))
    assert list(out.glob("01_*.mp4"))

    probe, calls = indexer.probe, []

    def ctrl_c_on_the_second_video(path):
        calls.append(path)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return probe(path)

    monkeypatch.setattr(indexer, "probe", ctrl_c_on_the_second_video)
    interrupted = mm(*index, "--reindex", "--no-estimate")
    assert interrupted.exit_code != 0 and "[2/2]" not in interrupted.output
    assert _world(tmp_path, data, out) == before
    monkeypatch.setattr(indexer, "probe", probe)
    resumed = unchanged(before, *index, "--no-estimate")
    assert "[1/1]" in resumed.output, resumed.output
