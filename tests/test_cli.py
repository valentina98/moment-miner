import csv

from click.testing import CliRunner

from moment_miner.cli import main

from .conftest import requires_ffmpeg


def test_annotate_from_template_and_append(tmp_path):
    folder = tmp_path / "footage"
    folder.mkdir()
    (folder / "run1.mp4").write_bytes(b"x")
    (folder / "run2.mp4").write_bytes(b"x")

    result = CliRunner().invoke(main, [
        "annotate", str(folder), "--template", "parkour",
    ], input="kong vault over an obstacle\n2\n0:04\n0:09\n\n")
    assert result.exit_code == 0, result.output
    assert "created" in result.output and "1 label(s) appended" in result.output

    rows = list(csv.DictReader((folder / "labels.csv").open()))
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
    (folder / "labels.csv").write_text("query,video,start,end\n")

    result = CliRunner().invoke(main, [
        "annotate", str(folder),
    ], input="backflip\n1\n0:02\n0:05\n\n")
    assert result.exit_code == 0, result.output

    rows = list(csv.DictReader((folder / "labels.csv").open()))
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
