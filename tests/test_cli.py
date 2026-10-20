import csv
import json

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
    assert top.exit_code == 0 and "cut" in top.output and "annotate" in top.output
    assert {"mine", "export"}.isdisjoint(main.list_commands(None))

    sub = runner.invoke(main, ["help", "search"])
    assert sub.exit_code == 0 and "mm search" in sub.output and "-k" in sub.output

    bad = runner.invoke(main, ["help", "nope"])
    assert bad.exit_code != 0 and "unknown command" in bad.output

    ver = runner.invoke(main, ["--version"])
    assert ver.exit_code == 0 and "mm" in ver.output


def test_annotate_help_lists_templates():
    result = CliRunner().invoke(main, ["annotate", "--help"])
    assert result.exit_code == 0
    assert "parkour" in result.output and "wedding" in result.output


def test_annotate_unknown_template(tmp_path):
    (tmp_path / "footage").mkdir()
    result = CliRunner().invoke(main, ["--data-dir", str(tmp_path / "data"), "annotate",
                                       str(tmp_path / "footage"), "--template", "nope"])
    assert result.exit_code != 0
    assert "unknown template" in result.output


def _mm(data_dir, *args):
    return CliRunner().invoke(main, ["--data-dir", str(data_dir), *args])


def _indexed(tmp_path, synthetic_video):
    data_dir = tmp_path / "mm_data"
    result = _mm(data_dir, "index", str(synthetic_video.parent), "--backend", "mock",
                 "--no-asr", "--no-estimate")
    assert result.exit_code == 0, result.output
    return data_dir


@requires_ffmpeg
def test_search_writes_a_project_per_video_and_cut_cuts_them(tmp_path, synthetic_video):
    import subprocess

    import json5

    # Two videos: search merges overlapping hits, so one 10 s video gives one.
    # Not a copy: a video's id comes from its bytes.
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "testsrc2=duration=10:size=320x240:rate=15", "-c:v", "libx264",
                    "-preset", "ultrafast", str(synthetic_video.with_name("other.mp4"))],
                   check=True)
    data_dir = _indexed(tmp_path, synthetic_video)
    found = _mm(data_dir, "search", "test pattern", "-k", "2", "--backend", "mock",
                "-o", str(tmp_path / "projects"))
    assert found.exit_code == 0, found.output
    projects = sorted((tmp_path / "projects").iterdir())
    assert [p.name for p in projects] == ["other-test-pattern.llc",
                                          "test_video-test-pattern.llc"]
    assert all(f"project: {p}" in found.output for p in projects)
    names = [s["name"] for p in projects for s in json5.loads(p.read_text())["cutSegments"]]
    assert sorted(names) == ["test pattern? #1", "test pattern? #2"]

    cut = _mm(data_dir, "cut", *map(str, projects), "-o", str(tmp_path / "clips"))
    assert cut.exit_code == 0, cut.output
    clips = sorted(p.name for p in (tmp_path / "clips").iterdir())
    assert [c[:3] for c in clips] == ["01_", "02_"]
    assert all(c.endswith(".mp4") for c in clips)
    assert f"<- {synthetic_video}" in cut.output


@requires_ffmpeg
def test_search_widens_each_hit_by_pad_inside_the_video(tmp_path, synthetic_video):
    data_dir = _indexed(tmp_path, synthetic_video)

    def hits(pad):
        result = _mm(data_dir, "search", "test pattern", "-k", "5", "--backend", "mock",
                     "--json", "--pad", str(pad), "-o", str(tmp_path / f"pad{pad}"))
        assert result.exit_code == 0, result.output
        return sorted((h["t0"], h["t1"]) for h in json.loads(result.stdout))

    from moment_miner.probe import probe

    duration = probe(str(synthetic_video))["duration"]
    exact, padded = hits(0), hits(1)
    assert any(t0 < 1 or t1 + 1 > duration for t0, t1 in exact), "no end to clamp"
    assert padded == [(max(0.0, t0 - 1), min(duration, t1 + 1)) for t0, t1 in exact]


@requires_ffmpeg
def test_cut_without_output_writes_a_run_folder_named_after_the_project(tmp_path,
                                                                       synthetic_video):
    data_dir = _indexed(tmp_path, synthetic_video)
    assert _mm(data_dir, "search", "test pattern", "-k", "1", "--backend", "mock",
               "-o", str(tmp_path / "p")).exit_code == 0
    result = _mm(data_dir, "cut", str(tmp_path / "p" / "test_video-test-pattern.llc"))
    assert result.exit_code == 0, result.output
    [run] = (data_dir / "cuts").iterdir()
    assert run.name.startswith("test_video-test-pattern_")
    assert [p.name[:3] for p in run.iterdir()] == ["01_"]


@requires_ffmpeg
def test_search_without_output_writes_a_run_folder_named_after_the_query(tmp_path,
                                                                        synthetic_video):
    data_dir = _indexed(tmp_path, synthetic_video)
    result = _mm(data_dir, "search", "Test pattern?", "-k", "1", "--backend", "mock")
    assert result.exit_code == 0, result.output
    [run] = (data_dir / "cuts").iterdir()
    assert run.name.startswith("test-pattern_")
    assert [p.name for p in run.iterdir()] == ["test_video-test-pattern.llc"]


@requires_ffmpeg
def test_a_reindex_keeps_the_stored_captions(tmp_path, synthetic_video):
    from moment_miner.manifest import Manifest
    from moment_miner.store import AxisStore, SegmentStore

    data_dir = _indexed(tmp_path, synthetic_video)
    axes = {"action": "kong vault", "who": "one person", "scene": "park", "light": "sunny"}
    AxisStore(data_dir).add([{"id": s["id"], "path": s["path"], "t0": s["t0"], "t1": s["t1"], **axes}
                             for s in SegmentStore(data_dir, "mock").segments()])
    Manifest(data_dir / "manifest.db").reset(synthetic_video.parent)

    result = _mm(data_dir, "index", str(synthetic_video.parent), "--backend", "mock",
                 "--no-asr", "--no-estimate")
    assert result.exit_code == 0, result.output
    assert "indexed': 1" in result.output
    texts = {s["text"] for s in SegmentStore(data_dir, "mock").segments()}
    assert texts == {"kong vault, one person, park, sunny"}


# cli spec rule 2: one name per option. These are the names it replaced.
RETIRED = {"--out", "--llc-dir", "--llc", "--top", "--caption"}
WRITES_RESULTS = {"search", "cut", "caption-compare", "locate"}


def test_every_option_has_one_name_for_one_thing():
    import click

    ctx = click.Context(main)
    for name in main.list_commands(ctx):
        cmd = main.get_command(ctx, name)
        opts = {p.name: p.opts for p in cmd.params if isinstance(p, click.Option)}
        used = {o for names in opts.values() for o in names}
        assert not used & RETIRED, (name, used & RETIRED)
        for param, names in opts.items():
            if param == "output":
                assert names == ["-o", "--output"], (name, names)
            if param in ("k", "top_k"):
                assert names == ["-k"], (name, names)
            if "labels" in param:
                assert names == ["--labels"], (name, names)
        if name in WRITES_RESULTS:
            assert opts.get("output") == ["-o", "--output"], name
        else:
            assert "output" not in opts, name


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
    _forget_missing(m, counts, tmp_path / "data", folder)
    assert counts["removed"] == 1
    assert [r["path"] for r in m.conn.execute("SELECT path FROM videos")] == [
        str((folder / "b.mp4").resolve())]


def test_every_video_missing_looks_like_a_new_mount_and_removes_nothing(tmp_path):
    from moment_miner.cli import _forget_missing
    m, folder = _two_indexed(tmp_path)
    for p in folder.iterdir():
        p.unlink()
    counts = m.scan(folder)
    _forget_missing(m, counts, tmp_path / "data", folder)
    assert counts["removed"] == 0
    assert m.conn.execute("SELECT count(*) FROM videos").fetchone()[0] == 2


@requires_ffmpeg
def _segments_for_two_models(data_dir, m):
    """A row per indexed video in two models' tables, as two backends leave them."""
    from moment_miner.store import SegmentStore
    for backend, dim in (("mock", 4), ("other", 8)):
        SegmentStore(data_dir, backend).add([
            {"id": f"{row['id']}:0.0", "video_id": row["id"], "path": row["path"],
             "t0": 0.0, "t1": 8.0, "text": "", "vector": [1.0] + [0.0] * (dim - 1),
             "motion": 0.0}
            for row in m.conn.execute("SELECT id, path FROM videos")])


def _paths_in(data_dir, backend):
    from moment_miner.store import SegmentStore
    return sorted(r["path"] for r in SegmentStore(data_dir, backend).segments())


def test_a_deleted_video_leaves_every_models_table(tmp_path):
    from moment_miner.cli import _forget_missing
    m, folder = _two_indexed(tmp_path)
    _segments_for_two_models(tmp_path / "data", m)
    (folder / "a.mp4").unlink()
    _forget_missing(m, m.scan(folder), tmp_path / "data", folder)
    kept = [str((folder / "b.mp4").resolve())]
    assert _paths_in(tmp_path / "data", "mock") == kept
    assert _paths_in(tmp_path / "data", "other") == kept


def test_prune_asks_and_removes_what_index_keeps(tmp_path):
    m, folder = _two_indexed(tmp_path)
    data_dir = tmp_path / "data"
    _segments_for_two_models(data_dir, m)
    for p in folder.iterdir():
        p.unlink()
    no = CliRunner().invoke(main, ["--data-dir", str(data_dir), "prune"], input="n\n")
    assert no.exit_code == 1 and "every indexed video is missing" in no.output
    assert len(_paths_in(data_dir, "other")) == 2
    yes = CliRunner().invoke(main, ["--data-dir", str(data_dir), "prune"], input="y\n")
    assert yes.exit_code == 0, yes.output
    assert "removed: 2 video(s)" in yes.output
    assert _paths_in(data_dir, "mock") == _paths_in(data_dir, "other") == []
    assert m.conn.execute("SELECT count(*) FROM videos").fetchone()[0] == 0


def test_prune_keeps_what_is_on_disk_and_stays_inside_its_folder(tmp_path):
    m, folder = _two_indexed(tmp_path)
    data_dir = tmp_path / "data"
    other = tmp_path / "elsewhere"
    other.mkdir()
    (other / "c.mp4").write_bytes(b"z" * 64)
    m.scan(other)
    _segments_for_two_models(data_dir, m)
    (folder / "a.mp4").unlink()
    (other / "c.mp4").unlink()
    result = CliRunner().invoke(main, ["--data-dir", str(data_dir), "prune", str(folder), "--yes"])
    assert result.exit_code == 0, result.output
    assert _paths_in(data_dir, "other") == sorted(
        [str((folder / "b.mp4").resolve()), str((other / "c.mp4").resolve())])
    again = CliRunner().invoke(main, ["--data-dir", str(data_dir), "prune", str(folder)])
    assert "nothing to prune" in again.output


def test_eval_scores_the_labels_in_the_data_folder_and_names_the_file(tmp_path, synthetic_video):
    data_dir = tmp_path / "mm_data"

    def run(*args):
        return CliRunner().invoke(main, ["--data-dir", str(data_dir), *args])

    assert run("eval", "--backend", "mock").exit_code != 0
    assert run("index", str(synthetic_video.parent), "--backend", "mock", "--no-asr",
               "--no-estimate").exit_code == 0
    (data_dir / "labels.csv").write_text(
        f"query,video,start,end\ntest pattern,{synthetic_video.name},0,4\n")
    result = run("eval", "--backend", "mock")
    assert result.exit_code == 0, result.output
    assert f"labels: {data_dir / 'labels.csv'}" in result.output
    assert "(1 labels)" in result.output
