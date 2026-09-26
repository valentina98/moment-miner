import os

from moment_miner.manifest import Manifest, fingerprint


def _fake_video(path, content=b"x" * 4096):
    path.write_bytes(content)


SETTINGS = {"win": 8.0, "stride": 4.0, "frames_per_window": 8,
            "frame_w": 456, "frame_h": 256, "backend": "mock"}


def _indexed(tmp_path, settings=SETTINGS):
    """One video, scanned and marked done under `settings`."""
    archive = tmp_path / "archive"
    archive.mkdir(parents=True)
    _fake_video(archive / "a.mp4")
    m = Manifest(tmp_path / "mm.db")
    m.scan(archive)
    path = m.pending()[0]["path"]
    m.mark_done(path, 10.0, settings)
    return m, path


def test_scan_new_unchanged_changed(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    _fake_video(archive / "a.mp4")
    (archive / "sub").mkdir()
    _fake_video(archive / "sub" / "b.mov")
    (archive / "notes.txt").write_text("not a video")

    m = Manifest(tmp_path / "mm.db")
    assert m.scan(archive) == {"new": 2, "changed": 0, "unchanged": 0, "missing": []}
    assert m.scan(archive) == {"new": 0, "changed": 0, "unchanged": 2, "missing": []}

    _fake_video(archive / "a.mp4", b"y" * 8192)
    os.utime(archive / "a.mp4", (1, 1))
    counts = m.scan(archive)
    assert counts["changed"] == 1 and counts["unchanged"] == 1
    assert len(m.pending()) == 2


def test_fingerprint_stable_and_content_sensitive(tmp_path):
    f = tmp_path / "v.mp4"
    _fake_video(f, b"a" * (3 << 20))
    fp1 = fingerprint(str(f))
    assert fingerprint(str(f)) == fp1
    _fake_video(f, b"a" * ((3 << 20) - 1) + b"b")
    assert fingerprint(str(f)) != fp1


def test_scan_skips_mined_folders(tmp_path):
    archive = tmp_path / "archive"
    (archive / "sub_mined" / "run").mkdir(parents=True)
    _fake_video(archive / "real.mp4")
    _fake_video(archive / "sub_mined" / "run" / "clip.mp4")
    m = Manifest(tmp_path / "mm.db")
    assert m.scan(archive) == {"new": 1, "changed": 0, "unchanged": 0, "missing": []}


def test_reset_marks_pending(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    _fake_video(archive / "a.mp4")
    m = Manifest(tmp_path / "mm.db")
    m.scan(archive)
    row = m.pending()[0]
    m.mark_done(row["path"], 10.0, SETTINGS)
    assert m.pending() == []
    m.reset(archive)
    assert len(m.pending()) == 1


def test_transcript_window(tmp_path):
    m = Manifest(tmp_path / "mm.db")
    m.add_transcript("vid1", [(0.0, 2.0, "hello"), (5.0, 7.0, "kong vault"), (9.0, 11.0, "bye")])
    assert m.transcript_between("vid1", 4.0, 8.0) == "kong vault"
    assert m.transcript_between("vid1", 0.0, 12.0) == "hello kong vault bye"
    assert m.transcript_between("vid1", 2.5, 4.5) == ""


def test_mark_done_records_the_settings_it_indexed_under(tmp_path):
    m, path = _indexed(tmp_path)
    row = m.recorded_settings(path)
    assert {k: row[k] for k in SETTINGS} == SETTINGS
    assert row["indexed_at"] > 0


def test_unchanged_settings_skip_a_reindex(tmp_path):
    m, _ = _indexed(tmp_path)
    assert m.pending(SETTINGS) == []


def test_any_changed_setting_forces_a_reindex(tmp_path):
    """Segment ids are `{video_id}:{t0}`, so each of these shifts every id the
    video would produce while the old rows keep answering queries."""
    changes = {"win": 12.0, "stride": 2.0, "frames_per_window": 16,
               "frame_w": 224, "frame_h": 224, "backend": "siglip"}
    assert set(changes) == set(SETTINGS), "a setting is recorded but not compared"
    for key, value in changes.items():
        m, path = _indexed(tmp_path / key)
        assert [r["path"] for r in m.pending({**SETTINGS, key: value})] == [path], key


def test_an_index_with_no_recorded_settings_needs_indexing(tmp_path):
    """What every index built before this table existed looks like: done, with
    nothing saying what produced it."""
    m, path = _indexed(tmp_path)
    with m.conn:
        m.conn.execute("DELETE FROM index_settings")
    assert m.recorded_settings(path) is None
    assert [r["path"] for r in m.pending(SETTINGS)] == [path]


def test_an_errored_video_stays_out_of_pending(tmp_path):
    """It has no settings row either, but it needs --reindex, not a retry on
    every run."""
    m, path = _indexed(tmp_path)
    m.mark_error(path, "RuntimeError: no readable windows")
    assert m.pending(SETTINGS) == [] and m.pending() == []


def test_pending_in_a_folder_leaves_other_folders_alone(tmp_path):
    """Adding footage in a new folder must not re-index the ones already done,
    even when those are stale."""
    m, old = _indexed(tmp_path)
    new = tmp_path / "new"
    new.mkdir()
    _fake_video(new / "b.mp4")
    m.scan(new)
    stale = {**SETTINGS, "stride": 2.0}
    assert [r["path"] for r in m.pending(stale, new)] == [str((new / "b.mp4").resolve())]
    assert old in [r["path"] for r in m.pending(stale)]


def test_why_pending_names_the_reason(tmp_path):
    m, path = _indexed(tmp_path)
    row = m.pending({**SETTINGS, "stride": 2.0})[0]
    assert m.why_pending(row, {**SETTINGS, "stride": 2.0}) == "settings differ: stride 4.0 -> 2.0"
    with m.conn:
        m.conn.execute("DELETE FROM index_settings")
    assert m.why_pending(m.pending(SETTINGS)[0], SETTINGS) == "no recorded settings"
    m.reset(tmp_path / "archive")
    assert m.why_pending(m.pending(SETTINGS)[0], SETTINGS) == "new or changed file"


def test_scan_lists_a_deleted_video_and_forget_drops_it(tmp_path):
    m, path = _indexed(tmp_path)
    _fake_video(tmp_path / "archive" / "b.mp4")
    os.remove(path)
    assert m.scan(tmp_path / "archive")["missing"] == [path]
    m.forget(path)
    assert m.conn.execute("SELECT 1 FROM videos WHERE path = ?", (path,)).fetchone() is None
    assert m.recorded_settings(path) is None
    assert m.scan(tmp_path / "archive")["missing"] == []
