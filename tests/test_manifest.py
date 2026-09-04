import os

from moment_miner.manifest import Manifest, fingerprint


def _fake_video(path, content=b"x" * 4096):
    path.write_bytes(content)


def test_scan_new_unchanged_changed(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    _fake_video(archive / "a.mp4")
    (archive / "sub").mkdir()
    _fake_video(archive / "sub" / "b.mov")
    (archive / "notes.txt").write_text("not a video")

    m = Manifest(tmp_path / "mm.db")
    assert m.scan(archive) == {"new": 2, "changed": 0, "unchanged": 0}
    assert m.scan(archive) == {"new": 0, "changed": 0, "unchanged": 2}

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
    assert m.scan(archive) == {"new": 1, "changed": 0, "unchanged": 0}


def test_reset_marks_pending(tmp_path):
    archive = tmp_path / "archive"
    archive.mkdir()
    _fake_video(archive / "a.mp4")
    m = Manifest(tmp_path / "mm.db")
    m.scan(archive)
    row = m.pending()[0]
    m.mark_done(row["path"], 10.0, "mock")
    assert m.pending() == []
    m.reset(archive)
    assert len(m.pending()) == 1


def test_transcript_window(tmp_path):
    m = Manifest(tmp_path / "mm.db")
    m.add_transcript("vid1", [(0.0, 2.0, "hello"), (5.0, 7.0, "kong vault"), (9.0, 11.0, "bye")])
    assert m.transcript_between("vid1", 4.0, 8.0) == "kong vault"
    assert m.transcript_between("vid1", 0.0, 12.0) == "hello kong vault bye"
    assert m.transcript_between("vid1", 2.5, 4.5) == ""
