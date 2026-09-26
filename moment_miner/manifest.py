import hashlib
import os
import sqlite3
import time
from pathlib import Path

VIDEO_EXTS = {
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mts", ".m2ts",
    ".wmv", ".flv", ".ts", ".3gp", ".mpg", ".mpeg",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    path TEXT PRIMARY KEY,
    id TEXT NOT NULL,
    size INTEGER NOT NULL,
    mtime REAL NOT NULL,
    duration REAL,
    status TEXT NOT NULL DEFAULT 'pending',
    error TEXT,
    backend TEXT,
    indexed_at REAL
);
CREATE INDEX IF NOT EXISTS idx_videos_id ON videos(id);
CREATE TABLE IF NOT EXISTS transcripts (
    video_id TEXT NOT NULL,
    start REAL NOT NULL,
    end REAL NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_transcripts_video ON transcripts(video_id);
CREATE TABLE IF NOT EXISTS index_settings (
    path TEXT PRIMARY KEY,
    win REAL NOT NULL,
    stride REAL NOT NULL,
    frames_per_window INTEGER NOT NULL,
    frame_w INTEGER NOT NULL,
    frame_h INTEGER NOT NULL,
    backend TEXT NOT NULL,
    indexed_at REAL NOT NULL
);
"""

# Every one of these feeds the segment geometry, and segment ids are
# `{video_id}:{t0}`, so changing any of them shifts every id a video would
# produce while its old rows stay in the table answering queries.
SETTINGS_COLUMNS = ("win", "stride", "frames_per_window", "frame_w", "frame_h", "backend")


def _settings_match(row: sqlite3.Row | None, settings: dict) -> bool:
    # No row at all means an index built before this table existed. Re-index
    # rather than guess what produced it -- guessing is what let two geometry
    # changes ship unnoticed in three days.
    return row is not None and all(row[c] == settings[c] for c in SETTINGS_COLUMNS)


def _prefix(folder: str | Path) -> str:
    return str(Path(folder).resolve()).rstrip("/") + "/"


def fingerprint(path: str, chunk: int = 1 << 20) -> str:
    # Hash size + first/last MB instead of whole file: archives are TBs.
    st = os.stat(path)
    h = hashlib.sha1(str(st.st_size).encode())
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if st.st_size > 2 * chunk:
            f.seek(-chunk, os.SEEK_END)
            h.update(f.read(chunk))
    return h.hexdigest()[:16]


class Manifest:
    def __init__(self, db_path: str | Path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    def scan(self, folder: str | Path) -> dict:
        """Register new/changed videos under folder. Returns counts."""
        counts = {"new": 0, "changed": 0, "unchanged": 0}
        for p in sorted(Path(folder).resolve().rglob("*")):
            if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
                continue
            # `mm mine` exports into *_mined folders; never re-ingest them.
            if any(part.endswith("_mined") for part in p.parent.parts):
                continue
            st = p.stat()
            row = self.conn.execute(
                "SELECT id, size, mtime FROM videos WHERE path = ?", (str(p),)
            ).fetchone()
            if row and row["size"] == st.st_size and row["mtime"] == st.st_mtime:
                counts["unchanged"] += 1
                continue
            vid = fingerprint(str(p))
            with self.conn:
                if row:
                    self.conn.execute(
                        "DELETE FROM transcripts WHERE video_id = ?", (row["id"],)
                    )
                    counts["changed"] += 1
                else:
                    counts["new"] += 1
                self.conn.execute(
                    "INSERT OR REPLACE INTO videos (path, id, size, mtime, status)"
                    " VALUES (?, ?, ?, ?, 'pending')",
                    (str(p), vid, st.st_size, st.st_mtime),
                )
        return counts

    def reset(self, folder: str | Path):
        """Mark all videos under folder pending so they re-index."""
        prefix = _prefix(folder)
        with self.conn:
            self.conn.execute(
                "UPDATE videos SET status='pending' WHERE path LIKE ? ESCAPE '\\'",
                (prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%",),
            )

    def pending(
        self, settings: dict | None = None, folder: str | Path | None = None
    ) -> list[sqlite3.Row]:
        """Videos needing work: new, changed, or indexed under other settings.

        `settings` is what the caller is about to index with; a done video
        whose recorded row differs comes back too, so a geometry change
        re-indexes without `--reindex`. Omit it and only new/changed videos
        come back, which is what a caller that is not about to index wants.
        Errored videos stay out either way -- they need `--reindex`, the same
        as before. `folder` limits the answer to videos under it, so indexing
        one folder never touches another.
        """
        prefix = None if folder is None else _prefix(folder)
        if settings is None:
            return [
                row for row in self.conn.execute(
                    "SELECT * FROM videos WHERE status = 'pending' ORDER BY path")
                if prefix is None or row["path"].startswith(prefix)
            ]
        recorded = {
            r["path"]: r
            for r in self.conn.execute("SELECT * FROM index_settings")
        }
        return [
            row
            for row in self.conn.execute(
                "SELECT * FROM videos WHERE status IN ('pending', 'done')"
                " ORDER BY path"
            )
            if (prefix is None or row["path"].startswith(prefix))
            and (row["status"] == "pending"
                 or not _settings_match(recorded.get(row["path"]), settings))
        ]

    def why_pending(self, row: sqlite3.Row, settings: dict) -> str:
        """Why `pending(settings)` returned `row`, for the index log."""
        if row["status"] == "pending":
            return "new or changed file"
        recorded = self.recorded_settings(row["path"])
        if recorded is None:
            return "no recorded settings"
        return "settings differ: " + ", ".join(
            f"{c} {recorded[c]} -> {settings[c]}"
            for c in SETTINGS_COLUMNS if recorded[c] != settings[c])

    def recorded_settings(self, path: str) -> sqlite3.Row | None:
        """What the last index of `path` was built under, or None if unknown."""
        return self.conn.execute(
            "SELECT * FROM index_settings WHERE path=?", (path,)
        ).fetchone()

    def mark_done(self, path: str, duration: float, settings: dict):
        now = time.time()
        with self.conn:
            self.conn.execute(
                "UPDATE videos SET status='done', duration=?, backend=?,"
                " indexed_at=?, error=NULL WHERE path=?",
                (duration, settings["backend"], now, path),
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO index_settings"
                f" (path, {', '.join(SETTINGS_COLUMNS)}, indexed_at)"
                f" VALUES ({', '.join('?' * (len(SETTINGS_COLUMNS) + 2))})",
                (path, *(settings[c] for c in SETTINGS_COLUMNS), now),
            )

    def mark_error(self, path: str, error: str):
        with self.conn:
            self.conn.execute(
                "UPDATE videos SET status='error', error=? WHERE path=?", (error, path)
            )

    def add_transcript(self, vid: str, segments: list[tuple[float, float, str]]):
        with self.conn:
            self.conn.execute("DELETE FROM transcripts WHERE video_id=?", (vid,))
            self.conn.executemany(
                "INSERT INTO transcripts (video_id, start, end, text) VALUES (?,?,?,?)",
                [(vid, s, e, t) for s, e, t in segments],
            )

    def transcript_between(self, vid: str, t0: float, t1: float) -> str:
        rows = self.conn.execute(
            "SELECT text FROM transcripts WHERE video_id=? AND end>? AND start<?"
            " ORDER BY start",
            (vid, t0, t1),
        ).fetchall()
        return " ".join(r["text"].strip() for r in rows)

    def stats(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT status, COUNT(*) n, COALESCE(SUM(duration),0) dur"
            " FROM videos GROUP BY status"
        ).fetchall()

    def errors(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT path, error FROM videos WHERE status='error' ORDER BY path"
        ).fetchall()

    def video(self, vid: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM videos WHERE id=? OR path=?", (vid, vid)
        ).fetchone()
