from pathlib import Path

import lancedb
import numpy as np
from lancedb.index import FTS
from lancedb.query import MatchQuery
from lancedb.rerankers import RRFReranker

from .guard import writable

COLUMNS = ["id", "video_id", "path", "t0", "t1", "text"]
AXIS_COLUMNS = ["action", "who", "scene", "light"]


class SegmentStore:
    """One LanceDB table per embedding backend (dimensions differ)."""

    def __init__(self, data_dir: str | Path, backend_name: str):
        self.db = lancedb.connect(str(writable(Path(data_dir) / "lance")))
        self.table_name = f"segments_{backend_name}"

    def _names(self) -> list[str]:
        return list(self.db.list_tables().tables)

    @staticmethod
    def backends(data_dir: str | Path) -> list[str]:
        """Every backend with a table here, so a removal reaches each model's rows."""
        db = lancedb.connect(str(writable(Path(data_dir) / "lance")))
        return [n[len("segments_"):] for n in db.list_tables().tables
                if n.startswith("segments_")]

    @property
    def table(self):
        if self.table_name not in self._names():
            raise RuntimeError(
                f"no index for backend {self.table_name!r};"
                f" available: {self._names()} — run `mm index` first"
            )
        return self.db.open_table(self.table_name)

    def add(self, rows: list[dict]):
        if not rows:
            return
        for r in rows:
            r["vector"] = np.asarray(r["vector"], dtype=np.float32)
        if self.table_name in self._names():
            # The same file reached under two mount roots fingerprints alike, so
            # its rows share an id and search would return the moment twice.
            (
                self.db.open_table(self.table_name)
                .merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(rows)
            )
        else:
            self.db.create_table(self.table_name, rows)

    def delete_path(self, path: str):
        if self.table_name in self._names():
            self.table.delete("path = '{}'".format(path.replace("'", "''")))

    def segments(self) -> list[dict]:
        """Every row without its vector, for passes that only need the spans."""
        return (
            self.table.search().select(COLUMNS)
            .limit(max(self.count(), 1)).to_list()
        )

    def set_text(self, seg_id: str, text: str):
        self.table.update(where="id = '{}'".format(seg_id.replace("'", "''")),
                          values={"text": text})

    def rebuild_fts(self):
        self.table.create_index("text", config=FTS(), replace=True)

    def search(self, vec: np.ndarray, query: str, k: int = 50,
               where: str | None = None) -> list[dict]:
        """The `k` best of the vector and full-text rankings fused by
        reciprocal rank (K=60), as `score`. `where` filters both rankings
        before they are cut to `k`."""
        q = (
            self.table.search(query_type="hybrid")
            .vector(np.asarray(vec, dtype=np.float32))
            # Matches the words. A plain string would be parsed, so a quoted
            # phrase fails for want of token positions, and it needs a
            # full-text index; this scans rows no index covers yet.
            .text(MatchQuery(query, "text"))
            .rerank(RRFReranker())
            .limit(k)
        )
        if where:
            q = q.where(where)
        # No select(): naming the columns makes lance warn on every query that
        # the rankings' own score columns were left out.
        return [dict({c: r[c] for c in COLUMNS}, score=r["_relevance_score"])
                for r in q.to_list()]

    def count(self) -> int:
        if self.table_name not in self._names():
            return 0
        return self.table.count_rows()


class AxisStore:
    """The caption's axes, one row per segment.

    Its own table: the axes are a property of the captions, not of the
    embedding, so keeping them in `segments_<backend>` would store them once
    per backend and make adding an axis rewrite the table holding the
    vectors. Here an axis can be added,
    recomputed or dropped on its own, and no re-index is needed to gain one.

    The joined caption still goes into the segment row's `text`, because that
    is what the full-text index reads. This table is what a per-axis query
    needs: a search constrains the axes it names, and `who = nobody` is a
    filter, not a phrase to match.
    """

    TABLE = "axes"

    def __init__(self, data_dir: str | Path):
        self.db = lancedb.connect(str(writable(Path(data_dir) / "lance")))

    def _exists(self) -> bool:
        return self.TABLE in list(self.db.list_tables().tables)

    def add(self, rows: list[dict]):
        if not rows:
            return
        if self._exists():
            (
                self.db.open_table(self.TABLE)
                .merge_insert("id")
                .when_matched_update_all()
                .when_not_matched_insert_all()
                .execute(rows)
            )
        else:
            self.db.create_table(self.TABLE, rows)

    def delete_path(self, path: str):
        if self._exists():
            self.db.open_table(self.TABLE).delete(
                "path = '{}'".format(path.replace("'", "''")))

    def all(self) -> list[dict]:
        if not self._exists():
            return []
        return self.db.open_table(self.TABLE).search().limit(0).to_list()

    def get(self, ids: list[str]) -> dict[str, dict]:
        """Axes by segment id. Ids with no row are simply absent."""
        if not ids or not self._exists():
            return {}
        quoted = ", ".join("'{}'".format(i.replace("'", "''")) for i in ids)
        rows = (
            self.db.open_table(self.TABLE)
            .search().where(f"id IN ({quoted})").limit(len(ids)).to_list()
        )
        return {r["id"]: {k: r[k] for k in AXIS_COLUMNS} for r in rows}

