from pathlib import Path

import lancedb
import numpy as np
from lancedb.index import FTS
from lancedb.query import MatchQuery
from lancedb.rerankers import RRFReranker

from .guard import writable

COLUMNS = ["id", "video_id", "path", "t0", "t1", "text"]
AXIS_COLUMNS = ["action", "who", "scene", "light"]


class _InSegmentOrder(RRFReranker):
    """Reciprocal-rank fusion of the `k` best of each ranking, in which rows
    with equal scores rank by video, then by start time.

    Segments of one scene often share a caption word for word, so their BM25
    scores are equal, and the order lance returns equal scores in follows how
    the rows are stored: the ranking would change with a re-index of unchanged
    footage. Lance cuts each ranking to the query's limit before it reaches a
    reranker, so the query asks for more than `k` and the cut to `k` is made
    here, after the ordering, where it cannot split a tied group by storage.
    """

    # shortcut: a group of tied rows that reaches past OVERFETCH * k is still
    # cut by storage order; raise it if such a group shows up (dozens of
    # segments sharing one caption were the most seen, at k = 50).
    OVERFETCH = 4
    TIES = [("video_id", "ascending"), ("t0", "ascending")]

    def __init__(self, k: int, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def rerank_hybrid(self, query, vector_results, fts_results):
        fused = super().rerank_hybrid(
            query,
            vector_results.sort_by([("_distance", "ascending"), *self.TIES]).slice(0, self.k),
            fts_results.sort_by([("_score", "descending"), *self.TIES]).slice(0, self.k))
        return fused.sort_by([("_relevance_score", "descending"), *self.TIES]).slice(0, self.k)


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
        reciprocal rank (K=60), as `score`, with each row's own BM25 score as
        `text_score` (0 when only the vector ranking found it). `where`
        filters both rankings before they are cut to `k`."""
        q = (
            self.table.search(query_type="hybrid")
            .vector(np.asarray(vec, dtype=np.float32))
            # Matches the words. A plain string would be parsed, so a quoted
            # phrase fails for want of token positions, and it needs a
            # full-text index; this scans rows no index covers yet.
            .text(MatchQuery(query, "text"))
            .rerank(_InSegmentOrder(k, return_score="all"))
            .limit(k * _InSegmentOrder.OVERFETCH)
        )
        if where:
            q = q.where(where)
        # No select(): naming the columns makes lance warn that the rankings'
        # own score columns were left out, and `text_score` reads one of them.
        # return_score="all" keeps `_score` on a row both rankings found only
        # from lancedb 0.41 (lancedb#4677), hence the pin in pyproject.toml.
        rows = q.to_list()
        # The text ranking scores only its own top `k`, so a row the vector
        # ranking alone brought in is scored here, by the same full-text query.
        unscored = [r["id"] for r in rows if r["_score"] is None]
        own = {}
        if unscored:
            quoted = ", ".join("'{}'".format(i.replace("'", "''")) for i in unscored)
            own = {r["id"]: r["_score"] for r in
                   self.table.search(MatchQuery(query, "text"), query_type="fts")
                   .where(f"id IN ({quoted})", prefilter=True)
                   .limit(len(unscored)).to_list()}
        return [dict({c: r[c] for c in COLUMNS}, score=r["_relevance_score"],
                     text_score=r["_score"] or own.get(r["id"], 0.0))
                for r in rows]

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

