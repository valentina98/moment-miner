from pathlib import Path

import lancedb
import numpy as np
from lancedb.index import FTS

COLUMNS = ["id", "video_id", "path", "t0", "t1", "text"]


class SegmentStore:
    """One LanceDB table per embedding backend (dimensions differ)."""

    def __init__(self, data_dir: str | Path, backend_name: str):
        self.db = lancedb.connect(str(Path(data_dir) / "lance"))
        self.table_name = f"segments_{backend_name}"

    def _names(self) -> list[str]:
        return list(self.db.list_tables().tables)

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
            # its rows share an id and rrf_fuse would credit that id twice.
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

    def rebuild_fts(self):
        self.table.create_index("text", config=FTS(), replace=True)

    def vector_search(self, vec: np.ndarray, k: int = 50) -> list[dict]:
        return (
            self.table.search(np.asarray(vec, dtype=np.float32))
            .limit(k).select([*COLUMNS, "_distance"]).to_list()
        )

    def text_search(self, query: str, k: int = 50) -> list[dict]:
        try:
            return (
                self.table.search(query, query_type="fts")
                .limit(k).select([*COLUMNS, "_score"]).to_list()
            )
        except Exception:
            # No FTS index yet, or query contains no indexable terms.
            return []

    def count(self) -> int:
        if self.table_name not in self._names():
            return 0
        return self.table.count_rows()
