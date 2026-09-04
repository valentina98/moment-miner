import csv
from collections import defaultdict
from pathlib import Path

from .embeddings.base import EmbeddingBackend
from .search import search
from .store import SegmentStore
from .timefmt import parse_ts


def load_labels(path: str) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or set(rows[0]) < {"query", "video", "start", "end"}:
        raise ValueError("labels CSV needs columns: query,video,start,end")
    # Unfilled template rows (video/time left blank) are skipped, so the
    # shipped templates can be copied and filled incrementally.
    return [
        {"query": r["query"].strip(), "video": r["video"].strip(),
         "t0": parse_ts(r["start"]), "t1": parse_ts(r["end"])}
        for r in rows
        if r["video"].strip() and r["start"].strip() and r["end"].strip()
    ]


def _matches(hit: dict, label: dict) -> bool:
    return (
        Path(hit["path"]).name == label["video"]
        and hit["t1"] > label["t0"]
        and hit["t0"] < label["t1"]
    )


def evaluate(
    labels: list[dict],
    backend: EmbeddingBackend,
    store: SegmentStore,
    k: int = 10,
) -> dict:
    """Recall@k and MRR: a label is found if any top-k hit overlaps it."""
    by_query: dict[str, list[dict]] = defaultdict(list)
    for lb in labels:
        by_query[lb["query"]].append(lb)

    per_query = []
    found_total = 0
    reciprocal_ranks = []
    for query, query_labels in by_query.items():
        hits = search(query, backend, store, k=k)
        found = 0
        for lb in query_labels:
            rank = next(
                (i + 1 for i, h in enumerate(hits) if _matches(h, lb)), None
            )
            if rank is not None:
                found += 1
                reciprocal_ranks.append(1.0 / rank)
        found_total += found
        per_query.append({"query": query, "found": found, "total": len(query_labels)})

    return {
        "per_query": per_query,
        "recall": found_total / len(labels) if labels else 0.0,
        "mrr": sum(reciprocal_ranks) / len(labels) if labels else 0.0,
        "k": k,
        "n_labels": len(labels),
    }
