from moment_miner.embeddings.mock import MockBackend
from moment_miner.evaluate import evaluate, load_labels
from moment_miner.store import SegmentStore


def _store_with(tmp_path, be, docs):
    store = SegmentStore(tmp_path, be.name)
    store.add([
        {"id": f"v{i}:0", "video_id": f"v{i}", "path": f"/x/clip{i}.mp4",
         "t0": 10.0, "t1": 18.0, "text": text,
         "vector": be.embed_text([text])[0]}
        for i, text in enumerate(docs)
    ])
    store.rebuild_fts()
    return store


def test_evaluate_recall_and_mrr(tmp_path):
    be = MockBackend()
    store = _store_with(tmp_path, be, ["kong vault", "backflip", "crowd"])
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text(
        "query,video,start,end\n"
        "kong vault,clip0.mp4,0:12,0:16\n"      # overlaps stored 10-18 → found
        "kong vault,clip4.mp4,1:00,1:05\n"      # no such video → miss
        "backflip,clip1.mp4,0:20,0:25\n"        # right video, no overlap → miss
        "unfilled template row,,,\n"            # skipped, not a miss
    )
    labels = load_labels(str(labels_csv))
    assert len(labels) == 3
    assert labels[0]["t0"] == 12.0

    report = evaluate(labels, be, store, k=5)
    assert report["n_labels"] == 3
    assert abs(report["recall"] - 1 / 3) < 1e-6
    assert report["mrr"] > 0.3
    by_q = {q["query"]: q for q in report["per_query"]}
    assert by_q["kong vault"]["found"] == 1
    assert by_q["backflip"]["found"] == 0
