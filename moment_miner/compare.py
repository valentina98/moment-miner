"""Compare caption sets side by side and render a judging page.

Several caption backends over the same segments produce several `captions.csv`
files. This turns them into one HTML page: the segment's stills, the captions
under them, and a mark where the models disagree about what is in the shot.

Disagreement here is **lexical, not semantic**: a segment is marked when the
captions share no content word at all. That is a fact about the text rather
than a tuned threshold, and on the first real bake-off it fired on 9 of 18
segments where a graded score fired on 17 and so told you nothing.

It catches the obvious cases ("dam wall" vs "bridge" vs "embankment") and
misses paraphrase ("bails a trick" vs "fails a landing"), which reads as
disagreement, and shared confident error, which reads as agreement. It is a
reading aid, not a metric; nothing in the index depends on it.
"""

import csv
import json
import random
from importlib.resources import files as pkg_files
from itertools import combinations, permutations
from pathlib import Path

# Words that carry no retrieval signal, so their presence or absence says
# nothing about whether two captions agree.
STOPWORDS = frozenset("""
a an the and or but of in on at to from with without over under through into
onto across along beside near by for as is are was were be been being it its
this that these those there here then than while during before after up down
out off again very some any no not one two three s
""".split())


def tokens(caption: str) -> set[str]:
    """Content words of a caption, crudely singularized."""
    out = set()
    for raw in caption.lower().replace("/", " ").replace("-", " ").split():
        word = "".join(c for c in raw if c.isalnum())
        if len(word) < 3 or word in STOPWORDS:
            continue
        if word.endswith("es") and len(word) > 4:
            word = word[:-2]
        elif word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        out.add(word)
    return out


def agreement(captions: list[str]) -> dict:
    """Shared terms, each caption's unique terms, and whether they disagree."""
    sets = [tokens(c) for c in captions]
    non_empty = [s for s in sets if s]
    shared = set.intersection(*non_empty) if len(non_empty) == len(sets) and sets else set()
    union_others = [set().union(*(sets[:i] + sets[i + 1:])) if len(sets) > 1 else set()
                    for i in range(len(sets))]
    # Containment, not Jaccard: with captions of different lengths Jaccard
    # scores the length gap as disagreement, which is not what is being asked.
    overlaps = [
        len(a & b) / min(len(a), len(b)) if (a and b) else 0.0
        for a, b in combinations(sets, 2)
    ]
    return {
        "shared": sorted(shared),
        "unique": [sorted(s - other) for s, other in zip(sets, union_others)],
        "min_overlap": round(min(overlaps), 3) if overlaps else 1.0,
        "disagree": len(sets) > 1 and not shared,
    }


def read_captions(path: Path) -> dict[str, dict]:
    """A caption CSV, keyed by segment id.

    Accepts the sidecar schema written by `mm index --caption`
    (id,t0,t1,caption,...) and the bare seg,caption shape, which needs a
    --segments file to supply timings.
    """
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no rows")
    key = "id" if "id" in rows[0] else "seg"
    if key not in rows[0] or "caption" not in rows[0]:
        raise ValueError(f"{path}: expected columns id|seg and caption, got {list(rows[0])}")
    return {r[key]: r for r in rows}


def read_segments(path: Path) -> dict[str, dict]:
    with Path(path).open(newline="") as f:
        return {r["seg"]: r for r in csv.DictReader(f)}


def load_frames(frames_dir: Path, seg_id: str) -> list[str]:
    """Stills already on disk, named `<seg id>_*.jpg` in shot order.

    Lets a comparison run when the archive itself is offline, which is the
    normal state of an external drive.
    """
    import base64

    return [base64.standard_b64encode(p.read_bytes()).decode()
            for p in sorted(Path(frames_dir).glob(f"{seg_id}_*.jpg"))]


def encode_frames(video: Path, t0: float, t1: float, n: int = 3) -> list[str]:
    from .captions import _encode, subsample
    from .frames import extract_frames

    frames = extract_frames(str(video), t0, t1, n=max(n, 3))
    return [_encode(f) for f in subsample(frames)]


def balanced_order(n_segments: int, n_sets: int, seed: int) -> list[list[int]]:
    """One column order per segment, using every ordering equally often.

    A free shuffle clusters: at 18 segments one set landed in the middle column
    on 10 of them, which is the position bias the shuffle exists to remove.
    """
    perms = [list(p) for p in permutations(range(n_sets))]
    deck = (perms * (n_segments // len(perms) + 1))[:n_segments]
    random.Random(seed).shuffle(deck)
    return deck


def build(
    caption_files: list[Path],
    videos: Path | None = None,
    segments_file: Path | None = None,
    frames_dir: Path | None = None,
    blind: bool = True,
    seed: int = 0,
    frames: bool = True,
) -> tuple[str, list[dict]]:
    """Render the page. Returns (html, order_map)."""
    sets = [read_captions(p) for p in caption_files]
    names = [Path(p).stem for p in caption_files]
    seg_meta = read_segments(segments_file) if segments_file else {}
    ids = [i for i in sets[0] if all(i in s for s in sets)]
    if not ids:
        raise ValueError("no segment id appears in every caption file")

    deck = balanced_order(len(ids), len(sets), seed) if blind else \
        [list(range(len(sets)))] * len(ids)

    data, order_map = [], []
    for seg_id, order in zip(ids, deck):
        row = sets[0][seg_id]
        meta = seg_meta.get(seg_id, row)
        t0, t1 = float(meta.get("t0", 0)), float(meta.get("t1", 0))
        shots = []
        if frames and frames_dir:
            shots = load_frames(frames_dir, seg_id)
        elif frames and videos:
            name = meta.get("video") or Path(row.get("path", "")).name
            path = Path(videos) / name if name else None
            if path and path.exists():
                shots = encode_frames(path, t0, t1)
        ordered = [sets[i][seg_id]["caption"] for i in order]
        data.append({
            "seg": seg_id,
            "t0": t0,
            "t1": t1,
            "frames": shots,
            "captions": ordered,
            "labels": ["ABCDEF"[i] for i in range(len(ordered))] if blind else
                      [names[i] for i in order],
            "agree": agreement(ordered),
        })
        order_map.append({"seg": seg_id, **{
            "ABCDEF"[slot]: names[i] for slot, i in enumerate(order)
        }})

    template = (pkg_files("moment_miner") / "templates" / "compare.html").read_text()
    html = (template
            .replace("__DATA__", json.dumps(data))
            .replace("__N__", str(len(data)))
            .replace("__BLIND__", "true" if blind else "false"))
    return html, order_map
