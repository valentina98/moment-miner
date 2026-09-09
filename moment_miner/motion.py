"""Whether a segment's shot is static or moving.

Nobody searches for "the camera moves", so this is not caption text — it is a
filter you apply to a result set: show me the locked-off shots, or only the
moving ones. The caption prompt is told not to describe the shot at all.

What it measures is **how much of the frame changed**, which is a proxy, not a
camera-motion estimator. A pan changes nearly every pixel; a locked-off camera
watching a small subject changes few. It therefore reads a whip-pan and a
close-up filling the frame the same way, and that is the known limit.

Computed from the frames the embedding pass already decoded, so it costs one
array subtraction per segment and no extra ffmpeg work. That is the only reason
it is measured inside `index_pending` at all: extraction is ~20% of index time,
and a separate pass would pay it twice. The *value* goes to its own `motion`
table (`store.MotionStore`), never into `segments_<backend>` -- it does not
depend on the embedding, and keeping it there stored it once per backend and
made adding or dropping the tag a schema change on the vectors.

**Calibrated 2026-09-06 on 29 full-length windows across both corpora.** Cut
clips (`footage/cuts/`, 19 windows) are almost all locked-off: 17 score 0.011
to 0.183, and the two hand-held ones 0.688 and 0.727. Raw camera clips
(`footage/src/canon_50fps/`, 10 windows) are all 0.480 to 0.903. So the classes
separate at 0.183 -> 0.480, a gap of 0.30, and the threshold below sits inside
it and classifies every one of the 29 correctly.

Two caveats on that. The gap was measured on one shooter's footage, so treat
0.35 as a good default rather than a constant of nature -- which is why the cut
is applied at search time (`--static-max`) rather than stored, so retuning it
does not cost a re-index. And an earlier pass
over the raw clips alone found no static shots at all and an apparent valley at
0.17-0.45 -- those low scores were degenerate tail windows a few tenths of a
second long, not locked-off shots. Exclude sub-window segments before reading
any distribution of this signal.
"""

import numpy as np

# 0-255. Below this a pixel counts as unchanged: sensor noise and compression
# move quiet pixels by a few levels between frames a second apart.
PIXEL_DELTA = 12

# The static/moving cut. Applied at search time, never stored: it is a
# judgement about the measurement, and baking it into the index made retuning
# cost a full re-index.
STATIC_MAX = 0.35


def moving_fraction(frames: np.ndarray) -> float:
    """Mean fraction of pixels that change between consecutive frames."""
    if frames is None or len(frames) < 2:
        return 0.0
    a = frames.astype(np.int16)
    changed = np.abs(a[1:] - a[:-1]).max(axis=-1) > PIXEL_DELTA
    return float(changed.mean())


def is_static(frames: np.ndarray, threshold: float) -> bool:
    return moving_fraction(frames) < threshold
