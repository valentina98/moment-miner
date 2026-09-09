import numpy as np
import pytest

from moment_miner.embeddings.mock import MockBackend
from moment_miner.locate import (
    FixedWindow,
    MaxSubarray,
    build_locator,
    heatmap,
)


def _curve(bump_from, bump_to, n=40, low=0.10, high=0.30):
    scores = np.full(n, low)
    scores[bump_from:bump_to] = high
    return scores, np.arange(n, dtype=float)


def test_heatmap_is_one_cosine_per_frame():
    be = MockBackend()
    frames = np.arange(3 * 4 * 4 * 3, dtype=np.uint8).reshape(3, 4, 4, 3)
    vecs = be.embed_frames(frames)
    q = be.embed_text(["a query"])[0]
    scores = heatmap(vecs, q)
    assert scores.shape == (3,)
    assert np.allclose(scores, [v @ q for v in vecs])


def test_embed_segment_is_the_mean_of_embed_frames():
    """The pooling moved to the base class; it must still produce the same vector."""
    be = MockBackend()
    frames = np.arange(5 * 4 * 4 * 3, dtype=np.uint8).reshape(5, 4, 4, 3)
    pooled = be.embed_frames(frames).mean(axis=0)
    expected = pooled / np.linalg.norm(pooled)
    assert np.allclose(be.embed_segment(frames), expected)
    assert abs(np.linalg.norm(be.embed_segment(frames)) - 1.0) < 1e-6


def test_maxsub_finds_the_bump_without_being_told_its_length():
    scores, times = _curve(10, 20)
    (region,) = MaxSubarray().locate(scores, times)
    assert (region.t0, region.t1) == (10.0, 20.0)


def test_maxsub_extent_follows_the_content():
    """The same locator, no parameter change, returns 4 s or 20 s as the curve dictates."""
    short = MaxSubarray().locate(*_curve(10, 14))[0]
    long_ = MaxSubarray().locate(*_curve(10, 30))[0]
    assert short.t1 - short.t0 == 4.0
    assert long_.t1 - long_.t0 == 20.0


def test_maxsub_prefers_a_long_run_over_a_single_taller_frame():
    """Sum over the centered curve, not peak height: 'in a row' is length, not amplitude."""
    scores = np.full(40, 0.10)
    scores[5] = 0.50
    scores[20:35] = 0.25
    (region,) = MaxSubarray().locate(scores, np.arange(40, dtype=float))
    assert (region.t0, region.t1) == (20.0, 35.0)


def test_maxsub_on_a_flat_curve_returns_a_single_frame():
    scores = np.full(10, 0.2)
    scores[7] = 0.2001
    (region,) = MaxSubarray().locate(scores, np.arange(10, dtype=float))
    assert region.t0 == 7.0


def test_maxsub_top_k_returns_disjoint_regions():
    scores = np.full(60, 0.10)
    scores[5:10] = 0.30
    scores[40:50] = 0.28
    regions = MaxSubarray(top_k=2).locate(scores, np.arange(60, dtype=float))
    assert len(regions) == 2
    a, b = sorted(regions, key=lambda r: r.t0)
    assert a.t1 <= b.t0
    assert (a.t0, a.t1) == (5.0, 10.0) and (b.t0, b.t1) == (40.0, 50.0)


def test_fixed_window_honours_the_requested_length():
    scores, times = _curve(10, 30)
    (region,) = FixedWindow(duration=5.0).locate(scores, times)
    assert region.t1 - region.t0 == 5.0
    assert region.t0 >= 10.0 and region.t1 <= 30.0


def test_fixed_window_respects_a_non_unit_frame_period():
    """At 4 fps a 5-second window is 20 frames, not 5."""
    scores = np.full(80, 0.1)
    scores[40:60] = 0.4
    times = np.arange(80) / 4.0
    (region,) = FixedWindow(duration=5.0).locate(scores, times)
    assert abs((region.t1 - region.t0) - 5.0) < 1e-9
    assert abs(region.t0 - 10.0) < 1e-9


def test_locators_are_interchangeable_behind_one_call():
    scores, times = _curve(10, 20)
    for loc in (MaxSubarray(), FixedWindow(duration=3.0)):
        (region,) = loc.locate(scores, times)
        assert region.t0 >= 10.0 and region.t1 <= 20.0


def test_empty_curve_locates_nothing():
    empty = np.array([])
    assert MaxSubarray().locate(empty, empty) == []
    assert FixedWindow(duration=2.0).locate(empty, empty) == []


def test_build_locator_requires_a_duration_only_where_it_means_something():
    assert isinstance(build_locator("maxsub"), MaxSubarray)
    assert isinstance(build_locator("window", duration=4.0), FixedWindow)
    with pytest.raises(ValueError, match="needs --duration"):
        build_locator("window")
    with pytest.raises(ValueError, match="infers the duration"):
        build_locator("maxsub", duration=4.0)
    with pytest.raises(ValueError, match="unknown locator"):
        build_locator("nope")


def test_median_baseline_merges_two_bumps_across_the_flat_between_them():
    """Why the default is the mean: on a flat floor the median sits on it, every floor
    frame centers to zero, and a zero neither extends a run nor ends one."""
    scores = np.full(60, 0.10)
    scores[5:10] = 0.30
    scores[40:50] = 0.28
    times = np.arange(60, dtype=float)
    (merged,) = MaxSubarray(baseline="median").locate(scores, times)
    assert merged.t0 == 5.0 and merged.t1 >= 50.0
    (first,) = MaxSubarray(baseline="mean").locate(scores, times)
    assert first.t1 - first.t0 <= 11.0


def test_percentile_baseline_handles_an_action_filling_most_of_the_clip():
    scores = np.full(40, 0.30)
    scores[:6] = 0.10
    scores[34:] = 0.10
    times = np.arange(40, dtype=float)
    (region,) = MaxSubarray(baseline=20).locate(scores, times)
    assert (region.t0, region.t1) == (6.0, 34.0)


def test_unknown_baseline_is_refused():
    with pytest.raises(ValueError, match="unknown baseline"):
        MaxSubarray(baseline="mode")
