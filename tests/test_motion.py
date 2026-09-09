import numpy as np

from moment_miner.motion import PIXEL_DELTA, is_static, moving_fraction


def frames(n, h=8, w=8):
    return np.zeros((n, h, w, 3), dtype=np.uint8)


def test_identical_frames_do_not_move():
    assert moving_fraction(frames(4)) == 0.0


def test_every_pixel_changing_reads_as_one():
    f = frames(2)
    f[1] = 255
    assert moving_fraction(f) == 1.0


def test_small_changes_are_ignored_as_sensor_noise():
    f = frames(2)
    f[1] = PIXEL_DELTA          # at the threshold, not over it
    assert moving_fraction(f) == 0.0


def test_a_small_moving_subject_barely_registers():
    """The case that matters here: a locked-off camera on a subject a few
    percent of frame height. Sixteen of 192 pixels move."""
    f = frames(2)
    f[1, :2, :2] = 255
    assert moving_fraction(f) == 4 / 64
    assert is_static(f, threshold=0.35)


def test_a_pan_changes_most_of_the_frame():
    f = frames(2)
    f[1, :, :6] = 255
    assert not is_static(f, threshold=0.35)


def test_one_frame_cannot_move():
    assert moving_fraction(frames(1)) == 0.0
    assert moving_fraction(None) == 0.0
