import numpy as np
import pytest

from lecturelens.bio.rppg import bandpass, blinks, green, hr_fft, hr_track, pos


def synthetic_face_rgb(bpm=72.0, fs=30.0, dur=30.0, seed=0, motion=True):
    """Skin colour = DC + small pulsatile component (strongest in green) + illumination/motion noise
    that scales all channels equally (the case POS is designed to cancel)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * dur)) / fs
    pulse = np.sin(2 * np.pi * bpm / 60 * t)
    base = np.array([180.0, 120.0, 100.0])
    pulsatile = np.array([0.33, 0.77, 0.53]) * 0.4       # relative PPG strength per channel
    illum = 1 + (0.03 * np.sin(2 * np.pi * 1.3 * t + 1) if motion else 0) + 0.002 * rng.standard_normal(len(t))
    rgb = (base[None, :] + pulsatile[None, :] * pulse[:, None]) * np.asarray(illum)[:, None]
    return rgb + 0.2 * rng.standard_normal(rgb.shape), t


@pytest.mark.parametrize("bpm", [55, 72, 110])
def test_pos_recovers_heart_rate(bpm):
    rgb, _ = synthetic_face_rgb(bpm)
    h = bandpass(pos(rgb, 30.0), 30.0)
    assert hr_fft(h, 30.0) == pytest.approx(bpm, abs=2)


def test_pos_beats_green_under_illumination_motion():
    """Illumination flicker at 78 'bpm' (1.3 Hz) sits in the cardiac band: green locks on to it, POS doesn't."""
    rgb, _ = synthetic_face_rgb(100)
    err_pos = abs(hr_fft(bandpass(pos(rgb, 30.0), 30.0), 30.0) - 100)
    err_green = abs(hr_fft(bandpass(green(rgb), 30.0), 30.0) - 100)
    assert err_pos < 2 < err_green


def test_hr_track_length():
    rgb, _ = synthetic_face_rgb(80, dur=40)
    t, hr = hr_track(bandpass(pos(rgb, 30.0), 30.0), 30.0)
    assert len(t) == 31 and np.median(hr) == pytest.approx(80, abs=2)


def test_blink_detection():
    e = np.full(300, 0.3)
    for s in (50, 150, 250):
        e[s:s + 4] = 0.1
    assert blinks(e, 30.0) == pytest.approx([50 / 30, 150 / 30, 250 / 30])
