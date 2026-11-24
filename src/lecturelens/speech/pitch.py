"""Pitch (F0) estimation two ways: time-domain autocorrelation and cepstral peak picking."""
from __future__ import annotations

import numpy as np

from .framing import frame_signal, frame_times, window
from .spectral import real_cepstrum


def autocorr(frame: np.ndarray) -> np.ndarray:
    """Biased autocorrelation r[k] = sum x[n]x[n+k], computed via FFT (Wiener–Khinchin)."""
    n = len(frame)
    f = np.fft.rfft(frame, 2 * n)
    return np.fft.irfft(np.abs(f) ** 2)[:n]


def f0_autocorr(frame: np.ndarray, sr: int, fmin=70.0, fmax=400.0, voicing=0.35) -> float:
    frame = frame - frame.mean()
    r = autocorr(frame * window("hamming", len(frame)))
    if r[0] <= 0:
        return 0.0
    r = r / r[0]
    lo, hi = int(sr / fmax), min(int(sr / fmin), len(r) - 1)
    k = lo + int(np.argmax(r[lo:hi]))
    if r[k] < voicing:
        return 0.0
    # parabolic interpolation for sub-sample lag
    if 0 < k < len(r) - 1:
        a, b, c = r[k - 1], r[k], r[k + 1]
        denom = a - 2 * b + c
        k = k + 0.5 * (a - c) / denom if denom != 0 else k
    return sr / k


def f0_cepstrum(frame: np.ndarray, sr: int, fmin=70.0, fmax=400.0, thresh=0.08) -> float:
    c = real_cepstrum(frame * window("hamming", len(frame)))
    lo, hi = int(sr / fmax), min(int(sr / fmin), len(c) // 2)
    q = lo + int(np.argmax(c[lo:hi]))
    return sr / q if c[q] > thresh else 0.0


def pitch_track(x: np.ndarray, sr: int, method: str = "autocorr", frame_ms=40, hop_ms=10):
    fl, hop = int(sr * frame_ms / 1000), int(sr * hop_ms / 1000)
    frames = frame_signal(x, fl, hop)
    fn = f0_autocorr if method == "autocorr" else f0_cepstrum
    f0 = np.array([fn(fr, sr) for fr in frames])
    return frame_times(len(frames), hop, sr, fl), f0
