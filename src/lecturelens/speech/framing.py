"""Short-term analysis primitives: pre-emphasis, framing and windowing (from scratch)."""
from __future__ import annotations

import numpy as np


def pre_emphasis(x: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    """y[n] = x[n] - alpha * x[n-1]; boosts high frequencies attenuated by glottal roll-off."""
    return np.append(x[0], x[1:] - alpha * x[:-1])


def window(kind: str, n: int) -> np.ndarray:
    """Analysis windows written out explicitly (periodic form, matching librosa/scipy 'fftbins')."""
    k = np.arange(n)
    if kind == "rect":
        return np.ones(n)
    if kind == "hamming":
        return 0.54 - 0.46 * np.cos(2 * np.pi * k / n)
    if kind == "hann":
        return 0.5 - 0.5 * np.cos(2 * np.pi * k / n)
    raise ValueError(f"unknown window {kind!r}")


def frame_signal(x: np.ndarray, frame_len: int, hop: int, pad: bool = True) -> np.ndarray:
    """Split a 1-D signal into overlapping frames, shape (n_frames, frame_len).

    With pad=True the signal is zero-padded so the last partial frame is kept.
    """
    if pad:
        n_frames = 1 + int(np.ceil(max(len(x) - frame_len, 0) / hop))
        need = (n_frames - 1) * hop + frame_len
        x = np.pad(x, (0, need - len(x)))
    else:
        n_frames = 1 + (len(x) - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
    return x[idx]


def frame_times(n_frames: int, hop: int, sr: int, frame_len: int) -> np.ndarray:
    """Centre time (s) of each frame."""
    return (np.arange(n_frames) * hop + frame_len / 2) / sr
