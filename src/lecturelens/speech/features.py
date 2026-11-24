"""Short-time energy, zero-crossing rate and an energy/ZCR voice-activity detector."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .framing import frame_signal, frame_times, window


def short_time_energy(frames: np.ndarray, win: np.ndarray | None = None) -> np.ndarray:
    if win is not None:
        frames = frames * win
    return np.sum(frames.astype(np.float64) ** 2, axis=1) / frames.shape[1]


def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    """Fraction of adjacent sample pairs whose sign differs."""
    s = np.sign(frames)
    s[s == 0] = 1
    return np.mean(np.abs(np.diff(s, axis=1)) / 2, axis=1)


@dataclass
class VADResult:
    times: np.ndarray       # frame centre times (s)
    energy_db: np.ndarray
    zcr: np.ndarray
    speech: np.ndarray      # bool per frame
    pauses: list[tuple[float, float]]  # (start, end) of silences >= min_pause


def _smooth_bool(mask: np.ndarray, min_run: int) -> np.ndarray:
    """Remove runs shorter than min_run (both True and False runs)."""
    out = mask.copy()
    for target in (True, False):
        i = 0
        n = len(out)
        while i < n:
            if out[i] == target:
                j = i
                while j < n and out[j] == target:
                    j += 1
                if j - i < min_run and i > 0 and j < n:
                    out[i:j] = not target
                i = j
            else:
                i += 1
    return out


def vad(x: np.ndarray, sr: int, frame_ms: float = 25, hop_ms: float = 10,
        min_pause_s: float = 0.35, energy_margin_db: float = 12.0) -> VADResult:
    """Energy + ZCR voice activity detection.

    The noise floor is estimated as the 10th percentile of frame energy. A frame is
    speech if its energy is `energy_margin_db` above the floor, or if it is moderately
    above the floor *and* has a high ZCR (catches unvoiced fricatives like /s/, /f/).
    """
    fl, hop = int(sr * frame_ms / 1000), int(sr * hop_ms / 1000)
    frames = frame_signal(x, fl, hop)
    e = short_time_energy(frames, window("hamming", fl))
    e_db = 10 * np.log10(e + 1e-12)
    z = zero_crossing_rate(frames)
    floor = np.percentile(e_db, 10)
    voiced = e_db > floor + energy_margin_db
    unvoiced = (e_db > floor + energy_margin_db / 2) & (z > 0.25)
    speech = _smooth_bool(voiced | unvoiced, min_run=int(0.1 / (hop_ms / 1000)))
    t = frame_times(len(frames), hop, sr, fl)

    pauses = []
    hop_s = hop / sr
    i = 0
    while i < len(speech):
        if not speech[i]:
            j = i
            while j < len(speech) and not speech[j]:
                j += 1
            start, end = i * hop_s, j * hop_s
            if end - start >= min_pause_s:
                pauses.append((start, end))
            i = j
        else:
            i += 1
    return VADResult(t, e_db, z, speech, pauses)
