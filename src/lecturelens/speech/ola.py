"""Short-term synthesis: overlap-add reconstruction, WSOLA time-scale modification, smart-speed.

Smart-speed = (1) cut pauses longer than `keep_pause_s` down to `keep_pause_s`, then
(2) WSOLA time compression so speech plays faster without raising pitch.
"""
from __future__ import annotations

import numpy as np

from .framing import frame_signal, window


def ola_reconstruct(x: np.ndarray, frame_len: int = 512, hop: int = 128) -> np.ndarray:
    """Analysis framing with a Hann window then overlap-add, normalised by the summed window.

    With Hann at 75 % overlap the window sum is constant (COLA), so this is a perfect
    reconstruction of x — the sanity check for every OLA-based synthesis.
    """
    w = window("hann", frame_len)
    frames = frame_signal(x, frame_len, hop) * w
    out = np.zeros((len(frames) - 1) * hop + frame_len)
    wsum = np.zeros_like(out)
    for i, fr in enumerate(frames):
        out[i * hop:i * hop + frame_len] += fr
        wsum[i * hop:i * hop + frame_len] += w
    out /= np.maximum(wsum, 1e-8)
    return out[:len(x)]


def wsola(x: np.ndarray, rate: float, sr: int, frame_ms=30.0, tol_ms=10.0) -> np.ndarray:
    """Waveform-Similarity Overlap-Add time-scale modification (rate>1 = faster).

    Each output frame is taken from around the nominal input position, shifted by up to
    ±tol so it best continues the previous frame's natural continuation (max cross-corr).
    """
    N = int(sr * frame_ms / 1000) // 2 * 2
    Hs = N // 2                       # synthesis hop (50 % overlap, Hann is COLA)
    Ha = Hs * rate                    # analysis hop
    tol = int(sr * tol_ms / 1000)
    w = window("hann", N)
    n_in = len(x)
    x = np.pad(x.astype(np.float64), (tol, N + tol + int(Ha) + 1))
    n_out = int((len(x) - 2 * tol - N) / Ha)
    out = np.zeros(n_out * Hs + N)
    prev = tol  # input position of previous chosen frame
    for k in range(n_out):
        nominal = int(round(k * Ha)) + tol
        if k == 0:
            pos = nominal
        else:
            target = x[prev + Hs: prev + Hs + N]          # natural continuation of last frame
            lo, hi = max(nominal - tol, 0), min(nominal + tol, len(x) - N)
            region = x[lo:hi + N]
            if len(region) < N or len(target) < N:
                break
            cc = np.correlate(region, target, mode="valid")
            pos = lo + int(np.argmax(cc))
        out[k * Hs:k * Hs + N] += x[pos:pos + N] * w
        prev = pos
    return out[: int(n_in / rate)]


def remove_pauses(x: np.ndarray, sr: int, pauses: list[tuple[float, float]], keep_pause_s=0.15,
                  fade_ms=10.0) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Shorten every pause to keep_pause_s. Returns new signal and a time map [(t_in, t_out)]."""
    fade = int(sr * fade_ms / 1000)
    pieces, tmap = [], [(0.0, 0.0)]
    cursor, out_len = 0, 0
    for (s, e) in pauses:
        a = int(s * sr)
        b = int(e * sr) - int(keep_pause_s * sr)
        if b <= a:
            continue
        seg = x[cursor:a].astype(np.float64).copy()
        if len(seg) > 2 * fade:
            seg[-fade:] *= np.linspace(1, 0, fade)
        pieces.append(seg)
        out_len += len(seg)
        tmap.append((a / sr, out_len / sr))
        cursor = b
    tail = x[cursor:].astype(np.float64).copy()
    if pieces and len(tail) > fade:
        tail[:fade] *= np.linspace(0, 1, fade)
    pieces.append(tail)
    return np.concatenate(pieces), tmap


def smart_speed(x: np.ndarray, sr: int, pauses, rate: float = 1.3, keep_pause_s=0.15) -> np.ndarray:
    y, _ = remove_pauses(x, sr, pauses, keep_pause_s)
    return wsola(y, rate, sr) if rate != 1.0 else y
