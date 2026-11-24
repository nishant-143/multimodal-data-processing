"""Spatio-temporal activity: frame differencing and dense optical flow (lecturer motion / writing)."""
from __future__ import annotations

import cv2
import numpy as np


def frame_diff_energy(grays: list[np.ndarray], size=(320, 180), thresh=20) -> np.ndarray:
    """Fraction of pixels that changed by > thresh between consecutive (downsampled) frames."""
    small = [cv2.resize(g, size, interpolation=cv2.INTER_AREA).astype(np.int16) for g in grays]
    out = [0.0]
    for a, b in zip(small, small[1:]):
        out.append(float((np.abs(b - a) > thresh).mean()))
    return np.array(out)


def flow_magnitude(grays: list[np.ndarray], size=(320, 180)) -> np.ndarray:
    """Mean Farneback optical-flow magnitude between consecutive frames (pixels/frame)."""
    small = [cv2.resize(g, size, interpolation=cv2.INTER_AREA) for g in grays]
    out = [0.0]
    for a, b in zip(small, small[1:]):
        f = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        out.append(float(np.linalg.norm(f, axis=2).mean()))
    return np.array(out)


def temporal_smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    return np.convolve(x, np.ones(k) / k, mode="same")
