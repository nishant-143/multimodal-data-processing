"""Slide-change detection with three feature descriptors, plus revisited-slide matching.

Descriptors per sampled frame:
  * hist  - 64-bin grey histogram + 4x4 spatial grid of mean intensities (cheap, global)
  * hog   - Histogram of Oriented Gradients on a 128x72 thumbnail (layout/text structure)
  * orb   - ORB keypoints; distance = 1 - fraction of good matches (local, viewpoint robust)
A change is declared where the frame-to-frame distance is a local peak above an
adaptive threshold (median + k * MAD), with a minimum gap between changes.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.feature import hog


def desc_hist(gray: np.ndarray) -> np.ndarray:
    h = np.bincount((gray // 4).ravel(), minlength=64).astype(float)
    h /= h.sum()
    small = cv2.resize(gray, (4, 4), interpolation=cv2.INTER_AREA).ravel() / 255.0
    return np.concatenate([h, small / 4])


def desc_hog(gray: np.ndarray) -> np.ndarray:
    small = cv2.resize(gray, (128, 72), interpolation=cv2.INTER_AREA)
    return hog(small, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)


_orb = None


def desc_orb(gray: np.ndarray):
    global _orb
    _orb = _orb or cv2.ORB_create(500)
    small = cv2.resize(gray, (640, 360), interpolation=cv2.INTER_AREA)
    return _orb.detectAndCompute(small, None)[1]


def dist(kind: str, a, b) -> float:
    if kind == "hist":
        return 0.5 * float(np.abs(a - b).sum())
    if kind == "hog":
        return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    if kind == "orb":
        if a is None or b is None or len(a) < 10 or len(b) < 10:
            return 1.0
        m = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(a, b)
        good = [x for x in m if x.distance < 40]
        return 1.0 - len(good) / min(len(a), len(b))
    raise ValueError(kind)


DESCRIPTORS = {"hist": desc_hist, "hog": desc_hog, "orb": desc_orb}


@dataclass
class SlideSegment:
    slide_id: int          # same id when a slide is revisited
    start: float
    end: float
    frame_idx: int         # representative (middle) sampled frame


def frame_distances(grays: list[np.ndarray], kind: str) -> np.ndarray:
    d = [DESCRIPTORS[kind](g) for g in grays]
    return np.array([0.0] + [dist(kind, d[i - 1], d[i]) for i in range(1, len(d))])


# Robust z-score threshold per descriptor: ORB distances are intrinsically noisier.
DEFAULT_K = {"hist": 8.0, "hog": 8.0, "orb": 4.0, "fused": 6.0}


def robust_z(d: np.ndarray, win: int | None = None) -> np.ndarray:
    """Robust z-score of each distance against its *local* neighbourhood (±win samples, excluding itself).

    Frame-to-frame distances are not stationary: static slides give ~0, a moving lecturer on camera
    gives a much higher noise floor. A global median/MAD therefore sets one threshold that is far too
    high for slides and too low for camera shots; a sliding-window baseline adapts to the shot type.
    A floor on the scale (10 % of the global MAD) stops tiny ink strokes on a static slide exploding.
    """
    gmed = np.median(d)
    gmad = 1.4826 * np.median(np.abs(d - gmed)) + 1e-9
    if win is None:
        return (d - gmed) / gmad
    z = np.zeros_like(d, dtype=float)
    for i in range(len(d)):
        nb = np.r_[d[max(0, i - win):i], d[i + 1:i + 1 + win]]
        med = np.median(nb)
        mad = 1.4826 * np.median(np.abs(nb - med))
        z[i] = (d[i] - med) / max(mad, 0.1 * gmad)
    return z


def fused_scores(dists_by_kind: dict[str, np.ndarray]) -> np.ndarray:
    """Late fusion of descriptors: max over each descriptor's z-score scaled by its own threshold."""
    return np.max([robust_z(d) / DEFAULT_K[k] for k, d in dists_by_kind.items()], axis=0) * DEFAULT_K["fused"]


def shot_aware_scores(dists_by_kind: dict[str, np.ndarray], types: list[str], pct: float = 99.0) -> np.ndarray:
    """Shot-type-aware late fusion (the detector used in the pipeline).

    * the grey histogram (global z) catches cuts between camera / slide / demo shots;
    * ORB is used only between two consecutive *slide* frames, where it catches slide builds that
      barely change the histogram. Its threshold is label-free: the `pct` percentile of all
      slide-to-slide ORB distances in this lecture (compression noise sits far below it).
    Returns scores where > 1 means "change".
    """
    t = np.asarray(types)
    both = np.r_[False, (t[1:] == "slide") & (t[:-1] == "slide")]
    zh = robust_z(dists_by_kind["hist"]) / DEFAULT_K["hist"]
    orb = dists_by_kind["orb"]
    if both.sum() < 10:
        return zh
    floor = max(np.percentile(orb[both], pct), 0.05)
    return np.maximum(zh, np.where(both, orb / floor, -np.inf))


def detect_changes(dists: np.ndarray, times: np.ndarray, k: float | str = "hist", min_gap_s: float = 5.0,
                   already_z: bool = False) -> list[float]:
    """Peaks whose robust z-score exceeds k (a number, or a descriptor name to use its default)."""
    k = DEFAULT_K[k] if isinstance(k, str) else k
    z = dists if already_z else robust_z(dists)
    cands = [i for i in range(1, len(z)) if z[i] > k]
    out: list[float] = []
    for i in sorted(cands, key=lambda i: -z[i]):            # greedy NMS by strength
        if all(abs(times[i] - t) >= min_gap_s for t in out):
            out.append(float(times[i]))
    return sorted(out)


def segment_slides(grays, times, changes: list[float], revisit_thresh: float = 0.12) -> list[SlideSegment]:
    bounds = [float(times[0])] + changes + [float(times[-1]) + (times[1] - times[0] if len(times) > 1 else 1)]
    segs, reps = [], []
    for s, e in zip(bounds, bounds[1:]):
        idx = [i for i, t in enumerate(times) if s <= t < e]
        if not idx:
            continue
        mid = idx[len(idx) // 2]
        d = desc_hog(grays[mid])
        sid = next((j for j, r in enumerate(reps) if dist("hog", r, d) < revisit_thresh), None)
        if sid is None:
            sid = len(reps)
            reps.append(d)
        segs.append(SlideSegment(sid, s, e, mid))
    return segs


def change_prf(pred: list[float], truth: list[float], tol: float = 3.0) -> tuple[float, float, float]:
    """Precision/recall/F1 with one-to-one matching within ±tol seconds."""
    used, tp = set(), 0
    for p in pred:
        j = next((j for j, t in enumerate(truth) if j not in used and abs(p - t) <= tol), None)
        if j is not None:
            used.add(j)
            tp += 1
    P = tp / len(pred) if pred else 0.0
    R = tp / len(truth) if truth else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)
