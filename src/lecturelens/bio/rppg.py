"""Remote photoplethysmography (rPPG): heart rate from subtle skin-colour changes in webcam video.

Pipeline: face landmarks -> forehead + cheek ROIs -> YCrCb skin mask (segmentation + morphology)
-> per-frame mean RGB -> POS projection (Wang et al., 2017) -> band-pass 0.7–3 Hz -> HR via FFT peak,
HRV (RMSSD) via peak detection. Also eye-aspect-ratio (EAR) blink detection (Soukupová & Čech, 2016).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

# MediaPipe face-mesh landmark indices
FOREHEAD = [10, 338, 297, 332, 333, 299, 337, 151, 108, 69, 104, 103, 67, 109]
L_CHEEK = [117, 118, 101, 36, 205, 187, 123, 116]
R_CHEEK = [346, 347, 330, 266, 425, 411, 352, 345]
L_EYE = [362, 385, 387, 263, 373, 380]
R_EYE = [33, 160, 158, 133, 153, 144]

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/latest/face_landmarker.task")


# ---- signal processing (pure numpy; unit-tested on synthetic data) -------------------------------

def pos(rgb: np.ndarray, fs: float, win_s: float = 1.6) -> np.ndarray:
    """Plane-Orthogonal-to-Skin projection. rgb: (T, 3) mean skin colour per frame."""
    T = len(rgb)
    l = int(round(win_s * fs))
    H = np.zeros(T)
    P = np.array([[0, 1, -1], [-2, 1, 1]], float)
    for n in range(l, T + 1):
        C = rgb[n - l:n].T                                 # (3, l)
        Cn = C / (C.mean(axis=1, keepdims=True) + 1e-9)    # temporal normalisation
        S = P @ Cn
        h = S[0] + (S[0].std() / (S[1].std() + 1e-9)) * S[1]
        H[n - l:n] += h - h.mean()                         # overlap-add
    return H


def green(rgb: np.ndarray) -> np.ndarray:
    """Baseline (Verkruysse et al., 2008): the detrended green channel."""
    g = rgb[:, 1]
    return g - np.convolve(g, np.ones(31) / 31, mode="same")


def bandpass(x: np.ndarray, fs: float, lo=0.7, hi=3.0, order=3) -> np.ndarray:
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def hr_fft(x: np.ndarray, fs: float, lo=0.7, hi=3.0, pad=8) -> float:
    """Dominant frequency in the cardiac band, in beats per minute."""
    n = len(x) * pad
    spec = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(len(x)), n))
    f = np.fft.rfftfreq(n, 1 / fs)
    band = (f >= lo) & (f <= hi)
    return float(f[band][np.argmax(spec[band])] * 60)


def hr_track(pulse: np.ndarray, fs: float, win_s=10.0, hop_s=1.0):
    w, h = int(win_s * fs), int(hop_s * fs)
    t, hr = [], []
    for s in range(0, max(len(pulse) - w, 0) + 1, h):
        t.append((s + w / 2) / fs)
        hr.append(hr_fft(pulse[s:s + w], fs))
    return np.array(t), np.array(hr)


def rmssd_ms(pulse: np.ndarray, fs: float) -> float:
    peaks, _ = find_peaks(pulse, distance=int(fs / 3.0))
    ibi = np.diff(peaks) / fs * 1000
    return float(np.sqrt(np.mean(np.diff(ibi) ** 2))) if len(ibi) > 2 else float("nan")


def ear(pts: np.ndarray) -> float:
    """Eye aspect ratio from 6 eye landmarks p1..p6."""
    p1, p2, p3, p4, p5, p6 = pts
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2 * np.linalg.norm(p1 - p4) + 1e-9))


def blinks(ear_series: np.ndarray, fs: float, thresh_ratio=0.75, min_frames=2) -> list[float]:
    """Blink = EAR below thresh_ratio * (median open-eye EAR) for >= min_frames frames."""
    thr = thresh_ratio * np.nanmedian(ear_series)
    below = np.nan_to_num(ear_series, nan=1.0) < thr
    out, i = [], 0
    while i < len(below):
        if below[i]:
            j = i
            while j < len(below) and below[j]:
                j += 1
            if j - i >= min_frames:
                out.append(i / fs)
            i = j
        else:
            i += 1
    return out


# ---- video front-end -------------------------------------------------------------------------------

def skin_mask_ycrcb(bgr: np.ndarray) -> np.ndarray:
    """Chrominance skin segmentation (Cr 133–173, Cb 77–127) cleaned with open/close."""
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    m = cv2.inRange(ycc, (0, 133, 77), (255, 173, 127))
    k = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)


def _landmarker(model_dir: Path):
    import urllib.request

    from mediapipe.tasks.python import BaseOptions, vision
    model = model_dir / "face_landmarker.task"
    if not model.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, model)
    opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model)),
                                        running_mode=vision.RunningMode.VIDEO, num_faces=1)
    return vision.FaceLandmarker.create_from_options(opts)


@dataclass
class FaceSignals:
    fs: float
    rgb: np.ndarray        # (T, 3) mean skin RGB, NaN where no face
    ear: np.ndarray        # (T,)


def extract_face_signals(video: str | Path, model_dir: Path = Path("data/models"),
                         max_seconds: float | None = None) -> FaceSignals:
    import mediapipe as mp
    cap = cv2.VideoCapture(str(video))
    fs = cap.get(cv2.CAP_PROP_FPS) or 30.0
    lm = _landmarker(model_dir)
    rgb, ears, i = [], [], 0
    while True:
        ok, bgr = cap.read()
        if not ok or (max_seconds and i / fs > max_seconds):
            break
        h, w = bgr.shape[:2]
        res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                           data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), int(i * 1000 / fs))
        if res.face_landmarks:
            pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])
            mask = np.zeros((h, w), np.uint8)
            for poly in (FOREHEAD, L_CHEEK, R_CHEEK):
                cv2.fillPoly(mask, [pts[poly].astype(np.int32)], 255)
            mask &= skin_mask_ycrcb(bgr)
            if mask.sum() > 0:
                b, g, r, _ = cv2.mean(bgr, mask=mask)
                rgb.append([r, g, b])
            else:
                rgb.append([np.nan] * 3)
            ears.append((ear(pts[L_EYE]) + ear(pts[R_EYE])) / 2)
        else:
            rgb.append([np.nan] * 3)
            ears.append(np.nan)
        i += 1
    cap.release()
    rgb = np.array(rgb, float)
    # fill short face-detection dropouts by linear interpolation (missing-data handling)
    for c in range(3):
        bad = np.isnan(rgb[:, c])
        if bad.any() and (~bad).any():
            rgb[bad, c] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), rgb[~bad, c])
    return FaceSignals(fs, rgb, np.array(ears, float))
