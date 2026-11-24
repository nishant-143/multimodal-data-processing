"""OCR front-end: segmentation (Otsu, from scratch) + morphology to find text lines, then Tesseract."""
from __future__ import annotations

import cv2
import numpy as np
import pytesseract

from .enhance import clahe, median_filter, unsharp_mask


def otsu_threshold(gray: np.ndarray) -> int:
    """Maximise between-class variance  sigma_b^2(t) = w0 w1 (mu0 - mu1)^2."""
    p = np.bincount(gray.ravel(), minlength=256) / gray.size
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mu_t * omega - mu) ** 2 / (omega * (1 - omega))
    return int(np.nanargmax(sb))


def binarize(gray: np.ndarray) -> np.ndarray:
    """Dark text on light background -> text = 255. Inverts automatically for dark slides."""
    t = otsu_threshold(gray)
    bw = (gray > t).astype(np.uint8) * 255
    if bw.mean() < 127:          # majority dark => light text on dark slide
        bw = 255 - bw
    return 255 - bw              # text pixels = 255


def text_line_boxes(gray: np.ndarray, min_h=8):
    """Morphology: opening removes speckle, wide dilation merges characters into line blobs."""
    bw = binarize(gray)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, gray.shape[1] // 60), 3))
    lines = cv2.dilate(bw, k, iterations=1)
    n, _, stats, _ = cv2.connectedComponentsWithStats(lines)
    boxes = [tuple(s[:4]) for s in stats[1:] if s[3] >= min_h and s[2] >= 2 * s[3]]
    return sorted(boxes, key=lambda b: (b[1], b[0])), bw


def preprocess_for_ocr(gray: np.ndarray, upscale_to: int = 1600, source: str = "screen") -> np.ndarray:
    """screen: unsharp mask + upscale (best on 124 lecture slides; median+CLAHE *hurt* clean captures).
    camera: median + CLAHE + upscale, for dim/noisy photographed screens."""
    g = unsharp_mask(gray, 1.0, 1.0) if source == "screen" else clahe(median_filter(gray, 3))
    if g.shape[1] < upscale_to:
        f = upscale_to / g.shape[1]
        g = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return g


def ocr(gray: np.ndarray, preprocess: bool = True) -> str:
    img = preprocess_for_ocr(gray) if preprocess else gray
    return pytesseract.image_to_string(img, config="--psm 3")


def char_accuracy(pred: str, truth: str) -> float:
    """1 - CER, using Levenshtein distance on whitespace-normalised lowercase strings."""
    a, b = " ".join(pred.lower().split()), " ".join(truth.lower().split())
    if not b:
        return 1.0 if not a else 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return max(0.0, 1 - prev[-1] / len(b))
