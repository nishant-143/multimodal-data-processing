"""Edge detection + geometric transform: locate the projected slide in a camera frame and rectify it."""
from __future__ import annotations

import cv2
import numpy as np


def sobel_edges(gray: np.ndarray) -> np.ndarray:
    """Gradient magnitude with explicit Sobel kernels (for figures/teaching; Canny used for detection)."""
    from scipy import ndimage
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
    gx = ndimage.convolve(gray.astype(float), kx)
    gy = ndimage.convolve(gray.astype(float), kx.T)
    return np.hypot(gx, gy)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Return corners as TL, TR, BR, BL."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def find_screen(gray: np.ndarray, min_area_frac: float = 0.15) -> np.ndarray | None:
    """Largest convex quadrilateral in the Canny edge map (after closing gaps), or None."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    v = np.median(blur)
    edges = cv2.Canny(blur, int(max(0, 0.66 * v)), int(min(255, 1.33 * v)))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_min = min_area_frac * gray.shape[0] * gray.shape[1]
    best, best_area = None, 0
    for c in cnts:
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        a = cv2.contourArea(approx)
        if len(approx) == 4 and cv2.isContourConvex(approx) and a > max(area_min, best_area):
            best, best_area = approx, a
    return order_corners(best) if best is not None else None


def homography_dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Direct Linear Transform from 4+ point pairs (from scratch; cross-checked with cv2)."""
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.asarray(A, float))
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def rectify(img: np.ndarray, corners: np.ndarray, out_w=1280, out_h=720) -> np.ndarray:
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32)
    H = homography_dlt(corners, dst)
    return cv2.warpPerspective(img, H, (out_w, out_h))
