"""Run all frame-level vision processing on one ingested lecture in a single streaming pass."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .ocr import text_line_boxes
from .slides import DESCRIPTORS, dist

SLIDE_BOX = (160, 1120)   # x-range of the 4:3 slide inside the 16:9 frame (1280 px wide)


def frame_type(gray: np.ndarray) -> str:
    """slide (4:3 white slide + black bars) | demo (full-width bright screen capture) | camera."""
    h, w = gray.shape
    x0, x1 = int(SLIDE_BOX[0] * w / 1280), int(SLIDE_BOX[1] * w / 1280)
    left, right, center = gray[:, : x0 - 10].mean(), gray[:, x1 + 10:].mean(), gray[:, x0:x1].mean()
    if left < 20 and right < 20 and center > 150:
        return "slide"
    if gray.mean() > 160 and left > 120:
        return "demo"
    return "camera"


def slide_crop(gray: np.ndarray) -> np.ndarray:
    w = gray.shape[1]
    return gray[:, int(SLIDE_BOX[0] * w / 1280): int(SLIDE_BOX[1] * w / 1280)]


@dataclass
class FrameStats:
    times: np.ndarray
    types: list[str]
    dists: dict[str, np.ndarray]     # descriptor distance to previous frame
    diff_energy: np.ndarray
    flow_mag: np.ndarray
    text_lines: np.ndarray
    thumbs: list[np.ndarray]         # 320x180 grey thumbnails (kept for later matching/plots)


def process_frames(frame_dir: Path, frame_rate: float = 1.0) -> FrameStats:
    files = sorted(frame_dir.glob("*.jpg"))
    prev_desc, prev_small = None, None
    dists = {k: [] for k in DESCRIPTORS}
    types, diffs, flows, lines, thumbs = [], [], [], [], []
    for f in files:
        g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        types.append(frame_type(g))
        desc = {k: fn(g) for k, fn in DESCRIPTORS.items()}
        small = cv2.resize(g, (320, 180), interpolation=cv2.INTER_AREA)
        thumbs.append(small)
        if prev_desc is None:
            for k in dists:
                dists[k].append(0.0)
            diffs.append(0.0)
            flows.append(0.0)
        else:
            for k in dists:
                dists[k].append(dist(k, prev_desc[k], desc[k]))
            diffs.append(float((np.abs(small.astype(np.int16) - prev_small) > 20).mean()))
            fl = cv2.calcOpticalFlowFarneback(prev_small, small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flows.append(float(np.linalg.norm(fl, axis=2).mean()))
        lines.append(len(text_line_boxes(slide_crop(g) if types[-1] == "slide" else g)[0])
                     if types[-1] != "camera" else 0)
        prev_desc, prev_small = desc, small
    times = (np.arange(len(files)) + 0.5) / frame_rate
    return FrameStats(times, types, {k: np.array(v) for k, v in dists.items()}, np.array(diffs),
                      np.array(flows), np.array(lines, float), thumbs)
