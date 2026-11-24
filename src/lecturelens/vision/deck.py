"""Video-to-deck alignment: match a slide frame to the page of the published slide PDF.

The PDF gives *exact* slide text, which serves two purposes:
  1. ground truth for measuring OCR accuracy, and
  2. clean slide text for search / off-slide insights when the deck is available.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .slides import desc_hog


@dataclass
class DeckPage:
    index: int
    text: str
    gray: np.ndarray    # rendered page, 960x720


def load_deck(pdf: Path, size=(960, 720)) -> list[DeckPage]:
    import fitz  # PyMuPDF
    doc = fitz.open(str(pdf))
    pages = []
    for i, p in enumerate(doc):
        pix = p.get_pixmap(dpi=110, colorspace=fitz.csGRAY)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
        text = p.get_text().replace("", "").replace("◦", "").replace(" ", " ")
        text = text.replace("\t", " ").replace("ﬃ", "ffi").replace("ﬀ", "ff").replace("ﬁ", "fi").replace("Aon", "tion")
        pages.append(DeckPage(i, " ".join(text.split()), cv2.resize(img, size, interpolation=cv2.INTER_AREA)))
    return pages


class DeckMatcher:
    def __init__(self, pages: list[DeckPage]):
        self.pages = pages
        self.hog = np.array([desc_hog(p.gray) for p in pages])
        self.hog /= np.linalg.norm(self.hog, axis=1, keepdims=True)

    def match(self, slide_gray: np.ndarray) -> tuple[int, float, float]:
        """Return (page index, cosine similarity, margin over the second-best page)."""
        d = desc_hog(cv2.resize(slide_gray, (960, 720)))
        d /= np.linalg.norm(d) + 1e-9
        sims = self.hog @ d
        order = np.argsort(-sims)
        return int(order[0]), float(sims[order[0]]), float(sims[order[0]] - sims[order[1]])
