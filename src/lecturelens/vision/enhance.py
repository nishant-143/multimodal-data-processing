"""Point processing, neighbourhood filtering and restoration for lecture frames.

Point ops (gamma, contrast stretch, histogram equalisation) are written from scratch;
neighbourhood ops use explicit kernels; the Wiener filter is implemented in the frequency domain.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


# ---- point processing ----------------------------------------------------------------------------

def gamma_correct(img: np.ndarray, gamma: float) -> np.ndarray:
    lut = (255 * (np.arange(256) / 255.0) ** gamma).astype(np.uint8)
    return lut[img]


def contrast_stretch(img: np.ndarray, lo_pct=2, hi_pct=98) -> np.ndarray:
    lo, hi = np.percentile(img, [lo_pct, hi_pct])
    out = (img.astype(float) - lo) * 255.0 / max(hi - lo, 1)
    return np.clip(out, 0, 255).astype(np.uint8)


def hist_equalize(img: np.ndarray) -> np.ndarray:
    """Global histogram equalisation via the CDF mapping s = (L-1) * CDF(r)."""
    hist = np.bincount(img.ravel(), minlength=256)
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0][0]
    lut = np.round((cdf - cdf_min) / max(cdf[-1] - cdf_min, 1) * 255).clip(0, 255).astype(np.uint8)
    return lut[img]


def clahe(img: np.ndarray, clip=2.0, tiles=8) -> np.ndarray:
    import cv2
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(img)


def auto_gamma(img: np.ndarray, target_mean=0.5) -> np.ndarray:
    """Choose gamma so the mean intensity maps to target_mean (brightens dark lecture halls)."""
    m = np.clip(img.mean() / 255.0, 1e-3, 0.999)
    return gamma_correct(img, np.log(target_mean) / np.log(m))


# ---- neighbourhood processing ----------------------------------------------------------------------

def gaussian_kernel(size=5, sigma=1.0) -> np.ndarray:
    ax = np.arange(size) - size // 2
    g = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * sigma ** 2))
    return g / g.sum()


def convolve(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    return ndimage.convolve(img.astype(float), k, mode="reflect")


def median_filter(img: np.ndarray, size=3) -> np.ndarray:
    return ndimage.median_filter(img, size=size)


def unsharp_mask(img: np.ndarray, amount=1.0, sigma=1.0) -> np.ndarray:
    blur = convolve(img, gaussian_kernel(int(6 * sigma) | 1, sigma))
    return np.clip(img + amount * (img - blur), 0, 255).astype(np.uint8)


# ---- restoration ---------------------------------------------------------------------------------

def motion_psf(length: int, angle_deg: float, size: int | None = None) -> np.ndarray:
    size = size or (length | 1) + 2
    psf = np.zeros((size, size))
    c = size // 2
    t = np.deg2rad(angle_deg)
    for i in np.linspace(-(length - 1) / 2, (length - 1) / 2, length * 4):
        x, y = int(round(c + i * np.cos(t))), int(round(c - i * np.sin(t)))
        psf[y, x] = 1
    return psf / psf.sum()


def _psf2otf(psf: np.ndarray, shape) -> np.ndarray:
    pad = np.zeros(shape)
    pad[:psf.shape[0], :psf.shape[1]] = psf
    pad = np.roll(pad, (-(psf.shape[0] // 2), -(psf.shape[1] // 2)), axis=(0, 1))
    return np.fft.fft2(pad)


def blur(img: np.ndarray, psf: np.ndarray) -> np.ndarray:
    H = _psf2otf(psf, img.shape)
    return np.real(np.fft.ifft2(np.fft.fft2(img.astype(float)) * H))


def wiener_deconvolve(img: np.ndarray, psf: np.ndarray, K: float = 0.01) -> np.ndarray:
    """F_hat = conj(H) / (|H|^2 + K) * G   (K ~ noise-to-signal power ratio)."""
    H = _psf2otf(psf, img.shape)
    G = np.fft.fft2(img.astype(float))
    F = np.conj(H) / (np.abs(H) ** 2 + K) * G
    return np.clip(np.real(np.fft.ifft2(F)), 0, 255).astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)
