"""FFT-based analysis: STFT/spectrogram, real cepstrum, mel filterbank and MFCC (from scratch)."""
from __future__ import annotations

import numpy as np
from scipy.fft import dct

from .framing import frame_signal, window


def stft(x: np.ndarray, n_fft: int, hop: int, win: str = "hann", center: bool = True) -> np.ndarray:
    """Complex STFT, shape (n_fft//2+1, n_frames). center=True reflects-pads like librosa."""
    if center:
        x = np.pad(x, n_fft // 2, mode="constant")
    frames = frame_signal(x, n_fft, hop, pad=False) * window(win, n_fft)
    return np.fft.rfft(frames, axis=1).T


def spectrogram_db(x: np.ndarray, sr: int, n_fft: int = 512, hop: int = 160, win: str = "hann"):
    S = np.abs(stft(x, n_fft, hop, win)) ** 2
    S_db = 10 * np.log10(S + 1e-10)
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    times = np.arange(S.shape[1]) * hop / sr
    return S_db, freqs, times


def real_cepstrum(frame: np.ndarray) -> np.ndarray:
    """c[n] = IFFT(log|FFT(x)|). Low quefrency = vocal-tract envelope, peak at 1/F0 = excitation."""
    spec = np.abs(np.fft.fft(frame))
    return np.real(np.fft.ifft(np.log(spec + 1e-10)))


def hz_to_mel(f):
    """Slaney-style mel scale (linear below 1 kHz, log above) — the librosa default."""
    f = np.asarray(f, dtype=float)
    f_sp = 200.0 / 3
    mels = f / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, mels)


def mel_to_hz(m):
    m = np.asarray(m, dtype=float)
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)


def mel_filterbank(sr: int, n_fft: int, n_mels: int = 40, fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    """Triangular, area-normalised (Slaney) mel filters, shape (n_mels, n_fft//2+1)."""
    fmax = fmax or sr / 2
    fft_freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    mel_pts = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_pts)
    ramps = mel_pts[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    fb = np.maximum(0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_pts[2:n_mels + 2] - mel_pts[:n_mels])
    return fb * enorm[:, None]


def mfcc(x: np.ndarray, sr: int, n_mfcc: int = 13, n_fft: int = 512, hop: int = 160,
         n_mels: int = 40, win: str = "hann") -> np.ndarray:
    """MFCC pipeline: |STFT|^2 -> mel filterbank -> log (dB) -> DCT-II (ortho). Shape (n_mfcc, n_frames)."""
    P = np.abs(stft(x, n_fft, hop, win)) ** 2
    M = mel_filterbank(sr, n_fft, n_mels) @ P
    log_M = 10 * np.log10(np.maximum(M, 1e-10))
    log_M = np.maximum(log_M, log_M.max() - 80.0)  # top_db=80 like librosa.power_to_db
    return dct(log_M, type=2, axis=0, norm="ortho")[:n_mfcc]
