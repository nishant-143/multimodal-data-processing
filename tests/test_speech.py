import numpy as np
import pytest

from lecturelens.speech.features import vad, zero_crossing_rate
from lecturelens.speech.framing import frame_signal
from lecturelens.speech.ola import ola_reconstruct, remove_pauses, wsola
from lecturelens.speech.pitch import pitch_track
from lecturelens.speech.spectral import mel_filterbank, mfcc

SR = 16000


def harmonic_tone(f0, dur, sr=SR):
    t = np.arange(int(dur * sr)) / sr
    return sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 8))


def test_framing_shape_and_content():
    x = np.arange(1000.0)
    fr = frame_signal(x, 400, 160)
    assert fr.shape[1] == 400
    assert np.array_equal(fr[1, :5], x[160:165])


def test_zcr_of_alternating_signal_is_one():
    assert zero_crossing_rate(np.array([[1.0, -1.0] * 50]))[0] == pytest.approx(1.0)


def test_mel_filterbank_matches_librosa():
    librosa = pytest.importorskip("librosa")
    ours = mel_filterbank(SR, 512, 40)
    ref = librosa.filters.mel(sr=SR, n_fft=512, n_mels=40)
    assert np.allclose(ours, ref, atol=1e-6)


def test_mfcc_matches_librosa():
    librosa = pytest.importorskip("librosa")
    rng = np.random.default_rng(0)
    x = harmonic_tone(150, 1.0) + 0.05 * rng.standard_normal(SR)
    ours = mfcc(x, SR, n_mfcc=13, n_fft=512, hop=160, n_mels=40)
    ref = librosa.feature.mfcc(y=x, sr=SR, n_mfcc=13, n_fft=512, hop_length=160, n_mels=40,
                               window="hann", center=True, pad_mode="constant")
    assert ours.shape == ref.shape
    assert np.max(np.abs(ours - ref)) < 1e-3 * np.max(np.abs(ref))


@pytest.mark.parametrize("method", ["autocorr", "cepstrum"])
def test_pitch_recovers_f0(method):
    _, f0 = pitch_track(harmonic_tone(160, 0.5), SR, method=method)
    voiced = f0[f0 > 0]
    assert len(voiced) > 0.8 * len(f0)
    assert np.median(voiced) == pytest.approx(160, rel=0.03)


def test_ola_perfect_reconstruction():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(SR)
    y = ola_reconstruct(x)
    snr = 10 * np.log10(np.sum(x**2) / np.sum((x - y) ** 2))
    assert snr > 40


def test_wsola_changes_duration_not_pitch():
    x = harmonic_tone(200, 2.0)
    y = wsola(x, 1.5, SR)
    assert len(y) == pytest.approx(len(x) / 1.5, rel=0.02)
    _, f0 = pitch_track(y[SR // 10: -SR // 10], SR)
    assert np.median(f0[f0 > 0]) == pytest.approx(200, rel=0.03)


def test_vad_finds_pause_and_removal_shortens():
    tone = harmonic_tone(150, 1.0)
    rng = np.random.default_rng(2)
    x = np.concatenate([tone, 0.001 * rng.standard_normal(SR), tone])
    res = vad(x, SR)
    assert len(res.pauses) == 1
    s, e = res.pauses[0]
    assert s == pytest.approx(1.0, abs=0.1) and e == pytest.approx(2.0, abs=0.1)
    y, _ = remove_pauses(x, SR, res.pauses, keep_pause_s=0.15)
    assert len(y) / SR == pytest.approx(2.15, abs=0.15)
