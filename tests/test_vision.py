import cv2
import numpy as np
import pytest

from lecturelens.vision.enhance import blur, hist_equalize, motion_psf, psnr, wiener_deconvolve
from lecturelens.vision.ocr import char_accuracy, ocr, otsu_threshold
from lecturelens.vision.screen import find_screen, homography_dlt, rectify
from lecturelens.vision.slides import change_prf, detect_changes, frame_distances, fused_scores


def slide(text: str, w=1280, h=720, bg=245) -> np.ndarray:
    img = np.full((h, w), bg, np.uint8)
    for i, line in enumerate(text.split("\n")):
        cv2.putText(img, line, (60, 140 + 90 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 20, 3, cv2.LINE_AA)
    return img


def test_hist_equalize_spreads_dark_image():
    rng = np.random.default_rng(0)
    dark = rng.integers(10, 60, (100, 100)).astype(np.uint8)
    eq = hist_equalize(dark)
    assert eq.min() == 0 and eq.max() == 255


def test_otsu_matches_opencv():
    img = slide("Otsu threshold")
    t_cv, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    assert abs(otsu_threshold(img) - t_cv) <= 1


def test_wiener_improves_psnr_on_motion_blur():
    img = slide("Wiener filter\nrestores blur").astype(float)
    psf = motion_psf(15, 0)
    rng = np.random.default_rng(1)
    blurred = np.clip(blur(img, psf) + rng.normal(0, 1, img.shape), 0, 255).astype(np.uint8)
    restored = wiener_deconvolve(blurred, psf, K=0.01)
    assert psnr(restored, img) > psnr(blurred, img) + 2


def test_homography_dlt_matches_opencv():
    src = np.array([[10, 20], [300, 40], [320, 250], [5, 230]], np.float32)
    dst = np.array([[0, 0], [399, 0], [399, 299], [0, 299]], np.float32)
    H = homography_dlt(src, dst)
    H_cv = cv2.getPerspectiveTransform(src, dst)
    assert np.allclose(H, H_cv / H_cv[2, 2], atol=1e-6)


def test_find_and_rectify_screen_in_synthetic_camera_frame():
    sl = slide("Cepstrum\nmel scale")
    frame = np.full((720, 1280), 40, np.uint8)  # dark lecture hall
    corners = np.array([[250, 120], [1050, 160], [1020, 600], [270, 640]], np.float32)
    H = cv2.getPerspectiveTransform(
        np.array([[0, 0], [1279, 0], [1279, 719], [0, 719]], np.float32), corners)
    warped = cv2.warpPerspective(sl, H, (1280, 720), borderValue=40)
    mask = cv2.warpPerspective(np.ones_like(sl), H, (1280, 720))
    frame[mask > 0] = warped[mask > 0]
    found = find_screen(frame)
    assert found is not None
    assert np.max(np.abs(found - corners)) < 8
    rect = rectify(frame, found)
    assert "cepstrum" in ocr(rect).lower()


def _sequence(make, n=4, per=20, seed=2):
    rng = np.random.default_rng(seed)
    frames, truth = [], []
    for i in range(n):
        if i:
            truth.append(float(len(frames)))
        s = make(i)
        for _ in range(per):
            frames.append(np.clip(s + rng.normal(0, 3, s.shape), 0, 255).astype(np.uint8))
    return frames, np.arange(len(frames), dtype=float), truth


def rich_slide(i):
    s = slide("\n".join(f"Topic {i} point {j}" for j in range(i + 2)))
    if i % 2:
        cv2.rectangle(s, (800, 300), (1200, 650), 120, -1)   # a figure
    return s


@pytest.mark.parametrize("kind", ["hist", "hog", "orb"])
def test_slide_change_detection_on_layout_changes(kind):
    frames, times, truth = _sequence(rich_slide)
    pred = detect_changes(frame_distances(frames, kind), times, k=kind)
    assert change_prf(pred, truth)[2] == pytest.approx(1.0)


def test_text_only_changes_need_local_features():
    """Global descriptors (histogram, HOG thumbnail) miss small text edits; ORB and the fusion catch them."""
    frames, times, truth = _sequence(lambda i: slide(f"Topic {i}\nbullet {i}"))
    d = {k: frame_distances(frames, k) for k in ["hist", "hog", "orb"]}
    f1 = {k: change_prf(detect_changes(v, times, k=k), truth)[2] for k, v in d.items()}
    f1["fused"] = change_prf(detect_changes(fused_scores(d), times, k="fused", already_z=True), truth)[2]
    assert f1["orb"] == pytest.approx(1.0) and f1["fused"] == pytest.approx(1.0)
    assert f1["hist"] < 0.5 and f1["hog"] < 0.5


def test_char_accuracy():
    assert char_accuracy("hello world", "hello world") == 1.0
    assert char_accuracy("helo world", "hello world") == pytest.approx(1 - 1 / 11)
