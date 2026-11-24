# LectureLens: seeing, hearing and reading a lecture

LectureLens takes a recorded lecture video and makes it navigable. It reads four signals from the video:

| Signal | What LectureLens extracts |
|---|---|
| **Speech** (audio) | pauses, loudness, pitch, MFCCs; a *smart-speed* version that is 47 % shorter with the same voice |
| **Vision** (frames) | whether a slide, the lecturer or live code is on screen; when the screen changes; slide text (OCR) |
| **Text** (transcript) | "ask the lecture" search, language-model statistics, things *said* but *not on the slide* |
| **Biosignal** (your webcam) | your heart rate and blink rate while you watch (remote photoplethysmography, the POS method) |

All four are aligned on one 10-second timeline and fused to find a lecture's **key moments**. The results
are shown in a **Streamlit dashboard**.

This is a course project for *Multimodal Data Processing*. Every algorithm from the syllabus is written **from
scratch** in NumPy/SciPy: windowing, energy and ZCR, autocorrelation and cepstral pitch, MFCC, overlap-add/WSOLA,
histogram equalisation, Otsu, morphology, Wiener deconvolution, homography, HOG/ORB matching, BPE, TF-IDF, BM25,
n-gram smoothing, Zipf/Heaps fits, and POS rPPG. Libraries are used only for file I/O, the Whisper (speech-to-text)
and Tesseract (OCR) engines, and as reference implementations inside the tests.

---

## 1. Requirements

- macOS or Linux, about 3 GB of free disk space, and an internet connection for the first download
- [uv](https://docs.astral.sh/uv/) (Python package manager). The project pins **Python 3.12**, and uv installs it for you.
- System tools: `ffmpeg` (audio/video) and `tesseract` (OCR)

```bash
brew install ffmpeg tesseract        # macOS (Linux: apt install ffmpeg tesseract-ocr)
```

## 2. Set up

```bash
cd lecturelens
uv sync                              # creates .venv with every Python dependency
uv run pytest                        # 34 tests, about 10 s; all should pass
```

The tests check the from-scratch code against references: MFCC vs librosa, BM25 by hand, OLA reconstruction SNR,
Otsu vs OpenCV, homography vs OpenCV, and heart rate recovered from synthetic pulses.

## 3. Get the data

```bash
bash scripts/download_data.sh        # 3 lectures (≈720 MB) + their slide PDFs
```

The data is three lectures of MIT OpenCourseWare **6.0001 Introduction to CS and Programming in Python (Fall 2016)**,
licensed CC BY-NC-SA 4.0. See `data/raw/SOURCES.md`. Any lecture video works: put an `.mp4` in `data/raw/`. A
matching slide PDF in `data/slides/lec<N>.pdf` is optional; it gives cleaner slide text and an OCR answer key.

## 4. Run the pipeline

```bash
uv run python -m lecturelens.run --smart-speed
```

For each lecture this runs, caching every step in `data/processed/<lecture>/`:

| Step | Output | Time per lecture (M-series laptop, CPU) |
|---|---|---|
| Ingest: 16 kHz mono audio, 1 frame/s, small proxy video | `audio.wav`, `frames/`, `proxy.mp4` | ~1 min |
| Speech-to-text with word timestamps (faster-whisper `small.en`) | `transcript.json` | ~5 min |
| Vision pass: frame type, change detection, optical flow | `frames.pkl` | ~45 s |
| Speech tracks: voice activity, pauses, pitch, MFCC | `speech.npz` | ~20 s |
| Slide segments + OCR + matching to slide-PDF pages | `slides.json` | ~20 s |
| 10 s timeline with features from every modality | `timeline.parquet` | ~30 s |
| Off-slide insights, smart-speed audio | `insights.json`, `smart_speed.wav` | ~20 s |

The first run takes about 25 minutes for all three lectures. Later runs reuse the cache and finish in about 2 minutes.

Then train and evaluate the key-moment models, and regenerate every figure and metric:

```bash
uv run python scripts/eval_fusion.py     # leave-one-lecture-out; writes the learned highlights into each timeline
uv run python scripts/make_figures.py    # reports/figures/*.png and reports/results_*.json
```

## 5. See it

**Terminal demo**
```bash
uv run python scripts/demo.py                                    # lecture 1
uv run python scripts/demo.py --lecture lec10 --query "big O notation"
```

**Dashboard**
```bash
uv run streamlit run app/streamlit_app.py                        # opens http://localhost:8501
```
- **Ask the lecture**: type a question. Each result is a button that seeks the video to that moment.
- **Highlights**: the learned model's top moments. You can switch to hand-weighted fusion with sliders.
- **Timeline**: importance, energy, pitch and visual activity, plus a slide/camera/demo bar, all synchronised.
- **Off-slide insights** and a **smart-speed audio** player.

## 6. Learner-pulse study (optional, needs a webcam)

```bash
uv run python scripts/learner_session.py record --lecture lec01 --minutes 10
#   start Lecture 1 at 0:00 when the countdown ends; face the camera in normal light.
#   Every ~2 min, write your smartwatch/phone heart rate to
#   annotations/learner_lec01_reference_hr.csv   (columns: lecture_time_s,bpm)
uv run python scripts/learner_session.py analyse --lecture lec01
```
This writes the heart-rate error against your watch, and the correlation of your heart rate and blink rate with
the key-moment curve, to `data/processed/lec01_*/learner.json` and `reports/figures/bio_learner.png`.
The video stays on your machine.

---

## Results (3 lectures, 2 h 16 min of video, laptop CPU)

| Modality | What | Result |
|---|---|---|
| Speech | MFCC from scratch vs librosa | max abs. difference 5 × 10⁻⁵ |
| Speech | Smart-speed (pauses → 0.15 s, WSOLA 1.3×) | 43.1 → 22.7 min; pitch 220 → 220 Hz (naive resampling: 276 Hz) |
| Vision | Screen-change detection, lecture 1 (139 hand-checked events) | F1 0.96; slide-build recall 1/6 → 4/6 with shot-aware fusion |
| Vision | OCR vs the slide PDF's own text (124 slides) | 72–73 % character accuracy; denoising *hurts* (−2.5 points) |
| Vision | Wiener deconvolution of a motion-blurred slide | PSNR 19.0 → 24.7 dB, OCR 44 → 66 % |
| Text | Search: 59 slide-title queries | BM25 MRR 0.60 (transcript only), 0.90 (+ slide text) |
| Text | Bigram LM perplexity on an unseen lecture | Laplace 170, add-k 89, Kneser–Ney 62 |
| Fusion | Key-moment average precision, leave-one-lecture-out | text 0.23, early fusion 0.20, chance 0.18; second human 0.65 |

**Main finding:** key moments are about *meaning*, not *emphasis*. Loudness, pitch and pauses barely beat chance.
The words carry most of the signal, and there is a large gap to human agreement. Semantic text features are the
next step.

## Ground truth and its limits

- **Screen changes (lecture 1).** Candidates came from matching frames to the slide PDF and from frame-type changes.
  All 139 were checked by eye on contact sheets. Slide builds that matched the same PDF page may be missing.
- **Key moments.** Each 10 s window was labelled from the transcript using `annotations/LABELING_GUIDE.md`
  (definition, rule, explicit flag, resolution). The labels were made by an LLM annotator. A second, independent
  LLM annotator relabelled lecture 1 (Cohen's κ = 0.74). Spot-check `annotations/labels_*.csv` before relying on
  the numbers. Because the labels come from text, they favour the text modality.
- **OCR accuracy** is measured against the text of the matched PDF page. Slide builds and pen ink cap it near 80 %.

## Project layout

```
src/lecturelens/
  ingest.py            ffmpeg -> audio (loudness-normalised), 1 fps frames, proxy video
  speech/              framing, features (energy/ZCR/VAD), pitch, spectral (STFT, cepstrum, mel, MFCC), ola (OLA/WSOLA)
  vision/              enhance (point ops, filters, Wiener), screen (Canny + homography), slides (hist/HOG/ORB,
                       shot-aware fusion), ocr (Otsu + morphology + Tesseract), deck (slide-PDF matching),
                       activity (frame difference, optical flow), lecture_vision (one streaming pass)
  text/                asr (faster-whisper), normalize (cleaning, lemma vs stem), bpe, retrieval (TF-IDF, BM25),
                       lm (Laplace/add-k/Kneser-Ney), stats (Zipf/Heaps), lecture_text (search, off-slide insights, cues)
  bio/rppg.py          face ROI + skin segmentation, POS, heart rate/HRV, eye-aspect-ratio blinks
  fusion/              align (10 s windows, missing modalities), importance (unimodal/early/late fusion)
  run.py               end-to-end pipeline
app/streamlit_app.py   dashboard
scripts/               download_data.sh, demo.py, eval_fusion.py, make_figures.py, learner_session.py, dump_windows.py
annotations/           ground truth (screen changes, key-moment labels + guide)
reports/               figures/ and results_*.json
tests/                 34 unit tests
```

## Troubleshooting

- **`tesseract` or `ffmpeg` not found**: install them with Homebrew (step 1).
- **The dashboard shows stale highlights after re-running the pipeline**: reload the page. Its cache follows the
  timeline file's modification time.
- **Webcam won't open** (`learner_session.py record`): macOS → System Settings → Privacy & Security → Camera →
  allow your terminal app.
- **First NLTK use is slow**: it downloads WordNet once (~10 MB).
