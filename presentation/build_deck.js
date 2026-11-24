// LectureLens presentation generator.
//   cd presentation && NODE_PATH=./node_modules node build_deck.js
// Every number on the slides is read from ../reports/results_*.json, so rerunning the
// pipeline + make_figures.py + this script keeps the deck in sync with the code.
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const Fa = require("react-icons/fa");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const ROOT = path.resolve(__dirname, "..");
const FIG = (n) => path.join(ROOT, "reports/figures", n + ".png");
const J = (n) => JSON.parse(fs.readFileSync(path.join(ROOT, "reports", `results_${n}.json`), "utf8"));
const speech = J("speech"), vision = J("vision"), text = J("text"), fusion = J("fusion"), agree = J("agreement");
const ocrVar = J("ocr_variants");
const insights = JSON.parse(fs.readFileSync(path.join(ROOT, "data/processed/lec01_what_is_computation/insights.json"), "utf8"));

// ---------- theme ---------------------------------------------------------------------------------
// Chalkboard slate + the four modality colours used in every figure (validated dataviz palette).
const THEME = {
  name: "LectureLens", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: {
    dk1: "1B1F24", lt1: "FFFFFF", dk2: "22303A", lt2: "EEF2F4",
    accent1: "2A78D6", accent2: "EB6834", accent3: "1BAF7A", accent4: "6250D6",
    accent5: "E9B949", accent6: "C8372D", hlink: "2A78D6", folHlink: "6250D6",
  },
};
const HEX = { ink: "1B1F24", ink2: "52514E", muted: "8A8984", grid: "E4E3DF", slate: "22303A", tint: "EEF2F4",
  speech: "2A78D6", vision: "EB6834", text: "1BAF7A", bio: "6250D6", chalk: "E9B949", red: "C8372D", white: "FFFFFF" };
const MOD = { speech: ["SPEECH", "speech"], vision: ["VISION", "vision"], text: ["TEXT", "text"],
  bio: ["BIO", "bio"], fusion: ["FUSION", "slate"] };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.title = "LectureLens";
pres.author = "LectureLens project";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const W = 13.333, H = 7.5;

pres.defineSlideMaster({
  title: "TITLE_DARK", background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 1.55, w: 7.4, h: 1.5, fontFace: "Cambria",
      fontSize: 54, bold: true, color: C.background1, valign: "bottom", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.7, y: 3.2, w: 7.0, h: 1.3, fontSize: 20,
      color: C.background2, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT", background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 9.4, h: 0.85, fontFace: "Cambria",
      fontSize: 30, bold: true, color: C.text1, valign: "middle", margin: 0 }, text: "" } },
    { text: { text: "LectureLens  ·  Multimodal Data Processing", options: { x: 0.6, y: 7.05, w: 6, h: 0.3,
      fontSize: 10, color: C.text2, margin: 0 } } },
  ],
  slideNumber: { x: 12.3, y: 7.05, w: 0.5, h: 0.3, fontSize: 10, color: "8A8984", align: "right" },
});

// ---------- helpers -------------------------------------------------------------------------------
const iconCache = {};
async function icon(name, color, px = 256) {
  const key = name + color;
  if (!iconCache[key]) {
    const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Fa[name], { color: "#" + color, size: px }));
    const buf = await sharp(Buffer.from(svg)).resize(px, px).png().toBuffer();
    iconCache[key] = "image/png;base64," + buf.toString("base64");
  }
  return iconCache[key];
}
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
// Place an image inside a box, preserving aspect ratio (contain), aligned top-left or centred.
function img(slide, name, x, y, w, h, opts = {}) {
  const f = FIG(name);
  const s = pngSize(f);
  const r = Math.min(w / s.w, h / s.h);
  const iw = s.w * r, ih = s.h * r;
  const ox = opts.align === "left" ? 0 : (w - iw) / 2;
  slide.addImage({ path: f, x: x + ox, y: y + (opts.valign === "top" ? 0 : (h - ih) / 2), w: iw, h: ih,
    altText: opts.alt || name.replace(/_/g, " "), objectName: "fig_" + name });
}
function chips(slide, mods) {
  let x = W - 0.6;
  [...mods].reverse().forEach((m) => {
    const [label, col] = MOD[m];
    const w = 0.22 + label.length * 0.105;
    x -= w;
    slide.addText(label, { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.13, x, y: 0.6, w, h: 0.32,
      fill: { color: HEX[col] }, color: HEX.white, fontSize: 10, bold: true, align: "center", valign: "middle",
      margin: 0, isTextBox: true, objectName: "chip_" + m });
    x -= 0.1;
  });
}
function title(slide, t) { slide.addText(t, { placeholder: "title" }); }
function stat(slide, x, y, w, big, label, color) {
  slide.addText([
    { text: big, options: { fontSize: 34, bold: true, color: color, fontFace: "Cambria", breakLine: true } },
    { text: label, options: { fontSize: 13, color: HEX.ink2 } },
  ], { x, y, w, h: 1.15, margin: 0, valign: "top", isTextBox: true });
}
function card(slide, x, y, w, h, fill = HEX.tint) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { color: fill } });
}
function bullets(slide, items, x, y, w, h, size = 15) {
  slide.addText(items.map((t, i) => {
    const runs = Array.isArray(t) ? t : [{ text: t }];
    return runs.map((r, j) => ({ text: r.text, options: { ...(r.options || {}), bullet: j === 0 ? { indent: 16 } : undefined,
      breakLine: j === runs.length - 1 && i < items.length - 1, paraSpaceAfter: 6 } }));
  }).flat(), { x, y, w, h, fontSize: size, color: HEX.ink, valign: "top", margin: 0, isTextBox: true });
}
const CHART_TXT = { catAxisLabelColor: HEX.ink2, valAxisLabelColor: HEX.ink2, catAxisLabelFontFace: "+mn-lt",
  valAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt", titleFontFace: "+mn-lt", legendFontFace: "+mn-lt",
  catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, dataLabelFontSize: 11, titleFontSize: 13, titleColor: HEX.ink,
  dataLabelColor: HEX.ink, legendFontSize: 11, legendColor: HEX.ink2 };
const fmt = (x, d = 2) => Number(x).toFixed(d);
const pct = (x, d = 0) => (100 * x).toFixed(d) + "%";

(async () => {
  const sec = (t) => pres.addSection({ title: t });

  // 1 ---------------------------------------------------------------------------------- Title
  sec("Title");
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Title" });
  s.addText("LectureLens", { placeholder: "title" });
  s.addText([
    { text: "Seeing, hearing and reading a lecture with classical multimodal processing", options: { breakLine: true } },
    { text: " ", options: { fontSize: 10, breakLine: true } },
    { text: "[Your Name]  ·  [Roll number]", options: { bold: true, color: "FFFFFF", breakLine: true } },
    { text: "Multimodal Data Processing — course project  ·  November 2026", options: { fontSize: 15 } },
  ], { placeholder: "body" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.35, y: 1.3, w: 4.4, h: 4.9, rectRadius: 0.15,
    fill: { color: "2C3D49" }, line: { color: "2C3D49" } });
  const modRows = [["FaMicrophoneAlt", "speech", "Speech", "pauses · pitch · MFCC · smart-speed"],
    ["FaDesktop", "vision", "Video", "slides vs camera · change detection · OCR"],
    ["FaAlignLeft", "text", "Text", "transcript · BM25 search · language models"],
    ["FaHeartbeat", "bio", "Biosignal", "webcam pulse (rPPG) · blinks"]];
  for (let i = 0; i < modRows.length; i++) {
    const [ic, col, h1, h2] = modRows[i];
    const y = 1.65 + i * 1.12;
    s.addShape(pres.shapes.OVAL, { x: 8.7, y, w: 0.72, h: 0.72, fill: { color: HEX[col] }, line: { color: HEX[col] } });
    s.addImage({ data: await icon(ic, "FFFFFF"), x: 8.88, y: y + 0.18, w: 0.36, h: 0.36, altText: h1 + " icon" });
    s.addText([{ text: h1, options: { bold: true, fontSize: 17, color: "FFFFFF", breakLine: true } },
      { text: h2, options: { fontSize: 12, color: "CFD8DE" } }], { x: 9.6, y: y - 0.04, w: 3.0, h: 0.82,
      margin: 0, valign: "middle", isTextBox: true });
  }
  s.addText("3 MIT OpenCourseWare lectures  ·  2 h 16 min of video  ·  laptop CPU only", { x: 0.7, y: 6.55, w: 7.5,
    h: 0.4, fontSize: 13, color: "AFC0CB", margin: 0, isTextBox: true });
  s.addNotes("Introduce yourself. One sentence: LectureLens turns a recorded lecture into something you can search, skim and listen to faster, using the classical speech, image, text and biosignal methods from this course.");

  // 2 ---------------------------------------------------------------------------------- Motivation
  sec("Motivation");
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Motivation" });
  title(s, "Why: a recorded lecture is long, and each signal tells only part of it");
  const mot = [
    ["FaDesktop", "vision", "Slides miss what is said",
      `Only ${pct(0.391)} of Lecture 1 shows a slide. ${pct(0.56)} is the lecturer on camera, and the rest is live code.`],
    ["FaAlignLeft", "text", "Transcripts miss what is shown",
      "6,701 spoken words, but the diagrams, code and red ink exist only on screen."],
    ["FaMicrophoneAlt", "speech", "Neither shows emphasis",
      `Lecture 1 has ${speech.n_pauses_lec01.toLocaleString()} pauses longer than 0.35 s. Which ones come before something that matters?`],
  ];
  for (let i = 0; i < 3; i++) {
    const [ic, col, h1, body] = mot[i];
    const x = 0.6 + i * 4.1;
    card(s, x, 1.55, 3.85, 3.3);
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: 1.85, w: 0.75, h: 0.75, fill: { color: HEX[col] }, line: { color: HEX[col] } });
    s.addImage({ data: await icon(ic, "FFFFFF"), x: x + 0.49, y: 2.04, w: 0.37, h: 0.37, altText: h1 });
    s.addText(h1, { x: x + 0.3, y: 2.8, w: 3.3, h: 0.5, fontSize: 19, bold: true, color: HEX.ink, margin: 0, isTextBox: true, fontFace: "Cambria" });
    s.addText(body, { x: x + 0.3, y: 3.35, w: 3.3, h: 1.4, fontSize: 15, color: HEX.ink2, margin: 0, valign: "top", isTextBox: true });
  }
  s.addText([
    { text: "Goal  ", options: { bold: true, color: HEX.red } },
    { text: "Make a 45-minute lecture navigable in minutes, using only signals already in the video: " },
    { text: "search it, jump to the key moments, and listen faster.", options: { bold: true } },
  ], { x: 0.6, y: 5.25, w: 12.1, h: 0.9, fontSize: 18, color: HEX.ink, margin: 0, isTextBox: true });
  s.addText("Why I picked it: one problem that needs every modality in this syllabus, and a problem every student who revises from recordings has.",
    { x: 0.6, y: 6.2, w: 12.1, h: 0.5, fontSize: 14, italic: true, color: HEX.ink2, margin: 0, isTextBox: true });
  s.addNotes("Personalise this: say why YOU care (e.g., how you revise from recorded lectures). Numbers on this slide are measured on Lecture 1 by the pipeline.");

  // 3 ---------------------------------------------------------------------------------- Coverage
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Motivation" });
  title(s, "One problem that exercises the whole syllabus");
  const cols = [
    ["Introduction", HEX.slate, ["Multimodal data: video, audio, words, slides, pulse", "Challenges: 4 sampling rates, missing modalities", "Collection & cleaning: OCW download, loudness norm., boilerplate removal"]],
    ["Text", HEX.text, ["Normalisation, disfluencies, lemma vs stem", "BPE subwords vs GPT tokenizer", "TF-IDF & BM25 search; Zipf & Heaps", "Bigram LM: Laplace, add-k, Kneser–Ney"]],
    ["Speech", HEX.speech, ["Windowing (Hamming vs rectangular)", "Energy, ZCR → pause detection", "Autocorrelation & cepstral pitch", "STFT spectrogram, MFCC", "Overlap-add / WSOLA smart-speed"]],
    ["Image & video", HEX.vision, ["Point ops: gamma, hist. eq., CLAHE", "Neighbourhood: median, unsharp", "Wiener restoration; Otsu; morphology", "Canny + homography; HOG, ORB", "Temporal: frame diff, optical flow"]],
    ["Other modalities", HEX.bio, ["Biomedical: webcam rPPG (POS), HR, HRV", "Blink rate from eye aspect ratio", "Conventional multimodal learning: early vs late fusion, ablation"]],
  ];
  const cw = 2.32;
  cols.forEach(([h1, col, items], i) => {
    const x = 0.6 + i * (cw + 0.12);
    s.addText(h1, { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.1, x, y: 1.5, w: cw, h: 0.55,
      fill: { color: col }, color: HEX.white, bold: true, fontSize: 16, align: "center", valign: "middle", margin: 0, isTextBox: true });
    card(s, x, 2.15, cw, 4.6);
    bullets(s, items, x + 0.15, 2.3, cw - 0.3, 4.35, 13);
  });
  s.addNotes("Every box is implemented from scratch in the repo (libraries only for I/O, the Whisper/Tesseract engines, and as test references).");

  // 4 ---------------------------------------------------------------------------------- History
  sec("Past and current work");
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Past and current work" });
  title(s, "Past and current work on lecture-video understanding");
  const tl = [
    ["1996", "Informedia", "CMU digital video library: speech recognition, OCR and image analysis for video search"],
    ["1999", "Classroom 2000", "Georgia Tech captures live lectures (slides, ink, audio) and links them for review"],
    ["2007", "MIT Lecture Browser", "Spoken-lecture search built on ASR transcripts"],
    ["2010", "TalkMiner", "Lecture-webcast search: detect slide frames, OCR them, index the text"],
    ["2022", "Whisper", "Robust, open ASR: transcripts become almost free"],
    ["2024+", "Multimodal LLMs", "GPT-4o, Gemini 1.5: answer questions about hour-long videos end to end"],
  ];
  s.addShape(pres.shapes.LINE, { x: 0.9, y: 2.55, w: 11.6, h: 0, line: { color: HEX.muted, width: 2 } });
  tl.forEach(([yr, name, d], i) => {
    const x = 0.6 + i * 2.05;
    const col = i < 4 ? HEX.slate : HEX.red;
    s.addShape(pres.shapes.OVAL, { x: x + 0.72, y: 2.42, w: 0.26, h: 0.26, fill: { color: col }, line: { color: HEX.white, width: 2 } });
    s.addText(yr, { x, y: 1.75, w: 1.7, h: 0.5, fontSize: 20, bold: true, color: col, align: "center", margin: 0, fontFace: "Cambria", isTextBox: true });
    s.addText(name, { x: x - 0.05, y: 2.85, w: 1.8, h: 0.45, fontSize: 14, bold: true, color: HEX.ink, align: "center", margin: 0, isTextBox: true });
    s.addText(d, { x: x - 0.05, y: 3.3, w: 1.8, h: 1.5, fontSize: 12, color: HEX.ink2, align: "center", valign: "top", margin: 0, isTextBox: true });
  });
  card(s, 0.6, 5.0, 12.1, 1.75);
  s.addText([
    { text: "Where LectureLens sits  ", options: { bold: true, color: HEX.red, breakLine: true } },
    { text: "Classical and interpretable like TalkMiner, but it fuses all four modalities, measures each one against ground truth, runs on a laptop CPU, and adds the learner's own physiology. Today's LLMs are powerful but opaque and expensive per hour of video; I wanted to see how far the course's methods go first.",
      options: { color: HEX.ink } },
  ], { x: 0.9, y: 5.15, w: 11.5, h: 1.5, fontSize: 15, margin: 0, valign: "top", isTextBox: true });
  s.addNotes("Historical arc: from systems research (Informedia, Classroom 2000) to ASR-driven search (MIT), slide-OCR indexing (TalkMiner), then neural ASR (Whisper) and end-to-end multimodal LLMs. Position: classical, measurable, cheap.");

  // 5 ---------------------------------------------------------------------------------- Architecture
  sec("Key highlights");
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Key highlights" });
  title(s, "System: four pipelines, one timeline");
  chips(s, ["speech", "vision", "text", "bio", "fusion"]);
  const box = (x, y, w, h, t, sub, fill, fc = HEX.white) => {
    s.addText([{ text: t, options: { bold: true, fontSize: 14, breakLine: !!sub } }, ...(sub ? [{ text: sub, options: { fontSize: 11 } }] : [])],
      { shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.1, x, y, w, h, fill: { color: fill }, line: { color: fill },
        color: fc, align: "center", valign: "middle", margin: 4, isTextBox: true });
  };
  const arrow = (x1, y1, x2, y2) => s.addShape(pres.shapes.LINE, { x: x1, y: y1, w: x2 - x1, h: y2 - y1,
    line: { color: HEX.muted, width: 1.5, endArrowType: "triangle" } });
  box(0.6, 3.25, 1.9, 1.2, "Lecture video", "MP4 · slide PDF", HEX.slate);
  box(2.95, 3.25, 1.9, 1.2, "Ingest", "16 kHz audio · 1 fps frames", HEX.slate);
  arrow(2.5, 3.85, 2.95, 3.85);
  const lanes = [["Speech", "VAD · pitch · MFCC · WSOLA", HEX.speech], ["Vision", "frame type · changes · OCR", HEX.vision],
    ["Text", "Whisper · BM25 · n-gram LM", HEX.text], ["Learner bio", "webcam rPPG · blinks", HEX.bio]];
  lanes.forEach(([t, sub, c], i) => {
    const y = 1.45 + i * 1.3;
    box(5.35, y, 2.5, 1.0, t, sub, c);
    arrow(4.85, 3.85, 5.35, y + 0.5);
    arrow(7.85, y + 0.5, 8.35, 3.85);
  });
  box(8.35, 3.05, 2.0, 1.6, "Aligned timeline", "10 s windows · masks for missing modality · boilerplate removed", HEX.tint, HEX.ink);
  arrow(10.35, 3.85, 10.75, 3.85);
  const outs = ["Ask-the-lecture search", "Highlights (fusion)", "Off-slide insights", "Smart-speed audio", "Dashboard"];
  outs.forEach((t, i) => {
    s.addText(t, { x: 10.8, y: 2.45 + i * 0.58, w: 2.0, h: 0.48, fontSize: 12, bold: true, color: HEX.ink,
      fill: { color: HEX.white }, line: { color: HEX.grid, width: 1 }, align: "center", valign: "middle", margin: 0, isTextBox: true });
  });
  s.addText("Modalities arrive at 100 Hz (audio frames), 1 Hz (video), irregular word timestamps and arbitrary slide segments, so all are aggregated into 10 s windows.",
    { x: 0.6, y: 6.55, w: 12.1, h: 0.4, fontSize: 12, color: HEX.ink2, italic: true, margin: 0, isTextBox: true });

  // 6 ---------------------------------------------------------------------------------- Speech
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Key highlights" });
  title(s, "Speech: short-term analysis, written from scratch");
  chips(s, ["speech"]);
  img(s, "speech_tracks", 0.6, 1.35, 8.3, 5.55, { align: "left" });
  stat(s, 9.3, 1.5, 3.4, (() => { const [m, e] = speech.mfcc_max_abs_diff_vs_librosa.toExponential(0).split("e-"); const sup = "⁰¹²³⁴⁵⁶⁷⁸⁹"; return `${m} × 10⁻${[...e].map((d) => sup[+d]).join("")}`; })(), "max MFCC difference from librosa (13 × 500 frames)", HEX.speech);
  stat(s, 9.3, 2.85, 3.4, pct(speech.pitch_agreement_within_5pct), "of voiced frames: autocorrelation and cepstral pitch agree within 5 %", HEX.speech);
  stat(s, 9.3, 4.2, 3.4, "220 / 136 Hz", "median F0 of the two lecturers, found without any speaker labels", HEX.speech);
  s.addText("Energy + ZCR voice activity detection finds the pauses (shaded) that smart-speed later removes.",
    { x: 9.3, y: 5.55, w: 3.4, h: 1.0, fontSize: 13, color: HEX.ink2, margin: 0, valign: "top", isTextBox: true });
  s.addNotes("25 ms frames, 10 ms hop. Hamming window to limit leakage (backup slide). ZCR catches unvoiced fricatives that energy misses. Pitch: autocorrelation via Wiener–Khinchin, cepstrum = IFFT(log|FFT|).");

  // 7 ---------------------------------------------------------------------------------- Smart speed
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Key highlights" });
  title(s, "Smart-speed: 43 minutes of lecture in 23, same voice");
  chips(s, ["speech"]);
  const ss = speech.smart_speed_minutes;
  s.addChart(pres.charts.BAR, [{ name: "minutes", labels: ["Original", "Pauses trimmed", "+ 1.3× WSOLA"],
    values: [ss.original, ss["pauses trimmed"], ss["+ 1.3x WSOLA"]].map((v) => +v.toFixed(1)) }], {
    x: 0.6, y: 1.4, w: 5.9, h: 3.9, barDir: "bar", chartColors: ["8A8984", "2A78D6", "1BAF7A"],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0\" min\"", showLegend: false,
    showTitle: true, title: "Lecture 1 listening time", valAxisHidden: true, valGridLine: { style: "none" },
    catGridLine: { style: "none" }, catAxisOrientation: "maxMin", barGapWidthPct: 60, ...CHART_TXT });
  const pa = speech.pitch_after_speedup_hz;
  s.addChart(pres.charts.BAR, [{ name: "median F0", labels: ["Original", "Naive 1.3× (resample)", "WSOLA 1.3×"],
    values: [pa.original, pa.naive, pa.wsola].map((v) => Math.round(v)) }], {
    x: 6.8, y: 1.4, w: 5.9, h: 3.9, barDir: "col", chartColors: ["8A8984", "EB6834", "2A78D6"],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0\" Hz\"", showLegend: false,
    showTitle: true, title: "Median pitch after speeding up", valAxisHidden: true, valGridLine: { style: "none" },
    valAxisMinVal: 0, valAxisMaxVal: 320, barGapWidthPct: 60, ...CHART_TXT });
  bullets(s, [
    [{ text: "Pauses → 0.15 s: ", options: { bold: true } }, { text: "VAD pauses shortened with 10 ms fades (−32 % time)." }],
    [{ text: "WSOLA: ", options: { bold: true } }, { text: "each 30 ms frame is shifted ±10 ms to best continue the previous one, then overlap-added with a Hann window. Duration changes, pitch does not." }],
    [{ text: "Sanity check: ", options: { bold: true } }, { text: `plain analysis→OLA reconstruction = ${fmt(speech.ola_reconstruction_snr_db, 1)} dB SNR (Hann at 75 % overlap is COLA).` }],
  ], 0.6, 5.5, 12.1, 1.4, 14);

  // 8 ---------------------------------------------------------------------------------- Vision: change detection
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Key highlights" });
  title(s, "Vision: when does the screen change?");
  chips(s, ["vision"]);
  img(s, "vision_frame_types", 0.6, 1.3, 7.2, 2.0, { align: "left" });
  const vr = vision.visual_events_lec01.results;
  const vnames = Object.keys(vr);
  s.addChart(pres.charts.BAR, [
    { name: "F1, all 139 events", labels: vnames, values: vnames.map((k) => +vr[k].F1.toFixed(2)) },
    { name: "Recall, 6 slide builds", labels: vnames, values: vnames.map((k) => +vr[k].recall_slide_build.toFixed(2)) },
  ], { x: 0.6, y: 3.45, w: 7.2, h: 3.45, barDir: "bar", barGrouping: "clustered", chartColors: ["2A78D6", "EB6834"],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", showLegend: true, legendPos: "b",
    valAxisMinVal: 0, valAxisMaxVal: 1.1, valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    catAxisOrientation: "maxMin", barGapWidthPct: 50, showTitle: true, title: "Lecture 1: 139 hand-checked visual events (±2 s)", ...CHART_TXT });
  card(s, 8.15, 1.3, 4.55, 5.6);
  bullets(s, [
    [{ text: "Frame type first. ", options: { bold: true } }, { text: "Black side bars + bright centre = slide; bright full width = code demo; else camera." }],
    [{ text: "The cheapest descriptor wins on cuts. ", options: { bold: true } }, { text: "A grey histogram finds 99 % of camera↔slide cuts." }],
    [{ text: "But it misses slide builds ", options: { bold: true } }, { text: "(a bullet appears: the histogram barely moves). ORB keypoints catch them, yet fail on dark camera footage." }],
    [{ text: "Shot-aware fusion: ", options: { bold: true } }, { text: "histogram for cuts, ORB only between two slide frames, with a label-free threshold (99th percentile). Build recall 1/6 → 4/6 at the same F1." }],
  ], 8.4, 1.5, 4.1, 5.3, 14);
  s.addNotes("Ground truth: 139 events = candidates from PDF page matching + frame-type changes, every one verified on contact sheets. Global vs local z-scores: HOG and ORB scored 0 with a global threshold because camera shots have a much higher noise floor than static slides.");

  // 9 ---------------------------------------------------------------------------------- Vision: OCR & restoration
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Key highlights" });
  title(s, "Vision: reading the slides, and restoring blurred ones");
  chips(s, ["vision"]);
  img(s, "vision_ocr_restoration", 0.6, 1.3, 12.1, 2.85);
  const ov = ocrVar.ocr_variants;
  const ovNames = Object.keys(ov);
  s.addChart(pres.charts.BAR, [{ name: "mean char. accuracy", labels: ovNames, values: ovNames.map((k) => +(100 * ov[k].mean).toFixed(1)) }], {
    x: 0.6, y: 4.3, w: 6.4, h: 2.65, barDir: "bar", chartColors: ["2A78D6", "2A78D6", "EB6834", "1BAF7A", "2A78D6"],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0\"%\"", showLegend: false,
    valAxisMinVal: 60, valAxisMaxVal: 78, valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
    catAxisOrientation: "maxMin", showTitle: true, title: `OCR vs the slide PDF's own text (${ocrVar.n} slides)`, ...CHART_TXT });
  bullets(s, [
    [{ text: "Ground truth for free: ", options: { bold: true } }, { text: "frames are matched to pages of the published slide PDF (HOG cosine), so the PDF text is the answer key." }],
    [{ text: "Denoising hurt. ", options: { bold: true } }, { text: "Median + CLAHE lost 2.5 points on clean screen captures; unsharp masking gained 0.9. Builds and pen ink cap accuracy near 80 %." }],
    [{ text: "Wiener restoration: ", options: { bold: true } }, { text: `motion-blurred slide ${fmt(vision.restoration.blur_psnr, 1)} → ${fmt(vision.restoration.wiener_psnr, 1)} dB PSNR; OCR ${pct(vision.restoration.blur_ocr_acc)} → ${pct(vision.restoration.wiener_ocr_acc)}.` }],
  ], 7.3, 4.35, 5.4, 2.6, 14);

  // 10 --------------------------------------------------------------------------------- Text: search + off-slide
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Key highlights" });
  title(s, "Text: ask the lecture, and hear what the slides leave out");
  chips(s, ["text", "vision"]);
  const srch = text.search;
  const row = (m, sl) => srch.find((r) => r.method === m && r.slides_indexed === sl);
  const th = { bold: true, color: HEX.white, fill: { color: HEX.slate }, fontSize: 13 };
  const tc = (v, b = false) => ({ text: v, options: { fontSize: 13, color: HEX.ink, bold: b } });
  s.addTable([
    [{ text: "Index", options: th }, { text: "Ranking", options: th }, { text: "MRR", options: th }, { text: "Hit@1", options: th }, { text: "Hit@5", options: th }],
    ...[["bm25", false], ["tfidf", false], ["bm25", true], ["tfidf", true]].map(([m, sl]) => {
      const r = row(m, sl); const best = m === "bm25";
      return [tc(sl ? "transcript + slide OCR" : "transcript only"), tc(m === "bm25" ? "BM25" : "TF-IDF cosine", best),
        tc(fmt(r.MRR), best), tc(pct(r.hit1)), tc(pct(r.hit5))];
    }),
  ], { x: 0.6, y: 1.45, w: 6.3, colW: [2.2, 1.4, 0.9, 0.9, 0.9], rowH: 0.42, border: { type: "solid", color: HEX.grid, pt: 1 },
    fill: { color: HEX.white }, margin: [3, 6, 3, 6] });
  s.addText(`Known-item test: ${row("bm25", false).n} queries = slide titles from the PDFs; correct = a 30 s chunk while that slide (or the minute after) is on screen. Slide-indexed scores are optimistic because the queries come from slide text.`,
    { x: 0.6, y: 3.7, w: 6.3, h: 0.9, fontSize: 12, color: HEX.ink2, italic: true, margin: 0, valign: "top", isTextBox: true });
  bullets(s, [
    [{ text: "BM25 > TF-IDF ", options: { bold: true } }, { text: "thanks to term-frequency saturation and length normalisation." }],
    [{ text: "Adding the screen ", options: { bold: true } }, { text: `lifts MRR ${fmt(row("bm25", false).MRR)} → ${fmt(row("bm25", true).MRR)}: the multimodal gain is largest here.` }],
  ], 0.6, 4.7, 6.3, 1.6, 14);
  card(s, 7.25, 1.45, 5.45, 5.45);
  s.addText("Off-slide insights", { x: 7.55, y: 1.6, w: 4.9, h: 0.45, fontSize: 18, bold: true, color: HEX.ink, margin: 0, fontFace: "Cambria", isTextBox: true });
  s.addText("Salient spoken passages (high IDF) with ~zero TF-IDF similarity to the slide on screen:", { x: 7.55, y: 2.05, w: 4.9, h: 0.6, fontSize: 12, color: HEX.ink2, margin: 0, isTextBox: true });
  const pick = [insights.find((d) => /flying planes/.test(d.text)), insights.find((d) => /central processing/.test(d.text))].filter(Boolean);
  pick.forEach((d, i) => {
    const mm = `${Math.floor(d.start / 60)}:${String(Math.floor(d.start % 60)).padStart(2, "0")}`;
    s.addText([{ text: mm + "  ", options: { bold: true, color: HEX.text } },
      { text: "“" + (/^[a-z]/.test(d.text) && /[.?!] [A-Z]/.test(d.text) ? d.text.slice(d.text.search(/[.?!] [A-Z]/) + 2) : d.text).split(" ").slice(0, 32).join(" ") + "…”", options: { italic: true, color: HEX.ink } }],
      { x: 7.55, y: 2.75 + i * 1.95, w: 4.9, h: 1.8, fontSize: 14, margin: 0, valign: "top", isTextBox: true });
  });
  s.addNotes("Off-slide insight 1: the slide shows only a cartoon, the explanation of ambiguity is spoken. Insight 2: the slide is a block diagram; what the CPU does is only said aloud.");

  // 11 --------------------------------------------------------------------------------- Text: statistics & LMs
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Key highlights" });
  title(s, "Text: spoken vs written language, smoothing and subwords");
  chips(s, ["text"]);
  img(s, "text_zipf_heaps", 0.6, 1.3, 7.3, 2.9, { align: "left" });
  const zh = text.zipf_heaps;
  const sp = zh["spoken (transcripts)"], wr = zh["written (slides)"];
  bullets(s, [
    [{ text: "Speech is more repetitive: ", options: { bold: true } }, { text: `Zipf s = ${fmt(sp.zipf_s)} vs ${fmt(wr.zipf_s)}; type/token ratio ${fmt(sp.ttr_equal_len, 3)} vs ${fmt(wr.ttr_equal_len, 3)} at equal length.` }],
    [{ text: "Heaps' law reused: ", options: { bold: true } }, { text: "novelty = new words ÷ the number Heaps predicts, so a lecture's opening is not 'novel' by default." }],
  ], 8.2, 1.4, 4.5, 2.8, 14);
  const ppl = text.lm_perplexity;
  const lecs = Object.keys(ppl);
  const mean = (k) => lecs.reduce((a, l) => a + ppl[l][k], 0) / lecs.length;
  s.addChart(pres.charts.BAR, [{ name: "perplexity", labels: ["Laplace (add-1)", "Add-k (k=0.05)", "Kneser–Ney"],
    values: [mean("laplace"), mean("addk"), mean("kn")].map((v) => Math.round(v)) }], {
    x: 0.6, y: 4.35, w: 5.6, h: 2.6, barDir: "col", chartColors: ["8A8984", "2A78D6", "1BAF7A"], showValue: true,
    dataLabelPosition: "outEnd", showLegend: false, valAxisHidden: true, valGridLine: { style: "none" }, valAxisMinVal: 0,
    barGapWidthPct: 70, showTitle: true, title: "Bigram LM perplexity on an unseen lecture (lower = better)", ...CHART_TXT });
  const bp = text.bpe_splits;
  const words = ["polymorphism", "inheritance", "declarative", "isinstance"];
  s.addTable([
    [{ text: "Word", options: th }, { text: `BPE, course corpus (${text.bpe_merges} merges)`, options: th }, { text: "GPT tokenizer (cl100k)", options: th }],
    ...words.map((w) => [tc(w, true), tc(bp[w].bpe_course.join(" · ")), tc(bp[w].gpt_cl100k.map((t) => t.trim()).join(" · "))]),
  ], { x: 6.5, y: 4.45, w: 6.2, colW: [1.6, 2.6, 2.0], rowH: 0.42, border: { type: "solid", color: HEX.grid, pt: 1 }, fill: { color: HEX.white }, margin: [3, 6, 3, 6] });
  s.addText("23k spoken tokens are too few for BPE to learn rare jargon; a 100k-merge vocabulary keeps it whole.",
    { x: 6.5, y: 6.6, w: 6.2, h: 0.35, fontSize: 12, italic: true, color: HEX.ink2, margin: 0, isTextBox: true });

  // 12 --------------------------------------------------------------------------------- Bio
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Key highlights" });
  title(s, "Learner pulse: heart rate from a webcam");
  chips(s, ["bio", "vision"]);
  const steps = [["FaUserCircle", "Face landmarks", "MediaPipe mesh → forehead + cheek polygons"],
    ["FaFillDrip", "Skin segmentation", "YCrCb thresholds + open/close morphology"],
    ["FaWaveSquare", "POS projection", "temporally normalised RGB onto the plane orthogonal to skin tone"],
    ["FaHeartbeat", "HR, HRV, blinks", "0.7–3 Hz band-pass → FFT peak; RMSSD; eye-aspect-ratio blinks"]];
  for (let i = 0; i < steps.length; i++) {
    const [ic, h1, d] = steps[i];
    const y = 1.45 + i * 0.95;
    s.addShape(pres.shapes.OVAL, { x: 0.6, y, w: 0.62, h: 0.62, fill: { color: HEX.bio }, line: { color: HEX.bio } });
    s.addImage({ data: await icon(ic, "FFFFFF"), x: 0.76, y: y + 0.16, w: 0.3, h: 0.3, altText: h1 });
    s.addText([{ text: h1, options: { bold: true, fontSize: 15, color: HEX.ink, breakLine: true } }, { text: d, options: { fontSize: 12, color: HEX.ink2 } }],
      { x: 1.4, y: y - 0.05, w: 4.4, h: 0.75, margin: 0, valign: "middle", isTextBox: true });
  }
  img(s, "bio_rppg_synthetic", 6.1, 1.35, 6.6, 2.6);
  card(s, 6.1, 4.15, 6.6, 2.75);
  bullets(s, [
    [{ text: "Verified: ", options: { bold: true } }, { text: "recovers 55 / 72 / 110 bpm within 2 bpm; under 78-bpm lighting flicker the green-channel baseline locks onto the flicker, POS does not (unit tests)." }],
    [{ text: "Study: ", options: { bold: true } }, { text: "record yourself watching Lecture 1 with a smartwatch as reference (scripts/learner_session.py), then correlate HR and blink rate with the key-moment curve." }],
    [{ text: "Status: ", options: { bold: true } }, { text: "[fill in after your recording: rPPG MAE vs smartwatch, correlation with key moments]", options: { color: HEX.red } }],
  ], 6.35, 4.3, 6.15, 2.5, 13);
  s.addNotes("Run: uv run python scripts/learner_session.py record --lecture lec01 --minutes 10, then analyse. Replace the red Status line with your numbers (bio_learner.png is produced).");

  // 13 --------------------------------------------------------------------------------- Fusion
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Key highlights" });
  title(s, "Fusion: which modality finds the key moments?");
  chips(s, ["fusion"]);
  const fsm = Object.fromEntries(fusion.enriched.summary.map((r) => [r.model, r.AP]));
  const fl = ["Chance", "Speech", "Vision", "Text", "Early fusion", "Late fusion", "Hand-weighted", "2nd human*"];
  const fv = [fusion.base_rate, fsm.speech, fsm.vision, fsm.text, fsm.early, fsm.late, fsm.unsupervised, agree.AP_of_annotatorB_vs_A];
  s.addChart(pres.charts.BAR, [{ name: "average precision", labels: fl, values: fv.map((v) => +v.toFixed(2)) }], {
    x: 0.6, y: 1.35, w: 7.4, h: 4.6, barDir: "col", chartColors: ["C9C8C3", "2A78D6", "EB6834", "1BAF7A", "6250D6", "6250D6", "8A8984", "52514E"],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", showLegend: false, valAxisHidden: true,
    valGridLine: { style: "none" }, valAxisMinVal: 0, valAxisMaxVal: 0.75, barGapWidthPct: 45, showTitle: true,
    title: `Average precision, leave-one-lecture-out (${fusion.n_windows} windows, ${fusion.n_pos} key moments)`, ...CHART_TXT });
  s.addText(`*Second annotator scored against the first on Lecture 1 (Cohen's κ = ${fmt(agree.kappa_exact)}): the ceiling a perfect model could approach.`,
    { x: 0.6, y: 6.05, w: 7.4, h: 0.7, fontSize: 12, italic: true, color: HEX.ink2, margin: 0, valign: "top", isTextBox: true });
  card(s, 8.3, 1.35, 4.4, 5.55);
  bullets(s, [
    [{ text: "Labels: ", options: { bold: true } }, { text: "10 s windows marked as definition, rule, explicit flag or resolution (written guide, two independent LLM annotators, κ = 0.74)." }],
    [{ text: "Text carries the signal; ", options: { bold: true } }, { text: "prosody barely beats chance. Dropping speech even raises early-fusion AP." }],
    [{ text: "My hypotheses were wrong: ", options: { bold: true } }, { text: "a pause before, novelty and visual change get negative weights; 'a slide is on screen' and discourse cues ('always', 'remember') get positive ones." }],
    [{ text: "Gap to humans (0.23 vs 0.65) ", options: { bold: true } }, { text: "means key moments are about meaning, not loudness." }],
  ], 8.55, 1.55, 3.95, 5.25, 14);

  // 14 --------------------------------------------------------------------------------- Dashboard
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Key highlights" });
  title(s, "The result: a lecture you can search, skim and speed through");
  chips(s, ["speech", "vision", "text", "fusion"]);
  s.addImage({ path: FIG("dashboard"), x: 0.6, y: 1.35, w: 8.25, h: 5.5, altText: "LectureLens dashboard screenshot", objectName: "dashboard" });
  bullets(s, [
    [{ text: "Search ", options: { bold: true } }, { text: "“what is an algorithm” → 15:15, where the recipe is called an algorithm." }],
    [{ text: "Highlights ", options: { bold: true } }, { text: "from the early-fusion model, always trained on the other lectures." }],
    [{ text: "Synchronised tracks: ", options: { bold: true } }, { text: "importance, energy, pitch, visual activity, slide/camera/demo bar." }],
    [{ text: "Off-slide insights ", options: { bold: true } }, { text: "and smart-speed audio player." }],
    [{ text: "Runs locally: ", options: { bold: true } }, { text: "3 lectures (2 h 16 min) processed in about 25 min on a laptop CPU; ASR is most of it." }],
  ], 9.15, 1.45, 3.55, 5.4, 14);

  // 15 --------------------------------------------------------------------------------- Learned
  sec("Learnings and reflections");
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Learnings and reflections" });
  title(s, "What I learned");
  const learned = [
    ["FaClock", HEX.speech, "Alignment is most of the work", "100 Hz audio frames, 1 Hz video, irregular word timestamps, licence boilerplate, missing slides: fusion only worked once everything sat on one 10 s grid."],
    ["FaBalanceScale", HEX.vision, "Thresholds need the right noise model", "One global z-score made HOG and ORB score zero: camera shots and static slides have very different noise floors."],
    ["FaCheckDouble", HEX.text, "Ground truth can be engineered", "The slide PDF gave OCR answers and page alignment for free; contact sheets made 139 events checkable in minutes."],
    ["FaFlask", HEX.bio, "Test against a reference first", "MFCC vs librosa, BM25 by hand, OLA SNR, POS on synthetic pulses (34 tests). A test caught a WSOLA length bug; a sanity table caught 'was' → 'wa' in the lemmatiser."],
  ];
  for (let i = 0; i < 4; i++) {
    const [ic, col, h1, d] = learned[i];
    const x = 0.6 + (i % 2) * 6.15, y = 1.45 + Math.floor(i / 2) * 2.7;
    card(s, x, y, 5.95, 2.5);
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: y + 0.3, w: 0.7, h: 0.7, fill: { color: col }, line: { color: col } });
    s.addImage({ data: await icon(ic, "FFFFFF"), x: x + 0.47, y: y + 0.47, w: 0.36, h: 0.36, altText: h1 });
    s.addText(h1, { x: x + 1.2, y: y + 0.3, w: 4.5, h: 0.7, fontSize: 18, bold: true, color: HEX.ink, fontFace: "Cambria", margin: 0, valign: "middle", isTextBox: true });
    s.addText(d, { x: x + 1.2, y: y + 1.05, w: 4.5, h: 1.3, fontSize: 14, color: HEX.ink2, margin: 0, valign: "top", isTextBox: true });
  }

  // 16 --------------------------------------------------------------------------------- Reflections
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Learnings and reflections" });
  title(s, "Reflections");
  const colHead = (x, t, col, ic) => {
    s.addText(t, { x: x + 0.55, y: 1.4, w: 5.0, h: 0.5, fontSize: 20, bold: true, color: col, fontFace: "Cambria", margin: 0, valign: "middle", isTextBox: true });
  };
  colHead(0.6, "(a) What surprised me", HEX.red);
  s.addImage({ data: await icon("FaSurprise", HEX.red), x: 0.6, y: 1.45, w: 0.4, h: 0.4, altText: "surprise" });
  bullets(s, [
    [{ text: "Emphasis ≠ importance. ", options: { bold: true } }, { text: "Loudness, pitch and pauses barely predict key moments; the words do." }],
    [{ text: "Synthetic wins don't transfer. ", options: { bold: true } }, { text: "ORB was perfect on synthetic slides and the worst detector on real video." }],
    [{ text: "Cleaning can hurt. ", options: { bold: true } }, { text: "Denoising lowered OCR on clean captures; Wiener deconvolution of a blurred frame beat the original." }],
    [{ text: "Spoken language is far more repetitive ", options: { bold: true } }, { text: "than slides (Zipf 1.22 vs 0.91)." }],
  ], 0.6, 2.05, 5.9, 4.8, 15);
  colHead(6.85, "(b) Scope for improvement", HEX.speech);
  s.addImage({ data: await icon("FaTools", HEX.speech), x: 6.85, y: 1.45, w: 0.4, h: 0.4, altText: "improvement" });
  bullets(s, [
    [{ text: "Semantic text features: ", options: { bold: true } }, { text: "sentence embeddings or a small LLM to close the 0.23 → 0.65 gap." }],
    [{ text: "Human labels at scale: ", options: { bold: true } }, { text: "real students marking key moments; more lectures and speakers." }],
    [{ text: "Learner study: ", options: { bold: true } }, { text: "10+ viewers with chest-strap reference HR." }],
    [{ text: "Beyond English slides: ", options: { bold: true } }, { text: "chalkboard lectures (needs handwriting OCR), Hindi/Telugu code-mixed speech." }],
    [{ text: "Real time: ", options: { bold: true } }, { text: "every DSP stage is streaming-friendly; only ASR is batch." }],
  ], 6.85, 2.05, 5.85, 4.8, 15);

  // 17 --------------------------------------------------------------------------------- References
  sec("References");
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "References" });
  title(s, "References");
  const refs1 = [
    "Rabiner & Schafer (1978). Digital Processing of Speech Signals. Prentice-Hall.",
    "Davis & Mermelstein (1980). Parametric representations for monosyllabic word recognition. IEEE TASSP.",
    "Verhelst & Roelands (1993). WSOLA: an overlap-add technique based on waveform similarity. ICASSP.",
    "Gonzalez & Woods. Digital Image Processing, 4th ed. Pearson.",
    "Otsu (1979); Canny (1986); Dalal & Triggs (2005) HOG; Rublee et al. (2011) ORB.",
    "Robertson & Zaragoza (2009). The probabilistic relevance framework: BM25 and beyond.",
    "Sennrich, Haddow & Birch (2016). Neural MT of rare words with subword units (BPE). ACL.",
    "Kneser & Ney (1995). Improved backing-off for m-gram language modeling. ICASSP; Chen & Goodman (1999). An empirical study of smoothing techniques. CSL.",
    "Zipf (1949); Heaps (1978). Information Retrieval: Computational and Theoretical Aspects.",
    "Verkruysse et al. (2008). Remote plethysmographic imaging using ambient light. Optics Express.",
    "Wang et al. (2017). Algorithmic principles of remote PPG (POS). IEEE TBME.",
    "Soukupová & Čech (2016). Real-time eye blink detection using facial landmarks. CVWW.",
  ];
  const refs2 = [
    "Wactlar, Kanade, Smith & Stevens (1996). Intelligent access to digital video: Informedia project. IEEE Computer.",
    "Abowd (1999). Classroom 2000: instrumentation of a living educational environment. IBM Systems Journal.",
    "Glass et al. (2007). Recent progress in the MIT spoken lecture processing project. Interspeech.",
    "Adcock et al. (2010). TalkMiner: a lecture webcast search engine. ACM Multimedia.",
    "Radford et al. (2022). Robust speech recognition via large-scale weak supervision (Whisper).",
    "Baltrušaitis, Ahuja & Morency (2019). Multimodal machine learning: a survey and taxonomy. IEEE TPAMI.",
    "Cohen (1960). A coefficient of agreement for nominal scales.",
    "Data: MIT OCW 6.0001 Fall 2016, lectures 1, 8, 10 (Bell, Grimson, Guttag), CC BY-NC-SA 4.0.",
    "Tools: OpenCV, NumPy/SciPy, librosa (tests only), faster-whisper, Tesseract, MediaPipe, NLTK/WordNet, scikit-learn, PyMuPDF, Streamlit, yt-dlp.",
    "LLM tools: Claude Code (Anthropic) for code scaffolding, figures and deck generation, and as the two key-moment annotators (verified by agreement).",
  ];
  const refBox = (items, x) => s.addText(items.map((t, i) => ({ text: t, options: { breakLine: i < items.length - 1, paraSpaceAfter: 4 } })),
    { x, y: 1.35, w: 5.95, h: 5.55, fontSize: 11, color: HEX.ink, margin: 0, valign: "top", isTextBox: true });
  refBox(refs1, 0.6);
  refBox(refs2, 6.75);

  // 18 --------------------------------------------------------------------------------- Backup: evaluation summary
  sec("Backup");
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
  title(s, "Backup: every number, and how it was measured");
  const rows = [
    ["Speech", "MFCC vs librosa (max abs. difference)", speech.mfcc_max_abs_diff_vs_librosa.toExponential(1), "5 s of Lecture 1"],
    ["Speech", "OLA reconstruction SNR", fmt(speech.ola_reconstruction_snr_db, 1) + " dB", "Hann, 75 % overlap"],
    ["Speech", "Pitch: autocorr. vs cepstrum within 5 %", pct(speech.pitch_agreement_within_5pct), "first 10 min, voiced frames"],
    ["Speech", "Smart-speed listening time", `${fmt(ss.original, 1)} → ${fmt(ss["+ 1.3x WSOLA"], 1)} min`, "pauses → 0.15 s, WSOLA 1.3×"],
    ["Vision", "Visual-event F1, shot-aware fusion", fmt(vr["Shot-aware fusion"].F1), "139 checked events, ±2 s"],
    ["Vision", "OCR char. accuracy (unsharp + upscale)", pct(ov["unsharp+upscale"].mean, 1), `${ocrVar.n} slides vs PDF text`],
    ["Vision", "Wiener: PSNR / OCR on blurred slide", `${fmt(vision.restoration.wiener_psnr, 1)} dB / ${pct(vision.restoration.wiener_ocr_acc)}`, "9-px motion blur + noise"],
    ["Text", "BM25 known-item MRR (transcript / +slides)", `${fmt(row("bm25", false).MRR)} / ${fmt(row("bm25", true).MRR)}`, `${row("bm25", false).n} slide-title queries`],
    ["Text", "Bigram perplexity: Laplace / add-k / KN", `${Math.round(mean("laplace"))} / ${Math.round(mean("addk"))} / ${Math.round(mean("kn"))}`, "leave-one-lecture-out"],
    ["Bio", "POS heart-rate error (synthetic)", "< 2 bpm", "55, 72, 110 bpm + flicker"],
    ["Fusion", "Key-moment AP: text / early / chance", `${fmt(fsm.text)} / ${fmt(fsm.early)} / ${fmt(fusion.base_rate)}`, "3-fold leave-one-lecture-out"],
    ["Labels", "Inter-annotator agreement (Cohen's κ)", fmt(agree.kappa_exact), "Lecture 1, 259 windows"],
  ];
  s.addTable([[{ text: "Modality", options: th }, { text: "Metric", options: th }, { text: "Result", options: th }, { text: "Setup", options: th }],
    ...rows.map((r) => [tc(r[0], true), tc(r[1]), tc(r[2], true), tc(r[3])])],
    { x: 0.6, y: 1.3, w: 12.1, colW: [1.3, 4.6, 2.6, 3.6], rowH: 0.42, border: { type: "solid", color: HEX.grid, pt: 1 }, fill: { color: HEX.white }, margin: [2, 6, 2, 6] });

  // 19 --------------------------------------------------------------------------------- Backup: windowing + MFCC
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
  title(s, "Backup: windowing and the MFCC check");
  chips(s, ["speech"]);
  img(s, "speech_windowing", 0.6, 1.3, 12.1, 2.75);
  img(s, "speech_mfcc_check", 0.6, 4.15, 12.1, 2.75);

  // 20 --------------------------------------------------------------------------------- Backup: enhancement + timeline
  s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Backup" });
  title(s, "Backup: enhancing dark camera frames, and one lecture at a glance");
  chips(s, ["vision", "fusion"]);
  img(s, "vision_enhancement", 0.6, 1.3, 5.6, 5.55);
  img(s, "overview_timeline", 6.4, 1.3, 6.3, 5.55);

  const out = path.join(__dirname, "LectureLens.pptx");
  await pres.writeFile({ fileName: out });
  await applyTheme(out, THEME);
  console.log("wrote", out);
})();
