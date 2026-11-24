#!/usr/bin/env bash
# Download the three MIT OCW 6.0001 (Fall 2016) lectures and their slide PDFs.
# License: CC BY-NC-SA 4.0 (non-commercial coursework use). Run from the project root.
set -euo pipefail
mkdir -p data/raw data/slides

# name:youtube-id pairs (plain list: macOS ships bash 3.2, which has no associative arrays)
for pair in "lec01_what_is_computation:nykOeWgQcHM" \
            "lec08_object_oriented_programming:-DP1i2ZU9gk" \
            "lec10_program_efficiency:o9nW0uBqvEo"; do
  name="${pair%%:*}"; vid="${pair#*:}"
  out="data/raw/${name}.mp4"
  [ -f "$out" ] && { echo "have $out"; continue; }
  uv run yt-dlp -q --no-warnings -f "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720]" \
    --merge-output-format mp4 -o "$out" -- "$vid"
  echo "downloaded $out"
done

BASE="https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016"
for n in 1 8 10; do
  out="data/slides/lec${n}.pdf"
  [ -f "$out" ] && { echo "have $out"; continue; }
  # the PDF link has a hash prefix, so read it from the lecture's resource page
  link=$(curl -sL -A "Mozilla/5.0" "$BASE/resources/mit6_0001f16_lec${n}/" | grep -oE "/courses/[^\"]*_Lec${n}\.pdf" | head -1)
  curl -sL -A "Mozilla/5.0" "https://ocw.mit.edu${link}" -o "$out"
  echo "downloaded $out"
done
