#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat <<USAGE
Usage: $0 <input_media> <silero_vad.onnx> <output_dir> [binvad_path]

Example:
  $0 input.mp4 silero_vad.onnx chunks ./build/binvad
USAGE
  exit 1
fi

INPUT="$1"
MODEL="$2"
OUT_DIR="$3"
BINVAD="${4:-./build/binvad}"
EXT="${INPUT##*.}"
if [[ "$EXT" == "$INPUT" ]]; then
  EXT="mka"
fi

if [[ ! -f "$INPUT" ]]; then
  echo "error: input file not found: $INPUT" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "error: model file not found: $MODEL" >&2
  exit 1
fi
if [[ ! -x "$BINVAD" ]]; then
  echo "error: binvad not executable: $BINVAD" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

TMP_RAW="$(mktemp)"
trap 'rm -f "$TMP_RAW"' EXIT

# 1) Decode to 16kHz mono s16le -> VAD (raw output: "start_sec end_sec")
ffmpeg -hide_banner -loglevel error -i "$INPUT" -f s16le -acodec pcm_s16le -ac 1 -ar 16000 - \
  | "$BINVAD" \
      --model "$MODEL" \
      --format raw \
      --speech-prob-thres 0.5 \
      --min-interval-sec 5 \
      --grace-sec 0.5 \
      --min-chunk-sec 1 \
  > "$TMP_RAW"

if [[ ! -s "$TMP_RAW" ]]; then
  echo "No speech chunks detected."
  exit 0
fi

# 2) Stream-copy chunk extraction from original media.
idx=0
while read -r start_sec end_sec; do
  printf -v chunk_name "chunk_%04d.%s" "$idx" "$EXT"
  out_path="$OUT_DIR/$chunk_name"
  echo "$out_path,$start_sec,$end_sec"

  ffmpeg -hide_banner -loglevel error \
    -ss "$start_sec" -to "$end_sec" -i "$INPUT" \
    -vn -c copy "$out_path" -nostdin

  idx=$((idx + 1))
done < "$TMP_RAW"

echo "Done. chunks=$idx output_dir=$OUT_DIR"
