#!/bin/bash

set -euo pipefail

ffmpeg -i $1 \
  -f s16le \
  -acodec pcm_s16le \
  -ac 1 \
  -ar 16000 - \
  | ./build/binvad \
      --model silero_vad_v6.2.onnx \
      --format raw \
      --speech-prob-thres 0.5 \
      --min-interval-sec 5 \
      --grace-sec 1 \
      --min-chunk-sec 1