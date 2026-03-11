# binvad

`stdin`으로 들어오는 `s16le` raw PCM(16kHz, mono)을 Silero VAD(ONNX Runtime)로 분석해 speech chunk를 출력합니다.
입력은 ffmpeg 파이프를 가정합니다.

## Silero VAD ONNX model 다운로드
```bash
$ wget https://huggingface.co/onnx-community/silero-vad/resolve/main/onnx/model.onnx
$ mv model.onnx silero_vad.onnx
```

## ONNX Runtime prebuilt 설치 + 빌드

### 1) prebuilt ONNX Runtime 다운로드

```bash
mkdir -p third_party
cd third_party
wget https://github.com/microsoft/onnxruntime/releases/download/v1.24.3/onnxruntime-linux-x64-1.24.3.tgz
tar -xzf onnxruntime-linux-x64-1.24.3.tgz
mv onnxruntime-linux-x64-1.24.3 onnxruntime
cd ..
```

필수 파일 확인:

- `third_party/onnxruntime/include/onnxruntime_cxx_api.h`
- `third_party/onnxruntime/lib/libonnxruntime.so` 또는 `libonnxruntime.so.1`

### 2) binvad 빌드

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Alternate: external ONNX Runtime install

Vendored 경로 대신 외부 설치를 쓰려면:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_DIR=/path/to/onnxruntime
cmake --build build -j
```

## 실행

`binvad`는 `stdin`으로 아래 포맷을 기대합니다.

- `s16le`
- `mono`
- `16kHz`

예시:

```bash
ffmpeg -i input.mp3 -f s16le -acodec pcm_s16le -ac 1 -ar 16000 - \
  | ./build/binvad \
      --model silero_vad.onnx \
      --format jsonl \
      --speech-prob-thres 0.5 \
      --min-interval-sec 0.3 \
      --grace-sec 0.05 \
      --min-chunk-sec 0.2
```

### Options

- `--model`: Silero VAD ONNX 모델 경로 (기본값: `silero_vad.onnx`)
- `--format`: `csv` 또는 `jsonl` 또는 `raw`
- `--speech-prob-thres`: speech 판정 확률 기준값
- `--min-interval-sec`: chunk 분리를 위한 최소 non-speech 구간 길이
- `--grace-sec`: chunk 앞/뒤로 추가할 시간(초)
- `--min-chunk-sec`: 최종 출력할 최소 chunk 길이

### Output

- `csv`: 헤더 포함 `start_sec,end_sec`
- `jsonl`: 줄 단위 JSON (`{"start_sec":...,"end_sec":...}`)
- `raw`: 줄 단위 공백 구분 (`<start_sec> <end_sec>`)

## E2E 청킹 스크립트 (stream copy)

`raw` 출력을 사용해 즉시 ffmpeg `-c copy` 청크를 생성합니다.

```bash
./scripts/e2e_vad_chunk_copy.sh input.mp4 silero_vad.onnx chunks ./build/binvad
```

출력:

- 각 청크를 `chunks/chunk_0000.mka` 형태로 생성
- 표준출력에 `output_path,start_sec,end_sec` 로그 출력
