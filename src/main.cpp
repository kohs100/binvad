#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
  std::string model_path = "silero_vad_v6.2.onnx";
  std::string format = "csv";  // csv|jsonl|raw
  float speech_prob_thres = 0.5f;
  double min_interval_sec = 0.3;
  double grace_sec = 0.05;
  double min_chunk_sec = 0.2;
};

struct Segment {
  double start_sec;
  double end_sec;
};

std::string FormatRawTs(double v) {
  std::ostringstream oss;
  oss.setf(std::ios::fixed);
  oss.precision(6);
  oss << v;
  std::string s = oss.str();
  if (!s.empty() && s[0] == '.') {
    return "0" + s;
  }
  if (s.size() > 1 && s[0] == '-' && s[1] == '.') {
    return "-0" + s.substr(1);
  }
  return s;
}

[[noreturn]] void Die(const std::string& msg) {
  throw std::runtime_error(msg);
}

bool StartsWith(const std::string& s, const std::string& p) {
  return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string token(argv[i]);
    auto take_value = [&](const std::string& key) -> std::string {
      if (i + 1 >= argc) {
        Die("missing value for " + key);
      }
      return std::string(argv[++i]);
    };

    if (token == "--model") {
      args.model_path = take_value(token);
    } else if (token == "--format") {
      args.format = take_value(token);
      if (args.format != "csv" && args.format != "jsonl" && args.format != "raw") {
        Die("--format must be csv, jsonl, or raw");
      }
    } else if (token == "--speech-prob-thres") {
      args.speech_prob_thres = std::stof(take_value(token));
    } else if (token == "--min-interval-sec") {
      args.min_interval_sec = std::stod(take_value(token));
    } else if (token == "--grace-sec") {
      args.grace_sec = std::stod(take_value(token));
    } else if (token == "--min-chunk-sec") {
      args.min_chunk_sec = std::stod(take_value(token));
    } else if (token == "--help" || token == "-h") {
      std::cout
          << "Usage: binvad [options]\n"
          << "  --model PATH                 Silero VAD ONNX path (default: silero_vad.onnx)\n"
          << "  --format csv|jsonl|raw       Output format (default: csv)\n"
          << "  --speech-prob-thres FLOAT    VAD probability threshold (default: 0.5)\n"
          << "  --min-interval-sec FLOAT     Minimum non-speech gap to split chunks\n"
          << "  --grace-sec FLOAT            Padding added before/after each chunk\n"
          << "  --min-chunk-sec FLOAT        Minimum chunk length to keep\n";
      std::exit(0);
    } else if (StartsWith(token, "--")) {
      Die("unknown option: " + token);
    } else {
      Die("unexpected positional argument: " + token);
    }
  }

  if (!(args.speech_prob_thres >= 0.0f && args.speech_prob_thres <= 1.0f)) {
    Die("--speech-prob-thres must be in [0, 1]");
  }
  if (args.min_interval_sec < 0.0) {
    Die("--min-interval-sec must be >= 0");
  }
  if (args.grace_sec < 0.0) {
    Die("--grace-sec must be >= 0");
  }
  if (args.grace_sec * 2.0 >= args.min_interval_sec && args.min_interval_sec > 0.0) {
    Die("--grace-sec must be smaller than half of --min-interval-sec");
  }
  if (args.min_chunk_sec < 0.0) {
    Die("--min-chunk-sec must be >= 0");
  }
  return args;
}

class SileroVadRunner {
 public:
  explicit SileroVadRunner(const std::string& model_path)
      : env_(ORT_LOGGING_LEVEL_WARNING, "binvad"),
        session_(nullptr) {
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(1);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_ = Ort::Session(env_, model_path.c_str(), so);

    InitMetadata();
    InitState();
  }

  int64_t sample_rate() const { return sample_rate_; }
  int64_t frame_samples() const { return frame_samples_; }

  float Infer(const std::vector<float>& frame) {
    if (static_cast<int64_t>(frame.size()) != frame_samples_) {
      Die("internal error: frame size mismatch");
    }

    // 1. Context와 현재 frame 병합
    std::vector<float> model_input(context_size_ + frame_samples_);
    std::memcpy(model_input.data(), context_.data(), context_size_ * sizeof(float));
    std::memcpy(model_input.data() + context_size_, frame.data(), frame_samples_ * sizeof(float));

    // 2. 다음 Inference를 위해 현재 frame의 마지막 context_size_ 만큼을 context_ 버퍼에 저장
    std::memcpy(context_.data(), frame.data() + frame_samples_ - context_size_, context_size_ * sizeof(float));

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<const char*> input_names_c;
    std::vector<Ort::Value> input_values;
    input_names_c.reserve(input_names_.size());
    input_values.reserve(input_names_.size());

    for (const auto& name : input_names_) {
      input_names_c.push_back(name.c_str());
      if (name == audio_input_name_) {
        // 병합된 [1, 576] 사이즈의 input tensor 주입
        std::vector<int64_t> shape{1, context_size_ + frame_samples_};
        input_values.push_back(Ort::Value::CreateTensor<float>(
            mem_info, model_input.data(), model_input.size(), shape.data(), shape.size()));
      } else if (!sr_input_name_.empty() && name == sr_input_name_) {
        std::vector<int64_t> shape{1};
        input_values.push_back(Ort::Value::CreateTensor<int64_t>(
            mem_info, &sample_rate_, 1, shape.data(), shape.size()));
      } else if (!state_input_name_.empty() && name == state_input_name_) {
        input_values.push_back(Ort::Value::CreateTensor<float>(
            mem_info, state_.data(), state_.size(), state_shape_.data(), state_shape_.size()));
      } else {
        Die("unsupported input tensor: " + name);
      }
    }

    std::vector<const char*> output_names_c;
    output_names_c.reserve(output_names_.size());
    for (const auto& n : output_names_) {
      output_names_c.push_back(n.c_str());
    }

    auto outputs = session_.Run(Ort::RunOptions{nullptr},
                                input_names_c.data(), input_values.data(), input_values.size(),
                                output_names_c.data(), output_names_c.size());

    if (outputs.empty()) {
      Die("model returned no outputs");
    }

    float prob = ExtractProbability(outputs);
    UpdateState(outputs);
    return prob;
  }

  // 스트림이 변경되거나 초기화가 필요할 때 호출
  void Reset() {
    InitState();
  }

 private:
  void Die(const std::string& msg) {
    std::cerr << "Fatal Error: " << msg << std::endl;
    std::exit(EXIT_FAILURE);
  }

  void InitMetadata() {
    Ort::AllocatorWithDefaultOptions allocator;

    const size_t input_count = session_.GetInputCount();
    const size_t output_count = session_.GetOutputCount();

    if (input_count == 0) {
      Die("model has no inputs");
    }

    input_names_.reserve(input_count);
    for (size_t i = 0; i < input_count; ++i) {
      auto n = session_.GetInputNameAllocated(i, allocator);
      input_names_.push_back(n.get());
    }

    output_names_.reserve(output_count);
    for (size_t i = 0; i < output_count; ++i) {
      auto n = session_.GetOutputNameAllocated(i, allocator);
      output_names_.push_back(n.get());
    }

    for (size_t i = 0; i < input_count; ++i) {
      Ort::TypeInfo ti = session_.GetInputTypeInfo(i);
      auto tsi = ti.GetTensorTypeAndShapeInfo();
      auto shape = tsi.GetShape();
      const std::string& name = input_names_[i];

      const std::string lname = Lower(name);
      if (lname.find("state") != std::string::npos) {
        state_input_name_ = name;
        state_shape_ = shape;
      } else if (lname == "sr" || lname.find("sample_rate") != std::string::npos) {
        sr_input_name_ = name;
      } else {
        audio_input_name_ = name;
      }
    }

    if (audio_input_name_.empty()) {
      Die("unable to locate audio input tensor");
    }

    // ONNX 모델의 동적 shape(-1)에 의존하지 않고 명시적으로 512 frame 유지
    context_size_ = (sample_rate_ == 16000) ? 64 : 32;
    frame_samples_ = (sample_rate_ == 16000) ? 512 : 256;

    if (state_shape_.empty() && !state_input_name_.empty()) {
      state_shape_ = {2, 1, 128};
    }

    if (output_names_.empty()) {
      Die("model has no outputs");
    }
  }

  static std::string Lower(const std::string& s) {
    std::string out(s);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    return out;
  }

  void InitState() {
    // Context 버퍼 초기화 (초기값 0.0f 할당)
    context_size_ = (sample_rate_ == 16000) ? 64 : 32;
    context_.assign(context_size_, 0.0f);

    if (state_input_name_.empty()) {
      return;
    }

    if (state_shape_.empty()) {
      state_shape_ = {2, 1, 128};
    }

    for (size_t i = 0; i < state_shape_.size(); ++i) {
      if (state_shape_[i] > 0) {
        continue;
      }
      if (state_shape_.size() == 3) {
        if (i == 0) {
          state_shape_[i] = 2;
        } else if (i == 1) {
          state_shape_[i] = 1;
        } else {
          state_shape_[i] = 128;
        }
      } else {
        state_shape_[i] = 1;
      }
    }

    size_t total = 1;
    for (int64_t d : state_shape_) {
      total *= static_cast<size_t>(d);
    }
    state_.assign(total, 0.0f);
  }

  float ExtractProbability(const std::vector<Ort::Value>& outputs) {
    float best = std::numeric_limits<float>::quiet_NaN();

    for (const auto& out : outputs) {
      if (!out.IsTensor()) {
        continue;
      }
      auto info = out.GetTensorTypeAndShapeInfo();
      if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        continue;
      }

      const size_t n = info.GetElementCount();
      if (n == 0) {
        continue;
      }

      const float* p = out.GetTensorData<float>();
      if (n == 1) {
        return p[0];
      }
      if (std::isnan(best)) {
        best = p[0];
      }
    }

    if (std::isnan(best)) {
      Die("failed to extract speech probability from model outputs");
    }
    return best;
  }

  void UpdateState(const std::vector<Ort::Value>& outputs) {
    if (state_.empty()) {
      return;
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
      const auto& out = outputs[i];
      if (!out.IsTensor()) {
        continue;
      }

      auto info = out.GetTensorTypeAndShapeInfo();
      if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        continue;
      }

      const size_t n = info.GetElementCount();
      if (n != state_.size()) {
        continue;
      }

      const float* p = out.GetTensorData<float>();
      std::memcpy(state_.data(), p, state_.size() * sizeof(float));
      return;
    }
  }

  Ort::Env env_;
  Ort::Session session_;

  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;

  std::string audio_input_name_;
  std::string sr_input_name_;
  std::string state_input_name_;

  int64_t sample_rate_ = 16000;
  int64_t frame_samples_ = 512;
  int64_t context_size_ = 64;  // v5/v6를 위한 context buffer 사이즈 추가

  std::vector<float> context_; // 이전 프레임의 마지막 샘플들을 저장하는 버퍼

  std::vector<int64_t> state_shape_;
  std::vector<float> state_;
};

void FinalizeSegments(std::vector<Segment>* segments, double total_sec, const Args& args) {
  auto& raw = *segments;
  for (auto& seg : raw) {
    seg.start_sec = std::max(0.0, seg.start_sec - args.grace_sec);
    seg.end_sec = std::min(total_sec, seg.end_sec + args.grace_sec);
  }

  std::sort(raw.begin(), raw.end(), [](const Segment& a, const Segment& b) {
    if (a.start_sec != b.start_sec) {
      return a.start_sec < b.start_sec;
    }
    return a.end_sec < b.end_sec;
  });

  std::vector<Segment> merged;
  merged.reserve(raw.size());
  for (const auto& seg : raw) {
    if (seg.end_sec <= seg.start_sec) {
      continue;
    }
    if (merged.empty() || seg.start_sec > merged.back().end_sec) {
      merged.push_back(seg);
    } else {
      merged.back().end_sec = std::max(merged.back().end_sec, seg.end_sec);
    }
  }

  raw.clear();
  for (const auto& seg : merged) {
    if ((seg.end_sec - seg.start_sec) >= args.min_chunk_sec) {
      raw.push_back(seg);
    }
  }
}

std::vector<Segment> DetectSegmentsFromStdin(SileroVadRunner* vad, const Args& args) {
  const int64_t sr = vad->sample_rate();
  const int64_t frame_samples = vad->frame_samples();
  const double frame_sec = static_cast<double>(frame_samples) / static_cast<double>(sr);

  std::vector<Segment> segments;
  bool in_speech = false;
  double active_start = 0.0;
  bool have_pending_silence = false;
  double pending_silence_start = 0.0;

  std::vector<float> frame(frame_samples, 0.0f);
  std::deque<float> sample_buffer;

  auto process_one_frame = [&](int64_t frame_index) {
    const double frame_start =
        static_cast<double>(frame_index * frame_samples) / static_cast<double>(sr);
    const double frame_end = frame_start + frame_sec;
    const float prob = vad->Infer(frame);
    const bool speech = prob >= args.speech_prob_thres;

    if (!in_speech) {
      if (speech) {
        in_speech = true;
        active_start = frame_start;
        have_pending_silence = false;
      }
      return;
    }

    if (speech) {
      have_pending_silence = false;
      return;
    }

    if (!have_pending_silence) {
      have_pending_silence = true;
      pending_silence_start = frame_start;
      return;
    }

    const double silence_dur = frame_end - pending_silence_start;
    if (silence_dur >= args.min_interval_sec) {
      segments.push_back({active_start, pending_silence_start});
      in_speech = false;
      have_pending_silence = false;
    }
  };

  constexpr size_t kChunkBytes = 1 << 15;
  std::vector<char> chunk(kChunkBytes);
  int64_t input_samples = 0;
  int64_t frame_index = 0;

  while (std::cin.good()) {
    std::cin.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
    const std::streamsize got = std::cin.gcount();
    if (got <= 0) {
      break;
    }
    const size_t bytes = static_cast<size_t>(got);
    if (bytes % 2 != 0) {
      Die("stdin byte count is not aligned for s16le");
    }

    const size_t n = bytes / sizeof(int16_t);
    for (size_t i = 0; i < n; ++i) {
      const uint8_t lo = static_cast<uint8_t>(chunk[2 * i]);
      const uint8_t hi = static_cast<uint8_t>(chunk[2 * i + 1]);
      const int16_t s = static_cast<int16_t>((static_cast<uint16_t>(hi) << 8) | lo);
      sample_buffer.push_back(static_cast<float>(s) / 32768.0f);
    }
    input_samples += static_cast<int64_t>(n);

    while (sample_buffer.size() >= static_cast<size_t>(frame_samples)) {
      for (int64_t i = 0; i < frame_samples; ++i) {
        frame[static_cast<size_t>(i)] = sample_buffer.front();
        sample_buffer.pop_front();
      }
      process_one_frame(frame_index++);
    }
  }

  if (std::cin.bad()) {
    Die("failed to read stdin");
  }

  if (!sample_buffer.empty()) {
    std::fill(frame.begin(), frame.end(), 0.0f);
    size_t i = 0;
    while (!sample_buffer.empty() && i < frame.size()) {
      frame[i++] = sample_buffer.front();
      sample_buffer.pop_front();
    }
    process_one_frame(frame_index++);
  }

  const double total_sec = static_cast<double>(input_samples) / static_cast<double>(sr);
  if (in_speech) {
    const double end_sec = have_pending_silence ? pending_silence_start : total_sec;
    segments.push_back({active_start, end_sec});
  }

  FinalizeSegments(&segments, total_sec, args);
  return segments;
}

void PrintSegments(const std::vector<Segment>& segs, const std::string& format) {
  std::cout.setf(std::ios::fixed);
  std::cout.precision(6);

  if (segs.empty()) {
    std::cerr << "Empty segment!!\n";
    return;
  }

  if (format == "csv") {
    std::cout << "start_sec,end_sec\n";
    for (const auto& s : segs) {
      std::cout << s.start_sec << ',' << s.end_sec << '\n';
    }
    return;
  }

  if (format == "raw") {
    for (const auto& s : segs) {
      std::cout << FormatRawTs(s.start_sec) << ' ' << FormatRawTs(s.end_sec) << '\n';
    }
    return;
  }

  for (const auto& s : segs) {
    std::cout << "{\"start_sec\":" << s.start_sec << ",\"end_sec\":" << s.end_sec << "}\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    const Args args = ParseArgs(argc, argv);

    SileroVadRunner vad(args.model_path);
    if (vad.sample_rate() != 16000) {
      Die("this program expects a 16kHz model/input");
    }

    const auto segments = DetectSegmentsFromStdin(&vad, args);
    PrintSegments(segments, args.format);
    return 0;
  } catch (const Ort::Exception& e) {
    std::cerr << "ONNX Runtime error: " << e.what() << '\n';
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
