#include "Engine.h"
#include "StudioVoice.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <mutex>
#include <numeric>

namespace ob {
#ifdef OB_LINUX_MIGRAPHX
static_assert(ORT_API_VERSION >= 24, "MIGraphX device verification requires ONNX Runtime API 1.24 or later");
#endif
Runtime::Runtime(const std::filesystem::path& directory) : directory(std::filesystem::absolute(directory)) {
    // ORT's C++ API table is process-global in this module; retain the DLL until
    // process exit so closing/reopening plugin instances cannot invalidate it.
    static std::once_flag once;
    static void* ort{};
    static void* dml{};
    std::call_once(once, [&] {
        const auto absoluteDirectory = std::filesystem::absolute(directory);
#ifdef _WIN32
        const auto flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
        dml = LoadLibraryExW((absoluteDirectory / "DirectML.dll").c_str(), nullptr, flags);
        ort = LoadLibraryExW((absoluteDirectory / "onnxruntime.dll").c_str(), nullptr, flags);
        if (!ort) throw std::runtime_error("Cannot load bundled onnxruntime.dll (Windows error " + std::to_string(GetLastError()) + ")");
        using GetApi = const OrtApiBase* (ORT_API_CALL*)();
        auto getApi = reinterpret_cast<GetApi>(GetProcAddress(static_cast<HMODULE>(ort), "OrtGetApiBase"));
#else
#ifdef __APPLE__
        const auto filename = absoluteDirectory / "libonnxruntime.dylib";
#else
        const auto filename = absoluteDirectory / "libonnxruntime.so";
#endif
        ort = dlopen(filename.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!ort) throw std::runtime_error(std::string("Cannot load bundled ONNX Runtime: ") + dlerror());
        using GetApi = const OrtApiBase* (ORT_API_CALL*)();
        auto getApi = reinterpret_cast<GetApi>(dlsym(ort, "OrtGetApiBase"));
#endif
        if (!getApi) throw std::runtime_error("Invalid ONNX Runtime library");
        auto api = getApi()->GetApi(ORT_API_VERSION);
        if (!api) throw std::runtime_error("Incompatible ONNX Runtime version");
        Ort::InitApi(api);
    });
    library = ort; directml = dml;
    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, "OpenBroadcastVST");
}
Runtime::~Runtime() = default;
void Runtime::configureDevice(Ort::SessionOptions& options, int device) const {
    if(device < 0)return;
#ifdef _WIN32
    if (!directml) throw std::runtime_error("DirectML.dll is missing");
    options.DisableMemPattern();
    using Append = OrtStatus* (ORT_API_CALL*)(OrtSessionOptions*, int);
    auto append = reinterpret_cast<Append>(GetProcAddress(static_cast<HMODULE>(library), "OrtSessionOptionsAppendExecutionProvider_DML"));
    if (!append) throw std::runtime_error("Runtime has no DirectML provider");
    Ort::ThrowOnError(append(options, device));
#elif defined(__APPLE__)
    // Match OpenBroadcast's system-managed CPU/GPU policy, not a Metal ordinal.
    if(device != 0)throw std::runtime_error("CoreML uses system-managed GPU selection");
    options.AppendExecutionProvider("CoreML", {{"MLComputeUnits", "CPUAndGPU"}});
#elif defined(OB_LINUX_MIGRAPHX)
    // Preserve model precision; let MIGraphX partition supported GPU subgraphs
    // while ORT handles any remaining CPU operators.
    OrtMIGraphXProviderOptions amd{};
    amd.device_id=device;
    amd.migraphx_mem_limit=std::numeric_limits<size_t>::max();
    options.AddConfigEntry("session.record_ep_graph_assignment_info", "1");
    Ort::ThrowOnError(Ort::GetApi().SessionOptionsAppendExecutionProvider_MIGraphX(options, &amd));
#else
    OrtCUDAProviderOptionsV2* raw{};
    const auto& api=Ort::GetApi();
    Ort::ThrowOnError(api.CreateCUDAProviderOptions(&raw));
    std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(api.ReleaseCUDAProviderOptions)> cuda(raw, api.ReleaseCUDAProviderOptions);
    const auto ordinal=std::to_string(device);
    const char* keys[]{"device_id", "cudnn_conv_algo_search"};
    const char* values[]{ordinal.c_str(), "HEURISTIC"};
    Ort::ThrowOnError(api.UpdateCUDAProviderOptions(cuda.get(), keys, values, 2));
    Ort::ThrowOnError(api.SessionOptionsAppendExecutionProvider_CUDA_V2(options, cuda.get()));
#endif
}
void Runtime::verifyDevice(const Ort::Session& session, int device) const {
#ifdef OB_LINUX_MIGRAPHX
    if(device < 0)return;
    const auto& api=Ort::GetApi();
    const OrtEpAssignedSubgraph* const* graphs{};
    size_t count=0;
    Ort::ThrowOnError(api.Session_GetEpGraphAssignmentInfo(session, &graphs, &count));
    for(size_t i=0;i<count;++i) {
        const char* name{};
        Ort::ThrowOnError(api.EpAssignedSubgraph_GetEpName(graphs[i], &name));
        if(name && std::string(name)=="MIGraphXExecutionProvider")return;
    }
    throw std::runtime_error("MIGraphX did not assign any model operations to the GPU");
#else
    (void)session; (void)device;
#endif
}
Coefficients::Coefficients(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    char magic[8]{}; file.read(magic, 8);
    if (std::string(magic, 8) != "OBDSP001") throw std::runtime_error("Missing or invalid dsp.bin");
    auto read = [&](float* data, uint32_t count) {
        uint32_t n{}; file.read(reinterpret_cast<char*>(&n), 4);
        if (n != count) throw std::runtime_error("Invalid DSP coefficient dimensions");
        file.read(reinterpret_cast<char*>(data), count * sizeof(float));
        if (!file) throw std::runtime_error("Truncated DSP coefficients");
        for (uint32_t i = 0; i < count; ++i)
            if (!std::isfinite(data[i])) throw std::runtime_error("Nonfinite DSP coefficients");
    };
    read(analysis.data(), block); read(synthesis.data(), block);
    std::vector<float> matrix(bins * bands); read(matrix.data(), bins * bands);
    for (int bin = 0; bin < bins; ++bin)
        for (int band = 0; band < bands; ++band)
            if (matrix[bin * bands + band] != 0) mel.push_back({bin, band, matrix[bin * bands + band]});
}
void Peak::add(float value) {
    int live = 0, minimum = -1;
    for (int i = 0; i < static_cast<int>(entries.size()); ++i) if (entries[i].active) {
        ++live;
        if (minimum < 0 || entries[i].value < entries[minimum].value) minimum = i;
    }
    if (live < 4 || value > entries[minimum].value) {
        if (live == 4) entries[minimum].active = false;
        entries.push_back({age, true, value}); age = 0;
    }
    while (!entries.empty() && !entries.front().active) {
        auto gap = entries.front().gap; entries.erase(entries.begin());
        if (!entries.empty()) entries.front().gap += gap;
    }
    if (!entries.empty()) {
        if (entries.front().gap <= 0) entries.erase(entries.begin());
        else --entries.front().gap;
    }
    age = std::min(timeout, age + 1);
}
float Peak::maximum() const {
    float result = 0; bool found = false;
    for (auto& e : entries) if (e.active) { result = std::max(result, e.value); found = true; }
    return found ? result : 1.f;
}
Stage::Stage(std::shared_ptr<Runtime> rt, std::shared_ptr<Coefficients> coefficients,
             const std::filesystem::path& model, int device) : runtime(std::move(rt)), dsp(std::move(coefficients)) {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.AddConfigEntry("session.intra_op.allow_spinning", "0");
    options.SetExecutionMode(ORT_SEQUENTIAL);
    runtime->configureDevice(options, device);
    session = Ort::Session(*runtime->env, model.c_str(), options);
    runtime->verifyDevice(session, device);
    if (session.GetInputCount() != 5 || session.GetOutputCount() != 5)
        throw std::runtime_error("Unexpected OpenBroadcast model interface");
    forward = kiss_fftr_alloc(block, 0, nullptr, nullptr);
    inverse = kiss_fftr_alloc(block, 1, nullptr, nullptr);
    if (!forward || !inverse) { free(forward); free(inverse); throw std::bad_alloc(); }
    states = {std::vector<float>(256), std::vector<float>(256), std::vector<float>(512), std::vector<float>(1024)};
    reset();
    infer(1.f); // Warm kernels and memory before starting audio.
    reset();
}
Stage::~Stage() { free(forward); free(inverse); }
void Stage::reset() {
    for (auto& s : states) std::fill(s.begin(), s.end(), 0.f);
    input.fill(0); output.fill(0); history.fill(0); overlap.fill(0); features.fill(0);
    longPeak = Peak(60000); shortPeak = Peak(8000);
    peakHistory.fill(1); flagHistory.fill(false); vad.fill(0);
    vadSum = 0; vadActive = false; pos = vadPos = 0;
}
float Stage::tick(float sample, float intensity) {
    auto result = output[pos]; input[pos] = sample;
    if (++pos == block) { infer(intensity); pos = 0; }
    return result;
}
void Stage::infer(float intensity) {
    float peak = 0;
    for (int i = 0; i < hop; ++i) peak = std::max(peak, std::abs(input[i]));
    longPeak.add(peak); shortPeak.add(peak);
    std::array<float, 7> peaks{1,1,1,1,1,1,1}; std::array<bool, 7> flags{};
    std::copy(peakHistory.begin(), peakHistory.end(), peaks.begin());
    std::copy(flagHistory.begin(), flagHistory.end(), flags.begin());
    for (int i = 0; i < 4; ++i) {
        peaks[3+i] = longPeak.maximum(); float p = peaks[3];
        if (!flags[2]) {
            flags[3+i] = vadActive && (p < .075f || shortPeak.maximum() < .075f);
            if (flags[3+i] && p > .075f) longPeak.copyFrom(shortPeak);
        } else flags[3+i] = p < .075f;
        if (std::abs(p) < std::numeric_limits<float>::epsilon()) flags[3+i] = false;
        if (flags[3+i]) for (int j = 0; j < hop; ++j) input[i*hop+j] *= .3f / p;
    }
    std::copy(peaks.begin()+4, peaks.end(), peakHistory.begin());
    std::copy(flags.begin()+4, flags.end(), flagHistory.begin());
    std::copy(history.begin(), history.end(), joined.begin());
    std::copy(input.begin(), input.end(), joined.begin()+1440);
    for (int frame = 0; frame < 4; ++frame) {
        for (int i = 0; i < block; ++i) fftTime[i] = joined[frame*hop+i] * dsp->analysis[i];
        kiss_fftr(forward, fftTime.data(), spectrum[frame].data());
        std::array<float, bins> power{};
        for (int i = 0; i < bins; ++i) power[i] = spectrum[frame][i].r*spectrum[frame][i].r + spectrum[frame][i].i*spectrum[frame][i].i;
        power[0] += power[bins-1]; power[bins-1] = 0;
        std::array<double, bands> mel{};
        for (auto& m : dsp->mel) mel[m.band] += static_cast<double>(power[m.bin] * m.value);
        for (int i = 0; i < bands; ++i)
            features[(6+frame)*bands+i] = std::clamp(std::log10(std::max(static_cast<float>(mel[i]), std::ldexp(1.f,-24))) / 8.8f + 4.f/8.8f, 0.f, 1.f);
    }
    const int64_t shape[]{1,1,10,bands};
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(memory, features.data(), features.size(), shape, 4));
    for (auto& state : states) {
        const int64_t stateShape[]{1, static_cast<int64_t>(state.size())};
        inputs.push_back(Ort::Value::CreateTensor<float>(memory, state.data(), state.size(), stateShape, 2));
    }
    const char* names[]{"main_input", "input_1", "input_2", "input_3", "input_4"};
    const char* outNames[]{"main_output", "output_1", "output_2", "output_3", "output_4"};
    auto outputs = session.Run(Ort::RunOptions{nullptr}, names, inputs.data(), 5, outNames, 5);
    if (outputs[0].GetTensorTypeAndShapeInfo().GetElementCount() != 4*bins)
        throw std::runtime_error("Invalid model mask dimensions");
    for (int i = 0; i < 4; ++i) {
        const auto count = outputs[i+1].GetTensorTypeAndShapeInfo().GetElementCount();
        if (count < states[i].size()) throw std::runtime_error("Invalid recurrent state dimensions");
        const float* last = outputs[i+1].GetTensorData<float>() + count - states[i].size();
        std::copy_n(last, states[i].size(), states[i].begin());
    }
    std::move(features.begin()+4*bands, features.end(), features.begin());
    std::copy(joined.end()-1440, joined.end(), history.begin());
    ola.fill(0); std::copy(overlap.begin(), overlap.end(), ola.begin());
    const float* mask = outputs[0].GetTensorData<float>();
    for (int frame = 0; frame < 4; ++frame) {
        for (int i = 0; i < bins; ++i) {
            float gain = 1.f - intensity * mask[frame*bins+(i == bins-1 ? 0 : i)];
            masked[i] = {spectrum[frame][i].r*gain, spectrum[frame][i].i*gain};
        }
        kiss_fftri(inverse, masked.data(), fftTime.data());
        for (int i = 0; i < block; ++i) ola[frame*hop+i] += fftTime[i] * dsp->synthesis[i] / block;
    }
    std::copy(ola.begin()+block, ola.end(), overlap.begin());
    std::copy_n(ola.begin(), block, output.begin());
    for (int frame = 0; frame < 4; ++frame) {
        vadSum -= vad[vadPos]; float energy = 0;
        for (int i = 0; i < hop; ++i) energy += std::abs(output[frame*hop+i]);
        vad[vadPos] = energy; vadPos = (vadPos+1)%15; vadSum += energy;
        vadActive = vadSum / 7200.f > 1e-4f;
        if (!vadActive) break;
    }
    for (int frame = 0; frame < 4; ++frame)
        if (flags[frame]) for (int i = 0; i < hop; ++i) output[frame*hop+i] *= peaks[frame]/.3f;
    for (float x : output) if (!std::isfinite(x)) throw std::runtime_error("Model produced nonfinite audio");
}
Resampler::Resampler(double in, double out, double delay) : ratio(in/out), cutoff(std::min(1., out/in)*.94), delay(delay) {
    for(int phase=0;phase<1024;++phase) {
        double weight=0;
        for(int k=-31;k<=32;++k) { auto w=kernel(k-phase/1024.); table[phase][k+31]=static_cast<float>(w); weight+=w; }
        for(auto& value:table[phase])value=static_cast<float>(value/weight);
    }
}
double Resampler::kernel(double x) const {
    constexpr double pi = 3.14159265358979323846;
    if (std::abs(x) >= 32.) return 0;
    const double sinc = std::abs(x) < 1e-9 ? cutoff : std::sin(pi*cutoff*x)/(pi*x);
    return sinc * (.42 + .5*std::cos(pi*x/32.) + .08*std::cos(2*pi*x/32.));
}
Channel::Channel(std::shared_ptr<Runtime> rt, std::shared_ptr<Coefficients> dsp,
                 const std::filesystem::path& assets, double rate, int device, bool studioMode)
    : down(rate, 48000),
      up(48000, rate, 32. + (std::ceil((studioMode ? studioBlock : 2*latency)*rate/48000 + 32 + 32*rate/48000 + 8) - ((studioMode ? studioBlock : 2*latency)*rate/48000 + 32 + 32*rate/48000 + 8))*48000/rate), native(rate == 48000) {
    if(studioMode) studio=std::make_unique<StudioVoice>(rt,assets/"models/studio_voice_48k.onnx",device);
    else {
        reverb=std::make_unique<Stage>(rt,dsp,assets/"models/dereverb_48k.onnx",device);
        noise=std::make_unique<Stage>(rt,dsp,assets/"models/denoiser_48k.onnx",device);
    }
}
Channel::~Channel() = default;
float Channel::tick(float input, float denoise, float dereverb) {
    auto process = [&](float x) { return studio ? studio->tick(x) : noise->tick(reverb->tick(x, dereverb), denoise); };
    if (native) return process(input);
    down.push(input, [&](float x) { up.push(process(x), [&](float y) { queue[write++ % queue.size()] = y; }); });
    if (read == write) throw std::runtime_error("Resampler underflow");
    return queue[read++ % queue.size()];
}
void Channel::reset() {
    if(studio)studio->reset();
    else { reverb->reset(); noise->reset(); }
    down.reset(); up.reset();
    queue.fill(0); read=0; write=8;
}
int Channel::delay(double rate, bool studio) {
    const int samples=studio ? studioBlock : 2*latency;
    return rate == 48000 ? samples : static_cast<int>(std::ceil(samples*rate/48000 + 32 + 32*rate/48000 + 8));
}
}
