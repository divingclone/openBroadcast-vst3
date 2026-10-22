#pragma once
#include <array>
#include <vector>
#include <memory>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <onnxruntime_cxx_api.h>
#include <kiss_fftr.h>

namespace ob {
class StudioVoice;
constexpr int hop = 480, block = 1920, bins = 961, bands = 320, latency = 3360;
struct Runtime {
    explicit Runtime(const std::filesystem::path& directory);
    ~Runtime();
    void configureDevice(Ort::SessionOptions&, int device) const;
    void verifyDevice(const Ort::Session&, int device) const;
    void* library{};
    void* directml{};
    std::unique_ptr<Ort::Env> env;
    std::filesystem::path directory;
};
struct Coefficients {
    explicit Coefficients(const std::filesystem::path& file);
    std::array<float, block> analysis{}, synthesis{};
    struct Mel { int bin, band; float value; };
    std::vector<Mel> mel;
};
class Peak {
public:
    explicit Peak(int timeout) : timeout(timeout), age(timeout) {}
    void add(float value);
    float maximum() const;
    void copyFrom(const Peak& other) { age = other.age; entries = other.entries; entries.erase(std::remove_if(entries.begin(),entries.end(),[](auto& e){return !e.active;}),entries.end()); }
private:
    struct Entry { int gap; bool active; float value; };
    int timeout, age;
    std::vector<Entry> entries;
};
class Stage {
public:
    Stage(std::shared_ptr<Runtime>, std::shared_ptr<Coefficients>, const std::filesystem::path&, int device = -1);
    ~Stage();
    void reset();
    float tick(float input, float intensity);
private:
    void infer(float intensity);
    std::shared_ptr<Runtime> runtime;
    std::shared_ptr<Coefficients> dsp;
    Ort::Session session{nullptr};
    Ort::MemoryInfo memory{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    kiss_fftr_cfg forward{}, inverse{};
    std::array<std::vector<float>, 4> states;
    std::array<float, block> input{}, output{}, fftTime{};
    std::array<float, 3360> joined{}, ola{};
    std::array<float, 1440> history{}, overlap{};
    std::array<float, 3200> features{};
    std::array<std::array<kiss_fft_cpx, bins>, 4> spectrum{};
    std::array<kiss_fft_cpx, bins> masked{};
    Peak longPeak{60000}, shortPeak{8000};
    std::array<float, 3> peakHistory{1,1,1};
    std::array<bool, 3> flagHistory{};
    std::array<float, 15> vad{};
    float vadSum{};
    bool vadActive{};
    int pos{}, vadPos{};
};

// Causal, windowed-sinc sample-rate converter. No allocation in push().
class Resampler {
public:
    Resampler(double inputRate, double outputRate, double delay = 32.);
    void reset() { inputIndex = outputIndex = 0; history.fill(0); }
    template<class Sink> void push(float sample, Sink&& sink) {
        history[static_cast<size_t>(inputIndex) % history.size()] = sample;
        while (static_cast<double>(outputIndex) * ratio <= static_cast<double>(inputIndex) + 1e-8) {
            const double center = static_cast<double>(outputIndex) * ratio - delay;
            const auto base = static_cast<int64_t>(std::floor(center));
            double sum = 0;
            const auto phase = std::min(1023, static_cast<int>((center-static_cast<double>(base))*1024));
            for (int k = -31; k <= 32; ++k) {
                const auto index = base + k;
                const double w = table[phase][k+31];
                if (index >= 0 && index <= inputIndex)
                    sum += history[static_cast<size_t>(index) % history.size()] * w;
            }
            sink(static_cast<float>(sum));
            ++outputIndex;
        }
        ++inputIndex;
    }
private:
    double kernel(double x) const;
    double ratio, cutoff, delay;
    int64_t inputIndex{}, outputIndex{};
    std::array<float, 256> history{};
    std::array<std::array<float,64>,1024> table{};
};
class Channel {
public:
    Channel(std::shared_ptr<Runtime>, std::shared_ptr<Coefficients>, const std::filesystem::path&, double rate, int device, bool studio = false);
    ~Channel();
    float tick(float input, float denoise, float dereverb);
    void reset();
    static int delay(double rate, bool studio = false);
private:
    std::unique_ptr<Stage> reverb, noise;
    std::unique_ptr<StudioVoice> studio;
    Resampler down, up;
    bool native;
    std::array<float, 256> queue{};
    size_t read{}, write{8};
};
}
