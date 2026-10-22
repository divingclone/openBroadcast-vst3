#pragma once
#include "Engine.h"
#include <string>

namespace ob {
// Sample-by-sample streaming needs one full block (80 ms). The Python
// 480-sample API can return its first output hop during the eighth input call.
constexpr int studioBlock = 3840;
class StudioVoice {
public:
    StudioVoice(std::shared_ptr<Runtime>, const std::filesystem::path& model, int device);
    ~StudioVoice();
    void reset();
    float tick(float sample);
private:
    void infer();
    std::shared_ptr<Runtime> runtime;
    void* native{};
    Ort::Session session{nullptr};
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
    Ort::MemoryInfo deviceMemory{nullptr};
    std::unique_ptr<Ort::IoBinding> deviceBinding;
    std::vector<std::string> inputNames, outputNames;
    std::vector<const char*> inputPointers, outputPointers;
    std::vector<Ort::Value> inputs;
    size_t audioIndex{};
    std::array<float, studioBlock> input{}, output{};
    int position{};
};
}
