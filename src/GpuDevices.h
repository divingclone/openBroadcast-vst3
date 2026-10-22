#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace ob {
enum class GpuBackend { platform, migraphx };
struct GpuDevice {
    int index{}; // Native DXGI/CUDA/HIP ordinal; CoreML uses system-managed index 0.
    std::wstring name;
    uint64_t dedicatedMemory{};
    int performanceRank{-1};
    uint64_t computeScore{};
    GpuBackend backend{GpuBackend::platform};
    int selectionId() const { return index+(backend==GpuBackend::migraphx ? 16 : 0); }
};
const char* gpuBackendName();
const char* gpuBackendName(GpuBackend backend);
std::vector<GpuDevice> discoverGpuDevices();
std::vector<GpuDevice> rankGpuDevices(std::vector<GpuDevice> devices);
}
