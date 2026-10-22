#include "GpuDevices.h"
#ifdef _WIN32
#include <windows.h>
#include <dxgi1_6.h>
#include <d3d12.h>
#include <wrl/client.h>
#elif !defined(__APPLE__)
#include <dlfcn.h>
#include <codecvt>
#include <locale>
#endif
#include <algorithm>
#include <limits>

namespace ob {
const char* gpuBackendName(GpuBackend backend) {
    return backend==GpuBackend::migraphx ? "MIGraphX" : gpuBackendName();
}
const char* gpuBackendName() {
#ifdef _WIN32
    return "DirectML";
#elif defined(__APPLE__)
    return "CoreML (system-managed CPU/GPU)";
#elif defined(OB_LINUX_MIGRAPHX)
    return "MIGraphX";
#else
    return "CUDA";
#endif
}
std::vector<GpuDevice> rankGpuDevices(std::vector<GpuDevice> devices) {
    std::stable_sort(devices.begin(), devices.end(), [](const auto& a, const auto& b) {
        const auto rank = [](int r) { return r < 0 ? std::numeric_limits<int>::max() : r; };
        if (rank(a.performanceRank) != rank(b.performanceRank))
            return rank(a.performanceRank) < rank(b.performanceRank);
        if (a.computeScore != b.computeScore) return a.computeScore > b.computeScore;
        if (a.dedicatedMemory != b.dedicatedMemory) return a.dedicatedMemory > b.dedicatedMemory;
        return a.index < b.index;
    });
    return devices;
}
std::vector<GpuDevice> discoverGpuDevices() {
#ifdef _WIN32
    using Microsoft::WRL::ComPtr;
    ComPtr<IDXGIFactory1> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return {};
    std::vector<GpuDevice> devices;
    std::vector<LUID> identities;
    // The existing saved GPU index parameter has a 0..15 range.
    for (UINT i = 0; i < 16; ++i) {
        ComPtr<IDXGIAdapter1> adapter;
        if (FAILED(factory->EnumAdapters1(i, &adapter))) break;
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(adapter->GetDesc1(&desc)) || (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) continue;
        if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), nullptr))) continue;
        devices.push_back({static_cast<int>(i), desc.Description, desc.DedicatedVideoMemory, -1});
        identities.push_back(desc.AdapterLuid);
    }
    ComPtr<IDXGIFactory6> preferred;
    if (SUCCEEDED(factory.As(&preferred))) {
        for (UINT rank = 0; rank < 16; ++rank) {
            ComPtr<IDXGIAdapter1> adapter;
            if (FAILED(preferred->EnumAdapterByGpuPreference(rank, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter)))) break;
            DXGI_ADAPTER_DESC1 desc{};
            if (FAILED(adapter->GetDesc1(&desc))) continue;
            for (size_t i = 0; i < devices.size(); ++i)
                if (identities[i].HighPart == desc.AdapterLuid.HighPart && identities[i].LowPart == desc.AdapterLuid.LowPart)
                    devices[i].performanceRank = static_cast<int>(rank);
        }
    }
    return rankGpuDevices(std::move(devices));
#elif defined(__APPLE__)
    return {{0, L"macOS GPU (system-managed)", 0, 0}};
#elif defined(OB_LINUX_MIGRAPHX)
    // HIP ordinals honor HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES. Query the
    // stable scalar API, avoiding the version-dependent hipDeviceProp_t layout.
    // Keep HIP loaded for the process lifetime: the provider may share its state.
    static void* library=[] {
        for(const char* path:{"libamdhip64.so.7", "libamdhip64.so",
                              "/opt/rocm/lib/libamdhip64.so.7", "/opt/rocm/lib/libamdhip64.so"})
            if(auto* handle=dlopen(path, RTLD_NOW | RTLD_LOCAL))return handle;
        return static_cast<void*>(nullptr);
    }();
    if(!library)return {};
    auto init=reinterpret_cast<int(*)(unsigned)>(dlsym(library,"hipInit"));
    auto countFn=reinterpret_cast<int(*)(int*)>(dlsym(library,"hipGetDeviceCount"));
    auto get=reinterpret_cast<int(*)(int*,int)>(dlsym(library,"hipDeviceGet"));
    auto nameFn=reinterpret_cast<int(*)(char*,int,int)>(dlsym(library,"hipDeviceGetName"));
    auto memoryFn=reinterpret_cast<int(*)(size_t*,int)>(dlsym(library,"hipDeviceTotalMem"));
    auto attribute=reinterpret_cast<int(*)(int*,int,int)>(dlsym(library,"hipDeviceGetAttribute"));
    std::vector<GpuDevice> devices;
    int count=0;
    if(init && countFn && get && nameFn && !init(0) && !countFn(&count)) {
        // hipDeviceAttribute_t ABI values from ROCm HIP; these are not CUDA values.
        constexpr int clockRate=5, integratedGpu=16, multiprocessorCount=63;
        for(int i=0;i<std::min(count,16);++i) {
            int device=0,units=0,clock=0,integrated=0; char name[256]{}; size_t memory=0;
            if(get(&device,i) || nameFn(name,sizeof(name)-1,device))continue;
            if(memoryFn && memoryFn(&memory,device))memory=0;
            int rank=-1;
            if(attribute) {
                if(attribute(&units,multiprocessorCount,device))units=0;
                if(attribute(&clock,clockRate,device))clock=0;
                if(!attribute(&integrated,integratedGpu,device))rank=integrated ? 1 : 0;
            }
            devices.push_back({i,std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(name),memory,rank,
                               static_cast<uint64_t>(std::max(0,units))*std::max(0,clock)});
        }
    }
    return rankGpuDevices(std::move(devices));
#else
    // Same CUDA driver discovery and SM-count * clock ranking as OpenBroadcast.
    // Driver ordinals honor CUDA_VISIBLE_DEVICES; never renumber after sorting.
    void* library=dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if(!library)return {};
    auto init=reinterpret_cast<int(*)(unsigned)>(dlsym(library,"cuInit"));
    auto countFn=reinterpret_cast<int(*)(int*)>(dlsym(library,"cuDeviceGetCount"));
    auto get=reinterpret_cast<int(*)(int*,int)>(dlsym(library,"cuDeviceGet"));
    auto nameFn=reinterpret_cast<int(*)(char*,int,int)>(dlsym(library,"cuDeviceGetName"));
    auto memoryFn=reinterpret_cast<int(*)(size_t*,int)>(dlsym(library,"cuDeviceTotalMem_v2"));
    auto attribute=reinterpret_cast<int(*)(int*,int,int)>(dlsym(library,"cuDeviceGetAttribute"));
    std::vector<GpuDevice> devices;
    int count=0;
    if(init && countFn && get && nameFn && !init(0) && !countFn(&count)) {
        for(int i=0;i<std::min(count,16);++i) {
            int device=0,sm=0,clock=0; char name[256]{}; size_t memory=0;
            if(get(&device,i) || nameFn(name,sizeof(name),device))continue;
            if(memoryFn && memoryFn(&memory,device))memory=0;
            if(attribute) { if(attribute(&sm,16,device))sm=0; if(attribute(&clock,13,device))clock=0; }
            devices.push_back({i,std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(name),memory,-1,
                               static_cast<uint64_t>(std::max(0,sm))*std::max(0,clock)});
        }
    }
    dlclose(library);
    return rankGpuDevices(std::move(devices));
#endif
}
}
