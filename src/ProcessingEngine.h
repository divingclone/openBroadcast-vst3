#pragma once
#include "Engine.h"
#include "GpuDevices.h"
#include "Transport.h"
namespace ob {
std::vector<GpuDevice> discoverProcessingDevices(const std::filesystem::path& resources);
class ProcessingEngine {
public:
    ProcessingEngine(const std::filesystem::path& resources,double rate,int channels,
                     const GpuDevice* gpu,bool studio);
    ~ProcessingEngine();
    void process(Packet&);
    void reset();
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}
