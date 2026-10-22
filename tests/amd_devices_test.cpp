#include "GpuDevices.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>
static void require(bool condition,const char* message) {
    if(!condition)throw std::runtime_error(message);
}
int main() {
    try {
        unsetenv("OB_TEST_HIP_MODE");
        auto devices=ob::discoverGpuDevices();
        require(devices.size()==2,"Discovery must skip a failed name query");
        require(devices[0].index==2 && devices[1].index==0,"Prefer discrete GPU while retaining HIP ordinals");
        require(devices[0].name==L"AMD Test Discrete","Display the HIP GPU model");
        require(devices[0].computeScore==128000000,"HIP attribute values must match its ABI, not CUDA");
        require(std::string(ob::gpuBackendName())=="MIGraphX","AMD backend name");
        setenv("OB_TEST_HIP_MODE","attributes-unavailable",1);
        devices=ob::discoverGpuDevices();
        require(devices.size()==2 && devices[0].dedicatedMemory==0 && devices[0].computeScore==0,
                "Missing optional attributes must preserve usable GPUs");
        for(const char* scenario:{"empty","init-failure"}) {
            setenv("OB_TEST_HIP_MODE",scenario,1);
            require(ob::discoverGpuDevices().empty(),"Unavailable HIP must permit CPU fallback");
        }
        std::cout<<"AMD DISCOVERY TESTS PASSED (simulated HIP)\n";
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
