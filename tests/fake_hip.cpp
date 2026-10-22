// A test-only HIP discovery ABI. This library cannot execute GPU kernels.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
static bool mode(const char* name) {
    const auto* value=std::getenv("OB_TEST_HIP_MODE");
    return value && std::strcmp(value,name)==0;
}
extern "C" {
int hipInit(unsigned) { return mode("init-failure") ? 100 : 0; }
int hipGetDeviceCount(int* count) { *count=mode("empty") ? 0 : 3; return 0; }
int hipDeviceGet(int* device,int ordinal) { *device=ordinal; return 0; }
int hipDeviceGetName(char* name,int size,int device) {
    if(device==1)return 1; // A filtered device must not renumber later ordinals.
    std::snprintf(name,static_cast<size_t>(size),"%s",device==0 ? "AMD Test Integrated" : "AMD Test Discrete");
    return 0;
}
int hipDeviceTotalMem(size_t* size,int device) {
    *size=(device==0 ? 32ull : 16ull)<<30;
    return mode("attributes-unavailable") ? 1 : 0;
}
int hipDeviceGetAttribute(int* value,int attribute,int device) {
    if(mode("attributes-unavailable"))return 1;
    switch(attribute) {
        case 5: *value=2000000; return 0;
        case 16: *value=device==0 ? 1 : 0; return 0;
        case 63: *value=device==0 ? 8 : 64; return 0;
        default: return 1;
    }
}
}
