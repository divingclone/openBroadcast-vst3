// Test-only discovery API. GPU execution is intentionally unsupported.
#include <cstdlib>
#include <cstdio>
#include <cstddef>
extern "C" {
int cuInit(unsigned) { return 0; }
int cuDeviceGetCount(int* count) { *count=std::getenv("OB_TEST_CUDA") ? 1 : 0; return 0; }
int cuDeviceGet(int* device,int ordinal) { *device=ordinal; return 0; }
int cuDeviceGetName(char* name,int size,int) { std::snprintf(name,static_cast<size_t>(size),"NVIDIA Test GPU");return 0; }
int cuDeviceTotalMem_v2(size_t* bytes,int) { *bytes=8ull<<30;return 0; }
int cuDeviceGetAttribute(int* value,int attribute,int) { *value=attribute==16 ? 10 : 1000000;return 0; }
}
