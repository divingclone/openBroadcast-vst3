// Imported from OpenBroadcast audio/native/explicit_dml.cpp.
// Built into the plugin; only the export macro differs from the upstream helper.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "dml_provider_factory.h"
#include <DirectML.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <windows.h>
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;
using Clock = std::chrono::steady_clock;
static thread_local std::string error;
static double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}
static void hr(HRESULT h) {
  if (FAILED(h)) {
    char s[40];
    sprintf_s(s, "HRESULT 0x%08X", (unsigned)h);
    throw std::runtime_error(s);
  }
}
struct Timing {
  double cpu_upload_ms, cpu_submit_ms, cpu_wait_ms, cpu_download_ms, total_ms,
      gpu_upload_ms, gpu_inference_queue_ms, gpu_download_ms;
};
struct Tensor {
  std::string name;
  std::vector<int64_t> shape;
  ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  size_t bytes = 0;
  ComPtr<ID3D12Resource> resource;
  void *wrapped = nullptr;
  OrtValue *value = nullptr;
};
struct Engine {
  HMODULE ortlib = nullptr, dmllib = nullptr;
  const OrtApi *api = nullptr;
  const OrtDmlApi *dmlapi = nullptr;
  OrtEnv *env = nullptr;
  OrtSession *session = nullptr;
  OrtSessionOptions *options = nullptr;
  OrtRunOptions *runopts = nullptr;
  OrtMemoryInfo *memory = nullptr;
  OrtAllocator *allocator = nullptr;
  OrtIoBinding *bindings[2] = {};
  ComPtr<ID3D12Device> device;
  ComPtr<ID3D12CommandQueue> queue;
  ComPtr<IDMLDevice> dmldevice;
  ComPtr<ID3D12Fence> fence;
  HANDLE event = nullptr;
  UINT64 fencevalue = 0, frequency = 0;
  ComPtr<ID3D12CommandAllocator> allocs[2];
  ComPtr<ID3D12GraphicsCommandList> lists[2];
  ComPtr<ID3D12Resource> upload, readback, queryread;
  ComPtr<ID3D12QueryHeap> queries;
  uint8_t *uploadptr = nullptr;
  uint8_t *readptr = nullptr;
  uint64_t *queryptr = nullptr;
  std::vector<Tensor> state[2];
  Tensor audioin, audioout;
  size_t staging = 0;
  int bank = 0;
  bool profile = false;
  std::wstring adaptername;
  void check(OrtStatus *s) {
    if (s) {
      std::string x = api->GetErrorMessage(s);
      api->ReleaseStatus(s);
      throw std::runtime_error(x);
    }
  }
  ComPtr<ID3D12Resource> buffer(size_t size, D3D12_HEAP_TYPE heap,
                                D3D12_RESOURCE_STATES initial) {
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = heap;
    D3D12_RESOURCE_DESC d = {};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = (std::max<size_t>(size, 4) + 3) & ~size_t(3);
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    d.Flags = heap == D3D12_HEAP_TYPE_DEFAULT
                  ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                  : D3D12_RESOURCE_FLAG_NONE;
    ComPtr<ID3D12Resource> r;
    hr(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &d, initial,
                                       nullptr, IID_PPV_ARGS(&r)));
    return r;
  }
  void wrap(Tensor &t) {
    t.resource = buffer(t.bytes, D3D12_HEAP_TYPE_DEFAULT,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    check(dmlapi->CreateGPUAllocationFromD3DResource(t.resource.Get(),
                                                     &t.wrapped));
    check(api->CreateTensorWithDataAsOrtValue(memory, t.wrapped, t.bytes,
                                              t.shape.data(), t.shape.size(),
                                              t.type, &t.value));
  }
  Tensor metadata(size_t i, bool input) {
    Tensor t;
    char *name = nullptr;
    OrtTypeInfo *ti = nullptr;
    const OrtTensorTypeAndShapeInfo *tsi = nullptr;
    check(input ? api->SessionGetInputName(session, i, allocator, &name)
                : api->SessionGetOutputName(session, i, allocator, &name));
    t.name = name;
    allocator->Free(allocator, name);
    check(input ? api->SessionGetInputTypeInfo(session, i, &ti)
                : api->SessionGetOutputTypeInfo(session, i, &ti));
    auto releaseType = [this](OrtTypeInfo *v) { api->ReleaseTypeInfo(v); };
    std::unique_ptr<OrtTypeInfo, decltype(releaseType)> typeOwner(ti,
                                                                  releaseType);
    check(api->CastTypeInfoToTensorInfo(ti, &tsi));
    if (!tsi)
      throw std::runtime_error(
          "Only tensor model inputs/outputs are supported");
    size_t n = 0;
    check(api->GetDimensionsCount(tsi, &n));
    t.shape.resize(n);
    check(api->GetDimensions(tsi, t.shape.data(), n));
    check(api->GetTensorElementType(tsi, &t.type));
    if (t.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT &&
        t.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 &&
        t.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
      throw std::runtime_error("Unsupported tensor dtype: " + t.name);
    size_t count = 1;
    for (auto d : t.shape) {
      if (d <= 0 || size_t(d) > std::numeric_limits<size_t>::max() / count / 4)
        throw std::runtime_error("Invalid or overflowing tensor shape: " +
                                 t.name);
      count *= d;
    }
    t.bytes = count * (t.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ? 2 : 4);
    return t;
  }
  void barrier(ID3D12GraphicsCommandList *l, ID3D12Resource *r,
               D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    l->ResourceBarrier(1, &b);
  }
  ID3D12GraphicsCommandList *begin(int i) {
    hr(allocs[i]->Reset());
    hr(lists[i]->Reset(allocs[i].Get(), nullptr));
    return lists[i].Get();
  }
  void submit(int i) {
    hr(lists[i]->Close());
    ID3D12CommandList *l = lists[i].Get();
    queue->ExecuteCommandLists(1, &l);
  }
  void wait() {
    auto v = ++fencevalue;
    hr(queue->Signal(fence.Get(), v));
    if (fence->GetCompletedValue() < v) {
      hr(fence->SetEventOnCompletion(v, event));
      if (WaitForSingleObject(event, INFINITE) != WAIT_OBJECT_0)
        throw std::runtime_error("GPU fence wait failed");
    }
    hr(device->GetDeviceRemovedReason());
  }
  void init(const wchar_t *model, const wchar_t *runtime, int adapter,
            bool timing, int threads) {
    profile = timing;
    std::wstring dir = runtime;
    ortlib = LoadLibraryExW((dir + L"\\onnxruntime.dll").c_str(), nullptr,
                            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                                LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    dmllib = LoadLibraryExW((dir + L"\\DirectML.dll").c_str(), nullptr,
                            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                                LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!ortlib || !dmllib)
      throw std::runtime_error("Cannot load ORT/DirectML runtime DLL");
    auto base = (const OrtApiBase *(ORT_API_CALL *)())GetProcAddress(
        ortlib, "OrtGetApiBase");
    if (!base)
      throw std::runtime_error("OrtGetApiBase missing");
    api = base()->GetApi(ORT_API_VERSION);
    if (!api)
      throw std::runtime_error("ORT API version mismatch");
    check(api->GetExecutionProviderApi("DML", ORT_API_VERSION,
                                       (const void **)&dmlapi));
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter1> selected;
    hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
    hr(factory->EnumAdapters1(adapter, &selected));
    DXGI_ADAPTER_DESC1 desc;
    hr(selected->GetDesc1(&desc));
    adaptername = desc.Description;
    hr(D3D12CreateDevice(selected.Get(), D3D_FEATURE_LEVEL_11_0,
                         IID_PPV_ARGS(&device)));
    D3D12_COMMAND_QUEUE_DESC q = {};
    q.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    q.Flags = D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT;
    hr(device->CreateCommandQueue(&q, IID_PPV_ARGS(&queue)));
    auto create = (HRESULT(WINAPI *)(
        ID3D12Device *, DML_CREATE_DEVICE_FLAGS, DML_FEATURE_LEVEL, REFIID,
        void **))GetProcAddress(dmllib, "DMLCreateDevice1");
    if (!create)
      throw std::runtime_error("DMLCreateDevice1 missing");
    hr(create(device.Get(), DML_CREATE_DEVICE_FLAG_NONE, DML_FEATURE_LEVEL_5_0,
              IID_PPV_ARGS(&dmldevice)));
    hr(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
    event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event)
      throw std::runtime_error("Cannot create GPU fence event");
    hr(queue->GetTimestampFrequency(&frequency));
    for (int i = 0; i < 2; i++) {
      hr(device->CreateCommandAllocator(q.Type, IID_PPV_ARGS(&allocs[i])));
      hr(device->CreateCommandList(0, q.Type, allocs[i].Get(), nullptr,
                                   IID_PPV_ARGS(&lists[i])));
      hr(lists[i]->Close());
    }
    check(api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "explicit-dml", &env));
    check(api->CreateSessionOptions(&options));
    check(api->DisableMemPattern(options));
    check(api->SetSessionExecutionMode(options, ORT_SEQUENTIAL));
    check(api->SetIntraOpNumThreads(options, threads));
    check(api->SetSessionGraphOptimizationLevel(options, ORT_ENABLE_ALL));
    check(dmlapi->SessionOptionsAppendExecutionProvider_DML1(
        options, dmldevice.Get(), queue.Get()));
    check(api->CreateSession(env, model, options, &session));
    check(api->GetAllocatorWithDefaultOptions(&allocator));
    check(api->CreateMemoryInfo("DML", OrtDeviceAllocator, 0, OrtMemTypeDefault,
                                &memory));
    check(api->CreateRunOptions(&runopts));
    check(api->AddRunConfigEntry(
        runopts, "disable_synchronize_execution_providers", "1"));
    size_t count;
    check(api->SessionGetInputCount(session, &count));
    for (size_t i = 0; i < count; i++) {
      Tensor t = metadata(i, true);
      if (t.name == "input_audio")
        audioin = std::move(t);
      else {
        state[0].push_back(std::move(t));
      }
    }
    check(api->SessionGetOutputCount(session, &count));
    std::map<std::string, Tensor> outputInfo;
    for (size_t i = 0; i < count; i++) {
      Tensor t = metadata(i, false);
      if (t.name == "out_audio_pre_tanh" || t.name == "out_audio")
        audioout = std::move(t);
      else
        outputInfo.emplace(t.name, std::move(t));
    }
    if (audioin.name.empty() || audioout.name.empty() ||
        audioin.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        audioout.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        audioin.bytes != 3840 * sizeof(float) ||
        audioout.bytes != audioin.bytes)
      throw std::runtime_error(
          "Expected float32 input_audio and waveform output with 3840 samples");
    if (outputInfo.size() != state[0].size())
      throw std::runtime_error("State input/output count mismatch");
    for (const auto &t : state[0]) {
      auto found = outputInfo.find("out_" + t.name);
      if (found == outputInfo.end() || found->second.type != t.type ||
          found->second.shape != t.shape)
        throw std::runtime_error("State output metadata mismatch: " + t.name);
    }
    state[1].reserve(state[0].size());
    for (auto &t : state[0]) {
      Tensor c;
      c.name = t.name;
      c.shape = t.shape;
      c.type = t.type;
      c.bytes = t.bytes;
      state[1].push_back(std::move(c));
    }
    wrap(audioin);
    wrap(audioout);
    staging = std::max(audioin.bytes, audioout.bytes);
    for (auto &b : state)
      for (auto &t : b) {
        wrap(t);
        staging = std::max(staging, t.bytes);
      }
    upload = buffer(staging, D3D12_HEAP_TYPE_UPLOAD,
                    D3D12_RESOURCE_STATE_GENERIC_READ);
    readback = buffer(staging, D3D12_HEAP_TYPE_READBACK,
                      D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_RANGE empty = {0, 0};
    hr(upload->Map(0, &empty, (void **)&uploadptr));
    hr(readback->Map(0, nullptr, (void **)&readptr));
    if (profile) {
      D3D12_QUERY_HEAP_DESC qh = {};
      qh.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
      qh.Count = 4;
      hr(device->CreateQueryHeap(&qh, IID_PPV_ARGS(&queries)));
      queryread =
          buffer(32, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
      hr(queryread->Map(0, nullptr, (void **)&queryptr));
    }
    for (int b = 0; b < 2; b++) {
      check(api->CreateIoBinding(session, &bindings[b]));
      check(api->BindInput(bindings[b], audioin.name.c_str(), audioin.value));
      check(
          api->BindOutput(bindings[b], audioout.name.c_str(), audioout.value));
      for (size_t i = 0; i < state[b].size(); i++) {
        check(api->BindInput(bindings[b], state[b][i].name.c_str(),
                             state[b][i].value));
        check(api->BindOutput(bindings[b], ("out_" + state[b][i].name).c_str(),
                              state[1 - b][i].value));
      }
    }
    reset();
  }
  void reset() {
    wait();
    memset(uploadptr, 0, staging);
    auto l = begin(0);
    for (auto &b : state)
      for (auto &t : b) {
        barrier(l, t.resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_DEST);
        l->CopyBufferRegion(t.resource.Get(), 0, upload.Get(), 0, t.bytes);
        barrier(l, t.resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      }
    submit(0);
    wait();
    bank = 0;
  }
  void process(const float *input, float *output, Timing *times) {
    Timing t = {};
    auto start = Clock::now();
    memcpy(uploadptr, input, audioin.bytes);
    auto l = begin(0);
    if (profile)
      l->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
    barrier(l, audioin.resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_DEST);
    l->CopyBufferRegion(audioin.resource.Get(), 0, upload.Get(), 0,
                        audioin.bytes);
    barrier(l, audioin.resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (profile)
      l->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
    submit(0);
    auto submittedUpload = Clock::now();
    check(api->RunWithBinding(session, runopts, bindings[bank]));
    auto submittedInference = Clock::now();
    l = begin(1);
    if (profile)
      l->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 2);
    barrier(l, audioout.resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
    l->CopyBufferRegion(readback.Get(), 0, audioout.resource.Get(), 0,
                        audioout.bytes);
    barrier(l, audioout.resource.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (profile) {
      l->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 3);
      l->ResolveQueryData(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 4,
                          queryread.Get(), 0);
    }
    submit(1);
    auto submittedAll = Clock::now();
    wait();
    auto waited = Clock::now();
    memcpy(output, readptr, audioout.bytes);
    bank = 1 - bank;
    auto end = Clock::now();
    t.cpu_upload_ms = ms(start, submittedUpload);
    t.cpu_submit_ms = ms(submittedUpload, submittedAll);
    t.cpu_wait_ms = ms(submittedAll, waited);
    t.cpu_download_ms = ms(waited, end);
    t.total_ms = ms(start, end);
    if (profile) {
      t.gpu_upload_ms = (queryptr[1] - queryptr[0]) * 1000.0 / frequency;
      t.gpu_inference_queue_ms =
          (queryptr[2] - queryptr[1]) * 1000.0 / frequency;
      t.gpu_download_ms = (queryptr[3] - queryptr[2]) * 1000.0 / frequency;
    }
    if (times)
      *times = t;
  }
  void copy_state(int i, void *dst, size_t bytes) {
    if (i < 0 || i >= int(state[bank].size()))
      throw std::runtime_error("state index");
    auto &t = state[bank][i];
    if (bytes != t.bytes)
      throw std::runtime_error("state bytes");
    auto l = begin(0);
    barrier(l, t.resource.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
    l->CopyBufferRegion(readback.Get(), 0, t.resource.Get(), 0, t.bytes);
    barrier(l, t.resource.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    submit(0);
    wait();
    memcpy(dst, readptr, bytes);
  }
  ~Engine() {
    // Finish outstanding GPU work before releasing wrapped allocations or
    // bindings.
    if (queue && fence) {
      try {
        wait();
      } catch (...) {
      }
    }
    if (api) {
      for (auto b : bindings)
        if (b)
          api->ReleaseIoBinding(b);
      auto release = [&](Tensor &t) {
        if (t.value)
          api->ReleaseValue(t.value);
        if (t.wrapped)
          dmlapi->FreeGPUAllocation(t.wrapped);
      };
      release(audioin);
      release(audioout);
      for (auto &b : state)
        for (auto &t : b)
          release(t);
      if (session)
        api->ReleaseSession(session);
      if (runopts)
        api->ReleaseRunOptions(runopts);
      if (options)
        api->ReleaseSessionOptions(options);
      if (memory)
        api->ReleaseMemoryInfo(memory);
      if (env)
        api->ReleaseEnv(env);
    }
    if (uploadptr)
      upload->Unmap(0, nullptr);
    if (readptr)
      readback->Unmap(0, nullptr);
    if (queryptr)
      queryread->Unmap(0, nullptr);
    if (event)
      CloseHandle(event);
    // COM vtables can live in DirectML/ORT. Release every dependent object
    // before balancing the two LoadLibrary calls, including partially
    // initialized cases.
    for (auto &b : state)
      b.clear();
    audioin.resource.Reset();
    audioout.resource.Reset();
    upload.Reset();
    readback.Reset();
    queryread.Reset();
    queries.Reset();
    for (auto &l : lists)
      l.Reset();
    for (auto &a : allocs)
      a.Reset();
    fence.Reset();
    dmldevice.Reset();
    queue.Reset();
    device.Reset();
    if (dmllib)
      FreeLibrary(dmllib);
    if (ortlib)
      FreeLibrary(ortlib);
  }
};
#define EXPORT extern "C"
EXPORT const char *edml_error() { return error.c_str(); }
EXPORT void *edml_create(const wchar_t *model, const wchar_t *runtime,
                         int adapter, int profile, int threads) {
  try {
    if (!model || !runtime || adapter < 0 || threads < 1)
      throw std::runtime_error("Invalid create arguments");
    auto p = std::make_unique<Engine>();
    p->init(model, runtime, adapter, profile != 0, threads);
    return p.release();
  } catch (const std::exception &e) {
    error = e.what();
    return nullptr;
  }
}
EXPORT int edml_process(void *p, const float *input, float *output,
                        Timing *timing) {
  try {
    if (!p || !input || !output)
      throw std::runtime_error("Null process argument");
    ((Engine *)p)->process(input, output, timing);
    return 0;
  } catch (const std::exception &e) {
    error = e.what();
    return -1;
  }
}
EXPORT int edml_reset(void *p) {
  try {
    if (!p)
      throw std::runtime_error("Null engine");
    ((Engine *)p)->reset();
    return 0;
  } catch (const std::exception &e) {
    error = e.what();
    return -1;
  }
}
EXPORT void edml_destroy(void *p) { delete (Engine *)p; }
EXPORT const wchar_t *edml_adapter(void *p) {
  return p ? ((Engine *)p)->adaptername.c_str() : nullptr;
}
EXPORT size_t edml_audio_samples(void *p, int output) {
  if (!p)
    return 0;
  return (output ? ((Engine *)p)->audioout.bytes
                 : ((Engine *)p)->audioin.bytes) /
         4;
}
EXPORT int edml_state_count(void *p) {
  return p ? int(((Engine *)p)->state[0].size()) : 0;
}
EXPORT const char *edml_state_name(void *p, int i) {
  if (!p || i < 0 || i >= edml_state_count(p))
    return nullptr;
  return ((Engine *)p)->state[0][i].name.c_str();
}
EXPORT size_t edml_state_bytes(void *p, int i) {
  if (!p || i < 0 || i >= edml_state_count(p))
    return 0;
  return ((Engine *)p)->state[0][i].bytes;
}
EXPORT int edml_copy_state(void *p, int i, void *dst, size_t size) {
  try {
    if (!p || !dst)
      throw std::runtime_error("Null state copy argument");
    ((Engine *)p)->copy_state(i, dst, size);
    return 0;
  } catch (const std::exception &e) {
    error = e.what();
    return -1;
  }
}
