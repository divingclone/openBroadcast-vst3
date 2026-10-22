#include "StudioVoice.h"
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
struct Timing;
extern "C" {
const char* edml_error();
void* edml_create(const wchar_t*, const wchar_t*, int, int, int);
int edml_process(void*, const float*, float*, Timing*);
int edml_reset(void*);
void edml_destroy(void*);
size_t edml_audio_samples(void*, int);
}
#endif
namespace ob {
StudioVoice::StudioVoice(std::shared_ptr<Runtime> rt, const std::filesystem::path& model, int device)
    : runtime(std::move(rt)) {
#ifdef _WIN32
    if (device >= 0) {
        native = edml_create(std::filesystem::absolute(model).c_str(), runtime->directory.c_str(), device, 0, 1);
        if (!native) throw std::runtime_error(std::string("Studio Voice: ") + edml_error());
        try {
            if (edml_audio_samples(native, 0) != studioBlock || edml_audio_samples(native, 1) != studioBlock)
                throw std::runtime_error("Invalid Studio Voice audio dimensions");
            reset(); infer(); reset();
        } catch (...) { edml_destroy(native); native = nullptr; throw; }
        return;
    }
#endif
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetInterOpNumThreads(1);
    options.SetExecutionMode(ORT_SEQUENTIAL);
    options.AddConfigEntry("session.intra_op.allow_spinning", "0");
    runtime->configureDevice(options, device);
    session = Ort::Session(*runtime->env, model.c_str(), options);
    runtime->verifyDevice(session, device);
    const auto count = session.GetInputCount();
    if (count < 2 || session.GetOutputCount() != count)
        throw std::runtime_error("Invalid Studio Voice streaming interface");
    bool foundAudio = false;
    for (size_t i = 0; i < count; ++i) {
        inputNames.emplace_back(session.GetInputNameAllocated(i, allocator).get());
        outputNames.push_back(inputNames.back() == "input_audio" ? "out_audio_pre_tanh" : "out_" + inputNames.back());
        auto typeInfo = session.GetInputTypeInfo(i);
        auto info = typeInfo.GetTensorTypeAndShapeInfo();
        const auto shape = info.GetShape();
        if (std::any_of(shape.begin(), shape.end(), [](auto n) { return n <= 0; }))
            throw std::runtime_error("Studio Voice requires fixed streaming tensor shapes");
        if (inputNames.back() == "input_audio") {
            if (shape != std::vector<int64_t>{1, studioBlock} || info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                throw std::runtime_error("Invalid Studio Voice input waveform");
            audioIndex = i; foundAudio = true;
            inputs.push_back(Ort::Value::CreateTensor<float>(memory, input.data(), input.size(), shape.data(), shape.size()));
        } else {
            const auto type = info.GetElementType();
            if (type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 && type != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
                throw std::runtime_error("Unsupported Studio Voice state type");
            inputs.push_back(Ort::Value::CreateTensor(allocator, shape.data(), shape.size(), type));
        }
    }
    if (!foundAudio) throw std::runtime_error("Studio Voice input_audio is missing");
    for (auto& name : inputNames) inputPointers.push_back(name.c_str());
    for (auto& name : outputNames) outputPointers.push_back(name.c_str());
#if !defined(_WIN32) && !defined(__APPLE__) && !defined(OB_LINUX_MIGRAPHX)
    if(device >= 0) {
        deviceMemory=Ort::MemoryInfo("Cuda", OrtDeviceAllocator, device, OrtMemTypeDefault);
        deviceBinding=std::make_unique<Ort::IoBinding>(session);
    }
#endif
    reset(); infer(); reset();
}
StudioVoice::~StudioVoice() {
#ifdef _WIN32
    if (native) edml_destroy(native);
#endif
}
void StudioVoice::reset() {
#ifdef _WIN32
    if (native) {
        if (edml_reset(native)) throw std::runtime_error(edml_error());
    } else
#endif
    {
        if(deviceBinding) { deviceBinding->ClearBoundInputs(); deviceBinding->ClearBoundOutputs(); }
        for (size_t i = 0; i < inputs.size(); ++i) if (i != audioIndex) {
            auto info = inputs[i].GetTensorTypeAndShapeInfo();
            const size_t bytes = info.GetElementCount() * (info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ? 2 : 4);
            if(deviceBinding) {
                // Reset on the host; the first run uploads zeros once. Subsequent
                // recurrent states remain device tensors, never copied to CPU.
                const auto shape=info.GetShape();
                inputs[i]=Ort::Value::CreateTensor(allocator, shape.data(), shape.size(), info.GetElementType());
            }
            std::memset(inputs[i].GetTensorMutableRawData(), 0, bytes);
        }
    }
    input.fill(0); output.fill(0); position = 0;
}
void StudioVoice::infer() {
#ifdef _WIN32
    if (native) {
        if (edml_process(native, input.data(), output.data(), nullptr)) throw std::runtime_error(edml_error());
    } else
#endif
    {
        std::vector<Ort::Value> results;
        if(deviceBinding) {
            deviceBinding->ClearBoundInputs(); deviceBinding->ClearBoundOutputs();
            for(size_t i=0;i<inputs.size();++i) {
                deviceBinding->BindInput(inputPointers[i], inputs[i]);
                deviceBinding->BindOutput(outputPointers[i], i==audioIndex ? memory : deviceMemory);
            }
            session.Run(Ort::RunOptions{nullptr}, *deviceBinding);
            deviceBinding->SynchronizeOutputs();
            results=deviceBinding->GetOutputValues();
        } else {
            results=session.Run(Ort::RunOptions{nullptr}, inputPointers.data(), inputs.data(), inputs.size(), outputPointers.data(), outputPointers.size());
        }
        const auto info = results[audioIndex].GetTensorTypeAndShapeInfo();
        if (info.GetElementCount() != studioBlock || info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("Invalid Studio Voice output waveform");
        std::copy_n(results[audioIndex].GetTensorData<float>(), studioBlock, output.begin());
        for (size_t i = 0; i < inputs.size(); ++i) if (i != audioIndex) inputs[i] = std::move(results[i]);
    }
    for (auto& sample : output) {
        if (!std::isfinite(sample)) throw std::runtime_error("Studio Voice produced nonfinite audio");
        sample = Ort::Float16_t(std::tanh(sample)).ToFloat();
    }
}
float StudioVoice::tick(float sample) {
    const auto result = output[position];
    input[position] = sample;
    if (++position == studioBlock) { infer(); position = 0; }
    return result;
}
}
