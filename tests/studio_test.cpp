#include "StudioVoice.h"
#include "GpuDevices.h"
#include <fstream>
#include <iostream>
#include <chrono>
#include <stdexcept>
static std::vector<float> read(const std::filesystem::path& path) {
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if(!file)throw std::runtime_error("Missing Studio Voice fixture: "+path.string());
    std::vector<float> values(static_cast<size_t>(file.tellg())/sizeof(float));file.seekg(0);
    file.read(reinterpret_cast<char*>(values.data()),values.size()*sizeof(float));return values;
}
int main(int argc,char**argv) {
    try {
        if(argc<4)throw std::runtime_error("Expected assets runtime fixtures [GPU index]");
        const int device=argc>4 ? std::stoi(argv[4]) : -1;
        auto runtime=std::make_shared<ob::Runtime>(argv[2]);
        const auto input=read(std::filesystem::path(argv[3])/"studio_input.f32");
        const char* gpuFixture =
#ifdef _WIN32
            "studio_dml.f32";
#elif defined(__APPLE__)
            "studio_coreml.f32";
#else
            "studio_cuda.f32";
#endif
        const auto reference=read(std::filesystem::path(argv[3])/(device<0 ? "studio_cpu.f32" : gpuFixture));
        if(input.size()!=reference.size())throw std::runtime_error("Fixture size mismatch");
        auto engine=std::make_unique<ob::StudioVoice>(runtime,std::filesystem::path(argv[1])/"models/studio_voice_48k.onnx",device);
        std::vector<float> first;
        for(int pass=0;pass<2;++pass) {
            engine->reset();float maxError=0;double energy=0, firstBlockMs=0;
            auto start=std::chrono::steady_clock::now();
            for(size_t i=0;i<input.size();++i) {
                const auto value=engine->tick(input[i]);
                if(i+1==ob::studioBlock)firstBlockMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                if(!std::isfinite(value))throw std::runtime_error("Nonfinite Studio Voice output");
                if(pass==0)first.push_back(value);
                else if(std::abs(value-first[i])>1e-5f)throw std::runtime_error("Studio Voice reset did not clear caches");
                maxError=std::max(maxError,std::abs(value-reference[i]));energy+=value*value;
            }
            std::cout<<"Studio Voice "<<(device<0 ? "CPU" : ob::gpuBackendName())<<" pass="<<pass<<" max_error="<<maxError<<" energy="<<energy<<" ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()<<" first_block_ms="<<firstBlockMs<<std::endl;
            if(maxError>.001f || energy<1e-10)throw std::runtime_error("Studio Voice reference mismatch");
        }
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
