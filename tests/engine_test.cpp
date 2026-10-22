#include "Engine.h"
#include <fstream>
#include <iostream>
#include <chrono>
#include <stdexcept>
static std::vector<float> read(const std::filesystem::path& file) {
    std::ifstream f(file,std::ios::binary|std::ios::ate);
    if(!f)throw std::runtime_error("Missing reference "+file.string());
    std::vector<float> data(static_cast<size_t>(f.tellg())/4);f.seekg(0);
    f.read(reinterpret_cast<char*>(data.data()),data.size()*4);return data;
}
int main(int argc,char** argv) {
    try {
        if(argc<4)throw std::runtime_error("Expected assets runtime references [GPU index]");
        auto rt=std::make_shared<ob::Runtime>(argv[2]);
        auto dsp=std::make_shared<ob::Coefficients>(std::filesystem::path(argv[1])/"dsp.bin");
        auto input=read(std::filesystem::path(argv[3])/"input.f32");
        const int device=argc>4 ? std::stoi(argv[4]) : -1;
        for(auto name:{"denoiser","dereverb"}) {
            ob::Stage stage(rt,dsp,std::filesystem::path(argv[1])/"models"/(std::string(name)+"_48k.onnx"),device);
            for(auto intensity:{0.f,.37f,1.f}) {
                stage.reset(); char suffix[32]; snprintf(suffix,sizeof(suffix),"_%.2f.f32",intensity);
                auto reference=read(std::filesystem::path(argv[3])/(std::string(name)+suffix));
                if(reference.size()!=input.size())throw std::runtime_error("Reference size mismatch");
                float maximum=0;double squared=0;
                auto start=std::chrono::steady_clock::now();
                for(size_t i=0;i<input.size();++i) {
                    float value=stage.tick(input[i],intensity), diff=std::abs(value-reference[i]);
                    if(!std::isfinite(value))throw std::runtime_error("Nonfinite output");
                    maximum=std::max(maximum,diff);squared+=diff*diff;
                }
                auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                std::cout<<name<<" intensity="<<intensity<<" max_error="<<maximum<<" rms_error="<<std::sqrt(squared/input.size())<<" processing_ms="<<ms<<std::endl;
                if(maximum>(device<0 ? 0.0003f : 0.003f))throw std::runtime_error("Python parity mismatch");
            }
        }
        for(double rate:{8000.,44100.,48000.,96000.,192000.}) {
            ob::Channel channel(rt,dsp,argv[1],rate,-1);
            const int at=static_cast<int>(rate*.1), n=static_cast<int>(rate*.4);
            float peak=0; int peakIndex=0;
            for(int i=0;i<n;++i) {
                float value=channel.tick(i==at ? .1f : 0.f,0,0);
                if(!std::isfinite(value))throw std::runtime_error("Resampling nonfinite");
                if(std::abs(value)>peak){peak=std::abs(value);peakIndex=i;}
            }
            const int expected=at+ob::Channel::delay(rate);
            std::cout<<"rate="<<rate<<" impulse_peak="<<peakIndex<<" expected="<<expected<<" amplitude="<<peak<<std::endl;
            if(std::abs(peakIndex-expected)>1 || peak<.005f)throw std::runtime_error("Latency/resampler mismatch");
        }
        std::cout<<"ENGINE TESTS PASSED"<<std::endl;return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
