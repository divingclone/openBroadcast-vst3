#include "Processor.h"
#include <fstream>
#include <iostream>
#include <chrono>
#include <thread>
#include <stdexcept>
// Diagnostic renderer: the caller supplies local speech, never bundled demo assets.
int main(int argc,char** argv) {
    juce::ScopedJuceInitialiser_GUI gui;
    try {
        if(argc!=7 && argc!=8)throw std::runtime_error("Expected assets input.f32 output.f32 sample-rate realtime backend [input-channels]");
        const int inputChannels=argc==8 ? std::stoi(argv[7]) : 1;
        if(inputChannels!=1 && inputChannels!=2)throw std::runtime_error("Expected mono or interleaved stereo input");
#ifdef _WIN32
        _wputenv_s(L"OPENBROADCAST_VST_ASSETS",std::filesystem::path(argv[1]).c_str());
#else
        setenv("OPENBROADCAST_VST_ASSETS",argv[1],1);
#endif
        std::ifstream file(argv[2],std::ios::binary|std::ios::ate);
        if(!file)throw std::runtime_error("Input missing");
        std::vector<float> input(static_cast<size_t>(file.tellg())/sizeof(float));
        file.seekg(0);file.read(reinterpret_cast<char*>(input.data()),input.size()*sizeof(float));
        const size_t frames=input.size()/inputChannels;
        const double rate=std::stod(argv[4]); const bool realtime=std::stoi(argv[5])!=0;
        BroadcastProcessor p;
        auto set=[&](const char* id,float v) { auto* param=p.parameters.getParameter(id);param->setValueNotifyingHost(param->convertTo0to1(v)); };
        set("studioEnabled",1);set("backend",static_cast<float>(std::stoi(argv[6])));set("mix",1);
        p.setRateAndBufferSizeDetails(rate,512);p.setNonRealtime(!realtime);p.prepareToPlay(rate,512);
        if(p.status().startsWith("Dry fallback"))throw std::runtime_error(p.status().toStdString());
        const int latency=p.getLatencySamples();
        const size_t total=frames+latency;
        std::vector<float> output;output.reserve(total*inputChannels);
        auto next=std::chrono::steady_clock::now(); double maxCallback=0,maxLoad=0,stereoDifference=0;
        for(size_t pos=0;pos<total;pos+=512) {
            const int count=static_cast<int>(std::min(size_t(512),total-pos));
            juce::AudioBuffer<float> audio(2,count);juce::MidiBuffer midi;
            for(int i=0;i<count;++i)for(int c=0;c<2;++c)audio.setSample(c,i,pos+i<frames ? input[(pos+i)*inputChannels+(inputChannels==1 ? 0 : c)] : 0.f);
            auto start=std::chrono::steady_clock::now();p.processBlock(audio,midi);
            maxCallback=std::max(maxCallback,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
            maxLoad=std::max(maxLoad,p.processingLoad());
            for(int i=0;i<count;++i) {
                if(!std::isfinite(audio.getSample(0,i)))throw std::runtime_error("Nonfinite output");
                stereoDifference=std::max(stereoDifference,double(std::abs(audio.getSample(0,i)-audio.getSample(1,i))));
                output.push_back(audio.getSample(0,i));
                if(inputChannels==2)output.push_back(audio.getSample(1,i));
            }
            if(realtime) { next+=std::chrono::nanoseconds(static_cast<int64_t>(1e9*count/rate));std::this_thread::sleep_until(next); }
        }
        std::cout<<p.status()<<"\nmax_load="<<maxLoad<<" max_callback_ms="<<maxCallback<<" stereo_difference="<<stereoDifference<<std::endl;
        if(p.status().startsWith("Dry fallback"))throw std::runtime_error(p.status().toStdString());
        std::ofstream result(argv[3],std::ios::binary);
        result.write(reinterpret_cast<const char*>(output.data()+latency*inputChannels),frames*inputChannels*sizeof(float));
        if(!result)throw std::runtime_error("Cannot save output");
        p.releaseResources();return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<std::endl;return 1; }
}
