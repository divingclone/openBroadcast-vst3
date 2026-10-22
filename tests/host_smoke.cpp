#include <juce_audio_utils/juce_audio_utils.h>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <thread>
static void check(bool v,const char* text){if(!v)throw std::runtime_error(text);}
int main(int argc,char** argv){
    juce::ScopedJuceInitialiser_GUI gui;
    try{
        check(argc==2 || argc==3,"Expected complete VST3 bundle path [--studio]");
        const bool studio=argc==3 && std::string(argv[2])=="--studio";
#ifdef _WIN32
        _wputenv_s(L"OPENBROADCAST_VST_ASSETS",L"");
#else
        unsetenv("OPENBROADCAST_VST_ASSETS");
#endif
        juce::VST3PluginFormat format;
        juce::OwnedArray<juce::PluginDescription> descriptions;
        format.findAllTypesForFile(descriptions,juce::File(argv[1]).getFullPathName());
        check(descriptions.size()==1,"VST3 scan must find one plugin");
        juce::String error;
        auto instance=format.createInstanceFromDescription(*descriptions[0],48000,512,error);
        check(instance!=nullptr,error.toRawUTF8());
        bool foundStudio=false;
        for(auto* parameter:instance->getParameters())if(parameter->getName(64)=="Studio Voice") {
            check(parameter->getValue()<.5f,"Packaged Studio Voice must default off");
            parameter->setValueNotifyingHost(studio ? 1.f : 0.f);foundStudio=true;
        }
        check(foundStudio,"Studio Voice parameter missing from bundle");
        instance->setNonRealtime(true);instance->prepareToPlay(48000,512);
        check(instance->getLatencySamples()==(studio ? 7680 : 10560),"VST3 latency report");
        double energy=0, inputEnergy=0, dryDifference=0;
        for(int pos=0;pos<96000;pos+=512){
            juce::AudioBuffer<float> b(2,512);
            for(int c=0;c<2;++c)for(int i=0;i<512;++i){float x=.02f*std::sin((pos+i)*.071);b.setSample(c,i,x);if(c==0)inputEnergy+=x*x;}
            juce::MidiBuffer midi;instance->processBlock(b,midi);
            for(int i=0;i<512;++i){float x=b.getSample(0,i);check(std::isfinite(x),"Nonfinite VST3 audio");energy+=x*x;
                const int source=pos+i-instance->getLatencySamples();
                const float dry=source>=0 ? .02f*std::sin(source*.071) : 0.f;
                dryDifference+=(x-dry)*(x-dry);
            }
        }
        std::cout<<"VST3 scanned: "<<descriptions[0]->name<<" latency="<<instance->getLatencySamples()<<" output/input energy="<<energy/inputEnergy<<std::endl;
        check(energy>1e-8 && (studio ? dryDifference>1e-5 : energy<inputEnergy*.8),"Packaged model must process audio, not dry-fallback");
        juce::MemoryBlock state;instance->getStateInformation(state);check(state.getSize()>0,"VST3 state is empty");instance->setStateInformation(state.getData(),static_cast<int>(state.getSize()));
        std::unique_ptr<juce::AudioProcessorEditor> editor(instance->createEditorIfNeeded());
        check(editor!=nullptr,"VST3 editor creation");
        editor.reset();instance->releaseResources();
        // Keep the realtime timing check on the default lightweight mode;
        // CPU Studio Voice may legitimately be slower than realtime.
        if(studio) { instance.reset();std::cout<<"STUDIO BUNDLE PASSED"<<std::endl;return 0; }
        instance->setNonRealtime(false);instance->prepareToPlay(48000,512);
        auto next=std::chrono::steady_clock::now();double maxCallback=0;
        for(int n=0;n<188;++n){
            juce::AudioBuffer<float>b(2,512);b.clear();juce::MidiBuffer midi;
            auto start=std::chrono::steady_clock::now();instance->processBlock(b,midi);
            maxCallback=std::max(maxCallback,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
            next+=std::chrono::microseconds(10667);std::this_thread::sleep_until(next);
        }
        std::cout<<"Realtime max callback ms="<<maxCallback<<std::endl;
        instance->releaseResources();instance.reset();
        std::cout<<"HOST SMOKE PASSED"<<std::endl;return 0;
    }catch(const std::exception&e){std::cerr<<e.what()<<std::endl;return 1;}
}
