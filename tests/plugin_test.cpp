#include "Processor.h"
#include <iostream>
#include <random>
#include <stdexcept>
static void require(bool v,const char* message){if(!v)throw std::runtime_error(message);}
static void parameter(BroadcastProcessor& p,const char* name,float value){auto* x=p.parameters.getParameter(name);x->setValueNotifyingHost(x->convertTo0to1(value));}
static void waitForConfiguration(BroadcastProcessor& p) {
    // Cold CoreML compilation of Studio Voice can exceed 45 seconds on hosted
    // macOS runners. This bounds loading, not UI responsiveness (checked below).
#ifdef __APPLE__
    constexpr int loadingSeconds=180;
#else
    constexpr int loadingSeconds=45;
#endif
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(loadingSeconds);
    std::cout<<"Waiting for configuration: backend="<<p.parameters.getRawParameterValue("backend")->load()
             <<" studio="<<p.parameters.getRawParameterValue("studioEnabled")->load()<<std::endl;
    while(std::chrono::steady_clock::now()<deadline) {
        const auto beforeDispatch=std::chrono::steady_clock::now();
        juce::Timer::callPendingTimersSynchronously();
        require(std::chrono::steady_clock::now()-beforeDispatch<std::chrono::milliseconds(250),"Model loading blocked the message thread");
        const auto status=p.status();
        if(!status.contains("Loading models") && !status.contains("Updating settings")) {
            require(!status.startsWith("Dry fallback"),status.toRawUTF8());
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    throw std::runtime_error("Automatic configuration timed out: "+p.status().toStdString());
}
static std::vector<float> render(BroadcastProcessor& p) {
    p.setRateAndBufferSizeDetails(48000,512);p.setNonRealtime(true);p.prepareToPlay(48000,512);
    require(!p.status().startsWith("Dry fallback"),p.status().toRawUTF8());
    std::vector<float> result;
    std::mt19937 random(1234);std::uniform_real_distribution<float> noise(-.02f,.02f);
    for(int pos=0;pos<48000;pos+=480) {
        juce::AudioBuffer<float>b(2,480);b.clear();
        for(int i=0;i<480;++i)b.setSample(0,i,.03f*std::sin((pos+i)*.037f)+noise(random));
        juce::MidiBuffer midi;p.processBlock(b,midi);
        result.insert(result.end(),b.getReadPointer(0),b.getReadPointer(0)+480);
    }
    require(p.underruns.load()==0,"Switch render missed packet");
    p.releaseResources();return result;
}
int main(int argc,char** argv){
    juce::ScopedJuceInitialiser_GUI gui;
    try{
        require(argc==2 || (argc==3 && (std::string(argv[2])=="--device-only" || std::string(argv[2])=="--studio-device")),"Expected asset path [--device-only|--studio-device]");
#ifdef _WIN32
        _wputenv_s(L"OPENBROADCAST_VST_ASSETS",std::filesystem::path(argv[1]).c_str());
#else
        setenv("OPENBROADCAST_VST_ASSETS",argv[1],1);
#endif
        BroadcastProcessor p;
        require(p.parameters.getRawParameterValue("backend")->load()==2,"Fresh instances must use Auto");
        require(p.parameters.getRawParameterValue("denoiseEnabled")->load()==1,"Denoise default");
        require(p.parameters.getRawParameterValue("dereverbEnabled")->load()==0,"Dereverb default");
        require(p.parameters.getRawParameterValue("studioEnabled")->load()==0,"Studio Voice must default off");
        if(argc==3 && std::string(argv[2])=="--studio-device")parameter(p,"studioEnabled",1);
        require(p.parameters.getRawParameterValue("reverb")->load()==1,"Dereverb must have a usable amount when enabled");
        auto ranked=ob::rankGpuDevices({{0,L"Integrated",512ull<<20,1},{3,L"Discrete",8ull<<30,0}});
        require(ranked.front().index==3,"GPU preference must preserve native ordinal");
        ranked=ob::rankGpuDevices({{0,L"Integrated",512ull<<20,-1},{3,L"Discrete",8ull<<30,-1}});
        require(ranked.front().index==3,"VRAM fallback ranking");
        require(ob::rankGpuDevices({}).empty(),"No GPU ranking");
        ranked=ob::rankGpuDevices({{0,L"Large slow GPU",24ull<<30,-1,10},{3,L"Fast GPU",8ull<<30,-1,100}});
        require(ranked.front().index==3,"CUDA compute ranking must preserve native ordinal");
        p.setRateAndBufferSizeDetails(48000,512);p.prepareToPlay(48000,512);
        require(!p.status().startsWith("Dry fallback"),p.status().toRawUTF8());
        std::cout<<"Auto device: "<<p.status()<<std::endl;
        const bool namedGpu=std::any_of(p.devices.begin(),p.devices.end(),[&](const auto& gpu){return p.status().contains(juce::String(gpu.name.c_str()));});
        require(namedGpu || p.status().contains("using CPU"),"Auto must identify the active GPU or explain CPU fallback");
        const auto expectedBackend=juce::SystemStats::getEnvironmentVariable("OPENBROADCAST_EXPECT_BACKEND",{});
        if(expectedBackend=="CPU")require(p.status().startsWith("CPU") && p.status().contains("using CPU"),"Expected explicit CPU fallback");
        else if(expectedBackend.isNotEmpty())require(namedGpu && p.status().contains(expectedBackend) && !p.status().startsWith("CPU"),"Expected automatic GPU backend");
        p.releaseResources();
        if(argc==3) { std::cout<<"DEVICE TESTS PASSED"<<std::endl;return 0; }
        parameter(p,"backend",0);
        parameter(p,"noise",.37f);parameter(p,"reverb",.23f);
        juce::MemoryBlock saved;p.getStateInformation(saved);
        parameter(p,"noise",0);p.setStateInformation(saved.getData(),static_cast<int>(saved.getSize()));
        require(std::abs(p.parameters.getRawParameterValue("noise")->load()-.37f)<1e-5,"State roundtrip");
        // Simulate a 0.1.0 project: retain amounts and CPU choice; permit its
        // existing amount automation to enable either effect without new switches.
        auto legacy=p.parameters.copyState();
        for(auto id:{"denoiseEnabled","dereverbEnabled","studioEnabled"})legacy.removeChild(legacy.getChildWithProperty("id",id),nullptr);
        parameter(p,"studioEnabled",1);
        juce::MemoryBlock legacyData;juce::AudioProcessor::copyXmlToBinary(*legacy.createXml(),legacyData);
        p.setStateInformation(legacyData.getData(),static_cast<int>(legacyData.getSize()));
        require(p.parameters.getRawParameterValue("dereverbEnabled")->load()==1,"Legacy dereverb automation disabled");
        require(p.parameters.getRawParameterValue("backend")->load()==0,"Saved CPU choice overwritten");
        require(p.parameters.getRawParameterValue("studioEnabled")->load()==0,"Legacy projects must keep Studio Voice off");
        require(std::abs(p.parameters.getRawParameterValue("reverb")->load()-.23f)<1e-5,"Legacy amount changed");
        for(auto stage:{0,1}) {
            const char* amount=stage==0 ? "noise" : "reverb";
            const char* enabled=stage==0 ? "denoiseEnabled" : "dereverbEnabled";
            parameter(p,"denoiseEnabled",0);parameter(p,"dereverbEnabled",0);
            parameter(p,amount,.75f);
            auto off=render(p);
            require(std::abs(p.parameters.getRawParameterValue(amount)->load()-.75f)<1e-5f,"Switch off erased amount");
            parameter(p,enabled,1);auto on=render(p);
            parameter(p,amount,0);auto zero=render(p);
            float offError=0,onDifference=0;
            for(size_t i=0;i<off.size();++i){offError=std::max(offError,std::abs(off[i]-zero[i]));onDifference=std::max(onDifference,std::abs(off[i]-on[i]));}
            require(offError<1e-6f,"Effect off does not equal zero strength");
            require(onDifference>1e-5f,"Effect on did not process audio");
            parameter(p,amount,.75f);parameter(p,enabled,0);
            juce::MemoryBlock switchState;p.getStateInformation(switchState);
            parameter(p,enabled,1);parameter(p,amount,.2f);
            p.setStateInformation(switchState.getData(),static_cast<int>(switchState.getSize()));
            require(p.parameters.getRawParameterValue(enabled)->load()==0,"Switch state was not restored");
            require(std::abs(p.parameters.getRawParameterValue(amount)->load()-.75f)<1e-5f,"Switch amount was not restored");
            std::cout<<amount<<" switch off_error="<<offError<<" on_difference="<<onDifference<<std::endl;
        }
        parameter(p,"denoiseEnabled",1);parameter(p,"dereverbEnabled",1);
        for(auto rate:{44100.,48000.,96000.})for(auto channels:{1,2}){
            auto layout=p.getBusesLayout();layout.inputBuses.getReference(0)=juce::AudioChannelSet::canonicalChannelSet(channels);layout.outputBuses=layout.inputBuses;
            require(p.setBusesLayout(layout),"Mono/stereo layout rejected");
            p.setRateAndBufferSizeDetails(rate,1024);p.setNonRealtime(true);
            parameter(p,"noise",0);parameter(p,"reverb",0);parameter(p,"mix",1);
            p.prepareToPlay(rate,1024);
            require(!p.status().startsWith("Dry fallback"),p.status().toRawUTF8());
            int total=static_cast<int>(rate*.7), delay=p.getLatencySamples();
            std::vector<float> original(total),output(total);
            for(int i=0;i<total;++i)original[i]=.02f*std::sin(2*3.141592653589793*300*i/rate);
            int position=0, iteration=0;
            const int sizes[]{1,7,64,127,480,512,1024,3};
            while(position<total){
                int size=std::min(sizes[iteration++%8],total-position);
                juce::AudioBuffer<float> buffer(channels,size);buffer.clear();
                for(int i=0;i<size;++i)buffer.setSample(0,i,original[position+i]);
                juce::MidiBuffer midi;p.processBlock(buffer,midi);
                for(int i=0;i<size;++i){output[position+i]=buffer.getSample(0,i);if(channels==2)require(std::abs(buffer.getSample(1,i))<1e-7,"Stereo crosstalk");}
                position+=size;
            }
            require(p.underruns.load()==0,"Offline missed packet");
            double error=0,energy=0;
            for(int i=delay+500;i<total;++i){double d=output[i]-original[i-delay];error+=d*d;energy+=original[i-delay]*original[i-delay];}
            const double relative=std::sqrt(error/energy);
            std::cout<<"rate="<<rate<<" channels="<<channels<<" latency="<<delay<<" relative_error="<<relative<<std::endl;
            require(relative<.0002,"Wet identity/delay mismatch");
            p.releaseResources();
        }
        p.setRateAndBufferSizeDetails(48000,512);p.prepareToPlay(48000,512);
        parameter(p,"bypass",1);
        int n=p.getLatencySamples()+1000;std::vector<float> result;
        for(int pos=0;pos<n;pos+=127){int count=std::min(127,n-pos);juce::AudioBuffer<float> b(2,count);b.clear();if(pos==0)b.setSample(0,0,.5f);juce::MidiBuffer m;p.processBlockBypassed(b,m);for(int i=0;i<count;++i)result.push_back(b.getSample(0,i));}
        require(std::abs(result[p.getLatencySamples()]-.5f)<1e-6,"Bypass latency mismatch");
        // A host transport reset must discard queued audio and recurrent state.
        parameter(p,"bypass",0);p.reset();
        for(int block=0;block<100;++block){
            juce::AudioBuffer<float>b(2,127);b.clear();juce::MidiBuffer m;p.processBlock(b,m);
            require(b.getMagnitude(0,127)<1e-7f,"Transport reset leaked previous audio");
        }
        p.releaseResources();
        // Studio Voice replaces the two light stages and changes reported PDC.
        parameter(p,"backend",2);parameter(p,"studioEnabled",1);
        auto studioAudio=render(p);
        require(p.getLatencySamples()==7680,"Studio Voice latency report");
        parameter(p,"noise",.1f);parameter(p,"reverb",.8f);
        auto studioAgain=render(p);
        float ignoredDifference=0;double studioEnergy=0;
        for(size_t i=0;i<studioAudio.size();++i) {
            ignoredDifference=std::max(ignoredDifference,std::abs(studioAudio[i]-studioAgain[i]));
            studioEnergy+=studioAudio[i]*studioAudio[i];
        }
        require(ignoredDifference<1e-5f,"Studio Voice did not replace Denoise / Dereverb");
        require(studioEnergy>1e-8,"Studio Voice plugin output was silent");
        juce::MemoryBlock studioState;p.getStateInformation(studioState);
        parameter(p,"studioEnabled",0);p.setStateInformation(studioState.getData(),static_cast<int>(studioState.getSize()));
        require(p.parameters.getRawParameterValue("studioEnabled")->load()==1,"Studio Voice state not restored");
        p.setRateAndBufferSizeDetails(44100,512);p.prepareToPlay(44100,512);
        require(!p.status().startsWith("Dry fallback"),p.status().toRawUTF8());
        require(p.getLatencySamples()==4*882+ob::Channel::delay(44100,true),"Resampled Studio Voice PDC");
        parameter(p,"bypass",1);
        const auto studioDelay=p.getLatencySamples();std::vector<float> bypassOutput;
        for(int pos=0;pos<studioDelay+1024;pos+=127) {
            juce::AudioBuffer<float>b(2,127);b.clear();if(pos==0)b.setSample(0,0,.5f);
            juce::MidiBuffer m;p.processBlockBypassed(b,m);
            bypassOutput.insert(bypassOutput.end(),b.getReadPointer(0),b.getReadPointer(0)+127);
        }
        require(std::abs(bypassOutput[studioDelay]-.5f)<1e-6f,"Studio Voice bypass alignment");
        parameter(p,"bypass",0);parameter(p,"studioEnabled",0);
        std::atomic<bool> probeRunning{true}, sawSuspension{false};double maxLockWait=0;
        std::thread callbackProbe([&] {
            while(probeRunning.load()) {
                const auto start=std::chrono::steady_clock::now();
                {
                    const juce::ScopedLock lock(p.getCallbackLock());
                    if(p.isSuspended())sawSuspension=true;
                }
                maxLockWait=std::max(maxLockWait,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
        try { waitForConfiguration(p); }
        catch(...) { probeRunning=false;callbackProbe.join();throw; }
        probeRunning=false;callbackProbe.join();
        require(sawSuspension.load(),"Automatic switching did not suspend callbacks while loading");
        require(maxLockWait<100.,"Automatic switching held the audio callback lock during model loading");
        require(!p.isSuspended(),"Automatic switching left audio suspended");
        require(p.getLatencySamples()==4*882+ob::Channel::delay(44100,false),"Disabling Studio Voice did not restore PDC");
        require(!p.status().contains("Studio Voice"),"Studio Voice still active after automatic switch");
        // No editor is open: device changes and restored state must still apply.
        parameter(p,"backend",0);waitForConfiguration(p);
        require(p.status().startsWith("CPU"),"CPU selection did not apply automatically");
        p.setStateInformation(studioState.getData(),static_cast<int>(studioState.getSize()));
        const auto loadDeadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
        while(!p.status().contains("Loading models") && std::chrono::steady_clock::now()<loadDeadline) {
            juce::Timer::callPendingTimersSynchronously();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        require(p.status().contains("Loading models"),"Restored state did not start loading automatically");
        // Parameter edits must remain responsive while a model is loading.
        // The newest settings win even when changed several times mid-load.
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        parameter(p,"studioEnabled",0);parameter(p,"backend",0);
        parameter(p,"studioEnabled",1);parameter(p,"backend",2);
        parameter(p,"studioEnabled",0);parameter(p,"backend",0);
        waitForConfiguration(p);
        require(p.status().startsWith("CPU") && !p.status().contains("Studio Voice"),"Newest settings were lost during loading");
        p.setStateInformation(studioState.getData(),static_cast<int>(studioState.getSize()));
        waitForConfiguration(p);
        require(p.status().contains("Studio Voice"),"Latest Studio Voice setting not applied");
        require(p.getLatencySamples()==4*882+ob::Channel::delay(44100,true),"Automatic Studio Voice PDC");
        parameter(p,"studioEnabled",0);parameter(p,"backend",0);waitForConfiguration(p);
        require(p.status().startsWith("CPU") && !p.status().contains("Studio Voice"),"Combined mode/device switch failed");
        std::cout<<"Studio Voice plugin energy="<<studioEnergy<<" inactive_controls_error="<<ignoredDifference<<std::endl;
        p.releaseResources();
        parameter(p,"backend",2);parameter(p,"noise",.75f);parameter(p,"reverb",.5f);
        parameter(p,"denoiseEnabled",1);parameter(p,"dereverbEnabled",1);
        p.setRateAndBufferSizeDetails(48000,512);
        p.prepareToPlay(48000,512);
        std::unique_ptr<juce::AudioProcessorEditor> editor(p.createEditor());require(editor!=nullptr,"Editor creation failed");
        auto snapshot=editor->createComponentSnapshot(editor->getLocalBounds());
        for(auto* child:editor->getChildren())if(auto* button=dynamic_cast<juce::TextButton*>(child))
            require(button->getButtonText()!="Apply","Apply button still visible");
        juce::File image=juce::File::getCurrentWorkingDirectory().getChildFile("editor-v0.4.0.png");
        auto stream=image.createOutputStream();require(stream!=nullptr,"Cannot write editor preview");
        stream->setPosition(0);stream->truncate();
        juce::PNGImageFormat().writeImageToStream(snapshot,*stream);
        p.releaseResources();
        std::cout<<"PLUGIN TESTS PASSED"<<std::endl;return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
