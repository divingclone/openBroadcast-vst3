#include "Processor.h"
#include <chrono>

namespace {
constexpr int bufferingPackets = 4;
std::filesystem::path nativePath(const juce::String& value) {
#ifdef _WIN32
    return std::filesystem::path(value.toWideCharPointer());
#else
    return std::filesystem::u8path(value.toStdString());
#endif
}
class Editor final : public juce::AudioProcessorEditor, private juce::Timer {
public:
    explicit Editor(BroadcastProcessor& p) : AudioProcessorEditor(p), processor(p) {
        for (auto* c : {&noise, &reverb, &mix}) addAndMakeVisible(c);
        for (auto* slider : {&noise, &reverb, &mix}) {
            slider->setSliderStyle(juce::Slider::LinearHorizontal);
            slider->setTextBoxStyle(juce::Slider::TextBoxRight, false, 65, 24);
            slider->setColour(juce::Slider::trackColourId, juce::Colour(0xff79b84a));
        }
        noise.setName("Denoise amount"); reverb.setName("Dereverb amount"); mix.setName("Mix");
        noiseLink = std::make_unique<Attachment>(p.parameters,"noise",noise);
        reverbLink = std::make_unique<Attachment>(p.parameters,"reverb",reverb);
        mixLink = std::make_unique<Attachment>(p.parameters,"mix",mix);
        bypass.setButtonText("Bypass"); addAndMakeVisible(bypass);
        bypassLink = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(p.parameters,"bypass",bypass);
        denoiseOn.setButtonText("Denoise"); dereverbOn.setButtonText("Dereverb");
        addAndMakeVisible(denoiseOn); addAndMakeVisible(dereverbOn);
        denoiseOn.setTooltip("Enable denoising. Your amount is retained when switched off.");
        dereverbOn.setTooltip("Enable room reverb reduction. Your amount is retained when switched off.");
        denoiseLink = std::make_unique<ButtonAttachment>(p.parameters,"denoiseEnabled",denoiseOn);
        dereverbLink = std::make_unique<ButtonAttachment>(p.parameters,"dereverbEnabled",dereverbOn);
        studioOn.setButtonText("Studio Voice"); addAndMakeVisible(studioOn);
        studioOn.setTooltip("Uses AI to enhance your voice for studio-quality sound. High CPU/GPU and memory usage. Replaces Denoise / Dereverb. Changes apply automatically; audio pauses during loading.");
        studioLink = std::make_unique<ButtonAttachment>(p.parameters,"studioEnabled",studioOn);
        device.setName("Processing device");
#ifdef __APPLE__
        device.addItem("Auto (CoreML CPU/GPU)",1);
#else
        device.addItem("Auto (High Performance GPU)",1);
#endif
        device.addItem("CPU",2);
        device.addSeparator();
        for(const auto& gpu:p.devices)
            device.addItem(juce::String(gpu.name.c_str())+" - "+ob::gpuBackendName(gpu.backend)+"  ["+juce::String(gpu.index)+"]",100+gpu.selectionId());
        addAndMakeVisible(device);
#ifdef __APPLE__
        device.setTooltip("CoreML uses system-managed CPU/GPU scheduling; individual GPUs cannot be selected. Changes apply automatically.");
#else
        device.setTooltip("Auto selects a high-performance GPU. Changes take effect automatically; audio pauses during loading.");
#endif
        device.onChange=[this] {
            const auto id=device.getSelectedId();
            auto set=[this](const char* name,float value) {
                auto* parameter=processor.parameters.getParameter(name);
                parameter->beginChangeGesture();
                parameter->setValueNotifyingHost(parameter->convertTo0to1(value));
                parameter->endChangeGesture();
            };
            if(id>=100) { set("device",static_cast<float>(id-100)); set("backend",1); }
            else if(id==1 || id==2) set("backend",id==1 ? 2.f : 0.f);
        };
        message.setColour(juce::Label::textColourId,juce::Colour(0xffa7b1bf));
        message.setJustificationType(juce::Justification::topLeft); addAndMakeVisible(message);
        setSize(620, 534); timerCallback(); startTimerHz(10);
    }
    void paint(juce::Graphics& g) override {
        g.fillAll(juce::Colour(0xff121821)); g.setColour(juce::Colours::white);
        g.setFont(27.f); g.drawText("OpenBroadcast",24,18,400,42,juce::Justification::left);
        g.setFont(14.f); g.setColour(juce::Colour(0xffa7b1bf));
        g.drawText("Denoise / Dereverb / Studio Voice",24,60,440,24,juce::Justification::left);
        g.setColour(juce::Colour(0xffd8b77a));
        g.drawText("High resource usage - GPU recommended",190,220,406,24,juce::Justification::left);
        g.setColour(juce::Colour(0xffa7b1bf));
        g.setFont(14.f);
        g.drawText("Uses AI to enhance your voice for studio-quality sound.",32,251,560,20,juce::Justification::left);
        g.setFont(12.f);
        g.drawText("Replaces Denoise / Dereverb. Changes take effect automatically.",32,275,560,20,juce::Justification::left);
        g.setFont(14.f);
        g.drawText("Mix",32,314,140,24,juce::Justification::left);
        g.drawText("Device",32,377,140,24,juce::Justification::left);
        g.setColour(juce::Colour(0xff29343d));
        g.drawHorizontalLine(358,24.f,596.f);
    }
    void resized() override {
        denoiseOn.setBounds(24,110,150,32); dereverbOn.setBounds(24,165,150,32);
        noise.setBounds(190,110,406,32); reverb.setBounds(190,165,406,32); mix.setBounds(190,312,406,32);
        studioOn.setBounds(24,218,160,32);
        device.setBounds(110,374,486,30); bypass.setBounds(508,30,88,30);
        message.setBounds(24,427,572,90);
    }
private:
    void timerCallback() override {
        const auto mode=static_cast<int>(processor.parameters.getRawParameterValue("backend")->load());
        const auto index=static_cast<int>(processor.parameters.getRawParameterValue("device")->load());
        const auto id=mode==2 ? 1 : mode==0 ? 2 : 100+index;
        if(id>=100 && device.indexOfItemId(id)<0)device.addItem("Unavailable GPU ["+juce::String(index)+"]",id);
        device.setSelectedId(id,juce::dontSendNotification);
        const bool studio=processor.parameters.getRawParameterValue("studioEnabled")->load()>.5f;
        denoiseOn.setEnabled(!studio); dereverbOn.setEnabled(!studio);
        noise.setEnabled(!studio && processor.parameters.getRawParameterValue("denoiseEnabled")->load()>.5f);
        reverb.setEnabled(!studio && processor.parameters.getRawParameterValue("dereverbEnabled")->load()>.5f);
        auto status=processor.status();
        message.setText(status,juce::dontSendNotification); message.setTooltip(status);
    }
    using Attachment = juce::AudioProcessorValueTreeState::SliderAttachment;
    using ButtonAttachment = juce::AudioProcessorValueTreeState::ButtonAttachment;
    BroadcastProcessor& processor;
    juce::Slider noise,reverb,mix;
    juce::ToggleButton bypass,denoiseOn,dereverbOn,studioOn;
    juce::ComboBox device;
    juce::Label message;
    std::unique_ptr<Attachment> noiseLink,reverbLink,mixLink;
    std::unique_ptr<ButtonAttachment> bypassLink,denoiseLink,dereverbLink,studioLink;
};
}
juce::AudioProcessorValueTreeState::ParameterLayout BroadcastProcessor::layout() {
    juce::AudioProcessorValueTreeState::ParameterLayout result;
    const auto percent=juce::AudioParameterFloatAttributes()
        .withStringFromValueFunction([](float value,int) { return juce::String(value*100.f,1)+" %"; })
        .withValueFromStringFunction([](const juce::String& text) { return text.getFloatValue()/100.f; });
    for (const auto& item : std::array<std::pair<const char*,const char*>,3>{{{"noise","Denoise"},{"reverb","Dereverb"},{"mix","Mix"}}})
        result.add(std::make_unique<juce::AudioParameterFloat>(juce::ParameterID{item.first,1},item.second,juce::NormalisableRange<float>(0,1,.001f),1.f,percent));
    result.add(std::make_unique<juce::AudioParameterBool>(juce::ParameterID{"bypass",1},"Bypass",false));
#ifdef __linux__
    const juce::String gpuChoice="GPU (CUDA / MIGraphX)";
#else
    const juce::String gpuChoice=ob::gpuBackendName();
#endif
    result.add(std::make_unique<juce::AudioParameterChoice>(juce::ParameterID{"backend",1},"Engine",juce::StringArray{"CPU",gpuChoice,"Auto"},2,juce::AudioParameterChoiceAttributes().withAutomatable(false)));
    result.add(std::make_unique<juce::AudioParameterInt>(juce::ParameterID{"device",1},"GPU",0,31,0,juce::AudioParameterIntAttributes().withAutomatable(false)));
    // Append new parameters so existing IDs and parameter ordering remain stable.
    result.add(std::make_unique<juce::AudioParameterBool>(juce::ParameterID{"denoiseEnabled",1},"Denoise On",true));
    result.add(std::make_unique<juce::AudioParameterBool>(juce::ParameterID{"dereverbEnabled",1},"Dereverb On",false));
    result.add(std::make_unique<juce::AudioParameterBool>(juce::ParameterID{"studioEnabled",1},"Studio Voice",false,juce::AudioParameterBoolAttributes().withAutomatable(false)));
    return result;
}
BroadcastProcessor::BroadcastProcessor() : AudioProcessor(BusesProperties().withInput("Input",juce::AudioChannelSet::stereo(),true).withOutput("Output",juce::AudioChannelSet::stereo(),true)), devices(ob::discoverProcessingDevices(resourcePath())), parameters(*this,nullptr,"OpenBroadcastState",layout()) { startTimer(50); }
BroadcastProcessor::~BroadcastProcessor() {
    stopTimer();
    if(configurationWorker.joinable())configurationWorker.join();
    releaseResources();
}
bool BroadcastProcessor::isBusesLayoutSupported(const BusesLayout& l) const {
    auto in=l.getMainInputChannelSet(); return in == l.getMainOutputChannelSet() && (in == juce::AudioChannelSet::mono() || in == juce::AudioChannelSet::stereo());
}
std::filesystem::path BroadcastProcessor::resourcePath() const {
    // Override is useful for development/tests; distributed bundles are self-contained.
    auto override = juce::SystemStats::getEnvironmentVariable("OPENBROADCAST_VST_ASSETS",{});
    if (override.isNotEmpty()) return nativePath(override);
    auto file=juce::File::getSpecialLocation(juce::File::currentExecutableFile);
    return nativePath(file.getParentDirectory().getParentDirectory().getChildFile("Resources").getFullPathName());
}
void BroadcastProcessor::fail(const std::string& text) {
    { std::lock_guard<std::mutex> lock(statusMutex); error=text; }
    failed.store(true,std::memory_order_release);
}
void BroadcastProcessor::prepareToPlay(double rate,int) {
    const std::lock_guard<std::recursive_mutex> lock(lifecycleMutex);
    releaseResources(); failed=false; underruns=0;
    resetRequested=false; generation=0;
    { std::lock_guard<std::mutex> lock(statusMutex); error.clear(); deviceNote.clear(); activeDevice="Initializing"; }
    channels=getTotalNumInputChannels(); offset=0; sequence=0; haveWet=false; dryPos=0;
    for(auto& d:dry)d.clear();
    selectedBackend=static_cast<int>(parameters.getRawParameterValue("backend")->load());
    selectedDevice=static_cast<int>(parameters.getRawParameterValue("device")->load());
    selectedStudio=parameters.getRawParameterValue("studioEnabled")->load()>.5f;
    if (!std::isfinite(rate) || rate < 8000 || rate > 192000) { setLatencySamples(0); fail("Supported sample rates: 8–192 kHz"); return; }
    packetSize=static_cast<int>(std::lround(rate*.02));
    const int delay=bufferingPackets*packetSize+ob::Channel::delay(rate,selectedStudio.load());
    setLatencySamples(delay);
    for (auto& d : dry) d.assign(static_cast<size_t>(delay),0.f);
    incoming={}; outgoing={}; wetSmooth=parameters.getRawParameterValue("mix")->load();
    try {
        auto path=resourcePath();
        auto initialize=[&](const ob::GpuDevice* gpu) {
            engine=std::make_unique<ob::ProcessingEngine>(path,rate,channels,gpu,selectedStudio.load());
        };
        juce::String deviceName="CPU", note;
        if(selectedBackend.load()==2) {
            bool usingGpu=false;
            for(const auto& gpu:devices) {
                try {
                    initialize(&gpu); usingGpu=true;
                    deviceName=juce::String(gpu.name.c_str())+" - "+ob::gpuBackendName(gpu.backend);
                    break;
                } catch(const std::exception& e) {
                    note=juce::String(gpu.name.c_str())+" ("+ob::gpuBackendName(gpu.backend)+") initialization failed: "+juce::String(e.what());
                }
            }
            if(!usingGpu) {
                initialize(nullptr);
                note=devices.empty() ? "Auto: no compatible GPU; using CPU." : "Auto: using CPU. "+note;
            }
        } else if(selectedBackend.load()==1) {
            auto gpu=std::find_if(devices.begin(),devices.end(),[&](const auto& d){return d.selectionId()==selectedDevice.load();});
            if(gpu==devices.end())throw std::runtime_error("Selected GPU is unavailable. Choose another device.");
            initialize(&*gpu);
            deviceName=juce::String(gpu->name.c_str())+" - "+ob::gpuBackendName(gpu->backend);
        } else initialize(nullptr);
        { std::lock_guard<std::mutex> lock(statusMutex); activeDevice=deviceName; deviceNote=note; }
        queues=std::make_unique<Queues>(); stopping=false; ready=true;
        worker=std::thread([this]{runWorker();});
    } catch (const std::exception& e) { fail(e.what()); ready=false; }
    prepared.store(true,std::memory_order_release);
}
void BroadcastProcessor::releaseResources() {
    const std::lock_guard<std::recursive_mutex> lock(lifecycleMutex);
    prepared.store(false,std::memory_order_release);
    stopping.store(true,std::memory_order_release);
    if (worker.joinable()) worker.join();
    ready=false; queues.reset(); engine.reset();
}
bool BroadcastProcessor::configurationChanged() const {
    const auto requested=static_cast<int>(parameters.getRawParameterValue("backend")->load());
    return requested!=selectedBackend.load()
        || (requested==1 && static_cast<int>(parameters.getRawParameterValue("device")->load())!=selectedDevice.load())
        || (parameters.getRawParameterValue("studioEnabled")->load()>.5f)!=selectedStudio.load();
}
void BroadcastProcessor::timerCallback() {
    // Poll at processor level so host parameter edits and restored state work
    // with the editor closed, without posting messages from the audio thread.
    if(configurationLoading.load(std::memory_order_acquire))return;
    if(configurationWorker.joinable())configurationWorker.join();
    if(!prepared.load(std::memory_order_acquire) || !configurationChanged())return;
    configurationLoading.store(true,std::memory_order_release);
    configurationWorker=std::thread([this] {
        applyConfiguration();
        configurationLoading.store(false,std::memory_order_release);
    });
}
void BroadcastProcessor::applyConfiguration() {
    // Serialize with host activation/release. Model loading runs off both the
    // message thread and audio thread; later edits are picked up on completion.
    std::unique_lock<std::recursive_mutex> lifecycleLock(lifecycleMutex);
    if(!prepared.load(std::memory_order_acquire) || !configurationChanged() || getSampleRate()<=0)return;
    bool wasSuspended;
    {
        // A host may hold this lock while waiting for releaseResources().
        // Never wait on it while owning lifecycleMutex; retry next timer tick.
        const juce::ScopedTryLock lock(getCallbackLock());
        if(!lock.isLocked())return;
        wasSuspended=isSuspended();
        suspendProcessing(true);
    }
    try { prepareToPlay(getSampleRate(),getBlockSize()); }
    catch(const std::exception& e) { fail(e.what()); }
    // Release lifecycle ownership before reacquiring the host callback lock.
    lifecycleLock.unlock();
    suspendProcessing(wasSuspended);
}
void BroadcastProcessor::runWorker() {
    try {
        uint64_t expected=0, currentGeneration=0;
        while(!stopping.load(std::memory_order_acquire)) {
            if (const auto* packet=queues->input.front()) {
                if(packet->generation!=generation.load(std::memory_order_acquire)) { queues->input.pop(); continue; }
                if(packet->generation!=currentGeneration) {
                    currentGeneration=packet->generation; expected=0;
                    if(engine)engine->reset();
                }
                // An overrun invalidates the model timeline. Fail to aligned dry
                // instead of replaying stale wet audio or hiding missing input.
                if (packet->sequence != expected++) throw std::runtime_error("Inference overloaded; reactivate plugin or use CPU / fewer instances");
                ob::Packet result=*packet; queues->input.pop();
                engine->process(result);
                while(result.generation==generation.load(std::memory_order_acquire) && !stopping.load(std::memory_order_acquire)) {
                    if(queues->output.push(result))break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            } else std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    } catch(const std::exception& e) { fail(e.what()); }
}
void BroadcastProcessor::processBlock(juce::AudioBuffer<float>& b,juce::MidiBuffer& midi) { midi.clear(); process(b,false); }
void BroadcastProcessor::processBlockBypassed(juce::AudioBuffer<float>& b,juce::MidiBuffer& midi) { midi.clear(); process(b,true); }
void BroadcastProcessor::process(juce::AudioBuffer<float>& buffer,bool bypassed) {
    juce::ScopedNoDenormals denormals;
    if(resetRequested.exchange(false,std::memory_order_acq_rel)) {
        generation.fetch_add(1,std::memory_order_release);
        offset=0; sequence=0; dryPos=0; haveWet=false;
        for(auto& d:dry)std::fill(d.begin(),d.end(),0.f);
        if(queues)while(queues->output.front())queues->output.pop();
    }
    const auto count=std::min(channels,buffer.getNumChannels());
    for(int c=count;c<buffer.getNumChannels();++c)buffer.clear(c,0,buffer.getNumSamples());
    if(dry[0].empty()) { buffer.clear(); return; }
    const bool bypass=bypassed || parameters.getRawParameterValue("bypass")->load()>.5f;
    const float target=bypass ? 0.f : parameters.getRawParameterValue("mix")->load();
    for(int i=0;i<buffer.getNumSamples();++i) {
        if(offset==0) {
            haveWet=false;
            if(ready && sequence>=bufferingPackets && !failed.load(std::memory_order_acquire)) {
                const auto wanted=sequence-bufferingPackets;
                const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
                for(;;) {
                    const auto* packet=queues->output.front();
                    if(packet && (packet->generation!=generation.load(std::memory_order_relaxed) || packet->sequence<wanted)) { queues->output.pop(); continue; }
                    if(packet && packet->sequence==wanted) { outgoing=*packet; queues->output.pop(); haveWet=true; break; }
                    if(!isNonRealtime() || failed.load() || stopping.load())break;
                    if(std::chrono::steady_clock::now()>deadline) { fail("Offline inference timed out"); break; }
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
                if(!haveWet)++underruns;
            }
            incoming.sequence=sequence; incoming.count=packetSize; incoming.channels=count;
            incoming.generation=generation.load(std::memory_order_relaxed);
            incoming.noise=parameters.getRawParameterValue("denoiseEnabled")->load()>.5f ? parameters.getRawParameterValue("noise")->load() : 0.f;
            incoming.reverb=parameters.getRawParameterValue("dereverbEnabled")->load()>.5f ? parameters.getRawParameterValue("reverb")->load() : 0.f;
        }
        wetSmooth += (target-wetSmooth)*.002f;
        for(int c=0;c<count;++c) {
            float x=buffer.getSample(c,i); if(!std::isfinite(x))x=0;
            incoming.audio[c][offset]=x;
            float delayed=dry[c][dryPos]; dry[c][dryPos]=x;
            float wet=haveWet && !failed.load(std::memory_order_relaxed) ? outgoing.audio[c][offset] : delayed;
            buffer.setSample(c,i,delayed+(wet-delayed)*wetSmooth);
        }
        dryPos=(dryPos+1)%static_cast<int>(dry[0].size());
        if(++offset==packetSize) {
            if(ready && !failed.load(std::memory_order_relaxed)) {
                if(isNonRealtime()) {
                    while(!queues->input.push(incoming) && !failed.load() && !stopping.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
                } else if(!queues->input.push(incoming)) ++underruns;
            }
            offset=0; ++sequence;
        }
    }
}
juce::String BroadcastProcessor::status() const {
    std::lock_guard<std::mutex> lock(statusMutex);
    if(configurationLoading.load(std::memory_order_acquire))return "Loading models... Audio resumes automatically.";
    if(failed.load(std::memory_order_acquire)) return "Dry fallback: "+juce::String(error);
    juce::String text=activeDevice+(selectedStudio.load() ? " · Studio Voice" : "");
    if(configurationChanged())
        text+="\nUpdating settings...";
    else if(deviceNote.isNotEmpty())text+="\n"+deviceNote;
    return text+"\nLatency: "+juce::String(getSampleRate()>0 ? getLatencySamples()*1000./getSampleRate() : 0.,1)+" ms  ·  Late packets: "+juce::String(underruns.load());
}
juce::AudioProcessorEditor* BroadcastProcessor::createEditor(){return new Editor(*this);}
juce::AudioProcessorParameter* BroadcastProcessor::getBypassParameter() const {return parameters.getParameter("bypass");}
void BroadcastProcessor::getStateInformation(juce::MemoryBlock& data){auto xml=parameters.copyState().createXml();copyXmlToBinary(*xml,data);}
void BroadcastProcessor::setStateInformation(const void* data,int size) {
    auto xml=getXmlFromBinary(data,size);
    if(!xml || !xml->hasTagName(parameters.state.getType()))return;
    auto state=juce::ValueTree::fromXml(*xml);
    // Old projects only had amount controls. Both stages must remain enabled,
    // including zero amounts, so their existing amount automation still works.
    for(auto id:{"denoiseEnabled","dereverbEnabled"}) {
        if(!state.getChildWithProperty("id",id).isValid()) {
            juce::ValueTree parameter("PARAM"); parameter.setProperty("id",id,nullptr);
            parameter.setProperty("value",1.f,nullptr); state.appendChild(parameter,nullptr);
        }
    }
    if(!state.getChildWithProperty("id","studioEnabled").isValid()) {
        juce::ValueTree parameter("PARAM"); parameter.setProperty("id","studioEnabled",nullptr);
        parameter.setProperty("value",0.f,nullptr); state.appendChild(parameter,nullptr);
    }
    parameters.replaceState(state);
}
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter(){return new BroadcastProcessor();}
