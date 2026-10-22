#pragma once
#include <juce_audio_utils/juce_audio_utils.h>
#include "Engine.h"
#include "ProcessingEngine.h"
#include "Transport.h"
#include "GpuDevices.h"
#include <thread>
#include <mutex>

class BroadcastProcessor final : public juce::AudioProcessor, private juce::Timer {
public:
    BroadcastProcessor();
    ~BroadcastProcessor() override;
    void prepareToPlay(double, int) override;
    void releaseResources() override;
    void reset() override { resetRequested.store(true,std::memory_order_release); }
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    void processBlockBypassed(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    bool isBusesLayoutSupported(const BusesLayout&) const override;
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }
    const juce::String getName() const override { return "OpenBroadcast"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return .5; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return "Default"; }
    void changeProgramName(int, const juce::String&) override {}
    void getStateInformation(juce::MemoryBlock&) override;
    void setStateInformation(const void*, int) override;
    juce::AudioProcessorParameter* getBypassParameter() const override;
    juce::String status() const;
    const std::vector<ob::GpuDevice> devices;
    juce::AudioProcessorValueTreeState parameters;
    std::atomic<uint64_t> underruns{0};
private:
    void timerCallback() override;
    bool configurationChanged() const;
    void applyConfiguration();
    std::recursive_mutex lifecycleMutex;
    std::thread configurationWorker;
    std::atomic<bool> prepared{false}, configurationLoading{false};
    void process(juce::AudioBuffer<float>&, bool bypass);
    void runWorker();
    void fail(const std::string&);
    static juce::AudioProcessorValueTreeState::ParameterLayout layout();
    std::filesystem::path resourcePath() const;
    struct Queues { ob::Spsc<ob::Packet,16> input, output; };
    std::unique_ptr<Queues> queues;
    std::unique_ptr<ob::ProcessingEngine> engine;
    std::thread worker;
    std::atomic<bool> stopping{false}, failed{false};
    std::atomic<bool> resetRequested{false};
    std::atomic<uint64_t> generation{0};
    bool ready{}, haveWet{};
    int packetSize{960}, offset{}, channels{2}, dryPos{};
    std::atomic<int> selectedBackend{2}, selectedDevice{0};
    std::atomic<bool> selectedStudio{false};
    uint64_t sequence{};
    ob::Packet incoming{}, outgoing{};
    std::array<std::vector<float>,2> dry;
    float wetSmooth{1};
    mutable std::mutex statusMutex;
    std::string error;
    juce::String activeDevice{"Not active"}, deviceNote;
};
