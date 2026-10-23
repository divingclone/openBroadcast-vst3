#pragma once
#include <array>
#include <atomic>
#include <cstdint>

namespace ob {
// Worker-owned accumulation; the editor reads only the published atomics.
// Four 20 ms packets cover Studio Voice's full 80 ms inference cycle.
class PerformanceMeter {
public:
    void reset() {
        groups={}; index=0; count=0; packets=0;
        pendingCompute=0; pendingAudio=0;
        publishedAt.store(0,std::memory_order_release);
    }
    void processed(double computeMs,double audioMs,uint64_t generation,int64_t nowMs) {
        pendingCompute+=computeMs; pendingAudio+=audioMs;
        if(++packets<4)return;
        groups[index]={pendingCompute,pendingAudio};
        index=(index+1)%groups.size();
        if(count<groups.size())++count;
        pendingCompute=0; pendingAudio=0; packets=0;
        double compute=0,audio=0;
        for(std::size_t i=0;i<count;++i) { compute+=groups[i].compute; audio+=groups[i].audio; }
        percent.store(audio>0 ? 100.*compute/audio : 0.,std::memory_order_relaxed);
        measuredGeneration.store(generation,std::memory_order_relaxed);
        publishedAt.store(nowMs,std::memory_order_release);
    }
    // A negative reading means no recent measurement for this audio timeline.
    double load(uint64_t generation,int64_t nowMs) const {
        const auto at=publishedAt.load(std::memory_order_acquire);
        if(at==0 || nowMs-at>1500 || measuredGeneration.load(std::memory_order_relaxed)!=generation)return -1.;
        return percent.load(std::memory_order_relaxed);
    }
private:
    struct Group { double compute{},audio{}; };
    std::array<Group,12> groups{}; // Approximately one second of processed audio.
    std::size_t index{},count{};
    int packets{};
    double pendingCompute{},pendingAudio{};
    std::atomic<double> percent{0};
    std::atomic<uint64_t> measuredGeneration{0};
    std::atomic<int64_t> publishedAt{0};
};
}
