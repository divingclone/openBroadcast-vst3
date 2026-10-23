#include "PerformanceMeter.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
static void require(bool value,const char* message) { if(!value)throw std::runtime_error(message); }
int main() {
    try {
        ob::PerformanceMeter meter;
        require(meter.load(0,1000)<0,"No audio must not report zero load");
        // Studio Voice does most work once per 80 ms: 40 ms / 80 ms = 50%,
        // not the 200% implied by comparing its busy packet to just 20 ms.
        for(int i=0;i<3;++i)meter.processed(0,20,0,1000+i);
        require(meter.load(0,1002)<0,"Partial Studio Voice cycle published");
        meter.processed(40,20,0,1003);
        require(std::abs(meter.load(0,1003)-50)<1e-8,"Studio Voice load denominator");
        require(meter.load(0,2600)<0,"Stale load must expire when audio stops");
        require(meter.load(1,1004)<0,"Seek retained old timeline load");
        meter.reset();
        for(int i=0;i<4;++i)meter.processed(30,20,1,3000+i);
        require(std::abs(meter.load(1,3003)-150)<1e-8,"Overload must not be capped at 100%");
        // A recovered stream must replace the old slow window completely.
        for(int i=0;i<48;++i)meter.processed(5,20,1,3100+i);
        require(std::abs(meter.load(1,3148)-25)<1e-8,"Rolling load failed to recover");
        meter.processed(100,20,1,3200);meter.reset();
        for(int i=0;i<4;++i)meter.processed(2,20,2,3300+i);
        require(std::abs(meter.load(2,3304)-10)<1e-8,"Reset retained partial-cycle work");
        std::cout<<"PERFORMANCE TESTS PASSED\n";
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
