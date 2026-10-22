#include "ProcessingEngine.h"
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <signal.h>
#include <cstdlib>
static void require(bool value,const char* message) { if(!value)throw std::runtime_error(message); }
static std::vector<pid_t> children() {
    std::ifstream file("/proc/self/task/"+std::to_string(getpid())+"/children");
    std::vector<pid_t> result; pid_t child;
    while(file>>child)result.push_back(child);
    return result;
}
int main(int argc,char** argv) {
    try {
        require(argc==2,"Expected bundled resources");
        setenv("OB_TEST_CUDA","1",1);
        auto devices=ob::discoverProcessingDevices(argv[1]);
        require(devices.size()==3,"Discover both vendors in one bundle");
        require(devices.front().name==L"AMD Test Discrete","Prefer fast discrete GPU");
        auto amd=std::find_if(devices.begin(),devices.end(),[](auto& d){return d.backend==ob::GpuBackend::migraphx;});
        auto cuda=std::find_if(devices.begin(),devices.end(),[](auto& d){return d.backend==ob::GpuBackend::platform;});
        require(amd!=devices.end() && cuda!=devices.end() && amd->selectionId()!=cuda->selectionId(),"Backend selections must not collide");
        require(children().empty(),"Discovery workers must be reaped");
        // Exercise both real ORT versions simultaneously without requiring GPUs.
        // The negative ordinal selects CPU inside the AMD worker only in this test.
        ob::GpuDevice amdCpu; amdCpu.index=-1; amdCpu.backend=ob::GpuBackend::migraphx;
        for(bool studio:{false,true}) {
            {
                ob::ProcessingEngine native(argv[1],48000,2,nullptr,studio);
                auto nativePids=children(); require(nativePids.size()==1,"One worker per instance");
                ob::ProcessingEngine other(argv[1],48000,2,&amdCpu,studio);
                require(children().size()==2,"Concurrent isolated runtimes");
                std::vector<float> first;
                float maximum=0;
                for(int pass=0;pass<2;++pass) {
                    native.reset(); other.reset();
                    size_t position=0;
                    for(int n=0;n<20;++n) {
                        ob::Packet a; a.count=960;a.channels=2;a.sequence=n;a.generation=pass;a.noise=.7f;a.reverb=.5f;
                        for(int c=0;c<2;++c)for(int i=0;i<a.count;++i)
                            a.audio[c][i]=.03f*std::sin((n*960+i)*(.037f+c*.011f));
                        auto b=a; native.process(a);other.process(b);
                        for(int c=0;c<2;++c)for(int i=0;i<a.count;++i) {
                            require(std::isfinite(a.audio[c][i]) && std::isfinite(b.audio[c][i]),"Finite worker audio");
                            maximum=std::max(maximum,std::abs(a.audio[c][i]-b.audio[c][i]));
                            if(!pass)first.push_back(b.audio[c][i]);
                            else require(std::abs(b.audio[c][i]-first[position++])<1e-6f,"Worker reset must reproduce audio");
                        }
                    }
                }
                require(maximum<(studio ? .003f : .0003f),"Concurrent runtime audio parity");
                std::cout<<"Isolated runtime parity, studio="<<studio<<" max_error="<<maximum<<'\n';
                require(kill(nativePids.front(),SIGKILL)==0,"Kill this test's CUDA worker");
                bool detected=false;
                try { native.reset(); } catch(const std::exception&) { detected=true; }
                require(detected,"Worker crash must surface as an error, not terminate host");
                other.reset(); // The other runtime must remain usable.
            }
            require(children().empty(),"Inference workers must be reaped");
        }
        std::cout<<"UNIFIED RUNTIME TESTS PASSED (simulated discovery, real CPU runtimes)\n";
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
