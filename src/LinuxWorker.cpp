#include "Engine.h"
#include "GpuDevices.h"
#include "LinuxProtocol.h"
#include <sys/socket.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <codecvt>
#include <locale>

namespace {
bool transfer(int fd,void* data,size_t bytes,bool write) {
    auto* p=static_cast<char*>(data);
    while(bytes) {
        auto n=write ? send(fd,p,bytes,MSG_NOSIGNAL) : recv(fd,p,bytes,0);
        if(n<0 && errno==EINTR)continue;
        if(n<=0)return false;
        p+=n; bytes-=static_cast<size_t>(n);
    }
    return true;
}
}
int main(int argc,char** argv) {
    if(argc!=3)return 2;
    // The private socket is descriptor 3, supplied by posix_spawn file actions.
    const std::filesystem::path resources=argv[1], libraries=argv[2];
    std::shared_ptr<ob::Runtime> runtime;
    std::shared_ptr<ob::Coefficients> dsp;
    std::array<std::unique_ptr<ob::Channel>,2> channels;
    int activeChannels=0;
    ob::ipc::Request request;
    while(transfer(3,&request,sizeof(request),false)) {
        ob::ipc::Reply reply;
        try {
            if(request.signature!=ob::ipc::magic || request.abi!=ob::ipc::version)
                throw std::runtime_error("Incompatible inference worker protocol");
            switch(request.command) {
            case ob::ipc::discover: {
                auto devices=ob::discoverGpuDevices();
                reply.count=static_cast<int>(std::min<size_t>(devices.size(),16));
                for(int i=0;i<reply.count;++i) {
                    const auto& d=devices[i]; auto& wire=reply.devices[i];
                    wire.index=d.index; wire.rank=d.performanceRank; wire.memory=d.dedicatedMemory; wire.score=d.computeScore;
                    auto name=std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(d.name);
                    std::snprintf(wire.name,sizeof(wire.name),"%s",name.c_str());
                }
                break;
            }
            case ob::ipc::initialize: {
                if(request.channels<1 || request.channels>2 || !std::isfinite(request.rate)
                   || request.rate<8000 || request.rate>192000 || request.device<-1 || request.device>15)
                    throw std::runtime_error("Invalid inference configuration");
                if(!runtime)runtime=std::make_shared<ob::Runtime>(libraries);
                if(!dsp)dsp=std::make_shared<ob::Coefficients>(resources/"dsp.bin");
                std::array<std::unique_ptr<ob::Channel>,2> next;
                for(int c=0;c<request.channels;++c)
                    next[c]=std::make_unique<ob::Channel>(runtime,dsp,resources,request.rate,request.device,request.studio!=0);
                channels=std::move(next); activeChannels=request.channels;
                break;
            }
            case ob::ipc::process:
                if(!activeChannels || request.audio.channels!=activeChannels || request.audio.count<0 || request.audio.count>ob::maxPacket)
                    throw std::runtime_error("Invalid inference packet");
                reply.audio=request.audio;
                for(int i=0;i<reply.audio.count;++i)for(int c=0;c<activeChannels;++c)
                    reply.audio.audio[c][i]=channels[c]->tick(reply.audio.audio[c][i],reply.audio.noise,reply.audio.reverb);
                break;
            case ob::ipc::reset:
                for(auto& channel:channels)if(channel)channel->reset();
                break;
            default: throw std::runtime_error("Unknown inference command");
            }
        } catch(const std::exception& e) {
            reply.failed=1; std::snprintf(reply.error,sizeof(reply.error),"%s",e.what());
        } catch(...) {
            reply.failed=1; std::snprintf(reply.error,sizeof(reply.error),"Unknown inference error");
        }
        if(!transfer(3,&reply,sizeof(reply),true))break;
    }
    return 0;
}
