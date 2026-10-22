#include "ProcessingEngine.h"
#include <stdexcept>
#ifdef __linux__
#include "LinuxProtocol.h"
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <poll.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <cerrno>
#include <cstring>
#include <chrono>
#include <thread>
#include <codecvt>
#include <locale>
extern char** environ;
#endif

namespace ob {
#ifdef __linux__
namespace {
class Worker {
public:
    Worker(const std::filesystem::path& resources,GpuBackend backend) {
        auto directory=std::filesystem::absolute(resources)/"runtime"/(backend==GpuBackend::migraphx ? "migraphx" : "cuda");
        auto executable=(directory/"ob_inference_worker").string();
        auto assets=std::filesystem::absolute(resources).string(), libraries=directory.string();
        int sockets[2];
        if(socketpair(AF_UNIX,SOCK_STREAM|SOCK_CLOEXEC,0,sockets))throw std::runtime_error("Cannot create inference socket");
        // Move both ends above fd 3 before setting up the child's protocol fd.
        int parent=fcntl(sockets[0],F_DUPFD_CLOEXEC,4), child=fcntl(sockets[1],F_DUPFD_CLOEXEC,4);
        close(sockets[0]); close(sockets[1]);
        if(parent<0 || child<0) {
            if(parent>=0)close(parent); if(child>=0)close(child);
            throw std::runtime_error("Cannot allocate inference socket descriptors");
        }
        posix_spawn_file_actions_t actions;
        int error=posix_spawn_file_actions_init(&actions);
        if(error) { close(parent); close(child); throw std::runtime_error("Cannot initialize inference process actions"); }
        if(!(error=posix_spawn_file_actions_adddup2(&actions,child,3)))
            error=posix_spawn_file_actions_addclosefrom_np(&actions,4);
        char* args[]{executable.data(),assets.data(),libraries.data(),nullptr};
        if(!error)error=posix_spawn(&pid,executable.c_str(),&actions,nullptr,args,environ);
        posix_spawn_file_actions_destroy(&actions); close(child);
        if(error) { close(parent);pid=-1;throw std::runtime_error("Cannot start "+std::string(gpuBackendName(backend))+" worker: "+std::strerror(error)); }
        fd=parent;
    }
    ~Worker() {
        if(fd>=0)close(fd); // EOF requests normal worker cleanup.
        if(pid>0) {
            for(int i=0;i<50;++i) {
                auto result=waitpid(pid,nullptr,WNOHANG);
                if(result==pid || (result<0 && errno==ECHILD))return;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            kill(pid,SIGKILL);
            while(waitpid(pid,nullptr,0)<0 && errno==EINTR){}
        }
    }
    ipc::Reply exchange(const ipc::Request& request,int seconds=10) {
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(seconds);
        transfer(const_cast<ipc::Request*>(&request),sizeof(request),true,deadline);
        ipc::Reply reply;
        transfer(&reply,sizeof(reply),false,deadline);
        if(reply.signature!=ipc::magic || reply.abi!=ipc::version)throw std::runtime_error("Incompatible inference worker reply");
        if(reply.failed) { reply.error[sizeof(reply.error)-1]=0; throw std::runtime_error(reply.error); }
        return reply;
    }
private:
    void transfer(void* data,size_t bytes,bool writing,std::chrono::steady_clock::time_point deadline) {
        auto* p=static_cast<char*>(data);
        while(bytes) {
            auto remaining=std::chrono::duration_cast<std::chrono::milliseconds>(deadline-std::chrono::steady_clock::now()).count();
            if(remaining<=0)throw std::runtime_error("Inference worker timed out");
            pollfd event{fd,static_cast<short>(writing ? POLLOUT : POLLIN),0};
            auto ready=poll(&event,1,static_cast<int>(remaining));
            if(ready<0 && errno==EINTR)continue;
            if(ready<0)throw std::runtime_error("Inference socket poll failed");
            if(!ready)throw std::runtime_error("Inference worker timed out");
            auto n=writing ? send(fd,p,bytes,MSG_NOSIGNAL|MSG_DONTWAIT) : recv(fd,p,bytes,MSG_DONTWAIT);
            if(n<0 && (errno==EINTR || errno==EAGAIN))continue;
            if(n<=0)throw std::runtime_error("Inference worker exited or disconnected");
            p+=n; bytes-=static_cast<size_t>(n);
        }
    }
    int fd{-1}; pid_t pid{-1};
};
}
struct ProcessingEngine::Impl { std::unique_ptr<Worker> worker; };
std::vector<GpuDevice> discoverProcessingDevices(const std::filesystem::path& resources) {
    std::vector<GpuDevice> devices;
    for(auto backend:{GpuBackend::platform,GpuBackend::migraphx}) {
        try {
            Worker worker(resources,backend);
            ipc::Request request; request.command=ipc::discover;
            auto reply=worker.exchange(request);
            if(reply.count<0 || reply.count>16)continue;
            for(int i=0;i<reply.count;++i) {
                auto& d=reply.devices[i]; d.name[sizeof(d.name)-1]=0;
                if(d.index<0 || d.index>15)continue;
                devices.push_back({d.index,std::wstring_convert<std::codecvt_utf8<wchar_t>>().from_bytes(d.name),
                                   d.memory,d.rank<0 ? 0 : d.rank,d.score,backend});
            }
        } catch(const std::exception&) { /* A missing GPU stack must not prevent CPU use. */ }
    }
    return rankGpuDevices(std::move(devices));
}
ProcessingEngine::ProcessingEngine(const std::filesystem::path& resources,double rate,int channels,const GpuDevice* gpu,bool studio)
    : impl(std::make_unique<Impl>()) {
    auto initialize=[&](GpuBackend backend) {
        auto worker=std::make_unique<Worker>(resources,backend);
        ipc::Request request; request.command=ipc::initialize; request.device=gpu ? gpu->index : -1;
        request.channels=channels; request.rate=rate; request.studio=studio ? 1 : 0;
        worker->exchange(request,180); impl->worker=std::move(worker);
    };
    if(gpu)initialize(gpu->backend);
    else {
        try { initialize(GpuBackend::platform); }
        catch(const std::exception&) { initialize(GpuBackend::migraphx); }
    }
}
void ProcessingEngine::process(Packet& packet) {
    ipc::Request request; request.command=ipc::process; request.audio=packet;
    auto reply=impl->worker->exchange(request,30);
    if(reply.audio.count!=packet.count || reply.audio.channels!=packet.channels
       || reply.audio.sequence!=packet.sequence || reply.audio.generation!=packet.generation)
        throw std::runtime_error("Invalid inference worker audio reply");
    packet=reply.audio;
}
void ProcessingEngine::reset() { ipc::Request request; request.command=ipc::reset; impl->worker->exchange(request); }
#else
struct ProcessingEngine::Impl { std::array<std::unique_ptr<Channel>,2> channels; };
std::vector<GpuDevice> discoverProcessingDevices(const std::filesystem::path&) { return discoverGpuDevices(); }
ProcessingEngine::ProcessingEngine(const std::filesystem::path& resources,double rate,int channels,const GpuDevice* gpu,bool studio)
    : impl(std::make_unique<Impl>()) {
    auto runtime=std::make_shared<Runtime>(resources/"runtime");
    auto dsp=std::make_shared<Coefficients>(resources/"dsp.bin");
    for(int c=0;c<channels;++c)impl->channels[c]=std::make_unique<Channel>(runtime,dsp,resources,rate,gpu ? gpu->index : -1,studio);
}
void ProcessingEngine::process(Packet& packet) {
    for(int i=0;i<packet.count;++i)for(int c=0;c<packet.channels;++c)
        packet.audio[c][i]=impl->channels[c]->tick(packet.audio[c][i],packet.noise,packet.reverb);
}
void ProcessingEngine::reset() { for(auto& c:impl->channels)if(c)c->reset(); }
#endif
ProcessingEngine::~ProcessingEngine()=default;
}
