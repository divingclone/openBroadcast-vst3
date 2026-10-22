#include "Engine.h"
#include <iostream>
int main(int argc,char** argv) {
    try {
        if(argc!=3)throw std::runtime_error("Expected assets and runtime paths");
        ob::Runtime runtime(argv[2]);
        const auto providers=Ort::GetAvailableProviders();
        if(std::find(providers.begin(),providers.end(),"MIGraphXExecutionProvider")==providers.end())
            throw std::runtime_error("AMD package must contain a MIGraphX-enabled ONNX Runtime");
        Ort::SessionOptions options;
        options.AddConfigEntry("session.record_ep_graph_assignment_info","1");
        auto model=std::filesystem::path(argv[1])/"models/denoiser_48k.onnx";
        Ort::Session cpu(*runtime.env,model.c_str(),options);
        runtime.verifyDevice(cpu,-1);
        try { runtime.verifyDevice(cpu,0); }
        catch(const std::runtime_error& e) {
            if(std::string(e.what()).find("did not assign any model operations")!=std::string::npos) {
                std::cout<<"CPU-ONLY MODEL IS NOT REPORTED AS AMD GPU\n";
                return 0;
            }
            throw;
        }
        throw std::runtime_error("CPU model was falsely reported as GPU acceleration");
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';return 1; }
}
