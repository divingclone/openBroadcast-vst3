# OpenBroadcast VST3

An unofficial VST3 plugin based on [OpenBroadcast](https://github.com/divingclone/openBroadcast), with Denoise, Dereverb and Studio Voice; NVIDIA Broadcast itself is not required.
Auto selects a high-performance GPU and displays its model: DirectML on Windows, CUDA or MIGraphX on Linux, and system-managed CoreML CPU/GPU on macOS; unavailable GPU acceleration falls back to CPU. Studio Voice is off by default and uses substantial resources.
Windows requires the Visual C++ v14 x64 runtime. The single Linux x64 ZIP supports NVIDIA (CUDA 12.8+ and cuDNN 9) and AMD (compatible ROCm 7 / MIGraphX 3), automatically choosing the matching GPU backend; install the appropriate system drivers and compute libraries for GPU acceleration.

## Installation

Close your host, extract the ZIP for your operating system and architecture, copy the entire `OpenBroadcast.vst3` folder to the location below, then reopen your VST3 host and rescan (add the folder to its scan paths if needed).

| Platform | User installation folder |
| --- | --- |
| Windows x64 | `%LOCALAPPDATA%\Programs\Common\VST3` (or `C:\Program Files\Common Files\VST3` for all users) |
| Linux x64 (built on Ubuntu 24.04) | `~/.vst3` |
| macOS 14+ (Apple Silicon / Intel) | `~/Library/Audio/Plug-Ins/VST3` |

Download the ZIP for your platform from the latest successful [Actions build](https://github.com/divingclone/openBroadcast-vst3/actions/workflows/native-builds.yml). Choose `macOS-arm64` for Apple Silicon or `macOS-x64` for Intel; macOS packages are ad-hoc signed, not notarized. AMD GPU execution still needs validation on physical hardware; see [AMD's compatibility requirements](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html).
