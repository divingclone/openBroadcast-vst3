#pragma once
#include "Transport.h"
#include <cstdint>
#include <type_traits>

namespace ob::ipc {
constexpr uint32_t magic=0x4f425736, version=1;
enum Command : uint32_t { discover=1, initialize, process, reset };
struct Device {
    int32_t index{}, rank{-1};
    uint64_t memory{}, score{};
    char name[256]{};
};
struct Request {
    uint32_t signature{magic}, abi{version}, command{};
    int32_t device{-1}, channels{2}, studio{};
    double rate{48000};
    Packet audio{};
};
struct Reply {
    uint32_t signature{magic}, abi{version};
    int32_t failed{}, count{};
    Device devices[16]{};
    Packet audio{};
    char error[2048]{};
};
static_assert(std::is_trivially_copyable_v<Request> && std::is_trivially_copyable_v<Reply>);
}
