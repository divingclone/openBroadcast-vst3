#pragma once
#include <array>
#include <atomic>
#include <cstdint>
namespace ob {
constexpr int maxPacket = 3840;
struct Packet {
    uint64_t sequence{};
    uint64_t generation{};
    int count{}, channels{};
    float noise{1}, reverb{};
    std::array<std::array<float, maxPacket>, 2> audio{};
};
// One producer and one consumer, with ownership transferred by release/acquire.
template<class T, size_t Capacity> class Spsc {
public:
    bool push(const T& item) {
        const auto w = written.load(std::memory_order_relaxed);
        if (w - read.load(std::memory_order_acquire) == Capacity) return false;
        slots[w % Capacity] = item;
        written.store(w+1, std::memory_order_release); return true;
    }
    const T* front() const {
        auto r = read.load(std::memory_order_relaxed);
        return r == written.load(std::memory_order_acquire) ? nullptr : &slots[r % Capacity];
    }
    void pop() { read.fetch_add(1, std::memory_order_release); }
private:
    std::array<T, Capacity> slots{};
    alignas(64) std::atomic<uint64_t> written{0};
    alignas(64) std::atomic<uint64_t> read{0};
};
}
