// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Parse-cost variants inside nlohmann only (no new library), on one descriptor file:
//   stream : json::parse(std::ifstream)  -- what DescriptorLoader does today
//   buffer : read file into std::string, json::parse(string)
//   sax    : sax_parse over the buffer with a handler that builds nothing
//            (tokenizer floor: what a SAX-to-struct loader cannot go below)
//   arena  : DOM parse from buffer with a bump allocator whose free is a no-op
//            (what allocation + destruction cost inside the DOM)
// Build: clang++ -O2 -std=c++17 -I <nlohmann single_include> json_floor.cpp -o json_floor
// Usage: json_floor <file.json> [stream|buffer|sax|arena]  (one variant per process)
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b)
{
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// Bump arena: one big block, deallocate is a no-op. Probe-only.
struct Arena
{
    static char* base;
    static size_t used, cap;
};
char* Arena::base = nullptr;
size_t Arena::used = 0, Arena::cap = 0;

template <typename T>
struct BumpAllocator
{
    using value_type = T;
    BumpAllocator() = default;
    template <typename U>
    BumpAllocator(const BumpAllocator<U>&)
    {
    }
    T* allocate(size_t n)
    {
        size_t bytes = (n * sizeof(T) + 15) & ~size_t(15);
        if(Arena::used + bytes > Arena::cap)
        {
            std::abort();
        }
        T* p = reinterpret_cast<T*>(Arena::base + Arena::used);
        Arena::used += bytes;
        return p;
    }
    void deallocate(T*, size_t) {}
    template <typename U>
    bool operator==(const BumpAllocator<U>&) const
    {
        return true;
    }
    template <typename U>
    bool operator!=(const BumpAllocator<U>&) const
    {
        return false;
    }
};
using ArenaJson = nlohmann::basic_json<std::map,
                                       std::vector,
                                       std::string,
                                       bool,
                                       std::int64_t,
                                       std::uint64_t,
                                       double,
                                       BumpAllocator>;

struct NoopSax : nlohmann::json_sax<nlohmann::json>
{
    size_t events = 0;
    bool null() override
    {
        return ++events;
    }
    bool boolean(bool) override
    {
        return ++events;
    }
    bool number_integer(number_integer_t) override
    {
        return ++events;
    }
    bool number_unsigned(number_unsigned_t) override
    {
        return ++events;
    }
    bool number_float(number_float_t, const string_t&) override
    {
        return ++events;
    }
    bool string(string_t&) override
    {
        return ++events;
    }
    bool binary(binary_t&) override
    {
        return ++events;
    }
    bool start_object(std::size_t) override
    {
        return ++events;
    }
    bool key(string_t&) override
    {
        return ++events;
    }
    bool end_object() override
    {
        return ++events;
    }
    bool start_array(std::size_t) override
    {
        return ++events;
    }
    bool end_array() override
    {
        return ++events;
    }
    bool parse_error(std::size_t, const std::string&, const nlohmann::detail::exception&) override
    {
        return false;
    }
};

int main(int argc, char** argv)
{
    const char* path = argv[1];
    const std::string mode = argc > 2 ? argv[2] : "all";
    auto want = [&](const char* m) { return mode == "all" || mode == m; };

    if(want("stream"))
    {
        std::ifstream file(path, std::ios::binary);
        auto t0 = Clock::now();
        auto* doc = new nlohmann::json(nlohmann::json::parse(file, nullptr, true, true));
        auto t1 = Clock::now();
        delete doc;
        auto t2 = Clock::now();
        std::printf("stream   parse %8.1f ms  destroy %7.1f ms\n", ms(t0, t1), ms(t1, t2));
    }

    auto r0 = Clock::now();
    std::string buffer;
    {
        std::ifstream file(path, std::ios::binary);
        std::ostringstream ss;
        ss << file.rdbuf();
        buffer = std::move(ss).str();
    }
    auto r1 = Clock::now();
    std::printf("read     %8.1f ms (%zu bytes into memory)\n", ms(r0, r1), buffer.size());

    if(want("buffer"))
    {
        auto t0 = Clock::now();
        auto* doc = new nlohmann::json(nlohmann::json::parse(buffer, nullptr, true, true));
        auto t1 = Clock::now();
        delete doc;
        auto t2 = Clock::now();
        std::printf("buffer   parse %8.1f ms  destroy %7.1f ms\n", ms(t0, t1), ms(t1, t2));
    }

    if(want("sax"))
    {
        NoopSax sax;
        auto t0 = Clock::now();
        nlohmann::json::sax_parse(buffer, &sax, nlohmann::json::input_format_t::json, true, true);
        auto t1 = Clock::now();
        std::printf("sax_noop parse %8.1f ms  (%zu events)\n", ms(t0, t1), sax.events);
    }

    if(want("arena"))
    {
        Arena::cap = buffer.size() * 16 + (64u << 20);
        Arena::base = static_cast<char*>(std::malloc(Arena::cap));
        auto t0 = Clock::now();
        auto* doc = new ArenaJson(ArenaJson::parse(buffer, nullptr, true, true));
        auto t1 = Clock::now();
        delete doc;
        auto t2 = Clock::now();
        std::printf("arena    parse %8.1f ms  destroy %7.1f ms  (arena %.0f MiB)\n",
                    ms(t0, t1),
                    ms(t1, t2),
                    static_cast<double>(Arena::used) / 1048576.0);
    }
}
