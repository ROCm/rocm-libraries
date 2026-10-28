// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "host_memory.hpp"
#include "hipblaslt_ostream.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace
{
    std::optional<uint64_t> read_number(const std::filesystem::path& file)
    {
        std::ifstream in(file);
        std::string   token;
        if(!(in >> token) || token.empty()
           || token.find_first_not_of("0123456789") != std::string::npos)
            return std::nullopt;
        try
        {
            return std::stoull(token);
        }
        catch(...)
        {
            return std::nullopt;
        }
    }

    // The value after `key` in a "key value" file such as meminfo or memory.stat.
    std::optional<uint64_t> read_field(const std::filesystem::path& file, const std::string& key)
    {
        std::ifstream in(file);
        std::string   name;
        uint64_t      value = 0;
        while(in >> name >> value)
        {
            if(name == key)
                return value;
            in.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        }
        return std::nullopt;
    }

    // Memory left under one cgroup's limit, or nothing when the directory sets no limit. Page
    // cache the kernel can drop (inactive_file) does not count as used.
    std::optional<uint64_t> cgroup_headroom(const std::filesystem::path& dir, bool v2)
    {
        // v2 writes "max" for no limit, which is not a number. v1 writes a value near 2^63,
        // which MemAvailable is always below.
        auto limit = read_number(dir / (v2 ? "memory.max" : "memory.limit_in_bytes"));
        if(!limit)
            return std::nullopt;
        uint64_t used
            = read_number(dir / (v2 ? "memory.current" : "memory.usage_in_bytes")).value_or(0);
        uint64_t reclaimable
            = read_field(dir / "memory.stat", v2 ? "inactive_file" : "total_inactive_file")
                  .value_or(0);
        used = used > reclaimable ? used - reclaimable : 0;
        return *limit > used ? *limit - used : uint64_t(0);
    }

    // The smallest headroom from the process's cgroup up to the hierarchy root. A container may
    // list "/" or the host's path for its cgroup; either way its own limit is at the root it
    // sees, so the walk always ends there.
    std::optional<uint64_t> smallest_cgroup_headroom(const HostMemorySources& s)
    {
        std::ifstream           in(s.self_cgroup);
        std::string             line;
        std::optional<uint64_t> smallest;
        while(std::getline(in, line))
        {
            // hierarchy-id:controllers:path
            size_t a = line.find(':');
            size_t b = a == std::string::npos ? a : line.find(':', a + 1);
            if(b == std::string::npos)
                continue;
            std::string       controllers = line.substr(a + 1, b - a - 1);
            bool              v2          = line.substr(0, a) == "0" && controllers.empty();
            bool              v1          = false;
            std::stringstream list(controllers);
            for(std::string c; std::getline(list, c, ',');)
                v1 = v1 || c == "memory";
            if(!v2 && !v1)
                continue;

            std::filesystem::path root = s.cgroup_root;
            if(v1)
                root /= "memory";
            std::filesystem::path rel = std::filesystem::path(line.substr(b + 1)).relative_path();
            while(true)
            {
                if(auto h = cgroup_headroom(root / rel, v2))
                    smallest = std::min(smallest.value_or(*h), *h);
                if(rel.empty())
                    break;
                rel = rel.parent_path();
            }
        }
        return smallest;
    }
} // namespace

std::optional<size_t> parse_host_memory_size(const char* text)
{
    if(!text || !std::isdigit(static_cast<unsigned char>(*text)))
        return std::nullopt;
    char* end                = nullptr;
    errno                    = 0;
    unsigned long long value = std::strtoull(text, &end, 10);
    if(errno == ERANGE)
        return std::nullopt;
    int shift = 0;
    switch(std::toupper(static_cast<unsigned char>(*end)))
    {
    case '\0':
        break;
    case 'K':
        shift = 10;
        break;
    case 'M':
        shift = 20;
        break;
    case 'G':
        shift = 30;
        break;
    case 'T':
        shift = 40;
        break;
    default:
        return std::nullopt;
    }
    if(shift && *++end != '\0')
        return std::nullopt;
    if(value > (std::numeric_limits<size_t>::max() >> shift))
        return std::nullopt;
    return size_t(value) << shift;
}

std::optional<size_t> host_memory_limit_override()
{
    const char* text = std::getenv("HIPBLASLT_CLIENT_HOST_MEMORY_LIMIT");
    if(!text || !*text)
        return std::nullopt;
    auto                     limit = parse_host_memory_size(text);
    static std::atomic<bool> warned{false};
    if(!limit && !warned.exchange(true))
        hipblaslt_cerr << "warning: ignoring HIPBLASLT_CLIENT_HOST_MEMORY_LIMIT=" << text
                       << ", which is not a byte count such as 64G" << std::endl;
    return limit;
}

size_t available_host_memory_bytes(const HostMemorySources& sources)
{
    std::optional<size_t> available;
#ifdef _WIN32
    (void)sources;
    MEMORYSTATUSEX status = {};
    status.dwLength       = sizeof(status);
    if(GlobalMemoryStatusEx(&status))
        available = size_t(status.ullAvailPhys);
#else
    if(auto kib = read_field(sources.meminfo, "MemAvailable:"))
    {
        uint64_t bytes = *kib * 1024;
        if(auto headroom = smallest_cgroup_headroom(sources))
            bytes = std::min(bytes, *headroom);
        available = size_t(bytes);
    }
#endif
    // When the system cannot be read, the limit is the only budget there is.
    if(auto limit = host_memory_limit_override())
        return std::min(available.value_or(*limit), *limit);
    return available.value_or(0);
}
