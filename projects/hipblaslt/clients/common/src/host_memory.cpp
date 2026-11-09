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
    // cache the kernel can drop does not count as used: both file lists (active_file and
    // inactive_file; tmpfs and shared memory are not on them), less pages still waiting to be
    // written back. A pinned allocation reclaims active and inactive cache alike.
    std::optional<uint64_t> cgroup_headroom(const std::filesystem::path& dir, bool v2)
    {
        // v2 writes "max" for no limit, which is not a number. v1 writes a value near 2^63,
        // which MemAvailable is always below.
        auto limit = read_number(dir / (v2 ? "memory.max" : "memory.limit_in_bytes"));
        if(!limit)
            return std::nullopt;
        uint64_t used
            = read_number(dir / (v2 ? "memory.current" : "memory.usage_in_bytes")).value_or(0);
        auto stat = [&](const char* v2_key, const char* v1_key) {
            return read_field(dir / "memory.stat", v2 ? v2_key : v1_key).value_or(0);
        };
        uint64_t cache = stat("active_file", "total_active_file")
                         + stat("inactive_file", "total_inactive_file");
        uint64_t unwritten
            = stat("file_dirty", "total_dirty") + stat("file_writeback", "total_writeback");
        uint64_t reclaimable = cache > unwritten ? cache - unwritten : 0;
        used                 = used > reclaimable ? used - reclaimable : 0;
        return *limit > used ? *limit - used : uint64_t(0);
    }

    // Where a cgroup hierarchy is mounted (point), and which of its cgroups is at that point
    // (root). Outside a cgroup namespace a container sees the host's paths, and its mount may
    // start below the host's root.
    struct CgroupMount
    {
        std::filesystem::path root;
        std::filesystem::path point;
    };

    // mountinfo writes a space, tab, newline or backslash in a path as \ and three octal digits.
    std::string unescape_mount_path(const std::string& text)
    {
        std::string out;
        for(size_t i = 0; i < text.size(); i++)
        {
            if(text[i] == '\\' && i + 3 < text.size()
               && std::isdigit(static_cast<unsigned char>(text[i + 1]))
               && std::isdigit(static_cast<unsigned char>(text[i + 2]))
               && std::isdigit(static_cast<unsigned char>(text[i + 3])))
            {
                out += char((text[i + 1] - '0') * 64 + (text[i + 2] - '0') * 8
                            + (text[i + 3] - '0'));
                i += 3;
            }
            else
                out += text[i];
        }
        return out;
    }

    // The cgroup v2 mount, or the v1 mount that carries the memory controller, from
    // /proc/self/mountinfo. Each line is "id parent dev root point options ... - type source
    // super-options".
    std::optional<CgroupMount> find_cgroup_mount(const std::string& mountinfo, bool v2)
    {
        std::ifstream in(mountinfo);
        for(std::string line; std::getline(in, line);)
        {
            size_t dash = line.find(" - ");
            if(dash == std::string::npos)
                continue;
            std::istringstream before(line.substr(0, dash)), after(line.substr(dash + 3));
            std::string        id, parent, dev, root, point, type, source, options;
            if(!(before >> id >> parent >> dev >> root >> point) || !(after >> type >> source))
                continue;
            after >> options;
            bool              memory = false;
            std::stringstream list(options);
            for(std::string o; std::getline(list, o, ',');)
                memory = memory || o == "memory";
            if(v2 ? type == "cgroup2" : type == "cgroup" && memory)
                return CgroupMount{unescape_mount_path(root), unescape_mount_path(point)};
        }
        return std::nullopt;
    }

    // The smallest headroom from the process's cgroup up to the top of the hierarchy this
    // process can see. The cgroup's path is translated through its mount when mountinfo lists
    // one; otherwise the hierarchy is assumed at cgroup_root (v1: cgroup_root/memory). A
    // container may list "/" or the host's path for its cgroup; either way its own limit is at
    // the top it sees, so the walk always ends there.
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

            // A cgroup that has been removed is listed with this suffix.
            std::string       path    = line.substr(b + 1);
            const std::string deleted = " (deleted)";
            if(path.size() >= deleted.size()
               && path.compare(path.size() - deleted.size(), deleted.size(), deleted) == 0)
                path.erase(path.size() - deleted.size());

            std::filesystem::path top, rel;
            if(auto mount = find_cgroup_mount(s.mountinfo, v2))
            {
                // The part of the cgroup's path below the mount's root. A cgroup outside that
                // root (a container listing the host's path) leaves only the top visible.
                top      = mount->point;
                auto sub = std::filesystem::path(path).lexically_relative(mount->root);
                if(!sub.empty() && *sub.begin() != ".." && sub != ".")
                    rel = sub;
            }
            else
            {
                top = s.cgroup_root;
                if(v1)
                    top /= "memory";
                rel = std::filesystem::path(path).relative_path();
            }
            while(true)
            {
                if(auto h = cgroup_headroom(top / rel, v2))
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

std::optional<size_t> available_host_memory_bytes(const HostMemorySources& sources)
{
    std::optional<size_t> available;
#ifdef _WIN32
    (void)sources;
    MEMORYSTATUSEX status = {};
    status.dwLength       = sizeof(status);
    if(GlobalMemoryStatusEx(&status))
        available = size_t(status.ullAvailPhys);
#else
    // Each source that can be read caps the budget on its own.
    if(auto kib = read_field(sources.meminfo, "MemAvailable:"))
        available = size_t(*kib * 1024);
    if(auto headroom = smallest_cgroup_headroom(sources))
        available = std::min(available.value_or(size_t(*headroom)), size_t(*headroom));
#endif
    // When the system cannot be read, the limit is the only budget there is.
    if(auto limit = host_memory_limit_override())
        return std::min(available.value_or(*limit), *limit);
    return available;
}
