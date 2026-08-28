// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Host-only cgroup memory headroom probe; no GPU/HIP calls.

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <string>

namespace hipblaslt_client
{
    struct cgroup_paths
    {
        std::string v2; // relative path under /sys/fs/cgroup
        std::string v1; // relative path under /sys/fs/cgroup/memory
    };

    inline cgroup_paths parse_proc_self_cgroup(std::string const& content)
    {
        cgroup_paths paths;
        for(size_t at = 0; at <= content.size();)
        {
            auto end = content.find('\n', at);
            if(end == std::string::npos)
                end = content.size();
            std::string entry = content.substr(at, end - at);
            while(!entry.empty() && (entry.back() == '\n' || entry.back() == '\r'))
                entry.pop_back();
            if(!entry.empty())
            {
                if(entry.compare(0, 3, "0::") == 0)
                    paths.v2 = entry.substr(3);
                else
                {
                    // "<id>:<controllers>:<path>", where controllers may be co-mounted
                    // and comma-joined, so match a whole token rather than ":memory:".
                    auto ids = entry.find(':');
                    auto sep = entry.find(':', ids == std::string::npos ? 0 : ids + 1);
                    if(ids != std::string::npos && sep != std::string::npos)
                    {
                        std::string list = entry.substr(ids + 1, sep - ids - 1);
                        for(size_t token_at = 0; token_at <= list.size();)
                        {
                            auto token_end = list.find(',', token_at);
                            if(token_end == std::string::npos)
                                token_end = list.size();
                            if(list.compare(token_at, token_end - token_at, "memory") == 0)
                            {
                                paths.v1 = entry.substr(sep + 1);
                                break;
                            }
                            token_at = token_end + 1;
                        }
                    }
                }
            }
            at = end + 1;
        }
        return paths;
    }

    inline bool parse_cgroup_size_token(std::string const& token,
                                        size_t&            out,
                                        size_t             unlimited_sentinel)
    {
        if(token == "max")
        {
            out = std::numeric_limits<size_t>::max();
            return true;
        }
        char const*        cstr = token.c_str();
        char*              end  = nullptr;
        unsigned long long value = strtoull(cstr, &end, 10);
        if(end == cstr)
            return false;
        out = static_cast<size_t>(value);
        if(out >= unlimited_sentinel)
            out = std::numeric_limits<size_t>::max();
        return true;
    }

    inline bool read_cgroup_size_file(std::string const& path, size_t& out)
    {
        size_t const unlimited = static_cast<size_t>(1) << 62;

        FILE* f = fopen(path.c_str(), "r");
        if(!f)
            return false;
        char token[64] = {};
        bool ok        = fscanf(f, "%63s", token) == 1;
        fclose(f);
        if(!ok)
            return false;
        return parse_cgroup_size_token(token, out, unlimited);
    }

    namespace detail
    {
        using read_size_fn = bool (*)(std::string const& path, size_t& out, void* ctx);

        inline bool read_cgroup_size_file_ctx(std::string const& path, size_t& out, void*)
        {
            return read_cgroup_size_file(path, out);
        }

        inline bool read_fake_sysfs_ctx(std::string const&                       path,
                                        size_t&                                  out,
                                        void*                                    ctx)
        {
            size_t const unlimited = static_cast<size_t>(1) << 62;
            auto const&  fake      = *static_cast<std::map<std::string, std::string> const*>(ctx);
            auto         it        = fake.find(path);
            if(it == fake.end())
                return false;
            return parse_cgroup_size_token(it->second, out, unlimited);
        }

        inline size_t cgroup_headroom(cgroup_paths const& paths, read_size_fn read, void* ctx)
        {
            // A v1 limit reads as a huge sentinel rather than the word "max".
            size_t const unlimited = static_cast<size_t>(1) << 62;

            size_t available = std::numeric_limits<size_t>::max();

            auto consider = [&](std::string const& dir, char const* limit, char const* usage) {
                size_t cap = 0, used = 0;
                if(!read(dir + "/" + limit, cap, ctx) || cap >= unlimited)
                    return;
                if(!read(dir + "/" + usage, used, ctx))
                    used = 0;
                available = std::min(available, cap > used ? cap - used : static_cast<size_t>(0));
            };

            // A MemoryMax may sit on any ancestor, so the effective allowance is the tightest.
            auto walk = [&](std::string const& base,
                            std::string const& rel,
                            char const*        limit,
                            char const*        usage) {
                if(rel.empty())
                    return;
                std::string dir = base + rel;
                for(;;)
                {
                    consider(dir, limit, usage);
                    if(dir.size() <= base.size())
                        break;
                    auto slash = dir.rfind('/');
                    if(slash == std::string::npos || slash < base.size())
                        break;
                    dir.erase(slash);
                }
            };

            walk("/sys/fs/cgroup", paths.v2, "memory.max", "memory.current");
            walk("/sys/fs/cgroup/memory", paths.v1, "memory.limit_in_bytes", "memory.usage_in_bytes");
            return available;
        }
    } // namespace detail

    inline size_t cgroup_available_memory_from(cgroup_paths const& paths)
    {
        return detail::cgroup_headroom(paths, detail::read_cgroup_size_file_ctx, nullptr);
    }

    inline size_t cgroup_available_memory_from(cgroup_paths const&                     paths,
                                             std::map<std::string, std::string> const& fake_sysfs)
    {
        return detail::cgroup_headroom(
            paths, detail::read_fake_sysfs_ctx, const_cast<void*>(static_cast<void const*>(&fake_sysfs)));
    }

#ifdef __linux__
    inline size_t cgroup_available_memory_live()
    {
        std::string content;
        if(FILE* f = fopen("/proc/self/cgroup", "r"))
        {
            char line[512];
            while(fgets(line, sizeof(line), f))
                content += line;
            fclose(f);
        }

        return cgroup_available_memory_from(parse_proc_self_cgroup(content));
    }
#endif

} // namespace hipblaslt_client
