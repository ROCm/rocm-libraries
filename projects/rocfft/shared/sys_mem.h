// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef SYSMEM_H
#define SYSMEM_H

#include <fstream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/sysinfo.h>
#endif

static constexpr size_t ONE_GiB = 1 << 30;
static constexpr size_t ONE_MiB = 1 << 20;
static constexpr size_t ONE_KiB = 1 << 10;

inline size_t bytes_to_GiB(const size_t bytes)
{
    return bytes == 0 ? 0 : (bytes - 1 + ONE_GiB) / ONE_GiB;
}

// returns a string reporting a byte size in a readable format,
// rounding it up to the nearest KiB (MiB if `byte_sz` exceeds 1 GiB)
static std::string byte_size_to_str(size_t byte_sz)
{
    if(byte_sz == 0)
        return "0 B";
    std::string                          ret;
    const std::pair<size_t, std::string> mem_units[]
        = {{ONE_GiB, "GiB"}, {ONE_MiB, "MiB"}, {ONE_KiB, "KiB"}};
    for(const auto& u : mem_units)
    {
        auto tmp = byte_sz / u.first;
        byte_sz -= tmp * u.first;
        if(byte_sz > 0 && (u.first == ONE_KiB || !ret.empty()))
        {
            // round up to nearest relevant unit and finish
            tmp++;
            byte_sz = 0;
        }
        if(tmp > 0)
        {
            if(!ret.empty())
                ret += " ";
            ret += std::to_string(tmp) + " " + u.second;
        }
    }
    return ret;
}

// return a string reporting a vec of byte sizes in a readable format.
static std::string byte_sizes_to_str(const std::vector<size_t>& byte_sizes)
{
    std::string ret;
    for(auto i : byte_sizes)
    {
        if(!ret.empty())
            ret.push_back(',');
        ret += byte_size_to_str(i);
    }
    return ret;
}

struct system_memory
{
public:
    // acquire a reference to a singleton of this struct
    static system_memory& singleton()
    {
        static system_memory mem;
        return mem;
    }

    size_t get_total_bytes() const
    {
        return total_bytes;
    }

    size_t get_total_gbytes() const
    {
        return bytes_to_GiB(get_total_bytes());
    }

    void set_limit_bytes(size_t limit_bytes_)
    {
        std::unique_lock lock(sys_memory_mutex);
        // Don't let limit use the total available memory, leave at
        // least a 1GiB buffer, otherwise process may get OOM killed.
        limit_bytes = std::min(limit_bytes_, total_bytes >= ONE_GiB ? total_bytes - ONE_GiB : 0);
    }

    void set_limit_gbytes(size_t limit_gbytes_)
    {
        set_limit_bytes(limit_gbytes_ * ONE_GiB);
    }

    size_t get_usable_bytes()
    {
        update_free_bytes();

        // Limit the amount of usable memory. If we are too aggressive
        // with host memory usage, the host process may get OOM killed
        // on systems with little or no swap space.
        std::shared_lock lock(sys_memory_mutex);
        return std::min(free_bytes < ONE_GiB ? 0 : free_bytes,
                        used_bytes < limit_bytes ? limit_bytes - used_bytes : 0);
    }

    size_t get_limit_bytes() const
    {
        std::shared_lock lock(sys_memory_mutex);
        return limit_bytes;
    }

    size_t get_usable_gbytes()
    {
        return bytes_to_GiB(get_usable_bytes());
    }

    void record_used_bytes(size_t allocation_size)
    {
        std::unique_lock lock(sys_memory_mutex);
        used_bytes += allocation_size;
    }
    void release_used_bytes(size_t allocation_size)
    {
        std::unique_lock lock(sys_memory_mutex);
        // prevent underflows
        used_bytes -= std::min(allocation_size, used_bytes);
    }

    std::string get_details(bool double_tab = false)
    {
        const auto        usable_bytes = get_usable_bytes();
        std::shared_lock  lock(sys_memory_mutex);
        std::stringstream ss;
        const auto        incr = (double_tab ? "\t\t" : "\t");
        ss << incr << "Usable system memory: " << byte_size_to_str(usable_bytes) << "\n"
           << incr << "Free system memory: " << byte_size_to_str(free_bytes) << "\n"
           << incr << "Used system memory: " << byte_size_to_str(used_bytes) << "\n"
           << incr << "Enforced limit on memory usage: " << byte_size_to_str(limit_bytes);
        return ss.str();
    }

private:
    const size_t total_bytes;
    size_t       free_bytes;
    size_t       limit_bytes;
    size_t       used_bytes;

    mutable std::shared_mutex sys_memory_mutex;

    system_memory()
        : total_bytes(read_sys_mem<sys_mem_label::TOTAL>())
        , free_bytes(read_sys_mem<sys_mem_label::FREE>())
        , limit_bytes(total_bytes)
        , used_bytes(0)
    {
    }

    void update_free_bytes()
    {
        std::unique_lock lock(sys_memory_mutex);
        free_bytes = read_sys_mem<sys_mem_label::FREE>();
    }

    enum class sys_mem_label
    {
        TOTAL,
        FREE
    };

    template <sys_mem_label mem_label>
    static size_t read_sys_mem()
    {
        static_assert(mem_label == sys_mem_label::TOTAL || mem_label == sys_mem_label::FREE);
        size_t ret = 0;
#ifdef _WIN32
        MEMORYSTATUSEX info;
        info.dwLength = sizeof(info);
        if(!GlobalMemoryStatusEx(&info))
            return ret;
        if constexpr(mem_label == sys_mem_label::TOTAL)
            ret = info.ullTotalPhys;
        else
            ret = info.ullAvailPhys;
#else
        // Read from /proc/meminfo if possible
        try
        {
            std::ifstream proc_meminfo("/proc/meminfo");
            std::string   proc_line;
            while(std::getline(proc_meminfo, proc_line))
            {
                // meminfo counts in KiB
                if constexpr(mem_label == sys_mem_label::TOTAL)
                {
                    if(proc_line.compare(0, 9, "MemTotal:") == 0)
                        ret = std::stoull(proc_line.substr(9)) * 1024;
                }
                else
                {
                    if(proc_line.compare(0, 13, "MemAvailable:") == 0)
                        ret = std::stoull(proc_line.substr(13)) * 1024;
                }
                if(ret)
                    break;
            }
        }
        catch(std::exception&)
        {
            // If there was a problem, we'll fall back to sysinfo
        }

        if(ret == 0)
        {
            // Either something couldn't be parsed or proc is not
            // mounted.  Fall back to sysinfo which can tell total +
            // free, but not "available".
            struct sysinfo info;
            if(sysinfo(&info) != 0)
                return ret;
            if constexpr(mem_label == sys_mem_label::TOTAL)
                ret = info.totalram * info.mem_unit;
            else
                ret = (info.freeram + info.bufferram) * info.mem_unit;
        }

        // top-level memory cgroup may restrict this further
        const std::pair<std::string, std::string> memcg1_paths
            = {"/sys/fs/cgroup/memory/memory.limit_in_bytes",
               "/sys/fs/cgroup/memory/memory.usage_in_bytes"};
        const std::pair<std::string, std::string> memcg2_paths
            = {"/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"};
        for(const auto& memcg_paths : {memcg1_paths, memcg2_paths})
        {
            std::ifstream memcg_limit_file(memcg_paths.first.c_str());
            std::ifstream memcg_usage_file(memcg_paths.second.c_str());
            size_t        memcg_limit_bytes;
            size_t        memcg_usage_bytes;
            // use cgroup limit if we can read the cgroup files and it's smaller
            if((memcg_limit_file >> memcg_limit_bytes) && (memcg_usage_file >> memcg_usage_bytes))
            {
                if constexpr(mem_label == sys_mem_label::TOTAL)
                    ret = std::min(ret, memcg_limit_bytes);
                else
                    ret = std::min(ret,
                                   memcg_usage_bytes <= memcg_limit_bytes
                                       ? memcg_limit_bytes - memcg_usage_bytes
                                       : 0);
            }
        }
#endif
        return ret;
    }
};

#endif // SYSMEM_H
