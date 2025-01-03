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

#include "device_properties.h"

#ifdef WIN32
#include <windows.h>
#else
#include <sys/sysinfo.h>
#endif

static constexpr size_t ONE_GiB = 1 << 30;

inline size_t bytes_to_GiB(const size_t bytes)
{
    return bytes == 0 ? 0 : (bytes - 1 + ONE_GiB) / ONE_GiB;
}

struct host_memory
{
public:
    host_memory()
    {
        update();
        set_limit_bytes(total_bytes);
    }

    size_t get_total_bytes()
    {
        return total_bytes;
    }

    size_t get_total_gbytes()
    {
        return bytes_to_GiB(get_total_bytes());
    }

    void set_limit_bytes(size_t limit_bytes_)
    {
        // Don't let limit use the total available memory, leave at
        // least a 1GiB buffer, otherwise process may get OOM killed.
        limit_bytes
            = limit_bytes_ > (total_bytes - ONE_GiB) ? (total_bytes - ONE_GiB) : limit_bytes_;
    }

    void set_limit_gbytes(size_t limit_gbytes_)
    {
        set_limit_bytes(limit_gbytes_ * ONE_GiB);
    }

    size_t get_usable_bytes()
    {
        update();

        // Limit the amount of usable memory. If we are too aggressive
        // with host memory usage, the host process may get OOM killed
        // on systems with little or no swap space.
        auto usable_bytes = free_bytes < ONE_GiB ? 0 : free_bytes;
        usable_bytes      = usable_bytes > limit_bytes ? limit_bytes : usable_bytes;

        return usable_bytes;
    }

    size_t get_usable_gbytes()
    {
        return bytes_to_GiB(get_usable_bytes());
    }

private:
    size_t total_bytes = 0;
    size_t free_bytes  = 0;
    size_t limit_bytes = 0;

    void update()
    {
#ifdef WIN32
        MEMORYSTATUSEX info;
        info.dwLength = sizeof(info);
        if(!GlobalMemoryStatusEx(&info))
            return;
        total_bytes = info.ullTotalPhys;
        free_bytes  = info.ullAvailPhys;
#else
        struct sysinfo info;
        if(sysinfo(&info) != 0)
            return;
        total_bytes = info.totalram * info.mem_unit;
        free_bytes  = info.freeram * info.mem_unit;

        // top-level memory cgroup may restrict this further

        // check cgroup v1
        std::ifstream memcg1_limit_file("/sys/fs/cgroup/memory/memory.limit_in_bytes");
        std::ifstream memcg1_usage_file("/sys/fs/cgroup/memory/memory.usage_in_bytes");
        size_t        memcg1_limit_bytes;
        size_t        memcg1_usage_bytes;
        // use cgroupv1 limit if we can read the cgroup files and it's
        // smaller
        if((memcg1_limit_file >> memcg1_limit_bytes) && (memcg1_usage_file >> memcg1_usage_bytes))
        {
            total_bytes = std::min<size_t>(total_bytes, memcg1_limit_bytes);
            free_bytes  = memcg1_limit_bytes - memcg1_usage_bytes;
        }

        // check cgroup v2
        std::ifstream memcg2_max_file("/sys/fs/cgroup/memory.max");
        std::ifstream memcg2_current_file("/sys/fs/cgroup/memory.current");
        size_t        memcg2_max_bytes;
        size_t        memcg2_current_bytes;
        // use cgroupv2 limit if we can read the cgroup files and it's
        // smaller
        if((memcg2_max_file >> memcg2_max_bytes) && (memcg2_current_file >> memcg2_current_bytes))
        {
            total_bytes = std::min<size_t>(total_bytes, memcg2_max_bytes);
            free_bytes  = memcg2_max_bytes - memcg2_current_bytes;
        }

#endif

        auto deviceProp = get_curr_device_prop();
        // on integrated APU, we can't expect to reuse "device" memory
        // for "host" things.
        if(deviceProp.integrated)
        {
            total_bytes -= deviceProp.totalGlobalMem;
        }
    }
};

inline static host_memory host_mem_info;

#endif // SYSMEM_H
