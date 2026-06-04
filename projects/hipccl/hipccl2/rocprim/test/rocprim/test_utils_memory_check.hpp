// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef TEST_TEST_UTILS_MEMORY_CHECK_HPP_
#define TEST_TEST_UTILS_MEMORY_CHECK_HPP_

#include <hip/hip_runtime.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <fstream>
#include <string>
#include <unistd.h>
#endif

// Needed for ROCPRIM_MEMCHECK_LOGGING
#include <iostream>

#include "../../common/utils.hpp"

namespace test_utils
{
// 32GB
constexpr static size_t minimum_memory_required_bytes = 34359738368;

inline unsigned long long get_total_system_memory(bool is_apu)
{
    unsigned long long total_system_memory = 0;
#ifdef _WIN32
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    GlobalMemoryStatusEx (&statex);
    total_system_memory = statex.ullTotalPhys;
#else
    std::ifstream meminfo("/proc/meminfo");
    std::string label;
    unsigned long long value;
    std::string unit;

    while (meminfo >> label >> value >> unit)
    {
        if (label == "MemTotal:")
        {
            total_system_memory = value;
        }

        // Stop once totalMem is found
        if (total_system_memory > 0) break;
    }
#endif
    if(is_apu)
    {
        size_t gpu_free_memory, gpu_total_memory;
        HIP_CHECK(hipMemGetInfo(&gpu_free_memory, &gpu_total_memory));

        // For APUs, OS will share up to half of visible system memory.
        // "Visible system memory" is total CPU RAM minus the carved-out
        // dedicated GPU memory.
        unsigned long long shared_gpu_memory = total_system_memory / 2;
        unsigned long long dedicated_gpu_memory = gpu_total_memory - shared_gpu_memory;
        total_system_memory = total_system_memory + dedicated_gpu_memory;
    }

    return total_system_memory;
}


// MemCheck checks whether a prospective allocation would exceed currently available
// memory, querying the OS on every call to account for other processes. This is
// needed on APU systems where CPU and GPU share a single memory pool, making it
// easy to exhaust memory with large test inputs.
//
// The alloc() functions are called before the actual memory allocation so the
// code can gracefully handle an out-of-memory situation.
//
// For tests with a single input size (skip the whole test):
//
//   MemCheck mem_check;
//   if(!mem_check.alloc_device_bytes(sizeof(type) * size)) GTEST_SKIP();
//
// For tests with multiple input sizes (skip the size or break out):
//
//   for(auto size : sizes)
//   {
//       MemCheck mem_check;
//       if(!mem_check.alloc_host_bytes(sizeof(type) * size)) continue;
//       std::vector<type> host_vec(size);
//       if(!mem_check.alloc_device_bytes(sizeof(type) * size)) continue;
//       type* d_ptr;
//       HIP_CHECK(hipMalloc(&d_ptr, sizeof(type) * size));
//       // ... run test ...
//       HIP_CHECK(hipFree(d_ptr));
//   }
//
// Set the ROCPRIM_MEMCHECK_LOGGING environment variable to enable diagnostic output.

class MemCheck
{
public:
    // padding_factor is a value in [0, 1] that indicates how much of a buffer we should leave
    // below the available memory.
    // i.e. when a prospective allocation >= free_memory * (1 - padding_factor), assume OOM.
    MemCheck(const hipStream_t stream = 0, const float padding_factor = 0.1f)
        : padding_factor(padding_factor)
    {
        char* env = common::__get_env("ROCPRIM_MEMCHECK_LOGGING");
        logging_enabled = (env != nullptr) && (strcmp(env, "1") == 0);
        common::clean_env(env);
    }

    template<typename T>
    inline bool alloc_host(const size_t size)
    {
        return alloc_host_bytes(sizeof(T) * size);
    }

    inline bool alloc_host_bytes(const size_t bytes)
    {
        size_t free_host = get_free_host_bytes();
        bool success = bytes <= static_cast<size_t>(free_host * (1.0f - padding_factor));

        if(logging_enabled)
        {
            std::cout << "alloc_host_bytes: " << toMB(bytes) << " MiB, free_host="
                      << toMB(free_host) << " MiB"
                      << (success ? "" : " -- out of memory") << std::endl;
        }
        return success;
    }

    template<typename T>
    inline bool alloc_device(const size_t size)
    {
        return alloc_device_bytes(sizeof(T) * size);
    }

    inline bool alloc_device_bytes(const size_t bytes)
    {
        size_t free_dev = get_free_device_bytes();
        bool success = bytes <= static_cast<size_t>(free_dev * (1.0f - padding_factor));

        if(logging_enabled)
        {
            std::cout << "alloc_device_bytes: " << toMB(bytes) << " MiB, free_dev="
                      << toMB(free_dev) << " MiB" << std::endl;
        }

        if(logging_enabled && !success)
        {
            std::cout << "alloc_device_bytes: out of memory" << std::endl;
        }
        return success;
    }

private:
    static size_t toMB(size_t bytes) { return bytes >> 20; }

    static size_t get_free_device_bytes()
    {
        size_t free_dev, total_dev;
        HIP_CHECK(hipMemGetInfo(&free_dev, &total_dev));
        return free_dev;
    }

    static size_t get_free_host_bytes()
    {
#ifdef _WIN32
        MEMORYSTATUSEX mem_status;
        mem_status.dwLength = sizeof(mem_status);
        GlobalMemoryStatusEx(&mem_status);
        return static_cast<size_t>(mem_status.ullAvailPhys);
#else
        // MemAvailable accounts for reclaimable page cache and slab memory,
        // so it more accurately reflects what the OS can give to a new allocation
        // than MemFree alone.
        std::ifstream meminfo("/proc/meminfo");
        std::string   label;
        size_t        value;
        std::string   unit;
        while(meminfo >> label >> value >> unit)
        {
            if(label == "MemAvailable:")
                return value * 1024; // kB -> bytes
        }
        return 0;
#endif
    }

    bool  logging_enabled = false;
    float padding_factor;
};

// For brevity, these macros assume your MemCheck instance is named 'memcheck'
#define MEMCHECK_OR_BREAK_ALLOC_DEVICE(T, size) \
    if (!memcheck.alloc_device<T>(size)) { break; }

#define MEMCHECK_OR_BREAK_ALLOC_DEVICE_BYTES(size) \
    if (!memcheck.alloc_device_bytes(size)) { break; }

#define MEMCHECK_OR_BREAK_ALLOC_HOST(T, size) \
    if (!memcheck.alloc_host<T>(size)) { break; }

#define MEMCHECK_OR_BREAK_ALLOC_HOST_BYTES(size) \
    if (!memcheck.alloc_host_bytes(size)) { break; }

}

#endif // TEST_TEST_UTILS_MEMORY_CHECK_HPP_
