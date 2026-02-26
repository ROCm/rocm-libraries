/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

/*! \file
 *  \brief rocsparse_memory_check.hpp provides utilities to detect available memory
 *  on host and device to filter tests based on memory requirements.
 */

#pragma once

#include <cstddef>
#include <hip/hip_runtime.h>
#include <iostream>
#include <mutex>

// For host memory detection on Linux
#ifdef __linux__
#include <sys/sysinfo.h>
#include <unistd.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

namespace rocsparse_memory_check
{
    // Detect available host memory by querying system APIs
    // Returns available memory in GB
    inline size_t detect_host_memory_gb()
    {
        constexpr size_t GB = 1024ULL * 1024ULL * 1024ULL;

#ifdef __linux__
        // Use sysinfo to get available memory on Linux
        struct sysinfo info;
        if(sysinfo(&info) == 0)
        {
            // info.freeram gives free RAM, info.bufferram gives buffer cache
            // For test filtering, use available memory (free + buffers + cached)
            // This is a more conservative estimate
            unsigned long available_bytes = info.freeram * info.mem_unit;
            size_t        available_gb    = available_bytes / GB;

            // Return at least 1GB even if system reports less
            return (available_gb > 0) ? available_gb : 1;
        }

        // Fallback: try to read /proc/meminfo for MemAvailable
        FILE* meminfo = fopen("/proc/meminfo", "r");
        if(meminfo != nullptr)
        {
            char line[256];
            while(fgets(line, sizeof(line), meminfo))
            {
                unsigned long mem_available_kb = 0;
                if(sscanf(line, "MemAvailable: %lu kB", &mem_available_kb) == 1)
                {
                    fclose(meminfo);
                    size_t available_gb = (mem_available_kb * 1024ULL) / GB;
                    return (available_gb > 0) ? available_gb : 1;
                }
            }
            fclose(meminfo);
        }
#elif defined(_WIN32)
        // Use GlobalMemoryStatusEx on Windows
        MEMORYSTATUSEX statex;
        statex.dwLength = sizeof(statex);
        if(GlobalMemoryStatusEx(&statex))
        {
            size_t available_gb = statex.ullAvailPhys / GB;
            return (available_gb > 0) ? available_gb : 1;
        }
#endif

        // Fallback: return 2GB as a safe default if we can't detect
        return 2;
    }

    // Detect available device memory by querying HIP memory info
    // Returns available memory in GB
    inline size_t detect_device_memory_gb()
    {
        constexpr size_t GB = 1024ULL * 1024ULL * 1024ULL;

        size_t free_mem  = 0;
        size_t total_mem = 0;

        // Query device memory using hipMemGetInfo
        hipError_t err = hipMemGetInfo(&free_mem, &total_mem);
        if(err == hipSuccess)
        {
            // Use free memory for conservative estimate
            size_t available_gb = free_mem / GB;
            return (available_gb > 0) ? available_gb : 1;
        }

        // If hipMemGetInfo fails, try to get device properties
        int device_id = 0;
        err           = hipGetDevice(&device_id);
        if(err == hipSuccess)
        {
            hipDeviceProp_t prop;
            err = hipGetDeviceProperties(&prop, device_id);
            if(err == hipSuccess)
            {
                // Use total memory as fallback (less conservative)
                size_t available_gb = prop.totalGlobalMem / GB;
                return (available_gb > 0) ? available_gb : 1;
            }
        }

        // Fallback: return 2GB as a safe default if we can't detect
        return 2;
    }

    // Global variables to cache detected memory (initialized on first use)
    // Use inline to avoid ODR violations across translation units
    inline size_t     g_available_host_memory_gb   = 0;
    inline size_t     g_available_device_memory_gb = 0;
    inline bool       g_memory_detected            = false;
    inline std::mutex g_memory_mutex;

    // Initialize memory detection (thread-safe, called once)
    inline void ensure_memory_detected()
    {
        if(!g_memory_detected)
        {
            // Creates a scoped lock guard that acquires the global memory mutex (g_memory_mutex)
            // upon construction and automatically releases it when the lock goes out of scope.
            // This ensures thread-safe access to shared memory resources by preventing
            // concurrent access from multiple threads. The RAII pattern guarantees the mutex
            // is unlocked even if an exception is thrown within the protected scope.
            std::lock_guard<std::mutex> lock(g_memory_mutex);
            // Double-check after acquiring lock
            if(!g_memory_detected)
            {
                g_available_host_memory_gb   = detect_host_memory_gb();
                g_available_device_memory_gb = detect_device_memory_gb();
                g_memory_detected            = true;

                // Only print if ROCSPARSE_VERBOSE_MEMORY is set
                const char* verbose_env = std::getenv("ROCSPARSE_VERBOSE_MEMORY");
                if(verbose_env != nullptr && verbose_env[0] != '0')
                {
                    std::cout << "========================================" << std::endl;
                    std::cout << "Memory Detection Results:" << std::endl;
                    std::cout << "  Available host memory:   " << g_available_host_memory_gb
                              << " GB" << std::endl;
                    std::cout << "  Available device memory: " << g_available_device_memory_gb
                              << " GB" << std::endl;
                    std::cout << "========================================" << std::endl;
                }
            }
        }
    }

    // Get available host memory in GB (cached after first call)
    inline size_t get_available_host_memory_gb()
    {
        ensure_memory_detected();
        return g_available_host_memory_gb;
    }

    // Get available device memory in GB (cached after first call)
    inline size_t get_available_device_memory_gb()
    {
        ensure_memory_detected();
        return g_available_device_memory_gb;
    }

} // namespace rocsparse_memory_check
