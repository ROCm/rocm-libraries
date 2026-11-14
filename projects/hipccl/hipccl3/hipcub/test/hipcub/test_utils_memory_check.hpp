// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef HIPCUB_TEST_HIPCUB_TEST_UTILS_MEMORY_CHECK_HPP_
#define HIPCUB_TEST_HIPCUB_TEST_UTILS_MEMORY_CHECK_HPP_

// Uses HIP_CHECK and test_common_utils::__get_env, so it must be included
// after common_test_header.hpp (which has no include guard).

#include <hip/hip_runtime.h>

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#else
    #include <fstream>
    #include <string>
#endif

#include <cstring>
#include <iostream>

namespace test_utils
{

// MemCheck checks whether a prospective allocation would exceed currently available
// memory, querying the OS on every call to account for other processes. This is
// needed on APU systems where CPU and GPU share a single memory pool, making it
// easy to exhaust memory with large test inputs. This is a port of the rocPRIM
// test helper of the same name (rocprim/test/rocprim/test_utils_memory_check.hpp).
//
// MemCheck is intended to be used in tests that loop over increasing test sizes,
// where it's OK to bail out and still report test success if the test can't be run
// for large test sizes.
//
// The alloc() functions are called before the actual memory allocation so the
// code can gracefully handle an out-of-memory situation:
//
//   for(auto size : sizes)
//   {
//       MemCheck memcheck;
//       MEMCHECK_OR_BREAK_ALLOC_HOST(type, size)
//       std::vector<type> host_vec(size);
//       MEMCHECK_OR_BREAK_ALLOC_DEVICE(type, size)
//       type* d_ptr;
//       HIP_CHECK(hipMalloc(&d_ptr, sizeof(type) * size));
//       // ... run test ...
//       HIP_CHECK(hipFree(d_ptr));
//   }
//
// On an integrated GPU (APU) device memory is carved out of system memory, so
// device allocations are also checked against the available host memory.
//
// Set the HIPCUB_MEMCHECK_LOGGING environment variable to 1 to enable diagnostic output.
class MemCheck
{
public:
    // padding_factor is a value in [0, 1] that indicates how much of a buffer we should leave
    // below the available memory.
    // i.e. when a prospective allocation >= free_memory * (1 - padding_factor), assume OOM.
    MemCheck(const float padding_factor = 0.1f) : padding_factor(padding_factor)
    {
        char* env       = test_common_utils::__get_env("HIPCUB_MEMCHECK_LOGGING");
        logging_enabled = (env != nullptr) && (strcmp(env, "1") == 0);
        test_common_utils::clean_env(env);
    }

    template<typename T>
    inline bool alloc_host(const size_t size)
    {
        return alloc_host_bytes(sizeof(T) * size);
    }

    inline bool alloc_host_bytes(const size_t bytes)
    {
        const size_t free_host = get_free_host_bytes();
        const bool   success   = fits(bytes, free_host);

        if(logging_enabled)
        {
            std::cout << "alloc_host_bytes: " << toMB(bytes)
                      << " MiB, free_host=" << toMB(free_host) << " MiB"
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
        const size_t free_dev = get_free_device_bytes();
        bool         success  = fits(bytes, free_dev);

        const bool integrated = is_integrated_device();
        size_t     free_host  = 0;
        if(integrated)
        {
            free_host = get_free_host_bytes();
            success   = success && fits(bytes, free_host);
        }

        if(logging_enabled)
        {
            std::cout << "alloc_device_bytes: " << toMB(bytes)
                      << " MiB, free_dev=" << toMB(free_dev) << " MiB";
            if(integrated)
            {
                std::cout << ", free_host=" << toMB(free_host) << " MiB";
            }
            std::cout << (success ? "" : " -- out of memory") << std::endl;
        }
        return success;
    }

private:
    bool fits(const size_t bytes, const size_t free_bytes) const
    {
        return bytes <= static_cast<size_t>(free_bytes * (1.0f - padding_factor));
    }

    static size_t toMB(size_t bytes)
    {
        return bytes >> 20;
    }

    static bool is_integrated_device()
    {
        int device_id;
        HIP_CHECK(hipGetDevice(&device_id));
        int integrated = 0;
        HIP_CHECK(hipDeviceGetAttribute(&integrated, hipDeviceAttributeIntegrated, device_id));
        return integrated != 0;
    }

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
            {
                return value * 1024; // kB -> bytes
            }
        }
        return 0;
#endif
    }

    bool  logging_enabled = false;
    float padding_factor;
};

// For brevity, these macros assume your MemCheck instance is named 'memcheck'
#define MEMCHECK_OR_BREAK_ALLOC_DEVICE(T, size) \
    if(!memcheck.alloc_device<T>(size))         \
    {                                           \
        break;                                  \
    }

#define MEMCHECK_OR_BREAK_ALLOC_DEVICE_BYTES(size) \
    if(!memcheck.alloc_device_bytes(size))         \
    {                                              \
        break;                                     \
    }

#define MEMCHECK_OR_BREAK_ALLOC_HOST(T, size) \
    if(!memcheck.alloc_host<T>(size))         \
    {                                         \
        break;                                \
    }

#define MEMCHECK_OR_BREAK_ALLOC_HOST_BYTES(size) \
    if(!memcheck.alloc_host_bytes(size))         \
    {                                            \
        break;                                   \
    }

} // namespace test_utils

#endif // HIPCUB_TEST_HIPCUB_TEST_UTILS_MEMORY_CHECK_HPP_
