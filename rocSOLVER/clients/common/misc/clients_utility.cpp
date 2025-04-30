/* **************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#include <cstdlib>
#include <cstring>
#include <string>

#include <fmt/core.h>
#include <rocblas/rocblas.h>

#include "clients_utility.hpp"
#include "rocblas_random.hpp"

#ifdef _WIN32

#include <windows.h>

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#else
#include <fcntl.h>
#endif

// Random number generator
// Note: We do not use random_device to initialize the RNG, because we want
// repeatability in case of test failure. TODO: Add seed as an optional CLI
// argument, and print the seed on output, to ensure repeatability.
const rocblas_rng_t rocblas_seed(69069); // A fixed seed to start at

// This records the main thread ID at startup
const std::thread::id main_thread_id = std::this_thread::get_id();

// For the main thread, we use rocblas_seed; for other threads, we start with a
// different seed but deterministically based on the thread id's hash function.
thread_local rocblas_rng_t rocblas_rng = get_seed();

// Return the path to the currently running executable
std::string rocsolver_exepath()
{
#ifdef _WIN32

    std::vector<TCHAR> result(MAX_PATH + 1);
    DWORD length = 0;
    for(;;)
    {
        length = GetModuleFileNameA(nullptr, result.data(), result.size());
        if(length < result.size() - 1)
        {
            result.resize(length + 1);
            break;
        }
        result.resize(result.size() * 2);
    }

    fs::path exepath(result.begin(), result.end());
    exepath = exepath.remove_filename();
    exepath += exepath.empty() ? "" : "/";
    return exepath.string();

#else
    std::string pathstr;
    char* path = realpath("/proc/self/exe", 0);
    if(path)
    {
        char* p = strrchr(path, '/');
        if(p)
        {
            p[1] = 0;
            pathstr = path;
        }
        free(path);
    }
    return pathstr;
#endif
}

/* ============================================================================================
 */
/*  device query and print out their ID and name; return number of
 * compute-capable devices. */
rocblas_int query_device_property()
{
    int device_count = 0;
    hipError_t status = hipGetDeviceCount(&device_count);
    if(status != hipSuccess)
    {
        fmt::print(stderr, "Query device error: cannot get device count\n");
        return -1;
    }
    fmt::print("Query device success: there are {} devices\n", device_count);

    for(int i = 0;; i++)
    {
        fmt::print("{:-<79}\n", ""); // horizontal rule
        if(i >= device_count)
            break;

        hipDeviceProp_t props;
        status = hipGetDeviceProperties(&props, i);
        if(status != hipSuccess)
        {
            fmt::print(stderr, "Query device error: cannot get device ID {}'s property\n", i);
            continue;
        }

        fmt::print("Device ID {} : {}\nwith {:3.1f} GB memory, max. SCLK {} MHz, "
                   "max. MCLK {} MHz, compute capability {}.{}\nmaxGridDimX {}, "
                   "sharedMemPerBlock {:3.1f} KB, maxThreadsPerBlock {}, warpSize {}\n",
                   i, props.name, props.totalGlobalMem / 1e9, int(props.clockRate / 1000),
                   int(props.memoryClockRate / 1000), props.major, props.minor, props.maxGridSize[0],
                   props.sharedMemPerBlock / 1e3, props.maxThreadsPerBlock, props.warpSize);
    }
    return device_count;
}

/*  set current device to device_id */
void set_device(rocblas_int device_id)
{
    hipError_t status = hipSetDevice(device_id);
    if(status != hipSuccess)
        fmt::print(stderr, "Set device error: cannot set device ID {}\n", device_id);
}
