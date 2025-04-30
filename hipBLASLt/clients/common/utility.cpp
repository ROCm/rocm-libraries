/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "utility.hpp"
#include "d_vector.hpp"
#include <chrono>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <stdlib.h>

#include <fcntl.h>

#include "Tensile/Source/client/include/Utility.hpp"

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#error no filesystem found
#endif

/* ============================================================================================ */
// Return path of this executable
std::string hipblaslt_exepath()
{
    std::string pathstr;
    char*       path = realpath("/proc/self/exe", 0);
    if(path)
    {
        char* p = strrchr(path, '/');
        if(p)
        {
            p[1]    = 0;
            pathstr = path;
        }
        free(path);
    }
    return pathstr;
}

/* ============================================================================================ */
// Temp directory rooted random path
std::string hipblaslt_tempname()
{
    char tmp[] = "/tmp/hipblaslt-XXXXXX";
    int  fd    = mkostemp(tmp, O_CLOEXEC);
    if(fd == -1)
    {
        dprintf(STDERR_FILENO, "Cannot open temporary file: %m\n");
        exit(EXIT_FAILURE);
    }

    return std::string(tmp);
}

/* ============================================================================================ */
/*  memory allocation requirements :*/

/*! \brief Compute strided batched matrix allocation size allowing for strides smaller than full matrix */
size_t strided_batched_matrix_size(int rows, int cols, int lda, int64_t stride, int batch_count)
{
    size_t size = size_t(lda) * cols;
    if(batch_count > 1)
    {
        // for cases where batch_count strides may not exceed full matrix size use full matrix size
        // e.g. row walking a larger matrix we just use full matrix size
        size_t size_strides = (batch_count - 1) * stride;
        size += size < size_strides + (cols - 1) * size_t(lda) + rows ? size_strides : 0;
    }
    return size;
}

/* ============================================================================================ */
/*  timing:*/

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and return wall time */
double get_time_us_sync_device(void)
{
    if(hipDeviceSynchronize() != hipSuccess)
    {
        hipblaslt_cerr << "Synchronizing device failed" << std::endl;
    }

    auto now = std::chrono::steady_clock::now();
    // now.time_since_epoch() is the duration since epoch
    // which is converted to microseconds
    auto duration
        = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    return (static_cast<double>(duration));
};

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and return wall time */
double get_time_us_sync(hipStream_t stream)
{
    if(hipStreamSynchronize(stream) != hipSuccess)
    {
        hipblaslt_cerr << "Synchronizing stream failed" << std::endl;
    }

    auto now = std::chrono::steady_clock::now();
    // now.time_since_epoch() is the duration since epoch
    // which is converted to microseconds
    auto duration
        = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    return (static_cast<double>(duration));
};

/*! \brief  CPU Timer(in microsecond): no GPU synchronization */
double get_time_us_no_sync(void)
{
    auto now = std::chrono::steady_clock::now();
    // now.time_since_epoch() is the duration since epoch
    // which is converted to microseconds
    auto duration
        = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    return (static_cast<double>(duration));
};

/* ============================================================================================ */
/*  device query and print out their ID and name; return number of compute-capable devices. */
int64_t query_device_property(int device_id, hipDeviceProp_t &props)
{
    int             device_count;
    hipblasStatus_t status = (hipblasStatus_t)hipGetDeviceCount(&device_count);
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblaslt_cerr << "Query device error: cannot get device count" << std::endl;
        return -1;
    }
    else
    {
        hipblaslt_cout << "Query device success: there are " << device_count << " devices."
                       << " (Target device ID is " << device_id << ")" << std::endl;
    }

    if(device_id >= device_count)
    {
        hipblaslt_cerr << "Query device error: Target device ID is larger than device counts."
                       << std::endl;
        return device_count;
    }

    status = (hipblasStatus_t)hipGetDeviceProperties(&props, device_id);
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblaslt_cerr << "Query device error: cannot get device ID " << device_id << "'s property"
                       << std::endl;
    }
    else
    {
        char buf[320];
        snprintf(buf,
                 sizeof(buf),
                 "Device ID %d : %s %s\n"
                 "with %3.1f GB memory, max. SCLK %d MHz, max. MCLK %d MHz, compute capability "
                 "%d.%d\n"
                 "maxGridDimX %d, sharedMemPerBlock %3.1f KB, maxThreadsPerBlock %d, warpSize %d\n",
                 device_id,
                 props.name,
                 props.gcnArchName,
                 props.totalGlobalMem / 1e9,
                 (int)(props.clockRate / 1000),
                 (int)(props.memoryClockRate / 1000),
                 props.major,
                 props.minor,
                 props.maxGridSize[0],
                 props.sharedMemPerBlock / 1e3,
                 props.maxThreadsPerBlock,
                 props.warpSize);
        hipblaslt_cout << buf;
    }

    return device_count;
}

/*  set current device to device_id */
void set_device(int64_t device_id)
{
    hipblasStatus_t status = (hipblasStatus_t)hipSetDevice(device_id);
    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        hipblaslt_cerr << "Set device error: cannot set device ID " << device_id
                       << ", there may not be such device ID" << std::endl;
    }
}

/*****************
 * local handles *
 *****************/

hipblaslt_local_handle::hipblaslt_local_handle()
{
    auto status = hipblasLtCreate(&m_handle);
    if(status != HIPBLAS_STATUS_SUCCESS)
        throw std::runtime_error(hipblas_status_to_string(status));

#ifdef GOOGLE_TEST
    if(t_set_stream_callback)
    {
        (*t_set_stream_callback)(m_handle);
        t_set_stream_callback.reset();
    }
#endif
}

hipblaslt_local_handle::hipblaslt_local_handle(const Arguments& arg)
    : hipblaslt_local_handle()
{
    if(arg.tensile_solution_selection_method >= 0)
    {
        auto sol_selec_env = getenv("TENSILE_SOLUTION_SELECTION_METHOD");
        if(sol_selec_env)
            m_sol_selec_saved_status = std::string(sol_selec_env);
        m_sol_selec_env_set = true;
        setenv("TENSILE_SOLUTION_SELECTION_METHOD", std::to_string(arg.tensile_solution_selection_method).c_str(), true);
    }
    // memory guard control, with multi-threading should not change values across threads
    d_vector_set_pad_length(arg.pad);
}

hipblaslt_local_handle::~hipblaslt_local_handle()
{
    if(m_sol_selec_env_set)
    {
        setenv("TENSILE_SOLUTION_SELECTION_METHOD", m_sol_selec_saved_status.c_str(), true);
    }
    hipblasLtDestroy(m_handle);
}

std::vector<void*> benchmark_allocation()
{
    const char* env          = std::getenv("HIPBLASLT_BENCHMARK");
    bool        doAllocation = false;
    if(env)
        doAllocation = strtol(env, nullptr, 0) != 0;
    std::vector<void*> ptrs;
    if(doAllocation)
    {
        std::stringstream ss;
        ptrs = TensileLite::Client::benchmarkAllocation(ss);
        hipblaslt_cout << ss.str();
    }
    return ptrs;
}

int32_t hipblaslt_get_arch_major()
{
    int             deviceId;
    hipDeviceProp_t deviceProperties;

    auto removePrefix = [](const std::string& s) {
        size_t pos = s.find("gfx");
        if(pos != std::string::npos)
        {
            return s.substr(pos + 3);
        }
        return s;
    };

    static_cast<void>(hipGetDevice(&deviceId));
    static_cast<void>(hipGetDeviceProperties(&deviceProperties, deviceId));
    auto gpu_arch_no_prefix = removePrefix(deviceProperties.gcnArchName);
    return stoi(gpu_arch_no_prefix) / 100;
}

void hipblaslt_print_version()
{
    int                    version;
    char                   git_version[128];
    hipblaslt_local_handle handle;
    hipblasLtGetVersion(handle, &version);
    hipblasLtGetGitRevision(handle, &git_version[0]);
    hipblaslt_cout << "hipBLASLt version: " << version << std::endl;
    hipblaslt_cout << "hipBLASLt git version: " << git_version << std::endl;
}
