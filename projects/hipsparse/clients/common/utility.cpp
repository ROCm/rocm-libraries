/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include <limits>
#ifdef WIN32
#include <windows.h>
#endif
#include "utility.hpp"

#include <chrono>
#include <cstdlib>

#ifdef WIN32
#define strSUITEcmp(A, B) _stricmp(A, B)
#endif

#ifdef __cpp_lib_filesystem
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

/* ============================================================================================ */
// Return path of this executable
std::string hipsparse_exepath()
{
#ifdef WIN32
    std::vector<TCHAR> result(MAX_PATH + 1);
    // Ensure result is large enough to accomodate the path
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
#endif
}

/* ==================================================================================== */
// Return path where the test data file (hipsparse_test.data) is located
std::string hipsparse_datapath()
{
#ifdef WIN32
    fs::path        share_path = fs::path(hipsparse_exepath() + "../share/hipsparse/test");
    std::error_code ec;
    fs::path        path = fs::canonical(share_path, ec);
    if(!ec)
    {
        if(fs::exists(path, ec) && !ec)
        {
            path += path.empty() ? "" : "/";
            return path.string();
        }
    }
#else
    std::string pathstr;
    std::string share_path = hipsparse_exepath() + "../share/hipsparse/test";
    char*       path       = realpath(share_path.c_str(), 0);
    if(path != NULL)
    {
        pathstr = path;
        pathstr += "/";
        free(path);
        return pathstr;
    }
#endif

    return hipsparse_exepath();
}

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================================ */
/*  query for hipsparse version and git commit SHA-1. */
void query_version(char* version)
{
    int  hipsparse_ver;
    char hipsparse_rev[256];

    hipsparseStatus_t status;

    hipsparseHandle_t handle;
    status = hipsparseCreate(&handle);
    if(HIPSPARSE_STATUS_SUCCESS != status)
    {
        std::cerr << "The creation of the hipsparseHandle_t failed." << std::endl;
        throw(status);
    }

    status = hipsparseGetVersion(handle, &hipsparse_ver);
    if(HIPSPARSE_STATUS_SUCCESS != status)
    {
        std::cerr << "hipsparseGetVersion failed." << std::endl;
        throw(status);
    }

    status = hipsparseGetGitRevision(handle, hipsparse_rev);
    if(HIPSPARSE_STATUS_SUCCESS != status)
    {
        std::cerr << "hipsparseGetGitRevision failed." << std::endl;
        throw(status);
    }

    status = hipsparseDestroy(handle);

    if(HIPSPARSE_STATUS_SUCCESS != status)
    {
        std::cerr << "rocsparse_destroy_handle failed." << std::endl;
        throw(status);
    }

    snprintf(version,
             512,
             "v%d.%d.%d-%.256s",
             hipsparse_ver / 100000,
             hipsparse_ver / 100 % 1000,
             hipsparse_ver % 100,
             hipsparse_rev);
}

/* ============================================================================================ */
/*  device query and print out their ID and name; return number of compute-capable devices. */
int query_device_property()
{
    int device_count;
    {
        hipsparseStatus_t status = (hipsparseStatus_t)hipGetDeviceCount(&device_count);
        if(status != HIPSPARSE_STATUS_SUCCESS)
        {
            printf("Query device error: cannot get device count.\n");
            return -1;
        }
        else
        {
            printf("Query device success: there are %d devices\n", device_count);
        }
    }

    for(int i = 0; i < device_count; i++)
    {
        hipDeviceProp_t   props;
        hipsparseStatus_t status = (hipsparseStatus_t)hipGetDeviceProperties(&props, i);
        if(status != HIPSPARSE_STATUS_SUCCESS)
        {
            printf("Query device error: cannot get device ID %d's property\n", i);
        }
        else
        {
            printf("Device ID %d : %s\n", i, props.name);
            printf("-------------------------------------------------------------------------\n");
            printf("with %ldMB memory, clock rate %dMHz @ computing capability %d.%d \n",
                   props.totalGlobalMem >> 20,
                   (int)(props.clockRate / 1000),
                   props.major,
                   props.minor);
            printf("maxGridDimX %d, sharedMemPerBlock %ldKB, maxThreadsPerBlock %d, warpSize %d\n",
                   props.maxGridSize[0],
                   props.sharedMemPerBlock >> 10,
                   props.maxThreadsPerBlock,
                   props.warpSize);

            printf("-------------------------------------------------------------------------\n");
        }
    }

    return device_count;
}

/*  set current device to device_id */
void set_device(int device_id)
{
    hipsparseStatus_t status = (hipsparseStatus_t)hipSetDevice(device_id);
    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        printf("Set device error: cannot set device ID %d, there may not be such device ID\n",
               (int)device_id);
    }
}
/* ============================================================================================ */
/*  timing:*/

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and return wall time */
double get_time_us(void)
{
    std::ignore = hipDeviceSynchronize();
    auto now    = std::chrono::steady_clock::now();
    // struct timeval tv;
    // gettimeofday(&tv, NULL);
    //  return (tv.tv_sec * 1000 * 1000) + tv.tv_usec;
    auto duration
        = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    return (static_cast<double>(duration));

    // hipDeviceSynchronize();
    //struct timeval tv;
    //gettimeofday(&tv, NULL);
    //return (tv.tv_sec * 1000 * 1000) + tv.tv_usec;
};

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and return wall time */
double get_time_us_sync(hipStream_t stream)
{
    std::ignore = hipStreamSynchronize(stream);
    auto now    = std::chrono::steady_clock::now();

    // struct timeval tv;
    // gettimeofday(&tv, NULL);
    // return (tv.tv_sec * 1000 * 1000) + tv.tv_usec;
    auto duration
        = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    return (static_cast<double>(duration));
    // hipStreamSynchronize(stream);
    // struct timeval tv;
    // gettimeofday(&tv, NULL);
    // return (tv.tv_sec * 1000 * 1000) + tv.tv_usec;
};

#ifdef __cplusplus
}
#endif
