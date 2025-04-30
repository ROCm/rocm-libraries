/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <cstdlib>
#include <cstring>
#include <string>

#include "hipsolver.h"

#include "../rocblascommon/clients_utility.hpp"

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
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

// Return the path to the currently running executable
std::string hipsolver_exepath()
{
#ifdef _WIN32

    std::vector<TCHAR> result(MAX_PATH + 1);
    DWORD              length = 0;
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

/* ============================================================================================ */
/*  device query and print out their ID and name; return number of compute-capable devices. */
int query_device_property()
{
    int               device_count;
    hipsolverStatus_t count_status = (hipsolverStatus_t)hipGetDeviceCount(&device_count);
    if(count_status != HIPSOLVER_STATUS_SUCCESS)
    {
        printf("Query device error: cannot get device count \n");
        return -1;
    }
    else
    {
        printf("Query device success: there are %d devices \n", device_count);
    }

    for(int i = 0; i < device_count; i++)
    {
        hipDeviceProp_t   props;
        hipsolverStatus_t props_status = (hipsolverStatus_t)hipGetDeviceProperties(&props, i);
        if(props_status != HIPSOLVER_STATUS_SUCCESS)
        {
            printf("Query device error: cannot get device ID %d's property\n", i);
        }
        else
        {
            printf("Device ID %d : %s ------------------------------------------------------\n",
                   i,
                   props.name);
            printf("with %3.1f GB memory, clock rate %dMHz @ computing capability %d.%d \n",
                   props.totalGlobalMem / 1e9,
                   (int)(props.clockRate / 1000),
                   props.major,
                   props.minor);
            printf(
                "maxGridDimX %d, sharedMemPerBlock %3.1f KB, maxThreadsPerBlock %d, warpSize %d\n",
                props.maxGridSize[0],
                props.sharedMemPerBlock / 1e3,
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
    hipsolverStatus_t status = (hipsolverStatus_t)hipSetDevice(device_id);
    if(status != HIPSOLVER_STATUS_SUCCESS)
    {
        printf("Set device error: cannot set device ID %d, there may not be such device ID\n",
               (int)device_id);
    }
}
