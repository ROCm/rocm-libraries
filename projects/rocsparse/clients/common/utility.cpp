/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#ifdef WIN32
#include <windows.h>
#endif

#include "rocsparse_clients_envariables.hpp"
#include "utility.hpp"

#include <chrono>
#include <cstdlib>

#ifdef WIN32
#define strcasecmp(A, B) _stricmp(A, B)

#ifdef __cpp_lib_filesystem
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#else
#include <fcntl.h>
#endif

/* ============================================================================================ */
// Return path of this executable
std::string rocsparse_exepath()
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
// Return path where the rocsparse_gentest.py file is located
std::string rocsparse_gentestpath()
{
#ifdef WIN32
    fs::path        share_path = fs::path(rocsparse_exepath() + "../libexec/rocsparse/test");
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
    std::string share_path = rocsparse_exepath() + "../libexec/rocsparse/test";
    char*       path       = realpath(share_path.c_str(), 0);
    if(path != NULL)
    {
        pathstr = path;
        pathstr += "/";
        free(path);
        return pathstr;
    }
#endif

    return rocsparse_exepath();
}

/* ==================================================================================== */
// Return path where the test data file (rocsparse_test.data) is located
std::string rocsparse_datapath()
{
    // first check an environment variable
    if(rocsparse_clients_envariables::is_defined(rocsparse_clients_envariables::TEST_DATA_DIR))
    {
        return rocsparse_clients_envariables::get(rocsparse_clients_envariables::TEST_DATA_DIR);
    }

#ifdef WIN32
    fs::path        share_path = fs::path(rocsparse_exepath() + "../share/rocsparse/test");
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
    std::string share_path = rocsparse_exepath() + "../share/rocsparse/test";
    char*       path       = realpath(share_path.c_str(), 0);
    if(path != NULL)
    {
        pathstr = path;
        pathstr += "/";
        free(path);
        return pathstr;
    }
#endif

    return rocsparse_exepath();
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
};

rocsparse_clients::timer::timer(hipStream_t stream)
    : m_stream(stream)
{
    auto status = hipEventCreate(&this->m_start);
    if(status != hipSuccess)
    {
        throw status;
    }
    status = hipEventCreate(&this->m_stop);
    if(status != hipSuccess)
    {
        throw status;
    }
}

void rocsparse_clients::timer::start()
{
    CHECK_HIP_ERROR(hipEventRecord(this->m_start, this->m_stream));
}

float rocsparse_clients::timer::stop()
{
    float time;
    auto  status = hipEventRecord(this->m_stop, this->m_stream);
    if(status != hipSuccess)
    {
        throw status;
    }
    status = hipEventSynchronize(this->m_stop);
    if(status != hipSuccess)
    {
        throw status;
    }
    status = hipEventElapsedTime(&time, this->m_start, this->m_stop);
    if(status != hipSuccess)
    {
        throw status;
    }
    return time;
}

rocsparse_clients::timer::~timer()
{
    std::ignore = hipEventDestroy(this->m_start);
    std::ignore = hipEventDestroy(this->m_stop);
}
