/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022 Advanced Micro Devices, Inc.
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


/* ============================================================================================ */
// Return path of this executable
std::string hipsparselt_exepath()
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
std::string hipsparselt_tempname()
{
    char tmp[] = "/tmp/hipsparselt-XXXXXX";
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
        hipsparselt_cerr << "Synchronizing device failed" << std::endl;
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
    if(hipDeviceSynchronize() != hipSuccess)
    {
        hipsparselt_cerr << "Synchronizing device failed" << std::endl;
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
int64_t query_device_property()
{
    int               device_count;
    hipsparseStatus_t status = (hipsparseStatus_t)hipGetDeviceCount(&device_count);
    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        hipsparselt_cerr << "Query device error: cannot get device count" << std::endl;
        return -1;
    }
    else
    {
        hipsparselt_cout << "Query device success: there are " << device_count << " devices"
                         << std::endl;
    }

    for(int i = 0;; i++)
    {
        hipsparselt_cout
            << "-------------------------------------------------------------------------------"
            << std::endl;

        if(i >= device_count)
            break;

        hipDeviceProp_t   props;
        hipsparseStatus_t status = (hipsparseStatus_t)hipGetDeviceProperties(&props, i);
        if(status != HIPSPARSE_STATUS_SUCCESS)
        {
            hipsparselt_cerr << "Query device error: cannot get device ID " << i << "'s property"
                             << std::endl;
        }
        else
        {
            char buf[320];
            snprintf(
                buf,
                sizeof(buf),
                "Device ID %d : %s %s\n"
                "with %3.1f GB memory, max. SCLK %d MHz, max. MCLK %d MHz, compute capability "
                "%d.%d\n"
                "maxGridDimX %d, sharedMemPerBlock %3.1f KB, maxThreadsPerBlock %d, warpSize %d\n",
                i,
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
            hipsparselt_cout << buf;
        }
    }

    return device_count;
}

/*  set current device to device_id */
void set_device(int64_t device_id)
{
    hipsparseStatus_t status = (hipsparseStatus_t)hipSetDevice(device_id);
    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        hipsparselt_cerr << "Set device error: cannot set device ID " << device_id
                         << ", there may not be such device ID" << std::endl;
    }
}

/*****************
 * local handles *
 *****************/

hipsparselt_local_handle::hipsparselt_local_handle()
{
    auto status = hipsparseLtInit(&m_handle);
    if(status != HIPSPARSE_STATUS_SUCCESS)
        throw std::runtime_error(hipsparse_status_to_string(status));

#ifdef GOOGLE_TEST
    if(t_set_stream_callback)
    {
        (*t_set_stream_callback)(m_handle);
        t_set_stream_callback.reset();
    }
#endif
}

hipsparselt_local_handle::hipsparselt_local_handle(const Arguments& arg)
    : hipsparselt_local_handle()
{

    // If the test specifies user allocated workspace, allocate and use it
    if(arg.user_allocated_workspace)
    {
        if((hipMalloc)(&m_memory, arg.user_allocated_workspace) != hipSuccess)
            throw std::bad_alloc();
    }

    // memory guard control, with multi-threading should not change values across threads
    d_vector_set_pad_length(arg.pad);
}

hipsparselt_local_handle::~hipsparselt_local_handle()
{
    if(m_memory)
        if(hipFree(m_memory) != hipSuccess)
        {
            hipsparselt_cerr << "free device memory failed " << std::endl;
        }
    hipsparseLtDestroy(&m_handle);
}
