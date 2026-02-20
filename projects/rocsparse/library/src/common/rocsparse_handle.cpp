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

#include "rocsparse_handle.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_logging.hpp"
#include "rocsparse_utility.hpp"

#include "../conversion/rocsparse_convert_array.hpp"

#include <hip/hip_runtime.h>

ROCSPARSE_KERNEL(1) void init_kernel(){};

/*******************************************************************************
 * constructor
 ******************************************************************************/
_rocsparse_handle::_rocsparse_handle()
{
    ROCSPARSE_ROUTINE_TRACE;

    // Default device is active device
    THROW_IF_HIP_ERROR(hipGetDevice(&device));
    THROW_IF_HIP_ERROR(hipGetDeviceProperties(&properties, device));

    // Device wavefront size
    wavefront_size = properties.warpSize;

    // Shared memory per block opt-in
    shared_mem_per_block_optin = properties.sharedMemPerBlockOptin;

#if HIP_VERSION >= 307
    // ASIC revision
    asic_rev = properties.asicRevision;
#else
    asic_rev = 0;
#endif

    // Layer mode
    char* str_layer_mode;
    if((str_layer_mode = getenv("ROCSPARSE_LAYER")) == NULL)
    {
        layer_mode = rocsparse_layer_mode_none;
    }
    else
    {
        layer_mode = (rocsparse_layer_mode)(atoi(str_layer_mode));
    }

    // Obtain size for coomv device buffer
    rocsparse_int nthreads = properties.maxThreadsPerBlock;
    rocsparse_int nprocs   = 2 * properties.multiProcessorCount;
    rocsparse_int nblocks  = (nprocs * nthreads - 1) / 256 + 1;

    size_t coomv_size = (((sizeof(rocsparse_int) + 16) * nblocks - 1) / 256 + 1) * 256;

    // Allocate device buffer
    buffer_size = (coomv_size > 1024 * 1024) ? coomv_size : 1024 * 1024;
    THROW_IF_HIP_ERROR(rocsparse_hipMalloc(&buffer, buffer_size));

    // Device alpha and beta
    THROW_IF_HIP_ERROR(rocsparse_hipMalloc(&alpha, sizeof(double) * 2));
    THROW_IF_HIP_ERROR(rocsparse_hipMalloc(&beta, sizeof(double) * 2));

    // Device one
    THROW_IF_HIP_ERROR(rocsparse_hipMalloc(&sone, sizeof(float) * 2));
    THROW_IF_HIP_ERROR(rocsparse_hipMalloc(&done, sizeof(double) * 2));

    // Execute empty kernel for initialization

    THROW_WITH_MESSAGE_IF_HIP_ERROR(hipGetLastError(), "prior to hipLaunchKernelGGL");
    hipLaunchKernelGGL(init_kernel, dim3(1), dim3(1), 0, stream);
    THROW_WITH_MESSAGE_IF_HIP_ERROR(hipGetLastError(), "'empty kernel scheduling failed'");

    // Execute memset for initialization
    THROW_IF_HIP_ERROR(hipMemsetAsync(sone, 0, sizeof(float) * 2, stream));
    THROW_IF_HIP_ERROR(hipMemsetAsync(done, 0, sizeof(double) * 2, stream));

    const float  s_value = 1.0f;
    const double d_value = 1.0;
    THROW_IF_HIP_ERROR(
        hipMemcpyAsync(sone, &s_value, sizeof(float), hipMemcpyHostToDevice, stream));
    THROW_IF_HIP_ERROR(
        hipMemcpyAsync(done, &d_value, sizeof(double), hipMemcpyHostToDevice, stream));

    // Wait for device transfer to finish
    THROW_IF_HIP_ERROR(hipStreamSynchronize(stream));

    // create blas handle
    rocsparse::blas_impl blas_impl;

#ifdef ROCSPARSE_WITH_ROCBLAS

    blas_impl = rocsparse::blas_impl_rocblas;

#else

    //
    // Other implementation available? Otherwise, set it to none.
    //
    blas_impl = rocsparse::blas_impl_none;
#endif

    THROW_IF_ROCSPARSE_ERROR(rocsparse::blas_create_handle(&this->blas_handle, blas_impl));
    THROW_IF_ROCSPARSE_ERROR(rocsparse::blas_set_stream(this->blas_handle, this->stream));
    THROW_IF_ROCSPARSE_ERROR(
        rocsparse::blas_set_pointer_mode(this->blas_handle, this->pointer_mode));

    // Open log file
    if(layer_mode & rocsparse_layer_mode_log_trace)
    {
        rocsparse::open_log_stream(&log_trace_os, &log_trace_ofs, "ROCSPARSE_LOG_TRACE_PATH");
    }

    // Open log_bench file
    if(layer_mode & rocsparse_layer_mode_log_bench)
    {
        rocsparse::open_log_stream(&log_bench_os, &log_bench_ofs, "ROCSPARSE_LOG_BENCH_PATH");
    }

    // Open log_debug file
    if(layer_mode & rocsparse_layer_mode_log_debug)
    {
        rocsparse::open_log_stream(&log_debug_os, &log_debug_ofs, "ROCSPARSE_LOG_DEBUG_PATH");
    }
}

/*******************************************************************************
 * destructor
 ******************************************************************************/
_rocsparse_handle::~_rocsparse_handle()
{
    ROCSPARSE_ROUTINE_TRACE;

    // Due to the changes in the hipFree introduced in HIP 7.0
    // https://rocm.docs.amd.com/projects/HIP/en/latest/hip-7-changes.html#update-hipfree
    // we need to introduce a device synchronize here as the below hipFree calls are now asynchronous.
    // hipFree() previously had an implicit wait for synchronization purpose which is applicable for all memory allocations.
    // This wait has been disabled in the HIP 7.0 runtime for allocations made with hipMallocAsync and hipMallocFromPoolAsync.
    PRINT_IF_HIP_ERROR(hipDeviceSynchronize());

    PRINT_IF_HIP_ERROR(rocsparse_hipFree(buffer));
    PRINT_IF_HIP_ERROR(rocsparse_hipFree(sone));
    PRINT_IF_HIP_ERROR(rocsparse_hipFree(done));
    PRINT_IF_HIP_ERROR(rocsparse_hipFree(alpha));
    PRINT_IF_HIP_ERROR(rocsparse_hipFree(beta));

    // destroy blas handle
    rocsparse_status status = rocsparse::blas_destroy_handle(this->blas_handle);
    if(status != rocsparse_status_success)
    {
        ROCSPARSE_ERROR_MESSAGE(status, "handle error");
    }

    // Close log files
    if(log_trace_ofs.is_open())
    {
        log_trace_ofs.close();
    }
    if(log_bench_ofs.is_open())
    {
        log_bench_ofs.close();
    }
    if(log_debug_ofs.is_open())
    {
        log_debug_ofs.close();
    }
}

/*******************************************************************************
 * Exactly like cuSPARSE, rocSPARSE only uses one stream for one API routine
 ******************************************************************************/

/*******************************************************************************
 * set stream:
   This API assumes user has already created a valid stream
   Associate the following rocsparse API call with this user provided stream
 ******************************************************************************/
rocsparse_status _rocsparse_handle::set_stream(hipStream_t user_stream)
{
    ROCSPARSE_ROUTINE_TRACE;

    // TODO check if stream is valid
    stream = user_stream;

    // blas set stream
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::blas_set_stream(this->blas_handle, user_stream));

    return rocsparse_status_success;
}

/*******************************************************************************
 * get stream
 ******************************************************************************/
rocsparse_status _rocsparse_handle::get_stream(hipStream_t* user_stream) const
{
    ROCSPARSE_ROUTINE_TRACE;

    *user_stream = stream;
    return rocsparse_status_success;
}

/*******************************************************************************
 * set pointer mode
 ******************************************************************************/
rocsparse_status _rocsparse_handle::set_pointer_mode(rocsparse_pointer_mode user_mode)
{
    ROCSPARSE_ROUTINE_TRACE;

    // TODO check if stream is valid
    this->pointer_mode = user_mode;

    // blas set stream
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::blas_set_pointer_mode(this->blas_handle, user_mode));

    return rocsparse_status_success;
}

/*******************************************************************************
 * get pointer mode
 ******************************************************************************/
rocsparse_status _rocsparse_handle::get_pointer_mode(rocsparse_pointer_mode* user_mode) const
{
    ROCSPARSE_ROUTINE_TRACE;

    *user_mode = this->pointer_mode;
    return rocsparse_status_success;
}

// Emulate C++17 std::void_t
template <typename...>
using void_t = void;

// By default, use gcnArch converted to a string prepended by gfx
template <typename PROP, typename = void>
struct ArchName
{
    std::string operator()(const PROP& prop) const;
};

// If gcnArchName exists as a member, use it instead
template <typename PROP>
struct ArchName<PROP, void_t<decltype(PROP::gcnArchName)>>
{
    std::string operator()(const PROP& prop) const
    {
        // strip out xnack/ecc from name
        std::string gcnArchName(prop.gcnArchName);
        std::string gcnArch = gcnArchName.substr(0, gcnArchName.find(":"));
        return gcnArch;
    }
};

// If gcnArchName not present, no xnack mode
template <typename PROP, typename = void>
struct XnackMode
{
    std::string operator()(const PROP& prop) const
    {
        return "";
    }
};

// If gcnArchName exists as a member, use it
template <typename PROP>
struct XnackMode<PROP, void_t<decltype(PROP::gcnArchName)>>
{
    std::string operator()(const PROP& prop) const
    {
        // strip out xnack/ecc from name
        std::string gcnArchName(prop.gcnArchName);
        auto        loc = gcnArchName.find("xnack");
        std::string xnackMode;
        if(loc != std::string::npos)
        {
            xnackMode = gcnArchName.substr(loc, 6);
            // guard against missing +/- at end of xnack mode
            if(xnackMode.size() < 6)
                xnackMode = "";
        }
        return xnackMode;
    }
};

std::string rocsparse::handle_get_arch_name(rocsparse_handle handle)
{
    return ArchName<hipDeviceProp_t>{}(handle->properties);
}

std::string rocsparse::handle_get_xnack_mode(rocsparse_handle handle)
{
    return XnackMode<hipDeviceProp_t>{}(handle->properties);
}

// Convert the current C++ exception to rocsparse_status
// This allows extern "C" functions to return this function in a catch(...) block
// while converting all C++ exceptions to an equivalent rocsparse_status here
rocsparse_status rocsparse::exception_to_rocsparse_status(std::exception_ptr e)
try
{
    if(e)
        std::rethrow_exception(e);
    return rocsparse_status_success;
}
catch(const rocsparse_status& status)
{
    return status;
}
catch(const std::bad_alloc&)
{
    return rocsparse_status_memory_error;
}
catch(...)
{
    return rocsparse_status_thrown_exception;
}

/*******************************************************************************
 * \brief convert hipError_t to rocsparse_status
 * TODO - enumerate library calls to hip runtime, enumerate possible errors from
 * those calls
 ******************************************************************************/
rocsparse_status rocsparse::get_rocsparse_status_for_hip_status(hipError_t status)
{
    switch(status)
    {
    // success
    case hipSuccess:
        return rocsparse_status_success;

    // internal hip memory allocation
    case hipErrorMemoryAllocation:
    case hipErrorLaunchOutOfResources:
        return rocsparse_status_memory_error;

    // user-allocated hip memory
    case hipErrorInvalidDevicePointer: // hip memory
        return rocsparse_status_invalid_pointer;

    // user-allocated device, stream, event
    case hipErrorInvalidDevice:
    case hipErrorInvalidResourceHandle:
        return rocsparse_status_invalid_handle;

    // library using hip incorrectly
    case hipErrorInvalidValue:
        return rocsparse_status_internal_error;

    // hip runtime failing
    case hipErrorNoDevice: // no hip devices
    case hipErrorUnknown:
    default:
        return rocsparse_status_internal_error;
    }
}
