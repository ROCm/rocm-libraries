/* ************************************************************************
 * Copyright (C) 2016-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include "utility.hpp"
#include "handle.hpp"
#include "rocblas_ostream.hpp"

/*******************************************************************************
 * \brief convert hipError_t to rocblas_status
 * TODO - enumerate library calls to hip runtime, enumerate possible errors from those calls
 ******************************************************************************/
rocblas_status rocblas_internal_convert_hip_to_rocblas_status(hipError_t status)
{
    switch(status)
    {
    // success
    case hipSuccess:
        return rocblas_status_success;

    // internal hip memory allocation
    case hipErrorMemoryAllocation:
    case hipErrorLaunchOutOfResources:
        return rocblas_status_memory_error;

    // user-allocated hip memory
    case hipErrorInvalidDevicePointer: // hip memory
        return rocblas_status_invalid_pointer;

    // user-allocated device, stream, event
    case hipErrorInvalidDevice:
    case hipErrorInvalidResourceHandle:
        return rocblas_status_invalid_handle;

    // library using hip incorrectly
    case hipErrorInvalidValue:
        return rocblas_status_internal_error;

    // hip catch find matching ISA
    case hipErrorNoBinaryForGpu:
        return rocblas_status_arch_mismatch;

    case hipErrorNotFound: // most likely tensile runtime error
    // hip runtime failing
    case hipErrorNoDevice: // no hip devices
    case hipErrorUnknown:
    default:
        return rocblas_status_internal_error;
    }
}

rocblas_status rocblas_internal_convert_hip_to_rocblas_status_and_log(hipError_t status)
{
    rocblas_status lib_status = rocblas_internal_convert_hip_to_rocblas_status(status);
    rocblas_cerr << "rocBLAS error from hip error code: '" << hipGetErrorName(status)
                 << "':" << status << std::endl;
    return lib_status;
}

template <typename T>
static const T rocblas_host_one = T(1);

__constant__ float                  rocblas_internal_device_one_s = 1;
__constant__ double                 rocblas_internal_device_one_d = 1;
__constant__ rocblas_float_complex  rocblas_internal_device_one_c{1, 0};
__constant__ rocblas_double_complex rocblas_internal_device_one_z{1, 0};

template <typename T>
rocblas_status rocblas_internal_get_one_ptr(rocblas_handle handle, const T*& one)
{
    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        one = &rocblas_host_one<T>;
        return rocblas_status_success;
    }

    void* ptr = nullptr;
    if constexpr(std::is_same_v<T, float>)
        RETURN_IF_HIP_ERROR(hipGetSymbolAddress(&ptr, rocblas_internal_device_one_s));
    else if constexpr(std::is_same_v<T, double>)
        RETURN_IF_HIP_ERROR(hipGetSymbolAddress(&ptr, rocblas_internal_device_one_d));
    else if constexpr(std::is_same_v<T, rocblas_float_complex>)
        RETURN_IF_HIP_ERROR(hipGetSymbolAddress(&ptr, rocblas_internal_device_one_c));
    else if constexpr(std::is_same_v<T, rocblas_double_complex>)
        RETURN_IF_HIP_ERROR(hipGetSymbolAddress(&ptr, rocblas_internal_device_one_z));

    one = static_cast<const T*>(ptr);
    return rocblas_status_success;
}

#define INSTANTIATE_GET_ONE_PTR(T_) \
    template rocblas_status rocblas_internal_get_one_ptr<T_>(rocblas_handle, const T_*&);

INSTANTIATE_GET_ONE_PTR(float)
INSTANTIATE_GET_ONE_PTR(double)
INSTANTIATE_GET_ONE_PTR(rocblas_float_complex)
INSTANTIATE_GET_ONE_PTR(rocblas_double_complex)

#undef INSTANTIATE_GET_ONE_PTR
