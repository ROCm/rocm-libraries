/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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
