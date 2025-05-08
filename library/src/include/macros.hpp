/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <hip/hip_runtime.h>

/*******************************************************************************
 * Macros
 ******************************************************************************/

// note compiler differences may require variations in definitions
//#ifdef WIN32
//#else
//#endif

#define ROCBLAS_KERNEL(lb_) static __global__ __launch_bounds__((lb_)) void
#define ROCBLAS_KERNEL_NO_BOUNDS static __global__ void

// A storage-class-specifier other than thread_local shall not be specified in an explicit specialization.
// ROCBLAS_KERNEL_INSTANTIATE should be used where kernels are instantiated to avoid use of static.

#define ROCBLAS_KERNEL_INSTANTIATE __global__

#define ROCBLAS_KERNEL_ILF __device__ __attribute__((always_inline))

// we ignore pre-existing hipGetLastError as all internal hip calls should be guarded and so an external error
#define ROCBLAS_LAUNCH_KERNEL(...)                                                 \
    do                                                                             \
    {                                                                              \
        hipLaunchKernelGGL(__VA_ARGS__);                                           \
        hipError_t status = hipExtGetLastError();                                  \
        if(status != hipSuccess)                                                   \
            return rocblas_internal_convert_hip_to_rocblas_status_and_log(status); \
    } while(0)

#define ROCBLAS_LAUNCH_KERNEL_GRID(grid_, ...)                                         \
    do                                                                                 \
    {                                                                                  \
        if(grid_.x != 0 && grid_.y != 0 && grid_.z != 0)                               \
        {                                                                              \
            hipLaunchKernelGGL(__VA_ARGS__);                                           \
            hipError_t status = hipExtGetLastError();                                  \
            if(status != hipSuccess)                                                   \
                return rocblas_internal_convert_hip_to_rocblas_status_and_log(status); \
        }                                                                              \
    } while(0)
