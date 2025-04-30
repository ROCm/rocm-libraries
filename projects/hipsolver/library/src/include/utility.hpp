/* ************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 *
 * ************************************************************************ */

#pragma once

#include "hipsolver.h"

#include "rocblas/rocblas.h"
#include "rocsolver/rocsolver.h"

inline rocblas_status hipsolverManageWorkspace(rocblas_handle handle, size_t new_size)
{
    if(new_size < 0)
        return rocblas_status_memory_error;

    if(!rocblas_is_managing_device_memory(handle))
        return rocblas_set_workspace(handle, nullptr, 0);

    return rocblas_status_success;
}

inline rocblas_status
    hipsolverZeroInfo(rocblas_handle handle, rocblas_int* devInfo, rocblas_int batch_count)
{
    if(!handle)
        return rocblas_status_invalid_handle;
    if(!devInfo)
        return rocblas_status_invalid_pointer;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    if(hipMemsetAsync(devInfo, 0, sizeof(rocblas_int) * batch_count, stream) == hipSuccess)
        return rocblas_status_success;
    else
        return rocblas_status_internal_error;
}
