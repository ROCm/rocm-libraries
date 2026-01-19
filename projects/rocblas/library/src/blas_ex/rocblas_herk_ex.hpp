/* ************************************************************************
 * Copyright (C) 2016-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../blas3/rocblas_syrk_herk.hpp"
#include "handle.hpp"
#include "logging.hpp"

template <typename T>
inline size_t rocblas_internal_syrk_herk_ex_workspace(rocblas_handle handle,
                                                      rocblas_int    n,
                                                      rocblas_int    k,
                                                      rocblas_int    batch_count)
{
    size_t size = 1;

    //Allocating workspace memory when for high precision compute and result C
    if(n > 0 && batch_count > 0)
        size = int64_t(n) * (n) * sizeof(T) * batch_count;

    return size;
}

template <bool BATCHED>
rocblas_status rocblas_herk_ex_template(rocblas_handle    handle,
                                        rocblas_fill      uplo,
                                        rocblas_operation trans_a,
                                        rocblas_int       n,
                                        rocblas_int       k,
                                        const void*       alpha,
                                        const void*       a,
                                        rocblas_datatype  a_type,
                                        rocblas_stride    offsetAin,
                                        rocblas_int       lda,
                                        rocblas_stride    stride_a,
                                        const void*       beta,
                                        void*             c,
                                        rocblas_datatype  c_type,
                                        rocblas_stride    offsetCin,
                                        rocblas_int       ldc,
                                        rocblas_stride    stride_c,
                                        rocblas_datatype  compute_type,
                                        rocblas_int       batch_count = 1);

template <typename API_INT>
inline rocblas_status rocblas_herk_ex_arg_check(rocblas_handle    handle,
                                                rocblas_fill      uplo,
                                                rocblas_operation trans_a,
                                                API_INT           n,
                                                API_INT           k,
                                                const void*       alpha,
                                                const void*       a,
                                                rocblas_datatype  a_type,
                                                rocblas_stride    offsetAin,
                                                API_INT           lda,
                                                rocblas_stride    stride_a,
                                                const void*       beta,
                                                void*             c,
                                                rocblas_datatype  c_type,
                                                rocblas_stride    offsetCin,
                                                API_INT           ldc,
                                                rocblas_stride    stride_c,
                                                rocblas_datatype  compute_type,
                                                API_INT           batch_count = 1)
{
    // handle must be valid
    if(!handle)
        return rocblas_status_invalid_handle;

    if(trans_a != rocblas_operation_none && trans_a != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;

    // sizes must not be negative
    if(n < 0 || k < 0 || batch_count < 0)
        return rocblas_status_invalid_size;

    // leading dimensions must be valid
    if(ldc < n || lda < (trans_a == rocblas_operation_none ? n : k))
        return rocblas_status_invalid_size;

    if(!n || !batch_count)
        return rocblas_status_success;

    if(handle->is_device_memory_size_query())
        return rocblas_status_continue;

    // pointers must be valid
    if((k && !alpha) || !beta || !c)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}
