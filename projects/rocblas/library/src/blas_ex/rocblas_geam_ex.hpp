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

#pragma once

#include "handle.hpp"
#include "logging.hpp"

template <bool BATCHED = false>
rocblas_status rocblas_geam_ex_template(rocblas_handle            handle,
                                        rocblas_operation         trans_A,
                                        rocblas_operation         trans_B,
                                        rocblas_int               m,
                                        rocblas_int               n,
                                        rocblas_int               k,
                                        const void*               alpha,
                                        const void*               A,
                                        rocblas_datatype          A_type,
                                        rocblas_stride            offset_A,
                                        rocblas_int               lda,
                                        rocblas_stride            stride_A,
                                        const void*               B,
                                        rocblas_datatype          B_type,
                                        rocblas_stride            offset_B,
                                        rocblas_int               ldb,
                                        rocblas_stride            stride_B,
                                        const void*               beta,
                                        const void*               C,
                                        rocblas_datatype          C_type,
                                        rocblas_stride            offset_C,
                                        rocblas_int               ldc,
                                        rocblas_stride            stride_C,
                                        void*                     D,
                                        rocblas_datatype          D_type,
                                        rocblas_stride            offset_D,
                                        rocblas_int               ldd,
                                        rocblas_stride            stride_D,
                                        rocblas_int               batch_count,
                                        rocblas_datatype          compute_type,
                                        rocblas_geam_ex_operation geam_ex_op);

template <typename T>
rocblas_status rocblas_geam_ex_arg_check(rocblas_handle    handle,
                                         rocblas_operation trans_a,
                                         rocblas_operation trans_b,
                                         rocblas_int       m,
                                         rocblas_int       n,
                                         rocblas_int       k,
                                         const void*       alpha,
                                         const void*       A,
                                         rocblas_int       lda,
                                         const void*       B,
                                         rocblas_int       ldb,
                                         const void*       beta,
                                         const void*       C,
                                         rocblas_int       ldc,
                                         const void*       D,
                                         rocblas_int       ldd,
                                         rocblas_int       batch_count = 1)
{
    // handle must be valid
    if(!handle)
        return rocblas_status_invalid_handle;

    if(trans_a != rocblas_operation_none && trans_a != rocblas_operation_transpose)
        return rocblas_status_invalid_value;
    if(trans_b != rocblas_operation_none && trans_b != rocblas_operation_transpose)
        return rocblas_status_invalid_value;

    // sizes must not be negative
    if(m < 0 || n < 0 || k < 0 || batch_count < 0)
        return rocblas_status_invalid_size;

    // leading dimensions must be valid
    if(ldc < m || ldd < m || lda < (trans_a == rocblas_operation_none ? m : k)
       || ldb < (trans_b == rocblas_operation_none ? k : n))
        return rocblas_status_invalid_size;

    if(C == D && ldc != ldd)
        return rocblas_status_invalid_size;

    // quick return
    // Note: k == 0 is not a quick return
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    // pointers must be valid
    if(!beta || !D)
        return rocblas_status_invalid_pointer;

    if(k)
    {
        if(!alpha)
            return rocblas_status_invalid_pointer;

        if(handle->pointer_mode == rocblas_pointer_mode_host)
        {
            T alpha_f = *((T*)alpha);
            if(alpha_f && (!A || !B))
                return rocblas_status_invalid_pointer;
        }
    }

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        T beta_f = *((T*)beta);
        if(beta_f && !C)
            return rocblas_status_invalid_pointer;
    }

    return rocblas_status_continue;
}
