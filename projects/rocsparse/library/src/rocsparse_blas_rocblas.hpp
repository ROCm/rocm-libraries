/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#pragma once

#include "rocsparse_blas.h"

namespace rocsparse
{
    rocsparse_status blas_rocblas_create_handle(blas_rocblas_handle* handle);

    rocsparse_status blas_rocblas_destroy_handle(blas_rocblas_handle handle);

    rocsparse_status blas_rocblas_set_stream(blas_rocblas_handle handle, hipStream_t stream);

    rocsparse_status blas_rocblas_set_pointer_mode(blas_rocblas_handle    handle,
                                                   rocsparse_pointer_mode pointer_mode);

    rocsparse_status blas_rocblas_gemm_ex(blas_rocblas_handle handle,
                                          rocsparse_operation transA,
                                          rocsparse_operation transB,
                                          rocsparse_int       m,
                                          rocsparse_int       n,
                                          rocsparse_int       k,
                                          const void*         alpha,
                                          const void*         a,
                                          rocsparse_datatype  a_type,
                                          rocsparse_int       lda,
                                          const void*         b,
                                          rocsparse_datatype  b_type,
                                          rocsparse_int       ldb,
                                          const void*         beta,
                                          const void*         c,
                                          rocsparse_datatype  c_type,
                                          rocsparse_int       ldc,
                                          void*               d,
                                          rocsparse_datatype  d_type,
                                          rocsparse_int       ldd,
                                          rocsparse_datatype  compute_type,
                                          blas_gemm_alg       algo,
                                          int32_t             solution_index,
                                          uint32_t            flags);
}
