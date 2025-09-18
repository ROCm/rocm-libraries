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

#include "blas_ex/rocblas_gemmt.hpp"
#include "check_numerics_matrix.hpp"
#include "handle.hpp"
#include "int64_helpers.hpp"
#include "rocblas_gemmt_64.hpp"

template <typename API_INT, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_gemmt_launcher_64(rocblas_handle    handle,
                                                  rocblas_fill      uplo,
                                                  rocblas_operation trans_a,
                                                  rocblas_operation trans_b,
                                                  int64_t           n_64,
                                                  int64_t           k_64,
                                                  const TScal*      alpha,
                                                  TConstPtr         A,
                                                  int64_t           lda_64,
                                                  rocblas_stride    stride_a,
                                                  TConstPtr         B,
                                                  int64_t           ldb_64,
                                                  rocblas_stride    stride_b,
                                                  const TScal*      beta,
                                                  TPtr              C,
                                                  int64_t           ldc_64,
                                                  rocblas_stride    stride_c,
                                                  int64_t           batch_count_64)
{
    // quick return
    if(!n_64 || !k_64 || !batch_count_64)
        return rocblas_status_success;

    if(n_64 > c_i32_max)
        return rocblas_status_invalid_size; // defer adding new kernels as C is a n_64 * n_64  matrix and when n_64 > c_ILP64_i32_max it exceeds practical device memory.

    if(k_64 <= c_ILP64_i32_max && lda_64 <= c_ILP64_i32_max && ldb_64 <= c_ILP64_i32_max
       && ldc_64 <= c_ILP64_i32_max && batch_count_64 <= c_i64_grid_YZ_chunk)
    {
        return rocblas_internal_gemmt_launcher<rocblas_int>(handle,
                                                            uplo,
                                                            trans_a,
                                                            trans_b,
                                                            (rocblas_int)n_64,
                                                            k_64,
                                                            alpha,
                                                            A,
                                                            lda_64,
                                                            stride_a,
                                                            B,
                                                            ldb_64,
                                                            stride_b,
                                                            beta,
                                                            C,
                                                            ldc_64,
                                                            stride_c,
                                                            (rocblas_int)batch_count_64);
    }

    for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
    {
        int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

        auto A_ptr = adjust_ptr_batch(A, b_base, stride_a);
        auto B_ptr = adjust_ptr_batch(B, b_base, stride_b);
        auto C_ptr = adjust_ptr_batch(C, b_base, stride_c);

        rocblas_status status = rocblas_internal_gemmt_launcher<int64_t>(handle,
                                                                         uplo,
                                                                         trans_a,
                                                                         trans_b,
                                                                         rocblas_int(n_64),
                                                                         k_64,
                                                                         alpha,
                                                                         A_ptr,
                                                                         lda_64,
                                                                         stride_a,
                                                                         B_ptr,
                                                                         ldb_64,
                                                                         stride_b,
                                                                         beta,
                                                                         C_ptr,
                                                                         ldc_64,
                                                                         stride_c,
                                                                         batch_count);

        if(status != rocblas_status_success)
            return status;
    } // batch
    return rocblas_status_success;
}

#ifdef INSTANTIATE_GEMMT_LAUNCHER_64
#error INSTANTIATE_GEMMT_LAUNCHER_64 already defined
#endif

#define INSTANTIATE_GEMMT_LAUNCHER_64(TScal_, TConstPtr_, TPtr_)                \
    template rocblas_status                                                     \
        rocblas_internal_gemmt_launcher_64<int64_t, TScal_, TConstPtr_, TPtr_>( \
            rocblas_handle    handle,                                           \
            rocblas_fill      uplo,                                             \
            rocblas_operation transA,                                           \
            rocblas_operation transB,                                           \
            int64_t           n,                                                \
            int64_t           k,                                                \
            const TScal_*     alpha,                                            \
            TConstPtr_        dA_in,                                            \
            int64_t           lda,                                              \
            rocblas_stride    stride_a,                                         \
            TConstPtr_        dB_in,                                            \
            int64_t           ldb,                                              \
            rocblas_stride    stride_b,                                         \
            const TScal_*     beta,                                             \
            TPtr_             dC_in,                                            \
            int64_t           ldc,                                              \
            rocblas_stride    stride_c,                                         \
            int64_t           batch_count);

// non batched

INSTANTIATE_GEMMT_LAUNCHER_64(float, const float*, float*)
INSTANTIATE_GEMMT_LAUNCHER_64(double, const double*, double*)
INSTANTIATE_GEMMT_LAUNCHER_64(rocblas_float_complex,
                              const rocblas_float_complex*,
                              rocblas_float_complex*)
INSTANTIATE_GEMMT_LAUNCHER_64(rocblas_double_complex,
                              const rocblas_double_complex*,
                              rocblas_double_complex*)

// batched

INSTANTIATE_GEMMT_LAUNCHER_64(float, const float* const*, float* const*)
INSTANTIATE_GEMMT_LAUNCHER_64(double, const double* const*, double* const*)
INSTANTIATE_GEMMT_LAUNCHER_64(rocblas_float_complex,
                              const rocblas_float_complex* const*,
                              rocblas_float_complex* const*)
INSTANTIATE_GEMMT_LAUNCHER_64(rocblas_double_complex,
                              const rocblas_double_complex* const*,
                              rocblas_double_complex* const*)

#undef INSTANTIATE_GEMMT_LAUNCHER_64
