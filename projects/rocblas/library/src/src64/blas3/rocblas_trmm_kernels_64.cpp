/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "handle.hpp"
#include "int64_helpers.hpp"
#include "rocblas_block_sizes.h"

#include "rocblas_trmm_64.hpp"

#include "blas3/rocblas_trmm.hpp" // int32 API called

template <bool BATCHED, typename T, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_trmm_launcher_64(rocblas_handle    handle,
                                                 rocblas_side      side,
                                                 rocblas_fill      uplo,
                                                 rocblas_operation trans_a,
                                                 rocblas_diagonal  diag,
                                                 int64_t           m_64,
                                                 int64_t           n_64,
                                                 TScal*            alpha,
                                                 rocblas_stride    stride_alpha,
                                                 TConstPtr*        dA,
                                                 rocblas_stride    offset_a,
                                                 int64_t           lda_64,
                                                 rocblas_stride    stride_a,
                                                 TConstPtr*        dB,
                                                 rocblas_stride    offset_b,
                                                 int64_t           ldb_64,
                                                 rocblas_stride    stride_b,
                                                 TPtr*             dC,
                                                 rocblas_stride    offset_c,
                                                 int64_t           ldc_64,
                                                 rocblas_stride    stride_c,
                                                 int64_t           batch_count_64)
{
    // quick return
    if(!m_64 || !n_64 || !batch_count_64)
        return rocblas_status_success;

    int64_t k  = (side == rocblas_side_left) ? m_64 : n_64;
    int64_t k1 = (side == rocblas_side_left) ? n_64 : m_64;

    // Need reduction to chunk over K dimension, so for now not supporting K sizes greater
    // than X_chunk size (2^28), these sizes aren't feasible in memory anyway
    if(k > c_i64_grid_X_chunk)
    {
        // exceeds practical memory
        return rocblas_status_invalid_size;
    }

    constexpr int NB = rocblas_is_complex<T> ? ROCBLAS_CZTRMM_NB : ROCBLAS_SDTRMM_NB;

    for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
    {
        int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

        auto A_ptr = adjust_ptr_batch(dA, b_base, stride_a);
        auto B_ptr = adjust_ptr_batch(dB, b_base, stride_b);
        auto C_ptr = adjust_ptr_batch(dC, b_base, stride_c);

        // may be able to call 32-bit trmm while only iterating through batches.
        if(m_64 < c_ILP64_i32_max && n_64 < c_ILP64_i32_max)
        {
            RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_launcher<NB, BATCHED, T>(handle,
                                                                                    side,
                                                                                    uplo,
                                                                                    trans_a,
                                                                                    diag,
                                                                                    m_64,
                                                                                    n_64,
                                                                                    alpha,
                                                                                    stride_alpha,
                                                                                    A_ptr,
                                                                                    offset_a,
                                                                                    lda_64,
                                                                                    stride_a,
                                                                                    B_ptr,
                                                                                    offset_b,
                                                                                    ldb_64,
                                                                                    stride_b,
                                                                                    C_ptr,
                                                                                    offset_c,
                                                                                    ldc_64,
                                                                                    stride_c,
                                                                                    batch_count)));
        }
        else
        {
            // K is 32-bit size, only K1 may need chunking

            for(int64_t k1_base = 0; k1_base < k1; k1_base += c_i64_grid_X_chunk)
            {
                int32_t k1_32 = int32_t(std::min(k1 - k1_base, int64_t(c_i64_grid_X_chunk)));
                int32_t m     = side == rocblas_side_left ? int32_t(m_64) : k1_32;
                int32_t n     = side == rocblas_side_left ? k1_32 : int32_t(n_64);

                RETURN_IF_ROCBLAS_ERROR((rocblas_internal_trmm_launcher<NB, BATCHED, T>(
                    handle,
                    side,
                    uplo,
                    trans_a,
                    diag,
                    m,
                    n,
                    alpha,
                    stride_alpha,
                    A_ptr,
                    offset_a, // blocking over k1 doesn't affect A matrix
                    lda_64,
                    stride_a,
                    B_ptr,
                    offset_b + (k1_base * (side == rocblas_side_left ? ldb_64 : 1)),
                    ldb_64,
                    stride_b,
                    C_ptr,
                    offset_c + (k1_base * (side == rocblas_side_left ? ldc_64 : 1)),
                    ldc_64,
                    stride_c,
                    batch_count)));
            }
        }
    }

    return rocblas_status_success;
}

#define TRMM_TEMPLATE_PARAMS                                                                       \
    handle, side, uplo, trans_a, diag, m, n, alpha, stride_alpha, dA, offset_a, lda, stride_a, dB, \
        offset_b, ldb, stride_b, dC, offset_c, ldc, stride_c, batch_count

template <typename T>
rocblas_status rocblas_internal_trmm_template_64(rocblas_handle    handle,
                                                 rocblas_side      side,
                                                 rocblas_fill      uplo,
                                                 rocblas_operation trans_a,
                                                 rocblas_diagonal  diag,
                                                 int64_t           m,
                                                 int64_t           n,
                                                 const T*          alpha,
                                                 rocblas_stride    stride_alpha,
                                                 const T*          dA,
                                                 rocblas_stride    offset_a,
                                                 int64_t           lda,
                                                 rocblas_stride    stride_a,
                                                 const T*          dB,
                                                 rocblas_stride    offset_b,
                                                 int64_t           ldb,
                                                 rocblas_stride    stride_b,
                                                 T*                dC,
                                                 rocblas_stride    offset_c,
                                                 int64_t           ldc,
                                                 rocblas_stride    stride_c,
                                                 int64_t           batch_count)
{
    return rocblas_internal_trmm_launcher_64<false, T>(TRMM_TEMPLATE_PARAMS);
}

template <typename T>
rocblas_status rocblas_internal_trmm_batched_template_64(rocblas_handle    handle,
                                                         rocblas_side      side,
                                                         rocblas_fill      uplo,
                                                         rocblas_operation trans_a,
                                                         rocblas_diagonal  diag,
                                                         int64_t           m,
                                                         int64_t           n,
                                                         const T*          alpha,
                                                         rocblas_stride    stride_alpha,
                                                         const T* const*   dA,
                                                         rocblas_stride    offset_a,
                                                         int64_t           lda,
                                                         rocblas_stride    stride_a,
                                                         const T* const*   dB,
                                                         rocblas_stride    offset_b,
                                                         int64_t           ldb,
                                                         rocblas_stride    stride_b,
                                                         T* const*         dC,
                                                         rocblas_stride    offset_c,
                                                         int64_t           ldc,
                                                         rocblas_stride    stride_c,
                                                         int64_t           batch_count)
{
    return rocblas_internal_trmm_launcher_64<true, T>(TRMM_TEMPLATE_PARAMS);
}

#undef TRMM_TEMPLATE_PARAMS

#ifdef INST_TRMM_TEMPLATE_64
#error INST_TRMM_TEMPLATE_64 already defined
#endif

#define INST_TRMM_TEMPLATE_64(T_)                                                                 \
    template rocblas_status rocblas_internal_trmm_template_64<T_>(rocblas_handle    handle,       \
                                                                  rocblas_side      side,         \
                                                                  rocblas_fill      uplo,         \
                                                                  rocblas_operation trans_a,      \
                                                                  rocblas_diagonal  diag,         \
                                                                  int64_t           m,            \
                                                                  int64_t           n,            \
                                                                  const T_*         alpha,        \
                                                                  rocblas_stride    stride_alpha, \
                                                                  const T_*         dA,           \
                                                                  rocblas_stride    offset_a,     \
                                                                  int64_t           lda,          \
                                                                  rocblas_stride    stride_a,     \
                                                                  const T_*         dB,           \
                                                                  rocblas_stride    offset_b,     \
                                                                  int64_t           ldb,          \
                                                                  rocblas_stride    stride_b,     \
                                                                  T_*               dC,           \
                                                                  rocblas_stride    offset_c,     \
                                                                  int64_t           ldc,          \
                                                                  rocblas_stride    stride_c,     \
                                                                  int64_t           batch_count);

#ifdef INST_TRMM_BATCHED_TEMPLATE_64
#error INST_TRMM_BATCHED_TEMPLATE_64 already defined
#endif

#define INST_TRMM_BATCHED_TEMPLATE_64(T_)                                  \
    template rocblas_status rocblas_internal_trmm_batched_template_64<T_>( \
        rocblas_handle    handle,                                          \
        rocblas_side      side,                                            \
        rocblas_fill      uplo,                                            \
        rocblas_operation trans_a,                                         \
        rocblas_diagonal  diag,                                            \
        int64_t           m,                                               \
        int64_t           n,                                               \
        const T_*         alpha,                                           \
        rocblas_stride    stride_alpha,                                    \
        const T_* const*  dA,                                              \
        rocblas_stride    offset_a,                                        \
        int64_t           lda,                                             \
        rocblas_stride    stride_a,                                        \
        const T_* const*  dB,                                              \
        rocblas_stride    offset_b,                                        \
        int64_t           ldb,                                             \
        rocblas_stride    stride_b,                                        \
        T_* const*        dC,                                              \
        rocblas_stride    offset_c,                                        \
        int64_t           ldc,                                             \
        rocblas_stride    stride_c,                                        \
        int64_t           batch_count);

INST_TRMM_TEMPLATE_64(float)
INST_TRMM_TEMPLATE_64(double)
INST_TRMM_TEMPLATE_64(rocblas_float_complex)
INST_TRMM_TEMPLATE_64(rocblas_double_complex)

INST_TRMM_BATCHED_TEMPLATE_64(float)
INST_TRMM_BATCHED_TEMPLATE_64(double)
INST_TRMM_BATCHED_TEMPLATE_64(rocblas_float_complex)
INST_TRMM_BATCHED_TEMPLATE_64(rocblas_double_complex)
