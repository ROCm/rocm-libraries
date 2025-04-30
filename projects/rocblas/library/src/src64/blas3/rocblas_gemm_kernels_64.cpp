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

#include "rocblas_gemm_64.hpp"

#include "blas3/rocblas_gemm.hpp" // int32 API called
#include "blas3/rocblas_gemm_source.hpp"

template <bool BATCHED, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_gemm_64(rocblas_handle    handle,
                                        rocblas_operation trans_a,
                                        rocblas_operation trans_b,
                                        int64_t           m_64,
                                        int64_t           n_64,
                                        int64_t           k_64,
                                        const TScal*      alpha,
                                        TConstPtr         A,
                                        rocblas_stride    offset_a,
                                        int64_t           lda_64,
                                        rocblas_stride    stride_a,
                                        TConstPtr         B,
                                        rocblas_stride    offset_b,
                                        int64_t           ldb_64,
                                        rocblas_stride    stride_b,
                                        const TScal*      beta,
                                        TPtr              C,
                                        rocblas_stride    offset_c,
                                        int64_t           ldc_64,
                                        rocblas_stride    stride_c,
                                        int64_t           batch_count_64)
{
    // Note: k==0 is not a quick return, because C must still be multiplied by beta
    if(!m_64 || !n_64 || !batch_count_64)
        return rocblas_status_success;

    bool dims_32bit = m_64 <= c_ILP64_i32_max && n_64 <= c_ILP64_i32_max && k_64 <= c_ILP64_i32_max;
    bool leading_32bit = lda_64 <= c_i32_max && ldb_64 <= c_i32_max && ldc_64 <= c_i32_max;

    rocblas_status status = rocblas_status_success;

    if(dims_32bit && leading_32bit)
    {
        for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
        {
            auto A_ptr = adjust_ptr_batch(A, b_base, stride_a);
            auto B_ptr = adjust_ptr_batch(B, b_base, stride_b);
            auto C_ptr = adjust_ptr_batch(C, b_base, stride_c);

            int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

            status = rocblas_internal_gemm<BATCHED>(handle,
                                                    trans_a,
                                                    trans_b,
                                                    m_64,
                                                    n_64,
                                                    k_64,
                                                    alpha,
                                                    A_ptr,
                                                    offset_a,
                                                    lda_64,
                                                    stride_a,
                                                    B_ptr,
                                                    offset_b,
                                                    ldb_64,
                                                    stride_b,
                                                    beta,
                                                    C_ptr,
                                                    offset_c,
                                                    ldc_64,
                                                    stride_c,
                                                    batch_count);
            if(status != rocblas_status_success)
                return status;
        }
    }
    else
    {
        constexpr int64_t limit = c_i32_max * 16; // source kernels must have m and n blocks >= 16
        bool              source_dims_supported = m_64 <= limit && n_64 <= limit;

        TScal alpha_h, beta_h;
        RETURN_IF_ROCBLAS_ERROR(rocblas_copy_alpha_beta_to_host_if_on_device(
            handle, alpha, beta, alpha_h, beta_h, k_64));
        auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

        hipStream_t rocblas_stream = handle->get_stream();

        for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
        {
            auto A_ptr = adjust_ptr_batch(A, b_base, stride_a);
            auto B_ptr = adjust_ptr_batch(B, b_base, stride_b);
            auto C_ptr = adjust_ptr_batch(C, b_base, stride_c);

            int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

            if(k_64 == 0 || (alpha && *alpha == 0))
            {
                status = rocblas_gemm_scale_launcher_64(
                    handle, m_64, n_64, *beta, C, offset_c, ldc_64, stride_c, batch_count);
            }
            else
            {
                if(!source_dims_supported)
                    return rocblas_status_invalid_size;

                for(int64_t n_base = 0; n_base < n_64; n_base += c_i64_grid_YZ_chunk)
                {
                    int32_t n = int32_t(std::min(n_64 - n_base, c_i64_grid_YZ_chunk));

                    for(int64_t m_base = 0; m_base < m_64; m_base += c_i64_grid_X_chunk)
                    {
                        int32_t m = int32_t(std::min(m_64 - m_base, c_i64_grid_X_chunk));

                        status = rocblas_gemm_source_solution_64<BATCHED>(
                            handle,
                            trans_a,
                            trans_b,
                            m,
                            n,
                            k_64,
                            *alpha,
                            A_ptr,
                            lda_64,
                            stride_a,
                            offset_a
                                + (trans_a == rocblas_operation_none ? m_base : m_base * lda_64),
                            B_ptr,
                            ldb_64,
                            stride_b,
                            offset_b
                                + (trans_b == rocblas_operation_none ? n_base * ldb_64 : n_base),
                            *beta,
                            (TConstPtr)C_ptr,
                            ldc_64,
                            stride_c,
                            offset_c + m_base + n_base * ldc_64,
                            C_ptr,
                            ldc_64,
                            stride_c,
                            offset_c + m_base + n_base * ldc_64,
                            batch_count);

                        if(status != rocblas_status_success)
                            return status;
                    }
                }
            }
        }
    }

    return status;
}

template <typename T>
rocblas_status rocblas_internal_gemm_template_64(rocblas_handle    handle,
                                                 rocblas_operation trans_a,
                                                 rocblas_operation trans_b,
                                                 int64_t           m,
                                                 int64_t           n,
                                                 int64_t           k,
                                                 const T*          alpha,
                                                 const T*          A,
                                                 rocblas_stride    offset_a,
                                                 int64_t           lda,
                                                 rocblas_stride    stride_a,
                                                 const T*          B,
                                                 rocblas_stride    offset_b,
                                                 int64_t           ldb,
                                                 rocblas_stride    stride_b,
                                                 const T*          beta,
                                                 T*                C,
                                                 rocblas_stride    offset_c,
                                                 int64_t           ldc,
                                                 rocblas_stride    stride_c,
                                                 int64_t           batch_count)
{
    return rocblas_internal_gemm_64<false>(handle,
                                           trans_a,
                                           trans_b,
                                           m,
                                           n,
                                           k,
                                           alpha,
                                           A,
                                           offset_a,
                                           lda,
                                           stride_a,
                                           B,
                                           offset_b,
                                           ldb,
                                           stride_b,
                                           beta,
                                           C,
                                           offset_c,
                                           ldc,
                                           stride_c,
                                           batch_count);
}

template <typename T>
rocblas_status rocblas_internal_gemm_batched_template_64(rocblas_handle    handle,
                                                         rocblas_operation trans_a,
                                                         rocblas_operation trans_b,
                                                         int64_t           m,
                                                         int64_t           n,
                                                         int64_t           k,
                                                         const T*          alpha,
                                                         const T* const*   A,
                                                         rocblas_stride    offset_a,
                                                         int64_t           lda,
                                                         rocblas_stride    stride_a,
                                                         const T* const*   B,
                                                         rocblas_stride    offset_b,
                                                         int64_t           ldb,
                                                         rocblas_stride    stride_b,
                                                         const T*          beta,
                                                         T* const*         C,
                                                         rocblas_stride    offset_c,
                                                         int64_t           ldc,
                                                         rocblas_stride    stride_c,
                                                         int64_t           batch_count)
{
    return rocblas_internal_gemm_64<true>(handle,
                                          trans_a,
                                          trans_b,
                                          m,
                                          n,
                                          k,
                                          alpha,
                                          A,
                                          offset_a,
                                          lda,
                                          stride_a,
                                          B,
                                          offset_b,
                                          ldb,
                                          stride_b,
                                          beta,
                                          C,
                                          offset_c,
                                          ldc,
                                          stride_c,
                                          batch_count);
}

#ifdef INST_GEMM_TEMPLATE_64
#error INST_GEMM_TEMPLATE_64 already defined
#endif

#define INST_GEMM_TEMPLATE_64(T_)                                                             \
    template rocblas_status rocblas_internal_gemm_template_64<T_>(rocblas_handle    handle,   \
                                                                  rocblas_operation trans_a,  \
                                                                  rocblas_operation trans_b,  \
                                                                  int64_t           m_64,     \
                                                                  int64_t           n_64,     \
                                                                  int64_t           k_64,     \
                                                                  const T_*         alpha,    \
                                                                  const T_*         A,        \
                                                                  rocblas_stride    offset_a, \
                                                                  int64_t           lda_64,   \
                                                                  rocblas_stride    stride_a, \
                                                                  const T_*         B,        \
                                                                  rocblas_stride    offset_b, \
                                                                  int64_t           ldb_64,   \
                                                                  rocblas_stride    stride_b, \
                                                                  const T_*         beta,     \
                                                                  T_*               C,        \
                                                                  rocblas_stride    offset_c, \
                                                                  int64_t           ldc_64,   \
                                                                  rocblas_stride    stride_c, \
                                                                  int64_t           batch_count_64);

INST_GEMM_TEMPLATE_64(rocblas_half)
INST_GEMM_TEMPLATE_64(float)
INST_GEMM_TEMPLATE_64(double)
INST_GEMM_TEMPLATE_64(rocblas_float_complex)
INST_GEMM_TEMPLATE_64(rocblas_double_complex)

#undef INST_GEMM_TEMPLATE_64

#ifdef INST_GEMM_BATCHED_TEMPLATE_64
#error INST_GEMM_BATCHED_TEMPLATE_64 already defined
#endif

#define INST_GEMM_BATCHED_TEMPLATE_64(T_)                                  \
    template rocblas_status rocblas_internal_gemm_batched_template_64<T_>( \
        rocblas_handle    handle,                                          \
        rocblas_operation trans_a,                                         \
        rocblas_operation trans_b,                                         \
        int64_t           m_64,                                            \
        int64_t           n_64,                                            \
        int64_t           k_64,                                            \
        const T_*         alpha,                                           \
        const T_* const*  A,                                               \
        rocblas_stride    offset_a,                                        \
        int64_t           lda_64,                                          \
        rocblas_stride    stride_a,                                        \
        const T_* const*  B,                                               \
        rocblas_stride    offset_b,                                        \
        int64_t           ldb_64,                                          \
        rocblas_stride    stride_b,                                        \
        const T_*         beta,                                            \
        T_* const*        C,                                               \
        rocblas_stride    offset_c,                                        \
        int64_t           ldc_64,                                          \
        rocblas_stride    stride_c,                                        \
        int64_t           batch_count);

INST_GEMM_BATCHED_TEMPLATE_64(rocblas_half)
INST_GEMM_BATCHED_TEMPLATE_64(float)
INST_GEMM_BATCHED_TEMPLATE_64(double)
INST_GEMM_BATCHED_TEMPLATE_64(rocblas_float_complex)
INST_GEMM_BATCHED_TEMPLATE_64(rocblas_double_complex)

#undef INST_GEMM_BATCHED_TEMPLATE_64
