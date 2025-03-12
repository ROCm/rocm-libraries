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
 * ************************************************************************ */

#include "blas3/rocblas_symm_hemm.hpp" // int32 API called
#include "handle.hpp"
#include "int64_helpers.hpp"
#include "rocblas_symm_hemm_64.hpp"

template <bool HERM, typename T_>
rocblas_status rocblas_internal_symm_hemm_launcher_64(rocblas_handle handle,
                                                      rocblas_side   side,
                                                      rocblas_fill   uplo,
                                                      int64_t        m_64,
                                                      int64_t        n_64,
                                                      const T_*      alpha,
                                                      const T_*      A,
                                                      rocblas_stride offsetA,
                                                      int64_t        lda_64,
                                                      rocblas_stride strideA,
                                                      const T_*      B,
                                                      rocblas_stride offsetB,
                                                      int64_t        ldb_64,
                                                      rocblas_stride strideB,
                                                      const T_*      beta,
                                                      T_*            C,
                                                      rocblas_stride offsetC,
                                                      int64_t        ldc_64,
                                                      rocblas_stride strideC,
                                                      int64_t        batch_count_64)
{
    if(!m_64 || !n_64 || !batch_count_64)
        return rocblas_status_success;

    if((m_64 > c_i32_max && side == rocblas_side_left)
       || (n_64 > c_i32_max && side == rocblas_side_right))
    {
        // exceeds practical memory
        return rocblas_status_invalid_size;
    }

    if(n_64 <= c_ILP64_i32_max && m_64 < c_ILP64_i32_max && lda_64 < c_ILP64_i32_max
       && ldb_64 < c_ILP64_i32_max && ldc_64 < c_ILP64_i32_max
       && batch_count_64 < c_i64_grid_YZ_chunk)
    {
        return rocblas_internal_symm_hemm_launcher<HERM>(handle,
                                                         side,
                                                         uplo,
                                                         (rocblas_int)m_64,
                                                         (rocblas_int)n_64,
                                                         alpha,
                                                         A,
                                                         offsetA,
                                                         lda_64,
                                                         strideA,
                                                         B,
                                                         offsetB,
                                                         ldb_64,
                                                         strideB,
                                                         beta,
                                                         C,
                                                         offsetC,
                                                         ldc_64,
                                                         strideC,
                                                         (rocblas_int)batch_count_64);
    }

    for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
    {
        int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

        auto A_ptr = adjust_ptr_batch(A, b_base, strideA);
        auto B_ptr = adjust_ptr_batch(B, b_base, strideB);
        auto C_ptr = adjust_ptr_batch(C, b_base, strideC);

        if(n_64 <= c_ILP64_i32_max && m_64 < c_ILP64_i32_max && lda_64 < c_ILP64_i32_max
           && ldb_64 < c_ILP64_i32_max && ldc_64 < c_ILP64_i32_max)
        {
            auto status = rocblas_internal_symm_hemm_launcher<HERM>(handle,
                                                                    side,
                                                                    uplo,
                                                                    (rocblas_int)m_64,
                                                                    (rocblas_int)n_64,
                                                                    alpha,
                                                                    A_ptr,
                                                                    offsetA,
                                                                    lda_64,
                                                                    strideA,
                                                                    B_ptr,
                                                                    offsetB,
                                                                    ldb_64,
                                                                    strideB,
                                                                    beta,
                                                                    C_ptr,
                                                                    offsetC,
                                                                    ldc_64,
                                                                    strideC,
                                                                    batch_count);
            if(status != rocblas_status_success)
                return status;
        }
        else
        {
            //Note: Use of an additional loop for reduction of partial dot products is skipped in this implementation,
            //as we are assuming the given problem can be decomposed into smaller chunks as symmetric matrices cannot be large due to memory constrains.
            //if side == Left,  m must be fairly small for given memory constrains
            //if side == right, n must be fairly small for given memory constrains
            for(int64_t n_base = 0; n_base < n_64; n_base += c_i64_grid_YZ_chunk)
            {
                int32_t n = int32_t(std::min(n_64 - n_base, c_i64_grid_YZ_chunk));

                for(int64_t m_base = 0; m_base < m_64; m_base += c_i64_grid_X_chunk)
                {
                    int32_t m = int32_t(std::min(m_64 - m_base, c_i64_grid_X_chunk));

                    auto status = rocblas_internal_symm_hemm_launcher<HERM>(
                        handle,
                        side,
                        uplo,
                        m,
                        n,
                        alpha,
                        A_ptr,
                        offsetA
                            + (side == rocblas_side_left ? m_base + m_base * lda_64
                                                         : n_base + n_base * lda_64),
                        lda_64,
                        strideA,
                        B_ptr,
                        offsetB + m_base + n_base * ldb_64,
                        ldb_64,
                        strideB,
                        beta,
                        C_ptr,
                        offsetC + m_base + n_base * ldc_64,
                        ldc_64,
                        strideC,
                        batch_count);
                    if(status != rocblas_status_success)
                        return status;
                }
            }
        }
    }
    return rocblas_status_success;
}

template <bool HERM, typename T_>
rocblas_status rocblas_internal_symm_hemm_batched_launcher_64(rocblas_handle   handle,
                                                              rocblas_side     side,
                                                              rocblas_fill     uplo,
                                                              int64_t          m_64,
                                                              int64_t          n_64,
                                                              const T_*        alpha,
                                                              const T_* const* A,
                                                              rocblas_stride   offsetA,
                                                              int64_t          lda_64,
                                                              rocblas_stride   strideA,
                                                              const T_* const* B,
                                                              rocblas_stride   offsetB,
                                                              int64_t          ldb_64,
                                                              rocblas_stride   strideB,
                                                              const T_*        beta,
                                                              T_* const*       C,
                                                              rocblas_stride   offsetC,
                                                              int64_t          ldc_64,
                                                              rocblas_stride   strideC,
                                                              int64_t          batch_count_64)

{
    if(!m_64 || !n_64 || !batch_count_64)
        return rocblas_status_success;

    if(n_64 <= c_ILP64_i32_max && m_64 < c_ILP64_i32_max && lda_64 < c_ILP64_i32_max
       && ldb_64 < c_ILP64_i32_max && ldc_64 < c_ILP64_i32_max
       && batch_count_64 < c_i64_grid_YZ_chunk)
    {

        return rocblas_internal_symm_hemm_batched_launcher<HERM>(handle,
                                                                 side,
                                                                 uplo,
                                                                 (rocblas_int)m_64,
                                                                 (rocblas_int)n_64,
                                                                 alpha,
                                                                 A,
                                                                 offsetA,
                                                                 lda_64,
                                                                 strideA,
                                                                 B,
                                                                 offsetB,
                                                                 ldb_64,
                                                                 strideB,
                                                                 beta,
                                                                 C,
                                                                 offsetC,
                                                                 ldc_64,
                                                                 strideC,
                                                                 (rocblas_int)batch_count_64);
    }

    for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
    {
        int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

        auto A_ptr = adjust_ptr_batch(A, b_base, strideA);
        auto B_ptr = adjust_ptr_batch(B, b_base, strideB);
        auto C_ptr = adjust_ptr_batch(C, b_base, strideC);

        if(n_64 <= c_ILP64_i32_max && m_64 < c_ILP64_i32_max && lda_64 < c_ILP64_i32_max
           && ldb_64 < c_ILP64_i32_max && ldc_64 < c_ILP64_i32_max)
        {
            auto status = rocblas_internal_symm_hemm_batched_launcher<HERM>(handle,
                                                                            side,
                                                                            uplo,
                                                                            (rocblas_int)m_64,
                                                                            (rocblas_int)n_64,
                                                                            alpha,
                                                                            A_ptr,
                                                                            offsetA,
                                                                            lda_64,
                                                                            strideA,
                                                                            B_ptr,
                                                                            offsetB,
                                                                            ldb_64,
                                                                            strideB,
                                                                            beta,
                                                                            C_ptr,
                                                                            offsetC,
                                                                            ldc_64,
                                                                            strideC,
                                                                            batch_count);
            if(status != rocblas_status_success)
                return status;
        }
        else
        {
            //Note: Use of an additional loop for reduction of partial dot products is skipped in this implementation,
            //as we are assuming the given problem can be decomposed into smaller chunks as symmetric matrices cannot be large due to memory constrains.
            //if side == Left,  m must be fairly small for given memory constrains
            //if side == right, n must be fairly small for given memory constrains
            for(int64_t n_base = 0; n_base < n_64; n_base += c_i64_grid_YZ_chunk)
            {
                int32_t n = int32_t(std::min(n_64 - n_base, c_i64_grid_YZ_chunk));

                for(int64_t m_base = 0; m_base < m_64; m_base += c_i64_grid_X_chunk)
                {
                    int32_t m = int32_t(std::min(m_64 - m_base, c_i64_grid_X_chunk));

                    auto status = rocblas_internal_symm_hemm_batched_launcher<HERM>(
                        handle,
                        side,
                        uplo,
                        m,
                        n,
                        alpha,
                        A_ptr,
                        offsetA
                            + (side == rocblas_side_left ? m_base + m_base * lda_64
                                                         : n_base + n_base * lda_64),
                        lda_64,
                        strideA,
                        B_ptr,
                        offsetB + m_base + n_base * ldb_64,
                        ldb_64,
                        strideB,
                        beta,
                        C_ptr,
                        offsetC + m_base + n_base * ldc_64,
                        ldc_64,
                        strideC,
                        batch_count);
                    if(status != rocblas_status_success)
                        return status;
                }
            }
        }
    }
    return rocblas_status_success;
}

template <typename T_>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_symm_template_64(rocblas_handle handle,
                                      rocblas_side   side,
                                      rocblas_fill   uplo,
                                      int64_t        m_64,
                                      int64_t        n_64,
                                      const T_*      alpha,
                                      const T_*      A,
                                      rocblas_stride offsetA,
                                      int64_t        lda_64,
                                      rocblas_stride strideA,
                                      const T_*      B,
                                      rocblas_stride offsetB,
                                      int64_t        ldb_64,
                                      rocblas_stride strideB,
                                      const T_*      beta,
                                      T_*            C,
                                      rocblas_stride offsetC,
                                      int64_t        ldc_64,
                                      rocblas_stride strideC,
                                      int64_t        batch_count_64)
{
    constexpr bool HERM = false;
    return rocblas_internal_symm_hemm_launcher_64<HERM>(handle,
                                                        side,
                                                        uplo,
                                                        m_64,
                                                        n_64,
                                                        alpha,
                                                        A,
                                                        offsetA,
                                                        lda_64,
                                                        strideA,
                                                        B,
                                                        offsetB,
                                                        ldb_64,
                                                        strideB,
                                                        beta,
                                                        C,
                                                        offsetC,
                                                        ldc_64,
                                                        strideC,
                                                        batch_count_64);
}

template <typename T_>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_symm_batched_template_64(rocblas_handle   handle,
                                              rocblas_side     side,
                                              rocblas_fill     uplo,
                                              int64_t          m_64,
                                              int64_t          n_64,
                                              const T_*        alpha,
                                              const T_* const* A,
                                              rocblas_stride   offsetA,
                                              int64_t          lda_64,
                                              rocblas_stride   strideA,
                                              const T_* const* B,
                                              rocblas_stride   offsetB,
                                              int64_t          ldb_64,
                                              rocblas_stride   strideB,
                                              const T_*        beta,
                                              T_* const*       C,
                                              rocblas_stride   offsetC,
                                              int64_t          ldc_64,
                                              rocblas_stride   strideC,
                                              int64_t          batch_count_64)
{
    constexpr bool HERM = false;
    return rocblas_internal_symm_hemm_batched_launcher_64<HERM>(handle,
                                                                side,
                                                                uplo,
                                                                m_64,
                                                                n_64,
                                                                alpha,
                                                                A,
                                                                offsetA,
                                                                lda_64,
                                                                strideA,
                                                                B,
                                                                offsetB,
                                                                ldb_64,
                                                                strideB,
                                                                beta,
                                                                C,
                                                                offsetC,
                                                                ldc_64,
                                                                strideC,
                                                                batch_count_64);
}

template <typename T_>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_hemm_template_64(rocblas_handle handle,
                                      rocblas_side   side,
                                      rocblas_fill   uplo,
                                      int64_t        m_64,
                                      int64_t        n_64,
                                      const T_*      alpha,
                                      const T_*      A,
                                      rocblas_stride offsetA,
                                      int64_t        lda_64,
                                      rocblas_stride strideA,
                                      const T_*      B,
                                      rocblas_stride offsetB,
                                      int64_t        ldb_64,
                                      rocblas_stride strideB,
                                      const T_*      beta,
                                      T_*            C,
                                      rocblas_stride offsetC,
                                      int64_t        ldc_64,
                                      rocblas_stride strideC,
                                      int64_t        batch_count_64)
{
    constexpr bool HERM = true;
    return rocblas_internal_symm_hemm_launcher_64<HERM>(handle,
                                                        side,
                                                        uplo,
                                                        m_64,
                                                        n_64,
                                                        alpha,
                                                        A,
                                                        offsetA,
                                                        lda_64,
                                                        strideA,
                                                        B,
                                                        offsetB,
                                                        ldb_64,
                                                        strideB,
                                                        beta,
                                                        C,
                                                        offsetC,
                                                        ldc_64,
                                                        strideC,
                                                        batch_count_64);
}

template <typename T_>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_hemm_batched_template_64(rocblas_handle   handle,
                                              rocblas_side     side,
                                              rocblas_fill     uplo,
                                              int64_t          m_64,
                                              int64_t          n_64,
                                              const T_*        alpha,
                                              const T_* const* A,
                                              rocblas_stride   offsetA,
                                              int64_t          lda_64,
                                              rocblas_stride   strideA,
                                              const T_* const* B,
                                              rocblas_stride   offsetB,
                                              int64_t          ldb_64,
                                              rocblas_stride   strideB,
                                              const T_*        beta,
                                              T_* const*       C,
                                              rocblas_stride   offsetC,
                                              int64_t          ldc_64,
                                              rocblas_stride   strideC,
                                              int64_t          batch_count_64)
{

    constexpr bool HERM = true;
    return rocblas_internal_symm_hemm_batched_launcher_64<HERM>(handle,
                                                                side,
                                                                uplo,
                                                                m_64,
                                                                n_64,
                                                                alpha,
                                                                A,
                                                                offsetA,
                                                                lda_64,
                                                                strideA,
                                                                B,
                                                                offsetB,
                                                                ldb_64,
                                                                strideB,
                                                                beta,
                                                                C,
                                                                offsetC,
                                                                ldc_64,
                                                                strideC,
                                                                batch_count_64);
}

#ifdef INST_SYMM_TEMPLATE_64
#error INST_SYMM_TEMPLATE_64 already defined
#endif

#define INST_SYMM_TEMPLATE_64(T_)                                     \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status          \
        rocblas_internal_symm_template_64<T_>(rocblas_handle handle,  \
                                              rocblas_side   side,    \
                                              rocblas_fill   uplo,    \
                                              int64_t        m_64,    \
                                              int64_t        n_64,    \
                                              const T_*      alpha,   \
                                              const T_*      A,       \
                                              rocblas_stride offsetA, \
                                              int64_t        lda_64,  \
                                              rocblas_stride strideA, \
                                              const T_*      B,       \
                                              rocblas_stride offsetB, \
                                              int64_t        ldb_64,  \
                                              rocblas_stride strideB, \
                                              const T_*      beta,    \
                                              T_*            C,       \
                                              rocblas_stride offsetC, \
                                              int64_t        ldc_64,  \
                                              rocblas_stride strideC, \
                                              int64_t        batch_count_64);

INST_SYMM_TEMPLATE_64(float)
INST_SYMM_TEMPLATE_64(double)
INST_SYMM_TEMPLATE_64(rocblas_float_complex)
INST_SYMM_TEMPLATE_64(rocblas_double_complex)

#undef INST_SYMM_TEMPLATE_64

#ifdef INST_SYMM_BATCHED_TEMPLATE_64
#error INST_SYMM_BATCHED_TEMPLATE_64 already defined
#endif

#define INST_SYMM_BATCHED_TEMPLATE_64(T_)                                       \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                    \
        rocblas_internal_symm_batched_template_64<T_>(rocblas_handle   handle,  \
                                                      rocblas_side     side,    \
                                                      rocblas_fill     uplo,    \
                                                      int64_t          m_64,    \
                                                      int64_t          n_64,    \
                                                      const T_*        alpha,   \
                                                      const T_* const* A,       \
                                                      rocblas_stride   offsetA, \
                                                      int64_t          lda_64,  \
                                                      rocblas_stride   strideA, \
                                                      const T_* const* B,       \
                                                      rocblas_stride   offsetB, \
                                                      int64_t          ldb_64,  \
                                                      rocblas_stride   strideB, \
                                                      const T_*        beta,    \
                                                      T_* const*       C,       \
                                                      rocblas_stride   offsetC, \
                                                      int64_t          ldc_64,  \
                                                      rocblas_stride   strideC, \
                                                      int64_t          batch_count_64);

INST_SYMM_BATCHED_TEMPLATE_64(float)
INST_SYMM_BATCHED_TEMPLATE_64(double)
INST_SYMM_BATCHED_TEMPLATE_64(rocblas_float_complex)
INST_SYMM_BATCHED_TEMPLATE_64(rocblas_double_complex)

#undef INST_SYMM_BATCHED_TEMPLATE_64

#ifdef INST_HEMM_TEMPLATE_64
#error INST_HEMM_TEMPLATE_64 already defined
#endif

#define INST_HEMM_TEMPLATE_64(T_)                                     \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status          \
        rocblas_internal_hemm_template_64<T_>(rocblas_handle handle,  \
                                              rocblas_side   side,    \
                                              rocblas_fill   uplo,    \
                                              int64_t        m_64,    \
                                              int64_t        n_64,    \
                                              const T_*      alpha,   \
                                              const T_*      A,       \
                                              rocblas_stride offsetA, \
                                              int64_t        lda_64,  \
                                              rocblas_stride strideA, \
                                              const T_*      B,       \
                                              rocblas_stride offsetB, \
                                              int64_t        ldb_64,  \
                                              rocblas_stride strideB, \
                                              const T_*      beta,    \
                                              T_*            C,       \
                                              rocblas_stride offsetC, \
                                              int64_t        ldc_64,  \
                                              rocblas_stride strideC, \
                                              int64_t        batch_count_64);

INST_HEMM_TEMPLATE_64(rocblas_float_complex)
INST_HEMM_TEMPLATE_64(rocblas_double_complex)

#undef INST_HEMM_TEMPLATE_64

#ifdef INST_HEMM_BATCHED_TEMPLATE_64
#error INST_HEMM_BATCHED_TEMPLATE_64 already defined
#endif

#define INST_HEMM_BATCHED_TEMPLATE_64(T_)                                       \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                    \
        rocblas_internal_hemm_batched_template_64<T_>(rocblas_handle   handle,  \
                                                      rocblas_side     side,    \
                                                      rocblas_fill     uplo,    \
                                                      int64_t          m_64,    \
                                                      int64_t          n_64,    \
                                                      const T_*        alpha,   \
                                                      const T_* const* A,       \
                                                      rocblas_stride   offsetA, \
                                                      int64_t          lda_64,  \
                                                      rocblas_stride   strideA, \
                                                      const T_* const* B,       \
                                                      rocblas_stride   offsetB, \
                                                      int64_t          ldb_64,  \
                                                      rocblas_stride   strideB, \
                                                      const T_*        beta,    \
                                                      T_* const*       C,       \
                                                      rocblas_stride   offsetC, \
                                                      int64_t          ldc_64,  \
                                                      rocblas_stride   strideC, \
                                                      int64_t          batch_count_64);

INST_HEMM_BATCHED_TEMPLATE_64(rocblas_float_complex)
INST_HEMM_BATCHED_TEMPLATE_64(rocblas_double_complex)

#undef INST_HEMM_BATCHED_TEMPLATE_64
