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

#include "handle.hpp"
#include "int64_helpers.hpp"
#include "rocblas-types.h"
#include "rocblas_dgmm_64.hpp"

/*
 * ===========================================================================
 *    template interface
 *    template specialization
 *    call DGMM C interfaces (see rocblas_dgmm*.cpp in the same dir)
 * ===========================================================================
 */

/**
 * TConstPtr is either: const T* OR const T* const*
 * TPtr      is either:       T* OR       T* const*
 * Where T is the base type (float, double, rocblas_complex, or rocblas_double_complex)
 */

template <typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_dgmm_launcher_64(rocblas_handle handle,
                                                 rocblas_side   side,
                                                 int64_t        m_64,
                                                 int64_t        n_64,
                                                 TConstPtr      A,
                                                 rocblas_stride offset_A,
                                                 int64_t        lda_64,
                                                 rocblas_stride stride_A,
                                                 TConstPtr      x,
                                                 rocblas_stride offset_x,
                                                 int64_t        incx_64,
                                                 rocblas_stride stride_x,
                                                 TPtr           C,
                                                 rocblas_stride offset_C,
                                                 int64_t        ldc_64,
                                                 rocblas_stride stride_C,
                                                 int64_t        batch_count_64)

{
    // Quick return if possible. Not Argument error
    if(!m_64 || !n_64 || !batch_count_64)
        return rocblas_status_success;

    bool dims_32bit = m_64 <= c_ILP64_i32_max && n_64 <= c_ILP64_i32_max;

    for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
    {
        auto    A_ptr       = adjust_ptr_batch(A, b_base, stride_A);
        auto    x_ptr       = adjust_ptr_batch(x, b_base, stride_x);
        auto    C_ptr       = adjust_ptr_batch(C, b_base, stride_C);
        int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

        if(dims_32bit)
        {
            rocblas_status status = rocblas_internal_dgmm_launcher(handle,
                                                                   side,
                                                                   (int)m_64,
                                                                   (int)n_64,
                                                                   A_ptr,
                                                                   offset_A,
                                                                   lda_64,
                                                                   stride_A,
                                                                   x_ptr,
                                                                   offset_x,
                                                                   incx_64,
                                                                   stride_x,
                                                                   C_ptr,
                                                                   offset_C,
                                                                   ldc_64,
                                                                   stride_C,
                                                                   batch_count);
            if(status != rocblas_status_success)
                return status;
        }
        else if(side == rocblas_side_left)
        {
            for(int64_t n_base = 0; n_base < n_64; n_base += c_i64_grid_YZ_chunk)
            {
                int32_t n = int32_t(std::min(n_64 - n_base, c_i64_grid_YZ_chunk));

                for(int64_t m_base = 0; m_base < m_64; m_base += c_i64_grid_X_chunk)
                {
                    int32_t m = int32_t(std::min(m_64 - m_base, c_i64_grid_X_chunk));

                    int64_t shift_x
                        = offset_x
                          + (incx_64 < 0 ? -incx_64 * (m_64 - m - m_base) : m_base * incx_64);

                    rocblas_status status
                        = rocblas_internal_dgmm_launcher(handle,
                                                         side,
                                                         m,
                                                         n,
                                                         A_ptr,
                                                         offset_A + m_base + n_base * lda_64,
                                                         lda_64,
                                                         stride_A,
                                                         x_ptr,
                                                         shift_x,
                                                         incx_64,
                                                         stride_x,
                                                         C_ptr,
                                                         offset_C + m_base + n_base * ldc_64,
                                                         ldc_64,
                                                         stride_C,
                                                         batch_count);
                    if(status != rocblas_status_success)
                        return status;
                }
            }
        }
        else // if(side == rocblas_side_right)
        {
            for(int64_t n_base = 0; n_base < n_64; n_base += c_i64_grid_YZ_chunk)
            {
                int32_t n = int32_t(std::min(n_64 - n_base, c_i64_grid_YZ_chunk));

                int64_t shift_x
                    = offset_x + (incx_64 < 0 ? -incx_64 * (n_64 - n - n_base) : n_base * incx_64);

                for(int64_t m_base = 0; m_base < m_64; m_base += c_i64_grid_X_chunk)
                {
                    int32_t m = int32_t(std::min(m_64 - m_base, c_i64_grid_X_chunk));

                    rocblas_status status
                        = rocblas_internal_dgmm_launcher(handle,
                                                         side,
                                                         m,
                                                         n,
                                                         A_ptr,
                                                         offset_A + m_base + n_base * lda_64,
                                                         lda_64,
                                                         stride_A,
                                                         x_ptr,
                                                         shift_x,
                                                         incx_64,
                                                         stride_x,
                                                         C_ptr,
                                                         offset_C + m_base + n_base * ldc_64,
                                                         ldc_64,
                                                         stride_C,
                                                         batch_count);
                    if(status != rocblas_status_success)
                        return status;
                }
            }
        }

    } // batch

    return rocblas_status_success;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files dgmm*.cpp

#ifdef INSTANTIATE_DGMM_LAUNCHER_64
#error INSTANTIATE_DGMM_LAUNCHER_64 already defined
#endif

#define INSTANTIATE_DGMM_LAUNCHER_64(TConstPtr_, TPtr_)                           \
    template rocblas_status rocblas_internal_dgmm_launcher_64<TConstPtr_, TPtr_>( \
        rocblas_handle handle,                                                    \
        rocblas_side   side,                                                      \
        int64_t        m_64,                                                      \
        int64_t        n_64,                                                      \
        TConstPtr_     A,                                                         \
        rocblas_stride offset_A,                                                  \
        int64_t        lda_64,                                                    \
        rocblas_stride stride_A,                                                  \
        TConstPtr_     X,                                                         \
        rocblas_stride offset_x,                                                  \
        int64_t        incx_64,                                                   \
        rocblas_stride stride_x,                                                  \
        TPtr_          C,                                                         \
        rocblas_stride offset_C,                                                  \
        int64_t        ldc_64,                                                    \
        rocblas_stride stride_C,                                                  \
        int64_t        batch_count_64);

// instantiate for rocblas_Xdgmm and rocblas_Xdgmm_strided_batched
INSTANTIATE_DGMM_LAUNCHER_64(float const*, float*)
INSTANTIATE_DGMM_LAUNCHER_64(double const*, double*)
INSTANTIATE_DGMM_LAUNCHER_64(rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_DGMM_LAUNCHER_64(rocblas_double_complex const*, rocblas_double_complex*)

// instantiate for rocblas_Xdgmm_batched
INSTANTIATE_DGMM_LAUNCHER_64(float const* const*, float* const*)
INSTANTIATE_DGMM_LAUNCHER_64(double const* const*, double* const*)
INSTANTIATE_DGMM_LAUNCHER_64(rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_DGMM_LAUNCHER_64(rocblas_double_complex const* const*, rocblas_double_complex* const*)
#undef INSTANTIATE_DGMM_LAUNCHER_64
