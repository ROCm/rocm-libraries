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

#include "blas2/rocblas_ger.hpp" // int32 API called
#include "blas2/rocblas_ger_kernels.hpp"

template <bool CONJ, typename T, typename U, typename V, typename W>
rocblas_status rocblas_internal_ger_launcher_64(rocblas_handle handle,
                                                int64_t        m_64,
                                                int64_t        n_64,
                                                const V*       alpha,
                                                rocblas_stride stride_alpha,
                                                U              x,
                                                rocblas_stride offsetx,
                                                int64_t        incx_64,
                                                rocblas_stride stridex,
                                                U              y,
                                                rocblas_stride offsety,
                                                int64_t        incy_64,
                                                rocblas_stride stridey,
                                                W              A,
                                                rocblas_stride offsetA,
                                                int64_t        lda_64,
                                                rocblas_stride strideA,
                                                int64_t        batch_count_64)
{
    // Quick return if possible. Not Argument error
    if(!m_64 || !n_64 || !batch_count_64)
        return rocblas_status_success;

    for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
    {
        auto    x_ptr       = adjust_ptr_batch(x, b_base, stridex);
        auto    y_ptr       = adjust_ptr_batch(y, b_base, stridey);
        auto    A_ptr       = adjust_ptr_batch(A, b_base, strideA);
        int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

        for(int64_t n_base = 0; n_base < n_64; n_base += c_i64_grid_X_chunk)
        {
            int32_t n = int32_t(std::min(n_64 - n_base, c_i64_grid_X_chunk));

            int64_t shifty = (incy_64 < 0 ? -incy_64 * (n_64 - n - n_base) : n_base * incy_64);

            shifty += offsety;

            for(int64_t m_base = 0; m_base < m_64; m_base += c_i64_grid_X_chunk)
            {
                int32_t m = int32_t(std::min(m_64 - m_base, c_i64_grid_X_chunk));

                int64_t shiftx = (incx_64 < 0 ? -incx_64 * (m_64 - m - m_base) : m_base * incx_64);

                shiftx += offsetx;

                auto shiftA = lda_64 * n_base + m_base + offsetA;

                rocblas_status status = rocblas_internal_ger_launcher<CONJ, T>(handle,
                                                                               m,
                                                                               n,
                                                                               alpha,
                                                                               stride_alpha,
                                                                               x_ptr,
                                                                               shiftx,
                                                                               incx_64,
                                                                               stridex,
                                                                               y_ptr,
                                                                               shifty,
                                                                               incy_64,
                                                                               stridey,
                                                                               A_ptr,
                                                                               shiftA,
                                                                               lda_64,
                                                                               strideA,
                                                                               batch_count);

                if(status != rocblas_status_success)
                    return status;

            } // m
        } // n
    } // batch
    return rocblas_status_success;
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_ger_template_64(rocblas_handle handle,
                                     int64_t        m,
                                     int64_t        n,
                                     const T*       alpha,
                                     rocblas_stride stride_alpha,
                                     const T*       x,
                                     rocblas_stride offsetx,
                                     int64_t        incx,
                                     rocblas_stride stridex,
                                     const T*       y,
                                     rocblas_stride offsety,
                                     int64_t        incy,
                                     rocblas_stride stridey,
                                     T*             A,
                                     rocblas_stride offsetA,
                                     int64_t        lda,
                                     rocblas_stride strideA,
                                     int64_t        batch_count)
{
    return rocblas_internal_ger_launcher_64<false, T>(handle,
                                                      m,
                                                      n,
                                                      alpha,
                                                      stride_alpha,
                                                      x,
                                                      offsetx,
                                                      incx,
                                                      stridex,
                                                      y,
                                                      offsety,
                                                      incy,
                                                      stridey,
                                                      A,
                                                      offsetA,
                                                      lda,
                                                      strideA,
                                                      batch_count);
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_gerc_template_64(rocblas_handle handle,
                                      int64_t        m,
                                      int64_t        n,
                                      const T*       alpha,
                                      rocblas_stride stride_alpha,
                                      const T*       x,
                                      rocblas_stride offsetx,
                                      int64_t        incx,
                                      rocblas_stride stridex,
                                      const T*       y,
                                      rocblas_stride offsety,
                                      int64_t        incy,
                                      rocblas_stride stridey,
                                      T*             A,
                                      rocblas_stride offsetA,
                                      int64_t        lda,
                                      rocblas_stride strideA,
                                      int64_t        batch_count)
{
    return rocblas_internal_ger_launcher_64<true, T>(handle,
                                                     m,
                                                     n,
                                                     alpha,
                                                     stride_alpha,
                                                     x,
                                                     offsetx,
                                                     incx,
                                                     stridex,
                                                     y,
                                                     offsety,
                                                     incy,
                                                     stridey,
                                                     A,
                                                     offsetA,
                                                     lda,
                                                     strideA,
                                                     batch_count);
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_ger_batched_template_64(rocblas_handle  handle,
                                             int64_t         m,
                                             int64_t         n,
                                             const T*        alpha,
                                             rocblas_stride  stride_alpha,
                                             const T* const* x,
                                             rocblas_stride  offsetx,
                                             int64_t         incx,
                                             rocblas_stride  stridex,
                                             const T* const* y,
                                             rocblas_stride  offsety,
                                             int64_t         incy,
                                             rocblas_stride  stridey,
                                             T* const*       A,
                                             rocblas_stride  offsetA,
                                             int64_t         lda,
                                             rocblas_stride  strideA,
                                             int64_t         batch_count)
{
    return rocblas_internal_ger_launcher_64<false, T>(handle,
                                                      m,
                                                      n,
                                                      alpha,
                                                      stride_alpha,
                                                      x,
                                                      offsetx,
                                                      incx,
                                                      stridex,
                                                      y,
                                                      offsety,
                                                      incy,
                                                      stridey,
                                                      A,
                                                      offsetA,
                                                      lda,
                                                      strideA,
                                                      batch_count);
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_gerc_batched_template_64(rocblas_handle  handle,
                                              int64_t         m,
                                              int64_t         n,
                                              const T*        alpha,
                                              rocblas_stride  stride_alpha,
                                              const T* const* x,
                                              rocblas_stride  offsetx,
                                              int64_t         incx,
                                              rocblas_stride  stridex,
                                              const T* const* y,
                                              rocblas_stride  offsety,
                                              int64_t         incy,
                                              rocblas_stride  stridey,
                                              T* const*       A,
                                              rocblas_stride  offsetA,
                                              int64_t         lda,
                                              rocblas_stride  strideA,
                                              int64_t         batch_count)
{
    return rocblas_internal_ger_launcher_64<true, T>(handle,
                                                     m,
                                                     n,
                                                     alpha,
                                                     stride_alpha,
                                                     x,
                                                     offsetx,
                                                     incx,
                                                     stridex,
                                                     y,
                                                     offsety,
                                                     incy,
                                                     stridey,
                                                     A,
                                                     offsetA,
                                                     lda,
                                                     strideA,
                                                     batch_count);
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *ger*.cpp

#ifdef INST_GER_TEMPLATE_64
#error INST_GER_TEMPLATE_64 already defined
#endif

#define INST_GER_TEMPLATE_64(API_INT_, T_)                                                         \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status rocblas_internal_ger_template_64<T_>( \
        rocblas_handle handle,                                                                     \
        API_INT_       m,                                                                          \
        API_INT_       n,                                                                          \
        const T_*      alpha,                                                                      \
        rocblas_stride stride_alpha,                                                               \
        const T_*      x,                                                                          \
        rocblas_stride offsetx,                                                                    \
        API_INT_       incx,                                                                       \
        rocblas_stride stridex,                                                                    \
        const T_*      y,                                                                          \
        rocblas_stride offsety,                                                                    \
        API_INT_       incy,                                                                       \
        rocblas_stride stridey,                                                                    \
        T_*            A,                                                                          \
        rocblas_stride offsetA,                                                                    \
        API_INT_       lda,                                                                        \
        rocblas_stride strideA,                                                                    \
        API_INT_       batch_count);

#ifdef INST_GERC_TEMPLATE_64
#error INST_GERC_TEMPLATE_64 already defined
#endif

#define INST_GERC_TEMPLATE_64(API_INT_, T_)                                \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status               \
        rocblas_internal_gerc_template_64<T_>(rocblas_handle handle,       \
                                              API_INT_       m,            \
                                              API_INT_       n,            \
                                              const T_*      alpha,        \
                                              rocblas_stride stride_alpha, \
                                              const T_*      x,            \
                                              rocblas_stride offsetx,      \
                                              API_INT_       incx,         \
                                              rocblas_stride stridex,      \
                                              const T_*      y,            \
                                              rocblas_stride offsety,      \
                                              API_INT_       incy,         \
                                              rocblas_stride stridey,      \
                                              T_*            A,            \
                                              rocblas_stride offsetA,      \
                                              API_INT_       lda,          \
                                              rocblas_stride strideA,      \
                                              API_INT_       batch_count);

#ifdef INST_GER_BATCHED_TEMPLATE_64
#error INST_GER_BATCHED_TEMPLATE_64 already defined
#endif

#define INST_GER_BATCHED_TEMPLATE_64(API_INT_, T_)                                  \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                        \
        rocblas_internal_ger_batched_template_64<T_>(rocblas_handle   handle,       \
                                                     API_INT_         m,            \
                                                     API_INT_         n,            \
                                                     const T_*        alpha,        \
                                                     rocblas_stride   stride_alpha, \
                                                     const T_* const* x,            \
                                                     rocblas_stride   offsetx,      \
                                                     API_INT_         incx,         \
                                                     rocblas_stride   stridex,      \
                                                     const T_* const* y,            \
                                                     rocblas_stride   offsety,      \
                                                     API_INT_         incy,         \
                                                     rocblas_stride   stridey,      \
                                                     T_* const*       A,            \
                                                     rocblas_stride   offsetA,      \
                                                     API_INT_         lda,          \
                                                     rocblas_stride   strideA,      \
                                                     API_INT_         batch_count);

#ifdef INST_GERC_BATCHED_TEMPLATE_64
#error INST_GERC_BATCHED_TEMPLATE_64 already defined
#endif

#define INST_GERC_BATCHED_TEMPLATE_64(API_INT_, T_)                                  \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                         \
        rocblas_internal_gerc_batched_template_64<T_>(rocblas_handle   handle,       \
                                                      API_INT_         m,            \
                                                      API_INT_         n,            \
                                                      const T_*        alpha,        \
                                                      rocblas_stride   stride_alpha, \
                                                      const T_* const* x,            \
                                                      rocblas_stride   offsetx,      \
                                                      API_INT_         incx,         \
                                                      rocblas_stride   stridex,      \
                                                      const T_* const* y,            \
                                                      rocblas_stride   offsety,      \
                                                      API_INT_         incy,         \
                                                      rocblas_stride   stridey,      \
                                                      T_* const*       A,            \
                                                      rocblas_stride   offsetA,      \
                                                      API_INT_         lda,          \
                                                      rocblas_stride   strideA,      \
                                                      API_INT_         batch_count);

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *ger*.cpp

INST_GER_TEMPLATE_64(int64_t, float)
INST_GER_TEMPLATE_64(int64_t, double)
INST_GER_TEMPLATE_64(int64_t, rocblas_float_complex)
INST_GER_TEMPLATE_64(int64_t, rocblas_double_complex)

INST_GERC_TEMPLATE_64(int64_t, rocblas_float_complex)
INST_GERC_TEMPLATE_64(int64_t, rocblas_double_complex)

INST_GER_BATCHED_TEMPLATE_64(int64_t, float)
INST_GER_BATCHED_TEMPLATE_64(int64_t, double)
INST_GER_BATCHED_TEMPLATE_64(int64_t, rocblas_float_complex)
INST_GER_BATCHED_TEMPLATE_64(int64_t, rocblas_double_complex)

INST_GERC_BATCHED_TEMPLATE_64(int64_t, rocblas_float_complex)
INST_GERC_BATCHED_TEMPLATE_64(int64_t, rocblas_double_complex)
