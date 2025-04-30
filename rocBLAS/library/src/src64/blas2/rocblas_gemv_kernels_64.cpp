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
#include "rocblas_gemv_64.hpp"

#include "blas2/rocblas_gemv.hpp" // int32 API called

template <typename Ti, typename Tex, typename To>
__attribute__((noinline)) rocblas_status
    rocblas_internal_gemv_launcher_64(rocblas_handle    handle,
                                      rocblas_operation transA,
                                      int64_t           m_64,
                                      int64_t           n_64,
                                      Tex const*        alpha,
                                      rocblas_stride    stride_alpha,
                                      Ti const*         A,
                                      rocblas_stride    offsetA,
                                      int64_t           lda_64,
                                      rocblas_stride    strideA,
                                      Ti const*         x,
                                      rocblas_stride    offsetx,
                                      int64_t           incx_64,
                                      rocblas_stride    stridex,
                                      Tex const*        beta,
                                      rocblas_stride    stride_beta,
                                      To*               y,
                                      rocblas_stride    offsety,
                                      int64_t           incy_64,
                                      rocblas_stride    stridey,
                                      int64_t           batch_count_64,
                                      Tex*              workspace)
{
    // Quick return if possible. Not Argument error
    if(!m_64 || !n_64 || !batch_count_64)
        return rocblas_status_success;

    bool dims_32bit = m_64 <= c_ILP64_i32_max && n_64 <= c_ILP64_i32_max;

    Tex* beta_one{nullptr};
    Tex  one(1.0);
    if(!dims_32bit)
    {
        if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
            beta_one = workspace++;
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                beta_one, &one, sizeof(Tex), hipMemcpyHostToDevice, handle->get_stream()));
        }
        else
        {
            beta_one = &one;
        }
    }

    for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
    {
        auto    x_ptr       = adjust_ptr_batch(x, b_base, stridex);
        auto    y_ptr       = adjust_ptr_batch(y, b_base, stridey);
        auto    A_ptr       = adjust_ptr_batch(A, b_base, strideA);
        auto    alpha_ptr   = adjust_ptr_batch(alpha, b_base, stride_alpha);
        auto    beta_ptr    = adjust_ptr_batch(beta, b_base, stride_beta);
        int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

        if(dims_32bit)
        {
            rocblas_status status = rocblas_internal_gemv_launcher(handle,
                                                                   transA,
                                                                   (int)m_64,
                                                                   (int)n_64,
                                                                   alpha_ptr,
                                                                   stride_alpha,
                                                                   A_ptr,
                                                                   offsetA,
                                                                   lda_64,
                                                                   strideA,
                                                                   x_ptr,
                                                                   offsetx,
                                                                   incx_64,
                                                                   stridex,
                                                                   beta_ptr,
                                                                   stride_beta,
                                                                   y_ptr,
                                                                   offsety,
                                                                   incy_64,
                                                                   stridey,
                                                                   batch_count,
                                                                   workspace);
            if(status != rocblas_status_success)
                return status;
        }
        else if(transA == rocblas_operation_none)
        {
            // outer loop over non reduction
            for(int64_t m_base = 0; m_base < m_64; m_base += c_i64_grid_X_chunk)
            {
                int32_t m = int32_t(std::min(m_64 - m_base, c_i64_grid_X_chunk));

                int64_t shifty
                    = offsety + (incy_64 < 0 ? -incy_64 * (m_64 - m - m_base) : m_base * incy_64);

                // reduction loop
                for(int64_t n_base = 0; n_base < n_64; n_base += c_i64_grid_YZ_chunk)
                {
                    int32_t n = int32_t(std::min(n_64 - n_base, c_i64_grid_YZ_chunk));

                    int64_t shiftx
                        = offsetx
                          + (incx_64 < 0 ? -incx_64 * (n_64 - n - n_base) : n_base * incx_64);

                    rocblas_status status
                        = rocblas_internal_gemv_launcher(handle,
                                                         transA,
                                                         m,
                                                         n,
                                                         alpha_ptr,
                                                         stride_alpha,
                                                         A_ptr,
                                                         offsetA + m_base + n_base * lda_64,
                                                         lda_64,
                                                         strideA,
                                                         x_ptr,
                                                         shiftx,
                                                         incx_64,
                                                         stridex,
                                                         n_base == 0 ? beta_ptr : beta_one,
                                                         n_base == 0 ? stride_beta : 0,
                                                         y_ptr, // used for accumulation
                                                         shifty,
                                                         incy_64,
                                                         stridey,
                                                         batch_count,
                                                         workspace);
                    if(status != rocblas_status_success)
                        return status;
                }
            }
        }
        else // if(transA != rocblas_operation_none)
        {

            // outer loop over non reduction
            for(int64_t n_base = 0; n_base < n_64; n_base += c_i64_grid_YZ_chunk)
            {
                int32_t n = int32_t(std::min(n_64 - n_base, c_i64_grid_YZ_chunk));

                int64_t shifty
                    = offsety + (incy_64 < 0 ? -incy_64 * (n_64 - n - n_base) : n_base * incy_64);

                // reduction loop
                for(int64_t m_base = 0; m_base < m_64; m_base += c_i64_grid_X_chunk)
                {
                    int32_t m = int32_t(std::min(m_64 - m_base, c_i64_grid_X_chunk));

                    int64_t shiftx
                        = offsetx
                          + (incx_64 < 0 ? -incx_64 * (m_64 - m - m_base) : m_base * incx_64);

                    rocblas_status status
                        = rocblas_internal_gemv_launcher(handle,
                                                         transA,
                                                         m,
                                                         n,
                                                         alpha_ptr,
                                                         stride_alpha,
                                                         A_ptr,
                                                         offsetA + n_base * lda_64 + m_base,
                                                         lda_64,
                                                         strideA,
                                                         x_ptr,
                                                         shiftx,
                                                         incx_64,
                                                         stridex,
                                                         m_base == 0 ? beta_ptr : beta_one,
                                                         m_base == 0 ? stride_beta : 0,
                                                         y_ptr, // used for accumulation
                                                         shifty,
                                                         incy_64,
                                                         stridey,
                                                         batch_count,
                                                         workspace);
                    if(status != rocblas_status_success)
                        return status;
                }
            }
        }

    } // batch

    return rocblas_status_success;
}

template <typename T>
rocblas_status rocblas_internal_gemv_template_64(rocblas_handle    handle,
                                                 rocblas_operation transA,
                                                 int64_t           m,
                                                 int64_t           n,
                                                 const T*          alpha,
                                                 rocblas_stride    stride_alpha,
                                                 const T*          A,
                                                 rocblas_stride    offseta,
                                                 int64_t           lda,
                                                 rocblas_stride    strideA,
                                                 const T*          x,
                                                 rocblas_stride    offsetx,
                                                 int64_t           incx,
                                                 rocblas_stride    stridex,
                                                 const T*          beta,
                                                 rocblas_stride    stride_beta,
                                                 T*                y,
                                                 rocblas_stride    offsety,
                                                 int64_t           incy,
                                                 rocblas_stride    stridey,
                                                 int64_t           batch_count,
                                                 T*                workspace)
{
    return rocblas_internal_gemv_launcher_64<T, T, T>(handle,
                                                      transA,
                                                      m,
                                                      n,
                                                      alpha,
                                                      stride_alpha,
                                                      A,
                                                      offseta,
                                                      lda,
                                                      strideA,
                                                      x,
                                                      offsetx,
                                                      incx,
                                                      stridex,
                                                      beta,
                                                      stride_beta,
                                                      y,
                                                      offsety,
                                                      incy,
                                                      stridey,
                                                      batch_count,
                                                      workspace);
}

template <typename T>
rocblas_status rocblas_internal_gemv_batched_template_64(rocblas_handle    handle,
                                                         rocblas_operation transA,
                                                         int64_t           m,
                                                         int64_t           n,
                                                         const T*          alpha,
                                                         rocblas_stride    stride_alpha,
                                                         const T* const*   A,
                                                         rocblas_stride    offseta,
                                                         int64_t           lda,
                                                         rocblas_stride    strideA,
                                                         const T* const*   x,
                                                         rocblas_stride    offsetx,
                                                         int64_t           incx,
                                                         rocblas_stride    stridex,
                                                         const T*          beta,
                                                         rocblas_stride    stride_beta,
                                                         T* const*         y,
                                                         rocblas_stride    offsety,
                                                         int64_t           incy,
                                                         rocblas_stride    stridey,
                                                         int64_t           batch_count,
                                                         T*                workspace)
{
    return rocblas_internal_gemv_launcher_64(handle,
                                             transA,
                                             m,
                                             n,
                                             alpha,
                                             stride_alpha,
                                             A,
                                             offseta,
                                             lda,
                                             strideA,
                                             x,
                                             offsetx,
                                             incx,
                                             stridex,
                                             beta,
                                             stride_beta,
                                             y,
                                             offsety,
                                             incy,
                                             stridey,
                                             batch_count,
                                             workspace);
}

#ifdef INST_GEMV_TEMPLATE_64
#error INST_GEMV_TEMPLATE_64 already defined
#endif

#define INST_GEMV_TEMPLATE_64(T_)                                             \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                  \
        rocblas_internal_gemv_template_64<T_>(rocblas_handle    handle,       \
                                              rocblas_operation transA,       \
                                              int64_t           m,            \
                                              int64_t           n,            \
                                              const T_*         alpha,        \
                                              rocblas_stride    stride_alpha, \
                                              const T_*         A,            \
                                              rocblas_stride    offseta,      \
                                              int64_t           lda,          \
                                              rocblas_stride    strideA,      \
                                              const T_*         x,            \
                                              rocblas_stride    offsetx,      \
                                              int64_t           incx,         \
                                              rocblas_stride    stridex,      \
                                              const T_*         beta,         \
                                              rocblas_stride    stride_beta,  \
                                              T_*               y,            \
                                              rocblas_stride    offsety,      \
                                              int64_t           incy,         \
                                              rocblas_stride    stridey,      \
                                              int64_t           batch_count,  \
                                              T_*               workspace);

INST_GEMV_TEMPLATE_64(float)
INST_GEMV_TEMPLATE_64(double)
INST_GEMV_TEMPLATE_64(rocblas_float_complex)
INST_GEMV_TEMPLATE_64(rocblas_double_complex)

#undef INST_GEMV_TEMPLATE_64

#ifdef INST_GEMV_BATCHED_TEMPLATE_64
#error INST_GEMV_BATCHED_TEMPLATE_64 already defined
#endif

#define INST_GEMV_BATCHED_TEMPLATE_64(T_)                                             \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                          \
        rocblas_internal_gemv_batched_template_64<T_>(rocblas_handle    handle,       \
                                                      rocblas_operation transA,       \
                                                      int64_t           m,            \
                                                      int64_t           n,            \
                                                      const T_*         alpha,        \
                                                      rocblas_stride    stride_alpha, \
                                                      const T_* const*  A,            \
                                                      rocblas_stride    offseta,      \
                                                      int64_t           lda,          \
                                                      rocblas_stride    strideA,      \
                                                      const T_* const*  x,            \
                                                      rocblas_stride    offsetx,      \
                                                      int64_t           incx,         \
                                                      rocblas_stride    stridex,      \
                                                      const T_*         beta,         \
                                                      rocblas_stride    stride_beta,  \
                                                      T_* const*        y,            \
                                                      rocblas_stride    offsety,      \
                                                      int64_t           incy,         \
                                                      rocblas_stride    stridey,      \
                                                      int64_t           batch_count,  \
                                                      T_*               workspace);

INST_GEMV_BATCHED_TEMPLATE_64(float)
INST_GEMV_BATCHED_TEMPLATE_64(double)
INST_GEMV_BATCHED_TEMPLATE_64(rocblas_float_complex)
INST_GEMV_BATCHED_TEMPLATE_64(rocblas_double_complex)

#undef INST_GEMV_BATCHED_TEMPLATE_64

// For mixed-precision gemv and used by trsv
#ifdef INST_GEMV_MIXED_LAUNCHER_64
#error INST_GEMV_MIXED_LAUNCHER_64 already defined
#endif

#define INST_GEMV_MIXED_LAUNCHER_64(Ti_, Tex_, To_)                            \
    template rocblas_status rocblas_internal_gemv_launcher_64<Ti_, Tex_, To_>( \
        rocblas_handle    handle,                                              \
        rocblas_operation transA,                                              \
        int64_t           m,                                                   \
        int64_t           n,                                                   \
        const Tex_*       alpha,                                               \
        rocblas_stride    stride_alpha,                                        \
        Ti_ const*        A,                                                   \
        rocblas_stride    offseta,                                             \
        int64_t           lda,                                                 \
        rocblas_stride    strideA,                                             \
        Ti_ const*        x,                                                   \
        rocblas_stride    offsetx,                                             \
        int64_t           incx,                                                \
        rocblas_stride    stridex,                                             \
        const Tex_*       beta,                                                \
        rocblas_stride    stride_beta,                                         \
        To_*              y,                                                   \
        rocblas_stride    offsety,                                             \
        int64_t           incy,                                                \
        rocblas_stride    stridey,                                             \
        int64_t           batch_count,                                         \
        Tex_*             workspace);

// non _template APIs
INST_GEMV_MIXED_LAUNCHER_64(rocblas_half, float, rocblas_half)
INST_GEMV_MIXED_LAUNCHER_64(rocblas_half const*, float, rocblas_half* const)
INST_GEMV_MIXED_LAUNCHER_64(rocblas_half, float, float)
INST_GEMV_MIXED_LAUNCHER_64(rocblas_half const*, float, float* const)
INST_GEMV_MIXED_LAUNCHER_64(rocblas_bfloat16, float, rocblas_bfloat16)
INST_GEMV_MIXED_LAUNCHER_64(rocblas_bfloat16 const*, float, rocblas_bfloat16* const)
INST_GEMV_MIXED_LAUNCHER_64(rocblas_bfloat16, float, float)
INST_GEMV_MIXED_LAUNCHER_64(rocblas_bfloat16 const*, float, float* const)

#undef INST_GEMV_MIXED_LAUNCHER_64
