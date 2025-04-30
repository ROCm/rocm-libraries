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
#include "rocblas_block_sizes.h"
#include "rocblas_scal_64.hpp"

#include "blas1/rocblas_scal.hpp" // rocblas_int API called
#include "blas1/rocblas_scal_kernels.hpp" // inst kernels with int64_t

template <typename API_INT, int NB, typename T, typename Tex, typename Ta, typename Tx>
rocblas_status rocblas_internal_scal_launcher_64(rocblas_handle handle,
                                                 API_INT        n_64,
                                                 const Ta*      alpha,
                                                 rocblas_stride stride_alpha,
                                                 Tx             x,
                                                 rocblas_stride offset_x,
                                                 API_INT        incx_64,
                                                 rocblas_stride stride_x,
                                                 API_INT        batch_count_64)
{
    // Quick return if possible. Not Argument error
    if(n_64 <= 0 || incx_64 <= 0 || batch_count_64 <= 0)
    {
        return rocblas_status_success;
    }

    if(incx_64 <= c_ILP64_i32_max)
    {
        for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
        {
            auto    x_ptr       = adjust_ptr_batch(x, b_base, stride_x);
            int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

            for(int64_t n_base = 0; n_base < n_64; n_base += c_i64_grid_X_chunk)
            {
                int32_t n         = int32_t(std::min(n_64 - n_base, c_i64_grid_X_chunk));
                auto    alpha_ptr = rocblas_pointer_mode_device == handle->pointer_mode
                                        ? alpha + n_base * stride_alpha
                                        : alpha;

                // 32bit API call
                rocblas_status status
                    = rocblas_internal_scal_launcher<rocblas_int, NB, T, Tex, Ta, Tx>(
                        handle,
                        rocblas_int(n),
                        alpha_ptr,
                        stride_alpha,
                        x_ptr,
                        offset_x + n_base * incx_64, // incx > 0
                        rocblas_int(incx_64),
                        stride_x,
                        batch_count);
                if(status != rocblas_status_success)
                    return status;
            }
        }
    }
    else
    {
        for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
        {
            auto    x_ptr       = adjust_ptr_batch(x, b_base, stride_x);
            int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

            for(int64_t n_base = 0; n_base < n_64; n_base += c_i64_grid_X_chunk)
            {
                int32_t n = int32_t(std::min(n_64 - n_base, c_i64_grid_X_chunk));

                int  blocks = (n - 1) / NB + 1;
                dim3 grid(blocks, 1, batch_count);
                dim3 threads(NB);

                int64_t shiftx = offset_x + n_base * incx_64;

                if(rocblas_pointer_mode_device == handle->pointer_mode)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_scal_kernel<int64_t, NB, T, Tex>),
                                          grid,
                                          threads,
                                          0,
                                          handle->get_stream(),
                                          n,
                                          alpha + n_base * stride_alpha,
                                          stride_alpha,
                                          x_ptr,
                                          shiftx,
                                          incx_64,
                                          stride_x,
                                          batch_count);
                else // single alpha is on host
                    ROCBLAS_LAUNCH_KERNEL((rocblas_scal_kernel<int64_t, NB, T, Tex>),
                                          grid,
                                          threads,
                                          0,
                                          handle->get_stream(),
                                          n,
                                          *alpha,
                                          stride_alpha,
                                          x_ptr,
                                          shiftx,
                                          incx_64,
                                          stride_x,
                                          batch_count);
            }
        }
    }

    return rocblas_status_success;
}

template <typename T, typename Ta>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_scal_template_64(rocblas_handle handle,
                                      int64_t        n,
                                      const Ta*      alpha,
                                      rocblas_stride stride_alpha,
                                      T*             x,
                                      rocblas_stride offset_x,
                                      int64_t        incx,
                                      rocblas_stride stride_x,
                                      int64_t        batch_count)
{
    return rocblas_internal_scal_launcher_64<int64_t, ROCBLAS_SCAL_NB, T, T>(
        handle, n, alpha, stride_alpha, x, offset_x, incx, stride_x, batch_count);
}

template <typename T, typename Ta>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_scal_batched_template_64(rocblas_handle handle,
                                              int64_t        n,
                                              const Ta*      alpha,
                                              rocblas_stride stride_alpha,
                                              T* const*      x,
                                              rocblas_stride offset_x,
                                              int64_t        incx,
                                              rocblas_stride stride_x,
                                              int64_t        batch_count)
{
    return rocblas_internal_scal_launcher_64<int64_t, ROCBLAS_SCAL_NB, T, T>(
        handle, n, alpha, stride_alpha, x, offset_x, incx, stride_x, batch_count);
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files scal*.cpp

// clang-format off
#ifdef INST_SCAL_TEMPLATE
#error INST_SCAL_TEMPLATE already defined
#endif

#define INST_SCAL_TEMPLATE(T_, Ta_)                              \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status            \
        rocblas_internal_scal_template_64<T_, Ta_>(rocblas_handle handle,  \
                                           int64_t    n,            \
                                           const Ta_*     alpha,        \
                                           rocblas_stride stride_alpha, \
                                           T_*            x,            \
                                           rocblas_stride offset_x,     \
                                           int64_t    incx,         \
                                           rocblas_stride stride_x,     \
                                           int64_t    batch_count);

// Not exporting execution type
INST_SCAL_TEMPLATE(rocblas_half, rocblas_half)
INST_SCAL_TEMPLATE(rocblas_half, float)
INST_SCAL_TEMPLATE(float, float)
INST_SCAL_TEMPLATE(double, double)
INST_SCAL_TEMPLATE(rocblas_float_complex, rocblas_float_complex)
INST_SCAL_TEMPLATE(rocblas_double_complex, rocblas_double_complex)
INST_SCAL_TEMPLATE(rocblas_float_complex, float)
INST_SCAL_TEMPLATE(rocblas_double_complex, double)

#undef INST_SCAL_TEMPLATE

#ifdef INST_SCAL_BATCHED_TEMPLATE
#error INST_SCAL_BATCHED_TEMPLATE already defined
#endif

#define INST_SCAL_BATCHED_TEMPLATE(T_, Ta_)                              \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                    \
        rocblas_internal_scal_batched_template_64<T_, Ta_>(rocblas_handle handle,  \
                                                   int64_t    n,            \
                                                   const Ta_*     alpha,        \
                                                   rocblas_stride stride_alpha, \
                                                   T_* const*     x,            \
                                                   rocblas_stride offset_x,     \
                                                   int64_t    incx,         \
                                                   rocblas_stride stride_x,     \
                                                   int64_t    batch_count);

INST_SCAL_BATCHED_TEMPLATE(rocblas_half, rocblas_half)
INST_SCAL_BATCHED_TEMPLATE(rocblas_half, float)
INST_SCAL_BATCHED_TEMPLATE(float, float)
INST_SCAL_BATCHED_TEMPLATE(double, double)
INST_SCAL_BATCHED_TEMPLATE(rocblas_float_complex, rocblas_float_complex)
INST_SCAL_BATCHED_TEMPLATE(rocblas_double_complex, rocblas_double_complex)
INST_SCAL_BATCHED_TEMPLATE(rocblas_float_complex, float)
INST_SCAL_BATCHED_TEMPLATE(rocblas_double_complex, double)

#undef INST_SCAL_BATCHED_TEMPLATE

#ifdef INST_SCAL_EX_LAUNCHER
#error INST_SCAL_EX_LAUNCHER already defined
#endif

#define INST_SCAL_EX_LAUNCHER(NB_, T_, Tex_, Ta_, Tx_)                                \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                                 \
        rocblas_internal_scal_launcher_64<int64_t, NB_, T_, Tex_, Ta_, Tx_>(rocblas_handle handle,       \
                                                                int64_t    n,            \
                                                                const Ta_*     alpha,        \
                                                                rocblas_stride stride_alpha, \
                                                                Tx_            x,            \
                                                                rocblas_stride offset_x,     \
                                                                int64_t    incx,         \
                                                                rocblas_stride stride_x,     \
                                                                int64_t    batch_count);

// Instantiations for scal_ex
INST_SCAL_EX_LAUNCHER(ROCBLAS_SCAL_NB, rocblas_half, float, rocblas_half, rocblas_half*)
INST_SCAL_EX_LAUNCHER(ROCBLAS_SCAL_NB, rocblas_half, float, float, rocblas_half*)
INST_SCAL_EX_LAUNCHER(ROCBLAS_SCAL_NB, rocblas_bfloat16, float, rocblas_bfloat16, rocblas_bfloat16*)
INST_SCAL_EX_LAUNCHER(ROCBLAS_SCAL_NB, rocblas_bfloat16, float, float, rocblas_bfloat16*)

INST_SCAL_EX_LAUNCHER(ROCBLAS_SCAL_NB, rocblas_half, float, rocblas_half, rocblas_half* const*)
INST_SCAL_EX_LAUNCHER(ROCBLAS_SCAL_NB, rocblas_half, float, float, rocblas_half* const*)
INST_SCAL_EX_LAUNCHER(ROCBLAS_SCAL_NB, rocblas_bfloat16, float, rocblas_bfloat16, rocblas_bfloat16* const*)
INST_SCAL_EX_LAUNCHER(ROCBLAS_SCAL_NB, rocblas_bfloat16, float, float, rocblas_bfloat16* const*)

INST_SCAL_EX_LAUNCHER(ROCBLAS_SCAL_NB, double, double, double, double*)
INST_SCAL_EX_LAUNCHER(ROCBLAS_SCAL_NB, double, double, double, double* const*)
INST_SCAL_EX_LAUNCHER(ROCBLAS_SCAL_NB, rocblas_float_complex, rocblas_float_complex, rocblas_float_complex, rocblas_float_complex*)
INST_SCAL_EX_LAUNCHER(ROCBLAS_SCAL_NB, rocblas_float_complex, rocblas_float_complex, rocblas_float_complex, rocblas_float_complex* const*)

#undef INST_SCAL_EX_LAUNCHER

// clang-format on
