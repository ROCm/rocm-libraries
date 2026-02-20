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

#include "blas1/rocblas_axpy.hpp"
#include "blas1/rocblas_axpy_kernels.hpp"
#include "int64_helpers.hpp"
#include "rocblas.h"
#include "rocblas_axpy_64.hpp"
#include "rocblas_block_sizes.h"

template <typename API_INT, rocblas_int NB, typename Tex, typename Ta, typename Tx, typename Ty>
rocblas_status rocblas_internal_axpy_launcher_64(rocblas_handle handle,
                                                 API_INT        n,
                                                 Ta             alpha,
                                                 rocblas_stride stride_alpha,
                                                 Tx             x,
                                                 rocblas_stride offset_x,
                                                 API_INT        incx,
                                                 rocblas_stride stride_x,
                                                 Ty             y,
                                                 rocblas_stride offset_y,
                                                 API_INT        incy,
                                                 rocblas_stride stride_y,
                                                 API_INT        batch_count)
{
    // Quick return if possible.
    if(n <= 0 || batch_count <= 0)
    {
        return rocblas_status_success;
    }

    if(incx < c_i32_max && incy < c_i32_max && incx > c_i32_min && incy > c_i32_min)
    {
        // increments can fit in int32_t
        for(int64_t b = 0; b < batch_count; b += c_i64_grid_YZ_chunk)
        {
            auto    x_ptr         = adjust_ptr_batch(x, b, stride_x);
            auto    y_ptr         = adjust_ptr_batch(y, b, stride_y);
            int32_t batch_count32 = int32_t(std::min(batch_count - b, c_i64_grid_YZ_chunk));

            auto alpha_ptr = rocblas_pointer_mode_device == handle->pointer_mode
                                 ? alpha + b * stride_alpha
                                 : alpha;

            for(int64_t n_base = 0; n_base < n; n_base += c_i64_grid_X_chunk)
            {
                int32_t n32 = int32_t(std::min(n - n_base, c_i64_grid_X_chunk));

                // If negative we inc we still need to start at end of array.
                // This offset shifts to start of 32-bit arrays beginning with last
                // 32-bit array, launcher shifts to end of array.
                int64_t shiftx
                    = offset_x + ((incx < 0) ? -incx * (n - n32 - n_base) : n_base * incx);
                int64_t shifty
                    = offset_y + ((incy < 0) ? -incy * (n - n32 - n_base) : n_base * incy);

                // 32-bit API call
                rocblas_status status
                    = rocblas_internal_axpy_launcher<rocblas_int, NB, Tex>(handle,
                                                                           rocblas_int(n32),
                                                                           alpha_ptr,
                                                                           stride_alpha,
                                                                           x_ptr,
                                                                           shiftx,
                                                                           rocblas_int(incx),
                                                                           stride_x,
                                                                           y_ptr,
                                                                           shifty,
                                                                           rocblas_int(incy),
                                                                           stride_y,
                                                                           batch_count32);

                if(status != rocblas_status_success)
                    return status;
            }
        }
    }
    else
    {
        // increments need 64 bits, can't ust 32-bit launcher
        for(int64_t b = 0; b < batch_count; b += c_i64_grid_YZ_chunk)
        {
            auto    x_ptr         = adjust_ptr_batch(x, b, stride_x);
            auto    y_ptr         = adjust_ptr_batch(y, b, stride_y);
            int32_t batch_count32 = int32_t(std::min(batch_count - b, c_i64_grid_YZ_chunk));

            for(int64_t n_base = 0; n_base < n; n_base += c_i64_grid_X_chunk)
            {
                int32_t n32 = int32_t(std::min(n - n_base, c_i64_grid_X_chunk));

                int  blocks = (n32 - 1) / NB + 1;
                dim3 grid(blocks, 1, batch_count32);
                dim3 threads(NB);

                // Not using launcher, so shifting to very end of 64-bit array
                int64_t shiftx
                    = offset_x + ((incx < 0) ? -incx * ((n - 1 - n_base)) : n_base * incx);
                int64_t shifty
                    = offset_y + ((incy < 0) ? -incy * ((n - 1 - n_base)) : n_base * incy);

                if(handle->pointer_mode == rocblas_pointer_mode_device)
                    ROCBLAS_LAUNCH_KERNEL((rocblas_axpy_kernel<int64_t, NB, Tex>),
                                          grid,
                                          threads,
                                          0,
                                          handle->get_stream(),
                                          n32,
                                          alpha + b * stride_alpha,
                                          stride_alpha,
                                          x_ptr,
                                          shiftx,
                                          incx,
                                          stride_x,
                                          y_ptr,
                                          shifty,
                                          incy,
                                          stride_y,
                                          batch_count);
                else
                    ROCBLAS_LAUNCH_KERNEL((rocblas_axpy_kernel<int64_t, NB, Tex>),
                                          grid,
                                          threads,
                                          0,
                                          handle->get_stream(),
                                          n32,
                                          *alpha,
                                          stride_alpha,
                                          x_ptr,
                                          shiftx,
                                          incx,
                                          stride_x,
                                          y_ptr,
                                          shifty,
                                          incy,
                                          stride_y,
                                          batch_count);
            }
        }
    }

    return rocblas_status_success;
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_axpy_template_64(rocblas_handle handle,
                                      int64_t        n,
                                      const T*       alpha,
                                      rocblas_stride stride_alpha,
                                      const T*       x,
                                      rocblas_stride offset_x,
                                      int64_t        incx,
                                      rocblas_stride stride_x,
                                      T*             y,
                                      rocblas_stride offset_y,
                                      int64_t        incy,
                                      rocblas_stride stride_y,
                                      int64_t        batch_count)
{
    return rocblas_internal_axpy_launcher_64<int64_t, ROCBLAS_AXPY_NB, T>(handle,
                                                                          n,
                                                                          alpha,
                                                                          stride_alpha,
                                                                          x,
                                                                          offset_x,
                                                                          incx,
                                                                          stride_x,
                                                                          y,
                                                                          offset_y,
                                                                          incy,
                                                                          stride_y,
                                                                          batch_count);
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_axpy_batched_template_64(rocblas_handle  handle,
                                              int64_t         n,
                                              const T*        alpha,
                                              rocblas_stride  stride_alpha,
                                              const T* const* x,
                                              rocblas_stride  offset_x,
                                              int64_t         incx,
                                              rocblas_stride  stride_x,
                                              T* const*       y,
                                              rocblas_stride  offset_y,
                                              int64_t         incy,
                                              rocblas_stride  stride_y,
                                              int64_t         batch_count)
{
    return rocblas_internal_axpy_launcher_64<int64_t, ROCBLAS_AXPY_NB, T>(handle,
                                                                          n,
                                                                          alpha,
                                                                          stride_alpha,
                                                                          x,
                                                                          offset_x,
                                                                          incx,
                                                                          stride_x,
                                                                          y,
                                                                          offset_y,
                                                                          incy,
                                                                          stride_y,
                                                                          batch_count);
}

#ifdef INSTANTIATE_AXPY_64_TEMPLATE
#error INSTANTIATE_AXPY_64_TEMPLATE already defined
#endif

#define INSTANTIATE_AXPY_64_TEMPLATE(T_)                                   \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status               \
        rocblas_internal_axpy_template_64<T_>(rocblas_handle handle,       \
                                              int64_t        n,            \
                                              const T_*      alpha,        \
                                              rocblas_stride stride_alpha, \
                                              const T_*      x,            \
                                              rocblas_stride offset_x,     \
                                              int64_t        incx,         \
                                              rocblas_stride stride_x,     \
                                              T_*            y,            \
                                              rocblas_stride offset_y,     \
                                              int64_t        incy,         \
                                              rocblas_stride stride_y,     \
                                              int64_t        batch_count);

INSTANTIATE_AXPY_64_TEMPLATE(rocblas_half)
INSTANTIATE_AXPY_64_TEMPLATE(float)
INSTANTIATE_AXPY_64_TEMPLATE(double)
INSTANTIATE_AXPY_64_TEMPLATE(rocblas_float_complex)
INSTANTIATE_AXPY_64_TEMPLATE(rocblas_double_complex)

#undef INSTANTIATE_AXPY_64_TEMPLATE

#ifdef INSTANTIATE_AXPY_BATCHED_64_TEMPLATE
#error INSTANTIATE_AXPY_BATCHED_64_TEMPLATE already defined
#endif

#define INSTANTIATE_AXPY_BATCHED_64_TEMPLATE(T_)                                     \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                         \
        rocblas_internal_axpy_batched_template_64<T_>(rocblas_handle   handle,       \
                                                      int64_t          n,            \
                                                      const T_*        alpha,        \
                                                      rocblas_stride   stride_alpha, \
                                                      const T_* const* x,            \
                                                      rocblas_stride   offset_x,     \
                                                      int64_t          incx,         \
                                                      rocblas_stride   stride_x,     \
                                                      T_* const*       y,            \
                                                      rocblas_stride   offset_y,     \
                                                      int64_t          incy,         \
                                                      rocblas_stride   stride_y,     \
                                                      int64_t          batch_count);

INSTANTIATE_AXPY_BATCHED_64_TEMPLATE(rocblas_half)
INSTANTIATE_AXPY_BATCHED_64_TEMPLATE(float)
INSTANTIATE_AXPY_BATCHED_64_TEMPLATE(double)
INSTANTIATE_AXPY_BATCHED_64_TEMPLATE(rocblas_float_complex)
INSTANTIATE_AXPY_BATCHED_64_TEMPLATE(rocblas_double_complex)

#undef INSTANTIATE_AXPY_BATCHED_64_TEMPLATE

#ifdef INSTANTIATE_AXPY_LAUNCHER
#error INSTANTIATE_AXPY_LAUNCHER already defined
#endif

#define INSTANTIATE_AXPY_LAUNCHER(NB_, Tex_, Ta_, Tx_, Ty_)                                       \
    template rocblas_status rocblas_internal_axpy_launcher_64<int64_t, NB_, Tex_, Ta_, Tx_, Ty_>( \
        rocblas_handle handle,                                                                    \
        int64_t        n,                                                                         \
        Ta_            alpha,                                                                     \
        rocblas_stride stride_alpha,                                                              \
        Tx_            x,                                                                         \
        rocblas_stride offset_x,                                                                  \
        int64_t        incx,                                                                      \
        rocblas_stride stride_x,                                                                  \
        Ty_            y,                                                                         \
        rocblas_stride offset_y,                                                                  \
        int64_t        incy,                                                                      \
        rocblas_stride stride_y,                                                                  \
        int64_t        batch_count);

// axpy_ex
INSTANTIATE_AXPY_LAUNCHER(
    ROCBLAS_AXPY_NB, float, const rocblas_bfloat16*, const rocblas_bfloat16*, rocblas_bfloat16*)
INSTANTIATE_AXPY_LAUNCHER(
    ROCBLAS_AXPY_NB, float, const float*, const rocblas_bfloat16*, rocblas_bfloat16*)
INSTANTIATE_AXPY_LAUNCHER(
    ROCBLAS_AXPY_NB, float, const rocblas_half*, const rocblas_half*, rocblas_half*)
INSTANTIATE_AXPY_LAUNCHER(ROCBLAS_AXPY_NB, float, const float*, const rocblas_half*, rocblas_half*)

// axpy_batched_ex
INSTANTIATE_AXPY_LAUNCHER(ROCBLAS_AXPY_NB,
                          float,
                          const rocblas_bfloat16*,
                          const rocblas_bfloat16* const*,
                          rocblas_bfloat16* const*)
INSTANTIATE_AXPY_LAUNCHER(
    ROCBLAS_AXPY_NB, float, const float*, const rocblas_bfloat16* const*, rocblas_bfloat16* const*)
INSTANTIATE_AXPY_LAUNCHER(
    ROCBLAS_AXPY_NB, float, const rocblas_half*, const rocblas_half* const*, rocblas_half* const*)
INSTANTIATE_AXPY_LAUNCHER(
    ROCBLAS_AXPY_NB, float, const float*, const rocblas_half* const*, rocblas_half* const*)

#undef INSTANTIATE_AXPY_LAUNCHER
// clang-format on
