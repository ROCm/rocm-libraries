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

#include "int64_helpers.hpp"
#include "rocblas.h"
#include "rocblas_block_sizes.h"
#include "rocblas_rot_64.hpp"

#include "blas1/rocblas_rot.hpp" // int32 API called
#include "blas1/rocblas_rot_kernels.hpp"

template <typename API_INT,
          rocblas_int NB,
          typename Tex,
          typename Tx,
          typename Ty,
          typename Tc,
          typename Ts>
rocblas_status rocblas_internal_rot_launcher_64(rocblas_handle handle,
                                                API_INT        n_64,
                                                Tx             x,
                                                rocblas_stride offset_x,
                                                int64_t        incx_64,
                                                rocblas_stride stride_x,
                                                Ty             y,
                                                rocblas_stride offset_y,
                                                int64_t        incy_64,
                                                rocblas_stride stride_y,
                                                Tc*            c,
                                                rocblas_stride c_stride,
                                                Ts*            s,
                                                rocblas_stride s_stride,
                                                API_INT        batch_count_64)
{
    // Quick return if possible
    if(n_64 <= 0 || batch_count_64 <= 0)
        return rocblas_status_success;

    if(std::abs(incx_64) <= c_ILP64_i32_max && std::abs(incy_64) < c_ILP64_i32_max)
    {
        if(n_64 <= c_ILP64_i32_max && batch_count_64 < c_i64_grid_YZ_chunk)
        {
            // valid to use original 32bit API with truncated 64bit args
            return rocblas_internal_rot_launcher<rocblas_int, NB, Tex, Tx>(handle,
                                                                           rocblas_int(n_64),
                                                                           x,
                                                                           offset_x,
                                                                           incx_64,
                                                                           stride_x,
                                                                           y,
                                                                           offset_y,
                                                                           incy_64,
                                                                           stride_y,
                                                                           c,
                                                                           c_stride,
                                                                           s,
                                                                           s_stride,
                                                                           batch_count_64);
        }

        for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
        {
            auto x_ptr = adjust_ptr_batch(x, b_base, stride_x);
            auto y_ptr = adjust_ptr_batch(y, b_base, stride_y);
            auto c_ptr = adjust_ptr_batch(c, b_base, c_stride);
            auto s_ptr = adjust_ptr_batch(s, b_base, s_stride);

            int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

            for(int64_t n_base = 0; n_base < n_64; n_base += c_i64_grid_X_chunk)
            {
                int32_t n = int32_t(std::min(n_64 - n_base, c_i64_grid_X_chunk));
                // 32bit API call as incx/y int64_t
                rocblas_status status = rocblas_internal_rot_launcher<rocblas_int, NB, Tex, Tx>(
                    handle,
                    rocblas_int(n),
                    x_ptr,
                    offset_x + n_base * incx_64,
                    incx_64,
                    stride_x,
                    y_ptr,
                    offset_y + n_base * incy_64,
                    incy_64,
                    stride_y,
                    c_ptr,
                    c_stride,
                    s_ptr,
                    s_stride,
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
            auto x_ptr = adjust_ptr_batch(x, b_base, stride_x);
            auto y_ptr = adjust_ptr_batch(y, b_base, stride_y);
            auto c_ptr = adjust_ptr_batch(c, b_base, c_stride);
            auto s_ptr = adjust_ptr_batch(s, b_base, s_stride);

            int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

            for(int64_t n_base = 0; n_base < n_64; n_base += c_i64_grid_X_chunk)
            {
                int32_t n = int32_t(std::min(n_64 - n_base, c_i64_grid_X_chunk));

                int64_t shiftx = incx_64 < 0 ? -incx_64 * n_base : incx_64 * n_base;
                int64_t shifty = incy_64 < 0 ? -incy_64 * n_base : incy_64 * n_base;

                shiftx += offset_x;
                shifty += offset_y;

                // new instantiation for 64bit incx/y
                rocblas_status status
                    = rocblas_internal_rot_launcher<rocblas_int, NB, Tex, Tx>(handle,
                                                                              n,
                                                                              x_ptr,
                                                                              shiftx,
                                                                              incx_64,
                                                                              stride_x,
                                                                              y_ptr,
                                                                              shifty,
                                                                              incy_64,
                                                                              stride_y,
                                                                              c_ptr,
                                                                              c_stride,
                                                                              s_ptr,
                                                                              s_stride,
                                                                              batch_count);

                if(status != rocblas_status_success)
                    return status;
            }
        }
    }
    return rocblas_status_success;
}

#ifdef INSTANTIATE_ROT_LAUNCHER_64
#error INSTANTIATE_ROT_LAUNCHER_64 already defined
#endif

#define INSTANTIATE_ROT_LAUNCHER_64(NB_, Tex_, Tx_, Ty_, Tc_, Ts_)                \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                      \
        rocblas_internal_rot_launcher_64<int64_t, NB_, Tex_, Tx_, Ty_, Tc_, Ts_>( \
            rocblas_handle handle,                                                \
            int64_t        n,                                                     \
            Tx_            x,                                                     \
            rocblas_stride offset_x,                                              \
            int64_t        incx,                                                  \
            rocblas_stride stride_x,                                              \
            Ty_            y,                                                     \
            rocblas_stride offset_y,                                              \
            int64_t        incy,                                                  \
            rocblas_stride stride_y,                                              \
            Tc_ * c,                                                              \
            rocblas_stride c_stride,                                              \
            Ts_ * s,                                                              \
            rocblas_stride s_stride,                                              \
            int64_t        batch_count);

//  instantiate for rocblas_Xrot and rocblas_Xrot_strided_batched
INSTANTIATE_ROT_LAUNCHER_64(ROCBLAS_ROT_NB, float, float*, float*, float const, float const)
INSTANTIATE_ROT_LAUNCHER_64(ROCBLAS_ROT_NB, double, double*, double*, double const, double const)
INSTANTIATE_ROT_LAUNCHER_64(ROCBLAS_ROT_NB,
                            float,
                            rocblas_bfloat16*,
                            rocblas_bfloat16*,
                            rocblas_bfloat16 const,
                            rocblas_bfloat16 const)
INSTANTIATE_ROT_LAUNCHER_64(
    ROCBLAS_ROT_NB, float, rocblas_half*, rocblas_half*, rocblas_half const, rocblas_half const)
INSTANTIATE_ROT_LAUNCHER_64(ROCBLAS_ROT_NB,
                            rocblas_float_complex,
                            rocblas_float_complex*,
                            rocblas_float_complex*,
                            float const,
                            float const)
INSTANTIATE_ROT_LAUNCHER_64(ROCBLAS_ROT_NB,
                            rocblas_float_complex,
                            rocblas_float_complex*,
                            rocblas_float_complex*,
                            float const,
                            rocblas_float_complex const)
INSTANTIATE_ROT_LAUNCHER_64(ROCBLAS_ROT_NB,
                            rocblas_float_complex,
                            rocblas_float_complex*,
                            rocblas_float_complex*,
                            rocblas_float_complex const,
                            rocblas_float_complex const)
INSTANTIATE_ROT_LAUNCHER_64(ROCBLAS_ROT_NB,
                            rocblas_double_complex,
                            rocblas_double_complex*,
                            rocblas_double_complex*,
                            double const,
                            double const)
INSTANTIATE_ROT_LAUNCHER_64(ROCBLAS_ROT_NB,
                            rocblas_double_complex,
                            rocblas_double_complex*,
                            rocblas_double_complex*,
                            double const,
                            rocblas_double_complex const)
INSTANTIATE_ROT_LAUNCHER_64(ROCBLAS_ROT_NB,
                            rocblas_double_complex,
                            rocblas_double_complex*,
                            rocblas_double_complex*,
                            rocblas_double_complex const,
                            rocblas_double_complex const)

//  instantiate for rocblas_Xrot__batched
INSTANTIATE_ROT_LAUNCHER_64(
    ROCBLAS_ROT_NB, float, float* const*, float* const*, float const, float const)
INSTANTIATE_ROT_LAUNCHER_64(
    ROCBLAS_ROT_NB, double, double* const*, double* const*, double const, double const)
INSTANTIATE_ROT_LAUNCHER_64(ROCBLAS_ROT_NB,
                            float,
                            rocblas_bfloat16* const*,
                            rocblas_bfloat16* const*,
                            rocblas_bfloat16 const,
                            rocblas_bfloat16 const)
INSTANTIATE_ROT_LAUNCHER_64(ROCBLAS_ROT_NB,
                            float,
                            rocblas_half* const*,
                            rocblas_half* const*,
                            rocblas_half const,
                            rocblas_half const)
INSTANTIATE_ROT_LAUNCHER_64(ROCBLAS_ROT_NB,
                            rocblas_float_complex,
                            rocblas_float_complex* const*,
                            rocblas_float_complex* const*,
                            float const,
                            float const)
INSTANTIATE_ROT_LAUNCHER_64(ROCBLAS_ROT_NB,
                            rocblas_float_complex,
                            rocblas_float_complex* const*,
                            rocblas_float_complex* const*,
                            float const,
                            rocblas_float_complex const)
INSTANTIATE_ROT_LAUNCHER_64(ROCBLAS_ROT_NB,
                            rocblas_float_complex,
                            rocblas_float_complex* const*,
                            rocblas_float_complex* const*,
                            rocblas_float_complex const,
                            rocblas_float_complex const)
INSTANTIATE_ROT_LAUNCHER_64(ROCBLAS_ROT_NB,
                            rocblas_double_complex,
                            rocblas_double_complex* const*,
                            rocblas_double_complex* const*,
                            rocblas_double_complex const,
                            rocblas_double_complex const)
INSTANTIATE_ROT_LAUNCHER_64(ROCBLAS_ROT_NB,
                            rocblas_double_complex,
                            rocblas_double_complex* const*,
                            rocblas_double_complex* const*,
                            double const,
                            double const)
INSTANTIATE_ROT_LAUNCHER_64(ROCBLAS_ROT_NB,
                            rocblas_double_complex,
                            rocblas_double_complex* const*,
                            rocblas_double_complex* const*,
                            double const,
                            rocblas_double_complex const)
#undef INSTANTIATE_ROT_LAUNCHER_64
// clang-format on
