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

#include "rocblas_spr_64.hpp"

#include "blas2/rocblas_spr.hpp" // int32 API called

template <typename API_INT, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_spr_launcher_64(rocblas_handle handle,
                                                rocblas_fill   uplo,
                                                API_INT        n_64,
                                                TScal const*   alpha,
                                                TConstPtr      x,
                                                rocblas_stride offset_x,
                                                int64_t        incx_64,
                                                rocblas_stride stride_x,
                                                TPtr           A,
                                                rocblas_stride offset_A,
                                                rocblas_stride stride_A,
                                                int64_t        batch_count_64)
{
    // Quick return if possible. Not Argument error
    if(!n_64 || !batch_count_64)
        return rocblas_status_success;

    // matrix A has size of at least (n * (n + 1)) / 2
    // n > INT_MAX => ~2^61 elements. For fp32 matrix A would need > 8192 petabytes of device memory
    if(n_64 > c_i32_max)
        return rocblas_status_invalid_size; // defer adding new kernels for sizes exceeding practical memory

    for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
    {
        auto    x_ptr       = adjust_ptr_batch(x, b_base, stride_x);
        auto    A_ptr       = adjust_ptr_batch(A, b_base, stride_A);
        int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

        auto shiftA = offset_A;

        rocblas_status status = rocblas_internal_spr_launcher(handle,
                                                              uplo,
                                                              (rocblas_int)n_64,
                                                              alpha,
                                                              x_ptr,
                                                              offset_x,
                                                              incx_64,
                                                              stride_x,
                                                              A_ptr,
                                                              shiftA,
                                                              stride_A,
                                                              batch_count);

        if(status != rocblas_status_success)
            return status;
    } // batch
    return rocblas_status_success;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *syr*.cpp

#ifdef INSTANTIATE_SPR_LAUNCHER_64
#error INSTANTIATE_SPR_LAUNCHER_64 already defined
#endif

#define INSTANTIATE_SPR_LAUNCHER_64(TI_, TScal_, TConstPtr_, TPtr_)                           \
    template rocblas_status rocblas_internal_spr_launcher_64<TI_, TScal_, TConstPtr_, TPtr_>( \
        rocblas_handle handle,                                                                \
        rocblas_fill   uplo,                                                                  \
        TI_            n,                                                                     \
        TScal_ const*  alpha,                                                                 \
        TConstPtr_     x,                                                                     \
        rocblas_stride offset_x,                                                              \
        int64_t        incx,                                                                  \
        rocblas_stride stride_x,                                                              \
        TPtr_          AP,                                                                    \
        rocblas_stride offset_A,                                                              \
        rocblas_stride stride_A,                                                              \
        int64_t        batch_count);

INSTANTIATE_SPR_LAUNCHER_64(int64_t, float, float const*, float*)
INSTANTIATE_SPR_LAUNCHER_64(int64_t, double, double const*, double*)
INSTANTIATE_SPR_LAUNCHER_64(int64_t,
                            rocblas_float_complex,
                            rocblas_float_complex const*,
                            rocblas_float_complex*)
INSTANTIATE_SPR_LAUNCHER_64(int64_t,
                            rocblas_double_complex,
                            rocblas_double_complex const*,
                            rocblas_double_complex*)
INSTANTIATE_SPR_LAUNCHER_64(int64_t, float, float const* const*, float* const*)
INSTANTIATE_SPR_LAUNCHER_64(int64_t, double, double const* const*, double* const*)
INSTANTIATE_SPR_LAUNCHER_64(int64_t,
                            rocblas_float_complex,
                            rocblas_float_complex const* const*,
                            rocblas_float_complex* const*)
INSTANTIATE_SPR_LAUNCHER_64(int64_t,
                            rocblas_double_complex,
                            rocblas_double_complex const* const*,
                            rocblas_double_complex* const*)

#undef INSTANTIATE_SPR_LAUNCHER_64
