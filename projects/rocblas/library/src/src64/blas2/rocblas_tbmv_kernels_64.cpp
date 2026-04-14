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

#include "rocblas_tbmv_64.hpp"

#include "blas2/rocblas_tbmv.hpp" // int32 API called

template <typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_tbmv_launcher_64(rocblas_handle    handle,
                                                 rocblas_fill      uplo,
                                                 rocblas_operation transA,
                                                 rocblas_diagonal  diag,
                                                 int64_t           n_64,
                                                 int64_t           k_64,
                                                 TConstPtr         A,
                                                 rocblas_stride    offseta,
                                                 int64_t           lda,
                                                 rocblas_stride    strideA,
                                                 TPtr              x,
                                                 rocblas_stride    offsetx,
                                                 int64_t           incx,
                                                 rocblas_stride    stridex,
                                                 int64_t           batch_count_64,
                                                 TPtr              w_x_copy)
{
    // Quick return if possible. Not Argument error
    if(!n_64 || !batch_count_64)
        return rocblas_status_success;

    if(n_64 > c_i32_max
       || k_64 > c_i32_max) // TODO: This isn't true for banded matrices, may need new kernels.
        return rocblas_status_invalid_size; // defer adding new kernels for sizes exceeding practical memory

    for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
    {
        auto    x_ptr       = adjust_ptr_batch(x, b_base, stridex);
        auto    A_ptr       = adjust_ptr_batch(A, b_base, strideA);
        int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

        rocblas_status status = rocblas_internal_tbmv_launcher(handle,
                                                               uplo,
                                                               transA,
                                                               diag,
                                                               (rocblas_int)n_64,
                                                               (rocblas_int)k_64,
                                                               A_ptr,
                                                               offseta,
                                                               lda,
                                                               strideA,
                                                               x_ptr,
                                                               offsetx,
                                                               incx,
                                                               stridex,
                                                               batch_count,
                                                               w_x_copy);

        if(status != rocblas_status_success)
            return status;
    } // batch
    return rocblas_status_success;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *tbmv*.cpp

#ifdef INSTANTIATE_TBMV_LAUNCHER
#error INSTANTIATE_TBMV_LAUNCHER  already defined
#endif

#define INSTANTIATE_TBMV_LAUNCHER(U_, V_)                                                            \
    template rocblas_status rocblas_internal_tbmv_launcher_64<U_, V_>(rocblas_handle    handle,      \
                                                                      rocblas_fill      uplo,        \
                                                                      rocblas_operation transA,      \
                                                                      rocblas_diagonal  diag,        \
                                                                      int64_t           n,           \
                                                                      int64_t           k,           \
                                                                      U_                A,           \
                                                                      rocblas_stride    offseta,     \
                                                                      int64_t           lda,         \
                                                                      rocblas_stride    strideA,     \
                                                                      V_                x,           \
                                                                      rocblas_stride    offsetx,     \
                                                                      int64_t           incx,        \
                                                                      rocblas_stride    stridex,     \
                                                                      int64_t           batch_count, \
                                                                      V_                w_x_copy);

INSTANTIATE_TBMV_LAUNCHER(float const*, float*)
INSTANTIATE_TBMV_LAUNCHER(double const*, double*)
INSTANTIATE_TBMV_LAUNCHER(rocblas_float_complex const*, rocblas_float_complex*)
INSTANTIATE_TBMV_LAUNCHER(rocblas_double_complex const*, rocblas_double_complex*)
INSTANTIATE_TBMV_LAUNCHER(float const* const*, float* const*)
INSTANTIATE_TBMV_LAUNCHER(double const* const*, double* const*)
INSTANTIATE_TBMV_LAUNCHER(rocblas_float_complex const* const*, rocblas_float_complex* const*)
INSTANTIATE_TBMV_LAUNCHER(rocblas_double_complex const* const*, rocblas_double_complex* const*)

#undef INSTANTIATE_TBMV_LAUNCHER
