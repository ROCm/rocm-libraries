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

#include "rocblas_tpmv_64.hpp"

#include "blas2/rocblas_tpmv.hpp" // int32 API called

template <typename TConstPtr, typename TPtr, typename TWork>
rocblas_status rocblas_internal_tpmv_launcher_64(rocblas_handle    handle,
                                                 rocblas_fill      uplo,
                                                 rocblas_operation transa,
                                                 rocblas_diagonal  diag,
                                                 int64_t           n_64,
                                                 TConstPtr         AP,
                                                 rocblas_stride    offset_AP,
                                                 rocblas_stride    stride_AP,
                                                 TPtr              x,
                                                 rocblas_stride    offset_x,
                                                 int64_t           incx_64,
                                                 rocblas_stride    stride_x,
                                                 TWork             workspace,
                                                 rocblas_stride    stride_w,
                                                 int64_t           batch_count_64)
{
    // Quick return if possible. Not Argument error
    if(!n_64 || !batch_count_64)
        return rocblas_status_success;

    if(n_64 > c_i32_max)
        return rocblas_status_invalid_size; // defer adding new kernels for sizes exceeding practical memory

    for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
    {
        auto    x_ptr         = adjust_ptr_batch(x, b_base, stride_x);
        auto    workspace_ptr = adjust_ptr_batch(workspace, b_base, stride_w);
        auto    AP_ptr        = adjust_ptr_batch(AP, b_base, stride_AP);
        int32_t batch_count   = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

        auto shift_AP = offset_AP;

        rocblas_status status = rocblas_internal_tpmv_launcher(handle,
                                                               uplo,
                                                               transa,
                                                               diag,
                                                               (rocblas_int)n_64,
                                                               AP_ptr,
                                                               shift_AP,
                                                               stride_AP,
                                                               x_ptr,
                                                               offset_x,
                                                               incx_64,
                                                               stride_x,
                                                               workspace_ptr,
                                                               stride_w,
                                                               batch_count);

        if(status != rocblas_status_success)
            return status;
    } // batch
    return rocblas_status_success;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *Tpmv*.cpp

#ifdef INSTANTIATE_TPMV_LAUNCHER_64
#error INSTANTIATE_TPMV_LAUNCHER_64  already defined
#endif
#define INSTANTIATE_TPMV_LAUNCHER_64(TConstPtr_, TPtr_, TWork_)                           \
    template rocblas_status rocblas_internal_tpmv_launcher_64<TConstPtr_, TPtr_, TWork_>( \
        rocblas_handle    handle,                                                         \
        rocblas_fill      uplo,                                                           \
        rocblas_operation transa,                                                         \
        rocblas_diagonal  diag,                                                           \
        int64_t           n_64,                                                           \
        TConstPtr_        a,                                                              \
        rocblas_stride    offset_AP,                                                      \
        rocblas_stride    stride_AP,                                                      \
        TPtr_             x,                                                              \
        rocblas_stride    offset_x,                                                       \
        int64_t           incx,                                                           \
        rocblas_stride    stride_x,                                                       \
        TWork_            workspace,                                                      \
        rocblas_stride    stride_w,                                                       \
        int64_t           batch_count_64);

INSTANTIATE_TPMV_LAUNCHER_64(float const*, float*, float*)
INSTANTIATE_TPMV_LAUNCHER_64(double const*, double*, double*)
INSTANTIATE_TPMV_LAUNCHER_64(rocblas_float_complex const*,
                             rocblas_float_complex*,
                             rocblas_float_complex*)
INSTANTIATE_TPMV_LAUNCHER_64(rocblas_double_complex const*,
                             rocblas_double_complex*,
                             rocblas_double_complex*)
INSTANTIATE_TPMV_LAUNCHER_64(float const* const*, float* const*, float*)
INSTANTIATE_TPMV_LAUNCHER_64(double const* const*, double* const*, double*)
INSTANTIATE_TPMV_LAUNCHER_64(rocblas_float_complex const* const*,
                             rocblas_float_complex* const*,
                             rocblas_float_complex*)
INSTANTIATE_TPMV_LAUNCHER_64(rocblas_double_complex const* const*,
                             rocblas_double_complex* const*,
                             rocblas_double_complex*)

#undef INSTANTIATE_TPMV_LAUNCHER_64
