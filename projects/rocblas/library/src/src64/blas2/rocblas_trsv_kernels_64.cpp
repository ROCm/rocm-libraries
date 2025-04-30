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

#include "rocblas_trsv_64.hpp"

#include "blas2/rocblas_trsv.hpp" // int32 API called

template <rocblas_int DIM_X, typename T, typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_trsv_substitution_template_64(rocblas_handle    handle,
                                                              rocblas_fill      uplo,
                                                              rocblas_operation transA,
                                                              rocblas_diagonal  diag,
                                                              int64_t           n_64,
                                                              TConstPtr         A,
                                                              rocblas_stride    offset_A,
                                                              int64_t           lda_64,
                                                              rocblas_stride    stride_A,
                                                              T const*          alpha,
                                                              TPtr              x,
                                                              rocblas_stride    offset_x,
                                                              int64_t           incx_64,
                                                              rocblas_stride    stride_x,
                                                              int64_t           batch_count_64,
                                                              rocblas_int*      w_completed_sec)
{
    // Quick return if possible. Not Argument error
    if(!n_64 || !batch_count_64)
        return rocblas_status_success;

    if(n_64 > c_i32_max)
        return rocblas_status_invalid_size; // defer adding new kernels for sizes exceeding practical memory

    for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
    {
        auto    x_ptr       = adjust_ptr_batch(x, b_base, stride_x);
        auto    A_ptr       = adjust_ptr_batch(A, b_base, stride_A);
        int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

        auto shift_A = offset_A;

        rocblas_status status = rocblas_internal_trsv_substitution_template<DIM_X, T>(
            handle,
            uplo,
            transA,
            diag,
            (rocblas_int)n_64,
            A_ptr,
            shift_A,
            lda_64,
            stride_A,
            alpha, // trsv doesn't need alpha, but trsm using this kernel may use.
            x_ptr,
            offset_x,
            incx_64,
            stride_x,
            batch_count,
            w_completed_sec);

        if(status != rocblas_status_success)
            return status;
    } // batch
    return rocblas_status_success;
}

#define TRSV_TEMPLATE_PARAMS_64                                                            \
    handle, uplo, transA, diag, n_64, A, offset_A, lda_64, stride_A, nullptr, x, offset_x, \
        incx_64, stride_x, batch_count_64, w_completed_sec

template <typename T>
rocblas_status rocblas_internal_trsv_template_64(rocblas_handle    handle,
                                                 rocblas_fill      uplo,
                                                 rocblas_operation transA,
                                                 rocblas_diagonal  diag,
                                                 int64_t           n_64,
                                                 const T*          A,
                                                 rocblas_stride    offset_A,
                                                 int64_t           lda_64,
                                                 rocblas_stride    stride_A,
                                                 T*                x,
                                                 rocblas_stride    offset_x,
                                                 int64_t           incx_64,
                                                 rocblas_stride    stride_x,
                                                 int64_t           batch_count_64,
                                                 rocblas_int*      w_completed_sec)
{
    if constexpr(std::is_same_v<T, float>)
        return rocblas_internal_trsv_substitution_template_64<ROCBLAS_SDCTRSV_NB, T>(
            TRSV_TEMPLATE_PARAMS_64);
    else if constexpr(std::is_same_v<T, double>)
        return rocblas_internal_trsv_substitution_template_64<ROCBLAS_SDCTRSV_NB, T>(
            TRSV_TEMPLATE_PARAMS_64);
    else if constexpr(std::is_same_v<T, rocblas_float_complex>)
        return rocblas_internal_trsv_substitution_template_64<ROCBLAS_SDCTRSV_NB, T>(
            TRSV_TEMPLATE_PARAMS_64);
    else if constexpr(std::is_same_v<T, rocblas_double_complex>)
        return rocblas_internal_trsv_substitution_template_64<ROCBLAS_ZTRSV_NB, T>(
            TRSV_TEMPLATE_PARAMS_64);

    return rocblas_status_internal_error;
}

template <typename T>
rocblas_status rocblas_internal_trsv_batched_template_64(rocblas_handle    handle,
                                                         rocblas_fill      uplo,
                                                         rocblas_operation transA,
                                                         rocblas_diagonal  diag,
                                                         int64_t           n_64,
                                                         const T* const*   A,
                                                         rocblas_stride    offset_A,
                                                         int64_t           lda_64,
                                                         rocblas_stride    stride_A,
                                                         T* const*         x,
                                                         rocblas_stride    offset_x,
                                                         int64_t           incx_64,
                                                         rocblas_stride    stride_x,
                                                         int64_t           batch_count_64,
                                                         rocblas_int*      w_completed_sec)
{
    if constexpr(std::is_same_v<T, float>)
        return rocblas_internal_trsv_substitution_template_64<ROCBLAS_SDCTRSV_NB, T>(
            TRSV_TEMPLATE_PARAMS_64);
    else if constexpr(std::is_same_v<T, double>)
        return rocblas_internal_trsv_substitution_template_64<ROCBLAS_SDCTRSV_NB, T>(
            TRSV_TEMPLATE_PARAMS_64);
    else if constexpr(std::is_same_v<T, rocblas_float_complex>)
        return rocblas_internal_trsv_substitution_template_64<ROCBLAS_SDCTRSV_NB, T>(
            TRSV_TEMPLATE_PARAMS_64);
    else if constexpr(std::is_same_v<T, rocblas_double_complex>)
        return rocblas_internal_trsv_substitution_template_64<ROCBLAS_ZTRSV_NB, T>(
            TRSV_TEMPLATE_PARAMS_64);

    return rocblas_status_internal_error;
}

#undef TRSV_TEMPLATE_PARAMS_64

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *trsv*.cpp

#ifdef INSTANTIATE_TRSV_TEMPLATE_64
#error INSTANTIATE_TRSV_TEMPLATE_64 already defined
#endif

#define INSTANTIATE_TRSV_TEMPLATE_64(T_)                                        \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                    \
        rocblas_internal_trsv_template_64<T_>(rocblas_handle    handle,         \
                                              rocblas_fill      uplo,           \
                                              rocblas_operation transA,         \
                                              rocblas_diagonal  diag,           \
                                              int64_t           n_64,           \
                                              const T_*         A,              \
                                              rocblas_stride    offseta,        \
                                              int64_t           lda_64,         \
                                              rocblas_stride    stridea,        \
                                              T_*               x,              \
                                              rocblas_stride    offsetx,        \
                                              int64_t           incx_64,        \
                                              rocblas_stride    stridex,        \
                                              int64_t           batch_count_64, \
                                              rocblas_int*      w_completed_sec);

INSTANTIATE_TRSV_TEMPLATE_64(float)
INSTANTIATE_TRSV_TEMPLATE_64(double)
INSTANTIATE_TRSV_TEMPLATE_64(rocblas_float_complex)
INSTANTIATE_TRSV_TEMPLATE_64(rocblas_double_complex)

#undef INSTANTIATE_TRSV_TEMPLATE_64

#ifdef INSTANTIATE_TRSV_TEMPLATE_64
#error INSTANTIATE_TRSV_TEMPLATE_64 already defined
#endif

#define INSTANTIATE_TRSV_BATCHED_TEMPLATE_64(T_)                                        \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                            \
        rocblas_internal_trsv_batched_template_64<T_>(rocblas_handle    handle,         \
                                                      rocblas_fill      uplo,           \
                                                      rocblas_operation transA,         \
                                                      rocblas_diagonal  diag,           \
                                                      int64_t           n_64,           \
                                                      const T_* const*  A,              \
                                                      rocblas_stride    offseta,        \
                                                      int64_t           lda_64,         \
                                                      rocblas_stride    stridea,        \
                                                      T_* const*        x,              \
                                                      rocblas_stride    offsetx,        \
                                                      int64_t           incx_64,        \
                                                      rocblas_stride    stridex,        \
                                                      int64_t           batch_count_64, \
                                                      rocblas_int*      w_completed_sec);

INSTANTIATE_TRSV_BATCHED_TEMPLATE_64(float)
INSTANTIATE_TRSV_BATCHED_TEMPLATE_64(double)
INSTANTIATE_TRSV_BATCHED_TEMPLATE_64(rocblas_float_complex)
INSTANTIATE_TRSV_BATCHED_TEMPLATE_64(rocblas_double_complex)

#undef INSTANTIATE_TRSV_BATCHED_TEMPLATE_64
