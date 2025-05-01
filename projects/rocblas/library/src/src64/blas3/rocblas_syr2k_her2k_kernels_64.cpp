/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "blas3/rocblas_syr2k_her2k.hpp" // int32 API called
#include "handle.hpp"
#include "int64_helpers.hpp"
#include "rocblas_block_sizes.h"
#include "rocblas_syr2k_her2k_64.hpp"

template <rocblas_int MIN_NB,
          bool        BATCHED,
          bool        TWOK,
          bool        HERK,
          typename T,
          typename TScala,
          typename TScalb,
          typename TConstPtr,
          typename TPtr>
rocblas_status rocblas_internal_syr2k_her2k_template_64(rocblas_handle    handle,
                                                        rocblas_fill      uplo,
                                                        rocblas_operation trans,
                                                        int64_t           n_64,
                                                        int64_t           k_64,
                                                        const TScala*     alpha,
                                                        TConstPtr         A,
                                                        rocblas_stride    offset_A,
                                                        int64_t           lda_64,
                                                        rocblas_stride    stride_A,
                                                        TConstPtr         B,
                                                        rocblas_stride    offset_B,
                                                        int64_t           ldb_64,
                                                        rocblas_stride    stride_B,
                                                        const TScalb*     beta,
                                                        TPtr              C,
                                                        rocblas_stride    offset_C,
                                                        int64_t           ldc_64,
                                                        rocblas_stride    stride_C,
                                                        int64_t           batch_count_64)
{
    // quick return
    if(!n_64 || !k_64 || !batch_count_64)
        return rocblas_status_success;

    if(n_64 > c_ILP64_i32_max)
        return rocblas_status_invalid_size; // defer adding new kernels for sizes exceeding practical memory

    if(n_64 <= c_ILP64_i32_max && k_64 <= c_ILP64_i32_max && lda_64 <= c_ILP64_i32_max
       && ldb_64 <= c_ILP64_i32_max && ldc_64 <= c_ILP64_i32_max
       && batch_count_64 <= c_i64_grid_YZ_chunk)
    {
        return rocblas_internal_syr2k_her2k_template<rocblas_int, MIN_NB, BATCHED, TWOK, HERK, T>(
            handle,
            uplo,
            trans,
            (rocblas_int)n_64,
            k_64,
            alpha,
            A,
            offset_A,
            lda_64,
            stride_A,
            B,
            offset_B,
            ldb_64,
            stride_B,
            beta,
            C,
            offset_C,
            ldc_64,
            stride_C,
            (rocblas_int)batch_count_64);
    }

    for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
    {
        auto A_ptr = adjust_ptr_batch(A, b_base, stride_A);
        auto B_ptr = adjust_ptr_batch(B, b_base, stride_B);
        auto C_ptr = adjust_ptr_batch(C, b_base, stride_C);

        int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

        rocblas_status status
            = rocblas_internal_syr2k_her2k_template<int64_t, MIN_NB, BATCHED, TWOK, HERK, T>(
                handle,
                uplo,
                trans,
                (rocblas_int)n_64,
                k_64,
                alpha,
                A_ptr,
                offset_A,
                lda_64,
                stride_A,
                B_ptr,
                offset_B,
                ldb_64,
                stride_B,
                beta,
                C_ptr,
                offset_C,
                ldc_64,
                stride_C,
                batch_count);

        if(status != rocblas_status_success)
            return status;
    } // batch
    return rocblas_status_success;
}

#define ROCBLAS_INTERNAL_SYR2K_HER2K_PARAMS_64                                                  \
    handle, uplo, trans, n_64, k_64, alpha, A, offset_A, lda_64, stride_A, B, offset_b, ldb_64, \
        stride_B, beta, C, offset_c, ldc_64, stride_C, batch_count_64

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syr2k_template_64(rocblas_handle    handle,
                                       rocblas_fill      uplo,
                                       rocblas_operation trans,
                                       int64_t           n_64,
                                       int64_t           k_64,
                                       const T*          alpha,
                                       const T*          A,
                                       rocblas_stride    offset_A,
                                       int64_t           lda_64,
                                       rocblas_stride    stride_A,
                                       const T*          B,
                                       rocblas_stride    offset_b,
                                       int64_t           ldb_64,
                                       rocblas_stride    stride_B,
                                       const T*          beta,
                                       T*                C,
                                       rocblas_stride    offset_c,
                                       int64_t           ldc_64,
                                       rocblas_stride    stride_C,
                                       int64_t           batch_count_64)
{
    constexpr bool BATCHED = false;
    constexpr bool TWOK    = true;
    constexpr bool HERM    = false;

    if constexpr(std::is_same_v<T, float>)
        return rocblas_internal_syr2k_her2k_template_64<ROCBLAS_SSYR2K_NB, BATCHED, TWOK, HERM, T>(
            ROCBLAS_INTERNAL_SYR2K_HER2K_PARAMS_64);
    else if constexpr(std::is_same_v<T, double>)
        return rocblas_internal_syr2k_her2k_template_64<ROCBLAS_DSYR2K_NB, BATCHED, TWOK, HERM, T>(
            ROCBLAS_INTERNAL_SYR2K_HER2K_PARAMS_64);
    else if constexpr(std::is_same_v<T, rocblas_float_complex>)
        return rocblas_internal_syr2k_her2k_template_64<ROCBLAS_CSYR2K_NB, BATCHED, TWOK, HERM, T>(
            ROCBLAS_INTERNAL_SYR2K_HER2K_PARAMS_64);
    else if constexpr(std::is_same_v<T, rocblas_double_complex>)
        return rocblas_internal_syr2k_her2k_template_64<ROCBLAS_ZSYR2K_NB, BATCHED, TWOK, HERM, T>(
            ROCBLAS_INTERNAL_SYR2K_HER2K_PARAMS_64);

    return rocblas_status_not_implemented;
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syr2k_batched_template_64(rocblas_handle    handle,
                                               rocblas_fill      uplo,
                                               rocblas_operation trans,
                                               int64_t           n_64,
                                               int64_t           k_64,
                                               const T*          alpha,
                                               const T* const*   A,
                                               rocblas_stride    offset_A,
                                               int64_t           lda_64,
                                               rocblas_stride    stride_A,
                                               const T* const*   B,
                                               rocblas_stride    offset_b,
                                               int64_t           ldb_64,
                                               rocblas_stride    stride_B,
                                               const T*          beta,
                                               T* const*         C,
                                               rocblas_stride    offset_c,
                                               int64_t           ldc_64,
                                               rocblas_stride    stride_C,
                                               int64_t           batch_count_64)
{
    constexpr bool BATCHED = true;
    constexpr bool TWOK    = true;
    constexpr bool HERM    = false;

    if constexpr(std::is_same_v<T, float>)
        return rocblas_internal_syr2k_her2k_template_64<ROCBLAS_SSYR2K_NB, BATCHED, TWOK, HERM, T>(
            ROCBLAS_INTERNAL_SYR2K_HER2K_PARAMS_64);
    else if constexpr(std::is_same_v<T, double>)
        return rocblas_internal_syr2k_her2k_template_64<ROCBLAS_DSYR2K_NB, BATCHED, TWOK, HERM, T>(
            ROCBLAS_INTERNAL_SYR2K_HER2K_PARAMS_64);
    else if constexpr(std::is_same_v<T, rocblas_float_complex>)
        return rocblas_internal_syr2k_her2k_template_64<ROCBLAS_CSYR2K_NB, BATCHED, TWOK, HERM, T>(
            ROCBLAS_INTERNAL_SYR2K_HER2K_PARAMS_64);
    else if constexpr(std::is_same_v<T, rocblas_double_complex>)
        return rocblas_internal_syr2k_her2k_template_64<ROCBLAS_ZSYR2K_NB, BATCHED, TWOK, HERM, T>(
            ROCBLAS_INTERNAL_SYR2K_HER2K_PARAMS_64);

    return rocblas_status_not_implemented;
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_her2k_template_64(rocblas_handle    handle,
                                       rocblas_fill      uplo,
                                       rocblas_operation trans,
                                       int64_t           n_64,
                                       int64_t           k_64,
                                       const T*          alpha,
                                       const T*          A,
                                       rocblas_stride    offset_A,
                                       int64_t           lda_64,
                                       rocblas_stride    stride_A,
                                       const T*          B,
                                       rocblas_stride    offset_b,
                                       int64_t           ldb_64,
                                       rocblas_stride    stride_B,
                                       const real_t<T>*  beta,
                                       T*                C,
                                       rocblas_stride    offset_c,
                                       int64_t           ldc_64,
                                       rocblas_stride    stride_C,
                                       int64_t           batch_count_64)
{
    constexpr bool BATCHED = false;
    constexpr bool TWOK    = true;
    constexpr bool HERM    = true;

    if constexpr(std::is_same_v<T, rocblas_float_complex>)
        return rocblas_internal_syr2k_her2k_template_64<ROCBLAS_CSYR2K_NB, BATCHED, TWOK, HERM, T>(
            ROCBLAS_INTERNAL_SYR2K_HER2K_PARAMS_64);
    else if constexpr(std::is_same_v<T, rocblas_double_complex>)
        return rocblas_internal_syr2k_her2k_template_64<ROCBLAS_ZSYR2K_NB, BATCHED, TWOK, HERM, T>(
            ROCBLAS_INTERNAL_SYR2K_HER2K_PARAMS_64);

    return rocblas_status_not_implemented;
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_her2k_batched_template_64(rocblas_handle    handle,
                                               rocblas_fill      uplo,
                                               rocblas_operation trans,
                                               int64_t           n_64,
                                               int64_t           k_64,
                                               const T*          alpha,
                                               const T* const*   A,
                                               rocblas_stride    offset_A,
                                               int64_t           lda_64,
                                               rocblas_stride    stride_A,
                                               const T* const*   B,
                                               rocblas_stride    offset_b,
                                               int64_t           ldb_64,
                                               rocblas_stride    stride_B,
                                               const real_t<T>*  beta,
                                               T* const*         C,
                                               rocblas_stride    offset_c,
                                               int64_t           ldc_64,
                                               rocblas_stride    stride_C,
                                               int64_t           batch_count_64)
{
    constexpr bool BATCHED = true;
    constexpr bool TWOK    = true;
    constexpr bool HERM    = true;

    if constexpr(std::is_same_v<T, rocblas_float_complex>)
        return rocblas_internal_syr2k_her2k_template_64<ROCBLAS_CSYR2K_NB, BATCHED, TWOK, HERM, T>(
            ROCBLAS_INTERNAL_SYR2K_HER2K_PARAMS_64);
    else if constexpr(std::is_same_v<T, rocblas_double_complex>)
        return rocblas_internal_syr2k_her2k_template_64<ROCBLAS_ZSYR2K_NB, BATCHED, TWOK, HERM, T>(
            ROCBLAS_INTERNAL_SYR2K_HER2K_PARAMS_64);

    return rocblas_status_not_implemented;
}

#undef ROCBLAS_INTERNAL_SYR2K_HER2K_PARAMS_64

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files syr2k*.cpp or her2k*.cpp

#ifdef INSTANTIATE_SYR2K_TEMPLATE_64
#error INSTANTIATE_SYR2K_TEMPLATE_64 already defined
#endif

#define INSTANTIATE_SYR2K_TEMPLATE_64(T_)                                  \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status               \
        rocblas_internal_syr2k_template_64<T_>(rocblas_handle    handle,   \
                                               rocblas_fill      uplo,     \
                                               rocblas_operation trans,    \
                                               int64_t           n_64,     \
                                               int64_t           k_64,     \
                                               const T_*         alpha,    \
                                               const T_*         A,        \
                                               rocblas_stride    offset_A, \
                                               int64_t           lda_64,   \
                                               rocblas_stride    stride_A, \
                                               const T_*         B,        \
                                               rocblas_stride    offset_b, \
                                               int64_t           ldb_64,   \
                                               rocblas_stride    stride_B, \
                                               const T_*         beta,     \
                                               T_*               C,        \
                                               rocblas_stride    offset_c, \
                                               int64_t           ldc_64,   \
                                               rocblas_stride    stride_C, \
                                               int64_t           batch_count_64);

INSTANTIATE_SYR2K_TEMPLATE_64(float)
INSTANTIATE_SYR2K_TEMPLATE_64(double)
INSTANTIATE_SYR2K_TEMPLATE_64(rocblas_float_complex)
INSTANTIATE_SYR2K_TEMPLATE_64(rocblas_double_complex)

#undef INSTANTIATE_SYR2K_TEMPLATE_64

#ifdef INSTANTIATE_SYR2K_BATCHED_TEMPLATE_64
#error INSTANTIATE_SYR2K_BATCHED_TEMPLATE_64 already defined
#endif

#define INSTANTIATE_SYR2K_BATCHED_TEMPLATE_64(T_)                                  \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                       \
        rocblas_internal_syr2k_batched_template_64<T_>(rocblas_handle    handle,   \
                                                       rocblas_fill      uplo,     \
                                                       rocblas_operation trans,    \
                                                       int64_t           n_64,     \
                                                       int64_t           k_64,     \
                                                       const T_*         alpha,    \
                                                       const T_* const*  A,        \
                                                       rocblas_stride    offset_A, \
                                                       int64_t           lda_64,   \
                                                       rocblas_stride    stride_A, \
                                                       const T_* const*  B,        \
                                                       rocblas_stride    offset_b, \
                                                       int64_t           ldb_64,   \
                                                       rocblas_stride    stride_B, \
                                                       const T_*         beta,     \
                                                       T_* const*        C,        \
                                                       rocblas_stride    offset_c, \
                                                       int64_t           ldc_64,   \
                                                       rocblas_stride    stride_C, \
                                                       int64_t           batch_count_64);

INSTANTIATE_SYR2K_BATCHED_TEMPLATE_64(float)
INSTANTIATE_SYR2K_BATCHED_TEMPLATE_64(double)
INSTANTIATE_SYR2K_BATCHED_TEMPLATE_64(rocblas_float_complex)
INSTANTIATE_SYR2K_BATCHED_TEMPLATE_64(rocblas_double_complex)

#undef INSTANTIATE_SYR2K_BATCHED_TEMPLATE_64

#ifdef INSTANTIATE_HER2K_TEMPLATE_64
#error INSTANTIATE_HER2K_TEMPLATE_64 already defined
#endif

#define INSTANTIATE_HER2K_TEMPLATE_64(T_)                                  \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status               \
        rocblas_internal_her2k_template_64<T_>(rocblas_handle    handle,   \
                                               rocblas_fill      uplo,     \
                                               rocblas_operation trans,    \
                                               int64_t           n_64,     \
                                               int64_t           k_64,     \
                                               const T_*         alpha,    \
                                               const T_*         A,        \
                                               rocblas_stride    offset_A, \
                                               int64_t           lda_64,   \
                                               rocblas_stride    stride_A, \
                                               const T_*         B,        \
                                               rocblas_stride    offset_b, \
                                               int64_t           ldb_64,   \
                                               rocblas_stride    stride_B, \
                                               const real_t<T_>* beta,     \
                                               T_*               C,        \
                                               rocblas_stride    offset_c, \
                                               int64_t           ldc_64,   \
                                               rocblas_stride    stride_C, \
                                               int64_t           batch_count_64);

INSTANTIATE_HER2K_TEMPLATE_64(rocblas_float_complex)
INSTANTIATE_HER2K_TEMPLATE_64(rocblas_double_complex)

#undef INSTANTIATE_HER2K_TEMPLATE_64

#ifdef INSTANTIATE_HER2K_BATCHED_TEMPLATE_64
#error INSTANTIATE_HER2K_BATCHED_TEMPLATE_64 already defined
#endif

#define INSTANTIATE_HER2K_BATCHED_TEMPLATE_64(T_)                                  \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                       \
        rocblas_internal_her2k_batched_template_64<T_>(rocblas_handle    handle,   \
                                                       rocblas_fill      uplo,     \
                                                       rocblas_operation trans,    \
                                                       int64_t           n_64,     \
                                                       int64_t           k_64,     \
                                                       const T_*         alpha,    \
                                                       const T_* const*  A,        \
                                                       rocblas_stride    offset_A, \
                                                       int64_t           lda_64,   \
                                                       rocblas_stride    stride_A, \
                                                       const T_* const*  B,        \
                                                       rocblas_stride    offset_b, \
                                                       int64_t           ldb_64,   \
                                                       rocblas_stride    stride_B, \
                                                       const real_t<T_>* beta,     \
                                                       T_* const*        C,        \
                                                       rocblas_stride    offset_c, \
                                                       int64_t           ldc_64,   \
                                                       rocblas_stride    stride_C, \
                                                       int64_t           batch_count_64);

INSTANTIATE_HER2K_BATCHED_TEMPLATE_64(rocblas_float_complex)
INSTANTIATE_HER2K_BATCHED_TEMPLATE_64(rocblas_double_complex)

#undef INSTANTIATE_HER2K_BATCHED_TEMPLATE_64
