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
#include "blas3/rocblas_syrk_herk.hpp" // int32 API called
#include "handle.hpp"
#include "int64_helpers.hpp"
#include "rocblas_block_sizes.h"
#include "rocblas_syrk_herk_64.hpp"

/**
  * T is base type, i.e. float, double, rocblas_float_complex, or rocblas_double_complex
  * TScal is base type of scalars, for HERM == false, TScal == T, for HERM == true, TScal == real_t<T>
  * TConstPtr is either: const T* OR const T* const*
  * TPtr      is either:       T* OR       T* const*
  */
template <int64_t NB,
          bool    BATCHED,
          bool    HERM,
          typename T,
          typename TScal,
          typename TConstPtr,
          typename TPtr>
rocblas_status rocblas_internal_syrk_herk_template_64(rocblas_handle    handle,
                                                      rocblas_fill      uplo,
                                                      rocblas_operation trans,
                                                      int64_t           n_64,
                                                      int64_t           k_64,
                                                      const TScal*      alpha,
                                                      TConstPtr         A,
                                                      rocblas_stride    offset_A,
                                                      int64_t           lda_64,
                                                      rocblas_stride    stride_A,
                                                      const TScal*      beta,
                                                      TPtr              C,
                                                      rocblas_stride    offset_C,
                                                      int64_t           ldc_64,
                                                      rocblas_stride    stride_C,
                                                      int64_t           batch_count_64)
{
    // quick return
    if(!n_64 || !k_64 || !batch_count_64)
        return rocblas_status_success;

    if(n_64 > c_i32_max)
        return rocblas_status_invalid_size; // defer adding new kernels for sizes exceeding practical memory

    static constexpr bool TWOK = false;

    if(n_64 <= c_ILP64_i32_max && k_64 <= c_ILP64_i32_max && lda_64 <= c_ILP64_i32_max
       && ldc_64 <= c_ILP64_i32_max && batch_count_64 <= c_i64_grid_YZ_chunk)
    {
        return rocblas_internal_syrk_herk_template<NB, BATCHED, HERM, T>(
            handle,
            uplo,
            trans,
            n_64,
            k_64,
            alpha,
            A,
            offset_A,
            lda_64,
            stride_A,
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
        auto C_ptr = adjust_ptr_batch(C, b_base, stride_C);

        int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

        rocblas_status status
            = rocblas_internal_syr2k_her2k_template<int64_t, NB, BATCHED, TWOK, HERM, T>(
                handle,
                uplo,
                trans,
                n_64,
                k_64,
                alpha,
                A_ptr,
                offset_A,
                lda_64,
                stride_A,
                A_ptr,
                offset_A,
                lda_64,
                stride_A,
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

#define ROCBLAS_INTERNAL_SYRK_HERK_PARAMS                                                       \
    handle, uplo, trans_a, n_64, k_64, alpha, A, offset_A, lda_64, stride_A, beta, C, offset_C, \
        ldc_64, stride_C, batch_count_64

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syrk_template_64(rocblas_handle    handle,
                                      rocblas_fill      uplo,
                                      rocblas_operation trans_a,
                                      int64_t           n_64,
                                      int64_t           k_64,
                                      const T*          alpha,
                                      const T*          A,
                                      rocblas_stride    offset_A,
                                      int64_t           lda_64,
                                      rocblas_stride    stride_A,
                                      const T*          beta,
                                      T*                C,
                                      rocblas_stride    offset_C,
                                      int64_t           ldc_64,
                                      rocblas_stride    stride_C,
                                      int64_t           batch_count_64)
{
    constexpr bool BATCHED = false;
    constexpr bool HERM    = false;
    if constexpr(std::is_same_v<T, float>)
        return rocblas_internal_syrk_herk_template_64<ROCBLAS_SDZSYRK_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);
    else if constexpr(std::is_same_v<T, double>)
        return rocblas_internal_syrk_herk_template_64<ROCBLAS_SDZSYRK_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_float_complex>)
        return rocblas_internal_syrk_herk_template_64<ROCBLAS_CSYRK_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_double_complex>)
        return rocblas_internal_syrk_herk_template_64<ROCBLAS_SDZSYRK_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);

    return rocblas_status_not_implemented;
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syrk_batched_template_64(rocblas_handle    handle,
                                              rocblas_fill      uplo,
                                              rocblas_operation trans_a,
                                              int64_t           n_64,
                                              int64_t           k_64,
                                              const T*          alpha,
                                              const T* const*   A,
                                              rocblas_stride    offset_A,
                                              int64_t           lda_64,
                                              rocblas_stride    stride_A,
                                              const T*          beta,
                                              T* const*         C,
                                              rocblas_stride    offset_C,
                                              int64_t           ldc_64,
                                              rocblas_stride    stride_C,
                                              int64_t           batch_count_64)
{
    constexpr bool BATCHED = true;
    constexpr bool HERM    = false;
    if constexpr(std::is_same_v<T, float>)
        return rocblas_internal_syrk_herk_template_64<ROCBLAS_SDSYRK_BATCHED_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);
    else if constexpr(std::is_same_v<T, double>)
        return rocblas_internal_syrk_herk_template_64<ROCBLAS_SDSYRK_BATCHED_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_float_complex>)
        return rocblas_internal_syrk_herk_template_64<ROCBLAS_CZSYRK_BATCHED_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_double_complex>)
        return rocblas_internal_syrk_herk_template_64<ROCBLAS_CZSYRK_BATCHED_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);

    return rocblas_status_not_implemented;
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_herk_template_64(rocblas_handle    handle,
                                      rocblas_fill      uplo,
                                      rocblas_operation trans_a,
                                      int64_t           n_64,
                                      int64_t           k_64,
                                      const real_t<T>*  alpha,
                                      const T*          A,
                                      rocblas_stride    offset_A,
                                      int64_t           lda_64,
                                      rocblas_stride    stride_A,
                                      const real_t<T>*  beta,
                                      T*                C,
                                      rocblas_stride    offset_C,
                                      int64_t           ldc_64,
                                      rocblas_stride    stride_C,
                                      int64_t           batch_count_64)
{
    constexpr bool BATCHED = false;
    constexpr bool HERM    = true;
    if constexpr(std::is_same_v<T, rocblas_float_complex>)
        return rocblas_internal_syrk_herk_template_64<ROCBLAS_CHERK_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_double_complex>)
        return rocblas_internal_syrk_herk_template_64<ROCBLAS_ZHERK_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);

    return rocblas_status_not_implemented;
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_herk_batched_template_64(rocblas_handle    handle,
                                              rocblas_fill      uplo,
                                              rocblas_operation trans_a,
                                              int64_t           n_64,
                                              int64_t           k_64,
                                              const real_t<T>*  alpha,
                                              const T* const*   A,
                                              rocblas_stride    offset_A,
                                              int64_t           lda_64,
                                              rocblas_stride    stride_A,
                                              const real_t<T>*  beta,
                                              T* const*         C,
                                              rocblas_stride    offset_C,
                                              int64_t           ldc_64,
                                              rocblas_stride    stride_C,
                                              int64_t           batch_count_64)
{
    constexpr bool BATCHED = true;
    constexpr bool HERM    = true;
    if constexpr(std::is_same_v<T, rocblas_float_complex>)
        return rocblas_internal_syrk_herk_template_64<ROCBLAS_HERK_BATCHED_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_double_complex>)
        return rocblas_internal_syrk_herk_template_64<ROCBLAS_HERK_BATCHED_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);

    return rocblas_status_not_implemented;
}

#undef ROCBLAS_INTERNAL_SYRK_HERK_PARAMS

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files syrk*.cpp or herk*.cpp

#ifdef INSTANTIATE_SYRK_TEMPLATE_64
#error INSTANTIATE_SYRK_TEMPLATE_64 already defined
#endif

#define INSTANTIATE_SYRK_TEMPLATE_64(T_)                                  \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status              \
        rocblas_internal_syrk_template_64<T_>(rocblas_handle    handle,   \
                                              rocblas_fill      uplo,     \
                                              rocblas_operation trans_a,  \
                                              int64_t           n_64,     \
                                              int64_t           k_64,     \
                                              const T_*         alpha,    \
                                              const T_*         A,        \
                                              rocblas_stride    offset_A, \
                                              int64_t           lda_64,   \
                                              rocblas_stride    stride_A, \
                                              const T_*         beta,     \
                                              T_*               C,        \
                                              rocblas_stride    offset_C, \
                                              int64_t           ldc_64,   \
                                              rocblas_stride    stride_C, \
                                              int64_t           batch_count_64);

INSTANTIATE_SYRK_TEMPLATE_64(float)
INSTANTIATE_SYRK_TEMPLATE_64(double)
INSTANTIATE_SYRK_TEMPLATE_64(rocblas_float_complex)
INSTANTIATE_SYRK_TEMPLATE_64(rocblas_double_complex)

#undef INSTANTIATE_SYRK_TEMPLATE_64

#ifdef INSTANTIATE_SYRK_BATCHED_TEMPLATE_64
#error INSTANTIATE_SYRK_BATCHED_TEMPLATE_64 already defined
#endif

#define INSTANTIATE_SYRK_BATCHED_TEMPLATE_64(T_)                                  \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                      \
        rocblas_internal_syrk_batched_template_64<T_>(rocblas_handle    handle,   \
                                                      rocblas_fill      uplo,     \
                                                      rocblas_operation trans_a,  \
                                                      int64_t           n_64,     \
                                                      int64_t           k_64,     \
                                                      const T_*         alpha,    \
                                                      const T_* const*  A,        \
                                                      rocblas_stride    offset_A, \
                                                      int64_t           lda_64,   \
                                                      rocblas_stride    stride_A, \
                                                      const T_*         beta,     \
                                                      T_* const*        C,        \
                                                      rocblas_stride    offset_C, \
                                                      int64_t           ldc_64,   \
                                                      rocblas_stride    stride_C, \
                                                      int64_t           batch_count_64);

INSTANTIATE_SYRK_BATCHED_TEMPLATE_64(float)
INSTANTIATE_SYRK_BATCHED_TEMPLATE_64(double)
INSTANTIATE_SYRK_BATCHED_TEMPLATE_64(rocblas_float_complex)
INSTANTIATE_SYRK_BATCHED_TEMPLATE_64(rocblas_double_complex)

#undef INSTANTIATE_SYRK_BATCHED_TEMPLATE_64

#ifdef INSTANTIATE_HERK_TEMPLATE_64
#error INSTANTIATE_HERK_TEMPLATE_64 already defined
#endif

#define INSTANTIATE_HERK_TEMPLATE_64(T_)                                  \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status              \
        rocblas_internal_herk_template_64<T_>(rocblas_handle    handle,   \
                                              rocblas_fill      uplo,     \
                                              rocblas_operation trans_a,  \
                                              int64_t           n_64,     \
                                              int64_t           k_64,     \
                                              const real_t<T_>* alpha,    \
                                              const T_*         A,        \
                                              rocblas_stride    offset_A, \
                                              int64_t           lda_64,   \
                                              rocblas_stride    stride_A, \
                                              const real_t<T_>* beta,     \
                                              T_*               C,        \
                                              rocblas_stride    offset_C, \
                                              int64_t           ldc_64,   \
                                              rocblas_stride    stride_C, \
                                              int64_t           batch_count_64);

INSTANTIATE_HERK_TEMPLATE_64(rocblas_float_complex)
INSTANTIATE_HERK_TEMPLATE_64(rocblas_double_complex)

#undef INSTANTIATE_HERK_TEMPLATE_64

#ifdef INSTANTIATE_HERK_BATCHED_TEMPLATE_64
#error INSTANTIATE_HERK_BATCHED_TEMPLATE_64 already defined
#endif

#define INSTANTIATE_HERK_BATCHED_TEMPLATE_64(T_)                                  \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                      \
        rocblas_internal_herk_batched_template_64<T_>(rocblas_handle    handle,   \
                                                      rocblas_fill      uplo,     \
                                                      rocblas_operation trans_a,  \
                                                      int64_t           n_64,     \
                                                      int64_t           k_64,     \
                                                      const real_t<T_>* alpha,    \
                                                      const T_* const*  A,        \
                                                      rocblas_stride    offset_A, \
                                                      int64_t           lda_64,   \
                                                      rocblas_stride    stride_A, \
                                                      const real_t<T_>* beta,     \
                                                      T_* const*        C,        \
                                                      rocblas_stride    offset_C, \
                                                      int64_t           ldc_64,   \
                                                      rocblas_stride    stride_C, \
                                                      int64_t           batch_count_64);

INSTANTIATE_HERK_BATCHED_TEMPLATE_64(rocblas_float_complex)
INSTANTIATE_HERK_BATCHED_TEMPLATE_64(rocblas_double_complex)

#undef INSTANTIATE_HERK_BATCHED_TEMPLATE_64
