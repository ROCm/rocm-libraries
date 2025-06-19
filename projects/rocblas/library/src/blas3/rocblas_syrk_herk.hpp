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

#pragma once

#include "check_numerics_matrix.hpp"
#include "handle.hpp"
#include "rocblas_gemm.hpp"
#include "rocblas_level3_threshold.hpp"

template <typename T>
inline bool rocblas_use_only_gemm(rocblas_handle handle, rocblas_int n, rocblas_int k)
{
    //Identifying the architecture to have an appropriate optimization
    bool is_gfx942 = handle->getArch() == 942 ? true : false;
    bool is_gfx90a = handle->getArch() == 910 ? true : false;

    //Identifying the precision to have an appropriate optimization
    constexpr bool is_float          = std::is_same_v<T, float>;
    constexpr bool is_double         = std::is_same_v<T, double>;
    constexpr bool is_complex_float  = std::is_same_v<T, rocblas_float_complex>;
    constexpr bool is_complex_double = std::is_same_v<T, rocblas_double_complex>;

    //Optimized kernel which uses only GEMM
    return k >= syrk_k_lower_threshold
           && ((is_gfx942
                && (((is_float || is_double) && n < sdsyrk_gfx942_n_higher_threshold)
                    || (is_complex_double && n < zsyrk_gfx942_n_higher_threshold)
                    || (is_complex_float && n < csyrk_gfx942_n_higher_threshold)))
               || (is_gfx90a
                   && (((is_float || is_double) && n < sdsyrk_gfx90a_n_higher_threshold)
                       || (is_complex_float || is_complex_double)
                              && n < czsyrk_gfx90a_n_higher_threshold)));
}

template <typename T>
inline size_t rocblas_internal_syrk_herk_workspace(rocblas_handle handle,
                                                   rocblas_int    n,
                                                   rocblas_int    k,
                                                   rocblas_int    batch_count)
{
    size_t size = 1;

    //Allocating workspace memory when only using gemm
    if(rocblas_use_only_gemm<T>(handle, n, k))
        if(n > 0 && batch_count > 0)
            size = ((int64_t(n) * (n - 1)) / 2) * sizeof(T) * batch_count;

    return size;
}

template <typename API_INT, typename TScal, typename TConstPtr, typename TPtr>
inline rocblas_status rocblas_syrk_arg_check(rocblas_handle    handle,
                                             rocblas_fill      uplo,
                                             rocblas_operation transA,
                                             API_INT           n,
                                             API_INT           k,
                                             const TScal*      alpha,
                                             TConstPtr         AP,
                                             rocblas_stride    offsetA,
                                             API_INT           lda,
                                             rocblas_stride    strideA,
                                             const TScal*      beta,
                                             TPtr              CP,
                                             rocblas_stride    offsetC,
                                             API_INT           ldc,
                                             rocblas_stride    strideC,
                                             API_INT           batch_count)
{
    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;

    if(rocblas_is_complex<TScal>)
    {
        if(transA != rocblas_operation_none && transA != rocblas_operation_transpose)
            return rocblas_status_invalid_value;
    }
    else
    {
        if(transA != rocblas_operation_none && transA != rocblas_operation_transpose
           && transA != rocblas_operation_conjugate_transpose)
            return rocblas_status_invalid_value;
    }

    if(n < 0 || k < 0 || batch_count < 0 || ldc < n || (transA == rocblas_operation_none && lda < n)
       || (transA != rocblas_operation_none && lda < k))
        return rocblas_status_invalid_size;

    if(!n || !batch_count)
        return rocblas_status_success;

    if((k > 0 && !alpha) || !beta)
        return rocblas_status_invalid_pointer;

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        bool calcA = k > 0 && *alpha != 0;

        if(!calcA && *beta == 1)
            return rocblas_status_success; // avoid slow kernel launches for no op

        if((calcA && !AP) || ((calcA || *beta != 1) && !CP))
            return rocblas_status_invalid_pointer;
    }

    return rocblas_status_continue;
}

template <typename API_INT, typename TScal, typename TConstPtr, typename TPtr>
inline rocblas_status rocblas_herk_arg_check(rocblas_handle    handle,
                                             rocblas_fill      uplo,
                                             rocblas_operation transA,
                                             API_INT           n,
                                             API_INT           k,
                                             TScal             alpha,
                                             TConstPtr         AP,
                                             rocblas_stride    offsetA,
                                             API_INT           lda,
                                             rocblas_stride    strideA,
                                             TScal             beta,
                                             TPtr              CP,
                                             rocblas_stride    offsetC,
                                             API_INT           ldc,
                                             rocblas_stride    strideC,
                                             API_INT           batch_count)
{
    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;
    if(transA != rocblas_operation_none && transA != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;

    if(n < 0 || k < 0 || batch_count < 0 || ldc < n || (transA == rocblas_operation_none && lda < n)
       || (transA != rocblas_operation_none && lda < k))
        return rocblas_status_invalid_size;

    if(!n || !batch_count)
        return rocblas_status_success;

    if((k > 0 && !alpha) || !beta)
        return rocblas_status_invalid_pointer;

    if(handle->pointer_mode == rocblas_pointer_mode_host)
    {
        bool calcA = k > 0 && *alpha != 0;

        if(!calcA && *beta == 1)
            return rocblas_status_success; // avoid slow kernel launches for no op

        if((calcA && !AP) || ((calcA || *beta != 1) && !CP))
            return rocblas_status_invalid_pointer;
    }

    return rocblas_status_continue;
}

template <rocblas_int NB,
          bool        BATCHED,
          bool        HERM,
          typename T,
          typename TScal,
          typename TConstPtr,
          typename TPtr>
rocblas_status rocblas_internal_syrk_herk_template(rocblas_handle    handle,
                                                   rocblas_fill      uplo,
                                                   rocblas_operation trans_A,
                                                   rocblas_int       n,
                                                   rocblas_int       k,
                                                   const TScal*      alpha_in,
                                                   TConstPtr         A,
                                                   rocblas_stride    offset_A,
                                                   rocblas_int       lda,
                                                   rocblas_stride    stride_A,
                                                   const TScal*      beta_in,
                                                   TPtr              C,
                                                   rocblas_stride    offset_C,
                                                   rocblas_int       ldc,
                                                   rocblas_stride    stride_C,
                                                   rocblas_int       batch_count);

template <bool HERM, typename TConstPtr, typename TPtr>
rocblas_status rocblas_herk_syrk_check_numerics(const char*       function_name,
                                                rocblas_handle    handle,
                                                rocblas_fill      uplo,
                                                rocblas_operation trans,
                                                int64_t           n_64,
                                                int64_t           k_64,
                                                TConstPtr         A,
                                                int64_t           lda_64,
                                                rocblas_stride    strideA,
                                                TPtr              C,
                                                int64_t           ldc_64,
                                                rocblas_stride    strideC,
                                                int64_t           batch_count_64,
                                                const int         check_numerics,
                                                bool              is_input);

/*
 * internal rocBLAS template function, also used by rocSOLVER.
 * Used for calls to rocblas_xsyrk() and rocblas_xsyrk_strided_batched()
 */
template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syrk_template(rocblas_handle    handle,
                                   rocblas_fill      uplo,
                                   rocblas_operation transA,
                                   rocblas_int       n,
                                   rocblas_int       k,
                                   const T*          alpha,
                                   const T*          A,
                                   rocblas_stride    offsetA,
                                   rocblas_int       lda,
                                   rocblas_stride    strideA,
                                   const T*          beta,
                                   T*                C,
                                   rocblas_stride    offsetC,
                                   rocblas_int       ldc,
                                   rocblas_stride    strideC,
                                   rocblas_int       batch_count);

/*
 * internal rocBLAS template function, also used by rocSOLVER.
 * Used for calls to rocblas_xsyrk_batched()
 */
template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syrk_batched_template(rocblas_handle    handle,
                                           rocblas_fill      uplo,
                                           rocblas_operation transA,
                                           rocblas_int       n,
                                           rocblas_int       k,
                                           const T*          alpha,
                                           const T* const*   A,
                                           rocblas_stride    offsetA,
                                           rocblas_int       lda,
                                           rocblas_stride    strideA,
                                           const T*          beta,
                                           T* const*         C,
                                           rocblas_stride    offsetC,
                                           rocblas_int       ldc,
                                           rocblas_stride    strideC,
                                           rocblas_int       batch_count);

/*
 * internal rocBLAS template function, also used by rocSOLVER.
 * Used for calls to rocblas_xherk() and rocblas_xherk_strided_batched()
 */
template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_herk_template(rocblas_handle    handle,
                                   rocblas_fill      uplo,
                                   rocblas_operation transA,
                                   rocblas_int       n,
                                   rocblas_int       k,
                                   const real_t<T>*  alpha,
                                   const T*          A,
                                   rocblas_stride    offsetA,
                                   rocblas_int       lda,
                                   rocblas_stride    strideA,
                                   const real_t<T>*  beta,
                                   T*                C,
                                   rocblas_stride    offsetC,
                                   rocblas_int       ldc,
                                   rocblas_stride    strideC,
                                   rocblas_int       batch_count);

/*
 * internal rocBLAS template function, also used by rocSOLVER.
 * Used for calls to rocblas_xherk_batched()
 */
template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_herk_batched_template(rocblas_handle    handle,
                                           rocblas_fill      uplo,
                                           rocblas_operation transA,
                                           rocblas_int       n,
                                           rocblas_int       k,
                                           const real_t<T>*  alpha,
                                           const T* const*   A,
                                           rocblas_stride    offsetA,
                                           rocblas_int       lda,
                                           rocblas_stride    strideA,
                                           const real_t<T>*  beta,
                                           T* const*         C,
                                           rocblas_stride    offsetC,
                                           rocblas_int       ldc,
                                           rocblas_stride    strideC,
                                           rocblas_int       batch_count);
