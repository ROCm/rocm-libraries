/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "int64_helpers.hpp" // c_i64_grid_YZ_chunk
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

// Upper bound on the syrk/herk gemm-path workspace.  Capping the chunk by bytes
// rather than by a batch count keeps the allocation bounded for every n: a cap
// expressed in batches still permits tri(n) * sizeof(T) * cap bytes, which at
// n=1024, batch_count=50000 is 98GB.
//
// The value trades peak memory against launch count, since a smaller budget
// means more chunks.  1GB keeps the common small-n, high-batch_count shapes on
// exactly the launch count the unchunked code used, while still bounding the
// large-n shapes that cannot run at all today.
constexpr size_t c_syrk_herk_workspace_max_bytes = size_t(1024) * 1024 * 1024;

// The budget in effect, with a test-only override.
//
// The override exists because the default is deliberately large, so every shape
// a test can afford to allocate fits in a single chunk: C is on the order of
// twice the unchunked workspace, so forcing a split at the default needs several
// GB of C. Without a way to lower the budget the whole chunk loop is unreachable
// from a test.
//
// Read on each call rather than cached, so a test can vary it between cases. The
// cost is one getenv against a call that already requires k >= 500.
inline size_t rocblas_syrk_herk_workspace_budget()
{
    if(const char* env = std::getenv("ROCBLAS_INTERNAL_SYRK_HERK_WORKSPACE_MAX_BYTES"))
    {
        const long long requested = std::atoll(env);
        if(requested > 0)
            return size_t(requested);
    }
    return c_syrk_herk_workspace_max_bytes;
}

// Batches processed per chunk by the gemm-path launcher.
//
// Both rocblas_internal_syrk_herk_workspace and the launcher in
// rocblas_syrk_herk_kernels.cpp MUST derive the chunk from this function: the
// query sizes the buffer for one chunk, so any disagreement lets the launcher
// write past the allocation.
//
// The chunk is the largest batch count whose triangle slots fit the byte budget,
// so a problem small enough to fit entirely takes a single pass and issues the
// same launches as the unchunked code.  When the budget does force a split, the
// chunk is rounded down to a multiple of c_i64_grid_YZ_chunk, which is the stride
// rocblas_internal_gemm_64 uses for its own batch loop; that keeps every full
// chunk exactly one GEMM launch instead of a full launch plus a short remainder.
inline rocblas_int
    rocblas_syrk_herk_chunk_size(rocblas_int n, rocblas_int batch_count, size_t elem_size)
{
    const size_t per_batch = (size_t(n) * size_t(n - 1) / 2) * elem_size;

    // tri(n) == 0 (n <= 1): no workspace is consumed, so one pass covers everything.
    if(!per_batch)
        return batch_count;

    size_t chunk = rocblas_syrk_herk_workspace_budget() / per_batch;
    if(chunk < 1)
        chunk = 1; // a single batch always has to fit
    if(chunk > size_t(batch_count))
        chunk = size_t(batch_count);

    if(chunk < size_t(batch_count) && chunk > size_t(c_i64_grid_YZ_chunk))
        chunk = (chunk / size_t(c_i64_grid_YZ_chunk)) * size_t(c_i64_grid_YZ_chunk);

    return rocblas_int(chunk);
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
        {
            // Peak allocation is one chunk's worth of triangle slots, not the whole
            // batch count, because the launcher reuses the buffer for every chunk.
            // All arithmetic uses size_t to prevent signed overflow in the product
            // tri(n) * sizeof(T) * chunk.
            size_t chunk = size_t(rocblas_syrk_herk_chunk_size(n, batch_count, sizeof(T)));
            size         = (size_t(n) * size_t(n - 1) / 2) * sizeof(T) * chunk;
        }

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

// helper
template <bool copy_from_C_to_W_C, bool is_upper, bool HERM, typename T, typename TPtr>
rocblas_status rocblas_copy_triangular_syrk_herk(rocblas_handle handle,
                                                 rocblas_int    n,
                                                 TPtr           C,
                                                 rocblas_int    ldc,
                                                 rocblas_stride stride_C,
                                                 T*             W_C,
                                                 rocblas_int    chunk_size,
                                                 rocblas_int    batch_offset);
