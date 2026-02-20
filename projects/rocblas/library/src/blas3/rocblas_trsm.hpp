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

#pragma once

#include "definitions.hpp"
#include "handle.hpp"
#include "rocblas.h"

template <typename API_INT, typename TScal, typename TPtr, typename TConstPtr>
inline rocblas_status rocblas_trsm_arg_check(rocblas_handle    handle,
                                             rocblas_side      side,
                                             rocblas_fill      uplo,
                                             rocblas_operation transA,
                                             rocblas_diagonal  diag,
                                             API_INT           m,
                                             API_INT           n,
                                             const TScal*      alpha,
                                             TConstPtr         A,
                                             API_INT           lda,
                                             TPtr              B,
                                             API_INT           ldb,
                                             API_INT           batch_count)
{
    if(side != rocblas_side_left && side != rocblas_side_right)
        return rocblas_status_invalid_value;

    if(uplo != rocblas_fill_lower && uplo != rocblas_fill_upper)
        return rocblas_status_invalid_value;

    if(transA != rocblas_operation_none && transA != rocblas_operation_transpose
       && transA != rocblas_operation_conjugate_transpose)
        return rocblas_status_invalid_value;

    if(diag != rocblas_diagonal_non_unit && diag != rocblas_diagonal_unit)
        return rocblas_status_invalid_value;

    // A is of size lda*k
    auto k = side == rocblas_side_left ? m : n;
    if(batch_count < 0 || m < 0 || n < 0 || lda < k || ldb < m)
        return rocblas_status_invalid_size;

    // quick return if possible.
    if(!m || !n || !batch_count)
        return handle->is_device_memory_size_query() ? rocblas_status_size_unchanged
                                                     : rocblas_status_success;

    if(!handle->is_device_memory_size_query())
    {
        if(!B || !alpha || (handle->pointer_mode == rocblas_pointer_mode_host && *alpha != 0 && !A))
            return rocblas_status_invalid_pointer;
    }

    return rocblas_status_continue;
}

template <typename T, typename U>
rocblas_status set_block_unit(rocblas_handle handle,
                              int64_t        m,
                              int64_t        n,
                              U              src,
                              int64_t        src_ld,
                              rocblas_stride src_stride,
                              rocblas_int    batch_count,
                              T              val        = 0.0,
                              rocblas_stride offset_src = 0);

template <bool BATCHED, typename T, typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_trsm_small_substitution_launcher(rocblas_handle    handle,
                                                                 rocblas_side      side,
                                                                 rocblas_fill      uplo,
                                                                 rocblas_operation transA,
                                                                 rocblas_diagonal  diag,
                                                                 int64_t           m,
                                                                 int64_t           n,
                                                                 T                 alpha_h,
                                                                 TConstPtr         A,
                                                                 rocblas_stride    offset_A,
                                                                 int64_t           lda,
                                                                 rocblas_stride    stride_A,
                                                                 TPtr              B,
                                                                 rocblas_stride    offset_B,
                                                                 int64_t           ldb,
                                                                 rocblas_stride    stride_B,
                                                                 int               batch_count,
                                                                 int               blksize);

template <rocblas_int BLOCK, rocblas_int DIM_X, bool BATCHED, typename T, typename U, typename V>
rocblas_status rocblas_internal_trsm_launcher(rocblas_handle    handle,
                                              rocblas_side      side,
                                              rocblas_fill      uplo,
                                              rocblas_operation transA,
                                              rocblas_diagonal  diag,
                                              rocblas_int       m,
                                              rocblas_int       n,
                                              const T*          alpha,
                                              U                 A,
                                              rocblas_stride    offset_A,
                                              rocblas_int       lda,
                                              rocblas_stride    stride_A,
                                              V                 B,
                                              rocblas_stride    offset_B,
                                              rocblas_int       ldb,
                                              rocblas_stride    stride_B,
                                              rocblas_int       batch_count,
                                              bool              optimal_mem,
                                              void*             w_x_temp,
                                              void*             w_x_temparr,
                                              void*             invA,
                                              void*             invAarr,
                                              U                 supplied_invA,
                                              rocblas_int       supplied_invA_size,
                                              rocblas_stride    offset_invA,
                                              rocblas_stride    stride_invA);

template <rocblas_int BLOCK, bool BATCHED, typename T>
rocblas_status rocblas_internal_trsm_workspace_size(rocblas_side      side,
                                                    rocblas_operation transA,
                                                    rocblas_int       m,
                                                    rocblas_int       n,
                                                    rocblas_int       batch_count,
                                                    rocblas_int       supplied_invA_size,
                                                    size_t*           w_x_tmp_size,
                                                    size_t*           w_x_tmp_arr_size,
                                                    size_t*           w_invA_size,
                                                    size_t*           w_invA_arr_size,
                                                    size_t*           w_x_tmp_size_backup);

template <bool BATCHED, typename T, typename U>
rocblas_status rocblas_internal_trsm_template_mem(rocblas_handle              handle,
                                                  rocblas_side                side,
                                                  rocblas_operation           transA,
                                                  rocblas_int                 m,
                                                  rocblas_int                 n,
                                                  rocblas_int                 lda,
                                                  rocblas_int                 ldb,
                                                  rocblas_int                 batch_count,
                                                  rocblas_device_malloc_base& w_mem,
                                                  void*&                      w_mem_x_temp,
                                                  void*&                      w_mem_x_temp_arr,
                                                  void*&                      w_mem_invA,
                                                  void*&                      w_mem_invA_arr,
                                                  U                           supplied_invA,
                                                  rocblas_int                 supplied_invA_size);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trsm_workspace_size(rocblas_side      side,
                                         rocblas_operation transA,
                                         rocblas_int       m,
                                         rocblas_int       n,
                                         rocblas_int       batch_count,
                                         rocblas_int       supplied_invA_size,
                                         size_t*           w_x_tmp_size,
                                         size_t*           w_x_tmp_arr_size,
                                         size_t*           w_invA_size,
                                         size_t*           w_invA_arr_size,
                                         size_t*           w_x_tmp_size_backup);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trsm_batched_workspace_size(rocblas_side      side,
                                                 rocblas_operation transA,
                                                 rocblas_int       m,
                                                 rocblas_int       n,
                                                 rocblas_int       batch_count,
                                                 rocblas_int       supplied_invA_size,
                                                 size_t*           w_x_tmp_size,
                                                 size_t*           w_x_tmp_arr_size,
                                                 size_t*           w_invA_size,
                                                 size_t*           w_invA_arr_size,
                                                 size_t*           w_x_tmp_size_backup);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trsm_template(rocblas_handle    handle,
                                   rocblas_side      side,
                                   rocblas_fill      uplo,
                                   rocblas_operation transA,
                                   rocblas_diagonal  diag,
                                   rocblas_int       m,
                                   rocblas_int       n,
                                   const T*          alpha,
                                   const T*          A,
                                   rocblas_stride    offset_A,
                                   rocblas_int       lda,
                                   rocblas_stride    stride_A,
                                   T*                B,
                                   rocblas_stride    offset_B,
                                   rocblas_int       ldb,
                                   rocblas_stride    stride_B,
                                   rocblas_int       batch_count,
                                   bool              optimal_mem,
                                   void*             w_x_temp,
                                   void*             w_x_temparr,
                                   void*             invA               = nullptr,
                                   void*             invAarr            = nullptr,
                                   const T*          supplied_invA      = nullptr,
                                   rocblas_int       supplied_invA_size = 0,
                                   rocblas_stride    offset_invA        = 0,
                                   rocblas_stride    stride_invA        = 0);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trsm_batched_template(rocblas_handle    handle,
                                           rocblas_side      side,
                                           rocblas_fill      uplo,
                                           rocblas_operation transA,
                                           rocblas_diagonal  diag,
                                           rocblas_int       m,
                                           rocblas_int       n,
                                           const T*          alpha,
                                           const T* const*   A,
                                           rocblas_stride    offset_A,
                                           rocblas_int       lda,
                                           rocblas_stride    stride_A,
                                           T* const*         B,
                                           rocblas_stride    offset_B,
                                           rocblas_int       ldb,
                                           rocblas_stride    stride_B,
                                           rocblas_int       batch_count,
                                           bool              optimal_mem,
                                           void*             w_x_temp,
                                           void*             w_x_temparr,
                                           void*             invA               = nullptr,
                                           void*             invAarr            = nullptr,
                                           const T* const*   supplied_invA      = nullptr,
                                           rocblas_int       supplied_invA_size = 0,
                                           rocblas_stride    offset_invA        = 0,
                                           rocblas_stride    stride_invA        = 0);
