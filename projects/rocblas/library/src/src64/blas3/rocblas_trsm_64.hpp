/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "handle.hpp"

template <int BLOCK, int DIM_X, bool BATCHED, typename T, typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_trsm_launcher_64(rocblas_handle    handle,
                                                 rocblas_side      side,
                                                 rocblas_fill      uplo,
                                                 rocblas_operation transA,
                                                 rocblas_diagonal  diag,
                                                 int64_t           m_64,
                                                 int64_t           n_64,
                                                 const T*          alpha,
                                                 TConstPtr         A,
                                                 rocblas_stride    offset_A,
                                                 int64_t           lda_64,
                                                 rocblas_stride    stride_A,
                                                 TPtr              B,
                                                 rocblas_stride    offset_B,
                                                 int64_t           ldb_64,
                                                 rocblas_stride    stride_B,
                                                 int64_t           batch_count_64,
                                                 bool              optimal_mem,
                                                 void*             w_x_temp,
                                                 void*             w_x_temparr,
                                                 void*             invA                  = nullptr,
                                                 void*             invAarr               = nullptr,
                                                 TConstPtr         supplied_invA         = nullptr,
                                                 int64_t           supplied_invA_size_64 = 0,
                                                 rocblas_stride    offset_invA           = 0,
                                                 rocblas_stride    stride_invA           = 0);

template <bool BATCHED, typename T, typename U>
rocblas_status rocblas_internal_trsm_template_mem_64(rocblas_handle              handle,
                                                     rocblas_side                side,
                                                     rocblas_operation           transA,
                                                     int64_t                     m_64,
                                                     int64_t                     n_64,
                                                     int64_t                     lda_64,
                                                     int64_t                     ldb_64,
                                                     int64_t                     batch_count_64,
                                                     rocblas_device_malloc_base& w_mem,
                                                     void*&                      w_mem_x_temp,
                                                     void*&                      w_mem_x_temp_arr,
                                                     void*&                      w_mem_invA,
                                                     void*&                      w_mem_invA_arr,
                                                     U                           supplied_invA,
                                                     int64_t supplied_invA_size);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trsm_workspace_size_64(rocblas_side      side,
                                            rocblas_operation transA,
                                            int64_t           m_64,
                                            int64_t           n_64,
                                            int64_t           lda_64,
                                            int64_t           ldb_64,
                                            int64_t           batch_count_64,
                                            int64_t           supplied_invA_size,
                                            size_t*           w_x_tmp_size,
                                            size_t*           w_x_tmp_arr_size,
                                            size_t*           w_invA_size,
                                            size_t*           w_invA_arr_size,
                                            size_t*           w_x_tmp_size_backup);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trsm_batched_workspace_size_64(rocblas_side      side,
                                                    rocblas_operation transA,
                                                    int64_t           m_64,
                                                    int64_t           n_64,
                                                    int64_t           lda_64,
                                                    int64_t           ldb_64,
                                                    int64_t           batch_count_64,
                                                    int64_t           supplied_invA_size,
                                                    size_t*           w_x_tmp_size,
                                                    size_t*           w_x_tmp_arr_size,
                                                    size_t*           w_invA_size,
                                                    size_t*           w_invA_arr_size,
                                                    size_t*           w_x_tmp_size_backup);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trsm_workspace_max_size_64(rocblas_side side,
                                                int64_t      m,
                                                int64_t      n,
                                                int64_t      batch_count,
                                                size_t*      w_x_tmp_size,
                                                size_t*      w_invA_size,
                                                size_t*      w_x_tmp_size_backup);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trsm_batched_workspace_max_size_64(rocblas_side side,
                                                        int64_t      m,
                                                        int64_t      n,
                                                        int64_t      batch_count,
                                                        size_t*      w_x_tmp_size,
                                                        size_t*      w_x_tmp_arr_size,
                                                        size_t*      w_invA_size,
                                                        size_t*      w_invA_arr_size,
                                                        size_t*      w_x_tmp_size_backup);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trsm_template_64(rocblas_handle    handle,
                                      rocblas_side      side,
                                      rocblas_fill      uplo,
                                      rocblas_operation transA,
                                      rocblas_diagonal  diag,
                                      int64_t           m_64,
                                      int64_t           n_64,
                                      const T*          alpha,
                                      const T*          A,
                                      rocblas_stride    offset_A,
                                      int64_t           lda_64,
                                      rocblas_stride    stride_A,
                                      T*                B,
                                      rocblas_stride    offset_B,
                                      int64_t           ldb_64,
                                      rocblas_stride    stride_B,
                                      int64_t           batch_count_64,
                                      bool              optimal_mem,
                                      void*             w_x_temp,
                                      void*             w_x_temparr,
                                      void*             invA                  = nullptr,
                                      void*             invAarr               = nullptr,
                                      const T*          supplied_invA         = nullptr,
                                      int64_t           supplied_invA_size_64 = 0,
                                      rocblas_stride    offset_invA           = 0,
                                      rocblas_stride    stride_invA           = 0);

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trsm_batched_template_64(rocblas_handle    handle,
                                              rocblas_side      side,
                                              rocblas_fill      uplo,
                                              rocblas_operation transA,
                                              rocblas_diagonal  diag,
                                              int64_t           m_64,
                                              int64_t           n_64,
                                              const T*          alpha,
                                              const T* const*   A,
                                              rocblas_stride    offset_A,
                                              int64_t           lda_64,
                                              rocblas_stride    stride_A,
                                              T* const*         B,
                                              rocblas_stride    offset_B,
                                              int64_t           ldb_64,
                                              rocblas_stride    stride_B,
                                              int64_t           batch_count_64,
                                              bool              optimal_mem,
                                              void*             w_x_temp,
                                              void*             w_x_temparr,
                                              void*             invA                  = nullptr,
                                              void*             invAarr               = nullptr,
                                              const T* const*   supplied_invA         = nullptr,
                                              int64_t           supplied_invA_size_64 = 0,
                                              rocblas_stride    offset_invA           = 0,
                                              rocblas_stride    stride_invA           = 0);
