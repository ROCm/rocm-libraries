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

#include "handle.hpp"
#include "int64_helpers.hpp"
#include "rocblas_block_sizes.h"

#include "rocblas_trsm_64.hpp"

#include "blas3/rocblas_trsm.hpp" // int32 API called

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
                                                 void*             invA,
                                                 void*             invAarr,
                                                 TConstPtr         supplied_invA,
                                                 int64_t           supplied_invA_size_64,
                                                 rocblas_stride    offset_invA,
                                                 rocblas_stride    stride_invA)
{
    if(!m_64 || !n_64 || !batch_count_64)
        return rocblas_status_success;

    if((m_64 > c_i32_max && side == rocblas_side_left)
       || (n_64 > c_i32_max && side == rocblas_side_right))
    {
        // exceeds practical memory
        return rocblas_status_invalid_size;
    }

    // may be able to call 32-bit trsm
    if(n_64 <= c_ILP64_i32_max && m_64 < c_ILP64_i32_max && lda_64 < c_ILP64_i32_max
       && ldb_64 < c_ILP64_i32_max && supplied_invA_size_64 < c_ILP64_i32_max
       && batch_count_64 < c_i64_grid_YZ_chunk)
        return rocblas_internal_trsm_launcher<BLOCK, DIM_X, BATCHED>(handle,
                                                                     side,
                                                                     uplo,
                                                                     transA,
                                                                     diag,
                                                                     m_64,
                                                                     n_64,
                                                                     alpha,
                                                                     A,
                                                                     offset_A,
                                                                     lda_64,
                                                                     stride_A,
                                                                     B,
                                                                     offset_B,
                                                                     ldb_64,
                                                                     stride_B,
                                                                     batch_count_64,
                                                                     optimal_mem,
                                                                     w_x_temp,
                                                                     w_x_temparr,
                                                                     invA,
                                                                     invAarr,
                                                                     supplied_invA,
                                                                     supplied_invA_size_64,
                                                                     offset_invA,
                                                                     stride_invA);

    for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
    {
        int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

        auto A_ptr = adjust_ptr_batch(A, b_base, stride_A);
        auto B_ptr = adjust_ptr_batch(B, b_base, stride_B);
        // with w_x_ptr we don't have to offset as we can reuse the memory between kernel launches
        // with invA we don't have to offset either. However, this is NOT true for trsm_ex. This adjustment should work
        // for trsm_ex, but it isn't supported yet so it isn't tested either
        auto invA_ptr
            = adjust_ptr_batch((const T*)invA, b_base, (supplied_invA ? supplied_invA_size_64 : 0));
        auto invAarr_ptr
            = adjust_ptr_batch((const T* const*)invAarr, (supplied_invA ? b_base : 0), 0);

        // may be able to call 32-bit trsm while only iterating through batches. Need to be careful about
        // allocated memory and offsets between groups of kernel launches
        if(n_64 <= c_ILP64_i32_max && m_64 < c_ILP64_i32_max && lda_64 < c_ILP64_i32_max
           && ldb_64 < c_ILP64_i32_max && supplied_invA_size_64 < c_ILP64_i32_max)
        {
            auto status
                = rocblas_internal_trsm_launcher<BLOCK, DIM_X, BATCHED>(handle,
                                                                        side,
                                                                        uplo,
                                                                        transA,
                                                                        diag,
                                                                        m_64,
                                                                        n_64,
                                                                        alpha,
                                                                        A_ptr,
                                                                        offset_A,
                                                                        lda_64,
                                                                        stride_A,
                                                                        B_ptr,
                                                                        offset_B,
                                                                        ldb_64,
                                                                        stride_B,
                                                                        batch_count,
                                                                        optimal_mem,
                                                                        w_x_temp,
                                                                        w_x_temparr,
                                                                        (void*)invA_ptr,
                                                                        (void*)invAarr_ptr,
                                                                        supplied_invA,
                                                                        supplied_invA_size_64,
                                                                        offset_invA,
                                                                        stride_invA);

            if(status != rocblas_status_success)
                return status;
        }
        else
        {
            // A couple cases are possible here:
            // 1. side == LEFT
            //     a) m > 32-bit
            //        - this isn't possible with current memory restrictions as A is m x m
            //     b) n > 32-bit
            //        - this is possible, but m and ldb will have to be fairly small given memory restrictions
            //        - this can use a simple substitution method that doesn't need GEMMs
            //     c) lda > 32-bit
            //        - this is possible, but m will have to be fairly small
            //        - can use simple substitution without GEMM
            //     d) ldb > 32-bit
            //        - this is possible, but n will have to be fairly small
            //        - here, m can be large. We can use a substitution method which uses GEMMs for better performance
            // 2. side == RIGHT
            //     a) m > 32-bit
            //        - this is possible, but n will be fairly small
            //        - can use substitution method with GEMMs
            //     b) n > 32-bit
            //        - this isn't possible with current memory restrictions
            //     c) lda > 32-bit
            //        - essentially the same as the large m case as n must be small
            //     d) ldb > 32-bit
            //        - essentially the same as the large m case as n must be small

            // In all cases we'll be using the substitution method in trsm for now.
            // Tensile doesn't currently support 64-bit params, so we need to have a solution which doesn't depend
            // on Tensile

            // not worrying about tuning block sizes right now
            rocblas_int blksize = 64;

            // Temporarily switch to host pointer mode, saving current pointer mode, restored on return
            auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

            // Get alpha - Check if zero for quick return
            T alpha_h;
            if(saved_pointer_mode == rocblas_pointer_mode_host)
                alpha_h = *alpha;
            else
            {
                RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                    &alpha_h, alpha, sizeof(T), hipMemcpyDeviceToHost, handle->get_stream()));
                RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->get_stream()));
            }

            if(alpha_h == T(0.0))
            {
                return set_block_unit<T>(
                    handle, m_64, n_64, B, ldb_64, stride_B, batch_count, 0.0, offset_B);
            }

            return rocblas_internal_trsm_small_substitution_launcher<BATCHED>(handle,
                                                                              side,
                                                                              uplo,
                                                                              transA,
                                                                              diag,
                                                                              m_64,
                                                                              n_64,
                                                                              alpha_h,
                                                                              A_ptr,
                                                                              offset_A,
                                                                              lda_64,
                                                                              stride_A,
                                                                              B_ptr,
                                                                              offset_B,
                                                                              ldb_64,
                                                                              stride_B,
                                                                              batch_count,
                                                                              blksize);
        }
    }

    return rocblas_status_success;
}

template <bool BATCHED, typename T, typename U>
rocblas_status rocblas_internal_trsm_template_mem_64(rocblas_handle              handle,
                                                     rocblas_side                side,
                                                     rocblas_operation           transA,
                                                     int64_t                     m,
                                                     int64_t                     n,
                                                     int64_t                     lda,
                                                     int64_t                     ldb,
                                                     int64_t                     batch_count,
                                                     rocblas_device_malloc_base& w_mem,
                                                     void*&                      w_mem_x_temp,
                                                     void*&                      w_mem_x_temp_arr,
                                                     void*&                      w_mem_invA,
                                                     void*&                      w_mem_invA_arr,
                                                     U                           supplied_invA,
                                                     int64_t                     supplied_invA_size)
{
    // Potentially use 32-bit trsm
    if(n <= c_ILP64_i32_max && m < c_ILP64_i32_max && lda < c_ILP64_i32_max && ldb < c_ILP64_i32_max
       && supplied_invA_size < c_ILP64_i32_max)
    {
        return rocblas_internal_trsm_template_mem<BATCHED, T>(
            handle,
            side,
            transA,
            m,
            n,
            lda,
            ldb,
            std::min(batch_count, c_i64_grid_YZ_chunk),
            w_mem,
            w_mem_x_temp,
            w_mem_x_temp_arr,
            w_mem_invA,
            w_mem_invA_arr,
            supplied_invA,
            supplied_invA_size);
    }

    auto& workspace = static_cast<decltype(handle->device_malloc(0))&>(w_mem);

    // Otherwsie we have 64-bi specific memory allocation
    size_t w_x_tmp_size, w_x_tmp_arr_size, w_invA_size, w_invA_arr_size, w_x_tmp_size_backup;
    auto   memory_status = rocblas_internal_trsm_workspace_size_64<ROCBLAS_TRSM_NB, BATCHED, T>(
        side,
        transA,
        m,
        n,
        lda,
        ldb,
        batch_count,
        supplied_invA_size,
        &w_x_tmp_size,
        &w_x_tmp_arr_size,
        &w_invA_size,
        &w_invA_arr_size,
        &w_x_tmp_size_backup);

    if(memory_status != rocblas_status_success && memory_status != rocblas_status_continue)
    {
        return memory_status;
    }

    if(handle->is_device_memory_size_query())
    {
        // indicates no memory needed
        if(memory_status == rocblas_status_continue)
        {
            return rocblas_status_size_unchanged;
        }
        else
        {
            return handle->set_optimal_device_memory_size(
                w_x_tmp_size, w_x_tmp_arr_size, w_invA_size, w_invA_arr_size);
        }
    }

    // only need memory if workspace_size query returned rocblas_status_success
    if(memory_status == rocblas_status_success)
    {
        workspace
            = handle->device_malloc(w_x_tmp_size, w_x_tmp_arr_size, w_invA_size, w_invA_arr_size);
        if(!workspace)
            return rocblas_status_memory_error;

        w_mem_x_temp     = workspace[0];
        w_mem_x_temp_arr = workspace[1];
        w_mem_invA       = workspace[2];
        w_mem_invA_arr   = workspace[3];
    }

    return rocblas_status_success;
}

template <int BLOCK, bool BATCHED, typename T>
rocblas_status rocblas_internal_trsm_workspace_size_64(rocblas_side      side,
                                                       rocblas_operation transA,
                                                       int64_t           m,
                                                       int64_t           n,
                                                       int64_t           lda,
                                                       int64_t           ldb,
                                                       int64_t           batch_count,
                                                       int64_t           supplied_invA_size,
                                                       size_t*           w_x_tmp_size,
                                                       size_t*           w_x_tmp_arr_size,
                                                       size_t*           w_invA_size,
                                                       size_t*           w_invA_arr_size,
                                                       size_t*           w_x_tmp_size_backup)
{
    if(!w_x_tmp_size || !w_x_tmp_arr_size || !w_invA_size || !w_invA_arr_size
       || !w_x_tmp_size_backup)
    {
        return rocblas_status_invalid_pointer;
    }

    // We need lda and ldb here to see if we can use 32-bit alogrithms or not
    // Note that we don't need to check batch_count as we use multiple calls of the 32-bit kernel if it's too large
    if(n <= c_ILP64_i32_max && m < c_ILP64_i32_max && lda < c_ILP64_i32_max && ldb < c_ILP64_i32_max
       && supplied_invA_size < c_ILP64_i32_max)
    {
        return rocblas_internal_trsm_workspace_size<BLOCK, BATCHED, T>(
            side,
            transA,
            m,
            n,
            std::min(batch_count, c_i64_grid_YZ_chunk),
            supplied_invA_size,
            w_x_tmp_size,
            w_x_tmp_arr_size,
            w_invA_size,
            w_invA_arr_size,
            w_x_tmp_size_backup);
    }

    // otherwise, we use 64-bit kernel which currently doesn't need any workspace memory
    // return rocblas_status_continue to indicate no memory is needed
    *w_x_tmp_size        = 0;
    *w_x_tmp_arr_size    = 0;
    *w_invA_size         = 0;
    *w_invA_arr_size     = 0;
    *w_x_tmp_size_backup = 0;
    return rocblas_status_continue;
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trsm_workspace_size_64(rocblas_side      side,
                                            rocblas_operation transA,
                                            int64_t           m,
                                            int64_t           n,
                                            int64_t           lda,
                                            int64_t           ldb,
                                            int64_t           batch_count,
                                            int64_t           supplied_invA_size,
                                            size_t*           w_x_tmp_size,
                                            size_t*           w_x_tmp_arr_size,
                                            size_t*           w_invA_size,
                                            size_t*           w_invA_arr_size,
                                            size_t*           w_x_tmp_size_backup)
{
    return rocblas_internal_trsm_workspace_size_64<ROCBLAS_TRSM_NB, false, T>(side,
                                                                              transA,
                                                                              m,
                                                                              n,
                                                                              lda,
                                                                              ldb,
                                                                              batch_count,
                                                                              supplied_invA_size,
                                                                              w_x_tmp_size,
                                                                              w_x_tmp_arr_size,
                                                                              w_invA_size,
                                                                              w_invA_arr_size,
                                                                              w_x_tmp_size_backup);
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trsm_batched_workspace_size_64(rocblas_side      side,
                                                    rocblas_operation transA,
                                                    int64_t           m,
                                                    int64_t           n,
                                                    int64_t           lda,
                                                    int64_t           ldb,
                                                    int64_t           batch_count,
                                                    int64_t           supplied_invA_size,
                                                    size_t*           w_x_tmp_size,
                                                    size_t*           w_x_tmp_arr_size,
                                                    size_t*           w_invA_size,
                                                    size_t*           w_invA_arr_size,
                                                    size_t*           w_x_tmp_size_backup)
{
    return rocblas_internal_trsm_workspace_size_64<ROCBLAS_TRSM_NB, true, T>(side,
                                                                             transA,
                                                                             m,
                                                                             n,
                                                                             lda,
                                                                             ldb,
                                                                             batch_count,
                                                                             supplied_invA_size,
                                                                             w_x_tmp_size,
                                                                             w_x_tmp_arr_size,
                                                                             w_invA_size,
                                                                             w_invA_arr_size,
                                                                             w_x_tmp_size_backup);
}

/*! \brief rocblas_internal_trsm_workspace_max_size_64

    Calculates the maximum needed memory allocation for trsm or trsm_strided_batched for sizes
    where m <= M and n <= N. Does not allocate any memory.

    @param[in]
    side rocblas_side
        Whether matrix A is located on the left or right of X
    @param[in]
    M int64_t
        Number of rows of matrix B
    @param[in]
    N int64_t
        Number of columns of matrix B
    @param[in]
    batch_count int64_t
        Number of batches
    @param[out]
    w_x_tmp_size size_t
        The bytes of workspace memory needed for x_tmp in the trsm calculations
    @param[out]
    w_invA_size size_t
        The bytes of workspace memory needed for invA in the trsm calculations
    @param[out]
    w_x_tmp_size_backup size_t
        If the user is unable to allocate w_x_tmp_arr_size bytes, w_x_tmp_size_backup
        bytes may be used in trsm with degraded performance.
    ********************************************************************/
template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trsm_workspace_max_size_64(rocblas_side side,
                                                int64_t      M,
                                                int64_t      N,
                                                int64_t      batch_count,
                                                size_t*      w_x_tmp_size,
                                                size_t*      w_invA_size,
                                                size_t*      w_x_tmp_size_backup)
{
    size_t tmp_x_arr_size, tmp_inva_arr_size;
    return rocblas_internal_trsm_workspace_max_size_64<false, T>(side,
                                                                 M,
                                                                 N,
                                                                 batch_count,
                                                                 w_x_tmp_size,
                                                                 &tmp_x_arr_size,
                                                                 w_invA_size,
                                                                 &tmp_inva_arr_size,
                                                                 w_x_tmp_size_backup);
}

template <bool BATCHED, typename T>
rocblas_status rocblas_internal_trsm_workspace_max_size_64(rocblas_side side,
                                                           int64_t      m,
                                                           int64_t      n,
                                                           int64_t      batch_count,
                                                           size_t*      w_x_tmp_size,
                                                           size_t*      w_x_tmp_arr_size,
                                                           size_t*      w_invA_size,
                                                           size_t*      w_invA_arr_size,
                                                           size_t*      w_x_tmp_size_backup)
{
    if(!w_x_tmp_size || !w_x_tmp_arr_size || !w_invA_size || !w_invA_arr_size
       || !w_x_tmp_size_backup)
    {
        return rocblas_status_invalid_pointer;
    }

    int64_t k  = side == rocblas_side_left ? m : n;
    int64_t k2 = side == rocblas_side_left ? n : m;

    bool is_small = (k <= 32) || (m <= 64 && n <= 64);

    // if the max m and n sizes fit within this "small" criteria, all smaller sizes will also use the "small" kernel, which
    // doesn't need any workspace memory
    // This isn't necessarily true for the substitution method, a larger size may use subsitution while a smaller size uses
    // inversion kernel
    if(is_small)
    {
        *w_x_tmp_size        = 0;
        *w_x_tmp_arr_size    = 0;
        *w_invA_size         = 0;
        *w_invA_arr_size     = 0;
        *w_x_tmp_size_backup = 0;
    }

    int64_t x_size1 = ROCBLAS_TRSM_NB * k / 4;
    int64_t x_size2 = ROCBLAS_TRSM_NB * ROCBLAS_TRTRI_NB * 2;
    int64_t x_size3 = ROCBLAS_TRSM_NB * k2;

    // this will usually be the largest. Many sizes, typically when divisible by ROCBLAS_TRSM_NB, don't need this much memory,
    // but not taking that into account here
    int64_t x_size4 = m * n;

    *w_x_tmp_size = sizeof(T) * batch_count
                    * std::max(std::max(x_size1, x_size2), std::max(x_size3, x_size4));
    *w_x_tmp_size_backup = sizeof(T) * batch_count * std::max(x_size4, int64_t(ROCBLAS_TRSM_NB));

    // temporary memory for invA
    *w_invA_size = ROCBLAS_TRSM_NB * k * sizeof(T) * batch_count;

    // if batched we need memory to hold pointers
    *w_x_tmp_arr_size = BATCHED ? sizeof(T*) * batch_count : 0;
    *w_invA_arr_size  = BATCHED ? sizeof(T*) * batch_count : 0;
    return rocblas_status_success;
}

/*! \brief rocblas_internal_trsm_batched_workspace_max_size_64

    Calculates the maximum needed memory allocation for trsm_batched for sizes
    where m <= M and n <= N. Does not allocate any memory.

    @param[in]
    side rocblas_side
        Whether matrix A is located on the left or right of X
    @param[in]
    M int64_t
        Number of rows of matrix B
    @param[in]
    N int64_t
        Number of columns of matrix B
    @param[in]
    batch_count int64_t
        Number of batches
    @param[out]
    w_x_tmp_size size_t
        The bytes of workspace memory needed for x_tmp in the trsm calculations
    @param[out]
    w_x_tmp_arr_size size_t
        The bytes of workspace memory needed for the array of pointers for x_tmp
    @param[out]
    w_invA_size size_t
        The bytes of workspace memory needed for invA in the trsm calculations
    @param[out]
    w_invA_arr_size size_t
        The bytes of workspace memory needed for the array of pointers for invA
    @param[out]
    w_x_tmp_size_backup size_t
        If the user is unable to allocate w_x_tmp_arr_size bytes, w_x_tmp_size_backup
        bytes may be used in trsm with degraded performance.
    ********************************************************************/
template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trsm_batched_workspace_max_size_64(rocblas_side side,
                                                        int64_t      M,
                                                        int64_t      N,
                                                        int64_t      batch_count,
                                                        size_t*      w_x_tmp_size,
                                                        size_t*      w_x_tmp_arr_size,
                                                        size_t*      w_invA_size,
                                                        size_t*      w_invA_arr_size,
                                                        size_t*      w_x_tmp_size_backup)
{
    return rocblas_internal_trsm_workspace_max_size_64<true, T>(side,
                                                                M,
                                                                N,
                                                                batch_count,
                                                                w_x_tmp_size,
                                                                w_x_tmp_arr_size,
                                                                w_invA_size,
                                                                w_invA_arr_size,
                                                                w_x_tmp_size_backup);
}

#define TRSM_TEMPLATE_PARAMS                                                                     \
    handle, side, uplo, transA, diag, m, n, alpha, A, offset_A, lda, stride_A, B, offset_B, ldb, \
        stride_B, batch_count, optimal_mem, w_x_temp, w_x_temparr, invA, invAarr, supplied_invA, \
        supplied_invA_size, offset_invA, stride_invA

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trsm_template_64(rocblas_handle    handle,
                                      rocblas_side      side,
                                      rocblas_fill      uplo,
                                      rocblas_operation transA,
                                      rocblas_diagonal  diag,
                                      int64_t           m,
                                      int64_t           n,
                                      const T*          alpha,
                                      const T*          A,
                                      rocblas_stride    offset_A,
                                      int64_t           lda,
                                      rocblas_stride    stride_A,
                                      T*                B,
                                      rocblas_stride    offset_B,
                                      int64_t           ldb,
                                      rocblas_stride    stride_B,
                                      int64_t           batch_count,
                                      bool              optimal_mem,
                                      void*             w_x_temp,
                                      void*             w_x_temparr,
                                      void*             invA,
                                      void*             invAarr,
                                      const T*          supplied_invA,
                                      int64_t           supplied_invA_size,
                                      rocblas_stride    offset_invA,
                                      rocblas_stride    stride_invA)
{
    if constexpr(std::is_same_v<T, float>)
        return rocblas_internal_trsm_launcher_64<ROCBLAS_TRSM_NB, ROCBLAS_SDCTRSV_NB, false>(
            TRSM_TEMPLATE_PARAMS);
    else if constexpr(std::is_same_v<T, double>)
        return rocblas_internal_trsm_launcher_64<ROCBLAS_TRSM_NB, ROCBLAS_SDCTRSV_NB, false>(
            TRSM_TEMPLATE_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_float_complex>)
        return rocblas_internal_trsm_launcher_64<ROCBLAS_TRSM_NB, ROCBLAS_SDCTRSV_NB, false>(
            TRSM_TEMPLATE_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_double_complex>)
        return rocblas_internal_trsm_launcher_64<ROCBLAS_TRSM_NB, ROCBLAS_ZTRSV_NB, false>(
            TRSM_TEMPLATE_PARAMS);
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_trsm_batched_template_64(rocblas_handle    handle,
                                              rocblas_side      side,
                                              rocblas_fill      uplo,
                                              rocblas_operation transA,
                                              rocblas_diagonal  diag,
                                              int64_t           m,
                                              int64_t           n,
                                              const T*          alpha,
                                              const T* const*   A,
                                              rocblas_stride    offset_A,
                                              int64_t           lda,
                                              rocblas_stride    stride_A,
                                              T* const*         B,
                                              rocblas_stride    offset_B,
                                              int64_t           ldb,
                                              rocblas_stride    stride_B,
                                              int64_t           batch_count,
                                              bool              optimal_mem,
                                              void*             w_x_temp,
                                              void*             w_x_temparr,
                                              void*             invA,
                                              void*             invAarr,
                                              const T* const*   supplied_invA,
                                              int64_t           supplied_invA_size,
                                              rocblas_stride    offset_invA,
                                              rocblas_stride    stride_invA)
{
    if constexpr(std::is_same_v<T, float>)
        return rocblas_internal_trsm_launcher_64<ROCBLAS_TRSM_NB, ROCBLAS_SDCTRSV_NB, true>(
            TRSM_TEMPLATE_PARAMS);
    else if constexpr(std::is_same_v<T, double>)
        return rocblas_internal_trsm_launcher_64<ROCBLAS_TRSM_NB, ROCBLAS_SDCTRSV_NB, true>(
            TRSM_TEMPLATE_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_float_complex>)
        return rocblas_internal_trsm_launcher_64<ROCBLAS_TRSM_NB, ROCBLAS_SDCTRSV_NB, true>(
            TRSM_TEMPLATE_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_double_complex>)
        return rocblas_internal_trsm_launcher_64<ROCBLAS_TRSM_NB, ROCBLAS_ZTRSV_NB, true>(
            TRSM_TEMPLATE_PARAMS);
}

#undef TRSM_TEMPLATE_PARAMS

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *ger*.cpp

#ifdef INST_TRSM_TEMPLATE_64
#error INST_TRSM_TEMPLATE_64 already defined
#endif

#define INST_TRSM_TEMPLATE_64(API_INT_, T_)                                         \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                        \
        rocblas_internal_trsm_template_64<T_>(rocblas_handle    handle,             \
                                              rocblas_side      side,               \
                                              rocblas_fill      uplo,               \
                                              rocblas_operation transA,             \
                                              rocblas_diagonal  diag,               \
                                              API_INT_          m,                  \
                                              API_INT_          n,                  \
                                              const T_*         alpha,              \
                                              const T_*         A,                  \
                                              rocblas_stride    offsetA,            \
                                              API_INT_          lda,                \
                                              rocblas_stride    strideA,            \
                                              T_*               B,                  \
                                              rocblas_stride    offsetB,            \
                                              API_INT_          ldb,                \
                                              rocblas_stride    strideB,            \
                                              API_INT_          batch_count,        \
                                              bool              optimal_mem,        \
                                              void*             w_x_temp,           \
                                              void*             w_x_temp_arr,       \
                                              void*             invA,               \
                                              void*             invAarr,            \
                                              const T_*         supplied_invA,      \
                                              API_INT_          supplied_invA_size, \
                                              rocblas_stride    offset_invA,        \
                                              rocblas_stride    stride_invA);

#ifdef INST_TRSM_BATCHED_TEMPLATE_64
#error INST_TRSM_BATCHED_TEMPLATE_64 already defined
#endif

#define INST_TRSM_BATCHED_TEMPLATE_64(API_INT_, T_)                                         \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                                \
        rocblas_internal_trsm_batched_template_64<T_>(rocblas_handle    handle,             \
                                                      rocblas_side      side,               \
                                                      rocblas_fill      uplo,               \
                                                      rocblas_operation transA,             \
                                                      rocblas_diagonal  diag,               \
                                                      API_INT_          m,                  \
                                                      API_INT_          n,                  \
                                                      const T_*         alpha,              \
                                                      const T_* const*  A,                  \
                                                      rocblas_stride    offsetA,            \
                                                      API_INT_          lda,                \
                                                      rocblas_stride    strideA,            \
                                                      T_* const*        B,                  \
                                                      rocblas_stride    offsetB,            \
                                                      API_INT_          ldb,                \
                                                      rocblas_stride    strideB,            \
                                                      API_INT_          batch_count,        \
                                                      bool              optimal_mem,        \
                                                      void*             w_x_temp,           \
                                                      void*             w_x_temp_arr,       \
                                                      void*             invA,               \
                                                      void*             invAarr,            \
                                                      const T_* const*  supplied_invA,      \
                                                      API_INT_          supplied_invA_size, \
                                                      rocblas_stride    offset_invA,        \
                                                      rocblas_stride    stride_invA);

#ifdef INST_TRSM_WORK_SIZE_64
#error INST_TRSM_WORK_SIZE_64 already defined
#endif

#define INST_TRSM_WORK_SIZE_64(T_)                                                        \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                              \
        rocblas_internal_trsm_workspace_size_64<T_>(rocblas_side      side,               \
                                                    rocblas_operation transA,             \
                                                    int64_t           m,                  \
                                                    int64_t           n,                  \
                                                    int64_t           lda,                \
                                                    int64_t           ldb,                \
                                                    int64_t           batch_count,        \
                                                    int64_t           supplied_invA_size, \
                                                    size_t * w_x_tmp_size,                \
                                                    size_t * w_x_tmp_arr_size,            \
                                                    size_t * w_invA_size,                 \
                                                    size_t * w_invA_arr_size,             \
                                                    size_t * w_x_tmp_size_backup);

#ifdef INST_TRSM_BATCHED_WORK_SIZE_64
#error INST_TRSM_BATCHED_WORK_SIZE_64 already defined
#endif

#define INST_TRSM_BATCHED_WORK_SIZE_64(T_)                                                        \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                                      \
        rocblas_internal_trsm_batched_workspace_size_64<T_>(rocblas_side      side,               \
                                                            rocblas_operation transA,             \
                                                            int64_t           m,                  \
                                                            int64_t           n,                  \
                                                            int64_t           lda,                \
                                                            int64_t           ldb,                \
                                                            int64_t           batch_count,        \
                                                            int64_t           supplied_invA_size, \
                                                            size_t * w_x_tmp_size,                \
                                                            size_t * w_x_tmp_arr_size,            \
                                                            size_t * w_invA_size,                 \
                                                            size_t * w_invA_arr_size,             \
                                                            size_t * w_x_tmp_size_backup);

#ifdef INSTANTIATE_TRSM_MEM_TEMPLATE_64
#error INSTANTIATE_TRSM_MEM_TEMPLATE_64 already defined
#endif

#define INSTANTIATE_TRSM_MEM_TEMPLATE_64(BATCHED_, T_, U_)                           \
    template rocblas_status rocblas_internal_trsm_template_mem_64<BATCHED_, T_, U_>( \
        rocblas_handle    handle,                                                    \
        rocblas_side      side,                                                      \
        rocblas_operation transA,                                                    \
        int64_t           m,                                                         \
        int64_t           n,                                                         \
        int64_t           lda,                                                       \
        int64_t           ldb,                                                       \
        int64_t           batch_count,                                               \
        rocblas_device_malloc_base & w_mem,                                          \
        void*&  w_mem_x_temp,                                                        \
        void*&  w_mem_x_temp_arr,                                                    \
        void*&  w_mem_invA,                                                          \
        void*&  w_mem_invA_arr,                                                      \
        U_      supplied_invA,                                                       \
        int64_t supplied_invA_size);

#ifdef INSTANTIATE_TRSM_WORKSPACE_MAX_SIZE_TEMPLATE_64
#error INSTANTIATE_TRSM_WORKSPACE_MAX_SIZE_TEMPLATE_64 already defined
#endif

#define INSTANTIATE_TRSM_WORKSPACE_MAX_SIZE_TEMPLATE_64(T_)                       \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                      \
        rocblas_internal_trsm_workspace_max_size_64<T_>(rocblas_side side,        \
                                                        int64_t      m,           \
                                                        int64_t      n,           \
                                                        int64_t      batch_count, \
                                                        size_t * w_x_tmp_size,    \
                                                        size_t * w_invA_size,     \
                                                        size_t * w_x_tmp_size_backup);

#ifdef INSTANTIATE_TRSM_BATCHED_WORKSPACE_MAX_SIZE_TEMPLATE_64
#error INSTANTIATE_TRSM_BATCHED_WORKSPACE_MAX_SIZE_TEMPLATE_64 already defined
#endif

#define INSTANTIATE_TRSM_BATCHED_WORKSPACE_MAX_SIZE_TEMPLATE_64(T_)                        \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                               \
        rocblas_internal_trsm_batched_workspace_max_size_64<T_>(rocblas_side side,         \
                                                                int64_t      m,            \
                                                                int64_t      n,            \
                                                                int64_t      batch_count,  \
                                                                size_t * w_x_tmp_size,     \
                                                                size_t * w_x_tmp_arr_size, \
                                                                size_t * w_invA_size,      \
                                                                size_t * w_invA_arr_size,  \
                                                                size_t * w_x_tmp_size_backup);

INST_TRSM_TEMPLATE_64(int64_t, float)
INST_TRSM_TEMPLATE_64(int64_t, double)
INST_TRSM_TEMPLATE_64(int64_t, rocblas_float_complex)
INST_TRSM_TEMPLATE_64(int64_t, rocblas_double_complex)

INST_TRSM_BATCHED_TEMPLATE_64(int64_t, float)
INST_TRSM_BATCHED_TEMPLATE_64(int64_t, double)
INST_TRSM_BATCHED_TEMPLATE_64(int64_t, rocblas_float_complex)
INST_TRSM_BATCHED_TEMPLATE_64(int64_t, rocblas_double_complex)

INSTANTIATE_TRSM_MEM_TEMPLATE_64(false, float, const float*)
INSTANTIATE_TRSM_MEM_TEMPLATE_64(false, double, const double*)
INSTANTIATE_TRSM_MEM_TEMPLATE_64(false, rocblas_float_complex, const rocblas_float_complex*)
INSTANTIATE_TRSM_MEM_TEMPLATE_64(false, rocblas_double_complex, const rocblas_double_complex*)

INSTANTIATE_TRSM_MEM_TEMPLATE_64(true, float, const float* const*)
INSTANTIATE_TRSM_MEM_TEMPLATE_64(true, double, const double* const*)
INSTANTIATE_TRSM_MEM_TEMPLATE_64(true, rocblas_float_complex, const rocblas_float_complex* const*)
INSTANTIATE_TRSM_MEM_TEMPLATE_64(true, rocblas_double_complex, const rocblas_double_complex* const*)

INST_TRSM_WORK_SIZE_64(float)
INST_TRSM_WORK_SIZE_64(double)
INST_TRSM_WORK_SIZE_64(rocblas_float_complex)
INST_TRSM_WORK_SIZE_64(rocblas_double_complex)

INST_TRSM_BATCHED_WORK_SIZE_64(float)
INST_TRSM_BATCHED_WORK_SIZE_64(double)
INST_TRSM_BATCHED_WORK_SIZE_64(rocblas_float_complex)
INST_TRSM_BATCHED_WORK_SIZE_64(rocblas_double_complex)

INSTANTIATE_TRSM_WORKSPACE_MAX_SIZE_TEMPLATE_64(float)
INSTANTIATE_TRSM_WORKSPACE_MAX_SIZE_TEMPLATE_64(double)
INSTANTIATE_TRSM_WORKSPACE_MAX_SIZE_TEMPLATE_64(rocblas_float_complex)
INSTANTIATE_TRSM_WORKSPACE_MAX_SIZE_TEMPLATE_64(rocblas_double_complex)

INSTANTIATE_TRSM_BATCHED_WORKSPACE_MAX_SIZE_TEMPLATE_64(float)
INSTANTIATE_TRSM_BATCHED_WORKSPACE_MAX_SIZE_TEMPLATE_64(double)
INSTANTIATE_TRSM_BATCHED_WORKSPACE_MAX_SIZE_TEMPLATE_64(rocblas_float_complex)
INSTANTIATE_TRSM_BATCHED_WORKSPACE_MAX_SIZE_TEMPLATE_64(rocblas_double_complex)
