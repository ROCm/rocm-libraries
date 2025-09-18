/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_csrmv.hpp"
#include "rocsparse_utility.hpp"

#include "csrmv_device.h"

#define BLOCK_MULTIPLIER 3
#define WG_SIZE 256
#define LR_THRESHOLD 11
#define VEC_THRESHOLD 5
#define CSRMV_LRB_SHORT_ROWS_2_LDS_ELEMS 1024

template <typename I, typename J, typename A>
rocsparse_status rocsparse::csrmv_analysis_lrb_template_dispatch(rocsparse_handle          handle,
                                                                 rocsparse_operation       trans,
                                                                 J                         m,
                                                                 J                         n,
                                                                 I                         nnz,
                                                                 const rocsparse_mat_descr descr,
                                                                 const A*                  csr_val,
                                                                 const I*              csr_row_ptr,
                                                                 const J*              csr_col_ind,
                                                                 rocsparse_csrmv_info* p_csrmv_info)
{
    ROCSPARSE_ROUTINE_TRACE;

    p_csrmv_info[0]                 = new _rocsparse_csrmv_info();
    rocsparse_csrmv_info csrmv_info = p_csrmv_info[0];

    // Stream
    hipStream_t stream = handle->stream;

    RETURN_IF_HIP_ERROR(
        rocsparse_hipMallocAsync(&csrmv_info->lrb.rows_offsets_scratch, sizeof(J) * m, stream));
    RETURN_IF_HIP_ERROR(
        rocsparse_hipMallocAsync(&csrmv_info->lrb.rows_bins, sizeof(J) * m, stream));
    RETURN_IF_HIP_ERROR(
        rocsparse_hipMallocAsync(&csrmv_info->lrb.n_rows_bins, sizeof(J) * 32, stream));

    RETURN_IF_HIP_ERROR(
        hipMemsetAsync(csrmv_info->lrb.rows_offsets_scratch, 0, sizeof(J) * m, stream));
    RETURN_IF_HIP_ERROR(hipMemsetAsync(csrmv_info->lrb.rows_bins, 0, sizeof(J) * m, stream));
    RETURN_IF_HIP_ERROR(hipMemsetAsync(csrmv_info->lrb.n_rows_bins, 0, sizeof(J) * 32, stream));

    dim3 blocks(256);
    dim3 threads(WG_SIZE);
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
        (rocsparse::csrmvn_preprocess_device_32_bins_3phase_phase1<WG_SIZE>),
        blocks,
        threads,
        0,
        stream,
        m,
        csr_row_ptr,
        static_cast<J*>(csrmv_info->lrb.rows_offsets_scratch),
        static_cast<J*>(csrmv_info->lrb.n_rows_bins));

    // Copy bin sizes to CPU for later workgroup size determination.
    // If we modify the phase-2 and phase-3 preprocessing kernels so we don't directly reuse
    // n_rows_bins as both input and output, we can make the memcpy async and parallelize it
    // with the phase-2 and phase-3 kernels. Alternatively, we could always launch a fixed grid
    // size and then do more in the (SpMV) kernels to compute intra-kernel iteration bounds.
    J temp[32];
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        temp, csrmv_info->lrb.n_rows_bins, sizeof(J) * 32, hipMemcpyDeviceToHost, stream));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    for(int i = 0; i < 32; i++)
    {
        csrmv_info->lrb.nRowsBins[i] = temp[i];
    }

    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrmvn_preprocess_device_32_bins_3phase_phase2),
                                       1,
                                       1,
                                       0,
                                       stream,
                                       static_cast<J*>(csrmv_info->lrb.n_rows_bins));

    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
        (rocsparse::csrmvn_preprocess_device_32_bins_3phase_phase3<WG_SIZE>),
        blocks,
        threads,
        0,
        stream,
        m,
        csr_row_ptr,
        static_cast<J*>(csrmv_info->lrb.rows_offsets_scratch),
        static_cast<J*>(csrmv_info->lrb.n_rows_bins),
        static_cast<J*>(csrmv_info->lrb.rows_bins));

    // Optionally, sort bins (adds preprocessing cost, but often substantially reduces SpMV consumer-kernel time)
    /*if(true)
    {
        uint32_t startbit = 0;
        uint32_t endbit   = rocsparse::clz(m);

        bool   temp_alloc;
        void*  temp_storage_ptr   = nullptr;
        size_t temp_storage_bytes = 0;

        RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_keys(
            nullptr,
            temp_storage_bytes,
            static_cast<J*>(csrmv_info->lrb.rows_bins),
            static_cast<J*>(csrmv_info->lrb.rows_offsets_scratch),
            m,
            (32 - 1),
            static_cast<J*>(csrmv_info->lrb.n_rows_bins),
            static_cast<J*>(csrmv_info->lrb.n_rows_bins) + 1,
            startbit,
            endbit,
            stream));

        if(handle->buffer_size >= temp_storage_bytes)
        {
            temp_storage_ptr = handle->buffer;
            temp_alloc       = false;
        }
        else
        {
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMallocAsync(&temp_storage_ptr, temp_storage_bytes, stream));
            temp_alloc = true;
        }

        RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_keys(
            temp_storage_ptr,
            temp_storage_bytes,
            static_cast<J*>(csrmv_info->lrb.rows_bins),
            static_cast<J*>(csrmv_info->lrb.rows_offsets_scratch),
            m,
            (32 - 1),
            static_cast<J*>(csrmv_info->lrb.n_rows_bins),
            static_cast<J*>(csrmv_info->lrb.n_rows_bins) + 1,
            startbit,
            endbit,
            stream));

        if(temp_alloc)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(temp_storage_ptr, stream));
        }

        // Swap
        void* tmp                              = csrmv_info->lrb.rows_bins;
        csrmv_info->lrb.rows_bins            = csrmv_info->lrb.rows_offsets_scratch;
        csrmv_info->lrb.rows_offsets_scratch = tmp;
    }*/

    // Determine how many cross-workgroup global synchronization flags we'll need for Longrows
    // and allocate device storage accordingly (note that the sync approach is basically a
    // direct copy-paste from rocSPARSE CSR-Adaptive at present).
    // We should be able to reduce this to one flag per *row* (rather than one per *WG*)
    // if we do bit of Longrows refactoring.

    // Longrows synchronization flags: these will be allocated below, during the preprocessing, based
    // on Longrows bin sizes. We use a different array per kernel launch in order to permit use of >1 stream.
    uint32_t max_required_grid = 0;
    for(int j = LR_THRESHOLD; j < 32; j++)
    {
        uint32_t block_size      = WG_SIZE;
        uint32_t bin_max_row_len = (1 << j);
        uint32_t num_wgs_per_row = (bin_max_row_len - 1) / (BLOCK_MULTIPLIER * block_size) + 1;
        uint32_t grid_size       = csrmv_info->lrb.nRowsBins[j] * num_wgs_per_row;

        max_required_grid = rocsparse::max(grid_size, max_required_grid);
    }

    if(max_required_grid != 0)
    {
        csrmv_info->lrb.size = max_required_grid;

        RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(
            &csrmv_info->lrb.wg_flags, sizeof(uint32_t) * csrmv_info->lrb.size, stream));
    }

    // Store some pointers to verify correct execution
    csrmv_info->trans       = trans;
    csrmv_info->m           = m;
    csrmv_info->n           = n;
    csrmv_info->nnz         = nnz;
    csrmv_info->descr       = descr;
    csrmv_info->csr_row_ptr = csr_row_ptr;
    csrmv_info->csr_col_ind = csr_col_ind;

    csrmv_info->index_type_I = rocsparse::get_indextype<I>();
    csrmv_info->index_type_J = rocsparse::get_indextype<J>();

    return rocsparse_status_success;
}

namespace rocsparse
{
    template <typename I, typename J, typename A, typename X, typename Y, typename T>
    ROCSPARSE_KERNEL(WG_SIZE)
    void csrmvn_lrb_short_rows_kernel(bool conj,
                                      I    nnz,
                                      J* __restrict__ rows_bins,
                                      J* __restrict__ n_rows_bins,
                                      const uint32_t bin_id,
                                      ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                      const I* __restrict__ csr_row_ptr,
                                      const J* __restrict__ csr_col_ind,
                                      const A* __restrict__ csr_val,
                                      const X* __restrict__ x,
                                      ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                      Y* __restrict__ y,
                                      rocsparse_index_base idx_base,
                                      bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);
        if(alpha != 0 || beta != 1)
        {
            rocsparse::csrmvn_lrb_short_rows_device<WG_SIZE>(conj,
                                                             nnz,
                                                             rows_bins,
                                                             n_rows_bins,
                                                             bin_id,
                                                             alpha,
                                                             csr_row_ptr,
                                                             csr_col_ind,
                                                             csr_val,
                                                             x,
                                                             beta,
                                                             y,
                                                             idx_base);
        }
    }

    template <typename I, typename J, typename A, typename X, typename Y, typename T>
    ROCSPARSE_KERNEL(WG_SIZE)
    void csrmvn_lrb_short_rows_2_kernel(bool conj,
                                        I    nnz,
                                        J* __restrict__ rows_bins,
                                        J* __restrict__ n_rows_bins,
                                        const uint32_t bin_id,
                                        ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                        const I* __restrict__ csr_row_ptr,
                                        const J* __restrict__ csr_col_ind,
                                        const A* __restrict__ csr_val,
                                        const X* __restrict__ x,
                                        ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                        Y* __restrict__ y,
                                        rocsparse_index_base idx_base,
                                        bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);
        if(alpha != 0 || beta != 1)
        {
            rocsparse::csrmvn_lrb_short_rows_2_device<WG_SIZE, CSRMV_LRB_SHORT_ROWS_2_LDS_ELEMS>(
                conj,
                nnz,
                rows_bins,
                n_rows_bins,
                bin_id,
                alpha,
                csr_row_ptr,
                csr_col_ind,
                csr_val,
                x,
                beta,
                y,
                idx_base);
        }
    }

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y,
              typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrmvn_lrb_medium_rows_warp_reduce_kernel(bool    conj,
                                                   I       nnz,
                                                   int64_t count,
                                                   J* __restrict__ rows_bins,
                                                   J* __restrict__ n_rows_bins,
                                                   const uint32_t bin_id,
                                                   ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                                   const I* __restrict__ csr_row_ptr,
                                                   const J* __restrict__ csr_col_ind,
                                                   const A* __restrict__ csr_val,
                                                   const X* __restrict__ x,
                                                   ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                                   Y* __restrict__ y,
                                                   rocsparse_index_base idx_base,
                                                   bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);
        if(alpha != 0 || beta != 1)
        {
            rocsparse::csrmvn_lrb_medium_rows_warp_reduce_device<BLOCKSIZE, WF_SIZE>(conj,
                                                                                     nnz,
                                                                                     count,
                                                                                     rows_bins,
                                                                                     n_rows_bins,
                                                                                     bin_id,
                                                                                     alpha,
                                                                                     csr_row_ptr,
                                                                                     csr_col_ind,
                                                                                     csr_val,
                                                                                     x,
                                                                                     beta,
                                                                                     y,
                                                                                     idx_base);
        }
    }

    template <uint32_t BLOCKSIZE,
              typename I,
              typename J,
              typename A,
              typename X,
              typename Y,
              typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrmvn_lrb_medium_rows_kernel(bool conj,
                                       I    nnz,
                                       J* __restrict__ rows_bins,
                                       J* __restrict__ n_rows_bins,
                                       const uint32_t bin_id,
                                       ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                       const I* __restrict__ csr_row_ptr,
                                       const J* __restrict__ csr_col_ind,
                                       const A* __restrict__ csr_val,
                                       const X* __restrict__ x,
                                       ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                       Y* __restrict__ y,
                                       rocsparse_index_base idx_base,
                                       bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);
        if(alpha != 0 || beta != 1)
        {
            rocsparse::csrmvn_lrb_medium_rows_device<BLOCKSIZE>(conj,
                                                                nnz,
                                                                rows_bins,
                                                                n_rows_bins,
                                                                bin_id,
                                                                alpha,
                                                                csr_row_ptr,
                                                                csr_col_ind,
                                                                csr_val,
                                                                x,
                                                                beta,
                                                                y,
                                                                idx_base);
        }
    }

    template <typename I, typename J, typename A, typename X, typename Y, typename T>
    ROCSPARSE_KERNEL(WG_SIZE)
    void csrmvn_lrb_long_rows_kernel(bool conj,
                                     I    nnz,
                                     uint32_t* __restrict__ wg_flags,
                                     J* __restrict__ rows_bins,
                                     J* __restrict__ n_rows_bins,
                                     const uint32_t bin_id,
                                     ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                     const I* __restrict__ csr_row_ptr,
                                     const J* __restrict__ csr_col_ind,
                                     const A* __restrict__ csr_val,
                                     const X* __restrict__ x,
                                     ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                     Y* __restrict__ y,
                                     rocsparse_index_base idx_base,
                                     bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);
        if(alpha != 0 || beta != 1)
        {
            rocsparse::csrmvn_lrb_long_rows_device<WG_SIZE, BLOCK_MULTIPLIER>(conj,
                                                                              nnz,
                                                                              wg_flags,
                                                                              rows_bins,
                                                                              n_rows_bins,
                                                                              bin_id,
                                                                              alpha,
                                                                              csr_row_ptr,
                                                                              csr_col_ind,
                                                                              csr_val,
                                                                              x,
                                                                              beta,
                                                                              y,
                                                                              idx_base);
        }
    }
}

template <typename T, typename I, typename J, typename A, typename X, typename Y>
rocsparse_status rocsparse::csrmv_lrb_template_dispatch(rocsparse_handle          handle,
                                                        rocsparse_operation       trans,
                                                        J                         m,
                                                        J                         n,
                                                        I                         nnz,
                                                        const T*                  alpha_device_host,
                                                        const rocsparse_mat_descr descr,
                                                        const A*                  csr_val,
                                                        const I*                  csr_row_ptr,
                                                        const J*                  csr_col_ind,
                                                        rocsparse_csrmv_info      info,
                                                        const X*                  x,
                                                        const T*                  beta_device_host,
                                                        Y*                        y,
                                                        bool                      force_conj)
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(6, descr);
    ROCSPARSE_CHECKARG_POINTER(10, info);

    bool conj = (trans == rocsparse_operation_conjugate_transpose || force_conj);

    // Check if info matches current matrix and options
    ROCSPARSE_CHECKARG_ENUM(1, trans);

    ROCSPARSE_CHECKARG(10, info, (info->trans != trans), rocsparse_status_invalid_value);

    ROCSPARSE_CHECKARG(
        1, trans, (trans != rocsparse_operation_none), rocsparse_status_not_implemented);

    ROCSPARSE_CHECKARG(10,
                       info,
                       (info->m != m || info->n != n || info->nnz != nnz),
                       rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG(10, info, (info->descr != descr), rocsparse_status_invalid_value);
    ROCSPARSE_CHECKARG(10,
                       info,
                       (info->csr_row_ptr != csr_row_ptr || info->csr_col_ind != csr_col_ind),
                       rocsparse_status_invalid_pointer);

    // Stream
    hipStream_t stream = handle->stream;

    if(descr->type == rocsparse_matrix_type_general
       || descr->type == rocsparse_matrix_type_triangular)
    {
        // Short rows
        for(int j = 0; j < VEC_THRESHOLD; j++)
        {
            if(info->lrb.nRowsBins[j] != 0)
            {
                uint32_t block_size = WG_SIZE;
                uint32_t lds_size   = (block_size << j) * sizeof(T);

                // Dynamic LDS allocation
                if(lds_size < CSRMV_LRB_SHORT_ROWS_2_LDS_ELEMS * sizeof(T))
                {
                    uint32_t grid_size
                        = rocsparse::ceil((float)info->lrb.nRowsBins[j] / block_size);

                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (csrmvn_lrb_short_rows_kernel),
                        grid_size,
                        block_size,
                        lds_size,
                        stream,
                        conj,
                        nnz,
                        static_cast<J*>(info->lrb.rows_bins),
                        static_cast<J*>(info->lrb.n_rows_bins),
                        j,
                        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        x,
                        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
                        y,
                        descr->base,
                        handle->pointer_mode == rocsparse_pointer_mode_host);
                }
                // Static LDS allocation, for when dynamic would grow too large
                else
                {
                    uint32_t rows_per_wg = CSRMV_LRB_SHORT_ROWS_2_LDS_ELEMS >> j;
                    uint32_t grid_size
                        = rocsparse::ceil((float)info->lrb.nRowsBins[j] / rows_per_wg);

                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (csrmvn_lrb_short_rows_2_kernel),
                        grid_size,
                        block_size,
                        0,
                        stream,
                        conj,
                        nnz,
                        static_cast<J*>(info->lrb.rows_bins),
                        static_cast<J*>(info->lrb.n_rows_bins),
                        j,
                        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        x,
                        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),

                        y,
                        descr->base,
                        handle->pointer_mode == rocsparse_pointer_mode_host);
                }
            }
        }

        // Medium rows
        for(int j = VEC_THRESHOLD; j < LR_THRESHOLD; j++)
        {
            if(info->lrb.nRowsBins[j] != 0)
            {
                // Max WG size == 1024 on gfx90a.
                // min() permits using LRB-Vector for arbitrary bins (not just bins 0-10).
                uint32_t block_size = rocsparse::min(1 << j, 1024);

                if(block_size <= 256) // One warp per row
                {
                    uint32_t wf_size   = handle->wavefront_size;
                    uint32_t grid_size = (info->lrb.nRowsBins[j] - 1) / (256 / wf_size) + 1;

                    if(handle->wavefront_size == 32)
                    {
                        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                            (csrmvn_lrb_medium_rows_warp_reduce_kernel<256, 32>),
                            grid_size,
                            256,
                            0,
                            stream,
                            conj,
                            nnz,
                            info->lrb.nRowsBins[j],
                            static_cast<J*>(info->lrb.rows_bins),
                            static_cast<J*>(info->lrb.n_rows_bins),
                            j,
                            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                            csr_row_ptr,
                            csr_col_ind,
                            csr_val,
                            x,
                            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
                            y,
                            descr->base,
                            handle->pointer_mode == rocsparse_pointer_mode_host);
                    }
                    else
                    {
                        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                            (csrmvn_lrb_medium_rows_warp_reduce_kernel<256, 64>),
                            grid_size,
                            256,
                            0,
                            stream,
                            conj,
                            nnz,
                            info->lrb.nRowsBins[j],
                            static_cast<J*>(info->lrb.rows_bins),
                            static_cast<J*>(info->lrb.n_rows_bins),
                            j,
                            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                            csr_row_ptr,
                            csr_col_ind,
                            csr_val,
                            x,
                            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
                            y,
                            descr->base,
                            handle->pointer_mode == rocsparse_pointer_mode_host);
                    }
                }
                else // One block per row
                {
                    uint32_t grid_size = info->lrb.nRowsBins[j]; // One WG per row

                    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                        (csrmvn_lrb_medium_rows_kernel<WG_SIZE>),
                        grid_size,
                        WG_SIZE,
                        0,
                        stream,
                        conj,
                        nnz,
                        static_cast<J*>(info->lrb.rows_bins),
                        static_cast<J*>(info->lrb.n_rows_bins),
                        j,
                        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        x,
                        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
                        y,
                        descr->base,
                        handle->pointer_mode == rocsparse_pointer_mode_host);
                }
            }
        }

        // Long rows
        for(int j = LR_THRESHOLD; j < 32; j++)
        {
            if(info->lrb.nRowsBins[j] != 0)
            {
                RETURN_IF_HIP_ERROR(hipMemsetAsync(
                    info->lrb.wg_flags, 0, sizeof(uint32_t) * info->lrb.size, stream));

                uint32_t block_size      = WG_SIZE;
                uint32_t bin_max_row_len = (1 << j);
                uint32_t num_wgs_per_row
                    = (bin_max_row_len - 1) / (BLOCK_MULTIPLIER * block_size) + 1;
                uint32_t grid_size = info->lrb.nRowsBins[j] * num_wgs_per_row;

                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (csrmvn_lrb_long_rows_kernel),
                    grid_size,
                    block_size,
                    0,
                    stream,
                    conj,
                    nnz,
                    info->lrb.wg_flags,
                    static_cast<J*>(info->lrb.rows_bins),
                    static_cast<J*>(info->lrb.n_rows_bins),
                    j,
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                    csr_row_ptr,
                    csr_col_ind,
                    csr_val,
                    x,
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
                    y,
                    descr->base,
                    handle->pointer_mode == rocsparse_pointer_mode_host);
            }
        }
    }
    else
    {
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        // LCOV_EXCL_STOP
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, JTYPE, ATYPE)                                       \
    template rocsparse_status rocsparse::csrmv_analysis_lrb_template_dispatch( \
        rocsparse_handle          handle,                                      \
        rocsparse_operation       trans,                                       \
        JTYPE                     m,                                           \
        JTYPE                     n,                                           \
        ITYPE                     nnz,                                         \
        const rocsparse_mat_descr descr,                                       \
        const ATYPE*              csr_val,                                     \
        const ITYPE*              csr_row_ptr,                                 \
        const JTYPE*              csr_col_ind,                                 \
        rocsparse_csrmv_info*     p_csrmv_info);

// Uniform precision
INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);

// Mixed precisions
INSTANTIATE(int32_t, int32_t, int8_t);
INSTANTIATE(int64_t, int32_t, int8_t);
INSTANTIATE(int64_t, int64_t, int8_t);
INSTANTIATE(int32_t, int32_t, _Float16);
INSTANTIATE(int64_t, int32_t, _Float16);
INSTANTIATE(int64_t, int64_t, _Float16);
INSTANTIATE(int32_t, int32_t, rocsparse_bfloat16);
INSTANTIATE(int64_t, int32_t, rocsparse_bfloat16);
INSTANTIATE(int64_t, int64_t, rocsparse_bfloat16);

#undef INSTANTIATE

#define INSTANTIATE(TTYPE, ITYPE, JTYPE, ATYPE, XTYPE, YTYPE)         \
    template rocsparse_status rocsparse::csrmv_lrb_template_dispatch( \
        rocsparse_handle          handle,                             \
        rocsparse_operation       trans,                              \
        JTYPE                     m,                                  \
        JTYPE                     n,                                  \
        ITYPE                     nnz,                                \
        const TTYPE*              alpha_device_host,                  \
        const rocsparse_mat_descr descr,                              \
        const ATYPE*              csr_val,                            \
        const ITYPE*              csr_row_ptr,                        \
        const JTYPE*              csr_col_ind,                        \
        rocsparse_csrmv_info      info,                               \
        const XTYPE*              x,                                  \
        const TTYPE*              beta_device_host,                   \
        YTYPE*                    y,                                  \
        bool                      force_conj);

// Uniform precision
INSTANTIATE(float, int32_t, int32_t, float, float, float);
INSTANTIATE(float, int64_t, int32_t, float, float, float);
INSTANTIATE(float, int64_t, int64_t, float, float, float);
INSTANTIATE(double, int32_t, int32_t, double, double, double);
INSTANTIATE(double, int64_t, int32_t, double, double, double);
INSTANTIATE(double, int64_t, int64_t, double, double, double);
INSTANTIATE(rocsparse_float_complex,
            int32_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int64_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int64_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);

// Mixed percision
INSTANTIATE(int32_t, int32_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE(int32_t, int64_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE(int32_t, int64_t, int64_t, int8_t, int8_t, int32_t);
INSTANTIATE(float, int32_t, int32_t, int8_t, int8_t, float);
INSTANTIATE(float, int64_t, int32_t, int8_t, int8_t, float);
INSTANTIATE(float, int64_t, int64_t, int8_t, int8_t, float);
INSTANTIATE(float, int32_t, int32_t, _Float16, _Float16, float);
INSTANTIATE(float, int64_t, int32_t, _Float16, _Float16, float);
INSTANTIATE(float, int64_t, int64_t, _Float16, _Float16, float);
INSTANTIATE(float, int32_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(float, int64_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(float, int64_t, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(rocsparse_float_complex,
            int32_t,
            int32_t,
            float,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int32_t,
            float,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int64_t,
            float,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(double, int32_t, int32_t, float, double, double);
INSTANTIATE(double, int64_t, int32_t, float, double, double);
INSTANTIATE(double, int64_t, int64_t, float, double, double);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            int32_t,
            double,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int32_t,
            double,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int64_t,
            double,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int64_t,
            rocsparse_float_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);

#undef INSTANTIATE
