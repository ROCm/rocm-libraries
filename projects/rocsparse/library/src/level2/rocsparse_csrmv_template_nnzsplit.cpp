/*! \file */
/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_common.h"
#include "rocsparse_control.hpp"
#include "rocsparse_csrmv.hpp"
#include "rocsparse_envariables.hpp"
#include "rocsparse_utility.hpp"

#include "internal/generic/rocsparse_v2_spmv.h"
#include "rocsparse_spmv_helpers.h"

#include "csrmv_device_nnzsplit.h"
#include "rocsparse_primitives.hpp"

#include <vector>

#define LAUNCH_CSRMV_ANALYSIS(BLOCKSIZE, NNZ_PER_THREAD) \
    csrmv_analysis_nnzsplit<BLOCKSIZE, NNZ_PER_THREAD>(  \
        handle, trans, m, n, nnz, descr, csr_val, csr_row_ptr, csr_col_ind, csrmv_info);

#define LAUNCH_CSRMV(BLOCKSIZE, NNZ_PER_THREAD)                               \
    uint32_t NNZ_PER_BLOCK  = NNZ_PER_THREAD * BLOCKSIZE;                     \
    uint32_t requiredBlocks = (nnz + NNZ_PER_BLOCK - 1) / NNZ_PER_BLOCK;      \
                                                                              \
    dim3 csrmv_threads(BLOCKSIZE);                                            \
    dim3 csrmv_phase_2(requiredBlocks);                                       \
                                                                              \
    if(handle->wavefront_size == 64)                                          \
    {                                                                         \
        if(csrmv_info->nnzsplit.use_starting_block_ids)                       \
        {                                                                     \
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                               \
                (csrmv_nnzsplit<BLOCKSIZE, NNZ_PER_THREAD, 64, true>),        \
                csrmv_phase_2,                                                \
                csrmv_threads,                                                \
                0,                                                            \
                handle->stream,                                               \
                conj,                                                         \
                nnz,                                                          \
                m,                                                            \
                n,                                                            \
                ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host), \
                csr_row_ptr,                                                  \
                (J*)csrmv_info->nnzsplit.starting_ids,                        \
                csr_col_ind,                                                  \
                csr_val,                                                      \
                x,                                                            \
                y,                                                            \
                (J*)csrmv_info->nnzsplit.starting_block_ids,                  \
                descr->base,                                                  \
                handle->pointer_mode == rocsparse_pointer_mode_host);         \
        }                                                                     \
        else                                                                  \
        {                                                                     \
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                               \
                (csrmv_nnzsplit<BLOCKSIZE, NNZ_PER_THREAD, 64, false>),       \
                csrmv_phase_2,                                                \
                csrmv_threads,                                                \
                0,                                                            \
                handle->stream,                                               \
                conj,                                                         \
                nnz,                                                          \
                m,                                                            \
                n,                                                            \
                ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host), \
                csr_row_ptr,                                                  \
                (J*)csrmv_info->nnzsplit.starting_ids,                        \
                csr_col_ind,                                                  \
                csr_val,                                                      \
                x,                                                            \
                y,                                                            \
                (J*)csrmv_info->nnzsplit.starting_block_ids,                  \
                descr->base,                                                  \
                handle->pointer_mode == rocsparse_pointer_mode_host);         \
        }                                                                     \
    }                                                                         \
    else                                                                      \
    {                                                                         \
        if(csrmv_info->nnzsplit.use_starting_block_ids)                       \
        {                                                                     \
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                               \
                (csrmv_nnzsplit<BLOCKSIZE, NNZ_PER_THREAD, 32, true>),        \
                csrmv_phase_2,                                                \
                csrmv_threads,                                                \
                0,                                                            \
                handle->stream,                                               \
                conj,                                                         \
                nnz,                                                          \
                m,                                                            \
                n,                                                            \
                ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host), \
                csr_row_ptr,                                                  \
                (J*)csrmv_info->nnzsplit.starting_ids,                        \
                csr_col_ind,                                                  \
                csr_val,                                                      \
                x,                                                            \
                y,                                                            \
                (J*)csrmv_info->nnzsplit.starting_block_ids,                  \
                descr->base,                                                  \
                handle->pointer_mode == rocsparse_pointer_mode_host);         \
        }                                                                     \
        else                                                                  \
        {                                                                     \
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                               \
                (csrmv_nnzsplit<BLOCKSIZE, NNZ_PER_THREAD, 32, false>),       \
                csrmv_phase_2,                                                \
                csrmv_threads,                                                \
                0,                                                            \
                handle->stream,                                               \
                conj,                                                         \
                nnz,                                                          \
                m,                                                            \
                n,                                                            \
                ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host), \
                csr_row_ptr,                                                  \
                (J*)csrmv_info->nnzsplit.starting_ids,                        \
                csr_col_ind,                                                  \
                csr_val,                                                      \
                x,                                                            \
                y,                                                            \
                (J*)csrmv_info->nnzsplit.starting_block_ids,                  \
                descr->base,                                                  \
                handle->pointer_mode == rocsparse_pointer_mode_host);         \
        }                                                                     \
    }

#define LAUNCH_CSRMVT(BLOCKSIZE, NNZ_PER_THREAD)                         \
    uint32_t NNZ_PER_BLOCK  = NNZ_PER_THREAD * BLOCKSIZE;                \
    uint32_t requiredBlocks = (nnz + NNZ_PER_BLOCK - 1) / NNZ_PER_BLOCK; \
                                                                         \
    dim3 csrmvt_threads(BLOCKSIZE);                                      \
    dim3 csrmvt_phase_2(requiredBlocks);                                 \
                                                                         \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                  \
        (csrmvt_nnzsplit<BLOCKSIZE, NNZ_PER_THREAD>),                    \
        dim3(csrmvt_phase_2),                                            \
        dim3(csrmvt_threads),                                            \
        0,                                                               \
        handle->stream,                                                  \
        skip_diag,                                                       \
        conj,                                                            \
        nnz,                                                             \
        m,                                                               \
        n,                                                               \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),    \
        csr_row_ptr,                                                     \
        (J*)csrmv_info->nnzsplit.starting_ids,                           \
        csr_col_ind,                                                     \
        csr_val,                                                         \
        x,                                                               \
        y,                                                               \
        descr->base,                                                     \
        handle->pointer_mode == rocsparse_pointer_mode_host);

// nnz-per-thread granularity tiers.
#define CSRMV_NNZ_PER_THREAD_0 1
#define CSRMV_NNZ_PER_THREAD_1 4
#define CSRMV_NNZ_PER_THREAD_2 8

// Average nnz/row density knees selecting the granularity. Sparse rows are
// latency-bound and prefer high occupancy (NNZ_PER_THREAD=1); denser rows
// amortize per-block overhead with more work per thread.
#define CSRMV_NNZSPLIT_LOW_DENSITY 32
#define CSRMV_NNZSPLIT_HIGH_DENSITY 96

// Number of wavefronts per block. The block size is wavefront-relative
// (4 wavefronts = 128 threads on wave32, 256 on wave64): a small block keeps
// each block spanning fewer rows, which shortens the per-element dichotomic row
// search and cuts atomic traffic at row/block boundaries, while wide-wavefront
// parts keep the historical 256-thread block instead of a half-occupancy 128.
#define CSRMV_NNZSPLIT_BLOCK_WAVES 4

// Row-length skew guard: if the longest row exceeds this multiple of the mean
// row length, avoid the finest granularity. On strongly skewed matrices the
// 1 nnz/thread path spreads a single long row across many small blocks, which
// serialises the row and generates heavy atomic contention at its boundaries;
// the coarse granularity keeps such rows within far fewer blocks.
#define CSRMV_NNZSPLIT_SKEW_FACTOR 64

namespace rocsparse
{
    // nnzsplit block size, wavefront-relative (128 on wave32, 256 on wave64).
    static inline uint32_t nnzsplit_block_size(rocsparse_handle handle)
    {
        return CSRMV_NNZSPLIT_BLOCK_WAVES * static_cast<uint32_t>(handle->wavefront_size);
    }

    // Number of thread-blocks the device can hold concurrently for a given block
    // size, from real occupancy (maxThreadsPerMultiProcessor) rather than a fixed
    // blocks-per-CU magic number.
    static inline uint64_t nnzsplit_saturation_blocks(const hipDeviceProp_t& prop, uint32_t block)
    {
        uint32_t per_cu = 1u;
        if(block > 0 && prop.maxThreadsPerMultiProcessor > 0)
        {
            per_cu = static_cast<uint32_t>(prop.maxThreadsPerMultiProcessor) / block;
            if(per_cu < 1u)
            {
                per_cu = 1u;
            }
        }
        return static_cast<uint64_t>(prop.multiProcessorCount) * static_cast<uint64_t>(per_cu);
    }

    // Choose the nnz-per-thread granularity. Precedence:
    //   (1) skew guard  -> coarse  (long-tailed rows: cut atomic/dichotomy traffic)
    //   (2) low volume  -> fine    (not enough work to fill the device: occupancy)
    //   (3) density buckets (sparse -> fine, medium -> med, dense -> coarse)
    static inline uint32_t nnzsplit_nnz_per_thread(
        const hipDeviceProp_t& prop, uint32_t block, int64_t m, int64_t nnz, int64_t max_row_nnz)
    {
        const double avg = (m > 0) ? static_cast<double>(nnz) / static_cast<double>(m)
                                   : static_cast<double>(nnz);

        // (1) skew guard
        if(m > 0 && max_row_nnz > 0)
        {
            const double mean = (avg > 1.0) ? avg : 1.0;
            if(static_cast<double>(max_row_nnz) > CSRMV_NNZSPLIT_SKEW_FACTOR * mean)
            {
                return CSRMV_NNZ_PER_THREAD_2;
            }
        }

        // (2) too little work to saturate the device -> finest granularity
        const uint64_t nnz_per_block_med
            = static_cast<uint64_t>(block) * static_cast<uint64_t>(CSRMV_NNZ_PER_THREAD_1);
        const uint64_t blocks_med
            = (static_cast<uint64_t>(nnz) + nnz_per_block_med - 1) / nnz_per_block_med;
        if(blocks_med < nnzsplit_saturation_blocks(prop, block))
        {
            return CSRMV_NNZ_PER_THREAD_0;
        }

        // (3) density buckets
        if(avg < static_cast<double>(CSRMV_NNZSPLIT_LOW_DENSITY))
        {
            return CSRMV_NNZ_PER_THREAD_0;
        }
        if(avg < static_cast<double>(CSRMV_NNZSPLIT_HIGH_DENSITY))
        {
            return CSRMV_NNZ_PER_THREAD_1;
        }
        return CSRMV_NNZ_PER_THREAD_2;
    }

    // Longest row length (max over row_ptr differences) - the row-skew signal.
    // Computed once at analysis time (amortised over many SpMV calls) with a
    // single host copy of row_ptr; the compute phase reads it from the info
    // struct and never touches row_ptr for tuning.
    template <typename I, typename J>
    static rocsparse_status csrmv_nnzsplit_max_row_nnz(rocsparse_handle handle,
                                                       J                m,
                                                       const I*         csr_row_ptr,
                                                       int64_t*         max_row_nnz)
    {
        *max_row_nnz = 0;
        if(m <= 0 || csr_row_ptr == nullptr)
        {
            return rocsparse_status_success;
        }

        hipStream_t    stream = handle->stream;
        std::vector<I> hptr(static_cast<size_t>(m) + 1);
        RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(hptr.data(),
                                                     csr_row_ptr,
                                                     sizeof(I) * (static_cast<size_t>(m) + 1),
                                                     hipMemcpyDeviceToHost,
                                                     stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(stream));

        int64_t mx = 0;
        for(J i = 0; i < m; ++i)
        {
            const int64_t len = static_cast<int64_t>(hptr[i + 1] - hptr[i]);
            if(len > mx)
            {
                mx = len;
            }
        }
        *max_row_nnz = mx;
        return rocsparse_status_success;
    }
}

// Dispatch to the compile-time-templated launch macro for the (block, npt) pair
// selected above. The block size is wavefront-relative (128 on wave32, 256 on
// wave64) and npt is the nnz-per-thread granularity (1 / 4 / 8). Analysis records
// the pair in the info struct and the compute phase replays it verbatim, so the
// two phases never disagree on the block layout of starting_ids/starting_block_ids.
#define DISPATCH_BLOCK_NPT(macro_to_launch, BLK, NPT) \
    if((BLK) == 128u)                                 \
    {                                                 \
        if((NPT) == 1u)                               \
        {                                             \
            macro_to_launch(128, 1);                  \
        }                                             \
        else if((NPT) == 4u)                          \
        {                                             \
            macro_to_launch(128, 4);                  \
        }                                             \
        else                                          \
        {                                             \
            macro_to_launch(128, 8);                  \
        }                                             \
    }                                                 \
    else                                              \
    {                                                 \
        if((NPT) == 1u)                               \
        {                                             \
            macro_to_launch(256, 1);                  \
        }                                             \
        else if((NPT) == 4u)                          \
        {                                             \
            macro_to_launch(256, 4);                  \
        }                                             \
        else                                          \
        {                                             \
            macro_to_launch(256, 8);                  \
        }                                             \
    }

template <uint32_t BLOCKSIZE, uint32_t NNZ_PER_THREAD, typename I, typename J, typename A>
rocsparse_status csrmv_analysis_nnzsplit(rocsparse_handle          handle,
                                         rocsparse_operation       trans,
                                         J                         m,
                                         J                         n,
                                         I                         nnz,
                                         const rocsparse_mat_descr descr,
                                         const A*                  csr_val,
                                         const I*                  csr_row_ptr,
                                         const J*                  csr_col_ind,
                                         rocsparse_csrmv_info      csrmv_info)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Stream
    hipStream_t stream = handle->stream;

    const bool use_starting_block_ids
        = csrmv_info == nullptr ? false : csrmv_info->nnzsplit.use_starting_block_ids;

    uint32_t NNZ_PER_BLOCK  = NNZ_PER_THREAD * BLOCKSIZE;
    uint32_t requiredBlocks = (nnz + NNZ_PER_BLOCK - 1) / NNZ_PER_BLOCK;

    csrmv_info->nnzsplit.size = requiredBlocks;

    csrmv_info->nnzsplit.use_starting_block_ids = use_starting_block_ids;

    dim3 csrmv_threads(BLOCKSIZE);
    dim3 csrmv_phase_1((rocsparse::max(m + 1, n) + BLOCKSIZE - 1) / BLOCKSIZE);

    if(!csrmv_info->nnzsplit.use_starting_block_ids)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(
            &csrmv_info->nnzsplit.starting_ids, sizeof(J) * (requiredBlocks + 1), stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(
            csrmv_info->nnzsplit.starting_ids, 0, sizeof(J) * (requiredBlocks + 1), stream));
    }
    else
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(
            &csrmv_info->nnzsplit.starting_ids, sizeof(J) * (2 * requiredBlocks + 2), stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(
            csrmv_info->nnzsplit.starting_ids, 0, sizeof(J) * (2 * requiredBlocks + 2), stream));
    }
    J* temp_buffer_j = reinterpret_cast<J*>(csrmv_info->nnzsplit.starting_ids);
    csrmv_info->nnzsplit.starting_block_ids = &temp_buffer_j[requiredBlocks + 1];

    if(csrmv_info->nnzsplit.use_starting_block_ids)
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrmv_determine_block_starts<BLOCKSIZE, NNZ_PER_THREAD, true, I, J>),
            csrmv_phase_1,
            csrmv_threads,
            0,
            stream,
            m,
            csr_row_ptr,
            temp_buffer_j,
            (J*)csrmv_info->nnzsplit.starting_block_ids,
            descr->base);
    else
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csrmv_determine_block_starts<BLOCKSIZE, NNZ_PER_THREAD, false, I, J>),
            csrmv_phase_1,
            csrmv_threads,
            0,
            stream,
            m,
            csr_row_ptr,
            temp_buffer_j,
            (J*)csrmv_info->nnzsplit.starting_block_ids,
            descr->base);

    if(csrmv_info->nnzsplit.use_starting_block_ids)
    {
        size_t buffer_size;
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::exclusive_scan_buffer_size<J, J>(
            handle, static_cast<J>(0), requiredBlocks + 1, &buffer_size)));
        bool  temp_alloc       = false;
        void* temp_storage_ptr = nullptr;
        if(handle->buffer_size >= buffer_size)
        {
            temp_storage_ptr = handle->buffer;
            temp_alloc       = false;
        }
        else
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(&temp_storage_ptr, buffer_size, stream));
            temp_alloc = true;
        }
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::primitives::exclusive_scan(handle,
                                                  (J*)csrmv_info->nnzsplit.starting_block_ids,
                                                  (J*)csrmv_info->nnzsplit.starting_block_ids,
                                                  static_cast<J>(0),
                                                  requiredBlocks + 1,
                                                  buffer_size,
                                                  temp_storage_ptr));
        if(temp_alloc)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(temp_storage_ptr, stream));
        }
    }

    return rocsparse_status_success;
}

template <typename I, typename J, typename A>
rocsparse_status
    rocsparse::csrmv_analysis_nnzsplit_template_dispatch(rocsparse_handle          handle,
                                                         rocsparse_operation       trans,
                                                         J                         m,
                                                         J                         n,
                                                         I                         nnz,
                                                         const rocsparse_mat_descr descr,
                                                         const A*                  csr_val,
                                                         const I*                  csr_row_ptr,
                                                         const J*                  csr_col_ind,
                                                         rocsparse_csrmv_info*     p_csrmv_info)
{
    ROCSPARSE_ROUTINE_TRACE;

    p_csrmv_info[0]                 = new _rocsparse_csrmv_info();
    rocsparse_csrmv_info csrmv_info = p_csrmv_info[0];

    // Choose the launch tuning once, here, and record it so the compute phase
    // replays the exact same (block, npt) - the block layout of the arrays
    // allocated below depends on both. Block size is wavefront-relative; the
    // nnz-per-thread granularity folds in the row-skew guard and real occupancy.
    const uint32_t block       = rocsparse::nnzsplit_block_size(handle);
    int64_t        max_row_nnz = 0;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::csrmv_nnzsplit_max_row_nnz(handle, m, csr_row_ptr, &max_row_nnz));
    const uint32_t npt = rocsparse::nnzsplit_nnz_per_thread(
        handle->properties, block, static_cast<int64_t>(m), static_cast<int64_t>(nnz), max_row_nnz);

    csrmv_info->nnzsplit.block_size     = block;
    csrmv_info->nnzsplit.nnz_per_thread = npt;
    csrmv_info->nnzsplit.max_row_nnz    = max_row_nnz;

    DISPATCH_BLOCK_NPT(LAUNCH_CSRMV_ANALYSIS, block, npt)

    // Store some pointers to verify correct execution
    csrmv_info->trans = trans;
    csrmv_info->m     = m;
    csrmv_info->n     = n;
    csrmv_info->nnz   = nnz;

    return rocsparse_status_success;
}

template <uint32_t BLOCKSIZE,
          uint32_t NNZ_PER_THREAD,
          uint32_t WFSIZE,
          bool     USE_STARTING_BLOCK_IDS,
          typename I,
          typename J,
          typename A,
          typename X,
          typename Y,
          typename T>
ROCSPARSE_KERNEL(BLOCKSIZE)
void csrmv_nnzsplit(bool conj,
                    I    nnz,
                    J    m,
                    J    n,
                    ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                    const I* csr_row_ptr_begin,
                    const J* __restrict__ startingIds,
                    const J* __restrict__ csr_col_ind,
                    const A* __restrict__ csr_val,
                    const X* __restrict__ x,
                    Y* __restrict__ y,
                    const J* __restrict__ starting_block_ids,
                    rocsparse_index_base idx_base,
                    bool                 is_host_mode)
{
    ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
    if(alpha != 0)
    {
        rocsparse::csrmv_nnzsplit_device<BLOCKSIZE, NNZ_PER_THREAD, WFSIZE, USE_STARTING_BLOCK_IDS>(
            conj,
            nnz,
            m,
            n,
            alpha,
            csr_row_ptr_begin,
            startingIds,
            csr_col_ind,
            csr_val,
            x,
            y,
            starting_block_ids,
            idx_base);
    }
}

template <uint32_t BLOCKSIZE,
          uint32_t NNZ_PER_THREAD,
          typename I,
          typename J,
          typename A,
          typename X,
          typename Y,
          typename T>
ROCSPARSE_KERNEL(BLOCKSIZE)
void csrmvt_nnzsplit(bool skip_diag,
                     bool conj,
                     I    nnz,
                     J    m,
                     J    n,
                     ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                     const I* csr_row_ptr_begin,
                     const J* __restrict__ startingIds,
                     const J* __restrict__ csr_col_ind,
                     const A* __restrict__ csr_val,
                     const X* __restrict__ x,
                     Y* __restrict__ y,
                     rocsparse_index_base idx_base,
                     bool                 is_host_mode)
{
    ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
    if(alpha != 0)
    {
        rocsparse::csrmvt_nnzsplit_device<BLOCKSIZE, NNZ_PER_THREAD>(skip_diag,
                                                                     conj,
                                                                     nnz,
                                                                     m,
                                                                     n,
                                                                     alpha,
                                                                     csr_row_ptr_begin,
                                                                     startingIds,
                                                                     csr_col_ind,
                                                                     csr_val,
                                                                     x,
                                                                     y,
                                                                     idx_base);
    }
}

template <typename T, typename I, typename J, typename A, typename X, typename Y>
rocsparse_status rocsparse::csrmv_nnzsplit_template_dispatch(rocsparse_handle    handle,
                                                             rocsparse_operation trans,
                                                             J                   m,
                                                             J                   n,
                                                             I                   nnz,
                                                             const T*            alpha_device_host,
                                                             const rocsparse_mat_descr descr,
                                                             const A*                  csr_val,
                                                             const I*                  csr_row_ptr,
                                                             const J*                  csr_col_ind,
                                                             rocsparse_csrmv_info      csrmv_info,
                                                             const X*                  x,
                                                             const T* beta_device_host,
                                                             Y*       y,
                                                             bool     force_conj)
{
    return rocsparse::csrmv_nnzsplit_template_dispatch(handle,
                                                       trans,
                                                       m,
                                                       n,
                                                       nnz,
                                                       alpha_device_host,
                                                       descr,
                                                       csr_val,
                                                       csr_row_ptr,
                                                       csr_col_ind,
                                                       csrmv_info,
                                                       x,
                                                       beta_device_host,
                                                       y,
                                                       0,
                                                       nullptr,
                                                       nullptr,
                                                       force_conj);
}

template <typename T, typename I, typename J, typename A, typename X, typename Y>
rocsparse_status rocsparse::csrmv_nnzsplit_template_dispatch(rocsparse_handle    handle,
                                                             rocsparse_operation trans,
                                                             J                   m,
                                                             J                   n,
                                                             I                   nnz,
                                                             const T*            alpha_device_host,
                                                             const rocsparse_mat_descr descr,
                                                             const A*                  csr_val,
                                                             const I*                  csr_row_ptr,
                                                             const J*                  csr_col_ind,
                                                             rocsparse_csrmv_info      csrmv_info,
                                                             const X*                  x,
                                                             const T*      beta_device_host,
                                                             Y*            y,
                                                             rocsparse_int num_extra,
                                                             rocsparse_const_dnvec_descr  gamma_vec,
                                                             rocsparse_const_dnvec_descr* z_vecs,
                                                             bool force_conj)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Extract gamma arrays and z vectors for batched operation
    using Z                      = Y;
    T*        gamma_device_array = nullptr;
    const Z** z_array            = nullptr;

    // Check if pre-extracted arrays are available in spmv descriptor
    if(num_extra > 0)
    {
        if(handle && handle->temp_spmv_descr && spmv_has_device_arrays(handle->temp_spmv_descr))
        {
            gamma_device_array = rocsparse::spmv_get_gamma_device_array<T>(handle->temp_spmv_descr);
            z_array            = rocsparse::spmv_get_z_array<Z>(handle->temp_spmv_descr);
        }
        else
        {
            // throw an error here as the extra data cannot be retrieved
            // LCOV_EXCL_START
            return rocsparse_status_invalid_value;
            // LCOV_EXCL_STOP
        }
    }

    const J ysize = (trans == rocsparse_operation_none) ? m : n;

    const bool skip_diag = (descr->type == rocsparse_matrix_type_symmetric);
    const bool conj      = (trans == rocsparse_operation_conjugate_transpose || force_conj);

    // Replay the (block, npt) tuning chosen during analysis. If analysis did not
    // populate it (defensive; the nnzsplit path always analyses first), recompute
    // from the same rules using the stored skew signal.
    uint32_t block = csrmv_info->nnzsplit.block_size;
    uint32_t npt   = csrmv_info->nnzsplit.nnz_per_thread;
    if(block == 0u)
    {
        block = rocsparse::nnzsplit_block_size(handle);
        npt   = rocsparse::nnzsplit_nnz_per_thread(handle->properties,
                                                 block,
                                                 static_cast<int64_t>(m),
                                                 static_cast<int64_t>(nnz),
                                                 csrmv_info->nnzsplit.max_row_nnz);
    }

    if(trans == rocsparse_operation_none || descr->type == rocsparse_matrix_type_symmetric)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::axpby_array_batched(
            handle, ysize, num_extra, gamma_device_array, z_array, beta_device_host, y));
        DISPATCH_BLOCK_NPT(LAUNCH_CSRMV, block, npt)
    }

    if(trans != rocsparse_operation_none || descr->type == rocsparse_matrix_type_symmetric)
    {
        if(descr->type != rocsparse_matrix_type_symmetric)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::axpby_array_batched(
                handle, ysize, num_extra, gamma_device_array, z_array, beta_device_host, y));
        }

        DISPATCH_BLOCK_NPT(LAUNCH_CSRMVT, block, npt)
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, JTYPE, ATYPE)                                            \
    template rocsparse_status rocsparse::csrmv_analysis_nnzsplit_template_dispatch( \
        rocsparse_handle          handle,                                           \
        rocsparse_operation       trans,                                            \
        JTYPE                     m,                                                \
        JTYPE                     n,                                                \
        ITYPE                     nnz,                                              \
        const rocsparse_mat_descr descr,                                            \
        const ATYPE*              csr_val,                                          \
        const ITYPE*              csr_row_ptr,                                      \
        const JTYPE*              csr_col_ind,                                      \
        rocsparse_csrmv_info*     p_csrmv_info);

// Uniform precision
INSTANTIATE(int32_t, int32_t, _Float16);
INSTANTIATE(int64_t, int32_t, _Float16);
INSTANTIATE(int64_t, int64_t, _Float16);
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
INSTANTIATE(int32_t, int32_t, rocsparse_bfloat16);
INSTANTIATE(int64_t, int32_t, rocsparse_bfloat16);
INSTANTIATE(int64_t, int64_t, rocsparse_bfloat16);

// Mixed precisions
INSTANTIATE(int32_t, int32_t, int8_t);
INSTANTIATE(int64_t, int32_t, int8_t);
INSTANTIATE(int64_t, int64_t, int8_t);

#undef INSTANTIATE

#define INSTANTIATE(TTYPE, ITYPE, JTYPE, ATYPE, XTYPE, YTYPE)                     \
    template rocsparse_status rocsparse::csrmv_nnzsplit_template_dispatch<TTYPE>( \
        rocsparse_handle          handle,                                         \
        rocsparse_operation       trans,                                          \
        JTYPE                     m,                                              \
        JTYPE                     n,                                              \
        ITYPE                     nnz,                                            \
        const TTYPE*              alpha_device_host,                              \
        const rocsparse_mat_descr descr,                                          \
        const ATYPE*              csr_val,                                        \
        const ITYPE*              csr_row_ptr,                                    \
        const JTYPE*              csr_col_ind,                                    \
        rocsparse_csrmv_info      csrmv_info,                                     \
        const XTYPE*              x,                                              \
        const TTYPE*              beta_device_host,                               \
        YTYPE*                    y,                                              \
        bool                      force_conj);

// Uniform precision
INSTANTIATE(float, int32_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(float, int64_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(float, int64_t, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(float, int32_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16);
INSTANTIATE(float, int64_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16);
INSTANTIATE(float, int64_t, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16);
INSTANTIATE(float, int32_t, int32_t, _Float16, _Float16, float);
INSTANTIATE(float, int64_t, int32_t, _Float16, _Float16, float);
INSTANTIATE(float, int64_t, int64_t, _Float16, _Float16, float);
INSTANTIATE(float, int32_t, int32_t, _Float16, _Float16, _Float16);
INSTANTIATE(float, int64_t, int32_t, _Float16, _Float16, _Float16);
INSTANTIATE(float, int64_t, int64_t, _Float16, _Float16, _Float16);
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
