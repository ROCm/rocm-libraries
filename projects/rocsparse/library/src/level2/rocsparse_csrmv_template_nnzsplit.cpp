/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "csrmv_device_nnzsplit.h"
#include "rocsparse_primitives.hpp"

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

#define LAUNCH_HELPER(macro_to_launch)                                                       \
    const int nprocs = 2 * handle->properties.multiProcessorCount;                           \
    if(nnz / (nprocs * CSRMV_BLOCKSIZE * CSRMV_NNZ_PER_THREAD_1) < CSRMV_BLOCKS_PER_CU)      \
    {                                                                                        \
        macro_to_launch(CSRMV_BLOCKSIZE, CSRMV_NNZ_PER_THREAD_0);                            \
    }                                                                                        \
    else if(nnz / (nprocs * CSRMV_BLOCKSIZE * CSRMV_NNZ_PER_THREAD_2) < CSRMV_BLOCKS_PER_CU) \
    {                                                                                        \
        macro_to_launch(CSRMV_BLOCKSIZE, CSRMV_NNZ_PER_THREAD_1);                            \
    }                                                                                        \
    else                                                                                     \
    {                                                                                        \
        macro_to_launch(CSRMV_BLOCKSIZE, CSRMV_NNZ_PER_THREAD_2);                            \
    }

#define CSRMV_BLOCKSIZE 256
#define CSRMV_NNZ_PER_THREAD_0 1
#define CSRMV_NNZ_PER_THREAD_1 4
#define CSRMV_NNZ_PER_THREAD_2 8
#define CSRMV_BLOCKS_PER_CU 10

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
        RETURN_IF_HIP_ERROR(hipMemsetAsync(
            csrmv_info->nnzsplit.starting_ids, 0, sizeof(J) * (requiredBlocks + 1), stream));
    }
    else
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(
            &csrmv_info->nnzsplit.starting_ids, sizeof(J) * (2 * requiredBlocks + 2), stream));
        RETURN_IF_HIP_ERROR(hipMemsetAsync(
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

    LAUNCH_HELPER(LAUNCH_CSRMV_ANALYSIS)

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
    ROCSPARSE_ROUTINE_TRACE;

    const J ysize = (trans == rocsparse_operation_none) ? m : n;

    const bool skip_diag = (descr->type == rocsparse_matrix_type_symmetric);
    const bool conj      = (trans == rocsparse_operation_conjugate_transpose || force_conj);

    if(trans == rocsparse_operation_none || descr->type == rocsparse_matrix_type_symmetric)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::scale_array(handle, ysize, beta_device_host, y));
        LAUNCH_HELPER(LAUNCH_CSRMV)
    }

    if(trans != rocsparse_operation_none || descr->type == rocsparse_matrix_type_symmetric)
    {
        if(descr->type != rocsparse_matrix_type_symmetric)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::scale_array(handle, ysize, beta_device_host, y));
        }

        LAUNCH_HELPER(LAUNCH_CSRMVT)
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
INSTANTIATE(float, int32_t, int32_t, _Float16, _Float16, float);
INSTANTIATE(float, int64_t, int32_t, _Float16, _Float16, float);
INSTANTIATE(float, int64_t, int64_t, _Float16, _Float16, float);
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
