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
#include "csrsv_device.h"
#include "rocsparse_common.h"
#include "rocsparse_control.hpp"
#include "rocsparse_csrsv.hpp"
#include "rocsparse_primitives.hpp"
#include "rocsparse_utility.hpp"
#include <vector>
template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, bool SLEEP, typename I, typename J>
rocsparse_status launch_csrsv_analysis_upper_kernel(rocsparse_handle    handle,
                                                    int64_t             m,
                                                    rocsparse_indextype csr_row_ptr_indextype,
                                                    const void* __restrict__ csr_row_ptr,
                                                    rocsparse_indextype csr_col_ind_indextype,
                                                    const void* __restrict__ csr_col_ind,
                                                    rocsparse_indextype diag_ind_indextype,
                                                    void* __restrict__ diag_ind,
                                                    int32_t* __restrict__ done_array,
                                                    void* __restrict__ max_nnz,
                                                    void* __restrict__ zero_pivot,
                                                    rocsparse_index_base idx_base,
                                                    rocsparse_diag_type  diag_type)
{
    dim3 csrsv_blocks((m * handle->wavefront_size - 1) / BLOCKSIZE + 1);
    dim3 csrsv_threads(BLOCKSIZE);
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
        (rocsparse::csrsv_analysis_upper_kernel<BLOCKSIZE, WF_SIZE, SLEEP>),
        csrsv_blocks,
        csrsv_threads,
        0,
        handle->stream,
        static_cast<J>(m),
        reinterpret_cast<const I*>(csr_row_ptr),
        reinterpret_cast<const J*>(csr_col_ind),
        reinterpret_cast<I*>(diag_ind),
        done_array,
        reinterpret_cast<I*>(max_nnz),
        reinterpret_cast<J*>(zero_pivot),
        idx_base,
        diag_type);
    return rocsparse_status_success;
}

template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, bool SLEEP, typename I, typename J>
rocsparse_status launch_csrsv_analysis_lower_kernel(rocsparse_handle    handle,
                                                    int64_t             m,
                                                    rocsparse_indextype csr_row_ptr_indextype,
                                                    const void* __restrict__ csr_row_ptr,
                                                    rocsparse_indextype csr_col_ind_indextype,
                                                    const void* __restrict__ csr_col_ind,
                                                    rocsparse_indextype diag_ind_indextype,
                                                    void* __restrict__ diag_ind,
                                                    int32_t* __restrict__ done_array,
                                                    void* __restrict__ max_nnz,
                                                    void* __restrict__ zero_pivot,
                                                    rocsparse_index_base idx_base,
                                                    rocsparse_diag_type  diag_type)
{
    dim3 csrsv_blocks((m * handle->wavefront_size - 1) / BLOCKSIZE + 1);
    dim3 csrsv_threads(BLOCKSIZE);
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
        (rocsparse::csrsv_analysis_lower_kernel<BLOCKSIZE, WF_SIZE, SLEEP>),
        csrsv_blocks,
        csrsv_threads,
        0,
        handle->stream,
        static_cast<J>(m),
        reinterpret_cast<const I*>(csr_row_ptr),
        reinterpret_cast<const J*>(csr_col_ind),
        reinterpret_cast<I*>(diag_ind),
        done_array,
        reinterpret_cast<I*>(max_nnz),
        reinterpret_cast<J*>(zero_pivot),
        idx_base,
        diag_type);

    return rocsparse_status_success;
}

typedef decltype(&launch_csrsv_analysis_lower_kernel<1024, 32, false, int32_t, int32_t>)
    csrsv_analysis_kernel_t;

template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, bool SLEEP, typename I, typename J>
static csrsv_analysis_kernel_t find_mode(rocsparse_fill_mode mode, rocsparse_operation operation)
{

    return (operation == rocsparse_operation_none)
               ? ((mode == rocsparse_fill_mode_lower)
                      ? launch_csrsv_analysis_lower_kernel<BLOCKSIZE, WF_SIZE, SLEEP, I, J>
                      : launch_csrsv_analysis_upper_kernel<BLOCKSIZE, WF_SIZE, SLEEP, I, J>)
               : ((mode == rocsparse_fill_mode_lower)
                      ? launch_csrsv_analysis_upper_kernel<BLOCKSIZE, WF_SIZE, SLEEP, I, J>
                      : launch_csrsv_analysis_lower_kernel<BLOCKSIZE, WF_SIZE, SLEEP, I, J>);
}

template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, bool SLEEP, typename I, typename... P>
static csrsv_analysis_kernel_t find_j(rocsparse_indextype j, P... p)
{
    return (j == rocsparse_indextype_i32)   ? find_mode<BLOCKSIZE, WF_SIZE, SLEEP, I, int32_t>(p...)
           : (j == rocsparse_indextype_i64) ? find_mode<BLOCKSIZE, WF_SIZE, SLEEP, I, int64_t>(p...)
                                            : nullptr;
}

template <uint32_t BLOCKSIZE, uint32_t WF_SIZE, bool SLEEP, typename... P>
static csrsv_analysis_kernel_t find_i(rocsparse_indextype i, P... p)
{
    return (i == rocsparse_indextype_i32)   ? find_j<BLOCKSIZE, WF_SIZE, SLEEP, int32_t>(p...)
           : (i == rocsparse_indextype_i64) ? find_j<BLOCKSIZE, WF_SIZE, SLEEP, int64_t>(p...)
                                            : nullptr;
}

static csrsv_analysis_kernel_t find_csrsv_analysis_kernel(uint32_t            blocksize_,
                                                          uint32_t            wfsize_,
                                                          bool                sleep_,
                                                          rocsparse_indextype i_type,
                                                          rocsparse_indextype j_type,
                                                          rocsparse_fill_mode mode_,
                                                          rocsparse_operation operation_)
{
    if(blocksize_ == 1024 && ((wfsize_ == 32) && (sleep_ == false)))
    {
        return find_i<1024, 32, false>(i_type, j_type, mode_, operation_);
    }
    else if(blocksize_ == 1024 && ((wfsize_ == 64) && (sleep_ == false)))
    {
        return find_i<1024, 64, false>(i_type, j_type, mode_, operation_);
    }
    else if(blocksize_ == 1024 && ((wfsize_ == 64) && (sleep_ == true)))
    {
        return find_i<1024, 64, true>(i_type, j_type, mode_, operation_);
    }
    else
    {
        return nullptr;
    }
}

rocsparse_status rocsparse::launch_csrsv_analysis_kernel(rocsparse_handle    handle,
                                                         rocsparse_operation trans,
                                                         int64_t             m,
                                                         rocsparse_indextype csr_row_ptr_indextype,
                                                         const void* __restrict__ csr_row_ptr,
                                                         rocsparse_indextype csr_col_ind_indextype,
                                                         const void* __restrict__ csr_col_ind,
                                                         rocsparse_indextype diag_ind_indextype,
                                                         void* __restrict__ diag_ind,
                                                         int32_t* __restrict__ done_array,
                                                         void* __restrict__ max_nnz,
                                                         void* __restrict__ zero_pivot,
                                                         rocsparse_index_base idx_base,
                                                         rocsparse_diag_type  diag_type,
                                                         rocsparse_fill_mode  fill_mode)
{
    // Determine archid and ASIC revision
    const std::string gcn_arch_name = rocsparse::handle_get_arch_name(handle);
    const bool SLEEP = ((gcn_arch_name == rocpsarse_arch_names::gfx908) && (handle->asic_rev < 2));
    csrsv_analysis_kernel_t launch = find_csrsv_analysis_kernel(1024,
                                                                handle->wavefront_size,
                                                                SLEEP,
                                                                csr_row_ptr_indextype,
                                                                csr_col_ind_indextype,
                                                                fill_mode,
                                                                trans);
    if(launch != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(launch(handle,
                                         m,
                                         csr_row_ptr_indextype,
                                         csr_row_ptr,
                                         csr_col_ind_indextype,
                                         csr_col_ind,
                                         diag_ind_indextype,
                                         diag_ind,
                                         done_array,
                                         max_nnz,
                                         zero_pivot,
                                         idx_base,
                                         diag_type));
        return rocsparse_status_success;
    }
    else
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_arch_mismatch);
    }
}
