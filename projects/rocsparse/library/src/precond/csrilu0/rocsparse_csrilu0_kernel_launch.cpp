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

#include "rocsparse_csrilu0_kernel_launch.hpp"
#include "rocsparse_csrilu0_kernel_binsearch.hpp"
#include "rocsparse_csrilu0_kernel_hash.hpp"
#include "rocsparse_utility.hpp"
rocsparse_status
    rocsparse::csrilu0_strided_batched_kernel_launch(rocsparse_handle    handle,
                                                     int64_t             batch_count,
                                                     int64_t             m,
                                                     rocsparse_indextype csr_ptr_row_indextype,
                                                     const void* __restrict__ csr_row_ptr,
                                                     rocsparse_indextype csr_col_ind_indextype,
                                                     const void* __restrict__ csr_col_ind,
                                                     rocsparse_datatype csr_val_datatype,
                                                     void* __restrict__ csr_val,
                                                     int64_t csr_val_stride,
                                                     const void* __restrict__ csr_diag_ind,
                                                     int32_t* __restrict__ done,
                                                     int64_t done_stride,
                                                     const void* __restrict__ map,
                                                     void* __restrict__ zero_pivot,
                                                     int64_t zero_pivot_stride,
                                                     void* __restrict__ singular_pivot,
                                                     int64_t              singular_pivot_stride,
                                                     double               tol,
                                                     rocsparse_index_base idx_base,
                                                     int                  enable_boost,
                                                     size_t               size_boost_tol,
                                                     const void*          boost_tol,
                                                     const void*          boost_val,
                                                     int64_t              max_nnz)
{
    const bool sleep
        = (rocsparse::handle_get_arch_name(handle) == rocpsarse_arch_names::gfx908 && //
           handle->asic_rev < 2);

    auto launch_kernel = (sleep || ((handle->wavefront_size == 32) && (max_nnz >= 512))
                          || ((handle->wavefront_size == 64) && (max_nnz >= 1024)))
                             ? rocsparse::find_launch_csrilu0_kernel_binsearch(
                                 256,
                                 (sleep) ? 64 : handle->wavefront_size,
                                 sleep,
                                 csr_val_datatype,
                                 csr_ptr_row_indextype,
                                 csr_col_ind_indextype)
                             : rocsparse::find_launch_csrilu0_kernel_hash(256,
                                                                          handle->wavefront_size,
                                                                          max_nnz,
                                                                          csr_val_datatype,
                                                                          csr_ptr_row_indextype,
                                                                          csr_col_ind_indextype);

    if(launch_kernel == nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_arch_mismatch);
    }

    RETURN_IF_ROCSPARSE_ERROR(launch_kernel(handle,
                                            batch_count,
                                            m,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            csr_val,
                                            csr_val_stride,
                                            csr_diag_ind,
                                            done,
                                            done_stride,
                                            map,
                                            zero_pivot,
                                            zero_pivot_stride,
                                            singular_pivot,
                                            singular_pivot_stride,
                                            tol,
                                            idx_base,
                                            enable_boost,
                                            size_boost_tol,
                                            boost_tol,
                                            boost_val));
    return rocsparse_status_success;
}
