/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_assign_async.hpp"
#include "rocsparse_cscsv.hpp"
#include "rocsparse_cscsv_common.hpp"

#include "cscsv_device.h"

rocsparse_status rocsparse::cscsv_column_analysis(rocsparse_handle            handle,
                                                  rocsparse_const_spmat_descr A,
                                                  rocsparse_cscsv_info        info,
                                                  void*                       temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;
    hipStream_t               stream = handle->stream;
    const rocsparse_mat_descr descr  = A->descr;
    const int64_t             m      = A->cols;

    // Column pointers and row indices
    const rocsparse_indextype I_type   = A->col_type;
    const rocsparse_indextype J_type   = A->row_type;
    const size_t              sizeof_I = rocsparse::indextype_sizeof(I_type);
    const size_t              sizeof_J = rocsparse::indextype_sizeof(J_type);

    std::unique_ptr<rocsparse::trm_info_t> trm_info(new rocsparse::trm_info_t());
    trm_info->set_m(m);
    trm_info->set_nnz(A->nnz);
    trm_info->set_descr(descr);
    trm_info->set_row_ptr(A->const_col_data);
    trm_info->set_col_ind(A->const_row_data);
    trm_info->set_offset_indextype(I_type);
    trm_info->set_index_indextype(J_type);

    void* d_max_nnz = temp_buffer;
    RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(d_max_nnz, 0, 256, stream));

    RETURN_IF_HIP_ERROR(
        rocsparse_hipMallocAsync(trm_info->get_ref_diag_ind(), sizeof_I * m, stream));
    RETURN_IF_HIP_ERROR(
        rocsparse_hipMallocAsync(info->get_ref_dependency_count(), sizeof_J * m, stream));
    RETURN_IF_ROCSPARSE_ERROR(info->create_zero_pivot_async(J_type, stream));

    void* zero_pivot = info->get_position();
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::assign_max_async(info->get_batch_count(), J_type, zero_pivot, stream));

    // -1 marks a missing diagonal entry
    RETURN_IF_HIP_ERROR(
        rocsparse_hipMemsetAsync(trm_info->get_diag_ind(), 0xFF, sizeof_I * m, stream));
    RETURN_IF_HIP_ERROR(
        rocsparse_hipMemsetAsync(info->get_dependency_count(), 0, sizeof_J * m, stream));

    if(m > 0)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::cscsv_dispatch_indextype(I_type, J_type, [&](auto i, auto j) {
                using I                             = decltype(i);
                using J                             = decltype(j);
                static constexpr uint32_t BLOCKSIZE = rocsparse::s_cscsv_blocksize;
                static constexpr uint32_t L         = rocsparse::s_cscsv_analysis_groupsize;
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::cscsv_analysis_kernel<BLOCKSIZE, L>),
                    dim3((m * L - 1) / BLOCKSIZE + 1),
                    dim3(BLOCKSIZE),
                    0,
                    stream,
                    static_cast<J>(m),
                    reinterpret_cast<const I*>(A->const_col_data),
                    reinterpret_cast<const J*>(A->const_row_data),
                    reinterpret_cast<I*>(trm_info->get_diag_ind()),
                    reinterpret_cast<J*>(info->get_dependency_count()),
                    reinterpret_cast<I*>(d_max_nnz),
                    reinterpret_cast<J*>(zero_pivot),
                    descr->base,
                    descr->diag_type,
                    descr->fill_mode);
                return rocsparse_status_success;
            }));
    }

    int64_t max_nnz = 0;
    RETURN_IF_HIP_ERROR(
        rocsparse_hipMemcpyAsync(&max_nnz, d_max_nnz, sizeof_I, hipMemcpyDeviceToHost, stream));
    RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(stream));
    trm_info->set_max_nnz(max_nnz);

    info->set(rocsparse_operation_none, descr->fill_mode, trm_info.release());
    return rocsparse_status_success;
}
