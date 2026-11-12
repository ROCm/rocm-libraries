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
#include "internal/level2/rocsparse_csrsv.h"
#include "rocsparse_csrsv.hpp"

#include "../conversion/rocsparse_coo2csr.hpp"
#include "../conversion/rocsparse_csr2coo.hpp"
#include "rocsparse_csrsv.hpp"

#include "../conversion/rocsparse_gcreate_identity_permutation.hpp"
#include "csrsv_device.h"
#include "rocsparse_assign_async.hpp"
#include "rocsparse_common.h"
#include "rocsparse_control.hpp"
#include "rocsparse_primitives.hpp"
#include "rocsparse_triangular_transpose.hpp"
#include "rocsparse_utility.hpp"

template <typename J>
rocsparse_status sort2(rocsparse_handle handle,
                       int64_t          m,
                       void*            done_array,
                       void*            workspace2,
                       void*            workspace,
                       void*            row_map,
                       void*            buffer)
{
    size_t                                        rocprim_size;
    uint32_t                                      startbit = 0;
    uint32_t                                      endbit   = rocsparse::clz(m);
    rocsparse::primitives::double_buffer<int32_t> keys(done_array, workspace2);
    rocsparse::primitives::double_buffer<J>       vals(workspace, row_map);
    RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::radix_sort_pairs_buffer_size<int32_t, J>(
        handle, m, startbit, endbit, &rocprim_size)));
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::radix_sort_pairs(
        handle, keys, vals, m, startbit, endbit, rocprim_size, buffer));
    RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(handle->stream));
    if(vals.current() != row_map)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(
            row_map, vals.current(), sizeof(J) * m, hipMemcpyDeviceToDevice, handle->stream));
    }
    RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(handle->stream));
    return rocsparse_status_success;
}

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void triangular_transposed_diag_ind_to_original_kernel(
        int64_t m,
        const I* __restrict__ transposed_diag_ind,
        const I* __restrict__ transposed_perm,
        I* __restrict__ diag_ind)
    {
        const int64_t row = static_cast<int64_t>(blockIdx.x) * BLOCKSIZE + threadIdx.x;
        if(row < m)
        {
            const I p     = transposed_diag_ind[row];
            diag_ind[row] = (p < 0) ? p : transposed_perm[p];
        }
    }

    static rocsparse_status
        triangular_transposed_diag_ind_to_original(rocsparse_handle    handle,
                                                   int64_t             m,
                                                   rocsparse_indextype indextype,
                                                   const void*         transposed_diag_ind,
                                                   const void*         transposed_perm,
                                                   void*               diag_ind)
    {
        static constexpr uint32_t BLOCKSIZE = 256;
        if(m == 0)
        {
            return rocsparse_status_success;
        }
        const dim3 blocks((m - 1) / BLOCKSIZE + 1);
        const dim3 threads(BLOCKSIZE);
        switch(indextype)
        {
        case rocsparse_indextype_i32:
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (triangular_transposed_diag_ind_to_original_kernel<BLOCKSIZE, int32_t>),
                blocks,
                threads,
                0,
                handle->stream,
                m,
                reinterpret_cast<const int32_t*>(transposed_diag_ind),
                reinterpret_cast<const int32_t*>(transposed_perm),
                reinterpret_cast<int32_t*>(diag_ind));
            return rocsparse_status_success;
        }
        case rocsparse_indextype_i64:
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (triangular_transposed_diag_ind_to_original_kernel<BLOCKSIZE, int64_t>),
                blocks,
                threads,
                0,
                handle->stream,
                m,
                reinterpret_cast<const int64_t*>(transposed_diag_ind),
                reinterpret_cast<const int64_t*>(transposed_perm),
                reinterpret_cast<int64_t*>(diag_ind));
            return rocsparse_status_success;
        }
        default:
        {
            break;
        }
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    }
}

rocsparse_status rocsparse::gtrm_analysis(rocsparse_handle                  handle,
                                          rocsparse_operation               operation,
                                          rocsparse_const_spmat_descr       A,
                                          rocsparse::triangular_info_t*     info,
                                          rocsparse::pivot_symbolic_info_t* pivot_info,
                                          rocsparse::idnvec_descr_t*        diag_ind_storage,
                                          rocsparse::transposed_sparsity_pattern_t* transposed,
                                          size_t                                    buffer_size,
                                          void*                                     buffer)
{
    ROCSPARSE_ROUTINE_TRACE;
    const rocsparse_operation trans                 = operation;
    const int64_t             m                     = A->rows;
    const int64_t             nnz                   = A->nnz;
    const rocsparse_mat_descr descr                 = A->descr;
    const rocsparse_indextype csr_row_ptr_indextype = A->row_type;
    const void*               csr_row_ptr           = A->const_row_data;
    const rocsparse_indextype csr_col_ind_indextype = A->col_type;
    const void*               csr_col_ind           = A->const_col_data;
    void*                     temp_buffer           = buffer;
    const size_t              sizeof_I = rocsparse::indextype_sizeof(csr_row_ptr_indextype);
    const size_t              sizeof_J = rocsparse::indextype_sizeof(csr_col_ind_indextype);
    // Stream
    hipStream_t stream = handle->stream;

#ifdef ROCSPARSE_WITH_TRSM_REFACTORING
    if(trans != rocsparse_operation_none)
#else
    if(trans == rocsparse_operation_transpose || trans == rocsparse_operation_conjugate_transpose)
#endif
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_triangular_transpose(handle,
                                                                 m,
                                                                 nnz,
                                                                 descr,
                                                                 csr_row_ptr_indextype,
                                                                 csr_row_ptr,
                                                                 csr_col_ind_indextype,
                                                                 csr_col_ind,
                                                                 transposed,
                                                                 temp_buffer));
    }

    {
        info->set_m(m);
        info->set_nnz(nnz);
        info->set_descr(descr);
    }

    char* ptr = reinterpret_cast<char*>(temp_buffer);
    // max_nnz
    void* d_max_nnz = ptr;
    ptr += 256;

    // done array
    const size_t done_array_size_in_bytes = ((sizeof(int32_t) * m - 1) / 256 + 1) * 256;
    int32_t*     done_array               = reinterpret_cast<int32_t*>(ptr);
    ptr += done_array_size_in_bytes;

    // Initialize temporary buffer with 0
    size_t zero_size_in_bytes = 256 + ((sizeof(int32_t) * m - 1) / 256 + 1) * 256;
    RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(temp_buffer, 0, zero_size_in_bytes, stream));

    // workspace
    const size_t              workspace_size_in_bytes = ((sizeof_J * m - 1) / 256 + 1) * 256;
    const rocsparse_indextype workspace_indextype     = csr_col_ind_indextype;
    void*                     workspace               = ptr;
    ptr += workspace_size_in_bytes;

    // workspace22
    const size_t workspace2_size_in_bytes = ((sizeof(int32_t) * m - 1) / 256 + 1) * 256;
    void*        workspace2               = ptr;
    ptr += workspace2_size_in_bytes;

    // rocprim buffer
    void* rocprim_buffer = ptr;

    // Allocate buffer to hold diagonal entry point
    RETURN_IF_ROCSPARSE_ERROR(diag_ind_storage->create_async(m, csr_row_ptr_indextype, stream));

    // Allocate buffer to hold zero pivot
    RETURN_IF_ROCSPARSE_ERROR(pivot_info->create_zero_pivot_async(csr_col_ind_indextype, stream));

    // Allocate buffer to hold row map
    RETURN_IF_ROCSPARSE_ERROR(info->get_row_map()->create_async(m, csr_col_ind_indextype, stream));

    //
    // Synchronization needed.
    //
    RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(stream));

    //
    // Initialize zero pivot
    //
    // Allocate buffer to hold zero pivot
    void* zero_pivot = pivot_info->get_position();
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_max_async(
        pivot_info->get_batch_count(), csr_col_ind_indextype, zero_pivot, stream));

    //  rocsparse_indextype row_map_indextype = csr_col_ind_indextype;
    void*               row_map            = info->get_row_map()->get_values();
    rocsparse_indextype diag_ind_indextype = csr_row_ptr_indextype;

    // The shared diag_ind holds positions in the original matrix; for a transposed
    // analysis the kernel output is in the transposed one and is mapped back below.
    void* diag_ind = diag_ind_storage->get_values();
    if(trans != rocsparse_operation_none)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(&diag_ind, sizeof_I * m, stream));
    }

    const rocsparse_indextype local_csr_row_ptr_indextype = csr_row_ptr_indextype;

    const void* local_csr_row_ptr = (trans == rocsparse_operation_none)
                                        ? csr_row_ptr
                                        : transposed->get_row_ptr()->get_values();

    const rocsparse_indextype local_csr_col_ind_indextype = csr_col_ind_indextype;

    const void* local_csr_col_ind = (trans == rocsparse_operation_none)
                                        ? csr_col_ind
                                        : transposed->get_col_ind()->get_values();

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::launch_csrsv_analysis_kernel(handle,
                                                                      trans,
                                                                      m,
                                                                      local_csr_row_ptr_indextype,
                                                                      local_csr_row_ptr,
                                                                      local_csr_col_ind_indextype,
                                                                      local_csr_col_ind,
                                                                      diag_ind_indextype,
                                                                      diag_ind,
                                                                      done_array,
                                                                      d_max_nnz,
                                                                      zero_pivot,
                                                                      descr->base,
                                                                      descr->diag_type,
                                                                      descr->fill_mode));

    if(trans != rocsparse_operation_none)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::triangular_transposed_diag_ind_to_original(
            handle,
            m,
            diag_ind_indextype,
            diag_ind,
            transposed->get_perm()->get_values(),
            diag_ind_storage->get_values()));
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(diag_ind, stream));
    }

    {
        // Post processing
        int64_t max_nnz = 0; // important to set it to zero since sizeof_I might be int32_t.
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMemcpyAsync(&max_nnz, d_max_nnz, sizeof_I, hipMemcpyDeviceToHost, stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(stream));
        info->set_max_nnz(max_nnz);
    }

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::gcreate_identity_permutation(handle, m, workspace_indextype, workspace));

    const rocsparse_indextype csrt_col_ind_indextype = csr_col_ind_indextype;
    auto                      apply_sort2            = sort2<int32_t>;

    if((csrt_col_ind_indextype == rocsparse_indextype_i32))
    {
        apply_sort2 = sort2<int32_t>;
    }
    else if((csrt_col_ind_indextype == rocsparse_indextype_i64))
    {
        apply_sort2 = sort2<int64_t>;
    }
    else
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    }

    RETURN_IF_ROCSPARSE_ERROR(
        apply_sort2(handle, m, done_array, workspace2, workspace, row_map, rocprim_buffer));

    return rocsparse_status_success;
}

rocsparse_status rocsparse::csrsv_analysis(rocsparse_handle            handle,
                                           rocsparse_operation         trans,
                                           rocsparse_const_spmat_descr A,
                                           rocsparse_analysis_policy   analysis_policy,
                                           rocsparse_solve_policy      solve_policy,
                                           rocsparse_csrsv_info*       p_csrsv_info,
                                           void*                       temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;
    // Quick return if possible
    if(A->rows == 0)
    {
        return rocsparse_status_success;
    }

    auto csrsv_info = p_csrsv_info[0];
    auto descr      = A->descr;
    // Check matrix type
    ROCSPARSE_CHECKARG(2,
                       A,
                       (descr->type != rocsparse_matrix_type_general
                        && descr->type != rocsparse_matrix_type_triangular),
                       rocsparse_status_not_implemented);

    // Check matrix sorting mode
    ROCSPARSE_CHECKARG(2,
                       A,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    auto info = A->info;
    // Differentiate the analysis policies
    if(analysis_policy == rocsparse_analysis_policy_reuse)
    {
        rocsparse::triangular_info_t* p = info->get_csrsv_info(trans, descr->fill_mode);

        if((descr->fill_mode == rocsparse_fill_mode_lower) && (trans == rocsparse_operation_none))
        {
            p = (p != nullptr) ? p : info->get_csrilu0_info(trans, descr->fill_mode);
            p = (p != nullptr) ? p : info->get_csric0_info(trans, descr->fill_mode);
        }

        p = (p != nullptr) ? p : info->get_csrsm_info(trans, descr->fill_mode);
        if(p != nullptr)
        {
            info->set_csrsv_info(trans, descr->fill_mode, p);
            return rocsparse_status_success;
        }
    }

    if(csrsv_info == nullptr)
    {
        csrsv_info      = new _rocsparse_csrsv_info(info->get_triangular_data(),
                                               info->get_singularity_numeric_exact());
        p_csrsv_info[0] = csrsv_info;
    }

    // Perform analysis
    RETURN_IF_ROCSPARSE_ERROR(
        info->recreate_triangular_info(handle, trans, descr->fill_mode, A, 0, temp_buffer));

    return rocsparse_status_success;
}
