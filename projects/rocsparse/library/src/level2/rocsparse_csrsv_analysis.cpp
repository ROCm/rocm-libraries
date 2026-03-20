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

#include "../conversion/rocsparse_gcoo2csr.hpp"
#include "../conversion/rocsparse_gcreate_identity_permutation.hpp"
#include "../conversion/rocsparse_gcsr2coo.hpp"
#include "../level1/rocsparse_gthr.hpp"
#include "csrsv_device.h"
#include "rocsparse_assign_async.hpp"
#include "rocsparse_common.h"
#include "rocsparse_control.hpp"
#include "rocsparse_primitives.hpp"
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
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
    if(vals.current() != row_map)
    {
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            row_map, vals.current(), sizeof(J) * m, hipMemcpyDeviceToDevice, handle->stream));
    }
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
    return rocsparse_status_success;
}

template <typename I, typename J>
rocsparse_status sort(rocsparse_handle     handle,
                      int64_t              m,
                      int64_t              nnz,
                      void*                csrt_col_ind,
                      void*                tmp_work1,
                      void*                csrt_perm,
                      void*                tmp_work2,
                      rocsparse_index_base base,
                      void*                buffer,
                      void**               p_sorted_perm,
                      void**               p_sorted_col_ind)
{
    uint32_t startbit = 0;
    uint32_t endbit   = rocsparse::clz(m);
    size_t   rocprim_size;

    RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::radix_sort_pairs_buffer_size<J, I>(
        handle, nnz, startbit, endbit, &rocprim_size, true)));

    rocsparse::primitives::double_buffer<J> keys(tmp_work1, csrt_col_ind);
    rocsparse::primitives::double_buffer<I> vals(csrt_perm, tmp_work2);
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::radix_sort_pairs(
        handle, keys, vals, nnz, startbit, endbit, rocprim_size, buffer));

    p_sorted_perm[0]    = vals.current();
    p_sorted_col_ind[0] = keys.current();
    return rocsparse_status_success;
}

rocsparse_status rocsparse_trm_transpose_buffer_size(int64_t             m,
                                                     int64_t             nnz,
                                                     rocsparse_indextype csr_row_ptr_indextype,
                                                     rocsparse_indextype csr_col_ind_indextype,
                                                     size_t*             p_buffer_size)
{
    const size_t sizeof_I = rocsparse::indextype_sizeof(csr_row_ptr_indextype);
    const size_t sizeof_J = rocsparse::indextype_sizeof(csr_col_ind_indextype);
    p_buffer_size[0]
        = ((sizeof_J * nnz - 1) / 256 + 1) * 256 + ((sizeof_I * nnz - 1) / 256 + 1) * 256;
    return rocsparse_status_success;
}

static rocsparse_status rocsparse_trm_transpose(rocsparse_handle          handle,
                                                int64_t                   m,
                                                int64_t                   nnz,
                                                const rocsparse_mat_descr descr,
                                                rocsparse_indextype       csr_row_ptr_indextype,
                                                const void*               csr_row_ptr,
                                                rocsparse_indextype       csr_col_ind_indextype,
                                                const void*               csr_col_ind,
                                                rocsparse::trm_info_t*    info,
                                                void*                     temp_buffer)
{
    const size_t              sizeof_I = rocsparse::indextype_sizeof(csr_row_ptr_indextype);
    const size_t              sizeof_J = rocsparse::indextype_sizeof(csr_col_ind_indextype);
    const rocsparse_indextype csrt_row_ptr_indextype = csr_row_ptr_indextype;
    const rocsparse_indextype csrt_perm_indextype    = csr_row_ptr_indextype;
    const rocsparse_indextype csrt_col_ind_indextype = csr_col_ind_indextype;

    hipStream_t stream = handle->stream;
    // TODO: this need to be changed.
    // LCOV_EXCL_START
    if(info->get_transposed_perm() != nullptr || info->get_transposed_row_ptr() != nullptr
       || info->get_transposed_col_ind() != nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }
    // LCOV_EXCL_STOP

    const size_t csrt_row_ptr_size_in_bytes = sizeof_I * (m + 1);
    void**       ref_csrt_row_ptr           = info->get_ref_transposed_row_ptr();

    RETURN_IF_HIP_ERROR(
        rocsparse_hipMallocAsync(ref_csrt_row_ptr, csrt_row_ptr_size_in_bytes, stream));
    void* csrt_row_ptr = ref_csrt_row_ptr[0];
    if(nnz == 0)
    {
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::valset(handle, m + 1, descr->base, csrt_row_ptr_indextype, csrt_row_ptr));
    }

    // Buffer
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    // work1 buffer
    const size_t tmp_work1_size = ((sizeof_J * nnz - 1) / 256 + 1) * 256;
    void*        tmp_work1      = ptr;
    ptr += tmp_work1_size;
    const rocsparse_indextype tmp_work1_indextype = csr_col_ind_indextype;

    // work2 buffer
    const size_t tmp_work2_size = ((sizeof_I * nnz - 1) / 256 + 1) * 256;
    void*        tmp_work2      = ptr;
    ptr += tmp_work2_size;

    // Load CSR column indices into work1 buffer
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(tmp_work1, csr_col_ind, sizeof_J * nnz, hipMemcpyDeviceToDevice, stream));

    void**       ref_csrt_perm           = info->get_ref_transposed_perm();
    const size_t csrt_perm_size_in_bytes = sizeof_I * nnz;

    void**       ref_csrt_col_ind           = info->get_ref_transposed_col_ind();
    const size_t csrt_col_ind_size_in_bytes = sizeof_J * nnz;

    RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(ref_csrt_perm, csrt_perm_size_in_bytes, stream));
    RETURN_IF_HIP_ERROR(
        rocsparse_hipMallocAsync(ref_csrt_col_ind, csrt_col_ind_size_in_bytes, stream));

    void* csrt_perm    = ref_csrt_perm[0];
    void* csrt_col_ind = ref_csrt_col_ind[0];

    // Create identity permutation
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::gcreate_identity_permutation(handle, nnz, csrt_perm_indextype, csrt_perm));

    // Stable sort COO by columns
    auto apply_sort = sort<int32_t, int32_t>;
    if((csrt_perm_indextype == rocsparse_indextype_i32)
       && (csrt_col_ind_indextype == rocsparse_indextype_i32))
    {
        apply_sort = sort<int32_t, int32_t>;
    }
    else if((csrt_perm_indextype == rocsparse_indextype_i32)
            && (csrt_col_ind_indextype == rocsparse_indextype_i64))
    {
        apply_sort = sort<int32_t, int64_t>;
    }
    else if((csrt_perm_indextype == rocsparse_indextype_i64)
            && (csrt_col_ind_indextype == rocsparse_indextype_i64))
    {
        apply_sort = sort<int64_t, int64_t>;
    }
    else if((csrt_perm_indextype == rocsparse_indextype_i64)
            && (csrt_col_ind_indextype == rocsparse_indextype_i32))
    {
        apply_sort = sort<int64_t, int32_t>;
    }
    else
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    }
    void* p_sorted_col_ind[1]{};
    void* p_sorted_perm[1]{};
    void* rocprim_buffer = ptr;
    apply_sort(handle,
               m,
               nnz,
               csrt_col_ind,
               tmp_work1,
               csrt_perm,
               tmp_work2,
               descr->base,
               rocprim_buffer,
               p_sorted_perm,
               p_sorted_col_ind);

    //
    // Copy permutation vector, if not already available
    //
    if(p_sorted_perm[0] != csrt_perm)
    {
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(csrt_perm,
                                           p_sorted_perm[0],
                                           csrt_perm_size_in_bytes,
                                           hipMemcpyDeviceToDevice,
                                           handle->stream));
    }

    //
    // Create column pointers
    //
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gcoo2csr(handle,
                                                  csr_col_ind_indextype,
                                                  p_sorted_col_ind[0],
                                                  nnz,
                                                  m,
                                                  csrt_row_ptr_indextype,
                                                  csrt_row_ptr,
                                                  descr->base));

    //
    // Create row indices
    //
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gcsr2coo(handle,
                                                  csr_row_ptr_indextype,
                                                  csr_row_ptr,
                                                  nnz,
                                                  m,
                                                  tmp_work1_indextype,
                                                  tmp_work1,
                                                  descr->base));

    //
    // Permute column indices
    //

    RETURN_IF_ROCSPARSE_ERROR((rocsparse::gthr_indices(handle,
                                                       nnz,
                                                       tmp_work1_indextype,
                                                       tmp_work1,
                                                       csrt_col_ind_indextype,
                                                       csrt_col_ind,
                                                       csrt_perm_indextype,
                                                       csrt_perm,
                                                       rocsparse_index_base_zero)));
    return rocsparse_status_success;
}

rocsparse_status rocsparse::gtrm_analysis(rocsparse_handle          handle,
                                          rocsparse_operation       trans,
                                          int64_t                   m,
                                          int64_t                   nnz,
                                          const rocsparse_mat_descr descr,
                                          rocsparse_datatype        csr_val_datatype,
                                          const void*               csr_val,
                                          rocsparse_indextype       csr_row_ptr_indextype,
                                          const void*               csr_row_ptr,
                                          rocsparse_indextype       csr_col_ind_indextype,
                                          const void*               csr_col_ind,
                                          rocsparse::trm_info_t*    info,
                                          rocsparse::pivot_info_t*  pivot_info,
                                          void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;
    const size_t sizeof_I = rocsparse::indextype_sizeof(csr_row_ptr_indextype);
    const size_t sizeof_J = rocsparse::indextype_sizeof(csr_col_ind_indextype);
    // Stream
    hipStream_t stream = handle->stream;

    if(trans == rocsparse_operation_transpose || trans == rocsparse_operation_conjugate_transpose)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_trm_transpose(handle,
                                                          m,
                                                          nnz,
                                                          descr,
                                                          csr_row_ptr_indextype,
                                                          csr_row_ptr,
                                                          csr_col_ind_indextype,
                                                          csr_col_ind,
                                                          info,
                                                          temp_buffer));
    }

    {
        info->set_m(m);
        info->set_nnz(nnz);
        info->set_descr(descr);
        info->set_row_ptr((trans == rocsparse_operation_none) ? csr_row_ptr
                                                              : info->get_transposed_row_ptr());
        info->set_col_ind((trans == rocsparse_operation_none) ? csr_col_ind
                                                              : info->get_transposed_col_ind());
        info->set_offset_indextype(csr_row_ptr_indextype);
        info->set_index_indextype(csr_col_ind_indextype);
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
    RETURN_IF_HIP_ERROR(hipMemsetAsync(temp_buffer, 0, zero_size_in_bytes, stream));

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
    RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(info->get_ref_diag_ind(), sizeof_I * m, stream));

    // Allocate buffer to hold zero pivot
    pivot_info->create_zero_pivot_async(csr_col_ind_indextype, stream);

    // Allocate buffer to hold row map
    RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(info->get_ref_row_map(), sizeof_J * m, stream));

    //
    // Synchronization needed.
    //
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    //
    // Initialize zero pivot
    //
    // Allocate buffer to hold zero pivot
    void* zero_pivot = pivot_info->get_position();
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::assign_max_async(
        pivot_info->get_batch_count(), csr_col_ind_indextype, zero_pivot, stream));

    //  rocsparse_indextype row_map_indextype = csr_col_ind_indextype;
    void*               row_map            = info->get_row_map();
    rocsparse_indextype diag_ind_indextype = csr_row_ptr_indextype;
    void*               diag_ind           = info->get_diag_ind();

    const rocsparse_indextype local_csr_row_ptr_indextype = (trans == rocsparse_operation_none)
                                                                ? csr_row_ptr_indextype
                                                                : info->get_offset_indextype();

    const void* local_csr_row_ptr
        = (trans == rocsparse_operation_none) ? csr_row_ptr : info->get_transposed_row_ptr();

    const rocsparse_indextype local_csr_col_ind_indextype
        = (trans == rocsparse_operation_none) ? csr_col_ind_indextype : info->get_index_indextype();

    const void* local_csr_col_ind
        = (trans == rocsparse_operation_none) ? csr_col_ind : info->get_transposed_col_ind();

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

    {
        // Post processing
        int64_t max_nnz = 0; // important to set it to zero since sizeof_I might be int32_t.
        RETURN_IF_HIP_ERROR(
            hipMemcpyAsync(&max_nnz, d_max_nnz, sizeof_I, hipMemcpyDeviceToHost, stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));
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

    apply_sort2(handle, m, done_array, workspace2, workspace, row_map, rocprim_buffer);
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
        rocsparse::trm_info_t* p = nullptr;
        p = (p != nullptr) ? p : info->get_csrsv_info(trans, descr->fill_mode);

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
        csrsv_info      = new _rocsparse_csrsv_info();
        p_csrsv_info[0] = csrsv_info;
    }

    // Perform analysis
    RETURN_IF_ROCSPARSE_ERROR(csrsv_info->recreate(handle,
                                                   trans,
                                                   A->rows,
                                                   A->nnz,
                                                   A->descr,
                                                   A->data_type,
                                                   A->const_val_data,
                                                   // csr_val_stride,
                                                   A->row_type,
                                                   A->const_row_data,
                                                   A->col_type,
                                                   A->const_col_data,
                                                   temp_buffer));

    return rocsparse_status_success;
}
