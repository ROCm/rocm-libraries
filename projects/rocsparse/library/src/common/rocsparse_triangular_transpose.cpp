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

#include "rocsparse_triangular_transpose.hpp"
#include "../conversion/rocsparse_gcoo2csr.hpp"
#include "../conversion/rocsparse_gcreate_identity_permutation.hpp"
#include "../conversion/rocsparse_gcsr2coo.hpp"
#include "../level1/rocsparse_gthr.hpp"
#include "rocsparse_common.h"
#include "rocsparse_control.hpp"
#include "rocsparse_primitives.hpp"
#include "rocsparse_utility.hpp"

template <typename I, typename J>
static rocsparse_status sort(rocsparse_handle     handle,
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

rocsparse_status
    rocsparse_triangular_transpose_buffer_size(int64_t             m,
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

rocsparse_status
    rocsparse_triangular_transpose(rocsparse_handle                          handle,
                                   int64_t                                   m,
                                   int64_t                                   nnz,
                                   const rocsparse_mat_descr                 descr,
                                   rocsparse_indextype                       csr_row_ptr_indextype,
                                   const void*                               csr_row_ptr,
                                   rocsparse_indextype                       csr_col_ind_indextype,
                                   const void*                               csr_col_ind,
                                   rocsparse::transposed_sparsity_pattern_t* transposed,
                                   void*                                     temp_buffer)
{
    const size_t              sizeof_I = rocsparse::indextype_sizeof(csr_row_ptr_indextype);
    const size_t              sizeof_J = rocsparse::indextype_sizeof(csr_col_ind_indextype);
    const rocsparse_indextype csrt_row_ptr_indextype = csr_row_ptr_indextype;
    const rocsparse_indextype csrt_perm_indextype    = csr_row_ptr_indextype;
    const rocsparse_indextype csrt_col_ind_indextype = csr_col_ind_indextype;

    hipStream_t stream = handle->stream;
    RETURN_IF_ROCSPARSE_ERROR(
        transposed->get_row_ptr()->create_async(m + 1, csrt_row_ptr_indextype, stream));
    void* csrt_row_ptr = transposed->get_row_ptr()->get_values();
    if(nnz == 0)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(stream));
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
    RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(
        tmp_work1, csr_col_ind, sizeof_J * nnz, hipMemcpyDeviceToDevice, stream));

    RETURN_IF_ROCSPARSE_ERROR(
        transposed->get_perm()->create_async(nnz, csrt_perm_indextype, stream));
    RETURN_IF_ROCSPARSE_ERROR(
        transposed->get_col_ind()->create_async(nnz, csrt_col_ind_indextype, stream));

    void* csrt_perm    = transposed->get_perm()->get_values();
    void* csrt_col_ind = transposed->get_col_ind()->get_values();

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
    RETURN_IF_ROCSPARSE_ERROR(apply_sort(handle,
                                         m,
                                         nnz,
                                         csrt_col_ind,
                                         tmp_work1,
                                         csrt_perm,
                                         tmp_work2,
                                         descr->base,
                                         rocprim_buffer,
                                         p_sorted_perm,
                                         p_sorted_col_ind));

    //
    // Copy permutation vector, if not already available
    //
    if(p_sorted_perm[0] != csrt_perm)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(
            csrt_perm, p_sorted_perm[0], sizeof_I * nnz, hipMemcpyDeviceToDevice, handle->stream));
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
