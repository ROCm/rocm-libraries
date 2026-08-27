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

#include "../../level2/rocsparse_csrsv.hpp"
#include "rocsparse_csrsm.hpp"

#include "rocsparse_enum_utils.hpp"
#include "rocsparse_primitives.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::csrsm_analysis_buffer_size(rocsparse_handle            handle,
                                                       const int64_t               nrhs,
                                                       rocsparse_operation         op_A,
                                                       rocsparse_operation         op_B,
                                                       rocsparse_const_dnvec_descr alpha,
                                                       rocsparse_const_spmat_descr A,
                                                       rocsparse_const_dnmat_descr B,
                                                       size_t*          p_buffer_size_in_bytes,
                                                       rocsparse_error* p_error)
{
    ROCSPARSE_ROUTINE_TRACE;
    const int64_t M = A->rows;

    if(M == 0 || nrhs == 0)
    {
        p_buffer_size_in_bytes[0] = 0;
        return rocsparse_status_success;
    }

    if(nrhs == 1)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::csrsv_analysis_buffer_size(handle, op_A, A, p_buffer_size_in_bytes));
        return rocsparse_status_success;
    }

    //
    // B cannot be batched with a zero stride since it is an output.
    //
    const int64_t B_batch_stride = B->batch_stride;
    RETURN_IF_ROCSPARSE_ERROR(((B_batch_stride == 0) && (B->batch_count > 1))
                                  ? rocsparse_status_invalid_value
                                  : rocsparse_status_success);

    const size_t sizeof_I = rocsparse::indextype_sizeof(A->row_type);
    const size_t sizeof_J = rocsparse::indextype_sizeof(A->col_type);

    // max_nnz
    p_buffer_size_in_bytes[0] = 256;

    // Each thread block performs at most blockdim columns of the
    // rhs matrix. Therefore, the number of blocks depend on nrhs
    // and the blocksize.
    // Because of this, we might need a larger done_array compared
    // to csrsv.
    // int done_array
    p_buffer_size_in_bytes[0] += rocsparse::align_size(sizeof(int32_t) * M);

    // workspace
    p_buffer_size_in_bytes[0] += rocsparse::align_size(sizeof_J * M);

    // int workspace2
    p_buffer_size_in_bytes[0] += rocsparse::align_size(sizeof(int32_t) * M);

    uint32_t              startbit            = 0;
    uint32_t              endbit              = rocsparse::clz(M);
    static constexpr bool using_double_buffer = true;

    {
        size_t rocprim_size{};
        decltype(rocsparse::primitives::radix_sort_pairs_buffer_size<int32_t, int32_t>)* calc{};
        if(A->col_type == rocsparse_indextype_i32)
        {
            calc = rocsparse::primitives::radix_sort_pairs_buffer_size<int32_t, int32_t>;
        }
        else if(A->col_type == rocsparse_indextype_i64)
        {
            calc = rocsparse::primitives::radix_sort_pairs_buffer_size<int32_t, int64_t>;
        }
        RETURN_IF_ROCSPARSE_ERROR(
            (calc(handle, M, startbit, endbit, &rocprim_size, using_double_buffer)));
        // rocprim buffer
        p_buffer_size_in_bytes[0] += rocprim_size;
    }

    if(op_A != rocsparse_operation_none)
    {
        decltype(rocsparse::primitives::radix_sort_pairs_buffer_size<int32_t, int32_t>)* calc{};
        if((A->row_type == rocsparse_indextype_i32) && (A->col_type == rocsparse_indextype_i32))
        {
            calc = rocsparse::primitives::radix_sort_pairs_buffer_size<int32_t, int32_t>;
        }
        else if((A->row_type == rocsparse_indextype_i64)
                && (A->col_type == rocsparse_indextype_i32))
        {
            calc = rocsparse::primitives::radix_sort_pairs_buffer_size<int32_t, int64_t>;
        }
        else if((A->row_type == rocsparse_indextype_i32)
                && (A->col_type == rocsparse_indextype_i64))
        {
            calc = rocsparse::primitives::radix_sort_pairs_buffer_size<int64_t, int32_t>;
        }
        else if((A->row_type == rocsparse_indextype_i64)
                && (A->col_type == rocsparse_indextype_i64))
        {
            calc = rocsparse::primitives::radix_sort_pairs_buffer_size<int64_t, int64_t>;
        }

        RETURN_IF_ROCSPARSE_ERROR((calc) ? rocsparse_status_success
                                         : rocsparse_status_internal_error);

        size_t transpose_size{};
        RETURN_IF_ROCSPARSE_ERROR(
            calc(handle, A->nnz, startbit, endbit, &transpose_size, using_double_buffer));

        transpose_size += rocsparse::align_size(sizeof_J * A->nnz);
        transpose_size += rocsparse::align_size(sizeof_I * A->nnz);
        p_buffer_size_in_bytes[0] += transpose_size;
    }

    return rocsparse_status_success;
}

rocsparse_status rocsparse::csrsm_solve_buffer_size(rocsparse_handle            handle,
                                                    const int64_t               nrhs,
                                                    rocsparse_operation         op_A,
                                                    rocsparse_operation         op_B,
                                                    rocsparse_const_dnvec_descr alpha,
                                                    rocsparse_const_spmat_descr A,
                                                    rocsparse_const_dnmat_descr B,
                                                    size_t*          p_buffer_size_in_bytes,
                                                    rocsparse_error* p_error)
{
    ROCSPARSE_ROUTINE_TRACE;
    const int64_t M = A->rows;

    if(M == 0 || nrhs == 0)
    {
        p_buffer_size_in_bytes[0] = 0;
        return rocsparse_status_success;
    }

    const size_t sizeof_B      = rocsparse::datatype_sizeof(B->data_type);
    const auto   B_batch_count = B->batch_count;
    const auto   batch_count   = B_batch_count;
    if(nrhs == 1)
    {
        const int64_t          y_size         = M;
        const int64_t          y_batch_count  = B->batch_count;
        const int64_t          y_batch_stride = (B->batch_count == 1) ? 0 : M;
        const int64_t          y_inc          = 1;
        const auto             y_datatype     = B->data_type;
        _rocsparse_dnvec_descr y_st(
            y_batch_count, y_size, y_datatype, (const void*)0x4, (void*)0x4, y_inc, y_batch_stride);

        rocsparse_dnvec_descr y = &y_st;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_solve_buffer_size(
            handle, op_A, A, nullptr, y, p_buffer_size_in_bytes));

        p_buffer_size_in_bytes[0] += rocsparse::align_size(sizeof_B * M * B_batch_count);

        return rocsparse_status_success;
    }

    const int64_t B_batch_stride = B->batch_stride;
    RETURN_IF_ROCSPARSE_ERROR(((B_batch_stride == 0) && (B_batch_count > 1))
                                  ? rocsparse_status_invalid_value
                                  : rocsparse_status_success);

    const int64_t A_batch_stride = A->batch_stride;
    const int64_t A_batch_count  = (A_batch_stride == 0) ? 1 : A->batch_count;

    const size_t sizeof_A = rocsparse::datatype_sizeof(A->data_type);

    const int64_t transform_B
        = (op_B == rocsparse_operation_none && B->order == rocsparse_order_column);

    // max_nnz
    p_buffer_size_in_bytes[0] = 256;

    // Each thread block performs at most blockdim columns of the
    // rhs matrix. Therefore, the number of blocks depend on nrhs
    // and the blocksize.
    // Because of this, we might need a larger done_array compared
    // to csrsv.
    int32_t blockdim = 512;
    while(nrhs <= blockdim && blockdim > 32)
    {
        blockdim >>= 1;
    }

    blockdim <<= 1;
    const int32_t narrays = (nrhs - 1) / blockdim + 1;

    // int done_array
    p_buffer_size_in_bytes[0] += rocsparse::align_size(sizeof(int32_t) * M * narrays * batch_count);

    if(transform_B)
    {
        p_buffer_size_in_bytes[0] += rocsparse::align_size(sizeof_B * M * nrhs * B_batch_count);
    }

    if(op_A != rocsparse_operation_none)
    {
        p_buffer_size_in_bytes[0] += rocsparse::align_size(sizeof_A * A->nnz * A_batch_count);
    }

    return rocsparse_status_success;
}

rocsparse_status rocsparse::csrsm_buffer_size(rocsparse_handle            handle,
                                              const int64_t               nrhs,
                                              rocsparse_operation         op_A,
                                              rocsparse_operation         op_B,
                                              rocsparse_const_dnvec_descr alpha,
                                              rocsparse_const_spmat_descr A,
                                              rocsparse_const_dnmat_descr B,
                                              size_t*                     p_buffer_size_in_bytes,
                                              rocsparse_error*            p_error)
{

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_analysis_buffer_size(
        handle, nrhs, op_A, op_B, alpha, A, B, p_buffer_size_in_bytes, p_error));
    size_t buffer_size_in_bytes = std::numeric_limits<size_t>::max();

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_solve_buffer_size(
        handle, nrhs, op_A, op_B, alpha, A, B, &buffer_size_in_bytes, p_error));

    p_buffer_size_in_bytes[0] = std::max(p_buffer_size_in_bytes[0], buffer_size_in_bytes);

    return rocsparse_status_success;
}
