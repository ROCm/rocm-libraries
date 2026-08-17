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

#include "rocsparse_control.hpp"
#include "rocsparse_csrsv.hpp"
#include "rocsparse_primitives.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::csrsv_analysis_buffer_size(rocsparse_handle            handle,
                                                       rocsparse_operation         trans,
                                                       rocsparse_const_spmat_descr A,
                                                       size_t*                     buffer_size)
{
    ROCSPARSE_CHECKARG(2, A, A->descr == nullptr, rocsparse_status_invalid_pointer);
    ROCSPARSE_CHECKARG(2,
                       A,
                       (A->descr->type != rocsparse_matrix_type_general
                        && A->descr->type != rocsparse_matrix_type_triangular),
                       rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(2,
                       A,
                       (A->descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    // Quick return if possible
    if(A->rows == 0)
    {
        *buffer_size = 0;
        return rocsparse_status_success;
    }

    const size_t sizeof_I = rocsparse::indextype_sizeof(A->row_type);
    const size_t sizeof_J = rocsparse::indextype_sizeof(A->col_type);

    // rocsparse_int max_nnz
    *buffer_size = 256;

    // rocsparse_int done_array[m]
    *buffer_size += ((sizeof(int32_t) * A->rows - 1) / 256 + 1) * 256;

    // rocsparse_int workspace
    *buffer_size += ((sizeof_J * A->rows - 1) / 256 + 1) * 256;

    // rocsparse_int workspace2
    *buffer_size += ((sizeof(int32_t) * A->rows - 1) / 256 + 1) * 256;

    uint32_t startbit = 0;
    uint32_t endbit   = rocsparse::clz(A->rows);

    size_t rocprim_size = 0;

    auto calculate_rocprim_size
        = rocsparse::find_radix_sort_pairs_buffer_size(rocsparse_indextype_i32, A->col_type);

    RETURN_IF_ROCSPARSE_ERROR(
        (calculate_rocprim_size(handle, A->rows, startbit, endbit, &rocprim_size, true)));

    // rocprim buffer
    *buffer_size += rocprim_size;

    // On transposed case, we might need more temporary storage for transposing
    if(trans == rocsparse_operation_transpose || trans == rocsparse_operation_conjugate_transpose)
    {
        size_t transpose_size;
        // Determine rocprim buffer size
        auto calculate_size
            = rocsparse::find_radix_sort_pairs_buffer_size(A->col_type, A->row_type);

        RETURN_IF_ROCSPARSE_ERROR(
            (calculate_size(handle, A->nnz, startbit, endbit, &transpose_size, true)));
        // rocPRIM does not support in-place sorting, so we need an additional buffer
        // rocsparse_int max_nnz
        transpose_size += 256 + ((sizeof(int32_t) * A->rows - 1) / 256 + 1) * 256;
        transpose_size += ((sizeof_J * A->nnz - 1) / 256 + 1) * 256;
        transpose_size += ((sizeof_I * A->nnz - 1) / 256 + 1) * 256;
        *buffer_size = rocsparse::max(*buffer_size, transpose_size);
    }

    return rocsparse_status_success;
}

rocsparse_status rocsparse::csrsv_solve_buffer_size(rocsparse_handle            handle,
                                                    rocsparse_operation         op,
                                                    rocsparse_const_spmat_descr A,
                                                    rocsparse_const_dnvec_descr x,
                                                    rocsparse_const_dnvec_descr y,
                                                    size_t*                     buffer_size)
{
    ROCSPARSE_CHECKARG(2, A, A->descr == nullptr, rocsparse_status_invalid_pointer);
    ROCSPARSE_CHECKARG(2,
                       A,
                       (A->descr->type != rocsparse_matrix_type_general
                        && A->descr->type != rocsparse_matrix_type_triangular),
                       rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(2,
                       A,
                       (A->descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    // Quick return if possible
    if(A->rows == 0)
    {
        buffer_size[0] = 0;
        return rocsparse_status_success;
    }

    // rocsparse_int max_nnz
    buffer_size[0] = 256;
    // rocsparse_int done_array[m]
    const int64_t batch_count = (y) ? y->batch_count : A->batch_count;

    buffer_size[0] += ((sizeof(int32_t) * A->rows * batch_count - 1) / 256 + 1) * 256;

    // On transposed case, we might need more temporary storage for transposing
    if(op != rocsparse_operation_none)
    {
        const size_t sizeof_T       = rocsparse::datatype_sizeof(A->data_type);
        const size_t transpose_size = ((sizeof_T * A->batch_count * A->nnz - 1) / 256 + 1) * 256;
        buffer_size[0] += transpose_size;
    }
    return rocsparse_status_success;
}
