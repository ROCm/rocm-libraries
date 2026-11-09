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

#include "rocsparse_cscsv.hpp"
#include "rocsparse_cscsv_kernel.hpp"
#include "rocsparse_utility.hpp"

uint32_t rocsparse::cscsv_num_lanes(int64_t min_lanes, uint32_t wavefront_size)
{
    uint32_t num_lanes = 2;
    while(static_cast<int64_t>(num_lanes) < min_lanes && num_lanes < wavefront_size)
    {
        num_lanes *= 2;
    }
    return num_lanes;
}

rocsparse_status rocsparse::cscsv_analysis_buffer_size(rocsparse_handle            handle,
                                                       rocsparse_operation         trans,
                                                       rocsparse_const_spmat_descr A,
                                                       size_t*                     buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;
    if(trans == rocsparse_operation_none)
    {
        buffer_size[0] = 0;
        if(A->cols > 0)
        {
            rocsparse::cscsv_build_tasks_buffer_size_launch_t launch;
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::cscsv_build_tasks_buffer_size_launch_find(&launch, A->col_type));
            RETURN_IF_ROCSPARSE_ERROR(launch(handle, A->cols, buffer_size));
        }
        return rocsparse_status_success;
    }
    _rocsparse_mat_descr   descr_csr;
    _rocsparse_spmat_descr mat_csr;
    rocsparse::build_csr_from_csc(*A, mat_csr, descr_csr);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_analysis_buffer_size(
        handle, rocsparse::cscsv_operation_to_csr(trans), &mat_csr, buffer_size));
    return rocsparse_status_success;
}

rocsparse_status rocsparse::cscsv_solve_buffer_size(rocsparse_handle            handle,
                                                    rocsparse_operation         trans,
                                                    rocsparse_const_spmat_descr A,
                                                    rocsparse_const_dnvec_descr x,
                                                    rocsparse_const_dnvec_descr y,
                                                    size_t*                     buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;
    if(trans == rocsparse_operation_none)
    {
        // Ticket, remaining dependencies and partial sums of each batch instance.
        const int64_t m           = A->cols;
        const int64_t batch_count = y->batch_count;
        buffer_size[0]
            = rocsparse::align_size(sizeof(int32_t) * batch_count)
              + rocsparse::align_size(rocsparse::indextype_sizeof(A->row_type) * m * batch_count)
              + rocsparse::align_size(rocsparse::datatype_sizeof(A->data_type) * m * batch_count);
        return rocsparse_status_success;
    }
    _rocsparse_mat_descr   descr_csr;
    _rocsparse_spmat_descr mat_csr;
    rocsparse::build_csr_from_csc(*A, mat_csr, descr_csr);

#ifdef ROCSPARSE_WITH_TRSM_REFACTORING
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_solve_buffer_size(
        handle, rocsparse::cscsv_operation_to_csr(trans), &mat_csr, x, y, buffer_size));
#else
    // conjugate_transpose runs as a non-transposed CSR solve with on-the-fly
    // conjugation, which needs the same extra buffer space as transposition.
    const rocsparse_operation trans_csr = (trans == rocsparse_operation_conjugate_transpose)
                                              ? rocsparse_operation_transpose
                                              : rocsparse::cscsv_operation_to_csr(trans);

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::csrsv_solve_buffer_size(handle, trans_csr, &mat_csr, x, y, buffer_size));
#endif
    return rocsparse_status_success;
}
