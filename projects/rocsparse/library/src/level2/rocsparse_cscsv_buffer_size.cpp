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

uint32_t rocsparse::cscsv_num_lanes(int64_t max_nnz, uint32_t wavefront_size)
{
    uint32_t num_lanes = 2;
    while(static_cast<int64_t>(num_lanes) < max_nnz && num_lanes < wavefront_size)
    {
        num_lanes *= 2;
    }
    return num_lanes;
}

size_t rocsparse::cscsv_column_solve_buffer_size(int64_t             m,
                                                 int64_t             batch_count,
                                                 rocsparse_indextype j_type,
                                                 rocsparse_datatype  t_type)
{
    return rocsparse::align_size(sizeof(int32_t) * batch_count)
           + rocsparse::align_size(rocsparse::indextype_sizeof(j_type) * m * batch_count)
           + rocsparse::align_size(rocsparse::datatype_sizeof(t_type) * m * batch_count);
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
        return rocsparse_status_success;
    }
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::cscsv_transposed_analysis_buffer_size(handle, trans, A, buffer_size));
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
        buffer_size[0] = rocsparse::cscsv_column_solve_buffer_size(
            A->cols, y->batch_count, A->row_type, A->data_type);
        return rocsparse_status_success;
    }
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::cscsv_transposed_solve_buffer_size(handle, trans, A, x, y, buffer_size));
    return rocsparse_status_success;
}
