/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#pragma once

#pragma once

#include "rocsparse_control.hpp"
#include "rocsparse_csrsv_info.hpp"
#include "rocsparse_datatype_utils.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_indextype_utils.hpp"
#include "rocsparse_logging.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{

    rocsparse_status csrsv_zero_pivot(rocsparse_handle     handle,
                                      rocsparse_csrsv_info info,
                                      rocsparse_indextype  indextype,
                                      void*                position);

    rocsparse_status csrsv_analysis_buffer_size(rocsparse_handle            handle,
                                                rocsparse_operation         trans,
                                                rocsparse_const_spmat_descr A,
                                                size_t*                     buffer_size);

    rocsparse_status csrsv_solve_buffer_size(rocsparse_handle            handle,
                                             rocsparse_operation         trans,
                                             rocsparse_const_spmat_descr A,
                                             rocsparse_const_dnvec_descr x,
                                             rocsparse_const_dnvec_descr y,
                                             size_t*                     buffer_size);

    inline rocsparse_status csrsv_solve_buffer_size(rocsparse_handle            handle,
                                                    rocsparse_operation         trans,
                                                    rocsparse_const_spmat_descr A,
                                                    size_t*                     buffer_size)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            csrsv_solve_buffer_size(handle, trans, A, nullptr, nullptr, buffer_size));
        return rocsparse_status_success;
    }

    rocsparse_status csrsv_analysis(rocsparse_handle            handle,
                                    rocsparse_operation         trans,
                                    rocsparse_const_spmat_descr A,
                                    rocsparse_analysis_policy   analysis_policy,
                                    rocsparse_solve_policy      solve_policy,
                                    rocsparse_csrsv_info*       p_csrsv_info,
                                    void*                       temp_buffer);

    rocsparse_status csrsv_solve(rocsparse_handle            handle,
                                 rocsparse_operation         trans,
                                 rocsparse_datatype          alpha_datatype,
                                 const void*                 alpha,
                                 int64_t                     alpha_stride,
                                 rocsparse_const_spmat_descr A,
                                 rocsparse_const_dnvec_descr x,
                                 rocsparse_dnvec_descr       y,
                                 rocsparse_solve_policy      policy,
                                 rocsparse_csrsv_info        csrsv_info,
                                 void*                       temp_buffer);

    rocsparse_status launch_csrsv_analysis_kernel(rocsparse_handle    handle,
                                                  rocsparse_operation trans,
                                                  int64_t             m,
                                                  rocsparse_indextype csr_row_ptr_indextype,
                                                  const void* __restrict__ csr_row_ptr,
                                                  rocsparse_indextype csr_col_ind_indextype,
                                                  const void* __restrict__ csr_col_ind,
                                                  rocsparse_indextype csr_diag_ind_indextype,
                                                  void* __restrict__ csr_diag_ind,
                                                  int32_t* __restrict__ done_array,
                                                  void* __restrict__ max_nnz,
                                                  void* __restrict__ zero_pivot,
                                                  rocsparse_index_base idx_base,
                                                  rocsparse_diag_type  diag_type,
                                                  rocsparse_fill_mode  mode);

}
