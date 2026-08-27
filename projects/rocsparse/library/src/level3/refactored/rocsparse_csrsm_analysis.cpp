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
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::csrsm_analysis(rocsparse_handle            handle,
                                           const int64_t               nrhs,
                                           rocsparse_operation         op_A,
                                           rocsparse_operation         op_B,
                                           rocsparse_const_dnvec_descr alpha,
                                           rocsparse_const_spmat_descr A,
                                           rocsparse_const_dnmat_descr B,
                                           rocsparse_analysis_policy   analysis,
                                           rocsparse_csrsm_info*       p_csrsm_info,
                                           size_t                      buffer_size_in_bytes,
                                           void*                       buffer,
                                           rocsparse_error*            p_error)
{
    ROCSPARSE_ROUTINE_TRACE;
    const int64_t M = A->rows;

    if(M == 0 || nrhs == 0)
    {
        return rocsparse_status_success;
    }

    if(nrhs == 1)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_analysis(
            handle, op_A, A, analysis, rocsparse_solve_policy_auto, p_csrsm_info, buffer));

        return rocsparse_status_success;
    }

    auto csrsm_info = p_csrsm_info[0];
    auto info       = A->info;
    auto descr      = A->descr;
    // Differentiate the analysis policies
    if(analysis == rocsparse_analysis_policy_reuse)
    {
        //
        //
        //
        rocsparse::trm_info_t* p = nullptr;

        p = (p) ? p : info->get_csrsm_info(op_A, descr->fill_mode);

        if((descr->fill_mode == rocsparse_fill_mode_lower) && (op_A == rocsparse_operation_none))
        {
            p = (p) ? p : info->get_csrilu0_info(op_A, descr->fill_mode);
            p = (p) ? p : info->get_csric0_info(op_A, descr->fill_mode);
        }

        p = (p) ? p : info->get_csrsv_info(op_A, descr->fill_mode);
        if(p)
        {
            info->set_csrsm_info(op_A, descr->fill_mode, p);
            return rocsparse_status_success;
        }
    }

    if(csrsm_info == nullptr)
    {
        csrsm_info      = new _rocsparse_csrsm_info();
        p_csrsm_info[0] = csrsm_info;
    }

    // Perform analysis
    RETURN_IF_ROCSPARSE_ERROR(csrsm_info->recreate(handle,
                                                   op_A,
                                                   M,
                                                   A->nnz,
                                                   A->descr,
                                                   A->data_type,
                                                   A->const_val_data,
                                                   A->row_type,
                                                   A->const_row_data,
                                                   A->col_type,
                                                   A->const_col_data,
                                                   buffer));

    return rocsparse_status_success;
}
