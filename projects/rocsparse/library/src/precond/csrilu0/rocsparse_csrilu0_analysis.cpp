/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "../level2/rocsparse_csrsv.hpp"
#include "rocsparse_csrilu0.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::csrilu0_analysis(rocsparse_handle          handle,
                                             rocsparse_spmat_descr     A,
                                             rocsparse_analysis_policy analysis,
                                             rocsparse_solve_policy    solve,
                                             rocsparse_csrilu0_info*   p_csrilu0_info,
                                             void*                     temp_buffer)
{

    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, A);
    ROCSPARSE_CHECKARG_ENUM(2, analysis);
    ROCSPARSE_CHECKARG_ENUM(3, solve);
    ROCSPARSE_CHECKARG_POINTER(4, p_csrilu0_info);

    if(A->rows == 0 || A->batch_count == 0)
    {
        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_POINTER(5, temp_buffer);

    ROCSPARSE_CHECKARG(1, A, (A->descr == nullptr), rocsparse_status_invalid_pointer);
    ROCSPARSE_CHECKARG(
        1, A, (A->descr->type != rocsparse_matrix_type_general), rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(1,
                       A,
                       (A->descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    auto info         = A->info;
    auto csrilu0_info = p_csrilu0_info[0];

    if(analysis == rocsparse_analysis_policy_reuse)
    {
        auto trm = csrilu0_info->get(rocsparse_operation_none, rocsparse_fill_mode_lower);
        if(trm == nullptr)
        {
            trm = (trm != nullptr)
                      ? trm
                      : info->get_csric0_info(rocsparse_operation_none, rocsparse_fill_mode_lower);
            trm = (trm != nullptr)
                      ? trm
                      : info->get_csrsv_info(rocsparse_operation_none, rocsparse_fill_mode_lower);
            trm = (trm != nullptr)
                      ? trm
                      : info->get_csrsm_info(rocsparse_operation_none, rocsparse_fill_mode_lower);
            trm = (trm != nullptr) ? trm
                                   : info->get_csrsv_info(rocsparse_operation_transpose,
                                                          rocsparse_fill_mode_upper);
            trm = (trm != nullptr) ? trm
                                   : info->get_csrsm_info(rocsparse_operation_transpose,
                                                          rocsparse_fill_mode_upper);

            if(trm != nullptr)
            {
                info->set_csrilu0_info(rocsparse_operation_none, rocsparse_fill_mode_lower, trm);
            }
        }

        if(trm != nullptr)
        {
            return rocsparse_status_success;
        }
    }

    if(csrilu0_info == nullptr)
    {
        csrilu0_info      = new _rocsparse_csrilu0_info();
        p_csrilu0_info[0] = csrilu0_info;
    }

    // Perform analysis
    RETURN_IF_ROCSPARSE_ERROR(csrilu0_info->recreate(rocsparse_operation_none,
                                                     rocsparse_fill_mode_lower,
                                                     handle,
                                                     rocsparse_operation_none,
                                                     A->rows,
                                                     A->nnz,
                                                     A->descr,
                                                     A->data_type,
                                                     A->const_val_data,
                                                     A->row_type,
                                                     A->const_row_data,
                                                     A->col_type,
                                                     A->const_col_data,
                                                     temp_buffer));

    return rocsparse_status_success;
}
