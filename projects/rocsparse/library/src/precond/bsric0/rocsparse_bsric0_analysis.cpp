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

#include "rocsparse_assign_async.hpp"
#include "rocsparse_bsric0.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::bsric0_analysis(rocsparse_handle          handle,
                                            rocsparse_spmat_descr     A,
                                            rocsparse_analysis_policy analysis,
                                            rocsparse_solve_policy    solve,
                                            rocsparse_bsric0_info*    p_bsric0_info,
                                            void*                     buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, A);
    ROCSPARSE_CHECKARG_ENUM(2, analysis);
    ROCSPARSE_CHECKARG_ENUM(3, solve);
    ROCSPARSE_CHECKARG_POINTER(4, p_bsric0_info);

    const bool quick_return = (A->rows == 0 || A->batch_count == 0);
    ROCSPARSE_CHECKARG(5,
                       buffer,
                       (buffer == nullptr) && (quick_return == false),
                       rocsparse_status_invalid_pointer);

    if(quick_return)
    {
        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG(1, A, (A->descr == nullptr), rocsparse_status_invalid_pointer);
    ROCSPARSE_CHECKARG(
        1, A, (A->descr->type != rocsparse_matrix_type_general), rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG(1,
                       A,
                       (A->descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);
    auto info        = A->info;
    auto bsric0_info = p_bsric0_info[0];
    switch(analysis)
    {
    case rocsparse_analysis_policy_reuse:
    {
        auto trm = bsric0_info->get(rocsparse_operation_none, rocsparse_fill_mode_lower);
        trm      = (trm != nullptr)
                       ? trm
                       : info->get_bsric0_info(rocsparse_operation_none, rocsparse_fill_mode_lower);

        trm = (trm != nullptr)
                  ? trm
                  : info->get_bsrsv_info(rocsparse_operation_none, rocsparse_fill_mode_lower);

        if(trm != nullptr)
        {
            bsric0_info->set(rocsparse_operation_none, rocsparse_fill_mode_lower, trm);
            return rocsparse_status_success;
        }
        break;
    }
    case rocsparse_analysis_policy_force:
    {
        break;
    }
    }

    if(bsric0_info == nullptr)
    {
        bsric0_info      = new _rocsparse_bsric0_info();
        p_bsric0_info[0] = bsric0_info;
    }

    // Perform analysis
    RETURN_IF_ROCSPARSE_ERROR(bsric0_info->recreate(rocsparse_operation_none,
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
                                                    buffer));

    return rocsparse_status_success;
}
