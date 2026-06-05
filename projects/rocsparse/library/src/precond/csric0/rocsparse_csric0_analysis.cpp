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

#include "../level2/rocsparse_csrsv.hpp"
#include "rocsparse_csric0.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::csric0_analysis(rocsparse_handle          handle,
                                            rocsparse_spmat_descr     A,
                                            rocsparse_analysis_policy analysis,
                                            rocsparse_csric0_info*    p_csric0_info,
                                            void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(A->rows == 0)
    {
        //
        // Quick return
        //
        return rocsparse_status_success;
    }

    auto info        = A->info;
    auto csric0_info = p_csric0_info[0];

    if(analysis == rocsparse_analysis_policy_reuse)
    {
        auto trm = csric0_info->get(rocsparse_operation_none, rocsparse_fill_mode_lower);
        if(trm == nullptr)
        {
            trm = (trm != nullptr)
                      ? trm
                      : info->get_csrilu0_info(rocsparse_operation_none, rocsparse_fill_mode_lower);

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
                info->set_csric0_info(rocsparse_operation_none, rocsparse_fill_mode_lower, trm);
            }
        }

        if(trm != nullptr)
        {
            return rocsparse_status_success;
        }
    }

    if(csric0_info == nullptr)
    {
        csric0_info      = new _rocsparse_csric0_info();
        p_csric0_info[0] = csric0_info;
    }

    RETURN_IF_ROCSPARSE_ERROR(csric0_info->recreate(rocsparse_operation_none,
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
