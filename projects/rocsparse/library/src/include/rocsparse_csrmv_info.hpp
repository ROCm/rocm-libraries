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

#pragma once

#include "rocsparse_adaptive_info.hpp"
#include "rocsparse_csrmv_alg.hpp"
#include "rocsparse_csrmv_nnzsplit_info.hpp"
#include "rocsparse_indextype_utils.hpp"
#include "rocsparse_lrb_info.hpp"
#include "rocsparse_mat_descr.hpp"

/********************************************************************************
 * \brief rocsparse_csrmv_info is a structure holding the rocsparse csrmv info
 * data gathered during csrmv_analysis.
 *******************************************************************************/
typedef struct _rocsparse_csrmv_info
{
    _rocsparse_adaptive_info       adaptive{};
    _rocsparse_lrb_info            lrb{};
    _rocsparse_csrmv_nnzsplit_info nnzsplit{};

    // some data to verify correct execution
    rocsparse_operation         trans = rocsparse_operation_none;
    int64_t                     m{};
    int64_t                     n{};
    int64_t                     nnz{};
    int64_t                     max_rows{};
    const _rocsparse_mat_descr* descr{};
    const void*                 csr_row_ptr{};
    const void*                 csr_col_ind{};

    rocsparse_indextype index_type_I = deprecated_rocsparse_indextype_u16;
    rocsparse_indextype index_type_J = deprecated_rocsparse_indextype_u16;

    // Concrete algorithm stored when analysis was asked for csrmv_alg_default.
    // Left as csrmv_alg_default when analysis was asked for an explicit
    // algorithm, so a later default compute does not reuse that choice.
    rocsparse::csrmv_alg alg = rocsparse::csrmv_alg_default;

    // Longest row, when analysis has already walked the host row pointers.
    // -1 until then. nnz-split reads this from the descriptor. The value stays
    // on the mat info so a later routine can reuse it.
    int64_t max_nnz_per_row = -1;

    _rocsparse_csrmv_info();
    ~_rocsparse_csrmv_info();

    void clear();

    // Stored algorithm when the caller passes csrmv_alg_default and operation
    // is the one analysis ran for. Otherwise the static default map. operation
    // is the caller's operation, compared with this->trans.
    rocsparse::csrmv_alg alg_for(rocsparse_operation operation) const;

    // Algorithm for a csrmv_alg_default compute. Uses the choice analysis stored
    // for this operation, or the static map when csrmv_info is null. Row-split
    // and an empty matrix leave no descriptor, and compute without analysis has
    // none either.
    static rocsparse::csrmv_alg choose_alg(const _rocsparse_csrmv_info* csrmv_info,
                                           rocsparse_operation          operation);

} * rocsparse_csrmv_info, *rocsparse_cscmv_info;

namespace rocsparse
{
    /********************************************************************************
   * \brief Copy csrmv info.
   *******************************************************************************/
    rocsparse_status copy_csrmv_info(rocsparse_csrmv_info dest, const rocsparse_csrmv_info src);
}
