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
#include "rocsparse_diagonal_solve.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::cscsv_diagonal_solve(rocsparse_handle            handle,
                                                 rocsparse_operation         trans,
                                                 rocsparse_diagonal_modifier modifier,
                                                 const void*                 alpha,
                                                 rocsparse_const_spmat_descr A,
                                                 rocsparse_const_dnvec_descr x,
                                                 rocsparse_dnvec_descr       y,
                                                 rocsparse_cscsv_info        info)
{
    ROCSPARSE_ROUTINE_TRACE;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::cscsv_checkarg_matrix_properties(A));
#if defined(ROCSPARSE_WITH_SPTRSV_CSC)
    if(trans == rocsparse_operation_none)
    {
        const int64_t batch_count = y->batch_count;

        const rocsparse::trm_info_t* trm_info
            = info->get(rocsparse_operation_none, A->descr->fill_mode);
        if(trm_info == nullptr)
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_invalid_pointer,
                "the CSC analysis was not performed for this fill mode");
        }

        // The CSC analysis indexes the diagonal directly in the CSC values.
        rocsparse::spdiag_view diag{};
        diag.offset_type     = trm_info->get_offset_indextype();
        diag.diag_ind        = trm_info->get_diag_ind();
        diag.transposed_perm = nullptr;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::cscsv_seed_zero_pivot(
            handle, A->row_type, A->descr->diag_type, batch_count, info));

        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::diagonal_solve(handle,
                                      rocsparse_operation_none,
                                      modifier,
                                      alpha,
                                      A,
                                      diag,
                                      static_cast<int64_t>(1),
                                      x->const_values,
                                      x->inc,
                                      static_cast<int64_t>(0),
                                      x->batch_stride,
                                      y->values,
                                      y->inc,
                                      static_cast<int64_t>(0),
                                      y->batch_stride,
                                      batch_count,
                                      false,
                                      info->get_singularity_numeric_exact()->get_position(),
                                      1,
                                      handle->pointer_mode == rocsparse_pointer_mode_host));
        return rocsparse_status_success;
    }
#endif
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::diagonal_solve_csc(handle,
                                                            trans,
                                                            modifier,
                                                            alpha,
                                                            A,
                                                            info->get_csrsv_info(),
                                                            static_cast<int64_t>(1),
                                                            x->const_values,
                                                            x->inc,
                                                            static_cast<int64_t>(0),
                                                            x->batch_stride,
                                                            y->values,
                                                            y->inc,
                                                            static_cast<int64_t>(0),
                                                            y->batch_stride,
                                                            y->batch_count,
                                                            false));
    return rocsparse_status_success;
}
