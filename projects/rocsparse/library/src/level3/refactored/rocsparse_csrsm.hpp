/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_control.hpp"
#include "rocsparse_csrsm_info.hpp"

namespace rocsparse
{
    rocsparse_status spmat_transpose_update_values(rocsparse_handle            handle,
                                                   rocsparse_spmat_descr       target,
                                                   rocsparse_const_spmat_descr source,
                                                   rocsparse::trm_info_t*      trm_info);

    rocsparse_status csrsm_zero_pivot(rocsparse_handle     handle,
                                      rocsparse_csrsm_info info,
                                      rocsparse_indextype  indextype,
                                      void*                position);

    rocsparse_status csrsm_analysis_buffer_size(rocsparse_handle            handle,
                                                const int64_t               nrhs,
                                                rocsparse_operation         op_A,
                                                rocsparse_operation         op_B,
                                                rocsparse_const_dnvec_descr alpha,
                                                rocsparse_const_spmat_descr A,
                                                rocsparse_const_dnmat_descr B,
                                                size_t*                     p_buffer_size_in_bytes,
                                                rocsparse_error*            p_error);

    rocsparse_status csrsm_solve_buffer_size(rocsparse_handle            handle,
                                             const int64_t               nrhs,
                                             rocsparse_operation         op_A,
                                             rocsparse_operation         op_B,
                                             rocsparse_const_dnvec_descr alpha,
                                             rocsparse_const_spmat_descr A,
                                             rocsparse_const_dnmat_descr B,
                                             size_t*                     p_buffer_size_in_bytes,
                                             rocsparse_error*            p_error);

    rocsparse_status csrsm_buffer_size(rocsparse_handle            handle,
                                       const int64_t               nrhs,
                                       rocsparse_operation         trans_A,
                                       rocsparse_operation         trans_B,
                                       rocsparse_const_dnvec_descr alpha,
                                       rocsparse_const_spmat_descr A,
                                       rocsparse_const_dnmat_descr B,
                                       size_t*                     p_buffer_size,
                                       rocsparse_error*            p_error);

    rocsparse_status csrsm_analysis(rocsparse_handle            handle,
                                    const int64_t               nrhs,
                                    rocsparse_operation         trans_A,
                                    rocsparse_operation         trans_B,
                                    rocsparse_const_dnvec_descr alpha,
                                    rocsparse_const_spmat_descr A,
                                    rocsparse_const_dnmat_descr B,
                                    rocsparse_analysis_policy   analysis,
                                    rocsparse_csrsm_info*       p_csrsm_info,
                                    size_t                      buffer_size_in_bytes,
                                    void*                       buffer,
                                    rocsparse_error*            p_error);

    rocsparse_status csrsm_compute(rocsparse_handle            handle,
                                   const int64_t               nrhs,
                                   rocsparse_operation         A_op,
                                   bool                        A_load_conjugate,
                                   rocsparse_operation         B_op,
                                   rocsparse_const_dnvec_descr alpha,
                                   rocsparse_const_spmat_descr A,
                                   rocsparse_dnmat_descr       B,
                                   rocsparse_csrsm_info        csrsm_info,
                                   size_t                      buffer_eize_in_bytes,
                                   void*                       buffer,
                                   rocsparse_error*            p_error);

    rocsparse_status csrsm_compute(rocsparse_handle            handle,
                                   const int64_t               nrhs,
                                   rocsparse_operation         op_A,
                                   rocsparse_operation         op_B,
                                   rocsparse_const_dnvec_descr alpha,
                                   rocsparse_const_spmat_descr A,
                                   rocsparse_dnmat_descr       B,
                                   rocsparse_csrsm_info        csrsm_info,
                                   size_t                      buffer_size_in_bytes,
                                   void*                       buffer,
                                   rocsparse_error*            p_error);

}
