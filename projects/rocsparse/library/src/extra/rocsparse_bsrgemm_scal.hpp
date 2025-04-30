/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "control.h"

namespace rocsparse
{
    rocsparse_status bsrgemm_scal_buffer_size_quickreturn(rocsparse_handle          handle,
                                                          int64_t                   mb,
                                                          int64_t                   nb,
                                                          const void*               beta,
                                                          const rocsparse_mat_descr descr_D,
                                                          int64_t                   nnzb_D,
                                                          const void*               bsr_row_ptr_D,
                                                          const void*               bsr_col_ind_D,
                                                          rocsparse_mat_info        info_C,
                                                          size_t*                   buffer_size);

    rocsparse_status bsrgemm_scal_quickreturn(rocsparse_handle          handle,
                                              int64_t                   mb,
                                              int64_t                   nb,
                                              int64_t                   block_dim,
                                              const void*               beta,
                                              const rocsparse_mat_descr descr_D,
                                              int64_t                   nnzb_D,
                                              const void*               bsr_val_D,
                                              const void*               bsr_row_ptr_D,
                                              const void*               bsr_col_ind_D,
                                              const rocsparse_mat_descr descr_C,
                                              void*                     bsr_val_C,
                                              const void*               bsr_row_ptr_C,
                                              void*                     bsr_col_ind_C,
                                              const rocsparse_mat_info  info_C,
                                              void*                     temp_buffer);

    template <typename I, typename J, typename T>
    rocsparse_status bsrgemm_scal_core(rocsparse_handle          handle,
                                       J                         mb,
                                       J                         nb,
                                       J                         block_dim,
                                       const T*                  beta,
                                       const rocsparse_mat_descr descr_D,
                                       I                         nnzb_D,
                                       const T*                  bsr_val_D,
                                       const I*                  bsr_row_ptr_D,
                                       const J*                  bsr_col_ind_D,
                                       const rocsparse_mat_descr descr_C,
                                       T*                        bsr_val_C,
                                       const I*                  bsr_row_ptr_C,
                                       J*                        bsr_col_ind_C,
                                       const rocsparse_mat_info  info_C,
                                       void*                     temp_buffer);

    template <typename... P>
    rocsparse_status bsrgemm_scal_template(P&&... p)
    {
        const rocsparse_status status = rocsparse::bsrgemm_scal_quickreturn(p...);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrgemm_scal_core(p...));
        return rocsparse_status_success;
    }
}
