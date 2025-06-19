/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_bsrgemm_mult.hpp"
#include "rocsparse_csrgemm_mult.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::bsrgemm_mult_buffer_size_quickreturn(rocsparse_handle          handle,
                                                                 rocsparse_operation       trans_A,
                                                                 rocsparse_operation       trans_B,
                                                                 int64_t                   mb,
                                                                 int64_t                   nb,
                                                                 int64_t                   kb,
                                                                 const void*               alpha,
                                                                 const rocsparse_mat_descr descr_A,
                                                                 int64_t                   nnzb_A,
                                                                 const void* bsr_row_ptr_A,
                                                                 const void* bsr_col_ind_A,
                                                                 const rocsparse_mat_descr descr_B,
                                                                 int64_t                   nnzb_B,
                                                                 const void*        bsr_row_ptr_B,
                                                                 const void*        bsr_col_ind_B,
                                                                 rocsparse_mat_info info_C,
                                                                 size_t*            buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    const rocsparse_status status = rocsparse::csrgemm_mult_buffer_size_quickreturn(handle,
                                                                                    trans_A,
                                                                                    trans_B,
                                                                                    mb,
                                                                                    nb,
                                                                                    kb,
                                                                                    alpha,
                                                                                    descr_A,
                                                                                    nnzb_A,
                                                                                    bsr_row_ptr_A,
                                                                                    bsr_col_ind_A,
                                                                                    descr_B,
                                                                                    nnzb_B,
                                                                                    bsr_row_ptr_B,
                                                                                    bsr_col_ind_B,
                                                                                    info_C,
                                                                                    buffer_size);

    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    return rocsparse_status_continue;
}
