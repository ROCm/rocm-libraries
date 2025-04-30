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
    rocsparse_status csr2ell_strided_batched_quickreturn(rocsparse_handle          handle,
                                                         int64_t                   batch_count,
                                                         int64_t                   m,
                                                         const rocsparse_mat_descr csr_descr,
                                                         const void*               csr_val,
                                                         int64_t                   csr_val_stride,
                                                         const void*               csr_row_ptr,
                                                         const void*               csr_col_ind,
                                                         const rocsparse_mat_descr ell_descr,
                                                         int64_t                   ell_width,
                                                         void*                     ell_val,
                                                         int64_t                   ell_val_stride,
                                                         void*                     ell_col_ind);

    template <typename T, typename I, typename J>
    rocsparse_status csr2ell_strided_batched_core(rocsparse_handle          handle,
                                                  int64_t                   batch_count,
                                                  J                         m,
                                                  const rocsparse_mat_descr csr_descr,
                                                  const T*                  csr_val,
                                                  int64_t                   csr_val_stride,
                                                  const I*                  csr_row_ptr,
                                                  const J*                  csr_col_ind,
                                                  const rocsparse_mat_descr ell_descr,
                                                  J                         ell_width,
                                                  T*                        ell_val,
                                                  int64_t                   ell_val_stride,
                                                  J*                        ell_col_ind);

    template <typename... P>
    rocsparse_status csr2ell_strided_batched_template(P&&... p)
    {
        const rocsparse_status status = rocsparse::csr2ell_strided_batched_quickreturn(p...);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2ell_strided_batched_core(p...));
        return rocsparse_status_success;
    }
}
