/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights Reserved.
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
    template <typename I>
    rocsparse_status bsr2csr_quickreturn(rocsparse_handle          handle,
                                         rocsparse_direction       dir,
                                         int64_t                   mb,
                                         int64_t                   nb,
                                         const rocsparse_mat_descr bsr_descr,
                                         const void*               bsr_val,
                                         const void*               bsr_row_ptr,
                                         const void*               bsr_col_ind,
                                         int64_t                   block_dim,
                                         const rocsparse_mat_descr csr_descr,
                                         void*                     csr_val,
                                         I*                        csr_row_ptr,
                                         void*                     csr_col_ind);

    template <typename T, typename I, typename J>
    rocsparse_status bsr2csr_core(rocsparse_handle          handle,
                                  rocsparse_direction       direction,
                                  J                         mb,
                                  J                         nb,
                                  const rocsparse_mat_descr bsr_descr,
                                  const T*                  bsr_val,
                                  const I*                  bsr_row_ptr,
                                  const J*                  bsr_col_ind,
                                  J                         block_dim,
                                  const rocsparse_mat_descr csr_descr,
                                  T*                        csr_val,
                                  I*                        csr_row_ptr,
                                  J*                        csr_col_ind);

    template <typename... P>
    static rocsparse_status bsr2csr_template(P&&... p)
    {

        const rocsparse_status status = rocsparse::bsr2csr_quickreturn(p...);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsr2csr_core(p...));
        return rocsparse_status_success;
    }
}
