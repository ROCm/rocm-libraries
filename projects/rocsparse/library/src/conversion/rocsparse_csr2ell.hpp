/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights Reserved.
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
    template <typename T, typename I, typename J>
    rocsparse_status csr2ell_core(rocsparse_handle          handle,
                                  J                         m,
                                  const rocsparse_mat_descr csr_descr,
                                  const T*                  csr_val,
                                  const I*                  csr_row_ptr,
                                  const J*                  csr_col_ind,
                                  const rocsparse_mat_descr ell_descr,
                                  J                         ell_width,
                                  T*                        ell_val,
                                  J*                        ell_col_ind);

    rocsparse_status csr2ell_quickreturn(rocsparse_handle          handle,
                                         int64_t                   m,
                                         const rocsparse_mat_descr csr_descr,
                                         const void*               csr_val,
                                         const void*               csr_row_ptr,
                                         const void*               csr_col_ind,
                                         const rocsparse_mat_descr ell_descr,
                                         int64_t                   ell_width,
                                         void*                     ell_val,
                                         void*                     ell_col_ind);

    template <typename... P>
    rocsparse_status csr2ell_template(P&&... p)
    {
        const rocsparse_status status = rocsparse::csr2ell_quickreturn(p...);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2ell_core(p...));
        return rocsparse_status_success;
    }

    template <typename I, typename J>
    rocsparse_status csr2ell_width_core(rocsparse_handle          handle,
                                        J                         m,
                                        const rocsparse_mat_descr csr_descr,
                                        const I*                  csr_row_ptr,
                                        const rocsparse_mat_descr ell_descr,
                                        J*                        ell_width);
    template <typename J>
    rocsparse_status csr2ell_width_quickreturn(rocsparse_handle          handle,
                                               int64_t                   m,
                                               const rocsparse_mat_descr csr_descr,
                                               const void*               csr_row_ptr,
                                               const rocsparse_mat_descr ell_descr,
                                               J*                        ell_width);

    template <typename... P>
    rocsparse_status csr2ell_width_template(P&&... p)
    {
        const rocsparse_status status = rocsparse::csr2ell_width_quickreturn(p...);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2ell_width_core(p...));
        return rocsparse_status_success;
    }
}
