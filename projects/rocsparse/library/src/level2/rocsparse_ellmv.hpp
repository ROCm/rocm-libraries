/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "handle.h"

namespace rocsparse
{
    template <typename T, typename I, typename A, typename X, typename Y>
    rocsparse_status ellmv_template(rocsparse_handle          handle,
                                    rocsparse_operation       trans,
                                    int64_t                   m,
                                    int64_t                   n,
                                    const void*               alpha,
                                    const rocsparse_mat_descr descr,
                                    const void*               ell_val,
                                    const void*               ell_col_ind,
                                    int64_t                   ell_width,
                                    const void*               x,
                                    const void*               beta,
                                    void*                     y);

    rocsparse_status ellmv(rocsparse_handle          handle,
                           rocsparse_operation       trans,
                           int64_t                   m,
                           int64_t                   n,
                           rocsparse_datatype        alpha_device_host_datatype,
                           const void*               alpha_device_host,
                           const rocsparse_mat_descr descr,
                           rocsparse_datatype        ell_val_datatype,
                           const void*               ell_val,
                           rocsparse_indextype       ell_col_ind_indextype,
                           const void*               ell_col_ind,
                           int64_t                   ell_width,
                           rocsparse_datatype        x_datatype,
                           const void*               x,
                           rocsparse_datatype        beta_device_host_datatype,
                           const void*               beta_device_host,
                           rocsparse_datatype        y_datatype,
                           void*                     y);

}
