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

#include "handle.h"

namespace rocsparse
{
    typedef enum rocsparse_coomv_alg_
    {
        rocsparse_coomv_alg_default = 0,
        rocsparse_coomv_alg_segmented,
        rocsparse_coomv_alg_atomic
    } rocsparse_coomv_alg;

    template <typename I, typename A>
    rocsparse_status coomv_analysis_template(rocsparse_handle          handle,
                                             rocsparse_operation       trans,
                                             rocsparse_coomv_alg       alg,
                                             I                         m,
                                             I                         n,
                                             int64_t                   nnz,
                                             const rocsparse_mat_descr descr,
                                             const A*                  coo_val,
                                             const I*                  coo_row_ind,
                                             const I*                  coo_col_ind);

    template <typename T, typename I, typename A, typename X, typename Y>
    rocsparse_status coomv_template(rocsparse_handle          handle,
                                    rocsparse_operation       trans,
                                    rocsparse_coomv_alg       alg,
                                    I                         m,
                                    I                         n,
                                    int64_t                   nnz,
                                    const T*                  alpha_device_host,
                                    const rocsparse_mat_descr descr,
                                    const A*                  coo_val,
                                    const I*                  coo_row_ind,
                                    const I*                  coo_col_ind,
                                    const X*                  x,
                                    const T*                  beta_device_host,
                                    Y*                        y);
}
