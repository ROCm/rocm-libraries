/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_csrmv.hpp"

namespace rocsparse
{
    template <typename I, typename J, typename A>
    rocsparse_status cscmv_analysis_template(rocsparse_handle          handle,
                                             rocsparse_operation       trans,
                                             rocsparse::csrmv_alg      alg,
                                             J                         m,
                                             J                         n,
                                             I                         nnz,
                                             const rocsparse_mat_descr descr,
                                             const A*                  csc_val,
                                             const I*                  csc_col_ptr,
                                             const J*                  csc_row_ind,
                                             rocsparse_mat_info        info);

    template <typename T, typename I, typename J, typename A, typename X, typename Y>
    rocsparse_status cscmv_template(rocsparse_handle          handle,
                                    rocsparse_operation       trans,
                                    rocsparse::csrmv_alg      alg,
                                    J                         m,
                                    J                         n,
                                    I                         nnz,
                                    const T*                  alpha,
                                    const rocsparse_mat_descr descr,
                                    const A*                  csc_val,
                                    const I*                  csc_col_ptr,
                                    const J*                  csc_row_ind,
                                    rocsparse_mat_info        info,
                                    const X*                  x,
                                    const T*                  beta,
                                    Y*                        y);
}
