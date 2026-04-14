/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_handle.hpp"

namespace rocsparse
{
    template <typename T, typename I, typename J>
    rocsparse_status bsrxmv_template_dispatch(rocsparse_handle          handle,
                                              rocsparse_direction       dir,
                                              rocsparse_operation       trans,
                                              J                         size_of_mask,
                                              J                         mb,
                                              J                         nb,
                                              I                         nnzb,
                                              const T*                  alpha_device_host,
                                              const rocsparse_mat_descr descr,
                                              const T*                  bsr_val,
                                              const J*                  bsr_mask_ptr,
                                              const I*                  bsr_row_ptr,
                                              const I*                  bsr_end_ptr,
                                              const J*                  bsr_col_ind,
                                              J                         block_dim,
                                              const T*                  x,
                                              const T*                  beta_device_host,
                                              T*                        y);

    template <typename T, typename I, typename J>
    rocsparse_status bsrxmv_template(rocsparse_handle          handle,
                                     rocsparse_direction       dir,
                                     rocsparse_operation       trans,
                                     J                         size_of_mask,
                                     J                         mb,
                                     J                         nb,
                                     I                         nnzb,
                                     const T*                  alpha_device_host,
                                     const rocsparse_mat_descr descr,
                                     const T*                  bsr_val,
                                     const J*                  bsr_mask_ptr,
                                     const I*                  bsr_row_ptr,
                                     const I*                  bsr_end_ptr,
                                     const J*                  bsr_col_ind,
                                     J                         block_dim,
                                     const T*                  x,
                                     const T*                  beta_device_host,
                                     T*                        y);
}
