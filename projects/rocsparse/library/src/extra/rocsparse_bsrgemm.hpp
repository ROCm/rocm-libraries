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

namespace rocsparse
{
    template <typename I, typename J, typename T>
    rocsparse_status bsrgemm_buffer_size_template(rocsparse_handle          handle,
                                                  rocsparse_direction       dir,
                                                  rocsparse_operation       trans_A,
                                                  rocsparse_operation       trans_B,
                                                  J                         mb,
                                                  J                         nb,
                                                  J                         kb,
                                                  J                         block_dim,
                                                  const T*                  alpha,
                                                  const rocsparse_mat_descr descr_A,
                                                  I                         nnzb_A,
                                                  const I*                  bsr_row_ptr_A,
                                                  const J*                  bsr_col_ind_A,
                                                  const rocsparse_mat_descr descr_B,
                                                  I                         nnzb_B,
                                                  const I*                  bsr_row_ptr_B,
                                                  const J*                  bsr_col_ind_B,
                                                  const T*                  beta,
                                                  const rocsparse_mat_descr descr_D,
                                                  I                         nnzb_D,
                                                  const I*                  bsr_row_ptr_D,
                                                  const J*                  bsr_col_ind_D,
                                                  rocsparse_mat_info        info_C,
                                                  size_t*                   buffer_size);

    template <typename I, typename J>
    rocsparse_status bsrgemm_nnzb_template(rocsparse_handle          handle,
                                           rocsparse_direction       dir,
                                           rocsparse_operation       trans_A,
                                           rocsparse_operation       trans_B,
                                           J                         mb,
                                           J                         nb,
                                           J                         kb,
                                           J                         block_dim,
                                           const rocsparse_mat_descr descr_A,
                                           I                         nnzb_A,
                                           const I*                  bsr_row_ptr_A,
                                           const J*                  bsr_col_ind_A,
                                           const rocsparse_mat_descr descr_B,
                                           I                         nnzb_B,
                                           const I*                  bsr_row_ptr_B,
                                           const J*                  bsr_col_ind_B,
                                           const rocsparse_mat_descr descr_D,
                                           I                         nnzb_D,
                                           const I*                  bsr_row_ptr_D,
                                           const J*                  bsr_col_ind_D,
                                           const rocsparse_mat_descr descr_C,
                                           I*                        bsr_row_ptr_C,
                                           I*                        nnzb_C,
                                           const rocsparse_mat_info  info_C,
                                           void*                     temp_buffer);

    template <typename I, typename J, typename T>
    rocsparse_status bsrgemm_template(rocsparse_handle          handle,
                                      rocsparse_direction       dir,
                                      rocsparse_operation       trans_A,
                                      rocsparse_operation       trans_B,
                                      J                         mb,
                                      J                         nb,
                                      J                         kb,
                                      J                         block_dim,
                                      const T*                  alpha,
                                      const rocsparse_mat_descr descr_A,
                                      I                         nnzb_A,
                                      const T*                  bsr_val_A,
                                      const I*                  bsr_row_ptr_A,
                                      const J*                  bsr_col_ind_A,
                                      const rocsparse_mat_descr descr_B,
                                      I                         nnzb_B,
                                      const T*                  bsr_val_B,
                                      const I*                  bsr_row_ptr_B,
                                      const J*                  bsr_col_ind_B,
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
}
