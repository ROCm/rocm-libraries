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

#include "handle.h"

namespace rocsparse
{
    template <typename T>
    rocsparse_status gebsr2gebsr_buffer_size_template(rocsparse_handle          handle,
                                                      rocsparse_direction       dir,
                                                      rocsparse_int             mb,
                                                      rocsparse_int             nb,
                                                      rocsparse_int             nnzb,
                                                      const rocsparse_mat_descr descr_A,
                                                      const T*                  bsr_val_A,
                                                      const rocsparse_int*      bsr_row_ptr_A,
                                                      const rocsparse_int*      bsr_col_ind_A,
                                                      rocsparse_int             row_block_dim_A,
                                                      rocsparse_int             col_block_dim_A,
                                                      rocsparse_int             row_block_dim_C,
                                                      rocsparse_int             col_block_dim_C,
                                                      size_t*                   buffer_size);

    template <typename T>
    rocsparse_status gebsr2gebsr_template(rocsparse_handle          handle,
                                          rocsparse_direction       dir,
                                          rocsparse_int             mb,
                                          rocsparse_int             nb,
                                          rocsparse_int             nnzb,
                                          const rocsparse_mat_descr descr_A,
                                          const T*                  bsr_val_A,
                                          const rocsparse_int*      bsr_row_ptr_A,
                                          const rocsparse_int*      bsr_col_ind_A,
                                          rocsparse_int             row_block_dim_A,
                                          rocsparse_int             col_block_dim_A,
                                          const rocsparse_mat_descr descr_C,
                                          T*                        bsr_val_C,
                                          rocsparse_int*            bsr_row_ptr_C,
                                          rocsparse_int*            bsr_col_ind_C,
                                          rocsparse_int             row_block_dim_C,
                                          rocsparse_int             col_block_dim_C,
                                          void*                     temp_buffer);
}
