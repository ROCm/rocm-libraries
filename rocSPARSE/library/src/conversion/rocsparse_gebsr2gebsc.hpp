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
    template <typename T>
    rocsparse_status gebsr2gebsc_template(rocsparse_handle     handle,
                                          rocsparse_int        mb,
                                          rocsparse_int        nb,
                                          rocsparse_int        nnzb,
                                          const T*             bsr_val,
                                          const rocsparse_int* bsr_row_ptr,
                                          const rocsparse_int* bsr_col_ind,
                                          rocsparse_int        row_block_dim,
                                          rocsparse_int        col_block_dim,
                                          T*                   bsc_val,
                                          rocsparse_int*       bsc_row_ind,
                                          rocsparse_int*       bsc_col_ptr,
                                          rocsparse_action     copy_values,
                                          rocsparse_index_base idx_base,
                                          void*                temp_buffer);
}
