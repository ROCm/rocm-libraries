/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse-types.h"

namespace rocsparse
{

    typedef rocsparse_status (*bsric0_kernel_general_launch_t)(rocsparse_handle    handle,
                                                               rocsparse_direction dir,
                                                               int64_t             batch_count,
                                                               int64_t             mb,
                                                               const void*         bsr_row_ptr,
                                                               const void*         bsr_col_ind,
                                                               void*               bsr_val,
                                                               int64_t             bsr_val_stride,
                                                               const void*         bsr_diag_ind,
                                                               int64_t             bsr_dim,
                                                               int32_t*            done_array,
                                                               int64_t     done_array_stride,
                                                               const void* map,
                                                               void*       zero_pivot,
                                                               int64_t     zero_pivot_stride,
                                                               rocsparse_index_base idx_base);

    bsric0_kernel_general_launch_t
        find_bsric0_strided_batched_kernel_general_launch(uint32_t            blocksize,
                                                          uint32_t            wfsize,
                                                          bool                sleep,
                                                          rocsparse_datatype  t_type,
                                                          rocsparse_indextype i_type,
                                                          rocsparse_indextype j_type);
}
