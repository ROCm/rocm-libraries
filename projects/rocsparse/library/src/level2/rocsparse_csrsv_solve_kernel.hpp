/*! \file */
/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

    typedef rocsparse_status (*csrsv_launch_kernel_t)(rocsparse_handle handle,
                                                      int64_t          batch_count,
                                                      int64_t          m,
                                                      const void*      alpha_,
                                                      int64_t          alpha_stride,
                                                      const void* __restrict__ csr_row_ptr,
                                                      const void* __restrict__ csr_col_ind,
                                                      const void* __restrict__ csr_val,
                                                      int64_t csr_val_inc,
                                                      int64_t csr_val_stride,
                                                      const void* __restrict__ x,
                                                      int64_t x_inc,
                                                      int64_t x_stride,
                                                      void* __restrict__ y,
                                                      int64_t y_inc,
                                                      int64_t y_stride,
                                                      int32_t* __restrict__ done_array,
                                                      const void* __restrict__ map,
                                                      int64_t offset,
                                                      void* __restrict__ zero_pivot,
                                                      int64_t              zero_pivot_stride,
                                                      rocsparse_index_base idx_base,
                                                      rocsparse_fill_mode  fill_mode,
                                                      rocsparse_diag_type  diag_type,
                                                      bool                 is_host_mode);

    rocsparse_status csrsv_launch_kernel_find(rocsparse::csrsv_launch_kernel_t* spmm_function_,
                                              uint32_t                          A,
                                              uint32_t                          B,
                                              bool                              C,
                                              rocsparse_indextype               i_type_,
                                              rocsparse_indextype               j_type_,
                                              rocsparse_datatype                a_type_);

}
