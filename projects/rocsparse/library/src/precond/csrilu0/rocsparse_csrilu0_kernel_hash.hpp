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

    typedef rocsparse_status (*launch_csrilu0_kernel_hash_t)(rocsparse_handle handle,
                                                             int64_t          batch_count,
                                                             int64_t          m,
                                                             const void* __restrict__ csr_row_ptr,
                                                             const void* __restrict__ csr_col_ind,
                                                             void* __restrict__ csr_val,
                                                             int64_t csr_val_stride,
                                                             const void* __restrict__ csr_diag_ind,
                                                             int32_t* __restrict__ done,
                                                             int64_t done_stride,
                                                             const void* __restrict__ map,
                                                             void* __restrict__ zero_pivot,
                                                             int64_t zero_pivot_stride,
                                                             void* __restrict__ singular_pivot,
                                                             int64_t singular_pivot_stride,
                                                             double  tol,
                                                             rocsparse_index_base idx_base,
                                                             int                  enable_boost,
                                                             size_t               size_boost_tol,
                                                             const void*          boost_tol,
                                                             const void*          boost_val_);

    launch_csrilu0_kernel_hash_t find_launch_csrilu0_kernel_hash(uint32_t            blocksize,
                                                                 uint32_t            wfsize,
                                                                 uint32_t            hash,
                                                                 rocsparse_datatype  t_type,
                                                                 rocsparse_indextype i_type,
                                                                 rocsparse_indextype j_type);

}
