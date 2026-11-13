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
#include "rocsparse-types.h"
#include "rocsparse_transposed_sparsity_pattern.hpp"

rocsparse_status
    rocsparse_triangular_transpose_buffer_size(int64_t             m,
                                               int64_t             nnz,
                                               rocsparse_indextype csr_row_ptr_indextype,
                                               rocsparse_indextype csr_col_ind_indextype,
                                               size_t*             p_buffer_size);

// Builds the transposed sparsity pattern (row_ptr, col_ind and permutation) of a CSR matrix.
rocsparse_status
    rocsparse_triangular_transpose(rocsparse_handle                          handle,
                                   int64_t                                   m,
                                   int64_t                                   nnz,
                                   const rocsparse_mat_descr                 descr,
                                   rocsparse_indextype                       csr_row_ptr_indextype,
                                   const void*                               csr_row_ptr,
                                   rocsparse_indextype                       csr_col_ind_indextype,
                                   const void*                               csr_col_ind,
                                   rocsparse::transposed_sparsity_pattern_t* transposed,
                                   void*                                     temp_buffer);
