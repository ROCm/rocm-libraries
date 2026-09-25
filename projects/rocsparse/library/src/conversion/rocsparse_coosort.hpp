/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights Reserved.
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
    typedef enum rocsparse_coosort_alg_
    {
        rocsparse_coosort_alg_default = 0
    } rocsparse_coosort_alg;

    template <typename J>
    rocsparse_status coosort_buffer_size_template(rocsparse_handle handle,
                                                  int64_t          m,
                                                  int64_t          n,
                                                  int64_t          nnz,
                                                  const void*      coo_row_ind,
                                                  const void*      coo_col_ind,
                                                  size_t*          buffer_size);
    template <typename J>
    rocsparse_status coosort_by_row_template(rocsparse_handle handle,
                                             int64_t          m,
                                             int64_t          n,
                                             int64_t          nnz,
                                             void*            coo_row_ind,
                                             void*            coo_col_ind,
                                             void*            perm,
                                             void*            temp_buffer);

    template <typename J>
    rocsparse_status coosort_by_column_template(rocsparse_handle handle,
                                                int64_t          m,
                                                int64_t          n,
                                                int64_t          nnz,
                                                void*            coo_row_ind,
                                                void*            coo_col_ind,
                                                void*            perm,
                                                void*            temp_buffer);

    rocsparse_status coosort_buffer_size(rocsparse_handle      handle,
                                         rocsparse_coosort_alg alg,
                                         rocsparse_direction   dir,
                                         int64_t               m,
                                         int64_t               n,
                                         int64_t               nnz,
                                         rocsparse_indextype   coo_row_indextype_A,
                                         const void*           coo_row_ind_A,
                                         rocsparse_indextype   coo_col_indextype_A,
                                         const void*           coo_col_ind_A,
                                         rocsparse_datatype    coo_val_datatype_A,
                                         const void*           coo_val_A,
                                         rocsparse_indextype   coo_row_indextype_B,
                                         const void*           coo_row_ind_B,
                                         rocsparse_indextype   coo_col_indextype_B,
                                         const void*           coo_col_ind_B,
                                         rocsparse_datatype    coo_val_datatype_B,
                                         const void*           coo_val_B,
                                         size_t*               buffer_size);

    // Sorts the row indices, column indices and values of the COO matrix A into the COO
    // matrix B. Each output array may either alias its input array, in which case it is
    // sorted in place, or not overlap it at all. The index types and data types of A and B
    // must be identical. Each of the batch_count matrices is sorted independently, with the
    // batch strides given in number of elements.
    rocsparse_status coosort(rocsparse_handle      handle,
                             rocsparse_coosort_alg alg,
                             rocsparse_direction   dir,
                             int64_t               m,
                             int64_t               n,
                             int64_t               nnz,
                             int64_t               batch_count_A,
                             int64_t               batch_stride_A,
                             rocsparse_indextype   coo_row_indextype_A,
                             const void*           coo_row_ind_A,
                             rocsparse_indextype   coo_col_indextype_A,
                             const void*           coo_col_ind_A,
                             rocsparse_datatype    coo_val_datatype_A,
                             const void*           coo_val_A,
                             int64_t               batch_count_B,
                             int64_t               batch_stride_B,
                             rocsparse_indextype   coo_row_indextype_B,
                             void*                 coo_row_ind_B,
                             rocsparse_indextype   coo_col_indextype_B,
                             void*                 coo_col_ind_B,
                             rocsparse_datatype    coo_val_datatype_B,
                             void*                 coo_val_B,
                             void*                 temp_buffer);
}
