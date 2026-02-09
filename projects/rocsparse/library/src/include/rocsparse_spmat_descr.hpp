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

struct _rocsparse_spmat_descr
{
    bool init{};

    mutable bool analysed{};

    int64_t rows{};
    int64_t cols{};
    int64_t nnz{};

    void* row_data{};
    void* col_data{};
    void* ind_data{};
    void* val_data{};

    const void* const_row_data{};
    const void* const_col_data{};
    const void* const_ind_data{};
    const void* const_val_data{};

    rocsparse_indextype row_type{};
    rocsparse_indextype col_type{};
    rocsparse_datatype  data_type{};

    rocsparse_index_base idx_base{};
    rocsparse_format     format{};

    rocsparse_mat_descr descr{};
    rocsparse_mat_info  info{};

    rocsparse_direction block_dir{};
    int64_t             block_dim{};
    int64_t             ell_cols{};
    int64_t             ell_width{};
    int64_t             sell_slice_size{};
    int64_t             sell_colval_size{};

    int64_t batch_count{};
    int64_t batch_stride{};
    int64_t offsets_batch_stride{};
    int64_t columns_values_batch_stride{};

    _rocsparse_spmat_descr() = default;
    _rocsparse_spmat_descr(rocsparse_format     format,
                           bool                 analysed,
                           int64_t              batch_count,
                           int64_t              m,
                           int64_t              n,
                           int64_t              nnz,
                           rocsparse_datatype   val_datatype,
                           const void*          const_val_data,
                           void*                val_data,
                           int64_t              val_stride,
                           rocsparse_indextype  row_indextype,
                           const void*          const_row_data,
                           void*                row_data,
                           int64_t              row_stride,
                           rocsparse_indextype  col_indextype,
                           const void*          const_col_data,
                           void*                col_data,
                           int64_t              col_stride,
                           rocsparse_index_base base,
                           rocsparse_mat_descr  descr,
                           rocsparse_mat_info   info);

    _rocsparse_spmat_descr(rocsparse_format     format,
                           bool                 analysed,
                           int64_t              batch_count,
                           int64_t              mb,
                           int64_t              nb,
                           int64_t              nnzb,
                           rocsparse_direction  block_dir,
                           int64_t              block_dim,
                           rocsparse_datatype   val_datatype,
                           const void*          const_val_data,
                           void*                val_data,
                           int64_t              val_stride,
                           rocsparse_indextype  row_indextype,
                           const void*          const_row_data,
                           void*                row_data,
                           int64_t              row_stride,
                           rocsparse_indextype  col_indextype,
                           const void*          const_col_data,
                           void*                col_data,
                           int64_t              col_stride,
                           rocsparse_index_base base,
                           rocsparse_mat_descr  descr,
                           rocsparse_mat_info   info);
};
