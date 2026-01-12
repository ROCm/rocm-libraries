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
#include "rocsparse_idvec_descr.hpp"

struct _rocsparse_spattern_descr
{
protected:
    rocsparse_mat_descr mat_descr{};

    rocsparse_format      format{};
    int64_t               rows{};
    int64_t               cols{};
    rocsparse_idvec_descr row_data{};
    rocsparse_idvec_descr col_data{};
    int64_t               nnz{};
    int64_t               ell_width{};
    int64_t               ell_cols{};
    bool                  own_data{};

public:
    ~_rocsparse_spattern_descr();
    _rocsparse_spattern_descr() {}

    _rocsparse_spattern_descr(rocsparse_format      format_,
                              int64_t               rows_,
                              int64_t               cols_,
                              int64_t               nnz_,
                              rocsparse_idvec_descr row_data_,
                              rocsparse_idvec_descr col_data_,
                              rocsparse_mat_descr   mat_descr_);

    void                              set_own_data(bool value);
    int64_t                           get_rows() const;
    int64_t                           get_cols() const;
    int64_t                           get_nnz() const;
    void                              set_nnz(int64_t value);
    int64_t*                          get_pnnz();
    int64_t                           get_ell_width() const;
    void                              set_ell_width(int64_t value);
    int64_t                           get_ell_cols() const;
    void                              set_ell_cols(int64_t value);
    rocsparse_format                  get_format() const;
    rocsparse_idvec_descr             get_row_data();
    rocsparse_idvec_descr             get_col_data();
    rocsparse_const_idvec_descr       get_row_data() const;
    rocsparse_const_idvec_descr       get_col_data() const;
    rocsparse_mat_descr               get_mat_descr();
    const _rocsparse_mat_descr*       get_mat_descr() const;
    int64_t                           get_batch_count() const;
    void                              set_row_data(rocsparse_idvec_descr data);
    void                              set_col_data(rocsparse_idvec_descr data);
    static _rocsparse_spattern_descr* create_csr(int64_t               rows,
                                                 int64_t               cols,
                                                 int64_t               nnz,
                                                 rocsparse_idvec_descr row_data,
                                                 rocsparse_idvec_descr col_data);
    static _rocsparse_spattern_descr* create_csc(int64_t               rows,
                                                 int64_t               cols,
                                                 int64_t               nnz,
                                                 rocsparse_idvec_descr row_data,
                                                 rocsparse_idvec_descr col_data);

    static _rocsparse_spattern_descr* create_sell(int64_t               rows,
                                                  int64_t               cols,
                                                  int64_t               nnz,
                                                  rocsparse_idvec_descr row_data,
                                                  rocsparse_idvec_descr col_data);

    static _rocsparse_spattern_descr* create_coo_aos(int64_t               rows,
                                                     int64_t               cols,
                                                     int64_t               nnz,
                                                     rocsparse_idvec_descr row_data,
                                                     rocsparse_idvec_descr col_data);

    static _rocsparse_spattern_descr* create_coo(int64_t               rows,
                                                 int64_t               cols,
                                                 int64_t               nnz,
                                                 rocsparse_idvec_descr row_data,
                                                 rocsparse_idvec_descr col_data);

    static _rocsparse_spattern_descr*
        create_ell(int64_t rows, int64_t cols, int64_t width, rocsparse_idvec_descr col_data);

    static _rocsparse_spattern_descr*
        create_bell(int64_t rows, int64_t cols, int64_t width, rocsparse_idvec_descr col_data);

    static _rocsparse_spattern_descr* create_bsr(int64_t               rows,
                                                 int64_t               cols,
                                                 int64_t               nnz,
                                                 rocsparse_idvec_descr row_data,
                                                 rocsparse_idvec_descr col_data);
};
