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
#include "rocsparse_mat_descr.hpp"
#include "rocsparse_mat_info.hpp"

struct _rocsparse_spattern_descr
{
private:
    _rocsparse_idvec_descr m_row_data{};
    _rocsparse_idvec_descr m_col_data{};
    _rocsparse_mat_descr   m_mat_descr{};
    _rocsparse_mat_info    m_mat_info{};

    rocsparse_idvec_descr m_p_row_data{&m_row_data};
    rocsparse_idvec_descr m_p_col_data{&m_col_data};
    rocsparse_mat_descr   m_p_mat_descr{&m_mat_descr};
    rocsparse_mat_info    m_p_mat_info{&m_mat_info};

protected:
    rocsparse_format    m_format{};
    int64_t             m_rows{};
    int64_t             m_cols{};
    int64_t             m_nnz{};
    int64_t             m_ell_width{};
    int64_t             m_ell_cols{};
    bool                m_own_data{};
    rocsparse_direction m_block_dir{rocsparse_direction_column};
    int64_t             m_block_dim{1};

    int64_t m_sell_slice_size{};
    int64_t m_sell_colval_size{};
    int64_t m_batch_count{1};

public:
    _rocsparse_spattern_descr()  = default;
    ~_rocsparse_spattern_descr() = default;

    rocsparse_status destroy(hipStream_t stream);
    rocsparse_status validate();

    void define(rocsparse_format      format,
                int64_t               rows,
                int64_t               cols,
                int64_t               nnz,
                rocsparse_idvec_descr row_data,
                rocsparse_idvec_descr col_data,
                rocsparse_mat_descr   mat_descr);

    void define_csr(int64_t               rows,
                    int64_t               cols,
                    int64_t               nnz,
                    rocsparse_idvec_descr row_data,
                    rocsparse_idvec_descr col_data,
                    rocsparse_mat_descr   mat_descr,
                    rocsparse_mat_info    mat_info);

    void define_csc(int64_t               rows,
                    int64_t               cols,
                    int64_t               nnz,
                    rocsparse_idvec_descr row_data,
                    rocsparse_idvec_descr col_data,
                    rocsparse_mat_descr   mat_descr,
                    rocsparse_mat_info    mat_info);

    void define_sell(int64_t               rows,
                     int64_t               cols,
                     int64_t               nnz,
                     int64_t               sell_slice_size,
                     int64_t               sell_colval_size,
                     rocsparse_idvec_descr row_data,
                     rocsparse_idvec_descr col_data,
                     rocsparse_mat_descr   mat_descr,
                     rocsparse_mat_info    mat_info);

    void define_coo_aos(int64_t               rows,
                        int64_t               cols,
                        int64_t               nnz,
                        rocsparse_idvec_descr row_data,
                        rocsparse_idvec_descr col_data,
                        rocsparse_mat_descr   mat_descr,
                        rocsparse_mat_info    mat_info);

    void define_coo(int64_t               rows,
                    int64_t               cols,
                    int64_t               nnz,
                    rocsparse_idvec_descr row_data,
                    rocsparse_idvec_descr col_data,
                    rocsparse_mat_descr   mat_descr,
                    rocsparse_mat_info    mat_info);

    void define_ell(int64_t               rows,
                    int64_t               cols,
                    int64_t               width,
                    rocsparse_idvec_descr col_data,
                    rocsparse_mat_descr   mat_descr,
                    rocsparse_mat_info    mat_info);

    void define_bell(int64_t               rows,
                     int64_t               cols,
                     int64_t               width,
                     rocsparse_direction   block_dir,
                     int64_t               block_dim,
                     rocsparse_idvec_descr col_data,
                     rocsparse_mat_descr   mat_descr,
                     rocsparse_mat_info    mat_info);

    void define_bsr(int64_t               rows,
                    int64_t               cols,
                    int64_t               nnz,
                    rocsparse_direction   block_dir,
                    int64_t               block_dim,
                    rocsparse_idvec_descr row_data,
                    rocsparse_idvec_descr col_data,
                    rocsparse_mat_descr   mat_descr,
                    rocsparse_mat_info    mat_info);

    void set_own_data(bool value);
    //
    rocsparse_mat_info get_info() const;
    void               set_info(rocsparse_mat_info);
    //
    int64_t get_batch_count() const;
    void    set_batch_count(int64_t value);
    //
    int64_t get_sell_slice_size() const;
    void    set_sell_slice_size(int64_t value);
    //
    void    set_sell_colval_size(int64_t value);
    int64_t get_sell_colval_size() const;
    //
    rocsparse_direction get_block_dir() const;
    void                set_block_dir(rocsparse_direction);
    //
    int64_t get_block_dim() const;
    void    set_block_dim(int64_t);
    //
    int64_t get_rows() const;
    void    set_rows(int64_t);
    //
    int64_t get_cols() const;
    void    set_cols(int64_t);
    //
    int64_t  get_nnz() const;
    void     set_nnz(int64_t value);
    int64_t* get_pnnz();
    //
    //
    int64_t get_ell_width() const;
    void    set_ell_width(int64_t value);
    //
    int64_t get_ell_cols() const;
    void    set_ell_cols(int64_t value);
    //
    rocsparse_format get_format() const;
    void             set_format(rocsparse_format);
    //
    rocsparse_const_idvec_descr get_row_data() const;
    rocsparse_idvec_descr       get_row_data();
    void                        set_row_data(rocsparse_idvec_descr data);
    //
    rocsparse_const_idvec_descr get_col_data() const;
    rocsparse_idvec_descr       get_col_data();
    void                        set_col_data(rocsparse_idvec_descr data);
    //
    rocsparse_mat_descr get_mat_descr();
    rocsparse_mat_descr get_mat_descr() const;
    //
};
