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
#include "rocsparse_spattern_descr.hpp"

struct _rocsparse_spmat_descr
{
protected:
    rocsparse_spattern_descr m_spattern{};
    rocsparse_dnvec_descr    m_values{};
    rocsparse_mat_info       m_mat_info{};
    bool                     init{};
    mutable bool             analysed{};
    rocsparse_direction      m_block_dir;
    int64_t                  m_block_dim;
    bool                     m_own_values{};
    bool                     m_own_spattern{};

public:
    ~_rocsparse_spmat_descr();
    void set_own_spattern(bool value);
    void set_own_values(bool value);

    static rocsparse_spmat_descr create_sell(int64_t              rows,
                                             int64_t              cols,
                                             int64_t              nnz,
                                             int64_t              sell_slice_size,
                                             int64_t              sell_colval_size,
                                             const void*          const_row_data,
                                             void*                row_data,
                                             const void*          const_col_data,
                                             void*                col_data,
                                             const void*          const_val_data,
                                             void*                val_data,
                                             rocsparse_indextype  row_type,
                                             rocsparse_indextype  col_type,
                                             rocsparse_index_base idx_base,
                                             rocsparse_datatype   val_type);

    static rocsparse_spmat_descr create_coo(int64_t              rows,
                                            int64_t              cols,
                                            int64_t              nnz,
                                            const void*          const_row_data,
                                            void*                row_data,
                                            const void*          const_col_data,
                                            void*                col_data,
                                            const void*          const_val,
                                            void*                val,
                                            rocsparse_indextype  idx_type,
                                            rocsparse_index_base idx_base,
                                            rocsparse_datatype   val_type);

    static rocsparse_spmat_descr create_coo_aos(int64_t     rows,
                                                int64_t     cols,
                                                int64_t     nnz,
                                                const void* const_ind_data,
                                                void*       ind_data,

                                                const void*          const_val,
                                                void*                val,
                                                rocsparse_indextype  idx_type,
                                                rocsparse_index_base idx_base,
                                                rocsparse_datatype   val_type);

    static rocsparse_spmat_descr create_bsr(int64_t             mb,
                                            int64_t             nb,
                                            int64_t             nnzb,
                                            rocsparse_direction block_dir,
                                            int64_t             block_dim,

                                            const void*          const_row_data,
                                            void*                row_data,
                                            const void*          const_col_data,
                                            void*                col_data,
                                            const void*          const_val,
                                            void*                val,
                                            rocsparse_indextype  row_type,
                                            rocsparse_indextype  col_type,
                                            rocsparse_index_base idx_base,
                                            rocsparse_datatype   val_type);

    static rocsparse_spmat_descr create_csr(int64_t              rows,
                                            int64_t              cols,
                                            int64_t              nnz,
                                            const void*          const_row_data,
                                            void*                row_data,
                                            const void*          const_col_data,
                                            void*                col_data,
                                            const void*          const_val,
                                            void*                val,
                                            rocsparse_indextype  row_type,
                                            rocsparse_indextype  col_type,
                                            rocsparse_index_base idx_base,
                                            rocsparse_datatype   val_type);

    static rocsparse_spmat_descr create_csc(int64_t              rows,
                                            int64_t              cols,
                                            int64_t              nnz,
                                            const void*          const_row_data,
                                            void*                row_data,
                                            const void*          const_col_data,
                                            void*                col_data,
                                            const void*          const_val,
                                            void*                val,
                                            rocsparse_indextype  row_type,
                                            rocsparse_indextype  col_type,
                                            rocsparse_index_base idx_base,
                                            rocsparse_datatype   val_type);

    static rocsparse_spmat_descr create_bell(int64_t              rows,
                                             int64_t              cols,
                                             rocsparse_direction  block_dir,
                                             int64_t              block_dim,
                                             const void*          const_ind_data,
                                             void*                ind_data,
                                             const void*          const_val,
                                             void*                val,
                                             int64_t              width,
                                             rocsparse_indextype  idx_type,
                                             rocsparse_index_base idx_base,
                                             rocsparse_datatype   val_type);

    static rocsparse_spmat_descr create_ell(int64_t              rows,
                                            int64_t              cols,
                                            const void*          const_ind_data,
                                            void*                ind_data,
                                            const void*          const_val,
                                            void*                val,
                                            int64_t              width,
                                            rocsparse_indextype  idx_type,
                                            rocsparse_index_base idx_base,
                                            rocsparse_datatype   val_type);

    _rocsparse_spmat_descr() = default;

    _rocsparse_spmat_descr(_rocsparse_spattern_descr* spattern,
                           rocsparse_dnvec_descr      values,
                           rocsparse_mat_info         info);

    _rocsparse_spmat_descr(_rocsparse_spattern_descr* spattern,
                           rocsparse_direction        block_dir,
                           int64_t                    block_dim,
                           rocsparse_dnvec_descr      values,
                           rocsparse_mat_info         info);

    const _rocsparse_spattern_descr* get_spattern() const;
    rocsparse_spattern_descr         get_spattern();
    void                             set_spattern(rocsparse_spattern_descr value);

    rocsparse_dnvec_descr         get_values();
    void                          set_values(rocsparse_dnvec_descr value);
    const _rocsparse_dnvec_descr* get_values() const;

    rocsparse_mat_info  get_info() const;
    rocsparse_mat_descr get_descr() const;

public:
    bool get_analysed() const;
    void set_analysed(bool value) const;

    bool get_init() const;
    void set_init(bool value);

    int64_t get_rows() const;

    int64_t get_cols() const;

    int64_t  get_nnz() const;
    int64_t* get_pnnz();

    void* get_row_data() const;
    void* get_col_data() const;
    void* get_ind_data() const;
    void* get_val_data() const;

    const void*          get_const_row_data() const;
    const void*          get_const_col_data() const;
    const void*          get_const_ind_data() const;
    const void*          get_const_val_data() const;
    rocsparse_index_base get_idx_base() const;
    rocsparse_format     get_format() const;
    rocsparse_direction  get_block_dir() const;
    int64_t              get_block_dim() const;

    int64_t get_ell_cols() const;
    int64_t get_ell_width() const;
    int64_t get_batch_count() const;
    int64_t get_row_batch_dist() const;
    int64_t get_col_batch_dist() const;
    int64_t get_val_batch_dist() const;

    rocsparse_indextype get_row_type() const;
    rocsparse_indextype get_col_type() const;
    rocsparse_datatype  get_data_type() const;

    void set_row_data(void* value);
    void set_col_data(void* value);
    void set_ind_data(void* value);
    void set_val_data(void* value);
    void set_const_row_data(const void* value);
    void set_const_col_data(const void* value);
    void set_const_ind_data(const void* value);
    void set_const_val_data(const void* value);
    void set_nnz(const int64_t value);
    void set_batch_count(int64_t value);

    void set_row_batch_dist(int64_t value);
    void set_col_batch_dist(int64_t value);
    void set_val_batch_dist(int64_t value);
    void set_ell_width(int64_t value);

    int64_t sell_slice_size{};
    int64_t sell_colval_size{};

#if 0
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
#endif
};
