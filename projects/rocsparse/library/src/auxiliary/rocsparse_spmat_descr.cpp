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

#include "internal/auxiliary/rocsparse_idvec_descr.h"
#include "rocsparse_argdescr.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_datatype_utils.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_logging.hpp"
#include "rocsparse_spattern_descr.hpp"

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spmat_prop value_)
{
    switch(value_)
    {
    case rocsparse_spmat_prop_format:
    case rocsparse_spmat_prop_rows:
    case rocsparse_spmat_prop_cols:
    case rocsparse_spmat_prop_nnz:
    case rocsparse_spmat_prop_batch_count:
    case rocsparse_spmat_prop_block_dim:
    case rocsparse_spmat_prop_block_dir:
    {
        return false;
    }
    }
    return true;
}

void _rocsparse_spmat_descr::set_own_values(bool value)
{
    this->m_own_values = value;
}

void _rocsparse_spmat_descr::set_own_spattern(bool value)
{
    this->m_own_spattern = value;
}

_rocsparse_spmat_descr::~_rocsparse_spmat_descr()
{
    WARNING_IF_ROCSPARSE_ERROR(rocsparse_destroy_mat_info(this->m_mat_info));
    if(this->m_own_spattern && this->m_spattern == nullptr)
        delete this->m_spattern;
    if(this->m_own_values && this->m_values == nullptr)
        delete this->m_values;
    this->m_spattern = nullptr;
    this->m_values   = nullptr;
}

#if 1

rocsparse_spmat_descr _rocsparse_spmat_descr::create_coo(int64_t              rows,
                                                         int64_t              cols,
                                                         int64_t              nnz,
                                                         const void*          const_row_data,
                                                         void*                row_data,
                                                         const void*          const_col_data,
                                                         void*                col_data,
                                                         const void*          const_val_data,
                                                         void*                val_data,
                                                         rocsparse_indextype  idx_type,
                                                         rocsparse_index_base idx_base,
                                                         rocsparse_datatype   val_type)
{
    rocsparse_idvec_descr    row_idvec{};
    rocsparse_idvec_descr    col_idvec{};
    rocsparse_dnvec_descr    val_dnvec{};
    rocsparse_spattern_descr spattern{};
    rocsparse_spmat_descr    spmat{};
    try
    {
        row_idvec
            = new _rocsparse_idvec_descr(idx_type, idx_base, nnz, 1, const_row_data, row_data);
        col_idvec
            = new _rocsparse_idvec_descr(idx_type, idx_base, nnz, 1, const_col_data, col_data);
        val_dnvec = new _rocsparse_dnvec_descr(val_type, nnz, 1, const_val_data, val_data);

        spattern = _rocsparse_spattern_descr::create_coo(rows, cols, nnz, row_idvec, col_idvec);

        spattern->set_own_data(true);

        spmat = new _rocsparse_spmat_descr(spattern, val_dnvec, new _rocsparse_mat_info());
        spmat->set_own_spattern(true);
        spmat->set_own_values(true);
    }
    catch(...)
    {
        if(row_idvec != nullptr)
            delete row_idvec;
        if(col_idvec != nullptr)
            delete col_idvec;
        if(val_dnvec != nullptr)
            delete val_dnvec;
        if(spattern != nullptr)
            delete spattern;
        if(spmat != nullptr)
            delete spmat;
        return nullptr;
    }

    return spmat;
}

rocsparse_spmat_descr _rocsparse_spmat_descr::create_coo_aos(int64_t     rows,
                                                             int64_t     cols,
                                                             int64_t     nnz,
                                                             const void* const_ind_data,
                                                             void*       ind_data,

                                                             const void*          const_val_data,
                                                             void*                val_data,
                                                             rocsparse_indextype  idx_type,
                                                             rocsparse_index_base idx_base,
                                                             rocsparse_datatype   val_type)
{
    rocsparse_idvec_descr    row_idvec{};
    rocsparse_idvec_descr    col_idvec{};
    rocsparse_dnvec_descr    val_dnvec{};
    rocsparse_spattern_descr spattern{};
    rocsparse_spmat_descr    spmat{};

    const void* s_const_ind_data
        = reinterpret_cast<const char*>(const_ind_data) + rocsparse::indextype_sizeof(idx_type);
    void* s_ind_data
        = (ind_data != nullptr)
              ? (reinterpret_cast<char*>(ind_data) + rocsparse::indextype_sizeof(idx_type))
              : nullptr;

    try
    {
        row_idvec
            = new _rocsparse_idvec_descr(idx_type, idx_base, nnz, 2, const_ind_data, ind_data);
        col_idvec
            = new _rocsparse_idvec_descr(idx_type, idx_base, nnz, 2, s_const_ind_data, s_ind_data);
        val_dnvec = new _rocsparse_dnvec_descr(val_type, nnz, 1, const_val_data, val_data);
        spattern  = _rocsparse_spattern_descr::create_coo(rows, cols, nnz, row_idvec, col_idvec);
        spattern->set_own_data(true);

        spmat = new _rocsparse_spmat_descr(spattern, val_dnvec, new _rocsparse_mat_info());
        spmat->set_own_spattern(true);
        spmat->set_own_values(true);
    }
    catch(...)
    {
        if(row_idvec != nullptr)
            delete row_idvec;
        if(col_idvec != nullptr)
            delete col_idvec;
        if(val_dnvec != nullptr)
            delete val_dnvec;
        if(spattern != nullptr)
            delete spattern;
        if(spmat != nullptr)
            delete spmat;
    }
    return spmat;
}

rocsparse_spmat_descr _rocsparse_spmat_descr::create_bsr(int64_t             mb,
                                                         int64_t             nb,
                                                         int64_t             nnzb,
                                                         rocsparse_direction block_dir,
                                                         int64_t             block_dim,

                                                         const void*          const_row_data,
                                                         void*                row_data,
                                                         const void*          const_col_data,
                                                         void*                col_data,
                                                         const void*          const_val_data,
                                                         void*                val_data,
                                                         rocsparse_indextype  row_type,
                                                         rocsparse_indextype  col_type,
                                                         rocsparse_index_base idx_base,
                                                         rocsparse_datatype   val_type)
{
    rocsparse_idvec_descr    row_idvec{};
    rocsparse_idvec_descr    col_idvec{};
    rocsparse_dnvec_descr    val_dnvec{};
    rocsparse_spattern_descr spattern{};
    rocsparse_spmat_descr    spmat{};
    try
    {
        row_idvec
            = new _rocsparse_idvec_descr(row_type, idx_base, mb + 1, 1, const_row_data, row_data);
        col_idvec
            = new _rocsparse_idvec_descr(col_type, idx_base, nnzb, 1, const_col_data, col_data);
        val_dnvec = new _rocsparse_dnvec_descr(
            val_type, nnzb * block_dim * block_dim, 1, const_val_data, val_data);

        spattern = _rocsparse_spattern_descr::create_bsr(mb, nb, nnzb, row_idvec, col_idvec);

        spattern->set_own_data(true);
        spmat = new _rocsparse_spmat_descr(
            spattern, block_dir, block_dim, val_dnvec, new _rocsparse_mat_info());
        spmat->set_own_spattern(true);
        spmat->set_own_values(true);
    }
    catch(...)
    {
        if(row_idvec != nullptr)
            delete row_idvec;
        if(col_idvec != nullptr)
            delete col_idvec;
        if(val_dnvec != nullptr)
            delete val_dnvec;
        if(spattern != nullptr)
            delete spattern;
        if(spmat != nullptr)
            delete spmat;
    }
    return spmat;
}

rocsparse_spmat_descr _rocsparse_spmat_descr::create_sell(int64_t              rows,
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
                                                          rocsparse_datatype   val_type)
{
    rocsparse_idvec_descr    row_idvec{};
    rocsparse_idvec_descr    col_idvec{};
    rocsparse_dnvec_descr    val_dnvec{};
    rocsparse_spattern_descr spattern{};
    rocsparse_spmat_descr    spmat{};
    try
    {
        row_idvec
            = new _rocsparse_idvec_descr(row_type, idx_base, rows + 1, 1, const_row_data, row_data);
        col_idvec
            = new _rocsparse_idvec_descr(col_type, idx_base, nnz, 1, const_col_data, col_data);
        val_dnvec = new _rocsparse_dnvec_descr(val_type, nnz, 1, const_val_data, val_data);
        spattern  = _rocsparse_spattern_descr::create_sell(rows, cols, nnz, row_idvec, col_idvec);

        spmat->sell_slice_size  = sell_slice_size;
        spmat->sell_colval_size = sell_colval_size;

        spattern->set_own_data(true);
        spmat = new _rocsparse_spmat_descr(spattern, val_dnvec, new _rocsparse_mat_info());
        spmat->set_own_spattern(true);
        spmat->set_own_values(true);
    }
    catch(...)
    {
        if(row_idvec != nullptr)
            delete row_idvec;
        if(col_idvec != nullptr)
            delete col_idvec;
        if(val_dnvec != nullptr)
            delete val_dnvec;
        if(spattern != nullptr)
            delete spattern;
        if(spmat != nullptr)
            delete spmat;
    }
    return spmat;
}

rocsparse_spmat_descr _rocsparse_spmat_descr::create_csr(int64_t              rows,
                                                         int64_t              cols,
                                                         int64_t              nnz,
                                                         const void*          const_row_data,
                                                         void*                row_data,
                                                         const void*          const_col_data,
                                                         void*                col_data,
                                                         const void*          const_val_data,
                                                         void*                val_data,
                                                         rocsparse_indextype  row_type,
                                                         rocsparse_indextype  col_type,
                                                         rocsparse_index_base idx_base,
                                                         rocsparse_datatype   val_type)
{
    rocsparse_idvec_descr    row_idvec{};
    rocsparse_idvec_descr    col_idvec{};
    rocsparse_dnvec_descr    val_dnvec{};
    rocsparse_spattern_descr spattern{};
    rocsparse_spmat_descr    spmat{};
    try
    {
        row_idvec
            = new _rocsparse_idvec_descr(row_type, idx_base, rows + 1, 1, const_row_data, row_data);
        col_idvec
            = new _rocsparse_idvec_descr(col_type, idx_base, nnz, 1, const_col_data, col_data);
        val_dnvec = new _rocsparse_dnvec_descr(val_type, nnz, 1, const_val_data, val_data);
        spattern  = _rocsparse_spattern_descr::create_csr(rows, cols, nnz, row_idvec, col_idvec);

        spattern->set_own_data(true);
        spmat = new _rocsparse_spmat_descr(spattern, val_dnvec, new _rocsparse_mat_info());
        spmat->set_own_spattern(true);
        spmat->set_own_values(true);
    }
    catch(...)
    {
        if(row_idvec != nullptr)
            delete row_idvec;
        if(col_idvec != nullptr)
            delete col_idvec;
        if(val_dnvec != nullptr)
            delete val_dnvec;
        if(spattern != nullptr)
            delete spattern;
        if(spmat != nullptr)
            delete spmat;
    }
    return spmat;
}

rocsparse_spmat_descr _rocsparse_spmat_descr::create_csc(int64_t              rows,
                                                         int64_t              cols,
                                                         int64_t              nnz,
                                                         const void*          const_row_data,
                                                         void*                row_data,
                                                         const void*          const_col_data,
                                                         void*                col_data,
                                                         const void*          const_val_data,
                                                         void*                val_data,
                                                         rocsparse_indextype  row_type,
                                                         rocsparse_indextype  col_type,
                                                         rocsparse_index_base idx_base,
                                                         rocsparse_datatype   val_type)
{
    rocsparse_idvec_descr    row_idvec{};
    rocsparse_idvec_descr    col_idvec{};
    rocsparse_dnvec_descr    val_dnvec{};
    rocsparse_spattern_descr spattern{};
    rocsparse_spmat_descr    spmat{};
    try
    {
        row_idvec
            = new _rocsparse_idvec_descr(row_type, idx_base, nnz, 1, const_row_data, row_data);
        col_idvec
            = new _rocsparse_idvec_descr(col_type, idx_base, cols + 1, 1, const_col_data, col_data);
        val_dnvec = new _rocsparse_dnvec_descr(val_type, nnz, 1, const_val_data, val_data);
        spattern  = _rocsparse_spattern_descr::create_csc(rows, cols, nnz, row_idvec, col_idvec);

        spattern->set_own_data(true);
        spmat = new _rocsparse_spmat_descr(spattern, val_dnvec, new _rocsparse_mat_info());
        spmat->set_own_spattern(true);
        spmat->set_own_values(true);
    }
    catch(...)
    {
        if(row_idvec != nullptr)
            delete row_idvec;
        if(col_idvec != nullptr)
            delete col_idvec;
        if(val_dnvec != nullptr)
            delete val_dnvec;
        if(spattern != nullptr)
            delete spattern;
        if(spmat != nullptr)
            delete spmat;
    }
    return spmat;
}

rocsparse_spmat_descr _rocsparse_spmat_descr::create_bell(int64_t              rows,
                                                          int64_t              cols,
                                                          rocsparse_direction  block_dir,
                                                          int64_t              block_dim,
                                                          const void*          const_ind_data,
                                                          void*                ind_data,
                                                          const void*          const_val_data,
                                                          void*                val_data,
                                                          int64_t              width,
                                                          rocsparse_indextype  idx_type,
                                                          rocsparse_index_base idx_base,
                                                          rocsparse_datatype   val_type)
{

    rocsparse_idvec_descr    col_idvec{};
    rocsparse_dnvec_descr    val_dnvec{};
    rocsparse_spattern_descr spattern{};
    rocsparse_spmat_descr    spmat{};
    try
    {
        const int64_t nnz_s = rows * width;
        col_idvec
            = new _rocsparse_idvec_descr(idx_type, idx_base, nnz_s, 1, const_ind_data, ind_data);
        const int64_t nnz_n = rows * width * block_dim * block_dim;
        val_dnvec = new _rocsparse_dnvec_descr(val_type, nnz_n, 1, const_val_data, val_data);
        spattern  = _rocsparse_spattern_descr::create_bell(rows, cols, nnz_s, col_idvec);

        spattern->set_own_data(true);
        spmat = new _rocsparse_spmat_descr(
            spattern, block_dir, block_dim, val_dnvec, new _rocsparse_mat_info());
        spmat->set_own_spattern(true);
        spmat->set_own_values(true);
    }
    catch(...)
    {
        if(col_idvec != nullptr)
            delete col_idvec;
        if(val_dnvec != nullptr)
            delete val_dnvec;
        if(spattern != nullptr)
            delete spattern;
        if(spmat != nullptr)
            delete spmat;
    }
    return spmat;
}

rocsparse_spmat_descr _rocsparse_spmat_descr::create_ell(int64_t              rows,
                                                         int64_t              cols,
                                                         const void*          const_ind_data,
                                                         void*                ind_data,
                                                         const void*          const_val_data,
                                                         void*                val_data,
                                                         int64_t              width,
                                                         rocsparse_indextype  idx_type,
                                                         rocsparse_index_base idx_base,
                                                         rocsparse_datatype   val_type)
{

    rocsparse_idvec_descr    col_idvec{};
    rocsparse_dnvec_descr    val_dnvec{};
    rocsparse_spattern_descr spattern{};
    rocsparse_spmat_descr    spmat{};

    try
    {
        const int64_t nnz_s = rows * width;
        col_idvec
            = new _rocsparse_idvec_descr(idx_type, idx_base, nnz_s, 1, const_ind_data, ind_data);
        const int64_t nnz_n = nnz_s;
        val_dnvec = new _rocsparse_dnvec_descr(val_type, nnz_n, 1, const_val_data, val_data);
        spattern  = _rocsparse_spattern_descr::create_ell(rows, cols, width, col_idvec);

        spattern->set_own_data(true);
        spmat = new _rocsparse_spmat_descr(spattern, val_dnvec, new _rocsparse_mat_info());

        spmat->set_own_spattern(true);
        spmat->set_own_values(true);
    }
    catch(...)
    {
        if(col_idvec != nullptr)
            delete col_idvec;
        if(val_dnvec != nullptr)
            delete val_dnvec;
        if(spattern != nullptr)
            delete spattern;
        if(spmat != nullptr)
            delete spmat;
    }
    return spmat;
}

#endif

_rocsparse_spmat_descr::_rocsparse_spmat_descr(rocsparse_spattern_descr spattern,
                                               rocsparse_dnvec_descr    values,
                                               rocsparse_mat_info       info)
    : m_spattern(spattern)
    , m_values(values)
    , m_mat_info(info)
{
    this->m_block_dim = 1;
    this->m_block_dir = rocsparse_direction_column;
}

_rocsparse_spmat_descr::_rocsparse_spmat_descr(rocsparse_spattern_descr spattern,
                                               rocsparse_direction      block_dir,
                                               int64_t                  block_dim,
                                               rocsparse_dnvec_descr    values,
                                               rocsparse_mat_info       info)
    : m_spattern(spattern)
    , m_values(values)
    , m_mat_info(info)
{
    this->m_block_dim = block_dim;
    this->m_block_dir = block_dir;
}

rocsparse_direction _rocsparse_spmat_descr::get_block_dir() const
{
    return this->m_block_dir;
}

int64_t _rocsparse_spmat_descr::get_batch_count() const
{
    return this->m_values->get_batch_count();
}

void _rocsparse_spmat_descr::set_batch_count(int64_t value)
{
    this->m_values->set_batch_count(value);
}

int64_t _rocsparse_spmat_descr::get_row_batch_dist() const
{
    return this->m_spattern->get_row_data()->get_batch_dist();
}
void _rocsparse_spmat_descr::set_row_batch_dist(int64_t value)
{
    this->m_spattern->get_row_data()->set_batch_dist(value);
}

int64_t _rocsparse_spmat_descr::get_col_batch_dist() const
{
    return this->m_spattern->get_col_data()->get_batch_dist();
}

void _rocsparse_spmat_descr::set_col_batch_dist(int64_t value)
{
    this->m_spattern->get_col_data()->set_batch_dist(value);
}

int64_t _rocsparse_spmat_descr::get_val_batch_dist() const
{
    return this->m_values->get_batch_dist();
}

void _rocsparse_spmat_descr::set_val_batch_dist(int64_t value)
{
    this->m_values->set_batch_dist(value);
}

int64_t _rocsparse_spmat_descr::get_block_dim() const
{
    return this->m_block_dim;
}
int64_t _rocsparse_spmat_descr::get_ell_width() const
{
    return this->m_spattern->get_ell_width();
}
void _rocsparse_spmat_descr::set_ell_width(int64_t value)
{
    this->m_spattern->set_ell_width(value);
}
void _rocsparse_spmat_descr::set_nnz(int64_t value)
{
    this->m_spattern->set_nnz(value);
}
int64_t _rocsparse_spmat_descr::get_ell_cols() const
{
    return this->m_spattern->get_ell_cols();
}
rocsparse_format _rocsparse_spmat_descr::get_format() const
{
    return this->m_spattern->get_format();
}
rocsparse_indextype _rocsparse_spmat_descr::get_row_type() const
{
    return this->m_spattern->get_row_data()->get_indextype();
}
rocsparse_indextype _rocsparse_spmat_descr::get_col_type() const
{
    return this->m_spattern->get_col_data()->get_indextype();
}
rocsparse_datatype _rocsparse_spmat_descr::get_data_type() const
{
    return this->m_values->get_datatype();
}
rocsparse_index_base _rocsparse_spmat_descr::get_idx_base() const
{
    return this->m_spattern->get_row_data()->get_base();
}

const _rocsparse_spattern_descr* _rocsparse_spmat_descr::get_spattern() const
{
    return this->m_spattern;
}
_rocsparse_spattern_descr* _rocsparse_spmat_descr::get_spattern()
{
    return this->m_spattern;
}

rocsparse_dnvec_descr _rocsparse_spmat_descr::get_values()
{
    return this->m_values;
}
rocsparse_const_dnvec_descr _rocsparse_spmat_descr::get_values() const
{
    return this->m_values;
}

void _rocsparse_spmat_descr::set_values(rocsparse_dnvec_descr value)
{
    this->m_values = value;
}
void _rocsparse_spmat_descr::set_spattern(rocsparse_spattern_descr value)
{
    this->m_spattern = value;
}

bool _rocsparse_spmat_descr::get_analysed() const
{
    return this->analysed;
}
void _rocsparse_spmat_descr::set_analysed(bool value) const
{
    this->analysed = value;
}

rocsparse_mat_info _rocsparse_spmat_descr::get_info() const
{
    return this->m_mat_info;
}
rocsparse_mat_descr _rocsparse_spmat_descr::get_descr() const
{
    return this->m_spattern->get_mat_descr();
}

bool _rocsparse_spmat_descr::get_init() const
{
    return this->init;
}
void _rocsparse_spmat_descr::set_init(bool value)
{
    this->init = value;
}
int64_t _rocsparse_spmat_descr::get_rows() const
{
    return this->m_spattern->get_rows();
}

int64_t _rocsparse_spmat_descr::get_cols() const
{
    return this->m_spattern->get_cols();
}
int64_t _rocsparse_spmat_descr::get_nnz() const
{
    return this->m_spattern->get_nnz();
}
int64_t* _rocsparse_spmat_descr::get_pnnz()
{
    return this->m_spattern->get_pnnz();
}

void* _rocsparse_spmat_descr::get_row_data() const
{
    return this->m_spattern->get_row_data()->data();
}
void* _rocsparse_spmat_descr::get_col_data() const
{
    return this->m_spattern->get_col_data()->data();
}
void* _rocsparse_spmat_descr::get_ind_data() const
{
    return this->m_spattern->get_row_data()->data();
}
void* _rocsparse_spmat_descr::get_val_data() const
{
    return this->m_values->data();
}
void _rocsparse_spmat_descr::set_row_data(void* value)
{
    this->m_spattern->get_row_data()->set_data(value);
}
void _rocsparse_spmat_descr::set_col_data(void* value)
{
    this->m_spattern->get_col_data()->set_data(value);
}
void _rocsparse_spmat_descr::set_ind_data(void* value)
{
    this->m_spattern->get_row_data()->set_data(value);
}
void _rocsparse_spmat_descr::set_val_data(void* value)
{
    this->m_values->set_data(value);
}

void _rocsparse_spmat_descr::set_const_row_data(const void* value)
{
    this->m_spattern->get_row_data()->set_const_data(value);
}
void _rocsparse_spmat_descr::set_const_col_data(const void* value)
{
    this->m_spattern->get_col_data()->set_const_data(value);
}
void _rocsparse_spmat_descr::set_const_ind_data(const void* value)
{
    this->m_spattern->get_row_data()->set_const_data(value);
}
void _rocsparse_spmat_descr::set_const_val_data(const void* value)
{
    this->m_values->set_const_data(value);
}

const void* _rocsparse_spmat_descr::get_const_row_data() const
{
    return this->m_spattern->get_row_data()->const_data();
}
const void* _rocsparse_spmat_descr::get_const_col_data() const
{
    return this->m_spattern->get_col_data()->const_data();
}
const void* _rocsparse_spmat_descr::get_const_ind_data() const
{
    return this->m_spattern->get_row_data()->const_data();
}
const void* _rocsparse_spmat_descr::get_const_val_data() const
{
    return this->m_values->const_data();
}

extern "C" rocsparse_status rocsparse_spmat_create(rocsparse_handle         handle,
                                                   rocsparse_spmat_descr*   p_descr,
                                                   rocsparse_spattern_descr spattern,
                                                   rocsparse_dnvec_descr    values,
                                                   rocsparse_error*         p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, p_descr);
    ROCSPARSE_CHECKARG_POINTER(2, spattern);
    ROCSPARSE_CHECKARG_POINTER(3, values);
    rocsparse_mat_info info;
    THROW_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&info));
    p_descr[0] = new _rocsparse_spmat_descr(spattern, values, info);
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_spmat_create_block(rocsparse_handle         handle,
                                                         rocsparse_spmat_descr*   p_descr,
                                                         rocsparse_spattern_descr spattern,
                                                         rocsparse_direction      block_dir,
                                                         int64_t                  block_dim,
                                                         rocsparse_dnvec_descr    values,
                                                         rocsparse_error*         p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, p_descr);
    ROCSPARSE_CHECKARG_POINTER(2, spattern);
    ROCSPARSE_CHECKARG_POINTER(3, values);
    rocsparse_mat_info info;
    THROW_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&info));
    p_descr[0] = new _rocsparse_spmat_descr(spattern, block_dir, block_dim, values, info);
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_spmat_get_prop(rocsparse_handle            handle,
                                                     rocsparse_const_spmat_descr descr,
                                                     rocsparse_spmat_prop        prop,
                                                     void*                       p_value,
                                                     size_t           value_size_in_bytes,
                                                     rocsparse_error* p_error)
{
    try
    {
        ROCSPARSE_ROUTINE_TRACE;
        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_POINTER(1, descr);
        ROCSPARSE_CHECKARG_ENUM(2, prop);
        ROCSPARSE_CHECKARG_POINTER(3, p_value);

        switch(prop)
        {
        case rocsparse_spmat_prop_format:
        {
            ROCSPARSE_CHECKARG(4,
                               value_size_in_bytes,
                               (sizeof(rocsparse_format) != value_size_in_bytes),
                               rocsparse_status_invalid_value);
            *reinterpret_cast<rocsparse_format*>(p_value) = descr->get_format();
            return rocsparse_status_success;
        }
        case rocsparse_spmat_prop_batch_count:
        {
            ROCSPARSE_CHECKARG(4,
                               value_size_in_bytes,
                               (sizeof(int64_t) != value_size_in_bytes),
                               rocsparse_status_invalid_value);

            *reinterpret_cast<int64_t*>(p_value) = descr->get_batch_count();
            return rocsparse_status_success;
        }
        case rocsparse_spmat_prop_rows:
        {
            ROCSPARSE_CHECKARG(4,
                               value_size_in_bytes,
                               (sizeof(int64_t) != value_size_in_bytes),
                               rocsparse_status_invalid_value);

            *reinterpret_cast<int64_t*>(p_value) = descr->get_rows();
            return rocsparse_status_success;
        }
        case rocsparse_spmat_prop_cols:
        {
            ROCSPARSE_CHECKARG(4,
                               value_size_in_bytes,
                               (sizeof(int64_t) != value_size_in_bytes),
                               rocsparse_status_invalid_value);

            *reinterpret_cast<int64_t*>(p_value) = descr->get_cols();
            return rocsparse_status_success;
        }
        case rocsparse_spmat_prop_nnz:
        {
            ROCSPARSE_CHECKARG(4,
                               value_size_in_bytes,
                               (sizeof(int64_t) != value_size_in_bytes),
                               rocsparse_status_invalid_value);
            *reinterpret_cast<int64_t*>(p_value) = descr->get_nnz();
            return rocsparse_status_success;
        }
        case rocsparse_spmat_prop_block_dir:
        {
            ROCSPARSE_CHECKARG(4,
                               value_size_in_bytes,
                               (sizeof(rocsparse_direction) != value_size_in_bytes),
                               rocsparse_status_invalid_value);
            *reinterpret_cast<rocsparse_direction*>(p_value) = descr->get_block_dir();
            return rocsparse_status_success;
        }
        case rocsparse_spmat_prop_block_dim:
        {
            ROCSPARSE_CHECKARG(4,
                               value_size_in_bytes,
                               (sizeof(int64_t) != value_size_in_bytes),
                               rocsparse_status_invalid_value);
            *reinterpret_cast<int64_t*>(p_value) = descr->get_block_dim();
            return rocsparse_status_success;
        }
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_START
    }
    catch(...)
    {
        RETURN_ROCSPARSE_EXCEPTION();
    }
    // LCOV_EXCL_STOP
}

extern "C" rocsparse_status rocsparse_spmat_set_prop(rocsparse_handle      handle,
                                                     rocsparse_spmat_descr descr,
                                                     rocsparse_spmat_prop  prop,
                                                     const void*           p_value,
                                                     size_t                value_size_in_bytes,
                                                     rocsparse_error*      p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG_ENUM(2, prop);
    ROCSPARSE_CHECKARG_POINTER(3, p_value);
#if 0
    switch(prop)
    {
    case rocsparse_spmat_prop_format:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(rocsparse_format) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        descr->set_format(*reinterpret_cast<const rocsparse_format*>(p_value));
        return rocsparse_status_success;
    }
    case rocsparse_spmat_prop_batch_count:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);

        descr->set_batch_count(*reinterpret_cast<const int64_t*>(p_value));
        return rocsparse_status_success;
    }
    case rocsparse_spmat_prop_rows:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);

        descr->set_rows(*reinterpret_cast<const int64_t*>(p_value));
        return rocsparse_status_success;
    }
    case rocsparse_spmat_prop_cols:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);

        descr->set_cols(*reinterpret_cast<const int64_t*>(p_value));
        return rocsparse_status_success;
    }
    case rocsparse_spmat_prop_nnz:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        descr->set_nnz(*reinterpret_cast<const int64_t*>(p_value));
        return rocsparse_status_success;
    }
    case rocsparse_spmat_prop_block_dir:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(rocsparse_direction) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        descr->set_block_dir(*reinterpret_cast<const rocsparse_direction*>(p_value));
        return rocsparse_status_success;
    }
    case rocsparse_spmat_prop_block_dim:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        descr->set_block_dim(*reinterpret_cast<const int64_t*>(p_value));
        return rocsparse_status_success;
    }
    }
#endif
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_spmat_get_spattern(rocsparse_handle          handle,
                                                         rocsparse_spmat_descr     descr,
                                                         rocsparse_spattern_descr* p_value,
                                                         rocsparse_error*          p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG_POINTER(2, p_value);
    p_value[0] = descr->get_spattern();
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_spmat_set_spattern(rocsparse_handle         handle,
                                                         rocsparse_spmat_descr    descr,
                                                         rocsparse_spattern_descr value,
                                                         rocsparse_error*         p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG_POINTER(2, value);
    descr->set_spattern(value);
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_spmat_get_data(rocsparse_handle       handle,
                                                     rocsparse_spmat_descr  descr,
                                                     rocsparse_dnvec_descr* p_value,
                                                     rocsparse_error*       p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG_POINTER(2, p_value);
    p_value[0] = descr->get_values();
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_spmat_set_data(rocsparse_handle      handle,
                                                     rocsparse_spmat_descr descr,
                                                     rocsparse_dnvec_descr value,
                                                     rocsparse_error*      p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG_POINTER(2, value);
    descr->set_values(value);
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
