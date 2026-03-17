/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_spmat_descr.hpp"
#include "rocsparse_memstat.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status _rocsparse_spmat_descr::validate()
{
    RETURN_IF_ROCSPARSE_ERROR(this->get_spattern()->validate());
    return rocsparse_status_success;
}

void _rocsparse_spmat_descr::set_own_spattern(bool value)
{
    this->m_own_spattern = value;
}

_rocsparse_spmat_descr::~_rocsparse_spmat_descr()
{
    hipStream_t default_stream{};
    this->destroy(default_stream);
}

rocsparse_status _rocsparse_spmat_descr::destroy(hipStream_t stream)
{

    if(&this->m_mat_info == this->m_p_mat_info)
    {
        RETURN_IF_ROCSPARSE_ERROR(this->m_mat_info.destroy(stream));
    }

    if(&this->m_spattern == this->m_p_spattern)
    {
        RETURN_IF_ROCSPARSE_ERROR(this->m_spattern.destroy(stream));
    }

    if(&this->m_values == this->m_p_values)
    {
        RETURN_IF_ROCSPARSE_ERROR(this->m_values.destroy(stream));
    }

    return rocsparse_status_success;
}

void _rocsparse_spmat_descr::define(rocsparse_spattern_descr spattern,
                                    rocsparse_dnvec_descr    values,
                                    rocsparse_mat_info       info)
{
    if(spattern != nullptr)
    {
        this->m_p_spattern = spattern;
    }

    if(values != nullptr)
    {
        this->m_p_values = values;
    }

    if(info != nullptr)
    {
        this->m_p_mat_info = info;
    }

    this->set_init(true);
}

rocsparse_direction _rocsparse_spmat_descr::get_block_dir() const
{
    return this->get_spattern()->get_block_dir();
}

int64_t _rocsparse_spmat_descr::get_batch_count() const
{
    return this->get_values()->get_batch_count();
}

void _rocsparse_spmat_descr::set_batch_count(int64_t value)
{
    this->get_values()->set_batch_count(value);
}

int64_t _rocsparse_spmat_descr::get_row_batch_dist() const
{
    return this->get_spattern()->get_row_data()->get_batch_dist();
}

void _rocsparse_spmat_descr::set_row_batch_dist(int64_t value)
{
    this->get_spattern()->get_row_data()->set_batch_dist(value);
}

int64_t _rocsparse_spmat_descr::get_col_batch_dist() const
{
    return this->get_spattern()->get_col_data()->get_batch_dist();
}

void _rocsparse_spmat_descr::set_col_batch_dist(int64_t value)
{
    this->get_spattern()->get_col_data()->set_batch_dist(value);
}

int64_t _rocsparse_spmat_descr::get_val_batch_dist() const
{
    return this->get_values()->get_batch_dist();
}

void _rocsparse_spmat_descr::set_val_batch_dist(int64_t value)
{
    this->get_values()->set_batch_dist(value);
}

void _rocsparse_spmat_descr::set_batch_stride(int64_t value)
{
    this->get_values()->set_batch_dist(value);
}

int64_t _rocsparse_spmat_descr::get_batch_stride() const
{
    return this->get_values()->get_batch_dist();
}

int64_t _rocsparse_spmat_descr::get_columns_values_batch_stride() const
{
    return this->get_values()->get_batch_dist();
}

void _rocsparse_spmat_descr::set_columns_values_batch_stride(int64_t value)
{

    this->get_values()->set_batch_dist(value);
}

void _rocsparse_spmat_descr::set_offsets_batch_stride(int64_t value)
{
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    if(this->get_spattern()->get_format() == rocsparse_format_csc)
    {
        this->get_spattern()->get_col_data()->set_batch_dist(value);
    }
    else
    {
        this->get_spattern()->get_row_data()->set_batch_dist(value);
    }
}

int64_t _rocsparse_spmat_descr::get_offsets_batch_stride() const
{
    if(this->get_spattern()->get_format() == rocsparse_format_csc)
    {
        return this->get_spattern()->get_col_data()->get_batch_dist();
    }
    else
    {
        return this->get_spattern()->get_row_data()->get_batch_dist();
    }
}

int64_t _rocsparse_spmat_descr::get_block_dim() const
{
    return this->get_spattern()->get_block_dim();
}

int64_t _rocsparse_spmat_descr::get_ell_width() const
{
    return this->get_spattern()->get_ell_width();
}

void _rocsparse_spmat_descr::set_ell_width(int64_t value)
{
    this->get_spattern()->set_ell_width(value);
}
void _rocsparse_spmat_descr::set_nnz(int64_t value)
{
    this->get_spattern()->set_nnz(value);
}
int64_t _rocsparse_spmat_descr::get_ell_cols() const
{
    return this->get_spattern()->get_ell_cols();
}

void _rocsparse_spmat_descr::set_block_dir(rocsparse_direction value)
{
    this->get_spattern()->set_block_dir(value);
}

void _rocsparse_spmat_descr::set_block_dim(int64_t value)
{
    this->get_spattern()->set_block_dim(value);
}

void _rocsparse_spmat_descr::set_format(rocsparse_format value)
{
    return this->get_spattern()->set_format(value);
}

rocsparse_format _rocsparse_spmat_descr::get_format() const
{
    return this->get_spattern()->get_format();
}

rocsparse_indextype _rocsparse_spmat_descr::get_row_type() const
{
    return this->get_spattern()->get_row_data()->get_indextype();
}
rocsparse_indextype _rocsparse_spmat_descr::get_col_type() const
{
    return this->get_spattern()->get_col_data()->get_indextype();
}
rocsparse_datatype _rocsparse_spmat_descr::get_data_type() const
{
    return this->get_values()->get_datatype();
}
rocsparse_index_base _rocsparse_spmat_descr::get_idx_base() const
{
    return this->get_spattern()->get_row_data()->get_base();
}

const _rocsparse_spattern_descr* _rocsparse_spmat_descr::get_spattern() const
{
    return this->m_p_spattern;
}
_rocsparse_spattern_descr* _rocsparse_spmat_descr::get_spattern()
{
    return this->m_p_spattern;
}

rocsparse_dnvec_descr _rocsparse_spmat_descr::get_values()
{
    return this->m_p_values;
}

rocsparse_const_dnvec_descr _rocsparse_spmat_descr::get_values() const
{
    return this->m_p_values;
}

void _rocsparse_spmat_descr::set_values(rocsparse_dnvec_descr value)
{
    this->m_p_values = value;
}
void _rocsparse_spmat_descr::set_spattern(rocsparse_spattern_descr value)
{
    this->m_p_spattern = value;
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
    return this->get_spattern()->get_info();
}

rocsparse_mat_descr _rocsparse_spmat_descr::get_descr() const
{
    return this->get_spattern()->get_mat_descr();
}

rocsparse_mat_descr _rocsparse_spmat_descr::get_descr()
{
    return this->get_spattern()->get_mat_descr();
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
    return this->get_spattern()->get_rows();
}

void _rocsparse_spmat_descr::set_rows(int64_t value)
{
    return this->get_spattern()->set_rows(value);
}

void _rocsparse_spmat_descr::set_cols(int64_t value)
{
    return this->get_spattern()->set_cols(value);
}

int64_t _rocsparse_spmat_descr::get_cols() const
{
    return this->get_spattern()->get_cols();
}
int64_t _rocsparse_spmat_descr::get_nnz() const
{
    return this->get_spattern()->get_nnz();
}
int64_t* _rocsparse_spmat_descr::get_pnnz()
{
    return this->get_spattern()->get_pnnz();
}

const void* _rocsparse_spmat_descr::get_row_data() const
{
    return this->get_spattern()->get_row_data()->data();
}

const void* _rocsparse_spmat_descr::get_col_data() const
{
    return this->get_spattern()->get_col_data()->data();
}

const void* _rocsparse_spmat_descr::get_ind_data() const
{
    return this->get_spattern()->get_row_data()->data();
}

const void* _rocsparse_spmat_descr::get_val_data() const
{
    return this->get_values()->data();
}

void* _rocsparse_spmat_descr::get_row_data()
{
    return this->get_spattern()->get_row_data()->data();
}

void* _rocsparse_spmat_descr::get_col_data()
{
    return this->get_spattern()->get_col_data()->data();
}

void* _rocsparse_spmat_descr::get_ind_data()
{
    return this->get_spattern()->get_row_data()->data();
}

void* _rocsparse_spmat_descr::get_val_data()
{
    return this->get_values()->data();
}

void _rocsparse_spmat_descr::set_row_data(void* value)
{
    this->get_spattern()->get_row_data()->set_data(value);
}
void _rocsparse_spmat_descr::set_col_data(void* value)
{
    this->get_spattern()->get_col_data()->set_data(value);
}
void _rocsparse_spmat_descr::set_ind_data(void* value)
{
    this->get_spattern()->get_row_data()->set_data(value);
}
void _rocsparse_spmat_descr::set_val_data(void* value)
{
    this->get_values()->set_data(value);
}

void _rocsparse_spmat_descr::set_const_row_data(const void* value)
{
    this->get_spattern()->get_row_data()->set_const_data(value);
}
void _rocsparse_spmat_descr::set_const_col_data(const void* value)
{
    this->get_spattern()->get_col_data()->set_const_data(value);
}
void _rocsparse_spmat_descr::set_const_ind_data(const void* value)
{
    this->get_spattern()->get_row_data()->set_const_data(value);
}
void _rocsparse_spmat_descr::set_const_val_data(const void* value)
{
    this->get_values()->set_const_data(value);
}

void _rocsparse_spmat_descr::set_sell_slice_size(int64_t value)
{
    this->get_spattern()->set_sell_slice_size(value);
}
void _rocsparse_spmat_descr::set_sell_colval_size(int64_t value)
{
    this->get_spattern()->set_sell_colval_size(value);
}
int64_t _rocsparse_spmat_descr::get_sell_slice_size() const
{
    return this->get_spattern()->get_sell_slice_size();
}
int64_t _rocsparse_spmat_descr::get_sell_colval_size() const
{
    return this->get_spattern()->get_sell_colval_size();
}

const void* _rocsparse_spmat_descr::get_const_row_data() const
{
    return this->get_spattern()->get_row_data()->const_data();
}

const void* _rocsparse_spmat_descr::get_const_col_data() const
{
    return this->get_spattern()->get_col_data()->const_data();
}

const void* _rocsparse_spmat_descr::get_const_ind_data() const
{
    return this->get_spattern()->get_row_data()->const_data();
}

const void* _rocsparse_spmat_descr::get_const_val_data() const
{
    return this->get_values()->const_data();
}
