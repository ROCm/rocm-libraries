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

#include "rocsparse_spattern_descr.hpp"
#include "rocsparse_memstat.hpp"
#include "rocsparse_utility.hpp"

rocsparse_mat_info _rocsparse_spattern_descr::get_info() const
{
    return this->m_p_mat_info;
}
void _rocsparse_spattern_descr::set_info(rocsparse_mat_info value)
{
    this->m_p_mat_info = value;
}

void _rocsparse_spattern_descr::set_sell_slice_size(int64_t value)
{
    this->m_sell_slice_size = value;
}
void _rocsparse_spattern_descr::set_sell_colval_size(int64_t value)
{
    this->m_sell_colval_size = value;
}
int64_t _rocsparse_spattern_descr::get_sell_slice_size() const
{
    return this->m_sell_slice_size;
}
int64_t _rocsparse_spattern_descr::get_sell_colval_size() const
{
    return this->m_sell_colval_size;
}

rocsparse_direction _rocsparse_spattern_descr::get_block_dir() const
{
    return this->m_block_dir;
}

void _rocsparse_spattern_descr::set_block_dir(rocsparse_direction value)
{
    this->m_block_dir = value;
}

void _rocsparse_spattern_descr::set_block_dim(int64_t value)
{
    this->m_block_dim = value;
}

int64_t _rocsparse_spattern_descr::get_block_dim() const
{
    return this->m_block_dim;
}

void _rocsparse_spattern_descr::set_row_data(rocsparse_idvec_descr data)
{
    this->m_p_row_data = data;
}

void _rocsparse_spattern_descr::set_col_data(rocsparse_idvec_descr data)
{
    this->m_p_col_data = data;
}

int64_t _rocsparse_spattern_descr::get_batch_count() const
{
    return this->m_batch_count;
}

void _rocsparse_spattern_descr::set_batch_count(int64_t value)
{
    this->m_batch_count = value;
}

int64_t* _rocsparse_spattern_descr::get_pnnz()
{
    return &this->m_nnz;
}

void _rocsparse_spattern_descr::set_own_data(bool value)
{
    this->m_own_data = value;
}

_rocsparse_spattern_descr::~_rocsparse_spattern_descr()
{
    hipStream_t default_stream{};
    this->destroy(default_stream);
}

rocsparse_status _rocsparse_spattern_descr::destroy(hipStream_t stream)
{

    if(this->m_p_row_data && (this->m_p_row_data == &this->m_row_data))
    {
        RETURN_IF_ROCSPARSE_ERROR(this->m_row_data.destroy(stream));
    }
    if(this->m_p_col_data && (this->m_p_col_data == &this->m_col_data))
    {
        RETURN_IF_ROCSPARSE_ERROR(this->m_col_data.destroy(stream));
    }
    this->m_p_row_data = nullptr;
    this->m_p_col_data = nullptr;
    return rocsparse_status_success;
}

int64_t _rocsparse_spattern_descr::get_rows() const
{
    return this->m_rows;
}
int64_t _rocsparse_spattern_descr::get_cols() const
{
    return this->m_cols;
}
int64_t _rocsparse_spattern_descr::get_nnz() const
{
    return this->m_nnz;
}

void _rocsparse_spattern_descr::set_rows(int64_t value)
{
    this->m_rows = value;
}
void _rocsparse_spattern_descr::set_cols(int64_t value)
{
    this->m_cols = value;
}

int64_t _rocsparse_spattern_descr::get_ell_width() const
{
    return this->m_ell_width;
}

void _rocsparse_spattern_descr::set_ell_width(int64_t value)
{
    this->m_ell_width = value;
}

void _rocsparse_spattern_descr::set_ell_cols(int64_t value)
{
    this->m_ell_cols = value;
}

void _rocsparse_spattern_descr::set_nnz(int64_t value)
{
    this->m_nnz = value;
}

int64_t _rocsparse_spattern_descr::get_ell_cols() const
{
    return this->m_ell_cols;
}
rocsparse_format _rocsparse_spattern_descr::get_format() const
{
    return this->m_format;
}

void _rocsparse_spattern_descr::set_format(rocsparse_format value)
{
    this->m_format = value;
}

rocsparse_idvec_descr _rocsparse_spattern_descr::get_row_data()
{
    return this->m_p_row_data;
}
rocsparse_idvec_descr _rocsparse_spattern_descr::get_col_data()
{
    return this->m_p_col_data;
}
rocsparse_const_idvec_descr _rocsparse_spattern_descr::get_row_data() const
{
    return this->m_p_row_data;
}
rocsparse_const_idvec_descr _rocsparse_spattern_descr::get_col_data() const
{
    return this->m_p_col_data;
}

rocsparse_mat_descr _rocsparse_spattern_descr::get_mat_descr()
{
    return this->m_p_mat_descr;
}

rocsparse_mat_descr _rocsparse_spattern_descr::get_mat_descr() const
{
    return this->m_p_mat_descr;
}

void _rocsparse_spattern_descr::define_csr(int64_t               rows,
                                           int64_t               cols,
                                           int64_t               nnz,
                                           rocsparse_idvec_descr row_data,
                                           rocsparse_idvec_descr col_data,
                                           rocsparse_mat_descr   mat_descr,
                                           rocsparse_mat_info    mat_info)
{
    this->m_format     = rocsparse_format_csr;
    this->m_rows       = rows;
    this->m_cols       = cols;
    this->m_nnz        = nnz;
    this->m_p_row_data = row_data;
    this->m_p_col_data = col_data;
    if(mat_descr != nullptr)
    {
        this->m_p_mat_descr = mat_descr;
    }
    else
    {
        // Initialize descriptor
        THROW_IF_ROCSPARSE_ERROR(
            rocsparse_set_mat_index_base(this->get_mat_descr(), row_data->get_base()));
    }
    if(mat_info != nullptr)
    {
        this->m_p_mat_info = mat_info;
    }
}

void _rocsparse_spattern_descr::define_csc(int64_t               rows,
                                           int64_t               cols,
                                           int64_t               nnz,
                                           rocsparse_idvec_descr row_data,
                                           rocsparse_idvec_descr col_data,
                                           rocsparse_mat_descr   mat_descr,
                                           rocsparse_mat_info    mat_info)
{
    this->m_format     = rocsparse_format_csc;
    this->m_rows       = rows;
    this->m_cols       = cols;
    this->m_nnz        = nnz;
    this->m_p_row_data = row_data;
    this->m_p_col_data = col_data;
    if(mat_descr != nullptr)
    {
        this->m_p_mat_descr = mat_descr;
    }
    else
    {
        // Initialize descriptor
        THROW_IF_ROCSPARSE_ERROR(
            rocsparse_set_mat_index_base(this->get_mat_descr(), row_data->get_base()));
    }
    if(mat_info != nullptr)
    {
        this->m_p_mat_info = mat_info;
    }
}

void _rocsparse_spattern_descr::define_coo_aos(int64_t               rows,
                                               int64_t               cols,
                                               int64_t               nnz,
                                               rocsparse_idvec_descr row_data,
                                               rocsparse_idvec_descr col_data,
                                               rocsparse_mat_descr   mat_descr,
                                               rocsparse_mat_info    mat_info)
{
    this->m_format     = rocsparse_format_coo_aos;
    this->m_rows       = rows;
    this->m_cols       = cols;
    this->m_nnz        = nnz;
    this->m_p_row_data = row_data;
    this->m_p_col_data = col_data;
    if(mat_descr != nullptr)
    {
        this->m_p_mat_descr = mat_descr;
    }
    else
    {
        // Initialize descriptor
        THROW_IF_ROCSPARSE_ERROR(
            rocsparse_set_mat_index_base(this->get_mat_descr(), row_data->get_base()));
    }
    if(mat_info != nullptr)
    {
        this->m_p_mat_info = mat_info;
    }
}

void _rocsparse_spattern_descr::define_coo(int64_t               rows,
                                           int64_t               cols,
                                           int64_t               nnz,
                                           rocsparse_idvec_descr row_data,
                                           rocsparse_idvec_descr col_data,
                                           rocsparse_mat_descr   mat_descr,
                                           rocsparse_mat_info    mat_info)
{
    this->m_format     = rocsparse_format_coo;
    this->m_rows       = rows;
    this->m_cols       = cols;
    this->m_nnz        = nnz;
    this->m_p_row_data = row_data;
    this->m_p_col_data = col_data;
    if(mat_descr != nullptr)
    {
        this->m_p_mat_descr = mat_descr;
    }
    else
    {
        // Initialize descriptor
        THROW_IF_ROCSPARSE_ERROR(
            rocsparse_set_mat_index_base(this->get_mat_descr(), row_data->get_base()));
    }
    if(mat_info != nullptr)
    {
        this->m_p_mat_info = mat_info;
    }
}

rocsparse_status _rocsparse_spattern_descr::validate()
{
    switch(this->m_format)
    {
    case rocsparse_format_csr:
    {
        RETURN_IF_ROCSPARSE_ERROR(this->m_p_row_data->validate());
        RETURN_IF_ROCSPARSE_ERROR((this->m_rows + 1 != this->m_p_row_data->get_size())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR((this->m_p_mat_descr->base != this->m_p_row_data->get_base())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR(this->m_p_col_data->validate());
        RETURN_IF_ROCSPARSE_ERROR((this->m_nnz != this->m_p_col_data->get_size())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR((this->m_p_mat_descr->base != this->m_p_col_data->get_base())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR((this->m_block_dim != 1) ? rocsparse_status_internal_error
                                                           : rocsparse_status_success);
        RETURN_IF_ROCSPARSE_ERROR((1 != this->m_p_col_data->get_inc())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);
        RETURN_IF_ROCSPARSE_ERROR((1 != this->m_p_col_data->get_inc())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);
        return rocsparse_status_success;
    }

    case rocsparse_format_ell:
    {
        RETURN_IF_ROCSPARSE_ERROR((this->m_p_row_data != nullptr) ? rocsparse_status_internal_error
                                                                  : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR(this->m_p_col_data->validate());
        RETURN_IF_ROCSPARSE_ERROR(
            ((this->m_rows * this->get_ell_width()) != this->m_p_col_data->get_size())
                ? rocsparse_status_internal_error
                : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR((this->m_p_mat_descr->base != this->m_p_col_data->get_base())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR((this->m_block_dim != 1) ? rocsparse_status_internal_error
                                                           : rocsparse_status_success);
        RETURN_IF_ROCSPARSE_ERROR((1 != this->m_p_col_data->get_inc())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);
        return rocsparse_status_success;
    }

    case rocsparse_format_bell:
    {
        RETURN_IF_ROCSPARSE_ERROR((this->m_p_row_data != nullptr) ? rocsparse_status_internal_error
                                                                  : rocsparse_status_success);
        RETURN_IF_ROCSPARSE_ERROR(this->m_p_col_data->validate());
        RETURN_IF_ROCSPARSE_ERROR(
            ((this->get_rows() * this->get_ell_cols()) != this->m_p_col_data->get_size())
                ? rocsparse_status_internal_error
                : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR((this->m_p_mat_descr->base != this->m_p_col_data->get_base())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR((this->m_block_dim != 1) ? rocsparse_status_internal_error
                                                           : rocsparse_status_success);
        RETURN_IF_ROCSPARSE_ERROR((1 != this->m_p_col_data->get_inc())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);
        return rocsparse_status_success;
    }

    case rocsparse_format_sell:
    {
        RETURN_IF_ROCSPARSE_ERROR(this->m_p_row_data->validate());
        const int64_t nslices = (this->get_rows() - 1) / this->get_sell_slice_size() + 1;

        RETURN_IF_ROCSPARSE_ERROR(((nslices + 1) != this->m_p_row_data->get_size())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR((this->m_p_mat_descr->base != this->m_p_row_data->get_base())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR(this->m_p_col_data->validate());
        RETURN_IF_ROCSPARSE_ERROR(((this->get_sell_colval_size()) != this->m_p_col_data->get_size())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR((this->m_p_mat_descr->base != this->m_p_col_data->get_base())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR((this->m_block_dim != 1) ? rocsparse_status_internal_error
                                                           : rocsparse_status_success);
        RETURN_IF_ROCSPARSE_ERROR((1 != this->m_p_col_data->get_inc())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);
        RETURN_IF_ROCSPARSE_ERROR((1 != this->m_p_col_data->get_inc())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);
        return rocsparse_status_success;
    }

    case rocsparse_format_bsr:
    {
        RETURN_IF_ROCSPARSE_ERROR(this->m_p_row_data->validate());
        RETURN_IF_ROCSPARSE_ERROR((this->m_rows + 1 != this->m_p_row_data->get_size())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR((this->m_p_mat_descr->base != this->m_p_row_data->get_base())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR(this->m_p_col_data->validate());
        RETURN_IF_ROCSPARSE_ERROR((this->m_nnz != this->m_p_col_data->get_size())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR((this->m_p_mat_descr->base != this->m_p_col_data->get_base())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR((this->m_block_dim == 0) ? rocsparse_status_internal_error
                                                           : rocsparse_status_success);
        RETURN_IF_ROCSPARSE_ERROR((1 != this->m_p_col_data->get_inc())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);
        RETURN_IF_ROCSPARSE_ERROR((1 != this->m_p_col_data->get_inc())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);
        return rocsparse_status_success;
    }

    case rocsparse_format_coo:
    {
        RETURN_IF_ROCSPARSE_ERROR(this->m_p_row_data->validate());
        RETURN_IF_ROCSPARSE_ERROR((this->m_nnz != this->m_p_row_data->get_size())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR((this->m_p_mat_descr->base != this->m_p_row_data->get_base())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR(this->m_p_col_data->validate());
        RETURN_IF_ROCSPARSE_ERROR((this->m_nnz != this->m_p_col_data->get_size())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR((this->m_p_mat_descr->base != this->m_p_col_data->get_base())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR((this->m_block_dim != 1) ? rocsparse_status_internal_error
                                                           : rocsparse_status_success);
        RETURN_IF_ROCSPARSE_ERROR((1 != this->m_p_col_data->get_inc())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);
        RETURN_IF_ROCSPARSE_ERROR((1 != this->m_p_col_data->get_inc())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);
        return rocsparse_status_success;
    }

    case rocsparse_format_coo_aos:
    {
        RETURN_IF_ROCSPARSE_ERROR(this->m_p_row_data->validate());
        RETURN_IF_ROCSPARSE_ERROR((this->m_nnz != this->m_p_row_data->get_size())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR((2 != this->m_p_row_data->get_inc())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR((this->m_p_mat_descr->base != this->m_p_row_data->get_base())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR(this->m_p_col_data->validate());
        RETURN_IF_ROCSPARSE_ERROR((this->m_nnz != this->m_p_col_data->get_size())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR((2 != this->m_p_col_data->get_inc())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR((this->m_p_mat_descr->base != this->m_p_col_data->get_base())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR((this->m_block_dim != 1) ? rocsparse_status_internal_error
                                                           : rocsparse_status_success);
        return rocsparse_status_success;
    }

    case rocsparse_format_csc:
    {
        RETURN_IF_ROCSPARSE_ERROR(this->m_p_col_data->validate());
        RETURN_IF_ROCSPARSE_ERROR((this->m_cols + 1 != this->m_p_col_data->get_size())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR((this->m_p_mat_descr->base != this->m_p_col_data->get_base())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR(this->m_p_row_data->validate());
        RETURN_IF_ROCSPARSE_ERROR((this->m_nnz != this->m_p_row_data->get_size())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR((this->m_p_mat_descr->base != this->m_p_row_data->get_base())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR((this->m_block_dim != 1) ? rocsparse_status_internal_error
                                                           : rocsparse_status_success);

        RETURN_IF_ROCSPARSE_ERROR((1 != this->m_p_col_data->get_inc())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);
        RETURN_IF_ROCSPARSE_ERROR((1 != this->m_p_col_data->get_inc())
                                      ? rocsparse_status_internal_error
                                      : rocsparse_status_success);
        return rocsparse_status_success;
    }
    }
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
}

void _rocsparse_spattern_descr::define_ell(int64_t               rows,
                                           int64_t               cols,
                                           int64_t               width,
                                           rocsparse_idvec_descr col_data,
                                           rocsparse_mat_descr   mat_descr,
                                           rocsparse_mat_info    mat_info)
{
    this->m_format    = rocsparse_format_ell;
    this->m_rows      = rows;
    this->m_cols      = cols;
    this->m_nnz       = rows * width;
    this->m_ell_width = width;
    // this->m_p_row_data = nullptr;
    this->m_p_col_data = col_data;
    if(col_data)
    {
        this->m_p_row_data->set_indextype(col_data->get_indextype());
    }
    if(mat_descr != nullptr)
    {
        this->m_p_mat_descr = mat_descr;
    }
    else
    {
        // Initialize descriptor
        THROW_IF_ROCSPARSE_ERROR(
            rocsparse_set_mat_index_base(this->get_mat_descr(), col_data->get_base()));
    }
    if(mat_info != nullptr)
    {
        this->m_p_mat_info = mat_info;
    }
}

void _rocsparse_spattern_descr::define_sell(int64_t               rows,
                                            int64_t               cols,
                                            int64_t               nnz,
                                            int64_t               sell_slice_size,
                                            int64_t               sell_colval_size,
                                            rocsparse_idvec_descr row_data,
                                            rocsparse_idvec_descr col_data,
                                            rocsparse_mat_descr   mat_descr,
                                            rocsparse_mat_info    mat_info)
{
    this->m_format     = rocsparse_format_sell;
    this->m_rows       = rows;
    this->m_cols       = cols;
    this->m_nnz        = nnz;
    this->m_p_row_data = row_data;
    this->m_p_col_data = col_data;
    this->set_sell_slice_size(sell_slice_size);
    this->set_sell_colval_size(sell_colval_size);
    if(mat_descr != nullptr)
    {
        this->m_p_mat_descr = mat_descr;
    }
    else
    {
        // Initialize descriptor
        THROW_IF_ROCSPARSE_ERROR(
            rocsparse_set_mat_index_base(this->get_mat_descr(), row_data->get_base()));
    }
    if(mat_info != nullptr)
    {
        this->m_p_mat_info = mat_info;
    }
}

void _rocsparse_spattern_descr::define_bell(int64_t               rows,
                                            int64_t               cols,
                                            int64_t               width,
                                            rocsparse_direction   block_direction,
                                            int64_t               block_dim,
                                            rocsparse_idvec_descr col_data,
                                            rocsparse_mat_descr   mat_descr,
                                            rocsparse_mat_info    mat_info)
{
    this->m_format   = rocsparse_format_bell;
    this->m_rows     = rows;
    this->m_cols     = cols;
    this->m_ell_cols = width;
    //    this->m_p_row_data = nullptr;
    if(col_data)
    {
        this->m_p_row_data->set_indextype(col_data->get_indextype());
    }

    this->m_p_col_data = col_data;
    this->m_nnz        = rows * width;
    this->m_block_dim  = block_dim;
    this->m_block_dir  = block_direction;
    if(mat_descr != nullptr)
    {
        this->m_p_mat_descr = mat_descr;
    }
    else
    {
        // Initialize descriptor
        THROW_IF_ROCSPARSE_ERROR(
            rocsparse_set_mat_index_base(this->get_mat_descr(), col_data->get_base()));
    }
    if(mat_info != nullptr)
    {
        this->m_p_mat_info = mat_info;
    }
}

void _rocsparse_spattern_descr::define_bsr(int64_t               rows,
                                           int64_t               cols,
                                           int64_t               nnz,
                                           rocsparse_direction   block_direction,
                                           int64_t               block_dim,
                                           rocsparse_idvec_descr row_data,
                                           rocsparse_idvec_descr col_data,
                                           rocsparse_mat_descr   mat_descr,
                                           rocsparse_mat_info    mat_info)
{
    this->m_format     = rocsparse_format_bsr;
    this->m_rows       = rows;
    this->m_cols       = cols;
    this->m_nnz        = nnz;
    this->m_p_row_data = row_data;
    this->m_p_col_data = col_data;
    this->m_block_dim  = block_dim;
    this->m_block_dir  = block_direction;
    if(mat_descr != nullptr)
    {
        this->m_p_mat_descr = mat_descr;
    }
    else
    {
        THROW_IF_ROCSPARSE_ERROR(
            rocsparse_set_mat_index_base(this->get_mat_descr(), col_data->get_base()));
    }
    if(mat_info != nullptr)
    {
        this->m_p_mat_info = mat_info;
    }
}
