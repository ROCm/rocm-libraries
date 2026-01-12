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
#include "internal/auxiliary/rocsparse_idvec_descr.h"
#include "rocsparse_argdescr.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_datatype_utils.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_logging.hpp"

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spattern_prop value_)
{
    switch(value_)
    {
    case rocsparse_spattern_prop_format:
    case rocsparse_spattern_prop_rows:
    case rocsparse_spattern_prop_cols:
    case rocsparse_spattern_prop_ell_width:
    case rocsparse_spattern_prop_bell_width:
    case rocsparse_spattern_prop_nnz:
    case rocsparse_spattern_prop_batchtype:
    case rocsparse_spattern_prop_batch_count:
    {
        return false;
    }
    }
    return true;
}

void _rocsparse_spattern_descr::set_row_data(rocsparse_idvec_descr data)
{
    this->row_data = data;
}

void _rocsparse_spattern_descr::set_col_data(rocsparse_idvec_descr data)
{
    this->row_data = data;
}

int64_t _rocsparse_spattern_descr::get_batch_count() const
{
    const int64_t row_batch_count
        = (this->get_row_data() != nullptr) ? this->get_row_data()->get_batch_count() : 0;
    const int64_t col_batch_count
        = (this->get_col_data() != nullptr) ? this->get_col_data()->get_batch_count() : 0;
    return std::max(row_batch_count, col_batch_count);
}

int64_t* _rocsparse_spattern_descr::get_pnnz()
{
    return &this->nnz;
}

void _rocsparse_spattern_descr::set_own_data(bool value)
{
    this->own_data = value;
}

_rocsparse_spattern_descr::~_rocsparse_spattern_descr()
{
    WARNING_IF_ROCSPARSE_ERROR(rocsparse_destroy_mat_descr(this->mat_descr));
    if(this->own_data)
    {
        if(this->row_data)
            delete this->row_data;
        if(this->col_data)
            delete this->col_data;
    }
    this->row_data = nullptr;
    this->col_data = nullptr;
}

int64_t _rocsparse_spattern_descr::get_rows() const
{
    return this->rows;
}
int64_t _rocsparse_spattern_descr::get_cols() const
{
    return this->cols;
}
int64_t _rocsparse_spattern_descr::get_nnz() const
{
    return this->nnz;
}
int64_t _rocsparse_spattern_descr::get_ell_width() const
{
    return this->ell_width;
}

void _rocsparse_spattern_descr::set_ell_width(int64_t value)
{
    this->ell_width = value;
}

void _rocsparse_spattern_descr::set_ell_cols(int64_t value)
{
    this->ell_cols = value;
}

void _rocsparse_spattern_descr::set_nnz(int64_t value)
{
    this->nnz = value;
}

int64_t _rocsparse_spattern_descr::get_ell_cols() const
{
    return this->ell_cols;
}
rocsparse_format _rocsparse_spattern_descr::get_format() const
{
    return this->format;
}
rocsparse_idvec_descr _rocsparse_spattern_descr::get_row_data()
{
    return this->row_data;
}
rocsparse_idvec_descr _rocsparse_spattern_descr::get_col_data()
{
    return this->col_data;
}
rocsparse_const_idvec_descr _rocsparse_spattern_descr::get_row_data() const
{
    return this->row_data;
}
rocsparse_const_idvec_descr _rocsparse_spattern_descr::get_col_data() const
{
    return this->col_data;
}

rocsparse_mat_descr _rocsparse_spattern_descr::get_mat_descr()
{
    return this->mat_descr;
}

const _rocsparse_mat_descr* _rocsparse_spattern_descr::get_mat_descr() const
{
    return this->mat_descr;
}

//
// Constructor.
//
_rocsparse_spattern_descr::_rocsparse_spattern_descr(rocsparse_format      format_,
                                                     int64_t               rows_,
                                                     int64_t               cols_,
                                                     int64_t               nnz_,
                                                     rocsparse_idvec_descr row_data_,
                                                     rocsparse_idvec_descr col_data_,
                                                     rocsparse_mat_descr   mat_descr_)
    : mat_descr(mat_descr_)
    , format(format_)
    , rows(rows_)
    , cols(cols_)
    , row_data(row_data_)
    , col_data(col_data_)
    , nnz(nnz_)
    , ell_width(0)
    , ell_cols(0)
    , own_data(false)
{
}

_rocsparse_spattern_descr* _rocsparse_spattern_descr::create_sell(int64_t               rows,
                                                                  int64_t               cols,
                                                                  int64_t               nnz,
                                                                  rocsparse_idvec_descr row_data,
                                                                  rocsparse_idvec_descr col_data)
{
    _rocsparse_spattern_descr* descr = new _rocsparse_spattern_descr();
    descr->format                    = rocsparse_format_sell;
    descr->rows                      = rows;
    descr->cols                      = cols;
    descr->nnz                       = nnz;
    descr->row_data                  = row_data;
    descr->col_data                  = col_data;
    THROW_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&descr->mat_descr));
    // Initialize descriptor
    THROW_IF_ROCSPARSE_ERROR(
        rocsparse_set_mat_index_base(descr->get_mat_descr(), row_data->get_base()));
    return descr;
}

#if 1

_rocsparse_spattern_descr* _rocsparse_spattern_descr::create_csr(int64_t               rows,
                                                                 int64_t               cols,
                                                                 int64_t               nnz,
                                                                 rocsparse_idvec_descr row_data,
                                                                 rocsparse_idvec_descr col_data)
{
    _rocsparse_spattern_descr* descr = new _rocsparse_spattern_descr();
    descr->format                    = rocsparse_format_csr;
    descr->rows                      = rows;
    descr->cols                      = cols;
    descr->nnz                       = nnz;
    descr->row_data                  = row_data;
    descr->col_data                  = col_data;
    THROW_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&descr->mat_descr));
    // Initialize descriptor
    THROW_IF_ROCSPARSE_ERROR(
        rocsparse_set_mat_index_base(descr->get_mat_descr(), row_data->get_base()));
    return descr;
}

_rocsparse_spattern_descr* _rocsparse_spattern_descr::create_csc(int64_t               rows,
                                                                 int64_t               cols,
                                                                 int64_t               nnz,
                                                                 rocsparse_idvec_descr row_data,
                                                                 rocsparse_idvec_descr col_data)
{
    _rocsparse_spattern_descr* descr = new _rocsparse_spattern_descr();
    descr->format                    = rocsparse_format_csc;
    descr->rows                      = rows;
    descr->cols                      = cols;
    descr->nnz                       = nnz;
    descr->row_data                  = row_data;
    descr->col_data                  = col_data;
    // Initialize descriptor
    THROW_IF_ROCSPARSE_ERROR(
        rocsparse_set_mat_index_base(descr->get_mat_descr(), row_data->get_base()));
    return descr;
}

_rocsparse_spattern_descr* _rocsparse_spattern_descr::create_coo_aos(int64_t               rows,
                                                                     int64_t               cols,
                                                                     int64_t               nnz,
                                                                     rocsparse_idvec_descr row_data,
                                                                     rocsparse_idvec_descr col_data)
{
    _rocsparse_spattern_descr* descr = new _rocsparse_spattern_descr();
    descr->format                    = rocsparse_format_coo_aos;
    descr->rows                      = rows;
    descr->cols                      = cols;
    descr->nnz                       = nnz;
    descr->row_data                  = row_data;
    descr->col_data                  = col_data;
    // Initialize descriptor
    THROW_IF_ROCSPARSE_ERROR(
        rocsparse_set_mat_index_base(descr->get_mat_descr(), row_data->get_base()));
    return descr;
}

_rocsparse_spattern_descr* _rocsparse_spattern_descr::create_coo(int64_t               rows,
                                                                 int64_t               cols,
                                                                 int64_t               nnz,
                                                                 rocsparse_idvec_descr row_data,
                                                                 rocsparse_idvec_descr col_data)
{
    _rocsparse_spattern_descr* descr = new _rocsparse_spattern_descr();
    descr->format                    = rocsparse_format_coo;
    descr->rows                      = rows;
    descr->cols                      = cols;
    descr->nnz                       = nnz;
    descr->row_data                  = row_data;
    descr->col_data                  = col_data;
    // Initialize descriptor
    THROW_IF_ROCSPARSE_ERROR(
        rocsparse_set_mat_index_base(descr->get_mat_descr(), row_data->get_base()));
    return descr;
}

_rocsparse_spattern_descr* _rocsparse_spattern_descr::create_ell(int64_t               rows,
                                                                 int64_t               cols,
                                                                 int64_t               width,
                                                                 rocsparse_idvec_descr col_data)
{
    _rocsparse_spattern_descr* descr = new _rocsparse_spattern_descr();
    descr->format                    = rocsparse_format_coo;
    descr->rows                      = rows;
    descr->cols                      = cols;
    descr->nnz                       = rows * width;
    descr->ell_width                 = width;
    descr->row_data                  = nullptr;
    descr->col_data                  = col_data;
    // Initialize descriptor
    THROW_IF_ROCSPARSE_ERROR(
        rocsparse_set_mat_index_base(descr->get_mat_descr(), col_data->get_base()));
    return descr;
}

_rocsparse_spattern_descr* _rocsparse_spattern_descr::create_bell(int64_t               rows,
                                                                  int64_t               cols,
                                                                  int64_t               width,
                                                                  rocsparse_idvec_descr col_data)
{
    _rocsparse_spattern_descr* descr = new _rocsparse_spattern_descr();
    descr->format                    = rocsparse_format_bell;
    descr->rows                      = rows;
    descr->cols                      = cols;
    descr->ell_cols                  = width;
    descr->row_data                  = nullptr;
    descr->col_data                  = col_data;
    descr->nnz                       = rows * width;
    // Initialize descriptor
    THROW_IF_ROCSPARSE_ERROR(
        rocsparse_set_mat_index_base(descr->get_mat_descr(), col_data->get_base()));
    return descr;
}

_rocsparse_spattern_descr* _rocsparse_spattern_descr::create_bsr(int64_t               rows,
                                                                 int64_t               cols,
                                                                 int64_t               nnz,
                                                                 rocsparse_idvec_descr row_data,
                                                                 rocsparse_idvec_descr col_data)
{
    _rocsparse_spattern_descr* descr = new _rocsparse_spattern_descr();
    descr->format                    = rocsparse_format_bsr;
    descr->rows                      = rows;
    descr->cols                      = cols;
    descr->nnz                       = nnz;
    descr->row_data                  = row_data;
    descr->col_data                  = col_data;
    return descr;
}

#endif

extern "C" rocsparse_status rocsparse_spattern_get_prop(rocsparse_handle               handle,
                                                        rocsparse_const_spattern_descr descr,
                                                        rocsparse_spattern_prop        prop,
                                                        void*                          p_value,
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
        case rocsparse_spattern_prop_format:
        {
            ROCSPARSE_CHECKARG(4,
                               value_size_in_bytes,
                               (sizeof(rocsparse_format) != value_size_in_bytes),
                               rocsparse_status_invalid_value);
            *reinterpret_cast<rocsparse_format*>(p_value) = descr->get_format();
            return rocsparse_status_success;
        }
        case rocsparse_spattern_prop_batchtype:
        {
            return rocsparse_status_success;
        }

        case rocsparse_spattern_prop_batch_count:
        {
            ROCSPARSE_CHECKARG(4,
                               value_size_in_bytes,
                               (sizeof(int64_t) != value_size_in_bytes),
                               rocsparse_status_invalid_value);

            *reinterpret_cast<int64_t*>(p_value) = descr->get_batch_count();
            return rocsparse_status_success;
        }

        case rocsparse_spattern_prop_ell_width:
        {
            ROCSPARSE_CHECKARG(4,
                               value_size_in_bytes,
                               (sizeof(int64_t) != value_size_in_bytes),
                               rocsparse_status_invalid_value);

            *reinterpret_cast<int64_t*>(p_value) = descr->get_ell_width();
            return rocsparse_status_success;
        }

        case rocsparse_spattern_prop_bell_width:
        {
            ROCSPARSE_CHECKARG(4,
                               value_size_in_bytes,
                               (sizeof(int64_t) != value_size_in_bytes),
                               rocsparse_status_invalid_value);

            *reinterpret_cast<int64_t*>(p_value) = descr->get_ell_cols();
            return rocsparse_status_success;
        }

        case rocsparse_spattern_prop_rows:
        {
            ROCSPARSE_CHECKARG(4,
                               value_size_in_bytes,
                               (sizeof(int64_t) != value_size_in_bytes),
                               rocsparse_status_invalid_value);

            *reinterpret_cast<int64_t*>(p_value) = descr->get_rows();
            return rocsparse_status_success;
        }
        case rocsparse_spattern_prop_cols:
        {
            ROCSPARSE_CHECKARG(4,
                               value_size_in_bytes,
                               (sizeof(int64_t) != value_size_in_bytes),
                               rocsparse_status_invalid_value);

            *reinterpret_cast<int64_t*>(p_value) = descr->get_cols();
            return rocsparse_status_success;
        }
        case rocsparse_spattern_prop_nnz:
        {
            ROCSPARSE_CHECKARG(4,
                               value_size_in_bytes,
                               (sizeof(int64_t) != value_size_in_bytes),
                               rocsparse_status_invalid_value);
            *reinterpret_cast<int64_t*>(p_value) = descr->get_nnz();
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

extern "C" rocsparse_status rocsparse_spattern_set_prop(rocsparse_handle         handle,
                                                        rocsparse_spattern_descr descr,
                                                        rocsparse_spattern_prop  prop,
                                                        const void*              p_value,
                                                        size_t           value_size_in_bytes,
                                                        rocsparse_error* p_error)
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
    case rocsparse_spattern_prop_format:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(rocsparse_format) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        descr->set_format(*reinterpret_cast<const rocsparse_format*>(p_value));
        return rocsparse_status_success;
    }
    case rocsparse_spattern_prop_batch_count:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);

        descr->set_batch_count(*reinterpret_cast<const int64_t*>(p_value));
        return rocsparse_status_success;
    }
    case rocsparse_spattern_prop_rows:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);

        descr->set_rows(*reinterpret_cast<const int64_t*>(p_value));
        return rocsparse_status_success;
    }
    case rocsparse_spattern_prop_cols:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);

        descr->set_cols(*reinterpret_cast<const int64_t*>(p_value));
        return rocsparse_status_success;
    }
    case rocsparse_spattern_prop_nnz:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        descr->set_nnz(*reinterpret_cast<const int64_t*>(p_value));
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

extern "C" rocsparse_status rocsparse_spattern_get_row_data(rocsparse_handle               handle,
                                                            rocsparse_const_spattern_descr descr,
                                                            rocsparse_idvec_descr*         p_data,
                                                            rocsparse_error*               p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG_POINTER(2, p_data);
    p_data[0] = (rocsparse_idvec_descr)descr->get_row_data();
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_spattern_get_col_data(rocsparse_handle               handle,
                                                            rocsparse_const_spattern_descr descr,
                                                            rocsparse_idvec_descr*         p_data,
                                                            rocsparse_error*               p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG_POINTER(2, p_data);
    p_data[0] = (rocsparse_idvec_descr)descr->get_col_data();
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_spattern_set_row_data(rocsparse_handle         handle,
                                                            rocsparse_spattern_descr descr,
                                                            rocsparse_idvec_descr    data,
                                                            rocsparse_error*         p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    descr->set_row_data(data);
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_spattern_set_col_data(rocsparse_handle         handle,
                                                            rocsparse_spattern_descr descr,
                                                            rocsparse_idvec_descr    data,
                                                            rocsparse_error*         p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    descr->set_col_data(data);
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_create_csr(rocsparse_handle          handle,
                                               rocsparse_spattern_descr* p_descr,
                                               int64_t                   rows,
                                               int64_t                   cols,
                                               int64_t                   nnz,
                                               rocsparse_idvec_descr     row_data,
                                               rocsparse_idvec_descr     col_data,
                                               rocsparse_error*          p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, p_descr);
    ROCSPARSE_CHECKARG_SIZE(2, rows);
    ROCSPARSE_CHECKARG_SIZE(3, cols);
    ROCSPARSE_CHECKARG_SIZE(4, nnz);
    ROCSPARSE_CHECKARG_POINTER(5, row_data);
    ROCSPARSE_CHECKARG_POINTER(6, col_data);
    p_descr[0] = _rocsparse_spattern_descr::create_csr(rows, cols, nnz, row_data, col_data);
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_create_bsr(rocsparse_handle          handle,
                                               rocsparse_spattern_descr* p_descr,
                                               int64_t                   rowsb,
                                               int64_t                   colsb,
                                               int64_t                   nnzb,
                                               rocsparse_idvec_descr     row_data,
                                               rocsparse_idvec_descr     col_data,
                                               rocsparse_error*          p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, p_descr);
    ROCSPARSE_CHECKARG_SIZE(2, rowsb);
    ROCSPARSE_CHECKARG_SIZE(3, colsb);
    ROCSPARSE_CHECKARG_SIZE(4, nnzb);
    ROCSPARSE_CHECKARG_POINTER(5, row_data);
    ROCSPARSE_CHECKARG_POINTER(6, col_data);
    p_descr[0] = _rocsparse_spattern_descr::create_bsr(rowsb, colsb, nnzb, row_data, col_data);
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_create_csc(rocsparse_handle          handle,
                                               rocsparse_spattern_descr* p_descr,
                                               int64_t                   rows,
                                               int64_t                   cols,
                                               int64_t                   nnz,
                                               rocsparse_idvec_descr     row_data,
                                               rocsparse_idvec_descr     col_data,
                                               rocsparse_error*          p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, p_descr);
    ROCSPARSE_CHECKARG_SIZE(2, rows);
    ROCSPARSE_CHECKARG_SIZE(3, cols);
    ROCSPARSE_CHECKARG_SIZE(4, nnz);
    ROCSPARSE_CHECKARG_POINTER(5, row_data);
    ROCSPARSE_CHECKARG_POINTER(6, col_data);
    p_descr[0] = _rocsparse_spattern_descr::create_csc(rows, cols, nnz, row_data, col_data);
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_create_coo(rocsparse_handle          handle,
                                               rocsparse_spattern_descr* p_descr,
                                               int64_t                   rows,
                                               int64_t                   cols,
                                               int64_t                   nnz,
                                               rocsparse_idvec_descr     row_data,
                                               rocsparse_idvec_descr     col_data,
                                               rocsparse_error*          p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, p_descr);
    ROCSPARSE_CHECKARG_SIZE(2, rows);
    ROCSPARSE_CHECKARG_SIZE(3, cols);
    ROCSPARSE_CHECKARG_SIZE(4, nnz);
    ROCSPARSE_CHECKARG_POINTER(5, row_data);
    ROCSPARSE_CHECKARG_POINTER(6, col_data);
    p_descr[0] = _rocsparse_spattern_descr::create_coo(rows, cols, nnz, row_data, col_data);
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_create_coo_aos(rocsparse_handle          handle,
                                                   rocsparse_spattern_descr* p_descr,
                                                   int64_t                   rows,
                                                   int64_t                   cols,
                                                   int64_t                   nnz,
                                                   rocsparse_idvec_descr     row_data,
                                                   rocsparse_idvec_descr     col_data,
                                                   rocsparse_error*          p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, p_descr);
    ROCSPARSE_CHECKARG_SIZE(2, rows);
    ROCSPARSE_CHECKARG_SIZE(3, cols);
    ROCSPARSE_CHECKARG_SIZE(4, nnz);
    ROCSPARSE_CHECKARG_POINTER(5, row_data);
    ROCSPARSE_CHECKARG_POINTER(6, col_data);
    p_descr[0] = _rocsparse_spattern_descr::create_coo_aos(rows, cols, nnz, row_data, col_data);
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_create_ell(rocsparse_handle          handle,
                                               rocsparse_spattern_descr* p_descr,
                                               int64_t                   rows,
                                               int64_t                   cols,
                                               int64_t                   width,
                                               rocsparse_idvec_descr     col_data,
                                               rocsparse_error*          p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, p_descr);
    ROCSPARSE_CHECKARG_SIZE(2, rows);
    ROCSPARSE_CHECKARG_SIZE(3, cols);
    ROCSPARSE_CHECKARG_SIZE(4, width);
    ROCSPARSE_CHECKARG_POINTER(5, col_data);
    p_descr[0] = _rocsparse_spattern_descr::create_ell(rows, cols, width, col_data);
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

ROCSPARSE_EXPORT
rocsparse_status rocsparse_spattern_create_bell(rocsparse_handle          handle,
                                                rocsparse_spattern_descr* p_descr,
                                                int64_t                   rowsb,
                                                int64_t                   colsb,
                                                int64_t                   width,
                                                rocsparse_idvec_descr     col_data,
                                                rocsparse_error*          p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, p_descr);
    ROCSPARSE_CHECKARG_SIZE(2, rowsb);
    ROCSPARSE_CHECKARG_SIZE(3, colsb);
    ROCSPARSE_CHECKARG_SIZE(4, width);
    ROCSPARSE_CHECKARG_POINTER(5, col_data);
    p_descr[0] = _rocsparse_spattern_descr::create_bell(rowsb, colsb, width, col_data);
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
