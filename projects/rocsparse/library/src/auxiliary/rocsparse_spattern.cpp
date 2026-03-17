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
bool rocsparse::enum_utils::is_invalid(rocsparse_spattern_prop value_)
{
    switch(value_)
    {
    case rocsparse_spattern_prop_format:
    case rocsparse_spattern_prop_rows:
    case rocsparse_spattern_prop_cols:
    case rocsparse_spattern_prop_ell_width:
    case rocsparse_spattern_prop_bell_width:
    case rocsparse_spattern_prop_sell_slice_size:
    case rocsparse_spattern_prop_sell_colval_size:
    case rocsparse_spattern_prop_nnz:
    case rocsparse_spattern_prop_block_dir:
    case rocsparse_spattern_prop_block_dim:
    case rocsparse_spattern_prop_batch_count:
    {
        return false;
    }
    }
    return true;
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spattern_data value_)
{
    switch(value_)
    {
    case rocsparse_spattern_data_row:
    case rocsparse_spattern_data_column:
    {
        return false;
    }
    }
    return true;
}

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

        case rocsparse_spattern_prop_sell_slice_size:
        {
            ROCSPARSE_CHECKARG(4,
                               value_size_in_bytes,
                               (sizeof(int64_t) != value_size_in_bytes),
                               rocsparse_status_invalid_value);

            *reinterpret_cast<int64_t*>(p_value) = descr->get_sell_slice_size();
            return rocsparse_status_success;
        }

        case rocsparse_spattern_prop_sell_colval_size:
        {
            ROCSPARSE_CHECKARG(4,
                               value_size_in_bytes,
                               (sizeof(int64_t) != value_size_in_bytes),
                               rocsparse_status_invalid_value);

            *reinterpret_cast<int64_t*>(p_value) = descr->get_sell_colval_size();
            return rocsparse_status_success;
        }

        case rocsparse_spattern_prop_block_dim:
        {
            ROCSPARSE_CHECKARG(4,
                               value_size_in_bytes,
                               (sizeof(int64_t) != value_size_in_bytes),
                               rocsparse_status_invalid_value);

            *reinterpret_cast<int64_t*>(p_value) = descr->get_block_dim();
            return rocsparse_status_success;
        }

        case rocsparse_spattern_prop_block_dir:
        {
            ROCSPARSE_CHECKARG(4,
                               value_size_in_bytes,
                               (sizeof(rocsparse_direction) != value_size_in_bytes),
                               rocsparse_status_invalid_value);

            *reinterpret_cast<rocsparse_direction*>(p_value) = descr->get_block_dir();
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
    case rocsparse_spattern_prop_block_dir:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(rocsparse_direction) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        descr->set_block_dir(*reinterpret_cast<const rocsparse_direction*>(p_value));
        return rocsparse_status_success;
    }

    case rocsparse_spattern_prop_ell_width:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        descr->set_ell_width(*reinterpret_cast<const int64_t*>(p_value));
        return rocsparse_status_success;
    }

    case rocsparse_spattern_prop_bell_width:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        descr->set_ell_cols(*reinterpret_cast<const int64_t*>(p_value));
        return rocsparse_status_success;
    }

    case rocsparse_spattern_prop_sell_slice_size:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        descr->set_sell_slice_size(*reinterpret_cast<const int64_t*>(p_value));
        return rocsparse_status_success;
    }

    case rocsparse_spattern_prop_sell_colval_size:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        descr->set_sell_colval_size(*reinterpret_cast<const int64_t*>(p_value));
        return rocsparse_status_success;
    }

    case rocsparse_spattern_prop_block_dim:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        descr->set_block_dim(*reinterpret_cast<const int64_t*>(p_value));
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

extern "C" rocsparse_status rocsparse_spattern_get_data(rocsparse_handle               handle,
                                                        rocsparse_const_spattern_descr descr,
                                                        rocsparse_spattern_data spattern_data,
                                                        rocsparse_idvec_descr*  p_data,
                                                        rocsparse_error*        p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG_ENUM(2, spattern_data);
    ROCSPARSE_CHECKARG_POINTER(3, p_data);
    switch(spattern_data)
    {
    case rocsparse_spattern_data_row:
    {
        p_data[0] = (rocsparse_idvec_descr)descr->get_row_data();
        return rocsparse_status_success;
    }
    case rocsparse_spattern_data_column:
    {
        p_data[0] = (rocsparse_idvec_descr)descr->get_col_data();
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

extern "C" rocsparse_status rocsparse_spattern_set_data(rocsparse_handle         handle,
                                                        rocsparse_spattern_descr descr,
                                                        rocsparse_spattern_data  spattern_data,
                                                        rocsparse_idvec_descr    data,
                                                        rocsparse_error*         p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG_ENUM(2, spattern_data);
    ROCSPARSE_CHECKARG_POINTER(3, data);
    switch(spattern_data)
    {
    case rocsparse_spattern_data_row:
    {
        descr->set_row_data(data);

        return rocsparse_status_success;
    }
    case rocsparse_spattern_data_column:
    {
        descr->set_col_data(data);
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

extern "C" rocsparse_status rocsparse_spattern_create_csr(rocsparse_handle          handle,
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
    p_descr[0] = new _rocsparse_spattern_descr;
    p_descr[0]->define_csr(rows, cols, nnz, row_data, col_data, nullptr, nullptr);
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_spattern_create_bsr(rocsparse_handle          handle,
                                                          rocsparse_spattern_descr* p_descr,
                                                          int64_t                   rowsb,
                                                          int64_t                   colsb,
                                                          int64_t                   nnzb,
                                                          rocsparse_direction       block_direction,
                                                          int64_t                   block_dim,
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
    ROCSPARSE_CHECKARG_ENUM(5, block_direction);
    ROCSPARSE_CHECKARG_SIZE(6, block_dim);
    ROCSPARSE_CHECKARG_POINTER(7, row_data);
    ROCSPARSE_CHECKARG_POINTER(8, col_data);
    p_descr[0] = new _rocsparse_spattern_descr;
    p_descr[0]->define_bsr(
        rowsb, colsb, nnzb, block_direction, block_dim, row_data, col_data, nullptr, nullptr);
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_spattern_create_csc(rocsparse_handle          handle,
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
    p_descr[0] = new _rocsparse_spattern_descr;
    p_descr[0]->define_csc(rows, cols, nnz, row_data, col_data, nullptr, nullptr);
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_spattern_create_coo(rocsparse_handle          handle,
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
    p_descr[0] = new _rocsparse_spattern_descr;
    p_descr[0]->define_coo(rows, cols, nnz, row_data, col_data, nullptr, nullptr);
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_spattern_create_coo_aos(rocsparse_handle          handle,
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
    p_descr[0] = new _rocsparse_spattern_descr;
    p_descr[0]->define_coo_aos(rows, cols, nnz, row_data, col_data, nullptr, nullptr);
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_spattern_create_ell(rocsparse_handle          handle,
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
    p_descr[0] = new _rocsparse_spattern_descr;
    p_descr[0]->define_ell(rows, cols, width, col_data, nullptr, nullptr);
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_spattern_destroy(rocsparse_handle         handle,
                                                       rocsparse_spattern_descr descr,
                                                       rocsparse_error*         p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    if(descr)
    {
        RETURN_IF_ROCSPARSE_ERROR(descr->destroy(handle->stream));
        delete descr;
    }
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

// LCOV_EXCL_STOP
extern "C" rocsparse_status rocsparse_spattern_create_bell(rocsparse_handle          handle,
                                                           rocsparse_spattern_descr* p_descr,
                                                           int64_t                   rowsb,
                                                           int64_t                   colsb,
                                                           int64_t                   width,
                                                           rocsparse_direction   block_direction,
                                                           int64_t               block_dim,
                                                           rocsparse_idvec_descr col_data,
                                                           rocsparse_error*      p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, p_descr);
    ROCSPARSE_CHECKARG_SIZE(2, rowsb);
    ROCSPARSE_CHECKARG_SIZE(3, colsb);
    ROCSPARSE_CHECKARG_SIZE(4, width);
    ROCSPARSE_CHECKARG_ENUM(5, block_direction);
    ROCSPARSE_CHECKARG_SIZE(6, block_dim);
    ROCSPARSE_CHECKARG_POINTER(7, col_data);
    p_descr[0] = new _rocsparse_spattern_descr;
    p_descr[0]->define_bell(
        rowsb, colsb, width, block_direction, block_dim, col_data, nullptr, nullptr);
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
