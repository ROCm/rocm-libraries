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

#include "internal/auxiliary/rocsparse_idvec_descr.h"
#include "rocsparse_argdescr.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_datatype_utils.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_logging.hpp"
#include "rocsparse_spattern_descr.hpp"

extern "C" rocsparse_status rocsparse_spmat_descr_destroy(rocsparse_handle      handle,
                                                          rocsparse_spmat_descr descr,
                                                          rocsparse_error*      p_error)
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
    {
        return false;
    }
    }
    return true;
}

extern "C" rocsparse_status rocsparse_spmat_descr_create(rocsparse_handle         handle,
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
    p_descr[0] = new _rocsparse_spmat_descr;
    p_descr[0]->define(spattern, values);
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

            *reinterpret_cast<int64_t*>(p_value) = descr->get_total_rows();
            return rocsparse_status_success;
        }
        case rocsparse_spmat_prop_cols:
        {
            ROCSPARSE_CHECKARG(4,
                               value_size_in_bytes,
                               (sizeof(int64_t) != value_size_in_bytes),
                               rocsparse_status_invalid_value);

            *reinterpret_cast<int64_t*>(p_value) = descr->get_total_cols();
            return rocsparse_status_success;
        }
        case rocsparse_spmat_prop_nnz:
        {
            ROCSPARSE_CHECKARG(4,
                               value_size_in_bytes,
                               (sizeof(int64_t) != value_size_in_bytes),
                               rocsparse_status_invalid_value);
            *reinterpret_cast<int64_t*>(p_value) = descr->get_total_nnz();
            return rocsparse_status_success;
        }
            // LCOV_EXCL_START
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    }
    catch(...)
    {
        RETURN_ROCSPARSE_EXCEPTION();
    }
    // LCOV_EXCL_STOP
}

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
