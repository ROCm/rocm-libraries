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

#include "rocsparse_dnmat_descr.hpp"
#include "internal/auxiliary/rocsparse_dnmat_descr.h"
#include "rocsparse_argdescr.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_datatype_utils.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_logging.hpp"

rocsparse_status _rocsparse_dnmat_descr::destroy(rocsparse_handle handle)
{
    return rocsparse_status_success;
}

_rocsparse_dnmat_descr::_rocsparse_dnmat_descr(rocsparse_datatype datatype_,
                                               rocsparse_order    order_,
                                               int64_t            rows_,
                                               int64_t            cols_,
                                               int64_t            ld_,
                                               const void*        const_values_,
                                               void*              values_)
    : init(true)
    , rows(rows_)
    , cols(cols_)
    , ld(ld_)
    , values(values_)
    , const_values(const_values_)
    , data_type(datatype_)
    , order(order_)
    , batch_count(1)
    , batch_stride(0)
    , batch_type(rocsparse_batchtype_strided)
{
}

_rocsparse_dnmat_descr::_rocsparse_dnmat_descr(rocsparse_datatype     datatype_,
                                               rocsparse_order        order_,
                                               int64_t                rows_,
                                               int64_t                cols_,
                                               int64_t                ld_,
                                               rocsparse_batchtype    batchtype_,
                                               rocsparse_batchstorage batch_storage_,
                                               int64_t                batch_count_,
                                               int64_t                batch_dist_,
                                               const void*            const_values_,
                                               void*                  values_)
    : init(true)
    , rows(rows_)
    , cols(cols_)
    , ld(ld_)
    , values(values_)
    , const_values(const_values_)
    , data_type(datatype_)
    , order(order_)
    , batch_count(batch_count_)
    , batch_stride(batch_dist_)
    , batch_type(batchtype_)
    , batch_storage(batch_storage_)
{
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_dnmat_prop value_)
{
    switch(value_)
    {
    case rocsparse_dnmat_prop_datatype:
    case rocsparse_dnmat_prop_order:
    case rocsparse_dnmat_prop_rows:
    case rocsparse_dnmat_prop_cols:
    case rocsparse_dnmat_prop_ld:
    case rocsparse_dnmat_prop_batchtype:
    case rocsparse_dnmat_prop_batchstorage:
    case rocsparse_dnmat_prop_batch_count:
    case rocsparse_dnmat_prop_batch_dist:
    {
        return false;
    }
    }
    return true;
}

#ifdef __cplusplus
extern "C" {
#endif

rocsparse_status rocsparse_dnmat_destroy(rocsparse_handle      handle,
                                         rocsparse_dnmat_descr descr,
                                         rocsparse_error*      p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    if(descr)
    {
        RETURN_IF_ROCSPARSE_ERROR(descr->destroy(handle));
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

rocsparse_status rocsparse_dnmat_create(rocsparse_handle       handle,
                                        rocsparse_dnmat_descr* p_descr,
                                        rocsparse_datatype     data_type,
                                        rocsparse_order        order,
                                        int64_t                rows,
                                        int64_t                cols,
                                        int64_t                ld,
                                        const void*            const_data,
                                        void*                  data,
                                        rocsparse_error*       p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, p_descr);
    ROCSPARSE_CHECKARG_ENUM(2, data_type);
    ROCSPARSE_CHECKARG_ENUM(3, order);
    ROCSPARSE_CHECKARG_SIZE(4, rows);
    ROCSPARSE_CHECKARG_SIZE(5, cols);
    // 6 ld is arbitrary
    ROCSPARSE_CHECKARG_ARRAY(7, rows * cols, const_data);
    ROCSPARSE_CHECKARG(
        8, data, (data != nullptr && data != const_data), rocsparse_status_invalid_pointer);
    p_descr[0] = new _rocsparse_dnmat_descr(data_type, order, rows, cols, ld, const_data, data);
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_dnmat_create_batched(rocsparse_handle       handle,
                                                rocsparse_dnmat_descr* p_descr,
                                                rocsparse_datatype     data_type,
                                                rocsparse_order        order,
                                                int64_t                rows,
                                                int64_t                cols,
                                                int64_t                ld,
                                                rocsparse_batchtype    batch_type,
                                                rocsparse_batchstorage batch_storage,
                                                int64_t                batch_count,
                                                int64_t                batch_dist,
                                                const void*            const_data,
                                                void*                  data,
                                                rocsparse_error*       p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, p_descr);
    ROCSPARSE_CHECKARG_ENUM(2, data_type);
    ROCSPARSE_CHECKARG_ENUM(3, order);
    ROCSPARSE_CHECKARG_SIZE(4, rows);
    ROCSPARSE_CHECKARG_SIZE(5, cols);
    // 6 ld is arbitrary
    ROCSPARSE_CHECKARG_ENUM(7, batch_type);
    ROCSPARSE_CHECKARG_ENUM(8, batch_storage);
    ROCSPARSE_CHECKARG_SIZE(9, batch_count);
    // 10 batch_dist is arbitrary
    ROCSPARSE_CHECKARG_ARRAY(11, rows * cols, const_data);
    ROCSPARSE_CHECKARG(
        12, data, (data != nullptr && data != const_data), rocsparse_status_invalid_pointer);

    p_descr[0] = new _rocsparse_dnmat_descr(data_type,
                                            order,
                                            rows,
                                            cols,
                                            ld,
                                            batch_type,
                                            batch_storage,
                                            batch_count,
                                            batch_dist,
                                            const_data,
                                            data);
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_dnmat_get_prop(rocsparse_handle            handle,
                                          rocsparse_const_dnmat_descr descr,
                                          rocsparse_dnmat_prop        prop,
                                          void*                       p_value,
                                          size_t                      value_size_in_bytes,
                                          rocsparse_error*            p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG_ENUM(2, prop);
    ROCSPARSE_CHECKARG_POINTER(3, p_value);

    switch(prop)
    {
    case rocsparse_dnmat_prop_datatype:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(rocsparse_datatype) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        *reinterpret_cast<rocsparse_datatype*>(p_value) = descr->data_type;
        return rocsparse_status_success;
    }
    case rocsparse_dnmat_prop_order:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(rocsparse_order) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        *reinterpret_cast<rocsparse_order*>(p_value) = descr->order;
        return rocsparse_status_success;
    }
    case rocsparse_dnmat_prop_rows:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);

        *reinterpret_cast<int64_t*>(p_value) = descr->rows;
        return rocsparse_status_success;
    }
    case rocsparse_dnmat_prop_cols:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);

        *reinterpret_cast<int64_t*>(p_value) = descr->cols;
        return rocsparse_status_success;
    }
    case rocsparse_dnmat_prop_ld:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        *reinterpret_cast<int64_t*>(p_value) = descr->ld;
        return rocsparse_status_success;
    }
    case rocsparse_dnmat_prop_batchtype:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(rocsparse_batchtype) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        *reinterpret_cast<rocsparse_batchtype*>(p_value) = descr->batch_type;
        return rocsparse_status_success;
    }
    case rocsparse_dnmat_prop_batchstorage:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(rocsparse_batchstorage) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        *reinterpret_cast<rocsparse_batchstorage*>(p_value) = descr->batch_storage;
        return rocsparse_status_success;
    }
    case rocsparse_dnmat_prop_batch_count:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);

        *reinterpret_cast<int64_t*>(p_value) = descr->batch_count;
        return rocsparse_status_success;
    }
    case rocsparse_dnmat_prop_batch_dist:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        *reinterpret_cast<int64_t*>(p_value) = descr->batch_stride;
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

rocsparse_status rocsparse_dnmat_set_prop(rocsparse_handle      handle,
                                          rocsparse_dnmat_descr descr,
                                          rocsparse_dnmat_prop  prop,
                                          const void*           p_const_value,
                                          size_t                value_size_in_bytes,
                                          rocsparse_error*      p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG_ENUM(2, prop);
    ROCSPARSE_CHECKARG_POINTER(3, p_const_value);

    switch(prop)
    {
    case rocsparse_dnmat_prop_datatype:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(rocsparse_datatype) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        descr->data_type = *reinterpret_cast<const rocsparse_datatype*>(p_const_value);
        return rocsparse_status_success;
    }
    case rocsparse_dnmat_prop_order:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(rocsparse_order) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        descr->order = *reinterpret_cast<const rocsparse_order*>(p_const_value);
        return rocsparse_status_success;
    }
    case rocsparse_dnmat_prop_rows:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);

        descr->rows = *reinterpret_cast<const int64_t*>(p_const_value);
        return rocsparse_status_success;
    }
    case rocsparse_dnmat_prop_cols:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);

        descr->cols = *reinterpret_cast<const int64_t*>(p_const_value);
        return rocsparse_status_success;
    }
    case rocsparse_dnmat_prop_ld:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        descr->ld = *reinterpret_cast<const int64_t*>(p_const_value);
        return rocsparse_status_success;
    }
    case rocsparse_dnmat_prop_batchtype:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(rocsparse_batchtype) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        descr->batch_type = *reinterpret_cast<const rocsparse_batchtype*>(p_const_value);
        return rocsparse_status_success;
    }
    case rocsparse_dnmat_prop_batchstorage:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(rocsparse_batchstorage) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        descr->batch_storage = *reinterpret_cast<const rocsparse_batchstorage*>(p_const_value);
        return rocsparse_status_success;
    }
    case rocsparse_dnmat_prop_batch_count:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);

        descr->batch_count = *reinterpret_cast<const int64_t*>(p_const_value);
        return rocsparse_status_success;
    }
    case rocsparse_dnmat_prop_batch_dist:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        descr->batch_stride = *reinterpret_cast<const int64_t*>(p_const_value);
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

rocsparse_status rocsparse_dnmat_get_data(rocsparse_handle            handle,
                                          rocsparse_const_dnmat_descr descr,
                                          void**                      p_data,
                                          rocsparse_error*            p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG_POINTER(2, p_data);

    p_data[0] = descr->values;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_dnmat_get_const_data(rocsparse_handle            handle,
                                                rocsparse_const_dnmat_descr descr,
                                                const void**                p_const_data,
                                                rocsparse_error*            p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG_POINTER(2, p_const_data);

    p_const_data[0] = descr->const_values;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_dnmat_set_data(rocsparse_handle      handle,
                                          rocsparse_dnmat_descr descr,
                                          void*                 data,
                                          rocsparse_error*      p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);

    descr->values       = data;
    descr->const_values = data;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_dnmat_set_const_data(rocsparse_handle      handle,
                                                rocsparse_dnmat_descr descr,
                                                const void*           data,
                                                rocsparse_error*      p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    descr->values       = nullptr;
    descr->const_values = data;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

#ifdef __cplusplus
}
#endif
