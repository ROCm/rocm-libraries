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

#include "internal/auxiliary/rocsparse_dnvec_descr.h"
#include "rocsparse_argdescr.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_datatype_utils.hpp"
#include "rocsparse_dnvec_descr.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_logging.hpp"
#include "rocsparse_memstat.hpp"

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_dnvec_prop value_)
{
    switch(value_)
    {
    case rocsparse_dnvec_prop_datatype:
    case rocsparse_dnvec_prop_size:
    case rocsparse_dnvec_prop_size_in_bytes:
    case rocsparse_dnvec_prop_inc:
    case rocsparse_dnvec_prop_batchtype:
    case rocsparse_dnvec_prop_batchstorage:
    case rocsparse_dnvec_prop_batch_count:
    case rocsparse_dnvec_prop_batch_dist:
    {
        return false;
    }
    }
    return true;
}

#ifdef __cplusplus
extern "C" {
#endif

rocsparse_status rocsparse_dnvec_descr_destroy(rocsparse_handle      handle,
                                               rocsparse_dnvec_descr descr,
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
// LCOV_EXCL_STOP

rocsparse_status rocsparse_dnvec_descr_create(rocsparse_handle       handle,
                                              rocsparse_dnvec_descr* p_descr,
                                              rocsparse_datatype     datatype,
                                              int64_t                size,
                                              int64_t                inc,
                                              const void*            const_data,
                                              void*                  data,
                                              rocsparse_error*       p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, p_descr);
    ROCSPARSE_CHECKARG_ENUM(2, datatype);
    ROCSPARSE_CHECKARG_SIZE(3, size);
    ROCSPARSE_CHECKARG_ARRAY(5, size, const_data);
    ROCSPARSE_CHECKARG(
        6, data, (data != nullptr && data != const_data), rocsparse_status_invalid_pointer);

    p_descr[0] = new _rocsparse_dnvec_descr(datatype, size, inc, const_data, data);
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_dnvec_descr_create_batch(rocsparse_handle       handle,
                                                    rocsparse_dnvec_descr* p_descr,
                                                    rocsparse_datatype     datatype,
                                                    int64_t                size,
                                                    int64_t                inc,
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
    ROCSPARSE_CHECKARG_ENUM(2, datatype);
    ROCSPARSE_CHECKARG_SIZE(3, size);
    // 5 inc is arbitrary
    ROCSPARSE_CHECKARG_ENUM(5, batch_type);
    ROCSPARSE_CHECKARG_ENUM(6, batch_storage);
    ROCSPARSE_CHECKARG_SIZE(7, batch_count);
    // 8 batch_dist is arbitrary
    ROCSPARSE_CHECKARG_ARRAY(9, size, const_data);
    ROCSPARSE_CHECKARG(
        11, data, (data != nullptr && data != const_data), rocsparse_status_invalid_pointer);
    p_descr[0] = new _rocsparse_dnvec_descr(
        datatype, size, inc, batch_type, batch_storage, batch_count, batch_dist, const_data, data);
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_dnvec_get_prop(rocsparse_handle            handle,
                                          rocsparse_const_dnvec_descr descr,
                                          rocsparse_dnvec_prop        prop,
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
    case rocsparse_dnvec_prop_datatype:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(rocsparse_datatype) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        *reinterpret_cast<rocsparse_datatype*>(p_value) = descr->get_datatype();
        return rocsparse_status_success;
    }
    case rocsparse_dnvec_prop_size:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);

        *reinterpret_cast<int64_t*>(p_value) = descr->get_size();
        return rocsparse_status_success;
    }
    case rocsparse_dnvec_prop_size_in_bytes:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(size_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        *reinterpret_cast<size_t*>(p_value)
            = descr->get_size() * rocsparse::datatype_sizeof(descr->get_datatype());
        return rocsparse_status_success;
    }
    case rocsparse_dnvec_prop_inc:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        *reinterpret_cast<int64_t*>(p_value) = descr->get_inc();
        return rocsparse_status_success;
    }
    case rocsparse_dnvec_prop_batchtype:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(rocsparse_batchtype) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        *reinterpret_cast<rocsparse_batchtype*>(p_value) = descr->get_batch_type();
        return rocsparse_status_success;
    }
    case rocsparse_dnvec_prop_batchstorage:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(rocsparse_batchstorage) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        *reinterpret_cast<rocsparse_batchstorage*>(p_value) = descr->get_batch_storage();
        return rocsparse_status_success;
    }
    case rocsparse_dnvec_prop_batch_count:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);

        *reinterpret_cast<int64_t*>(p_value) = descr->get_batch_count();
        return rocsparse_status_success;
    }
    case rocsparse_dnvec_prop_batch_dist:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        *reinterpret_cast<int64_t*>(p_value) = descr->get_batch_dist();
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

rocsparse_status rocsparse_dnvec_set_prop(rocsparse_handle      handle,
                                          rocsparse_dnvec_descr descr,
                                          rocsparse_dnvec_prop  prop,
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

    switch(prop)
    {
    case rocsparse_dnvec_prop_datatype:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(rocsparse_datatype) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        descr->set_datatype(*reinterpret_cast<const rocsparse_datatype*>(p_value));
        return rocsparse_status_success;
    }
    case rocsparse_dnvec_prop_size:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);

        descr->set_size(*reinterpret_cast<const int64_t*>(p_value));
        return rocsparse_status_success;
    }
    case rocsparse_dnvec_prop_size_in_bytes:
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            rocsparse_status_invalid_value,
            "rocsparse_dnvec_prop_size_in_bytes is a non-mutable property");
        return rocsparse_status_success;
    }
    case rocsparse_dnvec_prop_inc:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        descr->set_inc(*reinterpret_cast<const int64_t*>(p_value));
        return rocsparse_status_success;
    }
    case rocsparse_dnvec_prop_batchtype:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(rocsparse_batchtype) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        descr->set_batch_type(*reinterpret_cast<const rocsparse_batchtype*>(p_value));
        return rocsparse_status_success;
    }
    case rocsparse_dnvec_prop_batchstorage:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(rocsparse_batchstorage) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        descr->set_batch_storage(*reinterpret_cast<const rocsparse_batchstorage*>(p_value));
        return rocsparse_status_success;
    }
    case rocsparse_dnvec_prop_batch_count:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);

        descr->set_batch_count(*reinterpret_cast<const int64_t*>(p_value));
        return rocsparse_status_success;
    }
    case rocsparse_dnvec_prop_batch_dist:
    {
        ROCSPARSE_CHECKARG(4,
                           value_size_in_bytes,
                           (sizeof(int64_t) != value_size_in_bytes),
                           rocsparse_status_invalid_value);
        descr->set_batch_dist(*reinterpret_cast<const int64_t*>(p_value));
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

rocsparse_status rocsparse_dnvec_get_data(rocsparse_handle      handle,
                                          rocsparse_dnvec_descr descr,
                                          void**                p_data,
                                          rocsparse_error*      p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG_POINTER(2, p_data);
    p_data[0] = descr->data();
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_dnvec_get_const_data(rocsparse_handle            handle,
                                                rocsparse_const_dnvec_descr descr,
                                                const void**                p_const_data,
                                                rocsparse_error*            p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG_POINTER(2, p_const_data);

    p_const_data[0] = descr->const_data();
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_dnvec_set_data(rocsparse_handle      handle,
                                          rocsparse_dnvec_descr descr,
                                          void*                 data,
                                          rocsparse_error*      p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    descr->set_data(data);
    descr->set_const_data(data);
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_dnvec_set_const_data(rocsparse_handle      handle,
                                                rocsparse_dnvec_descr descr,
                                                const void*           const_data,
                                                rocsparse_error*      p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    descr->set_data(nullptr);
    descr->set_const_data(const_data);
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
