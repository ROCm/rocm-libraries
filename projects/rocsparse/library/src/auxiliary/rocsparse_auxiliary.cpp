/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse.h"
#include "rocsparse_control.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_utility.hpp"
#include <iomanip>
#include <map>

#include <hip/hip_runtime_api.h>

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_status status)
{
    switch(status)
    {
    case rocsparse_status_success:
        return "success";
    case rocsparse_status_invalid_handle:
        return "invalid handle";
    case rocsparse_status_not_implemented:
        return "not implemented";
    case rocsparse_status_invalid_pointer:
        return "invalid pointer";
    case rocsparse_status_invalid_size:
        return "invalid size";
    case rocsparse_status_memory_error:
        return "memory error";
    case rocsparse_status_internal_error:
        return "internal error";
        // LCOV_EXCL_START
    case rocsparse_status_invalid_value:
        return "invalid value";
        // LCOV_EXCL_STOP
    case rocsparse_status_arch_mismatch:
        return "arch mismatch";
    case rocsparse_status_zero_pivot:
        return "zero pivot";
    case rocsparse_status_not_initialized:
        return "not initialized";
    case rocsparse_status_type_mismatch:
        return "type mismatch";
    case rocsparse_status_requires_sorted_storage:
        return "requires sorted storage";
    case rocsparse_status_thrown_exception:
        return "thrown exception";
    case rocsparse_status_continue:
        return "continue";
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_pointer_mode value)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value)
    {
        CASE(rocsparse_pointer_mode_device);
        CASE(rocsparse_pointer_mode_host);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_spmat_attribute value)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value)
    {
        CASE(rocsparse_spmat_fill_mode);
        CASE(rocsparse_spmat_diag_type);
        CASE(rocsparse_spmat_matrix_type);
        CASE(rocsparse_spmat_storage_mode);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_diag_type value)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value)
    {
        CASE(rocsparse_diag_type_unit);
        CASE(rocsparse_diag_type_non_unit);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_fill_mode value_)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value_)
    {
        CASE(rocsparse_fill_mode_lower);
        CASE(rocsparse_fill_mode_upper);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_storage_mode value_)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value_)
    {
        CASE(rocsparse_storage_mode_sorted);
        CASE(rocsparse_storage_mode_unsorted);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_index_base value_)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value_)
    {
        CASE(rocsparse_index_base_zero);
        CASE(rocsparse_index_base_one);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_matrix_type value_)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value_)
    {
        CASE(rocsparse_matrix_type_general);
        CASE(rocsparse_matrix_type_symmetric);
        CASE(rocsparse_matrix_type_hermitian);
        CASE(rocsparse_matrix_type_triangular);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_direction value_)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value_)
    {
        CASE(rocsparse_direction_row);
        CASE(rocsparse_direction_column);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_operation value_)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value_)
    {
        CASE(rocsparse_operation_none);
        CASE(rocsparse_operation_transpose);
        CASE(rocsparse_operation_conjugate_transpose);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_indextype value_)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value_)
    {
        CASE(rocsparse_indextype_u16);
        CASE(rocsparse_indextype_i32);
        CASE(rocsparse_indextype_i64);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_datatype value_)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value_)
    {
        CASE(rocsparse_datatype_f16_r);
        CASE(rocsparse_datatype_f32_r);
        CASE(rocsparse_datatype_f64_r);
        CASE(rocsparse_datatype_f32_c);
        CASE(rocsparse_datatype_f64_c);
        CASE(rocsparse_datatype_i8_r);
        CASE(rocsparse_datatype_u8_r);
        CASE(rocsparse_datatype_i32_r);
        CASE(rocsparse_datatype_u32_r);
        CASE(rocsparse_datatype_bf16_r);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_order value_)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value_)
    {
        CASE(rocsparse_order_row);
        CASE(rocsparse_order_column);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_action value)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value)
    {
        CASE(rocsparse_action_numeric);
        CASE(rocsparse_action_symbolic);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_solve_policy value_)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value_)
    {
        CASE(rocsparse_solve_policy_auto);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_analysis_policy value_)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value_)
    {
        CASE(rocsparse_analysis_policy_reuse);
        CASE(rocsparse_analysis_policy_force);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_format value_)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value_)
    {
        CASE(rocsparse_format_coo);
        CASE(rocsparse_format_coo_aos);
        CASE(rocsparse_format_csr);
        CASE(rocsparse_format_csc);
        CASE(rocsparse_format_ell);
        CASE(rocsparse_format_bell);
        CASE(rocsparse_format_bsr);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spmat_attribute value)
{
    switch(value)
    {
    case rocsparse_spmat_fill_mode:
    case rocsparse_spmat_diag_type:
    case rocsparse_spmat_matrix_type:
    case rocsparse_spmat_storage_mode:
    {
        return false;
    }
    }
    return true;
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_pointer_mode value)
{
    switch(value)
    {
    case rocsparse_pointer_mode_device:
    case rocsparse_pointer_mode_host:
    {
        return false;
    }
    }
    return true;
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_diag_type value)
{
    switch(value)
    {
    case rocsparse_diag_type_unit:
    case rocsparse_diag_type_non_unit:
    {
        return false;
    }
    }
    return true;
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_fill_mode value_)
{
    switch(value_)
    {
    case rocsparse_fill_mode_lower:
    case rocsparse_fill_mode_upper:
    {
        return false;
    }
    }
    return true;
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_storage_mode value_)
{
    switch(value_)
    {
    case rocsparse_storage_mode_sorted:
    case rocsparse_storage_mode_unsorted:
    {
        return false;
    }
    }
    return true;
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_index_base value_)
{
    switch(value_)
    {
    case rocsparse_index_base_zero:
    case rocsparse_index_base_one:
    {
        return false;
    }
    }
    return true;
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_matrix_type value_)
{
    switch(value_)
    {
    case rocsparse_matrix_type_general:
    case rocsparse_matrix_type_symmetric:
    case rocsparse_matrix_type_hermitian:
    case rocsparse_matrix_type_triangular:
    {
        return false;
    }
    }
    return true;
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_direction value_)
{
    switch(value_)
    {
    case rocsparse_direction_row:
    case rocsparse_direction_column:
    {
        return false;
    }
    }
    return true;
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_operation value_)
{
    switch(value_)
    {
    case rocsparse_operation_none:
    case rocsparse_operation_transpose:
    case rocsparse_operation_conjugate_transpose:
    {
        return false;
    }
    }
    return true;
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_indextype value_)
{
    switch(value_)
    {
    case rocsparse_indextype_u16:
    case rocsparse_indextype_i32:
    case rocsparse_indextype_i64:
    {
        return false;
    }
    }
    return true;
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_datatype value_)
{
    switch(value_)
    {
    case rocsparse_datatype_f16_r:
    case rocsparse_datatype_f32_r:
    case rocsparse_datatype_f64_r:
    case rocsparse_datatype_f32_c:
    case rocsparse_datatype_f64_c:
    case rocsparse_datatype_i8_r:
    case rocsparse_datatype_u8_r:
    case rocsparse_datatype_i32_r:
    case rocsparse_datatype_u32_r:
    case rocsparse_datatype_bf16_r:
    {
        return false;
    }
    }
    return true;
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_order value_)
{
    switch(value_)
    {
    case rocsparse_order_row:
    case rocsparse_order_column:
    {
        return false;
    }
    }
    return true;
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_action value)
{
    switch(value)
    {
    case rocsparse_action_numeric:
    case rocsparse_action_symbolic:
    {
        return false;
    }
    }
    return true;
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_solve_policy value_)
{
    switch(value_)
    {
    case rocsparse_solve_policy_auto:
    {
        return false;
    }
    }
    return true;
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_analysis_policy value_)
{
    switch(value_)
    {
    case rocsparse_analysis_policy_reuse:
    case rocsparse_analysis_policy_force:
    {
        return false;
    }
    }
    return true;
}

#ifdef __cplusplus
extern "C" {
#endif

/********************************************************************************
 * \brief rocsparse_handle is a structure holding the rocsparse library context.
 * It must be initialized using rocsparse_create_handle()
 * and the returned handle must be passed
 * to all subsequent library function calls.
 * It should be destroyed at the end using rocsparse_destroy_handle().
 *******************************************************************************/
rocsparse_status rocsparse_create_handle(rocsparse_handle* handle)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, handle);
    *handle = new _rocsparse_handle();
    rocsparse::log_trace(*handle, "rocsparse_create_handle");
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief destroy handle
 *******************************************************************************/
rocsparse_status rocsparse_destroy_handle(rocsparse_handle handle)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    rocsparse::log_trace(handle, "rocsparse_destroy_handle");
    delete handle;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief Get rocSPARSE status enum name as a string
 *******************************************************************************/
const char* rocsparse_get_status_name(rocsparse_status status)
{
    switch(status)
    {
    case rocsparse_status_success:
        return "rocsparse_status_success";
    case rocsparse_status_invalid_handle:
        return "rocsparse_status_invalid_handle";
    case rocsparse_status_not_implemented:
        return "rocsparse_status_not_implemented";
    case rocsparse_status_invalid_pointer:
        return "rocsparse_status_invalid_pointer";
    case rocsparse_status_invalid_size:
        return "rocsparse_status_invalid_size";
    case rocsparse_status_memory_error:
        return "rocsparse_status_memory_error";
    case rocsparse_status_internal_error:
        return "rocsparse_status_internal_error";
        // LCOV_EXCL_START
    case rocsparse_status_invalid_value:
        return "rocsparse_status_invalid_value";
        // LCOV_EXCL_STOP
    case rocsparse_status_arch_mismatch:
        return "rocsparse_status_arch_mismatch";
    case rocsparse_status_zero_pivot:
        return "rocsparse_status_zero_pivot";
    case rocsparse_status_not_initialized:
        return "rocsparse_status_not_initialized";
    case rocsparse_status_type_mismatch:
        return "rocsparse_status_type_mismatch";
    case rocsparse_status_requires_sorted_storage:
        return "rocsparse_status_requires_sorted_storage";
    case rocsparse_status_thrown_exception:
        return "rocsparse_status_thrown_exception";
    case rocsparse_status_continue:
        return "rocsparse_status_continue";
    }

    return "Unrecognized status code";
}

/********************************************************************************
 * \brief Get rocSPARSE status enum description as a string
 *******************************************************************************/
const char* rocsparse_get_status_description(rocsparse_status status)
{
    switch(status)
    {
    case rocsparse_status_success:
        return "rocsparse operation was successful";
    case rocsparse_status_invalid_handle:
        return "handle not initialized, invalid or null";
    case rocsparse_status_not_implemented:
        return "function is not implemented";
    case rocsparse_status_invalid_pointer:
        return "invalid pointer parameter";
    case rocsparse_status_invalid_size:
        return "invalid size parameter";
    case rocsparse_status_memory_error:
        return "failed memory allocation, copy, dealloc";
    case rocsparse_status_internal_error:
        return "other internal library failure";
        // LCOV_EXCL_START
    case rocsparse_status_invalid_value:
        return "invalid value parameter";
        // LCOV_EXCL_STOP
    case rocsparse_status_arch_mismatch:
        return "device arch is not supported";
    case rocsparse_status_zero_pivot:
        return "encountered zero pivot";
    case rocsparse_status_not_initialized:
        return "descriptor has not been initialized";
    case rocsparse_status_type_mismatch:
        return "index types do not match";
    case rocsparse_status_requires_sorted_storage:
        return "sorted storage required";
    case rocsparse_status_thrown_exception:
        return "exception being thrown";
    case rocsparse_status_continue:
        return "nothing preventing function to proceed";
    }

    return "Unrecognized status code";
}

/********************************************************************************
 * \brief Indicates whether the scalar value pointers are on the host or device.
 * Set pointer mode, can be host or device
 *******************************************************************************/
rocsparse_status rocsparse_set_pointer_mode(rocsparse_handle handle, rocsparse_pointer_mode mode)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, mode);
    handle->pointer_mode = mode;
    rocsparse::log_trace(handle, "rocsparse_set_pointer_mode", mode);

    RETURN_IF_ROCSPARSE_ERROR(handle->set_pointer_mode(mode));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief Get pointer mode, can be host or device.
 *******************************************************************************/
rocsparse_status rocsparse_get_pointer_mode(rocsparse_handle handle, rocsparse_pointer_mode* mode)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, mode);
    *mode = handle->pointer_mode;
    rocsparse::log_trace(handle, "rocsparse_get_pointer_mode", *mode);

    RETURN_IF_ROCSPARSE_ERROR(handle->get_pointer_mode(mode));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 *! \brief Set rocsparse stream used for all subsequent library function calls.
 * If not set, all hip kernels will take the default NULL stream.
 *******************************************************************************/
rocsparse_status rocsparse_set_stream(rocsparse_handle handle, hipStream_t stream_id)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    rocsparse::log_trace(handle, "rocsparse_set_stream", stream_id);

    RETURN_IF_ROCSPARSE_ERROR(handle->set_stream(stream_id));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 *! \brief Get rocsparse stream used for all subsequent library function calls.
 *******************************************************************************/
rocsparse_status rocsparse_get_stream(rocsparse_handle handle, hipStream_t* stream_id)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    rocsparse::log_trace(handle, "rocsparse_get_stream", *stream_id);

    RETURN_IF_ROCSPARSE_ERROR(handle->get_stream(stream_id));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief Get rocSPARSE version
 * version % 100        = patch level
 * version / 100 % 1000 = minor version
 * version / 100000     = major version
 *******************************************************************************/
rocsparse_status rocsparse_get_version(rocsparse_handle handle, int* version)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    *version = ROCSPARSE_VERSION_MAJOR * 100000 + ROCSPARSE_VERSION_MINOR * 100
               + ROCSPARSE_VERSION_PATCH;

    rocsparse::log_trace(handle, "rocsparse_get_version", *version);

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief Get rocSPARSE git revision
 *******************************************************************************/
rocsparse_status rocsparse_get_git_rev(rocsparse_handle handle, char* rev)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, rev);

    static constexpr char v[] = TO_STR(ROCSPARSE_VERSION_TWEAK);

    memcpy(rev, v, sizeof(v));

    rocsparse::log_trace(handle, "rocsparse_get_git_rev", rev);

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_create_mat_descr_t is a structure holding the rocsparse matrix
 * descriptor. It must be initialized using rocsparse_create_mat_descr()
 * and the returned handle must be passed to all subsequent library function
 * calls that involve the matrix.
 * It should be destroyed at the end using rocsparse_destroy_mat_descr().
 *******************************************************************************/
rocsparse_status rocsparse_create_mat_descr(rocsparse_mat_descr* descr)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    *descr = new _rocsparse_mat_descr;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief copy matrix descriptor
 *******************************************************************************/
rocsparse_status rocsparse_copy_mat_descr(rocsparse_mat_descr dest, const rocsparse_mat_descr src)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, dest);
    ROCSPARSE_CHECKARG_POINTER(1, src);
    ROCSPARSE_CHECKARG(1, src, (src == dest), rocsparse_status_invalid_pointer);

    dest->type         = src->type;
    dest->fill_mode    = src->fill_mode;
    dest->diag_type    = src->diag_type;
    dest->base         = src->base;
    dest->storage_mode = src->storage_mode;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief destroy matrix descriptor
 *******************************************************************************/
rocsparse_status rocsparse_destroy_mat_descr(rocsparse_mat_descr descr)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    delete descr;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief Set the index base of the matrix descriptor.
 *******************************************************************************/
rocsparse_status rocsparse_set_mat_index_base(rocsparse_mat_descr descr, rocsparse_index_base base)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_ENUM(1, base);
    descr->base = base;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief Returns the index base of the matrix descriptor.
 *******************************************************************************/
rocsparse_index_base rocsparse_get_mat_index_base(const rocsparse_mat_descr descr)
{
    ROCSPARSE_ROUTINE_TRACE;

    // If descriptor is invalid, default index base is returned
    if(descr == nullptr)
    {
        return rocsparse_index_base_zero;
    }
    return descr->base;
}

/********************************************************************************
 * \brief Set the matrix type of the matrix descriptor.
 *******************************************************************************/
rocsparse_status rocsparse_set_mat_type(rocsparse_mat_descr descr, rocsparse_matrix_type type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_ENUM(1, type);

    descr->type = type;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief Returns the matrix type of the matrix descriptor.
 *******************************************************************************/
rocsparse_matrix_type rocsparse_get_mat_type(const rocsparse_mat_descr descr)
{
    ROCSPARSE_ROUTINE_TRACE;

    // If descriptor is invalid, default matrix type is returned
    if(descr == nullptr)
    {
        return rocsparse_matrix_type_general;
    }
    return descr->type;
}

rocsparse_status rocsparse_set_mat_fill_mode(rocsparse_mat_descr descr,
                                             rocsparse_fill_mode fill_mode)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_ENUM(1, fill_mode);

    descr->fill_mode = fill_mode;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_fill_mode rocsparse_get_mat_fill_mode(const rocsparse_mat_descr descr)
{
    ROCSPARSE_ROUTINE_TRACE;

    // If descriptor is invalid, default fill mode is returned
    if(descr == nullptr)
    {
        return rocsparse_fill_mode_lower;
    }
    return descr->fill_mode;
}

rocsparse_status rocsparse_set_mat_diag_type(rocsparse_mat_descr descr,
                                             rocsparse_diag_type diag_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_ENUM(1, diag_type);
    descr->diag_type = diag_type;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_diag_type rocsparse_get_mat_diag_type(const rocsparse_mat_descr descr)
{
    ROCSPARSE_ROUTINE_TRACE;

    // If descriptor is invalid, default diagonal type is returned
    if(descr == nullptr)
    {
        return rocsparse_diag_type_non_unit;
    }
    return descr->diag_type;
}

rocsparse_status rocsparse_set_mat_storage_mode(rocsparse_mat_descr    descr,
                                                rocsparse_storage_mode storage_mode)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_ENUM(1, storage_mode);
    descr->storage_mode = storage_mode;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_storage_mode rocsparse_get_mat_storage_mode(const rocsparse_mat_descr descr)
{
    ROCSPARSE_ROUTINE_TRACE;

    // If descriptor is invalid, default fill mode is returned
    if(descr == nullptr)
    {
        return rocsparse_storage_mode_sorted;
    }
    return descr->storage_mode;
}

/********************************************************************************
 * \brief rocsparse_create_hyb_mat is a structure holding the rocsparse HYB
 * matrix. It must be initialized using rocsparse_create_hyb_mat()
 * and the retured handle must be passed to all subsequent library function
 * calls that involve the HYB matrix.
 * It should be destroyed at the end using rocsparse_destroy_hyb_mat().
 *******************************************************************************/
rocsparse_status rocsparse_create_hyb_mat(rocsparse_hyb_mat* hyb)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, hyb);
    *hyb = new _rocsparse_hyb_mat;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief Copy HYB matrix.
 *******************************************************************************/
rocsparse_status rocsparse_copy_hyb_mat(rocsparse_hyb_mat dest, const rocsparse_hyb_mat src)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, dest);
    ROCSPARSE_CHECKARG_POINTER(1, src);
    ROCSPARSE_CHECKARG(1, src, (src == dest), rocsparse_status_invalid_pointer);

    // check if destination already contains data. If it does, verify its allocated arrays are the same size as source
    bool previously_created = false;
    previously_created |= (dest->m != 0);
    previously_created |= (dest->n != 0);
    previously_created |= (dest->partition != rocsparse_hyb_partition_auto);
    previously_created |= (dest->ell_nnz != 0);
    previously_created |= (dest->ell_width != 0);
    previously_created |= (dest->ell_col_ind != nullptr);
    previously_created |= (dest->ell_val != nullptr);
    previously_created |= (dest->coo_nnz != 0);
    previously_created |= (dest->coo_row_ind != nullptr);
    previously_created |= (dest->coo_col_ind != nullptr);
    previously_created |= (dest->coo_val != nullptr);
    previously_created |= (dest->data_type_T != rocsparse_datatype_f32_r);

    if(previously_created)
    {
        // Sparsity pattern of dest and src must match
        bool invalid = false;
        invalid |= (dest->m != src->m);
        invalid |= (dest->n != src->n);
        invalid |= (dest->partition != src->partition);
        invalid |= (dest->ell_width != src->ell_width);
        invalid |= (dest->ell_nnz != src->ell_nnz);
        invalid |= (dest->coo_nnz != src->coo_nnz);
        invalid |= (dest->data_type_T != src->data_type_T);

        if(invalid)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
        }
    }

    size_t T_size = sizeof(float);
    switch(src->data_type_T)
    {
    case rocsparse_datatype_f16_r:
    {
        T_size = sizeof(_Float16);
        break;
    }
    case rocsparse_datatype_bf16_r:
    {
        T_size = sizeof(rocsparse_bfloat16);
        break;
    }
    case rocsparse_datatype_f32_r:
    {
        T_size = sizeof(float);
        break;
    }
    case rocsparse_datatype_f64_r:
    {
        T_size = sizeof(double);
        break;
    }
    case rocsparse_datatype_f32_c:
    {
        T_size = sizeof(rocsparse_float_complex);
        break;
    }
    case rocsparse_datatype_f64_c:
    {
        T_size = sizeof(rocsparse_double_complex);
        break;
    }
    case rocsparse_datatype_i8_r:
    {
        T_size = sizeof(int8_t);
        break;
    }
    case rocsparse_datatype_u8_r:
    {
        T_size = sizeof(uint8_t);
        break;
    }
    case rocsparse_datatype_i32_r:
    {
        T_size = sizeof(int32_t);
        break;
    }
    case rocsparse_datatype_u32_r:
    {
        T_size = sizeof(uint32_t);
        break;
    }
    }

    if(src->ell_col_ind != nullptr)
    {
        if(dest->ell_col_ind == nullptr)
        {
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMalloc(&dest->ell_col_ind, sizeof(rocsparse_int) * src->ell_nnz));
        }
        RETURN_IF_HIP_ERROR(hipMemcpy(dest->ell_col_ind,
                                      src->ell_col_ind,
                                      sizeof(rocsparse_int) * src->ell_nnz,
                                      hipMemcpyDeviceToDevice));
    }

    if(src->ell_val != nullptr)
    {
        if(dest->ell_val == nullptr)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMalloc(&dest->ell_val, T_size * src->ell_nnz));
        }
        RETURN_IF_HIP_ERROR(
            hipMemcpy(dest->ell_val, src->ell_val, T_size * src->ell_nnz, hipMemcpyDeviceToDevice));
    }

    if(src->coo_row_ind != nullptr)
    {
        if(dest->coo_row_ind == nullptr)
        {
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMalloc(&dest->coo_row_ind, sizeof(rocsparse_int) * src->coo_nnz));
        }
        RETURN_IF_HIP_ERROR(hipMemcpy(dest->coo_row_ind,
                                      src->coo_row_ind,
                                      sizeof(rocsparse_int) * src->coo_nnz,
                                      hipMemcpyDeviceToDevice));
    }

    if(src->coo_col_ind != nullptr)
    {
        if(dest->coo_col_ind == nullptr)
        {
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMalloc(&dest->coo_col_ind, sizeof(rocsparse_int) * src->coo_nnz));
        }
        RETURN_IF_HIP_ERROR(hipMemcpy(dest->coo_col_ind,
                                      src->coo_col_ind,
                                      sizeof(rocsparse_int) * src->coo_nnz,
                                      hipMemcpyDeviceToDevice));
    }

    if(src->coo_val != nullptr)
    {
        if(dest->coo_val == nullptr)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMalloc(&dest->coo_val, T_size * src->coo_nnz));
        }
        RETURN_IF_HIP_ERROR(
            hipMemcpy(dest->coo_val, src->coo_val, T_size * src->coo_nnz, hipMemcpyDeviceToDevice));
    }

    dest->m           = src->m;
    dest->n           = src->n;
    dest->partition   = src->partition;
    dest->ell_width   = src->ell_width;
    dest->ell_nnz     = src->ell_nnz;
    dest->coo_nnz     = src->coo_nnz;
    dest->data_type_T = src->data_type_T;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief Destroy HYB matrix.
 *******************************************************************************/
rocsparse_status rocsparse_destroy_hyb_mat(rocsparse_hyb_mat hyb)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, hyb);

    // Due to the changes in the hipFree introduced in HIP 7.0
    // https://rocm.docs.amd.com/projects/HIP/en/latest/hip-7-changes.html#update-hipfree
    // we need to introduce a device synchronize here as the below hipFree calls are now asynchronous.
    // hipFree() previously had an implicit wait for synchronization purpose which is applicable for all memory allocations.
    // This wait has been disabled in the HIP 7.0 runtime for allocations made with hipMallocAsync and hipMallocFromPoolAsync.
    RETURN_IF_HIP_ERROR(hipDeviceSynchronize());

    // Clean up ELL part
    if(hyb->ell_col_ind != nullptr)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFree(hyb->ell_col_ind));
    }
    if(hyb->ell_val != nullptr)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFree(hyb->ell_val));
    }

    // Clean up COO part
    if(hyb->coo_row_ind != nullptr)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFree(hyb->coo_row_ind));
    }
    if(hyb->coo_col_ind != nullptr)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFree(hyb->coo_col_ind));
    }
    if(hyb->coo_val != nullptr)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFree(hyb->coo_val));
    }

    delete hyb;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_mat_info is a structure holding the matrix info data that is
 * gathered during the analysis routines. It must be initialized by calling
 * rocsparse_create_mat_info() and the returned info structure must be passed
 * to all subsequent function calls that require additional information. It
 * should be destroyed at the end using rocsparse_destroy_mat_info().
 *******************************************************************************/
rocsparse_status rocsparse_create_mat_info(rocsparse_mat_info* info)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, info);
    *info = new _rocsparse_mat_info;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief Copy mat info.
 *******************************************************************************/
rocsparse_status rocsparse_copy_mat_info(rocsparse_mat_info dest, const rocsparse_mat_info src)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, dest);
    ROCSPARSE_CHECKARG_POINTER(1, src);
    ROCSPARSE_CHECKARG(1, src, (src == dest), rocsparse_status_invalid_pointer);

    dest->duplicate_trdata(src, 0);

    rocsparse_csrmv_info src_csrmv_info  = src->get_csrmv_info();
    rocsparse_csrmv_info dest_csrmv_info = dest->get_csrmv_info();
    if(src_csrmv_info != nullptr)
    {
        if(dest_csrmv_info == nullptr)
        {
            dest_csrmv_info = new _rocsparse_csrmv_info();
            dest->set_csrmv_info(dest_csrmv_info);
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::copy_csrmv_info(dest_csrmv_info, src_csrmv_info));
    }

    rocsparse_bsrmv_info src_bsrmv_info  = src->get_bsrmv_info();
    rocsparse_bsrmv_info dest_bsrmv_info = dest->get_bsrmv_info();
    if(src_bsrmv_info != nullptr)
    {
        if(dest_bsrmv_info == nullptr)
        {
            dest_bsrmv_info = new _rocsparse_bsrmv_info();
            dest->set_bsrmv_info(dest_bsrmv_info);
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::copy_bsrmv_info(dest_bsrmv_info, src_bsrmv_info));
    }

    if(src->csrgemm_info != nullptr)
    {
        if(dest->csrgemm_info == nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::create_csrgemm_info(&dest->csrgemm_info));
        }
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::copy_csrgemm_info(dest->csrgemm_info, src->csrgemm_info));
    }

    if(src->csritsv_info != nullptr)
    {
        if(dest->csritsv_info == nullptr)
        {
            dest->csritsv_info = new _rocsparse_csritsv_info();
        }
        hipStream_t default_stream{};
        dest->csritsv_info->copy(src->csritsv_info, default_stream);
    }

    dest->boost_enable   = src->boost_enable;
    dest->boost_tol_size = src->boost_tol_size;
    dest->boost_tol      = src->boost_tol;
    dest->boost_val      = src->boost_val;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief Destroy mat info.
 *******************************************************************************/
rocsparse_status rocsparse_destroy_mat_info(rocsparse_mat_info info)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    if(info == nullptr)
    {
        return rocsparse_status_success;
    }

    delete info;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_color_info is a structure holding the color info data that is
 * gathered during the analysis routines. It must be initialized by calling
 * rocsparse_create_color_info() and the returned info structure must be passed
 * to all subsequent function calls that require additional information. It
 * should be destroyed at the end using rocsparse_destroy_color_info().
 *******************************************************************************/
rocsparse_status rocsparse_create_color_info(rocsparse_color_info* info)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, info);
    *info = new _rocsparse_color_info;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief Copy color info.
 *******************************************************************************/
rocsparse_status rocsparse_copy_color_info(rocsparse_color_info       dest,
                                           const rocsparse_color_info src)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, dest);
    ROCSPARSE_CHECKARG_POINTER(1, src);
    ROCSPARSE_CHECKARG(1, src, (src == dest), rocsparse_status_invalid_pointer);

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief Destroy color info.
 *******************************************************************************/
rocsparse_status rocsparse_destroy_color_info(rocsparse_color_info info)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    if(info == nullptr)
    {
        return rocsparse_status_success;
    }
    delete info;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_create_spvec_descr creates a descriptor holding the sparse
 * vector data, sizes and properties. It must be called prior to all subsequent
 * library function calls that involve sparse vectors. It should be destroyed at
 * the end using rocsparse_destroy_spvec_descr(). All data pointers remain valid.
 *******************************************************************************/
rocsparse_status rocsparse_create_spvec_descr(rocsparse_spvec_descr* descr,
                                              int64_t                size,
                                              int64_t                nnz,
                                              void*                  indices,
                                              void*                  values,
                                              rocsparse_indextype    idx_type,
                                              rocsparse_index_base   idx_base,
                                              rocsparse_datatype     data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, size);
    ROCSPARSE_CHECKARG_SIZE(2, nnz);
    ROCSPARSE_CHECKARG(2, nnz, (nnz > size), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_ARRAY(3, nnz, indices);
    ROCSPARSE_CHECKARG_ARRAY(4, nnz, values);
    ROCSPARSE_CHECKARG_ENUM(5, idx_type);
    ROCSPARSE_CHECKARG_ENUM(6, idx_base);
    ROCSPARSE_CHECKARG_ENUM(7, data_type);

    *descr = new _rocsparse_spvec_descr;

    (*descr)->init = true;

    (*descr)->size = size;
    (*descr)->nnz  = nnz;

    (*descr)->idx_data = indices;
    (*descr)->val_data = values;

    (*descr)->const_idx_data = indices;
    (*descr)->const_val_data = values;

    (*descr)->idx_type  = idx_type;
    (*descr)->data_type = data_type;

    (*descr)->idx_base = idx_base;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_create_const_spvec_descr(rocsparse_const_spvec_descr* descr,
                                                    int64_t                      size,
                                                    int64_t                      nnz,
                                                    const void*                  indices,
                                                    const void*                  values,
                                                    rocsparse_indextype          idx_type,
                                                    rocsparse_index_base         idx_base,
                                                    rocsparse_datatype           data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, size);
    ROCSPARSE_CHECKARG_SIZE(2, nnz);
    ROCSPARSE_CHECKARG(2, nnz, (nnz > size), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_ARRAY(3, nnz, indices);
    ROCSPARSE_CHECKARG_ARRAY(4, nnz, values);
    ROCSPARSE_CHECKARG_ENUM(5, idx_type);
    ROCSPARSE_CHECKARG_ENUM(6, idx_base);
    ROCSPARSE_CHECKARG_ENUM(7, data_type);

    rocsparse_spvec_descr new_descr = new _rocsparse_spvec_descr;

    new_descr->init = true;

    new_descr->size = size;
    new_descr->nnz  = nnz;

    new_descr->idx_data = nullptr;
    new_descr->val_data = nullptr;

    new_descr->const_idx_data = indices;
    new_descr->const_val_data = values;

    new_descr->idx_type  = idx_type;
    new_descr->data_type = data_type;

    new_descr->idx_base = idx_base;

    *descr = new_descr;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_destroy_spvec_descr destroys a sparse vector descriptor.
 *******************************************************************************/
rocsparse_status rocsparse_destroy_spvec_descr(rocsparse_const_spvec_descr descr)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);

    if(descr->init == false)
    {
        // Do nothing
        return rocsparse_status_success;
    }

    delete descr;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_spvec_get returns the sparse vector matrix data, sizes and
 * properties.
 *******************************************************************************/
rocsparse_status rocsparse_spvec_get(const rocsparse_spvec_descr descr,
                                     int64_t*                    size,
                                     int64_t*                    nnz,
                                     void**                      indices,
                                     void**                      values,
                                     rocsparse_indextype*        idx_type,
                                     rocsparse_index_base*       idx_base,
                                     rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, size);
    ROCSPARSE_CHECKARG_POINTER(2, nnz);
    ROCSPARSE_CHECKARG_POINTER(3, indices);
    ROCSPARSE_CHECKARG_POINTER(4, values);
    ROCSPARSE_CHECKARG_POINTER(5, idx_type);
    ROCSPARSE_CHECKARG_POINTER(6, idx_base);
    ROCSPARSE_CHECKARG_POINTER(7, data_type);

    *size = descr->size;
    *nnz  = descr->nnz;

    *indices = descr->idx_data;
    *values  = descr->val_data;

    *idx_type  = descr->idx_type;
    *idx_base  = descr->idx_base;
    *data_type = descr->data_type;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_const_spvec_get(rocsparse_const_spvec_descr descr,
                                           int64_t*                    size,
                                           int64_t*                    nnz,
                                           const void**                indices,
                                           const void**                values,
                                           rocsparse_indextype*        idx_type,
                                           rocsparse_index_base*       idx_base,
                                           rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, size);
    ROCSPARSE_CHECKARG_POINTER(2, nnz);
    ROCSPARSE_CHECKARG_POINTER(3, indices);
    ROCSPARSE_CHECKARG_POINTER(4, values);
    ROCSPARSE_CHECKARG_POINTER(5, idx_type);
    ROCSPARSE_CHECKARG_POINTER(6, idx_base);
    ROCSPARSE_CHECKARG_POINTER(7, data_type);
    *size = descr->size;
    *nnz  = descr->nnz;

    *indices = descr->const_idx_data;
    *values  = descr->const_val_data;

    *idx_type  = descr->idx_type;
    *idx_base  = descr->idx_base;
    *data_type = descr->data_type;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_spvec_get_index_base returns the sparse vector index base.
 *******************************************************************************/
rocsparse_status rocsparse_spvec_get_index_base(rocsparse_const_spvec_descr descr,
                                                rocsparse_index_base*       idx_base)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, idx_base);

    *idx_base = descr->idx_base;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_spvec_get_values returns the sparse vector value pointer.
 *******************************************************************************/
rocsparse_status rocsparse_spvec_get_values(const rocsparse_spvec_descr descr, void** values)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, values);
    *values = descr->val_data;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_const_spvec_get_values(rocsparse_const_spvec_descr descr,
                                                  const void**                values)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, values);
    *values = descr->const_val_data;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_spvec_set_values sets the sparse vector value pointer.
 *******************************************************************************/
rocsparse_status rocsparse_spvec_set_values(rocsparse_spvec_descr descr, void* values)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, values);
    descr->val_data       = values;
    descr->const_val_data = values;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_create_coo_descr creates a descriptor holding the COO matrix
 * data, sizes and properties. It must be called prior to all subsequent library
 * function calls that involve sparse matrices. It should be destroyed at the end
 * using rocsparse_destroy_spmat_descr(). All data pointers remain valid.
 *******************************************************************************/
rocsparse_status rocsparse_create_coo_descr(rocsparse_spmat_descr* descr,
                                            int64_t                rows,
                                            int64_t                cols,
                                            int64_t                nnz,
                                            void*                  coo_row_ind,
                                            void*                  coo_col_ind,
                                            void*                  coo_val,
                                            rocsparse_indextype    idx_type,
                                            rocsparse_index_base   idx_base,
                                            rocsparse_datatype     data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, rows);
    ROCSPARSE_CHECKARG_SIZE(2, cols);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG(3, nnz, (nnz > rows * cols), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_ARRAY(4, nnz, coo_row_ind);
    ROCSPARSE_CHECKARG_ARRAY(5, nnz, coo_col_ind);
    ROCSPARSE_CHECKARG_ARRAY(6, nnz, coo_val);
    ROCSPARSE_CHECKARG_ENUM(7, idx_type);
    ROCSPARSE_CHECKARG_ENUM(8, idx_base);
    ROCSPARSE_CHECKARG_ENUM(9, data_type);

    *descr = new _rocsparse_spmat_descr;

    (*descr)->init = true;

    (*descr)->rows = rows;
    (*descr)->cols = cols;
    (*descr)->nnz  = nnz;

    (*descr)->row_data = coo_row_ind;
    (*descr)->col_data = coo_col_ind;
    (*descr)->val_data = coo_val;

    (*descr)->const_row_data = coo_row_ind;
    (*descr)->const_col_data = coo_col_ind;
    (*descr)->const_val_data = coo_val;

    (*descr)->row_type  = idx_type;
    (*descr)->col_type  = idx_type;
    (*descr)->data_type = data_type;

    (*descr)->idx_base = idx_base;
    (*descr)->format   = rocsparse_format_coo;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&(*descr)->descr));
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&(*descr)->info));
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_set_mat_index_base((*descr)->descr, idx_base));

    (*descr)->batch_count                 = 1;
    (*descr)->batch_stride                = 0;
    (*descr)->offsets_batch_stride        = 0;
    (*descr)->columns_values_batch_stride = 0;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_create_const_coo_descr(rocsparse_const_spmat_descr* descr,
                                                  int64_t                      rows,
                                                  int64_t                      cols,
                                                  int64_t                      nnz,
                                                  const void*                  coo_row_ind,
                                                  const void*                  coo_col_ind,
                                                  const void*                  coo_val,
                                                  rocsparse_indextype          idx_type,
                                                  rocsparse_index_base         idx_base,
                                                  rocsparse_datatype           data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, rows);
    ROCSPARSE_CHECKARG_SIZE(2, cols);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG(3, nnz, (nnz > rows * cols), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_ARRAY(4, nnz, coo_row_ind);
    ROCSPARSE_CHECKARG_ARRAY(5, nnz, coo_col_ind);
    ROCSPARSE_CHECKARG_ARRAY(6, nnz, coo_val);
    ROCSPARSE_CHECKARG_ENUM(7, idx_type);
    ROCSPARSE_CHECKARG_ENUM(8, idx_base);
    ROCSPARSE_CHECKARG_ENUM(9, data_type);

    rocsparse_spmat_descr new_descr = new _rocsparse_spmat_descr;

    new_descr->init = true;

    new_descr->rows = rows;
    new_descr->cols = cols;
    new_descr->nnz  = nnz;

    new_descr->row_data = nullptr;
    new_descr->col_data = nullptr;
    new_descr->val_data = nullptr;

    new_descr->const_row_data = coo_row_ind;
    new_descr->const_col_data = coo_col_ind;
    new_descr->const_val_data = coo_val;

    new_descr->row_type  = idx_type;
    new_descr->col_type  = idx_type;
    new_descr->data_type = data_type;

    new_descr->idx_base = idx_base;
    new_descr->format   = rocsparse_format_coo;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&new_descr->descr));
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&new_descr->info));

    // Initialize descriptor
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(new_descr->descr, idx_base));

    new_descr->batch_count                 = 1;
    new_descr->batch_stride                = 0;
    new_descr->offsets_batch_stride        = 0;
    new_descr->columns_values_batch_stride = 0;

    *descr = new_descr;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_create_coo_aos_descr creates a descriptor holding the COO matrix
 * data, sizes and properties where the row pointer and column indices are stored
 * using array of structure (AoS) format. It must be called prior to all subsequent
 * library function calls that involve sparse matrices. It should be destroyed at
 * the end using rocsparse_destroy_spmat_descr(). All data pointers remain valid.
 *******************************************************************************/
rocsparse_status rocsparse_create_coo_aos_descr(rocsparse_spmat_descr* descr,
                                                int64_t                rows,
                                                int64_t                cols,
                                                int64_t                nnz,
                                                void*                  coo_ind,
                                                void*                  coo_val,
                                                rocsparse_indextype    idx_type,
                                                rocsparse_index_base   idx_base,
                                                rocsparse_datatype     data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, rows);
    ROCSPARSE_CHECKARG_SIZE(2, cols);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG(3, nnz, (nnz > rows * cols), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_ARRAY(4, nnz, coo_ind);
    ROCSPARSE_CHECKARG_ARRAY(5, nnz, coo_val);
    ROCSPARSE_CHECKARG_ENUM(6, idx_type);
    ROCSPARSE_CHECKARG_ENUM(7, idx_base);
    ROCSPARSE_CHECKARG_ENUM(8, data_type);

    *descr = new _rocsparse_spmat_descr;

    (*descr)->init = true;

    (*descr)->rows = rows;
    (*descr)->cols = cols;
    (*descr)->nnz  = nnz;

    (*descr)->ind_data = coo_ind;
    (*descr)->val_data = coo_val;

    (*descr)->const_ind_data = coo_ind;
    (*descr)->const_val_data = coo_val;

    (*descr)->row_type  = idx_type;
    (*descr)->col_type  = idx_type;
    (*descr)->data_type = data_type;

    (*descr)->idx_base = idx_base;
    (*descr)->format   = rocsparse_format_coo_aos;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&(*descr)->descr));
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&(*descr)->info));

    // Initialize descriptor
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_set_mat_index_base((*descr)->descr, idx_base));

    (*descr)->batch_count                 = 1;
    (*descr)->batch_stride                = 0;
    (*descr)->offsets_batch_stride        = 0;
    (*descr)->columns_values_batch_stride = 0;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_create_csr_descr creates a descriptor holding the CSR matrix
 * data, sizes and properties. It must be called prior to all subsequent library
 * function calls that involve sparse matrices. It should be destroyed at the end
 * using rocsparse_destroy_spmat_descr(). All data pointers remain valid.
 *******************************************************************************/
rocsparse_status rocsparse_create_csr_descr(rocsparse_spmat_descr* descr,
                                            int64_t                rows,
                                            int64_t                cols,
                                            int64_t                nnz,
                                            void*                  csr_row_ptr,
                                            void*                  csr_col_ind,
                                            void*                  csr_val,
                                            rocsparse_indextype    row_ptr_type,
                                            rocsparse_indextype    col_ind_type,
                                            rocsparse_index_base   idx_base,
                                            rocsparse_datatype     data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, rows);
    ROCSPARSE_CHECKARG_SIZE(2, cols);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG(3, nnz, (nnz > rows * cols), rocsparse_status_invalid_size);

    //
    // SWDEV-340500, this is a non-sense.
    // cusparse parity behavior should be fixed in hipsparse, not here.
    //
    //    ROCSPARSE_CHECKARG(4, (rows > 0 && nnz > 0 && csr_row_ptr == nullptr), csr_row_ptr, rocsparse_status_invalid_pointer);
    ROCSPARSE_CHECKARG_ARRAY(4, rows, csr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(5, nnz, csr_col_ind);
    ROCSPARSE_CHECKARG_ARRAY(6, nnz, csr_val);
    ROCSPARSE_CHECKARG_ENUM(7, row_ptr_type);
    ROCSPARSE_CHECKARG_ENUM(8, col_ind_type);
    ROCSPARSE_CHECKARG_ENUM(9, idx_base);
    ROCSPARSE_CHECKARG_ENUM(10, data_type);

    *descr = new _rocsparse_spmat_descr;

    (*descr)->init = true;

    (*descr)->rows = rows;
    (*descr)->cols = cols;
    (*descr)->nnz  = nnz;

    (*descr)->row_data = csr_row_ptr;
    (*descr)->col_data = csr_col_ind;
    (*descr)->val_data = csr_val;

    (*descr)->const_row_data = csr_row_ptr;
    (*descr)->const_col_data = csr_col_ind;
    (*descr)->const_val_data = csr_val;

    (*descr)->row_type  = row_ptr_type;
    (*descr)->col_type  = col_ind_type;
    (*descr)->data_type = data_type;

    (*descr)->idx_base = idx_base;
    (*descr)->format   = rocsparse_format_csr;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&(*descr)->descr));
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&(*descr)->info));

    // Initialize descriptor
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_set_mat_index_base((*descr)->descr, idx_base));

    (*descr)->batch_count                 = 1;
    (*descr)->batch_stride                = 0;
    (*descr)->offsets_batch_stride        = 0;
    (*descr)->columns_values_batch_stride = 0;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_create_const_csr_descr(rocsparse_const_spmat_descr* descr,
                                                  int64_t                      rows,
                                                  int64_t                      cols,
                                                  int64_t                      nnz,
                                                  const void*                  csr_row_ptr,
                                                  const void*                  csr_col_ind,
                                                  const void*                  csr_val,
                                                  rocsparse_indextype          row_ptr_type,
                                                  rocsparse_indextype          col_ind_type,
                                                  rocsparse_index_base         idx_base,
                                                  rocsparse_datatype           data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, rows);
    ROCSPARSE_CHECKARG_SIZE(2, cols);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG(3, nnz, (nnz > rows * cols), rocsparse_status_invalid_size);

    //
    // SWDEV-340500, this is a non-sense.
    // cusparse parity behavior should be fixed in hipsparse, not here.
    //
    //    ROCSPARSE_CHECKARG(4, (rows > 0 && nnz > 0 && csr_row_ptr == nullptr), csr_row_ptr, rocsparse_status_invalid_pointer);
    ROCSPARSE_CHECKARG_ARRAY(4, rows, csr_row_ptr);

    ROCSPARSE_CHECKARG_ARRAY(5, nnz, csr_col_ind);
    ROCSPARSE_CHECKARG_ARRAY(6, nnz, csr_val);
    ROCSPARSE_CHECKARG_ENUM(7, row_ptr_type);
    ROCSPARSE_CHECKARG_ENUM(8, col_ind_type);
    ROCSPARSE_CHECKARG_ENUM(9, idx_base);
    ROCSPARSE_CHECKARG_ENUM(10, data_type);

    rocsparse_spmat_descr new_descr = new _rocsparse_spmat_descr;

    new_descr->init = true;

    new_descr->rows = rows;
    new_descr->cols = cols;
    new_descr->nnz  = nnz;

    new_descr->row_data = nullptr;
    new_descr->col_data = nullptr;
    new_descr->val_data = nullptr;

    new_descr->const_row_data = csr_row_ptr;
    new_descr->const_col_data = csr_col_ind;
    new_descr->const_val_data = csr_val;

    new_descr->row_type  = row_ptr_type;
    new_descr->col_type  = col_ind_type;
    new_descr->data_type = data_type;

    new_descr->idx_base = idx_base;
    new_descr->format   = rocsparse_format_csr;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&new_descr->descr));
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&new_descr->info));

    // Initialize descriptor
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(new_descr->descr, idx_base));

    new_descr->batch_count                 = 1;
    new_descr->batch_stride                = 0;
    new_descr->offsets_batch_stride        = 0;
    new_descr->columns_values_batch_stride = 0;

    *descr = new_descr;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_create_csc_descr creates a descriptor holding the CSC matrix
 * data, sizes and properties. It must be called prior to all subsequent library
 * function calls that involve sparse matrices. It should be destroyed at the end
 * using rocsparse_destroy_spmat_descr(). All data pointers remain valid.
 *******************************************************************************/
rocsparse_status rocsparse_create_csc_descr(rocsparse_spmat_descr* descr,
                                            int64_t                rows,
                                            int64_t                cols,
                                            int64_t                nnz,
                                            void*                  csc_col_ptr,
                                            void*                  csc_row_ind,
                                            void*                  csc_val,
                                            rocsparse_indextype    col_ptr_type,
                                            rocsparse_indextype    row_ind_type,
                                            rocsparse_index_base   idx_base,
                                            rocsparse_datatype     data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, rows);
    ROCSPARSE_CHECKARG_SIZE(2, cols);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG(3, nnz, (nnz > rows * cols), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_ARRAY(4, cols, csc_col_ptr);
    ROCSPARSE_CHECKARG_ARRAY(5, nnz, csc_row_ind);
    ROCSPARSE_CHECKARG_ARRAY(6, nnz, csc_val);
    ROCSPARSE_CHECKARG_ENUM(7, col_ptr_type);
    ROCSPARSE_CHECKARG_ENUM(8, row_ind_type);
    ROCSPARSE_CHECKARG_ENUM(9, idx_base);
    ROCSPARSE_CHECKARG_ENUM(10, data_type);
    *descr = new _rocsparse_spmat_descr;

    (*descr)->init = true;

    (*descr)->rows = rows;
    (*descr)->cols = cols;
    (*descr)->nnz  = nnz;

    (*descr)->row_data = csc_row_ind;
    (*descr)->col_data = csc_col_ptr;
    (*descr)->val_data = csc_val;

    (*descr)->const_row_data = csc_row_ind;
    (*descr)->const_col_data = csc_col_ptr;
    (*descr)->const_val_data = csc_val;

    (*descr)->row_type  = row_ind_type;
    (*descr)->col_type  = col_ptr_type;
    (*descr)->data_type = data_type;

    (*descr)->idx_base = idx_base;
    (*descr)->format   = rocsparse_format_csc;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&(*descr)->descr));
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&(*descr)->info));

    // Initialize descriptor
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_set_mat_index_base((*descr)->descr, idx_base));

    (*descr)->batch_count                 = 1;
    (*descr)->batch_stride                = 0;
    (*descr)->offsets_batch_stride        = 0;
    (*descr)->columns_values_batch_stride = 0;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_create_const_csc_descr(rocsparse_const_spmat_descr* descr,
                                                  int64_t                      rows,
                                                  int64_t                      cols,
                                                  int64_t                      nnz,
                                                  const void*                  csc_col_ptr,
                                                  const void*                  csc_row_ind,
                                                  const void*                  csc_val,
                                                  rocsparse_indextype          col_ptr_type,
                                                  rocsparse_indextype          row_ind_type,
                                                  rocsparse_index_base         idx_base,
                                                  rocsparse_datatype           data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, rows);
    ROCSPARSE_CHECKARG_SIZE(2, cols);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG(3, nnz, (nnz > rows * cols), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_ARRAY(4, cols, csc_col_ptr);
    ROCSPARSE_CHECKARG_ARRAY(5, nnz, csc_row_ind);
    ROCSPARSE_CHECKARG_ARRAY(6, nnz, csc_val);
    ROCSPARSE_CHECKARG_ENUM(7, col_ptr_type);
    ROCSPARSE_CHECKARG_ENUM(8, row_ind_type);
    ROCSPARSE_CHECKARG_ENUM(9, idx_base);
    ROCSPARSE_CHECKARG_ENUM(10, data_type);

    rocsparse_spmat_descr new_descr = new _rocsparse_spmat_descr;

    new_descr->init = true;

    new_descr->rows = rows;
    new_descr->cols = cols;
    new_descr->nnz  = nnz;

    new_descr->row_data = nullptr;
    new_descr->col_data = nullptr;
    new_descr->val_data = nullptr;

    new_descr->const_row_data = csc_row_ind;
    new_descr->const_col_data = csc_col_ptr;
    new_descr->const_val_data = csc_val;

    new_descr->row_type  = row_ind_type;
    new_descr->col_type  = col_ptr_type;
    new_descr->data_type = data_type;

    new_descr->idx_base = idx_base;
    new_descr->format   = rocsparse_format_csc;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&new_descr->descr));
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&new_descr->info));

    // Initialize descriptor
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(new_descr->descr, idx_base));

    new_descr->batch_count                 = 1;
    new_descr->batch_stride                = 0;
    new_descr->offsets_batch_stride        = 0;
    new_descr->columns_values_batch_stride = 0;

    *descr = new_descr;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_create_ell_descr creates a descriptor holding the ELL matrix
 * data, sizes and properties. It must be called prior to all subsequent library
 * function calls that involve sparse matrices. It should be destroyed at the end
 * using rocsparse_destroy_spmat_descr(). All data pointers remain valid.
 *******************************************************************************/
rocsparse_status rocsparse_create_ell_descr(rocsparse_spmat_descr* descr,
                                            int64_t                rows,
                                            int64_t                cols,
                                            void*                  ell_col_ind,
                                            void*                  ell_val,
                                            int64_t                ell_width,
                                            rocsparse_indextype    idx_type,
                                            rocsparse_index_base   idx_base,
                                            rocsparse_datatype     data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, rows);
    ROCSPARSE_CHECKARG_SIZE(2, cols);
    ROCSPARSE_CHECKARG_SIZE(5, ell_width);
    ROCSPARSE_CHECKARG_ARRAY(3, rows * ell_width, ell_col_ind);
    ROCSPARSE_CHECKARG_ARRAY(4, rows * ell_width, ell_val);
    ROCSPARSE_CHECKARG(5, ell_width, (ell_width > cols), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_ENUM(6, idx_type);
    ROCSPARSE_CHECKARG_ENUM(7, idx_base);
    ROCSPARSE_CHECKARG_ENUM(8, data_type);

    *descr = new _rocsparse_spmat_descr;

    (*descr)->init = true;

    (*descr)->rows      = rows;
    (*descr)->cols      = cols;
    (*descr)->ell_width = ell_width;

    (*descr)->col_data = ell_col_ind;
    (*descr)->val_data = ell_val;

    (*descr)->const_col_data = ell_col_ind;
    (*descr)->const_val_data = ell_val;

    (*descr)->row_type  = idx_type;
    (*descr)->col_type  = idx_type;
    (*descr)->data_type = data_type;

    (*descr)->idx_base = idx_base;
    (*descr)->format   = rocsparse_format_ell;

    //
    // This is not really the number of non-zeros.
    // TODO: refactor the descriptors and having a proper design (get_nnz and different implementation for different format).
    // ell_width = nnz / rows.
    //
    (*descr)->nnz = ell_width * rows;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&(*descr)->descr));
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&(*descr)->info));

    // Initialize descriptor
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_set_mat_index_base((*descr)->descr, idx_base));

    (*descr)->batch_count                 = 1;
    (*descr)->batch_stride                = 0;
    (*descr)->offsets_batch_stride        = 0;
    (*descr)->columns_values_batch_stride = 0;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_create_bell_descr creates a descriptor holding the
 * BLOCKED ELL matrix data, sizes and properties. It must be called prior to all
 * subsequent library function calls that involve sparse matrices.
 * It should be destroyed at the end using rocsparse_destroy_spmat_descr().
 * All data pointers remain valid.
 *******************************************************************************/
rocsparse_status rocsparse_create_bell_descr(rocsparse_spmat_descr* descr,
                                             int64_t                rows,
                                             int64_t                cols,
                                             rocsparse_direction    ell_block_dir,
                                             int64_t                ell_block_dim,
                                             int64_t                ell_cols,
                                             void*                  ell_col_ind,
                                             void*                  ell_val,
                                             rocsparse_indextype    idx_type,
                                             rocsparse_index_base   idx_base,
                                             rocsparse_datatype     data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, rows);
    ROCSPARSE_CHECKARG_SIZE(2, cols);
    ROCSPARSE_CHECKARG_ENUM(3, ell_block_dir);
    ROCSPARSE_CHECKARG_SIZE(4, ell_block_dim);
    ROCSPARSE_CHECKARG(4, ell_block_dim, (ell_block_dim == 0), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_SIZE(5, ell_cols);
    ROCSPARSE_CHECKARG(5, ell_cols, (ell_cols > cols), rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG_ARRAY(6, ell_cols * ell_block_dim, ell_col_ind);
    ROCSPARSE_CHECKARG_ARRAY(7, ell_cols * ell_block_dim, ell_val);

    ROCSPARSE_CHECKARG_ENUM(8, idx_type);
    ROCSPARSE_CHECKARG_ENUM(9, idx_base);
    ROCSPARSE_CHECKARG_ENUM(10, data_type);

    *descr = new _rocsparse_spmat_descr;

    (*descr)->init = true;
    (*descr)->rows = rows;
    (*descr)->cols = cols;

    (*descr)->ell_cols  = ell_cols;
    (*descr)->block_dir = ell_block_dir;
    (*descr)->block_dim = ell_block_dim;

    (*descr)->col_data = ell_col_ind;
    (*descr)->val_data = ell_val;

    (*descr)->const_col_data = ell_col_ind;
    (*descr)->const_val_data = ell_val;

    (*descr)->row_type  = idx_type;
    (*descr)->col_type  = idx_type;
    (*descr)->data_type = data_type;

    (*descr)->idx_base = idx_base;
    (*descr)->format   = rocsparse_format_bell;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&(*descr)->descr));
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&(*descr)->info));

    // Initialize descriptor
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_set_mat_index_base((*descr)->descr, idx_base));

    (*descr)->batch_count                 = 1;
    (*descr)->batch_stride                = 0;
    (*descr)->offsets_batch_stride        = 0;
    (*descr)->columns_values_batch_stride = 0;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_create_const_bell_descr(rocsparse_const_spmat_descr* descr,
                                                   int64_t                      rows,
                                                   int64_t                      cols,
                                                   rocsparse_direction          ell_block_dir,
                                                   int64_t                      ell_block_dim,
                                                   int64_t                      ell_cols,
                                                   const void*                  ell_col_ind,
                                                   const void*                  ell_val,
                                                   rocsparse_indextype          idx_type,
                                                   rocsparse_index_base         idx_base,
                                                   rocsparse_datatype           data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, rows);
    ROCSPARSE_CHECKARG_SIZE(2, cols);
    ROCSPARSE_CHECKARG_ENUM(3, ell_block_dir);
    ROCSPARSE_CHECKARG_SIZE(4, ell_block_dim);
    ROCSPARSE_CHECKARG(4, ell_block_dim, (ell_block_dim == 0), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_SIZE(5, ell_cols);
    ROCSPARSE_CHECKARG(5, ell_cols, ell_cols > cols, rocsparse_status_invalid_size);

    ROCSPARSE_CHECKARG_ARRAY(6, rows * ell_cols * ell_block_dim, ell_col_ind);
    ROCSPARSE_CHECKARG_ARRAY(7, rows * ell_cols * ell_block_dim, ell_val);

    ROCSPARSE_CHECKARG_ENUM(8, idx_type);
    ROCSPARSE_CHECKARG_ENUM(9, idx_base);
    ROCSPARSE_CHECKARG_ENUM(10, data_type);

    rocsparse_spmat_descr new_descr = new _rocsparse_spmat_descr;

    new_descr->init = true;

    new_descr->rows = rows;
    new_descr->cols = cols;

    new_descr->ell_cols  = ell_cols;
    new_descr->block_dir = ell_block_dir;
    new_descr->block_dim = ell_block_dim;

    new_descr->col_data = nullptr;
    new_descr->val_data = nullptr;

    new_descr->const_col_data = ell_col_ind;
    new_descr->const_val_data = ell_val;

    new_descr->row_type  = idx_type;
    new_descr->col_type  = idx_type;
    new_descr->data_type = data_type;

    new_descr->idx_base = idx_base;
    new_descr->format   = rocsparse_format_bell;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&new_descr->descr));
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&new_descr->info));

    // Initialize descriptor
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_set_mat_index_base(new_descr->descr, idx_base));

    new_descr->batch_count                 = 1;
    new_descr->batch_stride                = 0;
    new_descr->offsets_batch_stride        = 0;
    new_descr->columns_values_batch_stride = 0;

    *descr = new_descr;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_create_bsr_descr creates a descriptor holding the BSR matrix
 * data, sizes and properties. It must be called prior to all subsequent library
 * function calls that involve sparse matrices. It should be destroyed at the end
 * using rocsparse_destroy_spmat_descr(). All data pointers remain valid.
 *******************************************************************************/
rocsparse_status rocsparse_create_bsr_descr(rocsparse_spmat_descr* descr,
                                            int64_t                mb,
                                            int64_t                nb,
                                            int64_t                nnzb,
                                            rocsparse_direction    block_dir,
                                            int64_t                block_dim,
                                            void*                  bsr_row_ptr,
                                            void*                  bsr_col_ind,
                                            void*                  bsr_val,
                                            rocsparse_indextype    row_ptr_type,
                                            rocsparse_indextype    col_ind_type,
                                            rocsparse_index_base   idx_base,
                                            rocsparse_datatype     data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, mb);
    ROCSPARSE_CHECKARG_SIZE(2, nb);
    ROCSPARSE_CHECKARG_SIZE(3, nnzb);
    ROCSPARSE_CHECKARG(3, nnzb, (nnzb > mb * nb), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_ENUM(4, block_dir);
    ROCSPARSE_CHECKARG_SIZE(5, block_dim);
    ROCSPARSE_CHECKARG(5, block_dim, (block_dim == 0), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_ARRAY(6, mb, bsr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(7, nnzb, bsr_col_ind);
    ROCSPARSE_CHECKARG_ARRAY(8, nnzb, bsr_val);
    ROCSPARSE_CHECKARG_ENUM(9, row_ptr_type);
    ROCSPARSE_CHECKARG_ENUM(10, col_ind_type);
    ROCSPARSE_CHECKARG_ENUM(11, idx_base);
    ROCSPARSE_CHECKARG_ENUM(12, data_type);

    *descr = new _rocsparse_spmat_descr;

    (*descr)->init = true;

    (*descr)->rows = mb;
    (*descr)->cols = nb;
    (*descr)->nnz  = nnzb;

    (*descr)->row_data = bsr_row_ptr;
    (*descr)->col_data = bsr_col_ind;
    (*descr)->val_data = bsr_val;

    (*descr)->const_row_data = bsr_row_ptr;
    (*descr)->const_col_data = bsr_col_ind;
    (*descr)->const_val_data = bsr_val;

    (*descr)->row_type  = row_ptr_type;
    (*descr)->col_type  = col_ind_type;
    (*descr)->data_type = data_type;

    (*descr)->idx_base = idx_base;
    (*descr)->format   = rocsparse_format_bsr;

    (*descr)->block_dim = block_dim;
    (*descr)->block_dir = block_dir;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&(*descr)->descr));
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&(*descr)->info));

    // Initialize descriptor
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_set_mat_index_base((*descr)->descr, idx_base));

    (*descr)->batch_count                 = 1;
    (*descr)->batch_stride                = 0;
    (*descr)->offsets_batch_stride        = 0;
    (*descr)->columns_values_batch_stride = 0;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_destroy_spmat_descr destroys a sparse matrix descriptor.
 *******************************************************************************/
rocsparse_status rocsparse_destroy_spmat_descr(rocsparse_const_spmat_descr descr)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);

    // Check if descriptor has been initialized
    if(descr->init == false)
    {
        // Do nothing
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_mat_descr(descr->descr));
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_mat_info(descr->info));

    delete descr;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_coo_get returns the sparse COO matrix data, sizes and
 * properties.
 *******************************************************************************/
rocsparse_status rocsparse_coo_get(const rocsparse_spmat_descr descr,
                                   int64_t*                    rows,
                                   int64_t*                    cols,
                                   int64_t*                    nnz,
                                   void**                      coo_row_ind,
                                   void**                      coo_col_ind,
                                   void**                      coo_val,
                                   rocsparse_indextype*        idx_type,
                                   rocsparse_index_base*       idx_base,
                                   rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, nnz);
    ROCSPARSE_CHECKARG_POINTER(4, coo_row_ind);
    ROCSPARSE_CHECKARG_POINTER(5, coo_col_ind);
    ROCSPARSE_CHECKARG_POINTER(6, coo_val);
    ROCSPARSE_CHECKARG_POINTER(7, idx_type);
    ROCSPARSE_CHECKARG_POINTER(8, idx_base);
    ROCSPARSE_CHECKARG_POINTER(9, data_type);

    *rows = descr->rows;
    *cols = descr->cols;
    *nnz  = descr->nnz;

    *coo_row_ind = descr->row_data;
    *coo_col_ind = descr->col_data;
    *coo_val     = descr->val_data;

    *idx_type  = descr->row_type;
    *idx_base  = descr->idx_base;
    *data_type = descr->data_type;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_const_coo_get(rocsparse_const_spmat_descr descr,
                                         int64_t*                    rows,
                                         int64_t*                    cols,
                                         int64_t*                    nnz,
                                         const void**                coo_row_ind,
                                         const void**                coo_col_ind,
                                         const void**                coo_val,
                                         rocsparse_indextype*        idx_type,
                                         rocsparse_index_base*       idx_base,
                                         rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, nnz);
    ROCSPARSE_CHECKARG_POINTER(4, coo_row_ind);
    ROCSPARSE_CHECKARG_POINTER(5, coo_col_ind);
    ROCSPARSE_CHECKARG_POINTER(6, coo_val);
    ROCSPARSE_CHECKARG_POINTER(7, idx_type);
    ROCSPARSE_CHECKARG_POINTER(8, idx_base);
    ROCSPARSE_CHECKARG_POINTER(9, data_type);

    *rows = descr->rows;
    *cols = descr->cols;
    *nnz  = descr->nnz;

    *coo_row_ind = descr->const_row_data;
    *coo_col_ind = descr->const_col_data;
    *coo_val     = descr->const_val_data;

    *idx_type  = descr->row_type;
    *idx_base  = descr->idx_base;
    *data_type = descr->data_type;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_coo_aos_get returns the sparse COO (AoS) matrix data, sizes and
 * properties.
 *******************************************************************************/
rocsparse_status rocsparse_coo_aos_get(const rocsparse_spmat_descr descr,
                                       int64_t*                    rows,
                                       int64_t*                    cols,
                                       int64_t*                    nnz,
                                       void**                      coo_ind,
                                       void**                      coo_val,
                                       rocsparse_indextype*        idx_type,
                                       rocsparse_index_base*       idx_base,
                                       rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, nnz);
    ROCSPARSE_CHECKARG_POINTER(4, coo_ind);
    ROCSPARSE_CHECKARG_POINTER(5, coo_val);
    ROCSPARSE_CHECKARG_POINTER(6, idx_type);
    ROCSPARSE_CHECKARG_POINTER(7, idx_base);
    ROCSPARSE_CHECKARG_POINTER(8, data_type);

    *rows = descr->rows;
    *cols = descr->cols;
    *nnz  = descr->nnz;

    *coo_ind = descr->ind_data;
    *coo_val = descr->val_data;

    *idx_type  = descr->row_type;
    *idx_base  = descr->idx_base;
    *data_type = descr->data_type;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_const_coo_aos_get(rocsparse_const_spmat_descr descr,
                                             int64_t*                    rows,
                                             int64_t*                    cols,
                                             int64_t*                    nnz,
                                             const void**                coo_ind,
                                             const void**                coo_val,
                                             rocsparse_indextype*        idx_type,
                                             rocsparse_index_base*       idx_base,
                                             rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, nnz);
    ROCSPARSE_CHECKARG_POINTER(4, coo_ind);
    ROCSPARSE_CHECKARG_POINTER(5, coo_val);
    ROCSPARSE_CHECKARG_POINTER(6, idx_type);
    ROCSPARSE_CHECKARG_POINTER(7, idx_base);
    ROCSPARSE_CHECKARG_POINTER(8, data_type);

    *rows = descr->rows;
    *cols = descr->cols;
    *nnz  = descr->nnz;

    *coo_ind = descr->const_ind_data;
    *coo_val = descr->const_val_data;

    *idx_type  = descr->row_type;
    *idx_base  = descr->idx_base;
    *data_type = descr->data_type;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_csr_get returns the sparse CSR matrix data, sizes and
 * properties.
 *******************************************************************************/
rocsparse_status rocsparse_csr_get(const rocsparse_spmat_descr descr,
                                   int64_t*                    rows,
                                   int64_t*                    cols,
                                   int64_t*                    nnz,
                                   void**                      csr_row_ptr,
                                   void**                      csr_col_ind,
                                   void**                      csr_val,
                                   rocsparse_indextype*        row_ptr_type,
                                   rocsparse_indextype*        col_ind_type,
                                   rocsparse_index_base*       idx_base,
                                   rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, nnz);
    ROCSPARSE_CHECKARG_POINTER(4, csr_row_ptr);
    ROCSPARSE_CHECKARG_POINTER(5, csr_col_ind);
    ROCSPARSE_CHECKARG_POINTER(6, csr_val);
    ROCSPARSE_CHECKARG_POINTER(7, row_ptr_type);
    ROCSPARSE_CHECKARG_POINTER(8, col_ind_type);
    ROCSPARSE_CHECKARG_POINTER(9, idx_base);
    ROCSPARSE_CHECKARG_POINTER(10, data_type);

    *rows = descr->rows;
    *cols = descr->cols;
    *nnz  = descr->nnz;

    *csr_row_ptr = descr->row_data;
    *csr_col_ind = descr->col_data;
    *csr_val     = descr->val_data;

    *row_ptr_type = descr->row_type;
    *col_ind_type = descr->col_type;
    *idx_base     = descr->idx_base;
    *data_type    = descr->data_type;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_const_csr_get(rocsparse_const_spmat_descr descr,
                                         int64_t*                    rows,
                                         int64_t*                    cols,
                                         int64_t*                    nnz,
                                         const void**                csr_row_ptr,
                                         const void**                csr_col_ind,
                                         const void**                csr_val,
                                         rocsparse_indextype*        row_ptr_type,
                                         rocsparse_indextype*        col_ind_type,
                                         rocsparse_index_base*       idx_base,
                                         rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, nnz);
    ROCSPARSE_CHECKARG_POINTER(4, csr_row_ptr);
    ROCSPARSE_CHECKARG_POINTER(5, csr_col_ind);
    ROCSPARSE_CHECKARG_POINTER(6, csr_val);
    ROCSPARSE_CHECKARG_POINTER(7, row_ptr_type);
    ROCSPARSE_CHECKARG_POINTER(8, col_ind_type);
    ROCSPARSE_CHECKARG_POINTER(9, idx_base);
    ROCSPARSE_CHECKARG_POINTER(10, data_type);

    *rows = descr->rows;
    *cols = descr->cols;
    *nnz  = descr->nnz;

    *csr_row_ptr = descr->const_row_data;
    *csr_col_ind = descr->const_col_data;
    *csr_val     = descr->const_val_data;

    *row_ptr_type = descr->row_type;
    *col_ind_type = descr->col_type;
    *idx_base     = descr->idx_base;
    *data_type    = descr->data_type;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_bsr_get returns the sparse BSR matrix data, sizes and
 * properties.
 *******************************************************************************/
rocsparse_status rocsparse_const_bsr_get(rocsparse_const_spmat_descr descr,
                                         int64_t*                    brows,
                                         int64_t*                    bcols,
                                         int64_t*                    bnnz,
                                         rocsparse_direction*        bdir,
                                         int64_t*                    bdim,
                                         const void**                bsr_row_ptr,
                                         const void**                bsr_col_ind,
                                         const void**                bsr_val,
                                         rocsparse_indextype*        row_ptr_type,
                                         rocsparse_indextype*        col_ind_type,
                                         rocsparse_index_base*       idx_base,
                                         rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for valid pointers
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for invalid size pointers
    if(brows == nullptr || bcols == nullptr || bnnz == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for invalid data pointers
    if(bsr_row_ptr == nullptr || bsr_col_ind == nullptr || bsr_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for invalid property pointers
    if(row_ptr_type == nullptr || col_ind_type == nullptr || idx_base == nullptr
       || data_type == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if descriptor has been initialized
    if(descr->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    *brows = descr->rows;
    *bcols = descr->cols;
    *bnnz  = descr->nnz;

    *bsr_row_ptr = descr->const_row_data;
    *bsr_col_ind = descr->const_col_data;
    *bsr_val     = descr->const_val_data;

    *row_ptr_type = descr->row_type;
    *col_ind_type = descr->col_type;
    *idx_base     = descr->idx_base;
    *data_type    = descr->data_type;
    *bdim         = descr->block_dim;
    *bdir         = descr->block_dir;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_bsr_get(const rocsparse_spmat_descr descr,
                                   int64_t*                    brows,
                                   int64_t*                    bcols,
                                   int64_t*                    bnnz,
                                   rocsparse_direction*        bdir,
                                   int64_t*                    bdim,
                                   void**                      bsr_row_ptr,
                                   void**                      bsr_col_ind,
                                   void**                      bsr_val,
                                   rocsparse_indextype*        row_ptr_type,
                                   rocsparse_indextype*        col_ind_type,
                                   rocsparse_index_base*       idx_base,
                                   rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for valid pointers
    if(descr == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for invalid size pointers
    if(brows == nullptr || bcols == nullptr || bnnz == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for invalid data pointers
    if(bsr_row_ptr == nullptr || bsr_col_ind == nullptr || bsr_val == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check for invalid property pointers
    if(row_ptr_type == nullptr || col_ind_type == nullptr || idx_base == nullptr
       || data_type == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    // Check if descriptor has been initialized
    if(descr->init == false)
    {
        return rocsparse_status_not_initialized;
    }

    *brows = descr->rows;
    *bcols = descr->cols;
    *bnnz  = descr->nnz;

    *bsr_row_ptr = descr->row_data;
    *bsr_col_ind = descr->col_data;
    *bsr_val     = descr->val_data;

    *row_ptr_type = descr->row_type;
    *col_ind_type = descr->col_type;
    *idx_base     = descr->idx_base;
    *data_type    = descr->data_type;
    *bdim         = descr->block_dim;
    *bdir         = descr->block_dir;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_csc_get returns the sparse CSC matrix data, sizes and
 * properties.
 *******************************************************************************/
rocsparse_status rocsparse_csc_get(const rocsparse_spmat_descr descr,
                                   int64_t*                    rows,
                                   int64_t*                    cols,
                                   int64_t*                    nnz,
                                   void**                      csc_col_ptr,
                                   void**                      csc_row_ind,
                                   void**                      csc_val,
                                   rocsparse_indextype*        col_ptr_type,
                                   rocsparse_indextype*        row_ind_type,
                                   rocsparse_index_base*       idx_base,
                                   rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, nnz);
    ROCSPARSE_CHECKARG_POINTER(4, csc_col_ptr);
    ROCSPARSE_CHECKARG_POINTER(5, csc_row_ind);
    ROCSPARSE_CHECKARG_POINTER(6, csc_val);
    ROCSPARSE_CHECKARG_POINTER(7, col_ptr_type);
    ROCSPARSE_CHECKARG_POINTER(8, row_ind_type);
    ROCSPARSE_CHECKARG_POINTER(9, idx_base);
    ROCSPARSE_CHECKARG_POINTER(10, data_type);

    *rows = descr->rows;
    *cols = descr->cols;
    *nnz  = descr->nnz;

    *csc_col_ptr = descr->col_data;
    *csc_row_ind = descr->row_data;
    *csc_val     = descr->val_data;

    *col_ptr_type = descr->col_type;
    *row_ind_type = descr->row_type;
    *idx_base     = descr->idx_base;
    *data_type    = descr->data_type;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_const_csc_get(rocsparse_const_spmat_descr descr,
                                         int64_t*                    rows,
                                         int64_t*                    cols,
                                         int64_t*                    nnz,
                                         const void**                csc_col_ptr,
                                         const void**                csc_row_ind,
                                         const void**                csc_val,
                                         rocsparse_indextype*        col_ptr_type,
                                         rocsparse_indextype*        row_ind_type,
                                         rocsparse_index_base*       idx_base,
                                         rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, nnz);
    ROCSPARSE_CHECKARG_POINTER(4, csc_col_ptr);
    ROCSPARSE_CHECKARG_POINTER(5, csc_row_ind);
    ROCSPARSE_CHECKARG_POINTER(6, csc_val);
    ROCSPARSE_CHECKARG_POINTER(7, col_ptr_type);
    ROCSPARSE_CHECKARG_POINTER(8, row_ind_type);
    ROCSPARSE_CHECKARG_POINTER(9, idx_base);
    ROCSPARSE_CHECKARG_POINTER(10, data_type);

    *rows = descr->rows;
    *cols = descr->cols;
    *nnz  = descr->nnz;

    *csc_col_ptr = descr->const_col_data;
    *csc_row_ind = descr->const_row_data;
    *csc_val     = descr->const_val_data;

    *row_ind_type = descr->row_type;
    *col_ptr_type = descr->col_type;
    *idx_base     = descr->idx_base;
    *data_type    = descr->data_type;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_ell_get returns the sparse ELL matrix data, sizes and
 * properties.
 *******************************************************************************/
rocsparse_status rocsparse_ell_get(const rocsparse_spmat_descr descr,
                                   int64_t*                    rows,
                                   int64_t*                    cols,
                                   void**                      ell_col_ind,
                                   void**                      ell_val,
                                   int64_t*                    ell_width,
                                   rocsparse_indextype*        idx_type,
                                   rocsparse_index_base*       idx_base,
                                   rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, ell_col_ind);
    ROCSPARSE_CHECKARG_POINTER(4, ell_val);
    ROCSPARSE_CHECKARG_POINTER(5, ell_width);
    ROCSPARSE_CHECKARG_POINTER(6, idx_type);
    ROCSPARSE_CHECKARG_POINTER(7, idx_base);
    ROCSPARSE_CHECKARG_POINTER(8, data_type);

    *rows = descr->rows;
    *cols = descr->cols;

    *ell_col_ind = descr->col_data;
    *ell_val     = descr->val_data;
    *ell_width   = descr->ell_width;

    *idx_type  = descr->row_type;
    *idx_base  = descr->idx_base;
    *data_type = descr->data_type;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_const_ell_get(rocsparse_const_spmat_descr descr,
                                         int64_t*                    rows,
                                         int64_t*                    cols,
                                         const void**                ell_col_ind,
                                         const void**                ell_val,
                                         int64_t*                    ell_width,
                                         rocsparse_indextype*        idx_type,
                                         rocsparse_index_base*       idx_base,
                                         rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, ell_col_ind);
    ROCSPARSE_CHECKARG_POINTER(4, ell_val);
    ROCSPARSE_CHECKARG_POINTER(5, ell_width);
    ROCSPARSE_CHECKARG_POINTER(6, idx_type);
    ROCSPARSE_CHECKARG_POINTER(7, idx_base);
    ROCSPARSE_CHECKARG_POINTER(8, data_type);

    *rows = descr->rows;
    *cols = descr->cols;

    *ell_col_ind = descr->const_col_data;
    *ell_val     = descr->const_val_data;
    *ell_width   = descr->ell_width;

    *idx_type  = descr->row_type;
    *idx_base  = descr->idx_base;
    *data_type = descr->data_type;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_bell_get returns the sparse BLOCKED ELL matrix data,
 * sizes and properties.
 *******************************************************************************/
rocsparse_status rocsparse_bell_get(const rocsparse_spmat_descr descr,
                                    int64_t*                    rows,
                                    int64_t*                    cols,
                                    rocsparse_direction*        ell_block_dir,
                                    int64_t*                    ell_block_dim,
                                    int64_t*                    ell_cols,
                                    void**                      ell_col_ind,
                                    void**                      ell_val,
                                    rocsparse_indextype*        idx_type,
                                    rocsparse_index_base*       idx_base,
                                    rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, ell_block_dir);
    ROCSPARSE_CHECKARG_POINTER(4, ell_block_dim);
    ROCSPARSE_CHECKARG_POINTER(5, ell_cols);
    ROCSPARSE_CHECKARG_POINTER(6, ell_col_ind);
    ROCSPARSE_CHECKARG_POINTER(7, ell_val);
    ROCSPARSE_CHECKARG_POINTER(8, idx_type);
    ROCSPARSE_CHECKARG_POINTER(9, idx_base);
    ROCSPARSE_CHECKARG_POINTER(10, data_type);

    *rows = descr->rows;
    *cols = descr->cols;

    *ell_col_ind   = descr->col_data;
    *ell_val       = descr->val_data;
    *ell_cols      = descr->ell_cols;
    *ell_block_dir = descr->block_dir;
    *ell_block_dim = descr->block_dim;

    *idx_type  = descr->row_type;
    *idx_base  = descr->idx_base;
    *data_type = descr->data_type;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_const_bell_get(rocsparse_const_spmat_descr descr,
                                          int64_t*                    rows,
                                          int64_t*                    cols,
                                          rocsparse_direction*        ell_block_dir,
                                          int64_t*                    ell_block_dim,
                                          int64_t*                    ell_cols,
                                          const void**                ell_col_ind,
                                          const void**                ell_val,
                                          rocsparse_indextype*        idx_type,
                                          rocsparse_index_base*       idx_base,
                                          rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, ell_block_dir);
    ROCSPARSE_CHECKARG_POINTER(4, ell_block_dim);
    ROCSPARSE_CHECKARG_POINTER(5, ell_cols);
    ROCSPARSE_CHECKARG_POINTER(6, ell_col_ind);
    ROCSPARSE_CHECKARG_POINTER(7, ell_val);
    ROCSPARSE_CHECKARG_POINTER(8, idx_type);
    ROCSPARSE_CHECKARG_POINTER(9, idx_base);
    ROCSPARSE_CHECKARG_POINTER(10, data_type);

    *rows = descr->rows;
    *cols = descr->cols;

    *ell_col_ind   = descr->const_col_data;
    *ell_val       = descr->const_val_data;
    *ell_cols      = descr->ell_cols;
    *ell_block_dir = descr->block_dir;
    *ell_block_dim = descr->block_dim;

    *idx_type  = descr->row_type;
    *idx_base  = descr->idx_base;
    *data_type = descr->data_type;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_coo_set_pointers sets the sparse COO matrix data pointers.
 *******************************************************************************/
rocsparse_status rocsparse_coo_set_pointers(rocsparse_spmat_descr descr,
                                            void*                 coo_row_ind,
                                            void*                 coo_col_ind,
                                            void*                 coo_val)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, coo_row_ind);
    ROCSPARSE_CHECKARG_POINTER(2, coo_col_ind);
    ROCSPARSE_CHECKARG_POINTER(3, coo_val);

    descr->row_data = coo_row_ind;
    descr->col_data = coo_col_ind;
    descr->val_data = coo_val;

    descr->const_row_data = coo_row_ind;
    descr->const_col_data = coo_col_ind;
    descr->const_val_data = coo_val;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_coo_aos_set_pointers sets the sparse COO (AoS) matrix data pointers.
 *******************************************************************************/
rocsparse_status
    rocsparse_coo_aos_set_pointers(rocsparse_spmat_descr descr, void* coo_ind, void* coo_val)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, coo_ind);
    ROCSPARSE_CHECKARG_POINTER(2, coo_val);

    descr->ind_data = coo_ind;
    descr->val_data = coo_val;

    descr->const_ind_data = coo_ind;
    descr->const_val_data = coo_val;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_csr_set_pointers sets the sparse CSR matrix data pointers.
 *******************************************************************************/
rocsparse_status rocsparse_csr_set_pointers(rocsparse_spmat_descr descr,
                                            void*                 csr_row_ptr,
                                            void*                 csr_col_ind,
                                            void*                 csr_val)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);

    ROCSPARSE_CHECKARG_POINTER(1, csr_row_ptr);
    ROCSPARSE_CHECKARG(
        2, csr_col_ind, descr->nnz > 0 && csr_col_ind == nullptr, rocsparse_status_invalid_pointer);
    ROCSPARSE_CHECKARG(
        3, csr_val, descr->nnz > 0 && csr_val == nullptr, rocsparse_status_invalid_pointer);

    // Sparsity structure might have changed, analysis is required before calling SpMV
    descr->analysed = false;

    descr->row_data = csr_row_ptr;
    descr->col_data = csr_col_ind;
    descr->val_data = csr_val;

    descr->const_row_data = csr_row_ptr;
    descr->const_col_data = csr_col_ind;
    descr->const_val_data = csr_val;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_csc_set_pointers sets the sparse CSR matrix data pointers.
 *******************************************************************************/
rocsparse_status rocsparse_csc_set_pointers(rocsparse_spmat_descr descr,
                                            void*                 csc_col_ptr,
                                            void*                 csc_row_ind,
                                            void*                 csc_val)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);

    ROCSPARSE_CHECKARG_POINTER(1, csc_col_ptr);
    ROCSPARSE_CHECKARG(
        2, csc_row_ind, descr->nnz > 0 && csc_row_ind == nullptr, rocsparse_status_invalid_pointer);
    ROCSPARSE_CHECKARG(
        3, csc_val, descr->nnz > 0 && csc_val == nullptr, rocsparse_status_invalid_pointer);

    // Sparsity structure might have changed, analysis is required before calling SpMV
    descr->analysed = false;

    descr->row_data = csc_row_ind;
    descr->col_data = csc_col_ptr;
    descr->val_data = csc_val;

    descr->const_row_data = csc_row_ind;
    descr->const_col_data = csc_col_ptr;
    descr->const_val_data = csc_val;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_ell_set_pointers sets the sparse ELL matrix data pointers.
 *******************************************************************************/
rocsparse_status
    rocsparse_ell_set_pointers(rocsparse_spmat_descr descr, void* ell_col_ind, void* ell_val)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, ell_col_ind);
    ROCSPARSE_CHECKARG_POINTER(2, ell_val);

    descr->col_data = ell_col_ind;
    descr->val_data = ell_val;

    descr->const_col_data = ell_col_ind;
    descr->const_val_data = ell_val;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_bsr_set_pointers sets the sparse BSR matrix data pointers.
 *******************************************************************************/
rocsparse_status rocsparse_bsr_set_pointers(rocsparse_spmat_descr descr,
                                            void*                 bsr_row_ptr,
                                            void*                 bsr_col_ind,
                                            void*                 bsr_val)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);

    ROCSPARSE_CHECKARG_POINTER(1, bsr_row_ptr);
    ROCSPARSE_CHECKARG(
        2, bsr_col_ind, descr->nnz > 0 && bsr_col_ind == nullptr, rocsparse_status_invalid_pointer);
    ROCSPARSE_CHECKARG(
        3, bsr_val, descr->nnz > 0 && bsr_val == nullptr, rocsparse_status_invalid_pointer);

    // Sparsity structure might have changed, analysis is required before calling SpMV
    descr->analysed = false;

    descr->row_data = bsr_row_ptr;
    descr->col_data = bsr_col_ind;
    descr->val_data = bsr_val;

    descr->const_row_data = bsr_row_ptr;
    descr->const_col_data = bsr_col_ind;
    descr->const_val_data = bsr_val;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_spmat_get_size returns the sparse matrix sizes.
 *******************************************************************************/
rocsparse_status rocsparse_spmat_get_size(rocsparse_const_spmat_descr descr,
                                          int64_t*                    rows,
                                          int64_t*                    cols,
                                          int64_t*                    nnz)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, nnz);

    *rows = descr->rows;
    *cols = descr->cols;
    *nnz  = descr->nnz;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_spmat_get_format returns the sparse matrix format.
 *******************************************************************************/
rocsparse_status rocsparse_spmat_get_format(rocsparse_const_spmat_descr descr,
                                            rocsparse_format*           format)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, format);

    *format = descr->format;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_spmat_get_index_base returns the sparse matrix index base.
 *******************************************************************************/
rocsparse_status rocsparse_spmat_get_index_base(rocsparse_const_spmat_descr descr,
                                                rocsparse_index_base*       idx_base)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, idx_base);

    *idx_base = descr->idx_base;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_spmat_get_values returns the sparse matrix value pointer.
 *******************************************************************************/
rocsparse_status rocsparse_spmat_get_values(rocsparse_spmat_descr descr, void** values)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, values);
    *values = descr->val_data;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_const_spmat_get_values(rocsparse_const_spmat_descr descr,
                                                  const void**                values)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, values);

    *values = descr->const_val_data;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_spmat_set_values sets the sparse matrix value pointer.
 *******************************************************************************/
rocsparse_status rocsparse_spmat_set_values(rocsparse_spmat_descr descr, void* values)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, values);

    descr->val_data       = values;
    descr->const_val_data = values;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_spmat_get_strided_batch gets the sparse matrix batch count.
 *******************************************************************************/
rocsparse_status rocsparse_spmat_get_strided_batch(rocsparse_const_spmat_descr descr,
                                                   rocsparse_int*              batch_count)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, batch_count);

    *batch_count = descr->batch_count;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/****************************************************************************
 * \brief rocsparse_spmat_get_nnz gets the sparse matrix number of non-zeros.
 ****************************************************************************/
rocsparse_status rocsparse_spmat_get_nnz(rocsparse_const_spmat_descr descr, int64_t* nnz)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, nnz);

    switch(descr->format)
    {
    case rocsparse_format_bell:
    {
        nnz[0] = descr->ell_cols * descr->rows * descr->block_dim * descr->block_dim;
        break;
    }

    case rocsparse_format_ell:
    {
        nnz[0] = descr->ell_width * descr->rows;
        break;
    }

    case rocsparse_format_bsr:
    {
        nnz[0] = descr->nnz * descr->block_dim * descr->block_dim;
        break;
    }

    case rocsparse_format_csc:
    case rocsparse_format_csr:
    case rocsparse_format_coo:
    case rocsparse_format_coo_aos:
    {
        nnz[0] = descr->nnz;
        break;
    }
    }

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/****************************************************************************
 * \brief rocsparse_spmat_get_nnz sets the sparse matrix number of non-zeros.
 ****************************************************************************/
rocsparse_status rocsparse_spmat_set_nnz(rocsparse_spmat_descr descr, int64_t nnz)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_SIZE(1, nnz);

    switch(descr->format)
    {
    case rocsparse_format_bell:
    {
        // LCOV_EXCL_START
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            rocsparse_status_invalid_value,
            "Cannot set the number of non-zeros of a Block ELL sparse matrix.");
        // LCOV_EXCL_STOP
        break;
    }

    case rocsparse_format_ell:
    {
        // LCOV_EXCL_START
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            rocsparse_status_invalid_value,
            "Cannot set the number of non-zeros of an ELL sparse matrix.");
        // LCOV_EXCL_STOP
        break;
    }

    case rocsparse_format_bsr:
    {
        descr->nnz = nnz;
        break;
    }
    case rocsparse_format_csc:
    {
        descr->nnz = nnz;
        break;
    }
    case rocsparse_format_csr:
    {
        descr->nnz = nnz;
        break;
    }
    case rocsparse_format_coo:
    {
        descr->nnz = nnz;
        break;
    }
    case rocsparse_format_coo_aos:
    {
        descr->nnz = nnz;
        break;
    }
    }

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_spmat_set_strided_batch sets the sparse matrix batch count.
 *******************************************************************************/
rocsparse_status rocsparse_spmat_set_strided_batch(rocsparse_spmat_descr descr,
                                                   rocsparse_int         batch_count)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(1, batch_count, (batch_count <= 0), rocsparse_status_invalid_value);

    descr->batch_count = batch_count;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_coo_set_strided_batch sets the COO sparse matrix batch count
 * and batch stride.
 *******************************************************************************/
rocsparse_status rocsparse_coo_set_strided_batch(rocsparse_spmat_descr descr,
                                                 rocsparse_int         batch_count,
                                                 int64_t               batch_stride)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(1, batch_count, (batch_count <= 0), rocsparse_status_invalid_value);
    ROCSPARSE_CHECKARG(2, batch_stride, (batch_stride < 0), rocsparse_status_invalid_value);

    descr->batch_count  = batch_count;
    descr->batch_stride = batch_stride;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_csr_set_strided_batch sets the CSR sparse matrix batch count
 * and batch stride.
 *******************************************************************************/
rocsparse_status rocsparse_csr_set_strided_batch(rocsparse_spmat_descr descr,
                                                 rocsparse_int         batch_count,
                                                 int64_t               offsets_batch_stride,
                                                 int64_t               columns_values_batch_stride)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(1, batch_count, (batch_count <= 0), rocsparse_status_invalid_value);
    ROCSPARSE_CHECKARG(
        2, offsets_batch_stride, (offsets_batch_stride < 0), rocsparse_status_invalid_value);
    ROCSPARSE_CHECKARG(3,
                       columns_values_batch_stride,
                       (columns_values_batch_stride < 0),
                       rocsparse_status_invalid_value);

    descr->batch_count                 = batch_count;
    descr->offsets_batch_stride        = offsets_batch_stride;
    descr->columns_values_batch_stride = columns_values_batch_stride;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_csc_set_strided_batch sets the CSC sparse matrix batch count
 * and batch stride.
 *******************************************************************************/
rocsparse_status rocsparse_csc_set_strided_batch(rocsparse_spmat_descr descr,
                                                 rocsparse_int         batch_count,
                                                 int64_t               offsets_batch_stride,
                                                 int64_t               rows_values_batch_stride)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(1, batch_count, (batch_count <= 0), rocsparse_status_invalid_value);
    ROCSPARSE_CHECKARG(
        2, offsets_batch_stride, (offsets_batch_stride < 0), rocsparse_status_invalid_value);
    ROCSPARSE_CHECKARG(3,
                       rows_values_batch_stride,
                       (rows_values_batch_stride < 0),
                       rocsparse_status_invalid_value);

    descr->batch_count                 = batch_count;
    descr->offsets_batch_stride        = offsets_batch_stride;
    descr->columns_values_batch_stride = rows_values_batch_stride;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_spmat_get_attribute gets the sparse matrix attribute.
 *******************************************************************************/
rocsparse_status rocsparse_spmat_get_attribute(rocsparse_const_spmat_descr descr,
                                               rocsparse_spmat_attribute   attribute,
                                               void*                       data,
                                               size_t                      data_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_ENUM(1, attribute);
    ROCSPARSE_CHECKARG_POINTER(2, data);
    switch(attribute)
    {
    case rocsparse_spmat_fill_mode:
    {
        ROCSPARSE_CHECKARG(3,
                           data_size,
                           data_size != sizeof(rocsparse_spmat_fill_mode),
                           rocsparse_status_invalid_size);
        rocsparse_fill_mode* uplo = reinterpret_cast<rocsparse_fill_mode*>(data);
        *uplo                     = rocsparse_get_mat_fill_mode(descr->descr);
        return rocsparse_status_success;
    }
    case rocsparse_spmat_diag_type:
    {
        ROCSPARSE_CHECKARG(3,
                           data_size,
                           data_size != sizeof(rocsparse_spmat_diag_type),
                           rocsparse_status_invalid_size);
        rocsparse_diag_type* uplo = reinterpret_cast<rocsparse_diag_type*>(data);
        *uplo                     = rocsparse_get_mat_diag_type(descr->descr);
        return rocsparse_status_success;
    }
    case rocsparse_spmat_matrix_type:
    {
        ROCSPARSE_CHECKARG(3,
                           data_size,
                           data_size != sizeof(rocsparse_spmat_matrix_type),
                           rocsparse_status_invalid_size);
        rocsparse_matrix_type* matrix = reinterpret_cast<rocsparse_matrix_type*>(data);
        *matrix                       = rocsparse_get_mat_type(descr->descr);
        return rocsparse_status_success;
    }
    case rocsparse_spmat_storage_mode:
    {
        ROCSPARSE_CHECKARG(3,
                           data_size,
                           data_size != sizeof(rocsparse_spmat_storage_mode),
                           rocsparse_status_invalid_size);
        rocsparse_storage_mode* storage = reinterpret_cast<rocsparse_storage_mode*>(data);
        *storage                        = rocsparse_get_mat_storage_mode(descr->descr);
        return rocsparse_status_success;
    }
    }

    // LCOV_EXCL_START
    return rocsparse_status_invalid_value;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_spmat_set_attribute sets the sparse matrix attribute.
 *******************************************************************************/
rocsparse_status rocsparse_spmat_set_attribute(rocsparse_spmat_descr     descr,
                                               rocsparse_spmat_attribute attribute,
                                               const void*               data,
                                               size_t                    data_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_ENUM(1, attribute);
    ROCSPARSE_CHECKARG_POINTER(2, data);

    switch(attribute)
    {
    case rocsparse_spmat_fill_mode:
    {
        ROCSPARSE_CHECKARG(3,
                           data_size,
                           data_size != sizeof(rocsparse_spmat_fill_mode),
                           rocsparse_status_invalid_size);
        rocsparse_fill_mode uplo = *reinterpret_cast<const rocsparse_fill_mode*>(data);
        return rocsparse_set_mat_fill_mode(descr->descr, uplo);
    }
    case rocsparse_spmat_diag_type:
    {
        ROCSPARSE_CHECKARG(3,
                           data_size,
                           data_size != sizeof(rocsparse_spmat_diag_type),
                           rocsparse_status_invalid_size);
        rocsparse_diag_type diag = *reinterpret_cast<const rocsparse_diag_type*>(data);
        return rocsparse_set_mat_diag_type(descr->descr, diag);
    }

    case rocsparse_spmat_matrix_type:
    {
        ROCSPARSE_CHECKARG(3,
                           data_size,
                           data_size != sizeof(rocsparse_spmat_matrix_type),
                           rocsparse_status_invalid_size);
        rocsparse_matrix_type matrix = *reinterpret_cast<const rocsparse_matrix_type*>(data);
        return rocsparse_set_mat_type(descr->descr, matrix);
    }
    case rocsparse_spmat_storage_mode:
    {
        ROCSPARSE_CHECKARG(3,
                           data_size,
                           data_size != sizeof(rocsparse_spmat_storage_mode),
                           rocsparse_status_invalid_size);
        rocsparse_storage_mode storage = *reinterpret_cast<const rocsparse_storage_mode*>(data);
        return rocsparse_set_mat_storage_mode(descr->descr, storage);
    }
    }
    // LCOV_EXCL_START
    return rocsparse_status_invalid_value;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_create_dnvec_descr creates a descriptor holding the dense
 * vector data, size and properties. It must be called prior to all subsequent
 * library function calls that involve the dense vector. It should be destroyed
 * at the end using rocsparse_destroy_dnvec_descr(). The data pointer remains
 * valid.
 *******************************************************************************/
rocsparse_status rocsparse_create_dnvec_descr(rocsparse_dnvec_descr* descr,
                                              int64_t                size,
                                              void*                  values,
                                              rocsparse_datatype     data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, size);
    ROCSPARSE_CHECKARG_ARRAY(2, size, values);
    ROCSPARSE_CHECKARG_ENUM(3, data_type);

    *descr = new _rocsparse_dnvec_descr;

    (*descr)->init = true;

    (*descr)->size         = size;
    (*descr)->values       = values;
    (*descr)->const_values = values;
    (*descr)->data_type    = data_type;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_create_const_dnvec_descr(rocsparse_const_dnvec_descr* descr,
                                                    int64_t                      size,
                                                    const void*                  values,
                                                    rocsparse_datatype           data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, size);
    ROCSPARSE_CHECKARG_ARRAY(2, size, values);
    ROCSPARSE_CHECKARG_ENUM(3, data_type);

    rocsparse_dnvec_descr new_descr = new _rocsparse_dnvec_descr;

    new_descr->init = true;

    new_descr->size         = size;
    new_descr->values       = nullptr;
    new_descr->const_values = values;
    new_descr->data_type    = data_type;

    *descr = new_descr;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_destroy_dnvec_descr destroys a dense vector descriptor.
 *******************************************************************************/
rocsparse_status rocsparse_destroy_dnvec_descr(rocsparse_const_dnvec_descr descr)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);

    delete descr;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_dnvec_get returns the dense vector data, size and properties.
 *******************************************************************************/
rocsparse_status rocsparse_dnvec_get(const rocsparse_dnvec_descr descr,
                                     int64_t*                    size,
                                     void**                      values,
                                     rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, size);
    ROCSPARSE_CHECKARG_POINTER(2, values);
    ROCSPARSE_CHECKARG_POINTER(3, data_type);

    *size      = descr->size;
    *values    = descr->values;
    *data_type = descr->data_type;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_const_dnvec_get(rocsparse_const_dnvec_descr descr,
                                           int64_t*                    size,
                                           const void**                values,
                                           rocsparse_datatype*         data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, size);
    ROCSPARSE_CHECKARG_POINTER(2, values);
    ROCSPARSE_CHECKARG_POINTER(3, data_type);

    *size      = descr->size;
    *values    = descr->const_values;
    *data_type = descr->data_type;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_dnvec_get_values returns the dense vector value pointer.
 *******************************************************************************/
rocsparse_status rocsparse_dnvec_get_values(const rocsparse_dnvec_descr descr, void** values)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, values);

    *values = descr->values;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_const_dnvec_get_values(rocsparse_const_dnvec_descr descr,
                                                  const void**                values)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, values);

    *values = descr->const_values;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_dnvec_set_values sets the dense vector value pointer.
 *******************************************************************************/
rocsparse_status rocsparse_dnvec_set_values(rocsparse_dnvec_descr descr, void* values)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, values);
    descr->values       = values;
    descr->const_values = values;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_create_dnmat_descr creates a descriptor holding the dense
 * matrix data, size and properties. It must be called prior to all subsequent
 * library function calls that involve the dense matrix. It should be destroyed
 * at the end using rocsparse_destroy_dnmat_descr(). The data pointer remains
 * valid.
 *******************************************************************************/
rocsparse_status rocsparse_create_dnmat_descr(rocsparse_dnmat_descr* descr,
                                              int64_t                rows,
                                              int64_t                cols,
                                              int64_t                ld,
                                              void*                  values,
                                              rocsparse_datatype     data_type,
                                              rocsparse_order        order)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, rows);
    ROCSPARSE_CHECKARG_SIZE(2, cols);
    ROCSPARSE_CHECKARG_ENUM(5, data_type);
    ROCSPARSE_CHECKARG_ENUM(6, order);

    switch(order)
    {
    case rocsparse_order_row:
    {
        ROCSPARSE_CHECKARG(
            3, ld, (ld < rocsparse::max(int64_t(1), cols)), rocsparse_status_invalid_size);
        break;
    }
    case rocsparse_order_column:
    {
        ROCSPARSE_CHECKARG(
            3, ld, (ld < rocsparse::max(int64_t(1), rows)), rocsparse_status_invalid_size);
        break;
    }
    }

    ROCSPARSE_CHECKARG_ARRAY(4, int64_t(rows) * cols, values);

    *descr = new _rocsparse_dnmat_descr;

    (*descr)->init = true;

    (*descr)->rows         = rows;
    (*descr)->cols         = cols;
    (*descr)->ld           = ld;
    (*descr)->values       = values;
    (*descr)->const_values = values;
    (*descr)->data_type    = data_type;
    (*descr)->order        = order;

    (*descr)->batch_count  = 1;
    (*descr)->batch_stride = 0;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_create_const_dnmat_descr(rocsparse_const_dnmat_descr* descr,
                                                    int64_t                      rows,
                                                    int64_t                      cols,
                                                    int64_t                      ld,
                                                    const void*                  values,
                                                    rocsparse_datatype           data_type,
                                                    rocsparse_order              order)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, rows);
    ROCSPARSE_CHECKARG_SIZE(2, cols);

    switch(order)
    {
    case rocsparse_order_row:
    {
        ROCSPARSE_CHECKARG(
            3, ld, (ld < rocsparse::max(int64_t(1), cols)), rocsparse_status_invalid_size);
        break;
    }
    case rocsparse_order_column:
    {
        ROCSPARSE_CHECKARG(
            3, ld, (ld < rocsparse::max(int64_t(1), rows)), rocsparse_status_invalid_size);
        break;
    }
    }

    ROCSPARSE_CHECKARG_ARRAY(4, int64_t(rows) * cols, values);
    ROCSPARSE_CHECKARG_ENUM(5, data_type);
    ROCSPARSE_CHECKARG_ENUM(6, order);

    rocsparse_dnmat_descr new_descr = new _rocsparse_dnmat_descr;
    new_descr->init                 = true;

    new_descr->rows         = rows;
    new_descr->cols         = cols;
    new_descr->ld           = ld;
    new_descr->values       = nullptr;
    new_descr->const_values = values;
    new_descr->data_type    = data_type;
    new_descr->order        = order;

    new_descr->batch_count  = 1;
    new_descr->batch_stride = 0;

    *descr = new_descr;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_destroy_dnmat_descr destroys a dense matrix descriptor.
 *******************************************************************************/
rocsparse_status rocsparse_destroy_dnmat_descr(rocsparse_const_dnmat_descr descr)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    delete descr;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_dnmat_get returns the dense matrix data, size and properties.
 *******************************************************************************/
rocsparse_status rocsparse_dnmat_get(const rocsparse_dnmat_descr descr,
                                     int64_t*                    rows,
                                     int64_t*                    cols,
                                     int64_t*                    ld,
                                     void**                      values,
                                     rocsparse_datatype*         data_type,
                                     rocsparse_order*            order)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, ld);
    ROCSPARSE_CHECKARG_POINTER(4, values);
    ROCSPARSE_CHECKARG_POINTER(5, data_type);
    ROCSPARSE_CHECKARG_POINTER(6, order);

    *rows      = descr->rows;
    *cols      = descr->cols;
    *ld        = descr->ld;
    *values    = descr->values;
    *data_type = descr->data_type;
    *order     = descr->order;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_const_dnmat_get(rocsparse_const_dnmat_descr descr,
                                           int64_t*                    rows,
                                           int64_t*                    cols,
                                           int64_t*                    ld,
                                           const void**                values,
                                           rocsparse_datatype*         data_type,
                                           rocsparse_order*            order)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, ld);
    ROCSPARSE_CHECKARG_POINTER(4, values);
    ROCSPARSE_CHECKARG_POINTER(5, data_type);
    ROCSPARSE_CHECKARG_POINTER(6, order);

    *rows      = descr->rows;
    *cols      = descr->cols;
    *ld        = descr->ld;
    *values    = descr->const_values;
    *data_type = descr->data_type;
    *order     = descr->order;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_dnmat_get_values returns the dense matrix value pointer.
 *******************************************************************************/
rocsparse_status rocsparse_dnmat_get_values(const rocsparse_dnmat_descr descr, void** values)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, values);
    *values = descr->values;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_const_dnmat_get_values(rocsparse_const_dnmat_descr descr,
                                                  const void**                values)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, values);

    *values = descr->const_values;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_dnmat_set_values sets the dense matrix value pointer.
 *******************************************************************************/
rocsparse_status rocsparse_dnmat_set_values(rocsparse_dnmat_descr descr, void* values)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, values);

    descr->values       = values;
    descr->const_values = values;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_dnmat_get_strided_batch gets the dense matrix batch count
 * and batch stride.
 *******************************************************************************/
rocsparse_status rocsparse_dnmat_get_strided_batch(rocsparse_const_dnmat_descr descr,
                                                   rocsparse_int*              batch_count,
                                                   int64_t*                    batch_stride)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, batch_count);
    ROCSPARSE_CHECKARG_POINTER(2, batch_stride);

    *batch_count  = descr->batch_count;
    *batch_stride = descr->batch_stride;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

/********************************************************************************
 * \brief rocsparse_dnmat_set_strided_batch sets the dense matrix batch count
 * and batch stride.
 *******************************************************************************/
rocsparse_status rocsparse_dnmat_set_strided_batch(rocsparse_dnmat_descr descr,
                                                   rocsparse_int         batch_count,
                                                   int64_t               batch_stride)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG(0, descr, (descr->init == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(1, batch_count, (batch_count <= 0), rocsparse_status_invalid_value);
    ROCSPARSE_CHECKARG(2, batch_stride, (batch_stride < 0), rocsparse_status_invalid_value);

    if(descr->order == rocsparse_order_column)
    {
        ROCSPARSE_CHECKARG(2,
                           batch_stride,
                           (batch_count > 1 && batch_stride < descr->ld * descr->cols),
                           rocsparse_status_invalid_value);
    }
    else if(descr->order == rocsparse_order_row)
    {
        ROCSPARSE_CHECKARG(2,
                           batch_stride,
                           (batch_count > 1 && batch_stride < descr->ld * descr->rows),
                           rocsparse_status_invalid_value);
    }

    descr->batch_count  = batch_count;
    descr->batch_stride = batch_stride;

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

ROCSPARSE_EXPORT
rocsparse_status rocsparse_create_csr_descr_SWDEV_453599(rocsparse_spmat_descr* descr,
                                                         int64_t                rows,
                                                         int64_t                cols,
                                                         int64_t                nnz,
                                                         void*                  csr_row_ptr,
                                                         void*                  csr_col_ind,
                                                         void*                  csr_val,
                                                         rocsparse_indextype    row_ptr_type,
                                                         rocsparse_indextype    col_ind_type,
                                                         rocsparse_index_base   idx_base,
                                                         rocsparse_datatype     data_type)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_SIZE(1, rows);
    ROCSPARSE_CHECKARG_SIZE(2, cols);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG(3, nnz, (nnz > rows * cols), rocsparse_status_invalid_size);

    // cusparse allows setting NULL for the pointers when nnz == 0. See SWDEV_453599 for reproducer.
    // This function exists so that hipsparse can follow this behaviour without affecting rocsparse.
    ROCSPARSE_CHECKARG_ARRAY(4, nnz, csr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(5, nnz, csr_col_ind);
    ROCSPARSE_CHECKARG_ARRAY(6, nnz, csr_val);

    ROCSPARSE_CHECKARG_ENUM(7, row_ptr_type);
    ROCSPARSE_CHECKARG_ENUM(8, col_ind_type);
    ROCSPARSE_CHECKARG_ENUM(9, idx_base);
    ROCSPARSE_CHECKARG_ENUM(10, data_type);

    *descr = new _rocsparse_spmat_descr;

    (*descr)->init = true;

    (*descr)->rows = rows;
    (*descr)->cols = cols;
    (*descr)->nnz  = nnz;

    (*descr)->row_data = csr_row_ptr;
    (*descr)->col_data = csr_col_ind;
    (*descr)->val_data = csr_val;

    (*descr)->const_row_data = csr_row_ptr;
    (*descr)->const_col_data = csr_col_ind;
    (*descr)->const_val_data = csr_val;

    (*descr)->row_type  = row_ptr_type;
    (*descr)->col_type  = col_ind_type;
    (*descr)->data_type = data_type;

    (*descr)->idx_base = idx_base;
    (*descr)->format   = rocsparse_format_csr;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&(*descr)->descr));
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&(*descr)->info));

    // Initialize descriptor
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_set_mat_index_base((*descr)->descr, idx_base));

    (*descr)->batch_count                 = 1;
    (*descr)->batch_stride                = 0;
    (*descr)->offsets_batch_stride        = 0;
    (*descr)->columns_values_batch_stride = 0;
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

rocsparse_status rocsparse_create_spgeam_descr(rocsparse_spgeam_descr* descr)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);

    *descr = new _rocsparse_spgeam_descr();
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

rocsparse_status rocsparse_destroy_spgeam_descr(rocsparse_spgeam_descr descr)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);

    // Due to the changes in the hipFree introduced in HIP 7.0
    // https://rocm.docs.amd.com/projects/HIP/en/latest/hip-7-changes.html#update-hipfree
    // we need to introduce a device synchronize here as the below hipFree calls are now asynchronous.
    // hipFree() previously had an implicit wait for synchronization purpose which is applicable for all memory allocations.
    // This wait has been disabled in the HIP 7.0 runtime for allocations made with hipMallocAsync and hipMallocFromPoolAsync.
    RETURN_IF_HIP_ERROR(hipDeviceSynchronize());

    // Clean up row pointer array
    if(descr->csr_row_ptr_C != nullptr)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFree(descr->csr_row_ptr_C));
    }

    // Clean up rocprim buffer
    if(descr->rocprim_buffer != nullptr && descr->rocprim_alloc)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFree(descr->rocprim_buffer));
    }

    delete descr;
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

/********************************************************************************
 * \brief rocsparse_spgeam_set_input gets the input on the SpGEAM descriptor.
 *******************************************************************************/
rocsparse_status rocsparse_spgeam_set_input(rocsparse_handle       handle,
                                            rocsparse_spgeam_descr descr,
                                            rocsparse_spgeam_input input,
                                            const void*            data,
                                            size_t                 data_size_in_bytes,
                                            rocsparse_error*       p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG_ENUM(2, input);
    ROCSPARSE_CHECKARG_POINTER(3, data);

    switch(input)
    {
    case rocsparse_spgeam_input_scalar_alpha:
    {
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(void*),
                           rocsparse_status_invalid_size);
        descr->set_scalar_A(data);
        return rocsparse_status_success;
    }
    case rocsparse_spgeam_input_scalar_beta:
    {
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(void*),
                           rocsparse_status_invalid_size);
        descr->set_scalar_B(data);
        return rocsparse_status_success;
    }

    case rocsparse_spgeam_input_alg:
    {
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(rocsparse_spgeam_alg),
                           rocsparse_status_invalid_size);
        const rocsparse_spgeam_alg alg = *reinterpret_cast<const rocsparse_spgeam_alg*>(data);
        descr->set_alg(alg);
        return rocsparse_status_success;
    }
    case rocsparse_spgeam_input_scalar_datatype:
    {
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(rocsparse_datatype),
                           rocsparse_status_invalid_size);
        const rocsparse_datatype scalar_type = *reinterpret_cast<const rocsparse_datatype*>(data);
        descr->set_scalar_datatype(scalar_type);
        return rocsparse_status_success;
    }
    case rocsparse_spgeam_input_compute_datatype:
    {
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(rocsparse_datatype),
                           rocsparse_status_invalid_size);
        const rocsparse_datatype compute_type = *reinterpret_cast<const rocsparse_datatype*>(data);
        descr->set_compute_datatype(compute_type);
        return rocsparse_status_success;
    }
    case rocsparse_spgeam_input_operation_A:
    {
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(rocsparse_operation),
                           rocsparse_status_invalid_size);
        const rocsparse_operation op_A = *reinterpret_cast<const rocsparse_operation*>(data);
        descr->set_operation_A(op_A);
        return rocsparse_status_success;
    }
    case rocsparse_spgeam_input_operation_B:
    {
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(rocsparse_operation),
                           rocsparse_status_invalid_size);
        const rocsparse_operation op_B = *reinterpret_cast<const rocsparse_operation*>(data);
        descr->set_operation_B(op_B);
        return rocsparse_status_success;
    }
    }
    return rocsparse_status_invalid_value;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

/********************************************************************************
 * \brief rocsparse_spgeam_get_output gets the output from the SpGEAM descriptor.
 *******************************************************************************/
rocsparse_status rocsparse_spgeam_get_output(rocsparse_handle        handle,
                                             rocsparse_spgeam_descr  descr,
                                             rocsparse_spgeam_output output,
                                             void*                   data,
                                             size_t                  data_size_in_bytes,
                                             rocsparse_error*        p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, descr);
    ROCSPARSE_CHECKARG_ENUM(2, output);
    ROCSPARSE_CHECKARG_POINTER(3, data);
    switch(output)
    {
    case rocsparse_spgeam_output_nnz:
    {
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(int64_t),
                           rocsparse_status_invalid_size);
        int64_t* nnz_C = reinterpret_cast<int64_t*>(data);
        *nnz_C         = descr->nnz_C;
        return rocsparse_status_success;
    }
    }

    return rocsparse_status_invalid_value;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

#ifdef __cplusplus
}
#endif
