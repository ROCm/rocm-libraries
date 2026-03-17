/* ************************************************************************
 * Copyright (C) 2018-2026 Advanced Micro Devices, Inc. All rights Reserved.
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
        CASE(rocsparse_format_sell);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_batchtype value_)
{
    switch(value_)
    {
    case rocsparse_batchtype_pointerarray:
    case rocsparse_batchtype_strided:
    {
        return false;
    }
    }
    return true;
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_batchstorage value_)
{
    switch(value_)
    {
    case rocsparse_batchstorage_soa:
    case rocsparse_batchstorage_aos:
    {
        return false;
    }
    }
    return true;
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

_rocsparse_spmat_descr::_rocsparse_spmat_descr(rocsparse_format     format_,
                                               bool                 analysed_,
                                               int64_t              batch_count_,
                                               int64_t              m_,
                                               int64_t              n_,
                                               int64_t              nnz_,
                                               rocsparse_direction  block_dir_,
                                               int64_t              block_dim_,
                                               rocsparse_datatype   val_datatype_,
                                               const void*          const_val_data_,
                                               void*                val_data_,
                                               int64_t              val_stride_,
                                               rocsparse_indextype  row_indextype_,
                                               const void*          const_row_data_,
                                               void*                row_data_,
                                               int64_t              row_stride_,
                                               rocsparse_indextype  col_indextype_,
                                               const void*          const_col_data_,
                                               void*                col_data_,
                                               int64_t              col_stride_,
                                               rocsparse_index_base base_,
                                               rocsparse_mat_descr  descr_,
                                               rocsparse_mat_info   info_)

{
    auto    spattern = this->get_spattern();
    auto    row      = spattern->get_row_data();
    auto    col      = spattern->get_col_data();
    auto    val      = this->get_values();
    int64_t row_size{};
    int64_t col_size{};
    int64_t val_size{};
    switch(format_)
    {
    case rocsparse_format_csr:
    {
        break;
    }

    case rocsparse_format_bsr:
    {
        row_size = m_ + 1;
        col_size = nnz_;
        val_size = nnz_ * block_dim_ * block_dim_;
        break;
    }

    case rocsparse_format_bell:
    {
#if 0
	row_size = m_ + 1;
	col_size = m_ * width_;
	val_size = m_ * width_ * block_dim_ * block_dim_;
#endif
        break;
    }

    case rocsparse_format_sell:
    {
#if 0
	row_size = m_ + 1;
	col_size = m_ * width_;
	val_size = m_ * width_;
#endif
        break;
    }

    case rocsparse_format_csc:
    {
        break;
    }

    case rocsparse_format_coo:
    {
        break;
    }

    case rocsparse_format_coo_aos:
    {
        break;
    }

    case rocsparse_format_ell:
    {
#if 0
	row_size = 0;
	col_size = m_ * width_;
	val_size = nnz_;
#endif
        break;
    }
    }

    row->define(row_indextype_, base_, row_size, 1, const_row_data_, row_data_);
    col->define(col_indextype_, base_, col_size, 1, const_col_data_, col_data_);
    val->define(val_datatype_,
                val_size,
                1,
                rocsparse_batchtype_strided,
                rocsparse_batchstorage_soa,
                batch_count_,
                val_stride_,
                const_val_data_,
                val_data_);

    switch(format_)
    {
    case rocsparse_format_csr:
    {
        break;
    }

    case rocsparse_format_bsr:
    {
        spattern->define_bsr(m_, n_, nnz_, block_dir_, block_dim_, row, col, descr_, info_);

        break;
    }

    case rocsparse_format_bell:
    {
#if 0
	spattern->define_bell(rows,
			      cols,
			      width,
			      row,
			      col,
			      descr_,
      info_);
#endif
        break;
    }

    case rocsparse_format_sell:
    {
        break;
    }

    case rocsparse_format_csc:
    case rocsparse_format_coo:
    case rocsparse_format_coo_aos:
    case rocsparse_format_ell:
    {
        break;
    }
    }

    this->define(spattern, val, info_);
    this->set_analysed(analysed_);
}

_rocsparse_spmat_descr::_rocsparse_spmat_descr(rocsparse_format     format_,
                                               bool                 analysed_,
                                               int64_t              batch_count_,
                                               int64_t              m_,
                                               int64_t              n_,
                                               int64_t              nnz_,
                                               rocsparse_datatype   val_datatype_,
                                               const void*          const_val_data_,
                                               void*                val_data_,
                                               int64_t              val_stride_,
                                               rocsparse_indextype  row_indextype_,
                                               const void*          const_row_data_,
                                               void*                row_data_,
                                               int64_t              row_stride_,
                                               rocsparse_indextype  col_indextype_,
                                               const void*          const_col_data_,
                                               void*                col_data_,
                                               int64_t              col_stride_,
                                               rocsparse_index_base base_,
                                               rocsparse_mat_descr  descr_,
                                               rocsparse_mat_info   info_)
{
    auto    spattern = this->get_spattern();
    auto    row      = spattern->get_row_data();
    auto    col      = spattern->get_col_data();
    auto    val      = this->get_values();
    int64_t row_size{};
    int64_t col_size{};
    int64_t val_size{};
    switch(format_)
    {
    case rocsparse_format_csr:
    {
        row_size = m_ + 1;
        col_size = nnz_;
        val_size = nnz_;
        break;
    }

    case rocsparse_format_bsr:
    case rocsparse_format_bell:
    {
        break;
    }

    case rocsparse_format_sell:
    {
#if 0
	row_size = m_ + 1;
	col_size = m_ * width_;
	val_size = m_ * width_;
#endif
        break;
    }

    case rocsparse_format_csc:
    {
        row_size = nnz_;
        col_size = n_ + 1;
        val_size = nnz_;
        break;
    }

    case rocsparse_format_coo:
    {
        row_size = nnz_;
        col_size = nnz_;
        val_size = nnz_;
        break;
    }

    case rocsparse_format_coo_aos:
    {
        row_size = nnz_;
        col_size = nnz_;
        val_size = nnz_;
        break;
    }

    case rocsparse_format_ell:
    {
#if 0
	row_size = 0;
	col_size = m_ * width_;
	val_size = m_ * width_;
#endif
        break;
    }
    }

    row->define(row_indextype_, base_, row_size, 1, const_row_data_, row_data_);

    col->define(col_indextype_, base_, col_size, 1, const_col_data_, col_data_);

    val->define(val_datatype_,
                val_size,
                1,
                rocsparse_batchtype_strided,
                rocsparse_batchstorage_soa,
                batch_count_,
                val_stride_,
                const_val_data_,
                val_data_);

    switch(format_)
    {

    case rocsparse_format_csr:
    {
        spattern->define_csr(m_, n_, nnz_, row, col, descr_, info_);
        break;
    }

    case rocsparse_format_bsr:
    {
#if 0
	spattern->define_bsr(rows,
			     cols,
			     nnz,
			     row,
			     col,
			     descr_,
      info_);
#endif
        break;
    }

    case rocsparse_format_bell:
    {
#if 0
	spattern->define_bell(rows,
			      cols,
			      width,
			      row,
			      col,
			      descr_,
      info_);
#endif
        break;
    }

    case rocsparse_format_sell:
    {
        break;
    }

    case rocsparse_format_csc:
    {

        spattern->define_csc(m_, n_, nnz_, row, col, descr_, info_);
        break;
    }

    case rocsparse_format_coo:
    {
        spattern->define_coo(m_, n_, nnz_, row, col, descr_, info_);
        break;
    }

    case rocsparse_format_coo_aos:
    {
        spattern->define_coo_aos(m_, n_, nnz_, row, col, descr_, info_);
        break;
    }

    case rocsparse_format_ell:
    {
        break;
    }
    }

    this->define(spattern, val);
    this->set_analysed(analysed_);
}

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
    static constexpr int64_t batch_count = 1;
    static constexpr int64_t inc         = 1;
    static constexpr int64_t batch_dist  = 0;
    descr[0]
        = new _rocsparse_dnvec_descr(batch_count, size, data_type, values, values, inc, batch_dist);
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

    static constexpr int64_t batch_count = 1;
    static constexpr int64_t inc         = 1;
    static constexpr int64_t batch_dist  = 0;
    descr[0]                             = new _rocsparse_dnvec_descr(
        batch_count, size, data_type, values, nullptr, inc, batch_dist);
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
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, size);
    ROCSPARSE_CHECKARG_POINTER(2, values);
    ROCSPARSE_CHECKARG_POINTER(3, data_type);

    *size      = descr->get_size();
    *values    = descr->get_values();
    *data_type = descr->get_data_type();

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
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, size);
    ROCSPARSE_CHECKARG_POINTER(2, values);
    ROCSPARSE_CHECKARG_POINTER(3, data_type);

    *size      = descr->get_size();
    *values    = descr->get_const_values();
    *data_type = descr->get_data_type();

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
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, values);

    *values = descr->get_values();

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
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, values);

    *values = descr->get_const_values();

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
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, values);
    descr->set_values(values);
    descr->set_const_values(values);

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

    *descr = new _rocsparse_dnmat_descr(data_type, order, rows, cols, ld, values, values);

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

    *descr = new _rocsparse_dnmat_descr(data_type, order, rows, cols, ld, values, nullptr);

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
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, ld);
    ROCSPARSE_CHECKARG_POINTER(4, values);
    ROCSPARSE_CHECKARG_POINTER(5, data_type);
    ROCSPARSE_CHECKARG_POINTER(6, order);

    *rows      = descr->get_rows();
    *cols      = descr->get_cols();
    *ld        = descr->get_ld();
    *values    = descr->get_values();
    *data_type = descr->get_data_type();
    *order     = descr->get_order();

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
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, rows);
    ROCSPARSE_CHECKARG_POINTER(2, cols);
    ROCSPARSE_CHECKARG_POINTER(3, ld);
    ROCSPARSE_CHECKARG_POINTER(4, values);
    ROCSPARSE_CHECKARG_POINTER(5, data_type);
    ROCSPARSE_CHECKARG_POINTER(6, order);

    *rows      = descr->get_rows();
    *cols      = descr->get_cols();
    *ld        = descr->get_ld();
    *values    = descr->get_const_values();
    *data_type = descr->get_data_type();
    *order     = descr->get_order();

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
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, values);
    *values = descr->get_values();

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
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, values);

    *values = descr->get_const_values();

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
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, values);

    descr->set_values(values);
    descr->set_const_values(values);

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
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG_POINTER(1, batch_count);
    ROCSPARSE_CHECKARG_POINTER(2, batch_stride);

    *batch_count  = descr->get_batch_count();
    *batch_stride = descr->get_batch_stride();

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
    ROCSPARSE_CHECKARG(0, descr, (descr->get_init() == false), rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(1, batch_count, (batch_count <= 0), rocsparse_status_invalid_value);
    ROCSPARSE_CHECKARG(2, batch_stride, (batch_stride < 0), rocsparse_status_invalid_value);

    if(descr->get_order() == rocsparse_order_column)
    {
        ROCSPARSE_CHECKARG(2,
                           batch_stride,
                           (batch_count > 1 && batch_stride < descr->get_ld() * descr->get_cols()),
                           rocsparse_status_invalid_value);
    }
    else if(descr->get_order() == rocsparse_order_row)
    {
        ROCSPARSE_CHECKARG(2,
                           batch_stride,
                           (batch_count > 1 && batch_stride < descr->get_ld() * descr->get_rows()),
                           rocsparse_status_invalid_value);
    }

    descr->set_batch_count(batch_count);
    descr->set_batch_stride(batch_stride);

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
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

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
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

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
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

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
