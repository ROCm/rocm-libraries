/* ************************************************************************
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/generic/rocsparse_check_spmat.h"
#include "rocsparse_control.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_check_matrix_coo.hpp"
#include "rocsparse_check_matrix_csc.hpp"
#include "rocsparse_check_matrix_csr.hpp"
#include "rocsparse_check_matrix_ell.hpp"
#include "rocsparse_check_matrix_gebsr.hpp"
#include "rocsparse_determine_indextype.hpp"

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_check_spmat_stage value_)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value_)
    {
        CASE(rocsparse_check_spmat_stage_buffer_size);
        CASE(rocsparse_check_spmat_stage_compute);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_data_status data_status)
{
    switch(data_status)
    {
    case rocsparse_data_status_success:
        return "No errors in data detected";
    case rocsparse_data_status_inf:
        return "An inf value was found in the values array.";
    case rocsparse_data_status_nan:
        return "An nan value was found in the values array.";
    case rocsparse_data_status_invalid_offset_ptr:
        return "An invalid offset pointer was detected.";
    case rocsparse_data_status_invalid_index:
        return "An invalid index was detected.";
    case rocsparse_data_status_duplicate_entry:
        return "A duplicate entry was detected.";
    case rocsparse_data_status_invalid_sorting:
        return "Sorting mode was detected to be invalid.";
    case rocsparse_data_status_invalid_fill:
        return "Fill mode was detected to be invalid.";
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_check_spmat_stage value_)
{
    switch(value_)
    {
    case rocsparse_check_spmat_stage_buffer_size:
    case rocsparse_check_spmat_stage_compute:
    {
        return false;
    }
    }
    return true;
}

namespace rocsparse
{

    template <typename I, typename J, typename T>
    rocsparse_status check_spmat_template(rocsparse_handle            handle,
                                          rocsparse_const_spmat_descr mat,
                                          rocsparse_data_status*      data_status,
                                          rocsparse_check_spmat_stage stage,
                                          size_t*                     buffer_size,
                                          void*                       temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(mat->format)
        {
        case rocsparse_format_coo:
        {
            switch(stage)
            {
            case rocsparse_check_spmat_stage_buffer_size:
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::check_matrix_coo_buffer_size_impl<T, I>(
                    handle,
                    (I)mat->rows,
                    (I)mat->cols,
                    mat->nnz,
                    (const T*)mat->const_val_data,
                    (const I*)mat->const_row_data,
                    (const I*)mat->const_col_data,
                    mat->idx_base,
                    (mat->descr)->type,
                    (mat->descr)->fill_mode,
                    (mat->descr)->storage_mode,
                    buffer_size)));
                return rocsparse_status_success;
            }
            case rocsparse_check_spmat_stage_compute:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    (rocsparse::check_matrix_coo_impl<T, I>(handle,
                                                            (I)mat->rows,
                                                            (I)mat->cols,
                                                            mat->nnz,
                                                            (const T*)mat->const_val_data,
                                                            (const I*)mat->const_row_data,
                                                            (const I*)mat->const_col_data,
                                                            mat->idx_base,
                                                            (mat->descr)->type,
                                                            (mat->descr)->fill_mode,
                                                            (mat->descr)->storage_mode,
                                                            data_status,
                                                            temp_buffer)));
                return rocsparse_status_success;
            }
            }
        }

        case rocsparse_format_csr:
        {
            switch(stage)
            {
            case rocsparse_check_spmat_stage_buffer_size:
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::check_matrix_csr_buffer_size_impl<T, I, J>(
                    handle,
                    (J)mat->rows,
                    (J)mat->cols,
                    (I)mat->nnz,
                    (const T*)mat->const_val_data,
                    (const I*)mat->const_row_data,
                    (const J*)mat->const_col_data,
                    mat->idx_base,
                    (mat->descr)->type,
                    (mat->descr)->fill_mode,
                    (mat->descr)->storage_mode,
                    buffer_size)));
                return rocsparse_status_success;
            }
            case rocsparse_check_spmat_stage_compute:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    (rocsparse::check_matrix_csr_impl<T, I, J>(handle,
                                                               (J)mat->rows,
                                                               (J)mat->cols,
                                                               (I)mat->nnz,
                                                               (const T*)mat->const_val_data,
                                                               (const I*)mat->const_row_data,
                                                               (const J*)mat->const_col_data,
                                                               mat->idx_base,
                                                               (mat->descr)->type,
                                                               (mat->descr)->fill_mode,
                                                               (mat->descr)->storage_mode,
                                                               data_status,
                                                               temp_buffer)));
                return rocsparse_status_success;
            }
            }
        }

        case rocsparse_format_csc:
        {
            switch(stage)
            {
            case rocsparse_check_spmat_stage_buffer_size:
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::check_matrix_csc_buffer_size_impl<T, I, J>(
                    handle,
                    (J)mat->rows,
                    (J)mat->cols,
                    (I)mat->nnz,
                    (const T*)mat->const_val_data,
                    (const I*)mat->const_col_data,
                    (const J*)mat->const_row_data,
                    mat->idx_base,
                    (mat->descr)->type,
                    (mat->descr)->fill_mode,
                    (mat->descr)->storage_mode,
                    buffer_size)));
                return rocsparse_status_success;
            }
            case rocsparse_check_spmat_stage_compute:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    (rocsparse::check_matrix_csc_impl<T, I, J>(handle,
                                                               (J)mat->rows,
                                                               (J)mat->cols,
                                                               (I)mat->nnz,
                                                               (const T*)mat->const_val_data,
                                                               (const I*)mat->const_col_data,
                                                               (const J*)mat->const_row_data,
                                                               mat->idx_base,
                                                               (mat->descr)->type,
                                                               (mat->descr)->fill_mode,
                                                               (mat->descr)->storage_mode,
                                                               data_status,
                                                               temp_buffer)));
                return rocsparse_status_success;
            }
            }
        }

        case rocsparse_format_ell:
        {
            switch(stage)
            {
            case rocsparse_check_spmat_stage_buffer_size:
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::check_matrix_ell_buffer_size_impl<T, I>(
                    handle,
                    (I)mat->rows,
                    (I)mat->cols,
                    (I)mat->ell_width,
                    (const T*)mat->const_val_data,
                    (const I*)mat->const_col_data,
                    mat->idx_base,
                    (mat->descr)->type,
                    (mat->descr)->fill_mode,
                    (mat->descr)->storage_mode,
                    buffer_size)));
                return rocsparse_status_success;
            }
            case rocsparse_check_spmat_stage_compute:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    (rocsparse::check_matrix_ell_impl<T, I>(handle,
                                                            (I)mat->rows,
                                                            (I)mat->cols,
                                                            (I)mat->ell_width,
                                                            (const T*)mat->const_val_data,
                                                            (const I*)mat->const_col_data,
                                                            mat->idx_base,
                                                            (mat->descr)->type,
                                                            (mat->descr)->fill_mode,
                                                            (mat->descr)->storage_mode,
                                                            data_status,
                                                            temp_buffer)));
                return rocsparse_status_success;
            }
            }
        }

        case rocsparse_format_bsr:
        {
            switch(stage)
            {
            case rocsparse_check_spmat_stage_buffer_size:
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse::check_matrix_gebsr_buffer_size_impl<T, I, J>(
                    handle,
                    mat->block_dir,
                    (J)mat->rows,
                    (J)mat->cols,
                    (I)mat->nnz,
                    (J)mat->block_dim,
                    (J)mat->block_dim,
                    (const T*)mat->const_val_data,
                    (const I*)mat->const_row_data,
                    (const J*)mat->const_col_data,
                    mat->idx_base,
                    (mat->descr)->type,
                    (mat->descr)->fill_mode,
                    (mat->descr)->storage_mode,
                    buffer_size)));
                return rocsparse_status_success;
            }
            case rocsparse_check_spmat_stage_compute:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    (rocsparse::check_matrix_gebsr_impl<T, I, J>(handle,
                                                                 mat->block_dir,
                                                                 (J)mat->rows,
                                                                 (J)mat->cols,
                                                                 (I)mat->nnz,
                                                                 (J)mat->block_dim,
                                                                 (J)mat->block_dim,
                                                                 (const T*)mat->const_val_data,
                                                                 (const I*)mat->const_row_data,
                                                                 (const J*)mat->const_col_data,
                                                                 mat->idx_base,
                                                                 (mat->descr)->type,
                                                                 (mat->descr)->fill_mode,
                                                                 (mat->descr)->storage_mode,
                                                                 data_status,
                                                                 temp_buffer)));
                return rocsparse_status_success;
            }
            }
        }

        case rocsparse_format_coo_aos:
        case rocsparse_format_bell:
        {
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            // LCOV_EXCL_STOP
        }
        }

        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    template <typename... Ts>
    rocsparse_status check_spmat_dynamic_dispatch(rocsparse_indextype itype,
                                                  rocsparse_indextype jtype,
                                                  rocsparse_datatype  ctype,
                                                  rocsparse_format    format,
                                                  Ts&&... ts)
    {
        ROCSPARSE_ROUTINE_TRACE;

#define DISPATCH_COMPUTE_TYPE(ITYPE, JTYPE, CTYPE)                                                 \
    switch(CTYPE)                                                                                  \
    {                                                                                              \
    case rocsparse_datatype_f32_r:                                                                 \
    {                                                                                              \
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::check_spmat_template<ITYPE, JTYPE, float>(ts...)));  \
        return rocsparse_status_success;                                                           \
    }                                                                                              \
    case rocsparse_datatype_f64_r:                                                                 \
    {                                                                                              \
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::check_spmat_template<ITYPE, JTYPE, double>(ts...))); \
        return rocsparse_status_success;                                                           \
    }                                                                                              \
    case rocsparse_datatype_f32_c:                                                                 \
    {                                                                                              \
        RETURN_IF_ROCSPARSE_ERROR(                                                                 \
            (rocsparse::check_spmat_template<ITYPE, JTYPE, rocsparse_float_complex>(ts...)));      \
        return rocsparse_status_success;                                                           \
    }                                                                                              \
    case rocsparse_datatype_f64_c:                                                                 \
    {                                                                                              \
        RETURN_IF_ROCSPARSE_ERROR(                                                                 \
            (rocsparse::check_spmat_template<ITYPE, JTYPE, rocsparse_double_complex>(ts...)));     \
        return rocsparse_status_success;                                                           \
    }                                                                                              \
    case rocsparse_datatype_i8_r:                                                                  \
    case rocsparse_datatype_u8_r:                                                                  \
    case rocsparse_datatype_i32_r:                                                                 \
    case rocsparse_datatype_u32_r:                                                                 \
    case rocsparse_datatype_f16_r:                                                                 \
    case rocsparse_datatype_bf16_r:                                                                \
    {                                                                                              \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);                               \
    }                                                                                              \
    }

        switch(itype)
        {
        case rocsparse_indextype_u16:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        case rocsparse_indextype_i32:
        {
            switch(jtype)
            {
            case rocsparse_indextype_u16:
            case rocsparse_indextype_i64:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            case rocsparse_indextype_i32:
            {
                DISPATCH_COMPUTE_TYPE(int32_t, int32_t, ctype);
            }
            }
        }
        case rocsparse_indextype_i64:
        {
            switch(jtype)
            {
            case rocsparse_indextype_u16:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            case rocsparse_indextype_i32:
            {
                DISPATCH_COMPUTE_TYPE(int64_t, int32_t, ctype);
            }
            case rocsparse_indextype_i64:
            {
                DISPATCH_COMPUTE_TYPE(int64_t, int64_t, ctype);
            }
            }
        }
        }

        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }
}

extern "C" rocsparse_status rocsparse_check_spmat(rocsparse_handle            handle,
                                                  rocsparse_const_spmat_descr mat,
                                                  rocsparse_data_status*      data_status,
                                                  rocsparse_check_spmat_stage stage,
                                                  size_t*                     buffer_size,
                                                  void*                       temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_check_spmat",
                         (const void*&)mat,
                         (const void*&)data_status,
                         stage,
                         (const void*&)buffer_size,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, mat);
    ROCSPARSE_CHECKARG_POINTER(2, data_status);
    ROCSPARSE_CHECKARG_ENUM(3, stage);

    ROCSPARSE_CHECKARG(4,
                       buffer_size,
                       ((temp_buffer == nullptr) && (buffer_size == nullptr)),
                       rocsparse_status_invalid_pointer);

    ROCSPARSE_CHECKARG(1, mat, (mat->init == false), rocsparse_status_not_initialized);

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::check_spmat_dynamic_dispatch(rocsparse::determine_I_indextype(mat),
                                                rocsparse::determine_J_indextype(mat),
                                                mat->data_type,
                                                mat->format,
                                                handle,
                                                mat,
                                                data_status,
                                                stage,
                                                buffer_size,
                                                temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
