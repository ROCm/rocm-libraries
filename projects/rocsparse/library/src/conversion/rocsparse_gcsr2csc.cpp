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

#include "rocsparse_gcsr2csc.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_convert_array.hpp"
#include "rocsparse_csr2csc.hpp"
#include "rocsparse_internal_convert_scalar.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::gcsr2csc_buffer_size(rocsparse_handle    handle,
                                                 int64_t             m,
                                                 int64_t             n,
                                                 int64_t             nnz,
                                                 rocsparse_indextype indextype_ptr,
                                                 rocsparse_indextype indextype_ind,
                                                 const void*         csr_row_ptr,
                                                 const void*         csr_col_ind,
                                                 rocsparse_action    copy_values,
                                                 size_t*             buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

#define CALL_TEMPLATE(PTRTYPE, INDTYPE)                                                          \
    PTRTYPE local_nnz;                                                                           \
    INDTYPE local_m, local_n;                                                                    \
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::internal_convert_scalar<INDTYPE>(m, local_m));          \
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::internal_convert_scalar<INDTYPE>(n, local_n));          \
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::internal_convert_scalar<PTRTYPE>(nnz, local_nnz));      \
    RETURN_IF_ROCSPARSE_ERROR(                                                                   \
        (rocsparse::csr2csc_buffer_size_template<PTRTYPE, INDTYPE>)(handle,                      \
                                                                    local_m,                     \
                                                                    local_n,                     \
                                                                    local_nnz,                   \
                                                                    (const PTRTYPE*)csr_row_ptr, \
                                                                    (const INDTYPE*)csr_col_ind, \
                                                                    copy_values,                 \
                                                                    buffer_size))

#define DISPATCH_INDEX_TYPE_IND(PTRTYPE)                             \
    switch(indextype_ind)                                            \
    {                                                                \
    case rocsparse_indextype_u16:                                    \
    {                                                                \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented); \
    }                                                                \
    case rocsparse_indextype_i32:                                    \
    {                                                                \
        CALL_TEMPLATE(PTRTYPE, int32_t);                             \
        return rocsparse_status_success;                             \
    }                                                                \
    case rocsparse_indextype_i64:                                    \
    {                                                                \
        CALL_TEMPLATE(PTRTYPE, int64_t);                             \
        return rocsparse_status_success;                             \
    }                                                                \
    }                                                                \
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value)

    switch(indextype_ptr)
    {
    case rocsparse_indextype_u16:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        return rocsparse_status_success;
    }
    case rocsparse_indextype_i32:
    {
        DISPATCH_INDEX_TYPE_IND(int32_t);
        return rocsparse_status_success;
    }
    case rocsparse_indextype_i64:
    {
        DISPATCH_INDEX_TYPE_IND(int64_t);
        return rocsparse_status_success;
    }
    }
    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP

#undef CALL_TEMPLATE
#undef DISPATCH_INDEX_TYPE_IND
}

rocsparse_status rocsparse::gcsr2csc(rocsparse_handle     handle,
                                     int64_t              m,
                                     int64_t              n,
                                     int64_t              nnz,
                                     rocsparse_datatype   datatype,
                                     rocsparse_indextype  indextype_ptr,
                                     rocsparse_indextype  indextype_ind,
                                     const void*          csr_val,
                                     const void*          csr_row_ptr,
                                     const void*          csr_col_ind,
                                     void*                csc_val,
                                     void*                csc_row_ind,
                                     void*                csc_col_ptr,
                                     rocsparse_action     copy_values,
                                     rocsparse_index_base idx_base,
                                     void*                temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

#define CALL_TEMPLATE(DATATYPE, PTRTYPE, INDTYPE)                                              \
    PTRTYPE local_nnz;                                                                         \
    INDTYPE local_m, local_n;                                                                  \
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::internal_convert_scalar<INDTYPE>(m, local_m));        \
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::internal_convert_scalar<INDTYPE>(n, local_n));        \
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::internal_convert_scalar<PTRTYPE>(nnz, local_nnz));    \
    RETURN_IF_ROCSPARSE_ERROR(                                                                 \
        (rocsparse::csr2csc_template<PTRTYPE, INDTYPE, DATATYPE>)(handle,                      \
                                                                  local_m,                     \
                                                                  local_n,                     \
                                                                  local_nnz,                   \
                                                                  (const DATATYPE*)csr_val,    \
                                                                  (const PTRTYPE*)csr_row_ptr, \
                                                                  (const INDTYPE*)csr_col_ind, \
                                                                  (DATATYPE*)csc_val,          \
                                                                  (INDTYPE*)csc_row_ind,       \
                                                                  (PTRTYPE*)csc_col_ptr,       \
                                                                  copy_values,                 \
                                                                  idx_base,                    \
                                                                  temp_buffer))

#define DISPATCH_INDEX_TYPE_IND(DATATYPE, PTRTYPE)                   \
    switch(indextype_ind)                                            \
    {                                                                \
    case rocsparse_indextype_u16:                                    \
    {                                                                \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented); \
        return rocsparse_status_success;                             \
    }                                                                \
    case rocsparse_indextype_i32:                                    \
    {                                                                \
        CALL_TEMPLATE(DATATYPE, PTRTYPE, int32_t);                   \
        return rocsparse_status_success;                             \
    }                                                                \
    case rocsparse_indextype_i64:                                    \
    {                                                                \
        CALL_TEMPLATE(DATATYPE, PTRTYPE, int64_t);                   \
        return rocsparse_status_success;                             \
    }                                                                \
    }                                                                \
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value)

#define DISPATCH_INDEX_TYPE_PTR(DATATYPE)                            \
    switch(indextype_ptr)                                            \
    {                                                                \
    case rocsparse_indextype_u16:                                    \
    {                                                                \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented); \
        return rocsparse_status_success;                             \
    }                                                                \
    case rocsparse_indextype_i32:                                    \
    {                                                                \
        DISPATCH_INDEX_TYPE_IND(DATATYPE, int32_t);                  \
        return rocsparse_status_success;                             \
    }                                                                \
    case rocsparse_indextype_i64:                                    \
    {                                                                \
        DISPATCH_INDEX_TYPE_IND(DATATYPE, int64_t);                  \
        return rocsparse_status_success;                             \
    }                                                                \
    }                                                                \
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value)

    switch(datatype)
    {
    case rocsparse_datatype_u8_r:
    {
        DISPATCH_INDEX_TYPE_PTR(uint8_t);
    }
    case rocsparse_datatype_i8_r:
    {
        DISPATCH_INDEX_TYPE_PTR(int8_t);
    }
    case rocsparse_datatype_f16_r:
    case rocsparse_datatype_bf16_r:
    {
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }
    case rocsparse_datatype_u32_r:
    {
        DISPATCH_INDEX_TYPE_PTR(uint32_t);
    }
    case rocsparse_datatype_i32_r:
    {
        DISPATCH_INDEX_TYPE_PTR(int32_t);
    }
    case rocsparse_datatype_f32_r:
    {
        DISPATCH_INDEX_TYPE_PTR(float);
    }
    case rocsparse_datatype_f32_c:
    {
        DISPATCH_INDEX_TYPE_PTR(rocsparse_float_complex);
    }
    case rocsparse_datatype_f64_r:
    {
        DISPATCH_INDEX_TYPE_PTR(double);
    }
    case rocsparse_datatype_f64_c:
    {
        DISPATCH_INDEX_TYPE_PTR(rocsparse_double_complex);
    }
    }

    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP

#undef CALL_TEMPLATE
#undef DISPATCH_INDEX_TYPE_IND
#undef DISPATCH_INDEX_TYPE_PTR
}

rocsparse_status rocsparse::spmat_csr2csc_buffer_size(rocsparse_handle            handle,
                                                      rocsparse_const_spmat_descr source_,
                                                      rocsparse_spmat_descr       target_,
                                                      size_t*                     buffer_size_)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gcsr2csc_buffer_size(
        handle,
        source_->rows,
        source_->cols,
        source_->nnz,
        //
        source_->row_type,
        source_->col_type,
        source_->const_row_data,
        source_->const_col_data,
        //
        (source_->const_val_data != nullptr && target_->val_data != nullptr)
            ? rocsparse_action_numeric
            : rocsparse_action_symbolic,
        buffer_size_));

    return rocsparse_status_success;
}

rocsparse_status rocsparse::spmat_csr2csc(rocsparse_handle            handle,
                                          rocsparse_const_spmat_descr source_,
                                          rocsparse_spmat_descr       target_,
                                          size_t                      buffer_size_,
                                          void*                       buffer_)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_ROCSPARSE_ERROR_IF(rocsparse_status_not_implemented,
                              source_->row_type != target_->col_type);
    RETURN_ROCSPARSE_ERROR_IF(rocsparse_status_not_implemented,
                              source_->col_type != target_->row_type);
    if(target_->val_data != nullptr && source_->const_val_data != nullptr)
    {
        RETURN_ROCSPARSE_ERROR_IF(rocsparse_status_not_implemented,
                                  source_->data_type != target_->data_type);
    }

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::gcsr2csc(handle,
                            source_->rows,
                            source_->cols,
                            source_->nnz,
                            //
                            source_->data_type,
                            source_->row_type,
                            source_->col_type,
                            //
                            source_->const_val_data,
                            source_->const_row_data,
                            source_->const_col_data,
                            //
                            target_->val_data,
                            target_->row_data,
                            target_->col_data,
                            //
                            (target_->val_data != nullptr && source_->const_val_data != nullptr)
                                ? rocsparse_action_numeric
                                : rocsparse_action_symbolic,
                            source_->idx_base,
                            buffer_));

    return rocsparse_status_success;
}
