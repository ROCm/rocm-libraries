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

#include "rocsparse_gbsr2csr.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_bsr2csr.hpp"

namespace rocsparse
{
    template <typename T, typename I>
    static rocsparse_status gbsr2csr_b(rocsparse_handle          handle,
                                       rocsparse_direction       dir,
                                       int64_t                   mb,
                                       int64_t                   nb,
                                       const rocsparse_mat_descr bsr_descr,
                                       const T*                  bsr_val,
                                       const I*                  bsr_row_ptr,
                                       rocsparse_indextype       bsr_col_ind_indextype,
                                       const void*               bsr_col_ind,
                                       int64_t                   block_dim,
                                       const rocsparse_mat_descr csr_descr,
                                       T*                        csr_val,
                                       I*                        csr_row_ptr,
                                       rocsparse_indextype       csr_col_ind_indextype,
                                       void*                     csr_col_ind)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(csr_col_ind_indextype != bsr_col_ind_indextype)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }

        switch(csr_col_ind_indextype)

        {
        case rocsparse_indextype_u16:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }

#define CASE(VAL, TYPE)                                                                 \
    case VAL:                                                                           \
    {                                                                                   \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsr2csr_template(handle,                   \
                                                              dir,                      \
                                                              (TYPE)mb,                 \
                                                              (TYPE)nb,                 \
                                                              bsr_descr,                \
                                                              bsr_val,                  \
                                                              bsr_row_ptr,              \
                                                              (const TYPE*)bsr_col_ind, \
                                                              (TYPE)block_dim,          \
                                                              csr_descr,                \
                                                              csr_val,                  \
                                                              csr_row_ptr,              \
                                                              (TYPE*)csr_col_ind));     \
        return rocsparse_status_success;                                                \
    }

            CASE(rocsparse_indextype_i32, int32_t);
            CASE(rocsparse_indextype_i64, int64_t);
#undef CASE
        }
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    template <typename T>
    static rocsparse_status gbsr2csr_a(rocsparse_handle          handle,
                                       rocsparse_direction       dir,
                                       int64_t                   mb,
                                       int64_t                   nb,
                                       const rocsparse_mat_descr bsr_descr,
                                       const T*                  bsr_val,
                                       rocsparse_indextype       bsr_row_ptr_indextype,
                                       const void*               bsr_row_ptr,
                                       rocsparse_indextype       bsr_col_ind_indextype,
                                       const void*               bsr_col_ind,
                                       int64_t                   block_dim,
                                       const rocsparse_mat_descr csr_descr,
                                       T*                        csr_val,
                                       rocsparse_indextype       csr_row_ptr_indextype,
                                       void*                     csr_row_ptr,
                                       rocsparse_indextype       csr_col_ind_indextype,
                                       void*                     csr_col_ind)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(csr_row_ptr_indextype)
        {
        case rocsparse_indextype_u16:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }

#define CASE(VAL, TYPE)                                                           \
    case VAL:                                                                     \
    {                                                                             \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gbsr2csr_b(handle,                   \
                                                        dir,                      \
                                                        mb,                       \
                                                        nb,                       \
                                                        bsr_descr,                \
                                                        bsr_val,                  \
                                                        (const TYPE*)bsr_row_ptr, \
                                                        bsr_col_ind_indextype,    \
                                                        bsr_col_ind,              \
                                                        block_dim,                \
                                                        csr_descr,                \
                                                        csr_val,                  \
                                                        (TYPE*)csr_row_ptr,       \
                                                        csr_col_ind_indextype,    \
                                                        csr_col_ind));            \
        return rocsparse_status_success;                                          \
    }

            CASE(rocsparse_indextype_i32, int32_t);
            CASE(rocsparse_indextype_i64, int64_t);
#undef CASE
        }
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }
}

rocsparse_status rocsparse::gbsr2csr(rocsparse_handle          handle,
                                     rocsparse_direction       dir,
                                     int64_t                   mb,
                                     int64_t                   nb,
                                     const rocsparse_mat_descr bsr_descr,
                                     rocsparse_datatype        bsr_val_datatype,
                                     const void*               bsr_val,
                                     rocsparse_indextype       bsr_row_ptr_indextype,
                                     const void*               bsr_row_ptr,
                                     rocsparse_indextype       bsr_col_ind_indextype,
                                     const void*               bsr_col_ind,
                                     int64_t                   block_dim,
                                     const rocsparse_mat_descr csr_descr,
                                     rocsparse_datatype        csr_val_datatype,
                                     void*                     csr_val,
                                     rocsparse_indextype       csr_row_ptr_indextype,
                                     void*                     csr_row_ptr,
                                     rocsparse_indextype       csr_col_ind_indextype,
                                     void*                     csr_col_ind)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_ROCSPARSE_ERROR_IF(rocsparse_status_not_implemented,
                              bsr_row_ptr_indextype != csr_row_ptr_indextype);
    RETURN_ROCSPARSE_ERROR_IF(rocsparse_status_not_implemented,
                              bsr_col_ind_indextype != csr_col_ind_indextype);
    if(csr_val != nullptr && bsr_val != nullptr)
    {
        RETURN_ROCSPARSE_ERROR_IF(rocsparse_status_not_implemented,
                                  bsr_val_datatype != csr_val_datatype);
    }

    switch(csr_val_datatype)
    {
    case rocsparse_datatype_i8_r:
    case rocsparse_datatype_u8_r:
    case rocsparse_datatype_u32_r:
    case rocsparse_datatype_f16_r:
    case rocsparse_datatype_bf16_r:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }

#define CASE(VAL, TYPE)                                                        \
    case VAL:                                                                  \
    {                                                                          \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gbsr2csr_a(handle,                \
                                                        dir,                   \
                                                        mb,                    \
                                                        nb,                    \
                                                        bsr_descr,             \
                                                        (const TYPE*)bsr_val,  \
                                                        bsr_row_ptr_indextype, \
                                                        bsr_row_ptr,           \
                                                        bsr_col_ind_indextype, \
                                                        bsr_col_ind,           \
                                                        block_dim,             \
                                                        csr_descr,             \
                                                        (TYPE*)csr_val,        \
                                                        csr_row_ptr_indextype, \
                                                        csr_row_ptr,           \
                                                        csr_col_ind_indextype, \
                                                        csr_col_ind));         \
        return rocsparse_status_success;                                       \
    }

        CASE(rocsparse_datatype_i32_r, int32_t);
        CASE(rocsparse_datatype_f32_r, float);
        CASE(rocsparse_datatype_f64_r, double);
        CASE(rocsparse_datatype_f32_c, rocsparse_float_complex);
        CASE(rocsparse_datatype_f64_c, rocsparse_double_complex);
#undef CASE
    }
    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

rocsparse_status rocsparse::spmat_bsr2csr_buffer_size(rocsparse_handle            handle,
                                                      rocsparse_const_spmat_descr source,
                                                      rocsparse_const_spmat_descr target,
                                                      size_t*                     buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    buffer_size[0] = 0;
    return rocsparse_status_success;
}

rocsparse_status rocsparse::spmat_bsr2csr(rocsparse_handle            handle,
                                          rocsparse_const_spmat_descr source,
                                          rocsparse_spmat_descr       target,
                                          size_t                      buffer_size,
                                          void*                       buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_ROCSPARSE_ERROR_IF(rocsparse_status_not_implemented,
                              source->row_type != target->row_type);
    RETURN_ROCSPARSE_ERROR_IF(rocsparse_status_not_implemented,
                              source->col_type != target->col_type);
    if(target->val_data != nullptr && source->const_val_data != nullptr)
    {
        RETURN_ROCSPARSE_ERROR_IF(rocsparse_status_not_implemented,
                                  source->data_type != target->data_type);
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gbsr2csr(handle,
                                                  source->block_dir,
                                                  source->rows,
                                                  source->cols,
                                                  source->descr,
                                                  source->data_type,
                                                  source->const_val_data,
                                                  source->row_type,
                                                  source->const_row_data,
                                                  source->col_type,
                                                  source->col_data,
                                                  source->block_dim,
                                                  target->descr,
                                                  target->data_type,
                                                  target->val_data,
                                                  target->row_type,
                                                  target->row_data,
                                                  target->col_type,
                                                  target->col_data));

    return rocsparse_status_success;
}
