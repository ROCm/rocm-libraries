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

#include "rocsparse_gell2csr.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_ell2csr.hpp"
#include "rocsparse_handle.hpp"

rocsparse_status rocsparse::gell2csr_nnz(rocsparse_handle          handle,
                                         int64_t                   m,
                                         int64_t                   n,
                                         const rocsparse_mat_descr ell_descr,
                                         int64_t                   ell_width,
                                         rocsparse_indextype       ell_col_ind_indextype,
                                         const void*               ell_col_ind,
                                         const rocsparse_mat_descr csr_descr,
                                         rocsparse_indextype       csr_row_ptr_indextype,
                                         void*                     csr_row_ptr,
                                         int64_t*                  csr_nnz)
{
    ROCSPARSE_ROUTINE_TRACE;

    switch(ell_col_ind_indextype)
    {
    case rocsparse_indextype_i32:
    {
        if(m > std::numeric_limits<int32_t>::max())
        {
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            // LCOV_EXCL_STOP
        }
        if(n > std::numeric_limits<int32_t>::max())
        {
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            // LCOV_EXCL_STOP
        }
        if(ell_width > std::numeric_limits<int32_t>::max())
        {
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            // LCOV_EXCL_STOP
        }
        int32_t m32         = m;
        int32_t n32         = n;
        int32_t ell_width32 = ell_width;

        switch(csr_row_ptr_indextype)
        {
        case rocsparse_indextype_i32:
        {
            int32_t csr_nnz_32;
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::ell2csr_nnz_template(handle,
                                                                      m32,
                                                                      n32,
                                                                      ell_descr,
                                                                      ell_width32,
                                                                      (const int32_t*)ell_col_ind,
                                                                      csr_descr,
                                                                      (int32_t*)csr_row_ptr,
                                                                      &csr_nnz_32));
            csr_nnz[0] = csr_nnz_32;
            return rocsparse_status_success;
        }
        case rocsparse_indextype_i64:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::ell2csr_nnz_template(handle,
                                                                      m32,
                                                                      n32,
                                                                      ell_descr,
                                                                      ell_width32,
                                                                      (const int32_t*)ell_col_ind,
                                                                      csr_descr,
                                                                      (int64_t*)csr_row_ptr,
                                                                      (int64_t*)csr_nnz));
            return rocsparse_status_success;
        }

        case rocsparse_indextype_u16:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        }
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    case rocsparse_indextype_i64:
    {
        switch(csr_row_ptr_indextype)
        {
        case rocsparse_indextype_i32:
        {
            int32_t csr_nnz_32;
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::ell2csr_nnz_template(handle,
                                                                      m,
                                                                      n,
                                                                      ell_descr,
                                                                      ell_width,
                                                                      (const int64_t*)ell_col_ind,
                                                                      csr_descr,
                                                                      (int32_t*)csr_row_ptr,
                                                                      &csr_nnz_32));
            csr_nnz[0] = csr_nnz_32;
            return rocsparse_status_success;
        }
        case rocsparse_indextype_i64:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::ell2csr_nnz_template(handle,
                                                                      m,
                                                                      n,
                                                                      ell_descr,
                                                                      ell_width,
                                                                      (const int64_t*)ell_col_ind,
                                                                      csr_descr,
                                                                      (int64_t*)csr_row_ptr,
                                                                      (int64_t*)csr_nnz));
            return rocsparse_status_success;
        }

        case rocsparse_indextype_u16:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        }
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    case rocsparse_indextype_u16:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }
    }
}

rocsparse_status rocsparse::gell2csr(rocsparse_handle          handle,
                                     int64_t                   m,
                                     int64_t                   n,
                                     const rocsparse_mat_descr ell_descr,
                                     int64_t                   ell_width,
                                     rocsparse_datatype        ell_val_datatype,
                                     const void*               ell_val,
                                     rocsparse_indextype       ell_col_ind_indextype,
                                     const void*               ell_col_ind,
                                     const rocsparse_mat_descr csr_descr,
                                     rocsparse_datatype        csr_val_datatype,
                                     void*                     csr_val,
                                     rocsparse_indextype       csr_row_ptr_indextype,
                                     const void*               csr_row_ptr,
                                     rocsparse_indextype       csr_col_ind_indextype,
                                     void*                     csr_col_ind)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(ell_val_datatype != csr_val_datatype)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }
    if(csr_col_ind_indextype != ell_col_ind_indextype)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }

#define CALL(T, I, J)                                                            \
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::ell2csr_template(handle,                \
                                                          (J)m,                  \
                                                          (J)n,                  \
                                                          ell_descr,             \
                                                          (J)ell_width,          \
                                                          (const T*)ell_val,     \
                                                          (const J*)ell_col_ind, \
                                                          csr_descr,             \
                                                          (T*)csr_val,           \
                                                          (const I*)csr_row_ptr, \
                                                          (J*)csr_col_ind));     \
    return rocsparse_status_success

    switch(ell_val_datatype)
    {
    case rocsparse_datatype_i32_r:
    case rocsparse_datatype_u32_r:
    case rocsparse_datatype_i8_r:
    case rocsparse_datatype_u8_r:
    case rocsparse_datatype_f16_r:
    case rocsparse_datatype_bf16_r:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }
    case rocsparse_datatype_f32_r:
    {
        switch(csr_row_ptr_indextype)
        {
        case rocsparse_indextype_i32:
        {
            switch(csr_col_ind_indextype)
            {
            case rocsparse_indextype_i32:
            {
                CALL(float, int32_t, int32_t);
            }
            case rocsparse_indextype_i64:
            {
                CALL(float, int32_t, int64_t);
            }
            case rocsparse_indextype_u16:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            // LCOV_EXCL_STOP
        }
        case rocsparse_indextype_i64:
        {
            switch(csr_col_ind_indextype)
            {
            case rocsparse_indextype_i32:
            {
                CALL(float, int64_t, int32_t);
            }
            case rocsparse_indextype_i64:
            {
                CALL(float, int64_t, int64_t);
            }
            case rocsparse_indextype_u16:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            // LCOV_EXCL_STOP
        }
        case rocsparse_indextype_u16:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        }
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    case rocsparse_datatype_f32_c:
    {
        switch(csr_row_ptr_indextype)
        {
        case rocsparse_indextype_i32:
        {
            switch(csr_col_ind_indextype)
            {
            case rocsparse_indextype_i32:
            {
                CALL(rocsparse_float_complex, int32_t, int32_t);
            }
            case rocsparse_indextype_i64:
            {
                CALL(rocsparse_float_complex, int32_t, int64_t);
            }
            case rocsparse_indextype_u16:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            // LCOV_EXCL_STOP
        }
        case rocsparse_indextype_i64:
        {
            switch(csr_col_ind_indextype)
            {
            case rocsparse_indextype_i32:
            {
                CALL(rocsparse_float_complex, int64_t, int32_t);
            }
            case rocsparse_indextype_i64:
            {
                CALL(rocsparse_float_complex, int64_t, int64_t);
            }
            case rocsparse_indextype_u16:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            // LCOV_EXCL_STOP
        }
        case rocsparse_indextype_u16:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        }
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    case rocsparse_datatype_f64_r:
    {
        switch(csr_row_ptr_indextype)
        {
        case rocsparse_indextype_i32:
        {
            switch(csr_col_ind_indextype)
            {
            case rocsparse_indextype_i32:
            {
                CALL(double, int32_t, int32_t);
            }
            case rocsparse_indextype_i64:
            {
                CALL(double, int32_t, int64_t);
            }
            case rocsparse_indextype_u16:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            // LCOV_EXCL_STOP
        }
        case rocsparse_indextype_i64:
        {
            switch(csr_col_ind_indextype)
            {
            case rocsparse_indextype_i32:
            {
                CALL(double, int64_t, int32_t);
            }
            case rocsparse_indextype_i64:
            {
                CALL(double, int64_t, int64_t);
            }
            case rocsparse_indextype_u16:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            // LCOV_EXCL_STOP
        }
        case rocsparse_indextype_u16:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        }
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }

    case rocsparse_datatype_f64_c:
    {
        switch(csr_row_ptr_indextype)
        {
        case rocsparse_indextype_i32:
        {
            switch(csr_col_ind_indextype)
            {
            case rocsparse_indextype_i32:
            {
                CALL(rocsparse_double_complex, int32_t, int32_t);
            }
            case rocsparse_indextype_i64:
            {
                CALL(rocsparse_double_complex, int32_t, int64_t);
            }
            case rocsparse_indextype_u16:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            // LCOV_EXCL_STOP
        }
        case rocsparse_indextype_i64:
        {
            switch(csr_col_ind_indextype)
            {
            case rocsparse_indextype_i32:
            {
                CALL(rocsparse_double_complex, int64_t, int32_t);
            }
            case rocsparse_indextype_i64:
            {
                CALL(rocsparse_double_complex, int64_t, int64_t);
            }
            case rocsparse_indextype_u16:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            // LCOV_EXCL_STOP
        }
        case rocsparse_indextype_u16:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        }
        // LCOV_EXCL_START
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
        // LCOV_EXCL_STOP
    }
    }
    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

rocsparse_status rocsparse::spmat_ell2csr_nnz(rocsparse_handle            handle,
                                              rocsparse_const_spmat_descr source,
                                              rocsparse_const_spmat_descr target,
                                              int64_t*                    out_csr_nnz)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gell2csr_nnz(handle,
                                                      source->rows,
                                                      source->cols,
                                                      source->descr,
                                                      source->ell_width,
                                                      source->col_type,
                                                      source->const_col_data,
                                                      target->descr,
                                                      target->row_type,
                                                      target->row_data,
                                                      out_csr_nnz));

    return rocsparse_status_success;
}

rocsparse_status rocsparse::spmat_ell2csr_buffer_size(rocsparse_handle            handle,
                                                      rocsparse_const_spmat_descr source,
                                                      rocsparse_const_spmat_descr target,
                                                      size_t*                     buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    buffer_size[0] = 0;
    return rocsparse_status_success;
}

rocsparse_status rocsparse::spmat_ell2csr(rocsparse_handle            handle,
                                          rocsparse_const_spmat_descr source,
                                          rocsparse_spmat_descr       target,
                                          size_t                      buffer_size,
                                          void*                       buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gell2csr(handle,
                                                  source->rows,
                                                  source->cols,
                                                  source->descr,
                                                  source->ell_width,
                                                  source->data_type,
                                                  source->const_val_data,
                                                  source->col_type,
                                                  source->const_col_data,
                                                  target->descr,
                                                  target->data_type,
                                                  target->val_data,
                                                  target->row_type,
                                                  target->row_data,
                                                  target->col_type,
                                                  target->col_data));

    return rocsparse_status_success;
}
