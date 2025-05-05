/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/generic/rocsparse_dense_to_sparse.h"
#include "control.h"
#include "handle.h"
#include "utility.h"

#include "rocsparse_dense2coo.hpp"
#include "rocsparse_dense2csx_impl.hpp"
#include "rocsparse_nnz_impl.hpp"

namespace rocsparse
{
    template <typename I, typename J, typename T>
    rocsparse_status dense_to_sparse_template(rocsparse_handle              handle,
                                              rocsparse_const_dnmat_descr   mat_A,
                                              rocsparse_spmat_descr         mat_B,
                                              rocsparse_dense_to_sparse_alg alg,
                                              size_t*                       buffer_size,
                                              void*                         temp_buffer);

    template <typename... P>
    rocsparse_status return_densetosparse(rocsparse_indextype itype,
                                          rocsparse_indextype jtype,
                                          rocsparse_datatype  ctype,
                                          P... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(itype == rocsparse_indextype_i32 && jtype == rocsparse_indextype_i32
           && ctype == rocsparse_datatype_f16_r)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::dense_to_sparse_template<int32_t, int32_t, _Float16>(p...)));
            return rocsparse_status_success;
        }
        if(itype == rocsparse_indextype_i32 && jtype == rocsparse_indextype_i32
           && ctype == rocsparse_datatype_f32_r)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::dense_to_sparse_template<int32_t, int32_t, float>(p...)));
            return rocsparse_status_success;
        }
        if(itype == rocsparse_indextype_i32 && jtype == rocsparse_indextype_i32
           && ctype == rocsparse_datatype_f64_r)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::dense_to_sparse_template<int32_t, int32_t, double>(p...)));
            return rocsparse_status_success;
        }
        if(itype == rocsparse_indextype_i32 && jtype == rocsparse_indextype_i32
           && ctype == rocsparse_datatype_f32_c)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::dense_to_sparse_template<int32_t, int32_t, rocsparse_float_complex>(
                    p...)));
            return rocsparse_status_success;
        }
        if(itype == rocsparse_indextype_i32 && jtype == rocsparse_indextype_i32
           && ctype == rocsparse_datatype_f64_c)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::dense_to_sparse_template<int32_t, int32_t, rocsparse_double_complex>(
                    p...)));
            return rocsparse_status_success;
        }
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i32
           && ctype == rocsparse_datatype_f16_r)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::dense_to_sparse_template<int64_t, int32_t, _Float16>(p...)));
            return rocsparse_status_success;
        }
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i32
           && ctype == rocsparse_datatype_f32_r)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::dense_to_sparse_template<int64_t, int32_t, float>(p...)));
            return rocsparse_status_success;
        }
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i32
           && ctype == rocsparse_datatype_f64_r)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::dense_to_sparse_template<int64_t, int32_t, double>(p...)));
            return rocsparse_status_success;
        }
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i32
           && ctype == rocsparse_datatype_f32_c)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::dense_to_sparse_template<int64_t, int32_t, rocsparse_float_complex>(
                    p...)));
            return rocsparse_status_success;
        }
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i32
           && ctype == rocsparse_datatype_f64_c)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::dense_to_sparse_template<int64_t, int32_t, rocsparse_double_complex>(
                    p...)));
            return rocsparse_status_success;
        }
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i64
           && ctype == rocsparse_datatype_f16_r)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::dense_to_sparse_template<int64_t, int64_t, _Float16>(p...)));
            return rocsparse_status_success;
        }
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i64
           && ctype == rocsparse_datatype_f32_r)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::dense_to_sparse_template<int64_t, int64_t, float>(p...)));
            return rocsparse_status_success;
        }
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i64
           && ctype == rocsparse_datatype_f64_r)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::dense_to_sparse_template<int64_t, int64_t, double>(p...)));
            return rocsparse_status_success;
        }
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i64
           && ctype == rocsparse_datatype_f32_c)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::dense_to_sparse_template<int64_t, int64_t, rocsparse_float_complex>(
                    p...)));
            return rocsparse_status_success;
        }
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i64
           && ctype == rocsparse_datatype_f64_c)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::dense_to_sparse_template<int64_t, int64_t, rocsparse_double_complex>(
                    p...)));
            return rocsparse_status_success;
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }

    template <typename I, typename J, typename T>
    rocsparse_status dense_to_sparse_template(rocsparse_handle              handle,
                                              rocsparse_const_dnmat_descr   mat_A,
                                              rocsparse_spmat_descr         mat_B,
                                              rocsparse_dense_to_sparse_alg alg,
                                              size_t*                       buffer_size,
                                              void*                         temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(temp_buffer == nullptr)
        {
            ROCSPARSE_CHECKARG_POINTER(4, buffer_size);
            if(mat_B->format == rocsparse_format_coo)
            {
                *buffer_size = sizeof(I) * mat_A->rows;
            }
            else if(mat_B->format == rocsparse_format_csr)
            {
                *buffer_size = sizeof(I) * mat_A->rows;
            }
            else if(mat_B->format == rocsparse_format_csc)
            {
                *buffer_size = sizeof(I) * mat_A->cols;
            }
            return rocsparse_status_success;
        }

        // If buffer_size is nullptr, perform analysis
        if(buffer_size == nullptr)
        {
            ROCSPARSE_CHECKARG_POINTER(5, temp_buffer);
            if(mat_B->format == rocsparse_format_coo)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::nnz_impl(handle,
                                                              rocsparse_direction_row,
                                                              mat_A->order,
                                                              (I)mat_A->rows,
                                                              (I)mat_A->cols,
                                                              mat_B->descr,
                                                              (const T*)mat_A->const_values,
                                                              mat_A->ld,
                                                              (I*)temp_buffer,
                                                              (I*)&mat_B->nnz));
                return rocsparse_status_success;
            }
            else if(mat_B->format == rocsparse_format_csr)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::nnz_impl(handle,
                                                              rocsparse_direction_row,
                                                              mat_A->order,
                                                              (J)mat_A->rows,
                                                              (J)mat_A->cols,
                                                              mat_B->descr,
                                                              (const T*)mat_A->const_values,
                                                              mat_A->ld,
                                                              (I*)temp_buffer,
                                                              (I*)&mat_B->nnz));
                return rocsparse_status_success;
            }
            else if(mat_B->format == rocsparse_format_csc)
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::nnz_impl(handle,
                                                              rocsparse_direction_column,
                                                              mat_A->order,
                                                              (J)mat_A->rows,
                                                              (J)mat_A->cols,
                                                              mat_B->descr,
                                                              (const T*)mat_A->const_values,
                                                              mat_A->ld,
                                                              (I*)temp_buffer,
                                                              (I*)&mat_B->nnz));
                return rocsparse_status_success;
            }
        }

        // COO
        if(mat_B->format == rocsparse_format_coo)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::dense2coo_template(handle,
                                                                    mat_A->order,
                                                                    (I)mat_A->rows,
                                                                    (I)mat_A->cols,
                                                                    mat_B->descr,
                                                                    (const T*)mat_A->const_values,
                                                                    mat_A->ld,
                                                                    (I*)temp_buffer,
                                                                    (T*)mat_B->val_data,
                                                                    (I*)mat_B->row_data,
                                                                    (I*)mat_B->col_data));
            return rocsparse_status_success;
        }

        // CSR
        if(mat_B->format == rocsparse_format_csr)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::dense2csx_impl<rocsparse_direction_row>(handle,
                                                                   mat_A->order,
                                                                   (J)mat_A->rows,
                                                                   (J)mat_A->cols,
                                                                   mat_B->descr,
                                                                   (const T*)mat_A->const_values,
                                                                   mat_A->ld,
                                                                   (I*)temp_buffer,
                                                                   (T*)mat_B->val_data,
                                                                   (I*)mat_B->row_data,
                                                                   (J*)mat_B->col_data));
            return rocsparse_status_success;
        }

        // CSC
        if(mat_B->format == rocsparse_format_csc)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::dense2csx_impl<rocsparse_direction_column>(handle,
                                                                      mat_A->order,
                                                                      (J)mat_A->rows,
                                                                      (J)mat_A->cols,
                                                                      mat_B->descr,
                                                                      (const T*)mat_A->const_values,
                                                                      mat_A->ld,
                                                                      (I*)temp_buffer,
                                                                      (T*)mat_B->val_data,
                                                                      (I*)mat_B->col_data,
                                                                      (J*)mat_B->row_data));
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_dense_to_sparse(rocsparse_handle              handle,
                                                      rocsparse_const_dnmat_descr   mat_A,
                                                      rocsparse_spmat_descr         mat_B,
                                                      rocsparse_dense_to_sparse_alg alg,
                                                      size_t*                       buffer_size,
                                                      void*                         temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_dense_sparse",
                         (const void*&)mat_A,
                         (const void*&)mat_B,
                         alg,
                         (const void*&)buffer_size,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    ROCSPARSE_CHECKARG_POINTER(1, mat_A);
    ROCSPARSE_CHECKARG(1, mat_A, (mat_A->init == false), rocsparse_status_not_initialized);

    ROCSPARSE_CHECKARG_POINTER(2, mat_B);
    ROCSPARSE_CHECKARG(2, mat_B, (mat_B->init == false), rocsparse_status_not_initialized);

    ROCSPARSE_CHECKARG_ENUM(3, alg);
    ROCSPARSE_CHECKARG(4,
                       buffer_size,
                       (buffer_size == nullptr && temp_buffer == nullptr),
                       rocsparse_status_invalid_pointer);

    if(mat_B->format == rocsparse_format_csc)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::return_densetosparse(mat_B->col_type,
                                                                  mat_B->row_type,
                                                                  mat_B->data_type,
                                                                  handle,
                                                                  mat_A,
                                                                  mat_B,
                                                                  alg,
                                                                  buffer_size,
                                                                  temp_buffer));

        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::return_densetosparse(mat_B->row_type,
                                                              mat_B->col_type,
                                                              mat_B->data_type,
                                                              handle,
                                                              mat_A,
                                                              mat_B,
                                                              alg,
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
