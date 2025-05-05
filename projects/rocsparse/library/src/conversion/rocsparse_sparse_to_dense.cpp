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

#include "internal/generic/rocsparse_sparse_to_dense.h"
#include "control.h"
#include "handle.h"
#include "utility.h"

#include "rocsparse_coo2dense.hpp"
#include "rocsparse_csx2dense_impl.hpp"

namespace rocsparse
{
    template <typename I, typename J, typename T>
    rocsparse_status sparse_to_dense_template(rocsparse_handle              handle,
                                              rocsparse_const_spmat_descr   mat_A,
                                              rocsparse_dnmat_descr         mat_B,
                                              rocsparse_sparse_to_dense_alg alg,
                                              size_t*                       buffer_size,
                                              void*                         temp_buffer);

    template <typename... P>
    rocsparse_status return_sparsetodense(rocsparse_indextype itype,
                                          rocsparse_indextype jtype,
                                          rocsparse_datatype  ctype,
                                          P... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(itype == rocsparse_indextype_i32 && jtype == rocsparse_indextype_i32
           && ctype == rocsparse_datatype_f16_r)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::sparse_to_dense_template<int32_t, int32_t, _Float16>(p...)));
            return rocsparse_status_success;
        }
        if(itype == rocsparse_indextype_i32 && jtype == rocsparse_indextype_i32
           && ctype == rocsparse_datatype_f32_r)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::sparse_to_dense_template<int32_t, int32_t, float>(p...)));
            return rocsparse_status_success;
        }
        if(itype == rocsparse_indextype_i32 && jtype == rocsparse_indextype_i32
           && ctype == rocsparse_datatype_f64_r)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::sparse_to_dense_template<int32_t, int32_t, double>(p...)));
            return rocsparse_status_success;
        }
        if(itype == rocsparse_indextype_i32 && jtype == rocsparse_indextype_i32
           && ctype == rocsparse_datatype_f32_c)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::sparse_to_dense_template<int32_t, int32_t, rocsparse_float_complex>(
                    p...)));
            return rocsparse_status_success;
        }
        if(itype == rocsparse_indextype_i32 && jtype == rocsparse_indextype_i32
           && ctype == rocsparse_datatype_f64_c)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::sparse_to_dense_template<int32_t, int32_t, rocsparse_double_complex>(
                    p...)));
            return rocsparse_status_success;
        }
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i32
           && ctype == rocsparse_datatype_f16_r)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::sparse_to_dense_template<int64_t, int32_t, _Float16>(p...)));
            return rocsparse_status_success;
        }
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i32
           && ctype == rocsparse_datatype_f32_r)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::sparse_to_dense_template<int64_t, int32_t, float>(p...)));
            return rocsparse_status_success;
        }
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i32
           && ctype == rocsparse_datatype_f64_r)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::sparse_to_dense_template<int64_t, int32_t, double>(p...)));
            return rocsparse_status_success;
        }
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i32
           && ctype == rocsparse_datatype_f32_c)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::sparse_to_dense_template<int64_t, int32_t, rocsparse_float_complex>(
                    p...)));
            return rocsparse_status_success;
        }
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i32
           && ctype == rocsparse_datatype_f64_c)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::sparse_to_dense_template<int64_t, int32_t, rocsparse_double_complex>(
                    p...)));
            return rocsparse_status_success;
        }
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i64
           && ctype == rocsparse_datatype_f16_r)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::sparse_to_dense_template<int64_t, int64_t, _Float16>(p...)));
            return rocsparse_status_success;
        }
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i64
           && ctype == rocsparse_datatype_f32_r)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::sparse_to_dense_template<int64_t, int64_t, float>(p...)));
            return rocsparse_status_success;
        }
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i64
           && ctype == rocsparse_datatype_f64_r)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::sparse_to_dense_template<int64_t, int64_t, double>(p...)));
            return rocsparse_status_success;
        }
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i64
           && ctype == rocsparse_datatype_f32_c)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::sparse_to_dense_template<int64_t, int64_t, rocsparse_float_complex>(
                    p...)));
            return rocsparse_status_success;
        }
        if(itype == rocsparse_indextype_i64 && jtype == rocsparse_indextype_i64
           && ctype == rocsparse_datatype_f64_c)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::sparse_to_dense_template<int64_t, int64_t, rocsparse_double_complex>(
                    p...)));
            return rocsparse_status_success;
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    }

    template <typename I, typename J, typename T>
    rocsparse_status sparse_to_dense_template(rocsparse_handle              handle,
                                              rocsparse_const_spmat_descr   mat_A,
                                              rocsparse_dnmat_descr         mat_B,
                                              rocsparse_sparse_to_dense_alg alg,
                                              size_t*                       buffer_size,
                                              void*                         temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // If temp_buffer is nullptr, return buffer_size
        if(temp_buffer == nullptr)
        {
            // We do not need a buffer
            *buffer_size = 4;

            return rocsparse_status_success;
        }

        // COO
        if(mat_A->format == rocsparse_format_coo)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::coo2dense_template(handle,
                                                                    (I)mat_A->rows,
                                                                    (I)mat_A->cols,
                                                                    mat_A->nnz,
                                                                    mat_A->descr,
                                                                    (const T*)mat_A->const_val_data,
                                                                    (const I*)mat_A->const_row_data,
                                                                    (const I*)mat_A->const_col_data,
                                                                    (T*)mat_B->values,
                                                                    mat_B->ld,
                                                                    mat_B->order));
            return rocsparse_status_success;
        }

        // CSR
        if(mat_A->format == rocsparse_format_csr)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::csx2dense_impl<rocsparse_direction_row>(handle,
                                                                   (J)mat_A->rows,
                                                                   (J)mat_A->cols,
                                                                   mat_A->descr,
                                                                   (const T*)mat_A->const_val_data,
                                                                   (const I*)mat_A->const_row_data,
                                                                   (const J*)mat_A->const_col_data,
                                                                   (T*)mat_B->values,
                                                                   mat_B->ld,
                                                                   mat_B->order));
            return rocsparse_status_success;
        }

        // CSC
        if(mat_A->format == rocsparse_format_csc)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::csx2dense_impl<rocsparse_direction_column>(
                handle,
                (J)mat_A->rows,
                (J)mat_A->cols,
                mat_A->descr,
                (const T*)mat_A->const_val_data,
                (const I*)mat_A->const_col_data,
                (const J*)mat_A->const_row_data,
                (T*)mat_B->values,
                mat_B->ld,
                mat_B->order));
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

extern "C" rocsparse_status rocsparse_sparse_to_dense(rocsparse_handle              handle,
                                                      rocsparse_const_spmat_descr   mat_A,
                                                      rocsparse_dnmat_descr         mat_B,
                                                      rocsparse_sparse_to_dense_alg alg,
                                                      size_t*                       buffer_size,
                                                      void*                         temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_sparse_dense",
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
                       (temp_buffer == nullptr && buffer_size == nullptr),
                       rocsparse_status_invalid_pointer);

    if(mat_A->format == rocsparse_format_csc)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::return_sparsetodense(mat_A->col_type,
                                                                  mat_A->row_type,
                                                                  mat_A->data_type,
                                                                  handle,
                                                                  mat_A,
                                                                  mat_B,
                                                                  alg,
                                                                  buffer_size,
                                                                  temp_buffer));
        return rocsparse_status_success;
    }
    else
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::return_sparsetodense(mat_A->row_type,
                                                                  mat_A->col_type,
                                                                  mat_A->data_type,
                                                                  handle,
                                                                  mat_A,
                                                                  mat_B,
                                                                  alg,
                                                                  buffer_size,
                                                                  temp_buffer));
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
