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
#include "rocsparse_control.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_dense2coo.hpp"
#include "rocsparse_dense2csx_impl.hpp"
#include "rocsparse_nnz_impl.hpp"

#include <map>

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_dense_to_sparse_alg value_)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value_)
    {
        CASE(rocsparse_dense_to_sparse_alg_default);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_dense_to_sparse_alg value_)
{
    switch(value_)
    {
    case rocsparse_dense_to_sparse_alg_default:
    {
        return false;
    }
    }
    return true;
}

namespace rocsparse
{
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

    typedef rocsparse_status (*dense_to_sparse_t)(rocsparse_handle              handle,
                                                  rocsparse_const_dnmat_descr   mat_A,
                                                  rocsparse_spmat_descr         mat_B,
                                                  rocsparse_dense_to_sparse_alg alg,
                                                  size_t*                       buffer_size,
                                                  void*                         temp_buffer);

    using dense_to_sparse_tuple
        = std::tuple<rocsparse_indextype, rocsparse_indextype, rocsparse_datatype>;

    // clang-format off
#define DENSE_TO_SPARSE_CONFIG(I, J, T)                                        \
    {dense_to_sparse_tuple(I, J, T),                                           \
     dense_to_sparse_template<typename rocsparse::indextype_traits<I>::type_t, \
                              typename rocsparse::indextype_traits<J>::type_t, \
                              typename rocsparse::datatype_traits<T>::type_t>}
    // clang-format on

    static const std::map<dense_to_sparse_tuple, dense_to_sparse_t> s_dense_to_sparse_dispatch{
        {DENSE_TO_SPARSE_CONFIG(
             rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_datatype_f16_r),
         DENSE_TO_SPARSE_CONFIG(
             rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_datatype_bf16_r),
         DENSE_TO_SPARSE_CONFIG(
             rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_datatype_f32_r),
         DENSE_TO_SPARSE_CONFIG(
             rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_datatype_f64_r),
         DENSE_TO_SPARSE_CONFIG(
             rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_datatype_f32_c),
         DENSE_TO_SPARSE_CONFIG(
             rocsparse_indextype_i32, rocsparse_indextype_i32, rocsparse_datatype_f64_c),

         DENSE_TO_SPARSE_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i32, rocsparse_datatype_f16_r),
         DENSE_TO_SPARSE_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i32, rocsparse_datatype_bf16_r),
         DENSE_TO_SPARSE_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i32, rocsparse_datatype_f32_r),
         DENSE_TO_SPARSE_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i32, rocsparse_datatype_f64_r),
         DENSE_TO_SPARSE_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i32, rocsparse_datatype_f32_c),
         DENSE_TO_SPARSE_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i32, rocsparse_datatype_f64_c),

         DENSE_TO_SPARSE_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i64, rocsparse_datatype_f16_r),
         DENSE_TO_SPARSE_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i64, rocsparse_datatype_bf16_r),
         DENSE_TO_SPARSE_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i64, rocsparse_datatype_f32_r),
         DENSE_TO_SPARSE_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i64, rocsparse_datatype_f64_r),
         DENSE_TO_SPARSE_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i64, rocsparse_datatype_f32_c),
         DENSE_TO_SPARSE_CONFIG(
             rocsparse_indextype_i64, rocsparse_indextype_i64, rocsparse_datatype_f64_c)}};

    static rocsparse_status dense_to_sparse_find(dense_to_sparse_t*  function_,
                                                 rocsparse_indextype i_type_,
                                                 rocsparse_indextype j_type_,
                                                 rocsparse_datatype  t_type_)
    {
        const auto& it = rocsparse::s_dense_to_sparse_dispatch.find(
            rocsparse::dense_to_sparse_tuple(i_type_, j_type_, t_type_));

        if(it != rocsparse::s_dense_to_sparse_dispatch.end())
        {
            function_[0] = it->second;
        }
        // LCOV_EXCL_START
        else
        {
#ifndef NDEBUG
            std::cout << "invalid precision configuration: "
                      << ", i_type: " << rocsparse::enum_utils::to_string(i_type_) << std::endl
                      << ", j_type: " << rocsparse::enum_utils::to_string(j_type_) << std::endl
                      << ", t_type: " << rocsparse::enum_utils::to_string(t_type_) << std::endl;

            std::cout << "available configuration are: " << std::endl;
            for(const auto& p : rocsparse::s_dense_to_sparse_dispatch)
            {
                const auto& t      = p.first;
                const auto  i_type = std::get<0>(t);
                const auto  j_type = std::get<1>(t);
                const auto  t_type = std::get<2>(t);
                std::cout << std::endl
                          << std::endl
                          << ", i_type: " << rocsparse::enum_utils::to_string(i_type) << std::endl
                          << ", j_type: " << rocsparse::enum_utils::to_string(j_type) << std::endl
                          << ", t_type: " << rocsparse::enum_utils::to_string(t_type) << std::endl;
            }
#endif

            std::stringstream sstr;
            sstr << "invalid precision configuration: "
                 << ", i_type: " << rocsparse::enum_utils::to_string(i_type_)
                 << ", j_type: " << rocsparse::enum_utils::to_string(j_type_)
                 << ", t_type: " << rocsparse::enum_utils::to_string(t_type_);

            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   sstr.str().c_str());
        }
        // LCOV_EXCL_STOP

        return rocsparse_status_success;
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

    rocsparse::dense_to_sparse_t f;
    if(mat_B->format == rocsparse_format_csc)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::dense_to_sparse_find(
            &f, mat_B->col_type, mat_B->row_type, mat_B->data_type));
    }
    else
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::dense_to_sparse_find(
            &f, mat_B->row_type, mat_B->col_type, mat_B->data_type));
    }

    RETURN_IF_ROCSPARSE_ERROR(f(handle, mat_A, mat_B, alg, buffer_size, temp_buffer));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
