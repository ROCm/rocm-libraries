/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "internal/util/rocsparse_check_matrix_csc.h"
#include "rocsparse_check_matrix_csc.hpp"
#include "rocsparse_check_matrix_csr.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_utility.hpp"

template <typename T, typename I, typename J>
rocsparse_status rocsparse::check_matrix_csc_core(rocsparse_handle       handle,
                                                  J                      m,
                                                  J                      n,
                                                  I                      nnz,
                                                  const T*               csc_val,
                                                  const I*               csc_col_ptr,
                                                  const J*               csc_row_ind,
                                                  rocsparse_index_base   idx_base,
                                                  rocsparse_matrix_type  matrix_type,
                                                  rocsparse_fill_mode    uplo,
                                                  rocsparse_storage_mode storage,
                                                  rocsparse_data_status* data_status,
                                                  void*                  temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::check_matrix_csr_core(handle,
                                                               n, // switch
                                                               m, // m and n
                                                               nnz,
                                                               csc_val,
                                                               csc_col_ptr,
                                                               csc_row_ind,
                                                               idx_base,
                                                               matrix_type,
                                                               uplo,
                                                               storage,
                                                               data_status,
                                                               temp_buffer));
    return rocsparse_status_success;
}

namespace rocsparse
{
    template <typename T, typename I, typename J>
    static rocsparse_status check_matrix_csc_quickreturn(rocsparse_handle       handle,
                                                         J                      m,
                                                         J                      n,
                                                         I                      nnz,
                                                         const T*               csc_val,
                                                         const I*               csc_col_ptr,
                                                         const J*               csc_row_ind,
                                                         rocsparse_index_base   idx_base,
                                                         rocsparse_matrix_type  matrix_type,
                                                         rocsparse_fill_mode    uplo,
                                                         rocsparse_storage_mode storage,
                                                         rocsparse_data_status* data_status,
                                                         void*                  temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        return rocsparse_status_continue;
    }
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse::check_matrix_csc_checkarg(rocsparse_handle       handle, //0
                                                      J                      m, //1
                                                      J                      n, //2
                                                      I                      nnz, //3
                                                      const T*               csc_val, //4
                                                      const I*               csc_col_ptr, //5
                                                      const J*               csc_row_ind, //6
                                                      rocsparse_index_base   idx_base, //7
                                                      rocsparse_matrix_type  matrix_type, //8
                                                      rocsparse_fill_mode    uplo, //9
                                                      rocsparse_storage_mode storage, //10
                                                      rocsparse_data_status* data_status, //11
                                                      void*                  temp_buffer) //12
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG_ARRAY(4, nnz, csc_val);
    ROCSPARSE_CHECKARG_ARRAY(5, m, csc_col_ptr);
    ROCSPARSE_CHECKARG_ARRAY(6, nnz, csc_row_ind);
    ROCSPARSE_CHECKARG_ENUM(7, idx_base);
    ROCSPARSE_CHECKARG_ENUM(8, matrix_type);
    ROCSPARSE_CHECKARG_ENUM(9, uplo);
    ROCSPARSE_CHECKARG_ENUM(10, storage);
    ROCSPARSE_CHECKARG_POINTER(11, data_status);
    ROCSPARSE_CHECKARG_POINTER(12, temp_buffer);

    if(matrix_type != rocsparse_matrix_type_general)
    {
        if(m != n)
        {
            rocsparse::log_debug(handle,
                                 ("Matrix was specified to be "
                                  + std::string(rocsparse::enum_utils::to_string(matrix_type))
                                  + " but m != n"));
        }
    }
    ROCSPARSE_CHECKARG(2,
                       n,
                       ((matrix_type != rocsparse_matrix_type_general) && (n != m)),
                       rocsparse_status_invalid_size);

    const rocsparse_status status = rocsparse::check_matrix_csc_quickreturn(handle,
                                                                            m,
                                                                            n,
                                                                            nnz,
                                                                            csc_val,
                                                                            csc_col_ptr,
                                                                            csc_row_ind,
                                                                            idx_base,
                                                                            matrix_type,
                                                                            uplo,
                                                                            storage,
                                                                            data_status,
                                                                            temp_buffer);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    return rocsparse_status_continue;
}

template <typename T, typename I, typename J, typename... P>
rocsparse_status rocsparse_check_matrix_csc_template(P&&... p)
{
    ROCSPARSE_ROUTINE_TRACE;

    const rocsparse_status status = rocsparse::check_matrix_csc_quickreturn<T, I, J>(p...);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR((rocsparse::check_matrix_csc_core<T, I, J>(p...)));
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(I, J, T)                                                 \
    template rocsparse_status rocsparse::check_matrix_csc_core<T, I, J>(     \
        rocsparse_handle       handle,                                       \
        J                      m,                                            \
        J                      n,                                            \
        I                      nnz,                                          \
        const T*               csc_val,                                      \
        const I*               csc_col_ptr,                                  \
        const J*               csc_row_ind,                                  \
        rocsparse_index_base   idx_base,                                     \
        rocsparse_matrix_type  matrix_type,                                  \
        rocsparse_fill_mode    uplo,                                         \
        rocsparse_storage_mode storage,                                      \
        rocsparse_data_status* data_status,                                  \
        void*                  temp_buffer);                                                  \
    template rocsparse_status rocsparse::check_matrix_csc_checkarg<T, I, J>( \
        rocsparse_handle       handle,                                       \
        J                      m,                                            \
        J                      n,                                            \
        I                      nnz,                                          \
        const T*               csc_val,                                      \
        const I*               csc_col_ptr,                                  \
        const J*               csc_row_ind,                                  \
        rocsparse_index_base   idx_base,                                     \
        rocsparse_matrix_type  matrix_type,                                  \
        rocsparse_fill_mode    uplo,                                         \
        rocsparse_storage_mode storage,                                      \
        rocsparse_data_status* data_status,                                  \
        void*                  temp_buffer);

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);
#undef INSTANTIATE

#define C_IMPL(NAME, T)                                                                        \
    extern "C" rocsparse_status NAME(rocsparse_handle       handle,                            \
                                     rocsparse_int          m,                                 \
                                     rocsparse_int          n,                                 \
                                     rocsparse_int          nnz,                               \
                                     const T*               csc_val,                           \
                                     const rocsparse_int*   csc_col_ptr,                       \
                                     const rocsparse_int*   csc_row_ind,                       \
                                     rocsparse_index_base   idx_base,                          \
                                     rocsparse_matrix_type  matrix_type,                       \
                                     rocsparse_fill_mode    uplo,                              \
                                     rocsparse_storage_mode storage,                           \
                                     rocsparse_data_status* data_status,                       \
                                     void*                  temp_buffer)                       \
    try                                                                                        \
    {                                                                                          \
        ROCSPARSE_ROUTINE_TRACE;                                                               \
        RETURN_IF_ROCSPARSE_ERROR(                                                             \
            (rocsparse::check_matrix_csc_impl<T, rocsparse_int, rocsparse_int>(handle,         \
                                                                               m,              \
                                                                               n,              \
                                                                               nnz,            \
                                                                               csc_val,        \
                                                                               csc_col_ptr,    \
                                                                               csc_row_ind,    \
                                                                               idx_base,       \
                                                                               matrix_type,    \
                                                                               uplo,           \
                                                                               storage,        \
                                                                               data_status,    \
                                                                               temp_buffer))); \
        return rocsparse_status_success;                                                       \
    }                                                                                          \
    catch(...)                                                                                 \
    {                                                                                          \
        RETURN_ROCSPARSE_EXCEPTION();                                                          \
    }

C_IMPL(rocsparse_scheck_matrix_csc, float);
C_IMPL(rocsparse_dcheck_matrix_csc, double);
C_IMPL(rocsparse_ccheck_matrix_csc, rocsparse_float_complex);
C_IMPL(rocsparse_zcheck_matrix_csc, rocsparse_double_complex);
#undef C_IMPL
