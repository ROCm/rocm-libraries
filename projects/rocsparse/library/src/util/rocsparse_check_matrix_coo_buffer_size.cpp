/*! \file */
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
#include "internal/util/rocsparse_check_matrix_coo.h"

#include "rocsparse_check_matrix_coo.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_utility.hpp"

#include "check_matrix_coo_device.h"

template <typename T, typename I>
rocsparse_status rocsparse::check_matrix_coo_buffer_size_core(rocsparse_handle       handle,
                                                              I                      m,
                                                              I                      n,
                                                              int64_t                nnz,
                                                              const T*               coo_val,
                                                              const I*               coo_row_ind,
                                                              const I*               coo_col_ind,
                                                              rocsparse_index_base   idx_base,
                                                              rocsparse_matrix_type  matrix_type,
                                                              rocsparse_fill_mode    uplo,
                                                              rocsparse_storage_mode storage,
                                                              size_t*                buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(nnz == 0)
    {
        buffer_size[0] = 0;
    }
    else
    {
        buffer_size[0] = ((sizeof(rocsparse_data_status) - 1) / 256 + 1) * 256;
    }
    return rocsparse_status_success;
}

namespace rocsparse
{
    template <typename T, typename I>
    static rocsparse_status
        check_matrix_coo_buffer_size_quickreturn(rocsparse_handle       handle,
                                                 I                      m,
                                                 I                      n,
                                                 int64_t                nnz,
                                                 const T*               coo_val,
                                                 const I*               coo_row_ind,
                                                 const I*               coo_col_ind,
                                                 rocsparse_index_base   idx_base,
                                                 rocsparse_matrix_type  matrix_type,
                                                 rocsparse_fill_mode    uplo,
                                                 rocsparse_storage_mode storage,
                                                 size_t*                buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        return rocsparse_status_continue;
    }
}

template <typename T, typename I>
rocsparse_status
    rocsparse::check_matrix_coo_buffer_size_checkarg(rocsparse_handle       handle, //0
                                                     I                      m, //1
                                                     I                      n, //2
                                                     int64_t                nnz, //3
                                                     const T*               coo_val, //4
                                                     const I*               coo_row_ind, //5
                                                     const I*               coo_col_ind, //6
                                                     rocsparse_index_base   idx_base, //7
                                                     rocsparse_matrix_type  matrix_type, //8
                                                     rocsparse_fill_mode    uplo, //9
                                                     rocsparse_storage_mode storage, //10
                                                     size_t*                buffer_size) //11
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG_ARRAY(4, nnz, coo_val);
    ROCSPARSE_CHECKARG_ARRAY(5, nnz, coo_row_ind);
    ROCSPARSE_CHECKARG_ARRAY(6, nnz, coo_col_ind);
    ROCSPARSE_CHECKARG_ENUM(7, idx_base);
    ROCSPARSE_CHECKARG_ENUM(8, matrix_type);
    ROCSPARSE_CHECKARG_ENUM(9, uplo);
    ROCSPARSE_CHECKARG_ENUM(10, storage);
    ROCSPARSE_CHECKARG_POINTER(11, buffer_size);

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

    const rocsparse_status status
        = rocsparse::check_matrix_coo_buffer_size_quickreturn(handle,
                                                              m,
                                                              n,
                                                              nnz,
                                                              coo_val,
                                                              coo_row_ind,
                                                              coo_col_ind,
                                                              idx_base,
                                                              matrix_type,
                                                              uplo,
                                                              storage,
                                                              buffer_size);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    return rocsparse_status_continue;
}

#define INSTANTIATE(I, T)                                                             \
    template rocsparse_status rocsparse::check_matrix_coo_buffer_size_core<T, I>(     \
        rocsparse_handle       handle,                                                \
        I                      m,                                                     \
        I                      n,                                                     \
        int64_t                nnz,                                                   \
        const T*               coo_val,                                               \
        const I*               coo_row_ind,                                           \
        const I*               coo_col_ind,                                           \
        rocsparse_index_base   idx_base,                                              \
        rocsparse_matrix_type  matrix_type,                                           \
        rocsparse_fill_mode    uplo,                                                  \
        rocsparse_storage_mode storage,                                               \
        size_t*                buffer_size);                                                         \
    template rocsparse_status rocsparse::check_matrix_coo_buffer_size_checkarg<T, I>( \
        rocsparse_handle       handle,                                                \
        I                      m,                                                     \
        I                      n,                                                     \
        int64_t                nnz,                                                   \
        const T*               coo_val,                                               \
        const I*               coo_row_ind,                                           \
        const I*               coo_col_ind,                                           \
        rocsparse_index_base   idx_base,                                              \
        rocsparse_matrix_type  matrix_type,                                           \
        rocsparse_fill_mode    uplo,                                                  \
        rocsparse_storage_mode storage,                                               \
        size_t*                buffer_size);

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
#undef INSTANTIATE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
#define C_IMPL(NAME, T)                                                                     \
    extern "C" rocsparse_status NAME(rocsparse_handle       handle,                         \
                                     rocsparse_int          m,                              \
                                     rocsparse_int          n,                              \
                                     rocsparse_int          nnz,                            \
                                     const T*               coo_val,                        \
                                     const rocsparse_int*   coo_row_ind,                    \
                                     const rocsparse_int*   coo_col_ind,                    \
                                     rocsparse_index_base   idx_base,                       \
                                     rocsparse_matrix_type  matrix_type,                    \
                                     rocsparse_fill_mode    uplo,                           \
                                     rocsparse_storage_mode storage,                        \
                                     size_t*                buffer_size)                    \
    try                                                                                     \
    {                                                                                       \
        ROCSPARSE_ROUTINE_TRACE;                                                            \
        RETURN_IF_ROCSPARSE_ERROR(                                                          \
            (rocsparse::check_matrix_coo_buffer_size_impl<T, rocsparse_int>(handle,         \
                                                                            m,              \
                                                                            n,              \
                                                                            nnz,            \
                                                                            coo_val,        \
                                                                            coo_row_ind,    \
                                                                            coo_col_ind,    \
                                                                            idx_base,       \
                                                                            matrix_type,    \
                                                                            uplo,           \
                                                                            storage,        \
                                                                            buffer_size))); \
        return rocsparse_status_success;                                                    \
    }                                                                                       \
    catch(...)                                                                              \
    {                                                                                       \
        RETURN_ROCSPARSE_EXCEPTION();                                                       \
    }

C_IMPL(rocsparse_scheck_matrix_coo_buffer_size, float);
C_IMPL(rocsparse_dcheck_matrix_coo_buffer_size, double);
C_IMPL(rocsparse_ccheck_matrix_coo_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zcheck_matrix_coo_buffer_size, rocsparse_double_complex);
#undef C_IMPL
