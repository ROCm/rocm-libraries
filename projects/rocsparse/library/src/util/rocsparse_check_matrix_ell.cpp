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
#include "internal/util/rocsparse_check_matrix_ell.h"
#include "rocsparse_check_matrix_ell.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_utility.hpp"

#include "check_matrix_ell_device.h"

template <typename T, typename I>
rocsparse_status rocsparse::check_matrix_ell_core(rocsparse_handle       handle,
                                                  I                      m,
                                                  I                      n,
                                                  I                      ell_width,
                                                  const T*               ell_val,
                                                  const I*               ell_col_ind,
                                                  rocsparse_index_base   idx_base,
                                                  rocsparse_matrix_type  matrix_type,
                                                  rocsparse_fill_mode    uplo,
                                                  rocsparse_storage_mode storage,
                                                  rocsparse_data_status* data_status,
                                                  void*                  temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    *data_status = rocsparse_data_status_success;

    // Temporary buffer entry points
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    rocsparse_data_status* d_data_status = reinterpret_cast<rocsparse_data_status*>(ptr);
    ptr += ((sizeof(rocsparse_data_status) - 1) / 256 + 1) * 256;

    RETURN_IF_HIP_ERROR(hipMemsetAsync(d_data_status, 0, sizeof(rocsparse_data_status)));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::check_matrix_ell_device<256>),
                                       dim3((m - 1) / 256 + 1),
                                       dim3(256),
                                       0,
                                       handle->stream,
                                       m,
                                       n,
                                       ell_width,
                                       ell_val,
                                       ell_col_ind,
                                       idx_base,
                                       matrix_type,
                                       uplo,
                                       storage,
                                       d_data_status);

    RETURN_IF_HIP_ERROR(hipMemcpyAsync(data_status,
                                       d_data_status,
                                       sizeof(rocsparse_data_status),
                                       hipMemcpyDeviceToHost,
                                       handle->stream));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

    if(*data_status != rocsparse_data_status_success)
    {
        rocsparse::log_debug(handle, rocsparse::enum_utils::to_string(*data_status));
    }

    return rocsparse_status_success;
}

namespace rocsparse
{
    template <typename T, typename I>
    static rocsparse_status check_matrix_ell_quickreturn(rocsparse_handle       handle,
                                                         I                      m,
                                                         I                      n,
                                                         I                      ell_width,
                                                         const T*               ell_val,
                                                         const I*               ell_col_ind,
                                                         rocsparse_index_base   idx_base,
                                                         rocsparse_matrix_type  matrix_type,
                                                         rocsparse_fill_mode    uplo,
                                                         rocsparse_storage_mode storage,
                                                         rocsparse_data_status* data_status,
                                                         void*                  temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(m * ell_width == 0)
        {
            // clear output status to success
            *data_status = rocsparse_data_status_success;
            return rocsparse_status_success;
        }

        return rocsparse_status_continue;
    }
}

template <typename T, typename I>
rocsparse_status rocsparse::check_matrix_ell_checkarg(rocsparse_handle       handle, //0
                                                      I                      m, //1
                                                      I                      n, //2
                                                      I                      ell_width, //3
                                                      const T*               ell_val, //4
                                                      const I*               ell_col_ind, //5
                                                      rocsparse_index_base   idx_base, //6
                                                      rocsparse_matrix_type  matrix_type, //7
                                                      rocsparse_fill_mode    uplo, //8
                                                      rocsparse_storage_mode storage, //9
                                                      rocsparse_data_status* data_status, //10
                                                      void*                  temp_buffer) //11
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_SIZE(3, ell_width);
    ROCSPARSE_CHECKARG_ARRAY(4, (m * ell_width), ell_val);
    ROCSPARSE_CHECKARG_ARRAY(5, (m * ell_width), ell_col_ind);
    ROCSPARSE_CHECKARG_ENUM(6, idx_base);
    ROCSPARSE_CHECKARG_ENUM(7, matrix_type);
    ROCSPARSE_CHECKARG(7,
                       matrix_type,
                       (matrix_type != rocsparse_matrix_type_general),
                       rocsparse_status_invalid_value);
    ROCSPARSE_CHECKARG_ENUM(8, uplo);
    ROCSPARSE_CHECKARG_ENUM(9, storage);
    ROCSPARSE_CHECKARG_POINTER(10, data_status);
    ROCSPARSE_CHECKARG_POINTER(11, temp_buffer);

    const rocsparse_status status = rocsparse::check_matrix_ell_quickreturn(handle,
                                                                            m,
                                                                            n,
                                                                            ell_width,
                                                                            ell_val,
                                                                            ell_col_ind,
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

#define INSTANTIATE(I, T)                                                 \
    template rocsparse_status rocsparse::check_matrix_ell_core<T, I>(     \
        rocsparse_handle       handle,                                    \
        I                      m,                                         \
        I                      n,                                         \
        I                      ell_width,                                 \
        const T*               ell_val,                                   \
        const I*               ell_col_ind,                               \
        rocsparse_index_base   idx_base,                                  \
        rocsparse_matrix_type  matrix_type,                               \
        rocsparse_fill_mode    uplo,                                      \
        rocsparse_storage_mode storage,                                   \
        rocsparse_data_status* data_status,                               \
        void*                  temp_buffer);                                               \
                                                                          \
    template rocsparse_status rocsparse::check_matrix_ell_checkarg<T, I>( \
        rocsparse_handle       handle,                                    \
        I                      m,                                         \
        I                      n,                                         \
        I                      ell_width,                                 \
        const T*               ell_val,                                   \
        const I*               ell_col_ind,                               \
        rocsparse_index_base   idx_base,                                  \
        rocsparse_matrix_type  matrix_type,                               \
        rocsparse_fill_mode    uplo,                                      \
        rocsparse_storage_mode storage,                                   \
        rocsparse_data_status* data_status,                               \
        void*                  temp_buffer);

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
#undef INSTANTIATE

#define C_IMPL(NAME, T)                                                         \
    extern "C" rocsparse_status NAME(rocsparse_handle       handle,             \
                                     rocsparse_int          m,                  \
                                     rocsparse_int          n,                  \
                                     rocsparse_int          ell_width,          \
                                     const T*               ell_val,            \
                                     const rocsparse_int*   ell_col_ind,        \
                                     rocsparse_index_base   idx_base,           \
                                     rocsparse_matrix_type  matrix_type,        \
                                     rocsparse_fill_mode    uplo,               \
                                     rocsparse_storage_mode storage,            \
                                     rocsparse_data_status* data_status,        \
                                     void*                  temp_buffer)        \
    try                                                                         \
    {                                                                           \
        ROCSPARSE_ROUTINE_TRACE;                                                \
        RETURN_IF_ROCSPARSE_ERROR(                                              \
            (rocsparse::check_matrix_ell_impl<T, rocsparse_int>(handle,         \
                                                                m,              \
                                                                n,              \
                                                                ell_width,      \
                                                                ell_val,        \
                                                                ell_col_ind,    \
                                                                idx_base,       \
                                                                matrix_type,    \
                                                                uplo,           \
                                                                storage,        \
                                                                data_status,    \
                                                                temp_buffer))); \
        return rocsparse_status_success;                                        \
    }                                                                           \
    catch(...)                                                                  \
    {                                                                           \
        RETURN_ROCSPARSE_EXCEPTION();                                           \
    }

C_IMPL(rocsparse_scheck_matrix_ell, float);
C_IMPL(rocsparse_dcheck_matrix_ell, double);
C_IMPL(rocsparse_ccheck_matrix_ell, rocsparse_float_complex);
C_IMPL(rocsparse_zcheck_matrix_ell, rocsparse_double_complex);
#undef C_IMPL
