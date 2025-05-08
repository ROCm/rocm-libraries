/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights Reserved.
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
#pragma once

#include "control.h"

namespace rocsparse
{
    template <typename T, typename I, typename J>
    rocsparse_status check_matrix_csr_buffer_size_core(rocsparse_handle       handle,
                                                       J                      m,
                                                       J                      n,
                                                       I                      nnz,
                                                       const T*               csr_val,
                                                       const I*               csr_row_ptr,
                                                       const J*               csr_col_ind,
                                                       rocsparse_index_base   idx_base,
                                                       rocsparse_matrix_type  matrix_type,
                                                       rocsparse_fill_mode    uplo,
                                                       rocsparse_storage_mode storage,
                                                       size_t*                buffer_size);

    template <typename T, typename I, typename J>
    rocsparse_status check_matrix_csr_buffer_size_checkarg(rocsparse_handle       handle,
                                                           J                      m,
                                                           J                      n,
                                                           I                      nnz,
                                                           const T*               csr_val,
                                                           const I*               csr_row_ptr,
                                                           const J*               csr_col_ind,
                                                           rocsparse_index_base   idx_base,
                                                           rocsparse_matrix_type  matrix_type,
                                                           rocsparse_fill_mode    uplo,
                                                           rocsparse_storage_mode storage,
                                                           size_t*                buffer_size);

    template <typename T, typename I, typename J, typename... P>
    inline rocsparse_status check_matrix_csr_buffer_size_impl(P&&... p)
    {
        const rocsparse_status status
            = rocsparse::check_matrix_csr_buffer_size_checkarg<T, I, J>(p...);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR((rocsparse::check_matrix_csr_buffer_size_core<T, I, J>(p...)));
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        return rocsparse_status_success;
    }

    template <typename T, typename I, typename J>
    rocsparse_status check_matrix_csr_core(rocsparse_handle       handle,
                                           J                      m,
                                           J                      n,
                                           I                      nnz,
                                           const T*               csr_val,
                                           const I*               csr_row_ptr,
                                           const J*               csr_col_ind,
                                           rocsparse_index_base   idx_base,
                                           rocsparse_matrix_type  matrix_type,
                                           rocsparse_fill_mode    uplo,
                                           rocsparse_storage_mode storage,
                                           rocsparse_data_status* data_status,
                                           void*                  temp_buffer);

    template <typename T, typename I, typename J>
    rocsparse_status check_matrix_csr_checkarg(rocsparse_handle       handle,
                                               J                      m,
                                               J                      n,
                                               I                      nnz,
                                               const T*               csr_val,
                                               const I*               csr_row_ptr,
                                               const J*               csr_col_ind,
                                               rocsparse_index_base   idx_base,
                                               rocsparse_matrix_type  matrix_type,
                                               rocsparse_fill_mode    uplo,
                                               rocsparse_storage_mode storage,
                                               rocsparse_data_status* data_status,
                                               void*                  temp_buffer);

    template <typename T, typename I, typename J, typename... P>
    inline rocsparse_status check_matrix_csr_impl(P&&... p)
    {
        const rocsparse_status status = rocsparse::check_matrix_csr_checkarg<T, I, J>(p...);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR((rocsparse::check_matrix_csr_core<T, I, J>(p...)));
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        return rocsparse_status_success;
    }
}
