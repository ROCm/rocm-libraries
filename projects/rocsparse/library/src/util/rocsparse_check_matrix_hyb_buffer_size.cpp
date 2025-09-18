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
#include "internal/util/rocsparse_check_matrix_hyb.h"

#include "rocsparse_check_matrix_coo.hpp"
#include "rocsparse_check_matrix_ell.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <typename I>
    static rocsparse_status
        check_matrix_ell_buffer_size_template_dispatch(rocsparse_datatype     type,
                                                       rocsparse_handle       handle,
                                                       I                      m,
                                                       I                      n,
                                                       I                      ell_width,
                                                       void*                  ell_val,
                                                       const I*               ell_col_ind,
                                                       rocsparse_index_base   idx_base,
                                                       rocsparse_matrix_type  matrix_type,
                                                       rocsparse_fill_mode    uplo,
                                                       rocsparse_storage_mode storage,
                                                       size_t*                buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(type)
        {
        case rocsparse_datatype_f32_r:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::check_matrix_ell_buffer_size_impl<float, I>(handle,
                                                                        m,
                                                                        n,
                                                                        ell_width,
                                                                        (float*)ell_val,
                                                                        ell_col_ind,
                                                                        idx_base,
                                                                        matrix_type,
                                                                        uplo,
                                                                        storage,
                                                                        buffer_size)));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_f64_r:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::check_matrix_ell_buffer_size_impl<double, I>(handle,
                                                                         m,
                                                                         n,
                                                                         ell_width,
                                                                         (double*)ell_val,
                                                                         ell_col_ind,
                                                                         idx_base,
                                                                         matrix_type,
                                                                         uplo,
                                                                         storage,
                                                                         buffer_size)));
            return rocsparse_status_success;
        }

        case rocsparse_datatype_f32_c:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::check_matrix_ell_buffer_size_impl<rocsparse_float_complex, I>(
                    handle,
                    m,
                    n,
                    ell_width,
                    (rocsparse_float_complex*)ell_val,
                    ell_col_ind,
                    idx_base,
                    matrix_type,
                    uplo,
                    storage,
                    buffer_size)));
            return rocsparse_status_success;
        }

        case rocsparse_datatype_f64_c:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::check_matrix_ell_buffer_size_impl<rocsparse_double_complex, I>(
                    handle,
                    m,
                    n,
                    ell_width,
                    (rocsparse_double_complex*)ell_val,
                    ell_col_ind,
                    idx_base,
                    matrix_type,
                    uplo,
                    storage,
                    buffer_size)));
            return rocsparse_status_success;
        }

        case rocsparse_datatype_i8_r:
        case rocsparse_datatype_u8_r:
        case rocsparse_datatype_i32_r:
        case rocsparse_datatype_u32_r:
        case rocsparse_datatype_f16_r:
        case rocsparse_datatype_bf16_r:
        {
            return rocsparse_status_not_implemented;
        }
        }
    }

    template <typename I>
    static rocsparse_status
        check_matrix_coo_buffer_size_template_dispatch(rocsparse_datatype     type,
                                                       rocsparse_handle       handle,
                                                       I                      m,
                                                       I                      n,
                                                       I                      nnz,
                                                       void*                  coo_val,
                                                       const I*               coo_row_ind,
                                                       const I*               coo_col_ind,
                                                       rocsparse_index_base   idx_base,
                                                       rocsparse_matrix_type  matrix_type,
                                                       rocsparse_fill_mode    uplo,
                                                       rocsparse_storage_mode storage,
                                                       size_t*                buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(type)
        {
        case rocsparse_datatype_f32_r:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::check_matrix_coo_buffer_size_impl<float, I>(handle,
                                                                        m,
                                                                        n,
                                                                        nnz,
                                                                        (float*)coo_val,
                                                                        coo_row_ind,
                                                                        coo_col_ind,
                                                                        idx_base,
                                                                        matrix_type,
                                                                        uplo,
                                                                        storage,
                                                                        buffer_size)));
            return rocsparse_status_success;
        }

        case rocsparse_datatype_f64_r:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::check_matrix_coo_buffer_size_impl<double, I>(handle,
                                                                         m,
                                                                         n,
                                                                         nnz,
                                                                         (double*)coo_val,
                                                                         coo_row_ind,
                                                                         coo_col_ind,
                                                                         idx_base,
                                                                         matrix_type,
                                                                         uplo,
                                                                         storage,
                                                                         buffer_size)));
            return rocsparse_status_success;
        }

        case rocsparse_datatype_f32_c:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::check_matrix_coo_buffer_size_impl<rocsparse_float_complex, I>(
                    handle,
                    m,
                    n,
                    nnz,
                    (rocsparse_float_complex*)coo_val,
                    coo_row_ind,
                    coo_col_ind,
                    idx_base,
                    matrix_type,
                    uplo,
                    storage,
                    buffer_size)));
            return rocsparse_status_success;
        }

        case rocsparse_datatype_f64_c:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::check_matrix_coo_buffer_size_impl<rocsparse_double_complex, I>(
                    handle,
                    m,
                    n,
                    nnz,
                    (rocsparse_double_complex*)coo_val,
                    coo_row_ind,
                    coo_col_ind,
                    idx_base,
                    matrix_type,
                    uplo,
                    storage,
                    buffer_size)));
            return rocsparse_status_success;
        }

        case rocsparse_datatype_i8_r:
        case rocsparse_datatype_u8_r:
        case rocsparse_datatype_i32_r:
        case rocsparse_datatype_u32_r:
        case rocsparse_datatype_f16_r:
        case rocsparse_datatype_bf16_r:
        {
            return rocsparse_status_not_implemented;
        }
        }
    }
}

rocsparse_status rocsparse_check_matrix_hyb_buffer_size(rocsparse_handle        handle,
                                                        const rocsparse_hyb_mat hyb,
                                                        rocsparse_index_base    idx_base,
                                                        rocsparse_matrix_type   matrix_type,
                                                        rocsparse_fill_mode     uplo,
                                                        rocsparse_storage_mode  storage,
                                                        size_t*                 buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, hyb);
    ROCSPARSE_CHECKARG_ENUM(2, idx_base);
    ROCSPARSE_CHECKARG_ENUM(3, matrix_type);
    ROCSPARSE_CHECKARG_ENUM(4, uplo);
    ROCSPARSE_CHECKARG_ENUM(5, storage);
    ROCSPARSE_CHECKARG_POINTER(6, buffer_size);

    size_t ell_buffer_size;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::check_matrix_ell_buffer_size_template_dispatch(hyb->data_type_T,
                                                                  handle,
                                                                  hyb->m,
                                                                  hyb->n,
                                                                  hyb->ell_width,
                                                                  hyb->ell_val,
                                                                  hyb->ell_col_ind,
                                                                  idx_base,
                                                                  matrix_type,
                                                                  uplo,
                                                                  storage,
                                                                  &ell_buffer_size));

    size_t coo_buffer_size;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::check_matrix_coo_buffer_size_template_dispatch(hyb->data_type_T,
                                                                  handle,
                                                                  hyb->m,
                                                                  hyb->n,
                                                                  hyb->coo_nnz,
                                                                  hyb->coo_val,
                                                                  hyb->coo_row_ind,
                                                                  hyb->coo_col_ind,
                                                                  idx_base,
                                                                  matrix_type,
                                                                  uplo,
                                                                  storage,
                                                                  &coo_buffer_size));

    *buffer_size = rocsparse::max(ell_buffer_size, coo_buffer_size);
    return rocsparse_status_success;
}
