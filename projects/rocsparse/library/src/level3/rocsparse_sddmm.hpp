/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_handle.hpp"

namespace rocsparse
{
    template <rocsparse_format FORMAT,
              typename T,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C>
    struct rocsparse_sddmm_st
    {

        static rocsparse_status buffer_size(rocsparse_handle     handle,
                                            rocsparse_operation  trans_A,
                                            rocsparse_operation  trans_B,
                                            rocsparse_order      order_A,
                                            rocsparse_order      order_B,
                                            J                    m,
                                            J                    n,
                                            J                    k,
                                            I                    nnz,
                                            const T*             alpha,
                                            const A*             A_val,
                                            int64_t              A_ld,
                                            const B*             B_val,
                                            int64_t              B_ld,
                                            const T*             beta,
                                            const I*             C_row_data,
                                            const J*             C_col_data,
                                            C*                   C_val_data,
                                            rocsparse_index_base C_base,
                                            rocsparse_mat_descr  C_descr,
                                            rocsparse_sddmm_alg  alg,
                                            size_t*              buffer_size);

        static rocsparse_status preprocess(rocsparse_handle     handle,
                                           rocsparse_operation  trans_A,
                                           rocsparse_operation  trans_B,
                                           rocsparse_order      order_A,
                                           rocsparse_order      order_B,
                                           J                    m,
                                           J                    n,
                                           J                    k,
                                           I                    nnz,
                                           const T*             alpha,
                                           const A*             A_val,
                                           int64_t              A_ld,
                                           const B*             B_val,
                                           int64_t              B_ld,
                                           const T*             beta,
                                           const I*             C_row_data,
                                           const J*             C_col_data,
                                           C*                   C_val_data,
                                           rocsparse_index_base C_base,
                                           rocsparse_mat_descr  C_descr,
                                           rocsparse_sddmm_alg  alg,
                                           void*                buffer);

        static rocsparse_status compute(rocsparse_handle     handle,
                                        rocsparse_operation  trans_A,
                                        rocsparse_operation  trans_B,
                                        rocsparse_order      order_A,
                                        rocsparse_order      order_B,
                                        J                    m,
                                        J                    n,
                                        J                    k,
                                        I                    nnz,
                                        const T*             alpha,
                                        const A*             A_val,
                                        int64_t              A_ld,
                                        const B*             B_val,
                                        int64_t              B_ld,
                                        const T*             beta,
                                        const I*             C_row_data,
                                        const J*             C_col_data,
                                        C*                   C_val_data,
                                        rocsparse_index_base C_base,
                                        rocsparse_mat_descr  C_descr,
                                        rocsparse_sddmm_alg  alg,
                                        void*                buffer);

        static rocsparse_status buffer_size_template(rocsparse_handle            handle,
                                                     rocsparse_operation         trans_A,
                                                     rocsparse_operation         trans_B,
                                                     const void*                 alpha,
                                                     rocsparse_const_dnmat_descr mat_A,
                                                     rocsparse_const_dnmat_descr mat_B,
                                                     const void*                 beta,
                                                     const rocsparse_spmat_descr mat_C,
                                                     rocsparse_datatype          compute_type,
                                                     rocsparse_sddmm_alg         alg,
                                                     size_t*                     out_buffer_size)
        {
            switch(FORMAT)
            {
            case rocsparse_format_csr:
            case rocsparse_format_coo:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    (rocsparse_sddmm_st<FORMAT, T, I, J, A, B, C>::buffer_size(
                        handle,
                        trans_A,
                        trans_B,
                        mat_A->order,
                        mat_B->order,
                        mat_C->rows,
                        mat_C->cols,
                        (trans_A == rocsparse_operation_none) ? mat_A->cols : mat_A->rows,
                        mat_C->nnz,
                        (const T*)alpha,
                        (const A*)mat_A->const_values,
                        mat_A->ld,
                        (const B*)mat_B->const_values,
                        mat_B->ld,
                        (const T*)beta,
                        (const I*)mat_C->const_row_data,
                        (const J*)mat_C->const_col_data,
                        (C*)mat_C->val_data,
                        mat_C->idx_base,
                        mat_C->descr,
                        alg,
                        out_buffer_size)));
                return rocsparse_status_success;
            }

            case rocsparse_format_csc:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    (rocsparse_sddmm_st<FORMAT, T, I, J, A, B, C>::buffer_size(
                        handle,
                        trans_A,
                        trans_B,
                        mat_A->order,
                        mat_B->order,
                        mat_C->rows,
                        mat_C->cols,
                        (trans_A == rocsparse_operation_none) ? mat_A->cols : mat_A->rows,
                        mat_C->nnz,
                        (const T*)alpha,
                        (const A*)mat_A->const_values,
                        mat_A->ld,
                        (const B*)mat_B->const_values,
                        mat_B->ld,
                        (const T*)beta,
                        (const I*)mat_C->const_col_data,
                        (const J*)mat_C->const_row_data,
                        (C*)mat_C->val_data,
                        mat_C->idx_base,
                        mat_C->descr,
                        alg,
                        out_buffer_size)));
                return rocsparse_status_success;
            }
            case rocsparse_format_ell:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    (rocsparse_sddmm_st<FORMAT, T, I, J, A, B, C>::buffer_size(
                        handle,
                        trans_A,
                        trans_B,
                        mat_A->order,
                        mat_B->order,
                        mat_C->rows,
                        mat_C->cols,
                        (trans_A == rocsparse_operation_none) ? mat_A->cols : mat_A->rows,
                        mat_C->nnz,
                        (const T*)alpha,
                        (const A*)mat_A->const_values,
                        mat_A->ld,
                        (const B*)mat_B->const_values,
                        mat_B->ld,
                        (const T*)beta,
                        (const I*)nullptr,
                        (const J*)mat_C->const_col_data,
                        (C*)mat_C->val_data,
                        mat_C->idx_base,
                        mat_C->descr,
                        alg,
                        out_buffer_size)));
                return rocsparse_status_success;
            }

            case rocsparse_format_coo_aos:
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    (rocsparse_sddmm_st<FORMAT, T, I, J, A, B, C>::buffer_size(
                        handle,
                        trans_A,
                        trans_B,
                        mat_A->order,
                        mat_B->order,
                        mat_C->rows,
                        mat_C->cols,
                        (trans_A == rocsparse_operation_none) ? mat_A->cols : mat_A->rows,
                        mat_C->nnz,
                        (const T*)alpha,
                        (const A*)mat_A->const_values,
                        mat_A->ld,
                        (const B*)mat_B->const_values,
                        mat_B->ld,
                        (const T*)beta,
                        (const I*)mat_C->const_ind_data,
                        (const J*)(((const I*)mat_C->const_ind_data) + 1),
                        (C*)mat_C->val_data,
                        mat_C->idx_base,
                        mat_C->descr,
                        alg,
                        out_buffer_size)));
                return rocsparse_status_success;
            }
            }
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            // LCOV_EXCL_STOP
        }

        static rocsparse_status preprocess_template(rocsparse_handle            handle,
                                                    rocsparse_operation         trans_A,
                                                    rocsparse_operation         trans_B,
                                                    const void*                 alpha,
                                                    rocsparse_const_dnmat_descr mat_A,
                                                    rocsparse_const_dnmat_descr mat_B,
                                                    const void*                 beta,
                                                    const rocsparse_spmat_descr mat_C,
                                                    rocsparse_datatype          compute_type,
                                                    rocsparse_sddmm_alg         alg,
                                                    void*                       buffer)
        {
            switch(FORMAT)
            {
            case rocsparse_format_csr:
            case rocsparse_format_coo:
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse_sddmm_st<FORMAT, T, I, J, A, B, C>::preprocess(
                    handle,
                    trans_A,
                    trans_B,
                    mat_A->order,
                    mat_B->order,
                    mat_C->rows,
                    mat_C->cols,
                    (trans_A == rocsparse_operation_none) ? mat_A->cols : mat_A->rows,
                    mat_C->nnz,
                    (const T*)alpha,
                    (const A*)mat_A->const_values,
                    mat_A->ld,
                    (const B*)mat_B->const_values,
                    mat_B->ld,
                    (const T*)beta,
                    (const I*)mat_C->const_row_data,
                    (const J*)mat_C->const_col_data,
                    (C*)mat_C->val_data,
                    mat_C->idx_base,
                    mat_C->descr,
                    alg,
                    buffer)));
                return rocsparse_status_success;
            }
            case rocsparse_format_csc:
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse_sddmm_st<FORMAT, T, I, J, A, B, C>::preprocess(
                    handle,
                    trans_A,
                    trans_B,
                    mat_A->order,
                    mat_B->order,
                    mat_C->rows,
                    mat_C->cols,
                    (trans_A == rocsparse_operation_none) ? mat_A->cols : mat_A->rows,
                    mat_C->nnz,
                    (const T*)alpha,
                    (const A*)mat_A->const_values,
                    mat_A->ld,
                    (const B*)mat_B->const_values,
                    mat_B->ld,
                    (const T*)beta,
                    (const I*)mat_C->const_col_data,
                    (const J*)mat_C->const_row_data,
                    (C*)mat_C->val_data,
                    mat_C->idx_base,
                    mat_C->descr,
                    alg,
                    buffer)));
                return rocsparse_status_success;
            }
            case rocsparse_format_ell:
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse_sddmm_st<FORMAT, T, I, J, A, B, C>::preprocess(
                    handle,
                    trans_A,
                    trans_B,
                    mat_A->order,
                    mat_B->order,
                    mat_C->rows,
                    mat_C->cols,
                    (trans_A == rocsparse_operation_none) ? mat_A->cols : mat_A->rows,
                    mat_C->nnz,
                    (const T*)alpha,
                    (const A*)mat_A->const_values,
                    mat_A->ld,
                    (const B*)mat_B->const_values,
                    mat_B->ld,
                    (const T*)beta,
                    (const I*)nullptr,
                    (const J*)mat_C->const_col_data,
                    (C*)mat_C->val_data,
                    mat_C->idx_base,
                    mat_C->descr,
                    alg,
                    buffer)));
                return rocsparse_status_success;
            }
            case rocsparse_format_coo_aos:
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse_sddmm_st<FORMAT, T, I, J, A, B, C>::preprocess(
                    handle,
                    trans_A,
                    trans_B,
                    mat_A->order,
                    mat_B->order,
                    mat_C->rows,
                    mat_C->cols,
                    (trans_A == rocsparse_operation_none) ? mat_A->cols : mat_A->rows,
                    mat_C->nnz,
                    (const T*)alpha,
                    (const A*)mat_A->const_values,
                    mat_A->ld,
                    (const B*)mat_B->const_values,
                    mat_B->ld,
                    (const T*)beta,
                    (const I*)mat_C->const_ind_data,
                    (const J*)(((const I*)mat_C->const_ind_data) + 1),
                    (C*)mat_C->val_data,
                    mat_C->idx_base,
                    mat_C->descr,
                    alg,
                    buffer)));
                return rocsparse_status_success;
            }
            }
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            // LCOV_EXCL_STOP
        }

        static rocsparse_status compute_template(rocsparse_handle            handle,
                                                 rocsparse_operation         trans_A,
                                                 rocsparse_operation         trans_B,
                                                 const void*                 alpha,
                                                 rocsparse_const_dnmat_descr mat_A,
                                                 rocsparse_const_dnmat_descr mat_B,
                                                 const void*                 beta,
                                                 const rocsparse_spmat_descr mat_C,
                                                 rocsparse_datatype          compute_type,
                                                 rocsparse_sddmm_alg         alg,
                                                 void*                       buffer)
        {
            switch(FORMAT)
            {
            case rocsparse_format_csr:
            case rocsparse_format_coo:
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse_sddmm_st<FORMAT, T, I, J, A, B, C>::compute(
                    handle,
                    trans_A,
                    trans_B,
                    mat_A->order,
                    mat_B->order,
                    mat_C->rows,
                    mat_C->cols,
                    (trans_A == rocsparse_operation_none) ? mat_A->cols : mat_A->rows,
                    mat_C->nnz,
                    (const T*)alpha,
                    (const A*)mat_A->const_values,
                    mat_A->ld,
                    (const B*)mat_B->const_values,
                    mat_B->ld,
                    (const T*)beta,
                    (const I*)mat_C->const_row_data,
                    (const J*)mat_C->const_col_data,
                    (C*)mat_C->val_data,
                    mat_C->idx_base,
                    mat_C->descr,
                    alg,
                    buffer)));
                return rocsparse_status_success;
            }
            case rocsparse_format_csc:
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse_sddmm_st<FORMAT, T, I, J, A, B, C>::compute(
                    handle,
                    trans_A,
                    trans_B,
                    mat_A->order,
                    mat_B->order,
                    mat_C->rows,
                    mat_C->cols,
                    (trans_A == rocsparse_operation_none) ? mat_A->cols : mat_A->rows,
                    mat_C->nnz,
                    (const T*)alpha,
                    (const A*)mat_A->const_values,
                    mat_A->ld,
                    (const B*)mat_B->const_values,
                    mat_B->ld,
                    (const T*)beta,
                    (const I*)mat_C->const_col_data,
                    (const J*)mat_C->const_row_data,
                    (C*)mat_C->val_data,
                    mat_C->idx_base,
                    mat_C->descr,
                    alg,
                    buffer)));
                return rocsparse_status_success;
            }
            case rocsparse_format_ell:
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse_sddmm_st<FORMAT, T, I, J, A, B, C>::compute(
                    handle,
                    trans_A,
                    trans_B,
                    mat_A->order,
                    mat_B->order,
                    mat_C->rows,
                    mat_C->cols,
                    (trans_A == rocsparse_operation_none) ? mat_A->cols : mat_A->rows,
                    mat_C->nnz,
                    (const T*)alpha,
                    (const A*)mat_A->const_values,
                    mat_A->ld,
                    (const B*)mat_B->const_values,
                    mat_B->ld,
                    (const T*)beta,
                    (const I*)nullptr,
                    (const J*)mat_C->const_col_data,
                    (C*)mat_C->val_data,
                    mat_C->idx_base,
                    mat_C->descr,
                    alg,
                    buffer)));
                return rocsparse_status_success;
            }
            case rocsparse_format_coo_aos:
            {
                RETURN_IF_ROCSPARSE_ERROR((rocsparse_sddmm_st<FORMAT, T, I, J, A, B, C>::compute(
                    handle,
                    trans_A,
                    trans_B,
                    mat_A->order,
                    mat_B->order,
                    mat_C->rows,
                    mat_C->cols,
                    (trans_A == rocsparse_operation_none) ? mat_A->cols : mat_A->rows,
                    mat_C->nnz,
                    (const T*)alpha,
                    (const A*)mat_A->const_values,
                    mat_A->ld,
                    (const B*)mat_B->const_values,
                    mat_B->ld,
                    (const T*)beta,
                    (const I*)mat_C->const_ind_data,
                    (const J*)(((const I*)mat_C->const_ind_data) + 1),
                    (C*)mat_C->val_data,
                    mat_C->idx_base,
                    mat_C->descr,
                    alg,
                    buffer)));
                return rocsparse_status_success;
            }
            }
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
            // LCOV_EXCL_STOP
        }
    };
}
