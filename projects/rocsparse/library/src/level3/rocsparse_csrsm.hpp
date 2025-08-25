/*! \file */
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

#pragma once

#include "rocsparse_control.hpp"
#include "rocsparse_csrsm_info.hpp"
namespace rocsparse
{
    rocsparse_status csrsm_zero_pivot(rocsparse_handle         handle,
                                      rocsparse::pivot_info_t* info,
                                      rocsparse_indextype      indextype,
                                      void*                    position);

    rocsparse_status csrsm_buffer_size_quickreturn(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_B,
                                                   int64_t                   m,
                                                   int64_t                   nrhs,
                                                   int64_t                   nnz,
                                                   const rocsparse_mat_descr descr,
                                                   const void*               csr_val,
                                                   const void*               csr_row_ptr,
                                                   const void*               csr_col_ind,
                                                   rocsparse_order           order_B,
                                                   rocsparse_mat_info        info,
                                                   rocsparse_solve_policy    policy,
                                                   size_t*                   buffer_size);

    template <typename I, typename J, typename T>
    rocsparse_status csrsm_buffer_size_core(rocsparse_handle          handle,
                                            rocsparse_operation       trans_A,
                                            rocsparse_operation       trans_B,
                                            int64_t                   m,
                                            int64_t                   nrhs,
                                            int64_t                   nnz,
                                            const rocsparse_mat_descr descr,
                                            const void*               csr_val,
                                            const void*               csr_row_ptr,
                                            const void*               csr_col_ind,
                                            rocsparse_order           order_B,
                                            rocsparse_mat_info        info,
                                            rocsparse_solve_policy    policy,
                                            size_t*                   buffer_size);

    template <typename I, typename J, typename T>
    rocsparse_status csrsm_buffer_size_template(rocsparse_handle          handle,
                                                rocsparse_operation       trans_A,
                                                rocsparse_operation       trans_B,
                                                int64_t                   m,
                                                int64_t                   nrhs,
                                                int64_t                   nnz,
                                                const rocsparse_mat_descr descr,
                                                const void*               csr_val,
                                                const void*               csr_row_ptr,
                                                const void*               csr_col_ind,
                                                rocsparse_order           order_B,
                                                rocsparse_mat_info        info,
                                                rocsparse_solve_policy    policy,
                                                size_t*                   buffer_size)
    {

        const rocsparse_status status = rocsparse::csrsm_buffer_size_quickreturn(handle,
                                                                                 trans_A,
                                                                                 trans_B,
                                                                                 m,
                                                                                 nrhs,
                                                                                 nnz,
                                                                                 descr,
                                                                                 csr_val,
                                                                                 csr_row_ptr,
                                                                                 csr_col_ind,
                                                                                 order_B,
                                                                                 info,
                                                                                 policy,
                                                                                 buffer_size);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrsm_buffer_size_core<I, J, T>(handle,
                                                                              trans_A,
                                                                              trans_B,
                                                                              m,
                                                                              nrhs,
                                                                              nnz,
                                                                              descr,
                                                                              csr_val,
                                                                              csr_row_ptr,
                                                                              csr_col_ind,
                                                                              order_B,
                                                                              info,
                                                                              policy,

                                                                              buffer_size)));
        return rocsparse_status_success;
    }

    rocsparse_status csrsm_analysis_quickreturn(rocsparse_handle          handle,
                                                rocsparse_operation       trans_A,
                                                rocsparse_operation       trans_B,
                                                int64_t                   m,
                                                int64_t                   nrhs,
                                                int64_t                   nnz,
                                                const void*               alpha,
                                                const rocsparse_mat_descr descr,
                                                const void*               csr_val,
                                                const void*               csr_row_ptr,
                                                const void*               csr_col_ind,
                                                const void*               B,
                                                int64_t                   ldb,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer);

    template <typename I, typename J, typename T>
    rocsparse_status csrsm_analysis_core(rocsparse_handle          handle,
                                         rocsparse_operation       trans_A,
                                         rocsparse_operation       trans_B,
                                         int64_t                   m,
                                         int64_t                   nrhs,
                                         int64_t                   nnz,
                                         const void*               alpha,
                                         const rocsparse_mat_descr descr,
                                         const void*               csr_val,
                                         const void*               csr_row_ptr,
                                         const void*               csr_col_ind,
                                         const void*               B,
                                         int64_t                   ldb,
                                         rocsparse_mat_info        info,
                                         rocsparse_analysis_policy analysis,
                                         rocsparse_solve_policy    solve,
                                         rocsparse_csrsm_info*     p_csrsm_info,
                                         void*                     temp_buffer);

    template <typename I, typename J, typename T>
    rocsparse_status csrsm_analysis_template(rocsparse_handle          handle,
                                             rocsparse_operation       trans_A,
                                             rocsparse_operation       trans_B,
                                             int64_t                   m,
                                             int64_t                   nrhs,
                                             int64_t                   nnz,
                                             const void*               alpha,
                                             const rocsparse_mat_descr descr,
                                             const void*               csr_val,
                                             const void*               csr_row_ptr,
                                             const void*               csr_col_ind,
                                             const void*               B,
                                             int64_t                   ldb,
                                             rocsparse_mat_info        info,
                                             rocsparse_analysis_policy analysis,
                                             rocsparse_solve_policy    solve,
                                             rocsparse_csrsm_info*     p_csrsm_info,
                                             void*                     temp_buffer)
    {

        const rocsparse_status status = rocsparse::csrsm_analysis_quickreturn(handle,
                                                                              trans_A,
                                                                              trans_B,
                                                                              m,
                                                                              nrhs,
                                                                              nnz,
                                                                              alpha,
                                                                              descr,
                                                                              csr_val,
                                                                              csr_row_ptr,
                                                                              csr_col_ind,
                                                                              B,
                                                                              ldb,
                                                                              info,
                                                                              analysis,
                                                                              solve,
                                                                              temp_buffer);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrsm_analysis_core<I, J, T>(handle,
                                                                           trans_A,
                                                                           trans_B,
                                                                           m,
                                                                           nrhs,
                                                                           nnz,
                                                                           alpha,
                                                                           descr,
                                                                           csr_val,
                                                                           csr_row_ptr,
                                                                           csr_col_ind,
                                                                           B,
                                                                           ldb,
                                                                           info,
                                                                           analysis,
                                                                           solve,
                                                                           p_csrsm_info,
                                                                           temp_buffer)));
        return rocsparse_status_success;
    }

    rocsparse_status csrsm_solve_quickreturn(rocsparse_handle          handle,
                                             rocsparse_operation       trans_A,
                                             rocsparse_operation       trans_B,
                                             int64_t                   m,
                                             int64_t                   nrhs,
                                             int64_t                   nnz,
                                             const void*               alpha,
                                             const rocsparse_mat_descr descr,
                                             const void*               csr_val,
                                             const void*               csr_row_ptr,
                                             const void*               csr_col_ind,
                                             void*                     B,
                                             int64_t                   ldb,
                                             rocsparse_order           order_B,
                                             rocsparse_mat_info        info,
                                             rocsparse_solve_policy    policy,
                                             void*                     temp_buffer);

    template <typename I, typename J, typename T>
    rocsparse_status csrsm_solve_core(rocsparse_handle          handle,
                                      rocsparse_operation       trans_A,
                                      rocsparse_operation       trans_B,
                                      int64_t                   m,
                                      int64_t                   nrhs,
                                      int64_t                   nnz,
                                      const void*               alpha_device_host,
                                      const rocsparse_mat_descr descr,
                                      const void*               csr_val,
                                      const void*               csr_row_ptr,
                                      const void*               csr_col_ind,
                                      void*                     B,
                                      int64_t                   ldb,
                                      rocsparse_order           order_B,
                                      rocsparse_mat_info        info,
                                      rocsparse_solve_policy    policy,
                                      rocsparse_csrsm_info      csrsm_info,
                                      void*                     temp_buffer);

    template <typename I, typename J, typename T>
    rocsparse_status csrsm_solve_template(rocsparse_handle          handle,
                                          rocsparse_operation       trans_A,
                                          rocsparse_operation       trans_B,
                                          int64_t                   m,
                                          int64_t                   nrhs,
                                          int64_t                   nnz,
                                          const void*               alpha_device_host,
                                          const rocsparse_mat_descr descr,
                                          const void*               csr_val,
                                          const void*               csr_row_ptr,
                                          const void*               csr_col_ind,
                                          void*                     B,
                                          int64_t                   ldb,
                                          rocsparse_order           order_B,
                                          rocsparse_mat_info        info,
                                          rocsparse_solve_policy    policy,
                                          rocsparse_csrsm_info      csrsm_info,
                                          void*                     temp_buffer)
    {

        const rocsparse_status status = rocsparse::csrsm_solve_quickreturn(handle,
                                                                           trans_A,
                                                                           trans_B,
                                                                           m,
                                                                           nrhs,
                                                                           nnz,
                                                                           alpha_device_host,
                                                                           descr,
                                                                           csr_val,
                                                                           csr_row_ptr,
                                                                           csr_col_ind,
                                                                           B,
                                                                           ldb,
                                                                           order_B,
                                                                           info,
                                                                           policy,
                                                                           temp_buffer);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR((rocsparse::csrsm_solve_core<I, J, T>(handle,
                                                                        trans_A,
                                                                        trans_B,
                                                                        m,
                                                                        nrhs,
                                                                        nnz,
                                                                        alpha_device_host,
                                                                        descr,
                                                                        csr_val,
                                                                        csr_row_ptr,
                                                                        csr_col_ind,
                                                                        B,
                                                                        ldb,
                                                                        order_B,
                                                                        info,
                                                                        policy,
                                                                        csrsm_info,
                                                                        temp_buffer)));
        return rocsparse_status_success;
    }

    rocsparse_status csrsm_buffer_size(rocsparse_handle          handle,
                                       rocsparse_operation       operation,
                                       rocsparse_operation       operation_X,
                                       int64_t                   m,
                                       int64_t                   nrhs,
                                       int64_t                   nnz,
                                       rocsparse_datatype        alpha_datatype,
                                       const rocsparse_mat_descr descr,
                                       rocsparse_datatype        csr_val_datatype,
                                       const void*               csr_val,
                                       rocsparse_indextype       csr_row_ptr_indextype,
                                       const void*               csr_row_ptr,
                                       rocsparse_indextype       csr_col_ind_indextype,
                                       const void*               csr_col_ind,
                                       rocsparse_datatype        datatype_X,
                                       rocsparse_order           order_X,
                                       rocsparse_mat_info        info,
                                       rocsparse_solve_policy    policy,
                                       size_t*                   buffer_size);

    rocsparse_status csrsm_analysis(rocsparse_handle          handle,
                                    rocsparse_operation       trans_A,
                                    rocsparse_operation       trans_B,
                                    int64_t                   m,
                                    int64_t                   nrhs,
                                    int64_t                   nnz,
                                    rocsparse_datatype        alpha_datatype,
                                    const void*               alpha,
                                    const rocsparse_mat_descr descr,
                                    rocsparse_datatype        csr_val_datatype,
                                    const void*               csr_val,
                                    rocsparse_indextype       csr_row_ptr_indextype,
                                    const void*               csr_row_ptr,
                                    rocsparse_indextype       csr_col_ind_indextype,
                                    const void*               csr_col_ind,
                                    rocsparse_datatype        B_datatype,
                                    const void*               B,
                                    int64_t                   ldb,
                                    rocsparse_mat_info        info,
                                    rocsparse_analysis_policy analysis,
                                    rocsparse_solve_policy    solve,
                                    rocsparse_csrsm_info*     p_csrsm_info,
                                    void*                     temp_buffer);

    rocsparse_status csrsm_solve(rocsparse_handle          handle,
                                 rocsparse_operation       trans_A,
                                 rocsparse_operation       trans_B,
                                 int64_t                   m,
                                 int64_t                   nrhs,
                                 int64_t                   nnz,
                                 rocsparse_datatype        alpha_datatype,
                                 const void*               alpha_device_host,
                                 const rocsparse_mat_descr descr,
                                 rocsparse_datatype        csr_val_datatype,
                                 const void*               csr_val,
                                 rocsparse_indextype       csr_row_ptr_indextype,
                                 const void*               csr_row_ptr,
                                 rocsparse_indextype       csr_col_ind_indextype,
                                 const void*               csr_col_ind,
                                 rocsparse_datatype        B_datatype,
                                 void*                     B,
                                 int64_t                   ldb,
                                 rocsparse_order           order_B,
                                 rocsparse_mat_info        info,
                                 rocsparse_solve_policy    policy,
                                 rocsparse_csrsm_info      csrsm_info,
                                 void*                     temp_buffer);

}
