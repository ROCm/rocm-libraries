/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights Reserved.
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
    rocsparse_status coosm_buffer_size_quickreturn(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_B,
                                                   int64_t                   m,
                                                   int64_t                   nrhs,
                                                   int64_t                   nnz,
                                                   const void*               alpha_device_host,
                                                   const rocsparse_mat_descr descr,
                                                   const void*               coo_val,
                                                   const void*               coo_row_ind,
                                                   const void*               coo_col_ind,
                                                   const void*               B,
                                                   int64_t                   ldb,
                                                   rocsparse_order           order_B,
                                                   rocsparse_mat_info        info,
                                                   rocsparse_solve_policy    policy,
                                                   size_t*                   buffer_size);

    template <typename I, typename T>
    rocsparse_status coosm_buffer_size_core(rocsparse_handle          handle,
                                            rocsparse_operation       trans_A,
                                            rocsparse_operation       trans_B,
                                            I                         m,
                                            I                         nrhs,
                                            int64_t                   nnz,
                                            const T*                  alpha_device_host,
                                            const rocsparse_mat_descr descr,
                                            const T*                  coo_val,
                                            const I*                  coo_row_ind,
                                            const I*                  coo_col_ind,
                                            const T*                  B,
                                            int64_t                   ldb,
                                            rocsparse_order           order_B,
                                            rocsparse_mat_info        info,
                                            rocsparse_solve_policy    policy,
                                            size_t*                   buffer_size);

    template <typename... P>
    rocsparse_status coosm_buffer_size_template(P&&... p)
    {

        const rocsparse_status status = rocsparse::coosm_buffer_size_quickreturn(p...);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosm_buffer_size_core(p...));
        return rocsparse_status_success;
    }

    rocsparse_status coosm_analysis_quickreturn(rocsparse_handle          handle,
                                                rocsparse_operation       trans_A,
                                                rocsparse_operation       trans_B,
                                                int64_t                   m,
                                                int64_t                   nrhs,
                                                int64_t                   nnz,
                                                const void*               alpha_device_host,
                                                const rocsparse_mat_descr descr,
                                                const void*               coo_val,
                                                const void*               coo_row_ind,
                                                const void*               coo_col_ind,
                                                const void*               B,
                                                int64_t                   ldb,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                void*                     temp_buffer);

    template <typename I, typename T>
    rocsparse_status coosm_analysis_core(rocsparse_handle          handle,
                                         rocsparse_operation       trans_A,
                                         rocsparse_operation       trans_B,
                                         I                         m,
                                         I                         nrhs,
                                         int64_t                   nnz,
                                         const T*                  alpha_device_host,
                                         const rocsparse_mat_descr descr,
                                         const T*                  coo_val,
                                         const I*                  coo_row_ind,
                                         const I*                  coo_col_ind,
                                         const T*                  B,
                                         int64_t                   ldb,
                                         rocsparse_mat_info        info,
                                         rocsparse_analysis_policy analysis,
                                         rocsparse_solve_policy    solve,
                                         void*                     temp_buffer);

    template <typename... P>
    rocsparse_status coosm_analysis_template(P&&... p)
    {

        const rocsparse_status status = rocsparse::coosm_analysis_quickreturn(p...);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosm_analysis_core(p...));
        return rocsparse_status_success;
    }

    rocsparse_status coosm_solve_quickreturn(rocsparse_handle          handle,
                                             rocsparse_operation       trans_A,
                                             rocsparse_operation       trans_B,
                                             int64_t                   m,
                                             int64_t                   nrhs,
                                             int64_t                   nnz,
                                             const void*               alpha_device_host,
                                             const rocsparse_mat_descr descr,
                                             const void*               coo_val,
                                             const void*               coo_row_ind,
                                             const void*               coo_col_ind,
                                             void*                     B,
                                             int64_t                   ldb,
                                             rocsparse_order           order_B,
                                             rocsparse_mat_info        info,
                                             rocsparse_solve_policy    policy,
                                             void*                     temp_buffer);

    template <typename I, typename T>
    rocsparse_status coosm_solve_core(rocsparse_handle          handle,
                                      rocsparse_operation       trans_A,
                                      rocsparse_operation       trans_B,
                                      I                         m,
                                      I                         nrhs,
                                      int64_t                   nnz,
                                      const T*                  alpha_device_host,
                                      const rocsparse_mat_descr descr,
                                      const T*                  coo_val,
                                      const I*                  coo_row_ind,
                                      const I*                  coo_col_ind,
                                      T*                        B,
                                      int64_t                   ldb,
                                      rocsparse_order           order_B,
                                      rocsparse_mat_info        info,
                                      rocsparse_solve_policy    policy,
                                      void*                     temp_buffer);

    template <typename... P>
    rocsparse_status coosm_solve_template(P&&... p)
    {
        const rocsparse_status status = rocsparse::coosm_solve_quickreturn(p...);

        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosm_solve_core(p...));
        return rocsparse_status_success;
    }
}
