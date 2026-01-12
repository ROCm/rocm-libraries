/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_csrsv.hpp"
#include "rocsparse_dnvec_descr.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <typename I, typename J, typename T>
    inline rocsparse_status csrsv_buffer_size_template(rocsparse_handle          handle,
                                                       rocsparse_operation       op,
                                                       int64_t                   m,
                                                       int64_t                   nnz,
                                                       const rocsparse_mat_descr descr,
                                                       const void*               csr_val,
                                                       const void*               csr_row_ptr,
                                                       const void*               csr_col_ind,
                                                       rocsparse_mat_info        info,
                                                       size_t* p_buffer_size_in_bytes)
    {
        const int64_t batch_count            = static_cast<int64_t>(1);
        const int64_t batch_dist_csr_val     = static_cast<int64_t>(0);
        const int64_t batch_dist_csr_row_ptr = static_cast<int64_t>(0);
        const int64_t batch_dist_csr_col_ind = static_cast<int64_t>(0);

        _rocsparse_idvec_descr row_idvec(rocsparse::get_indextype<I>(),
                                         descr->base,
                                         m + 1,
                                         static_cast<int64_t>(1),
                                         csr_row_ptr,
                                         nullptr);

        _rocsparse_idvec_descr col_idvec(rocsparse::get_indextype<J>(),
                                         descr->base,
                                         nnz,
                                         static_cast<int64_t>(1),
                                         csr_col_ind,
                                         nullptr);

        _rocsparse_dnvec_descr val_dnvec(
            rocsparse::get_datatype<T>(), nnz, static_cast<int64_t>(1), csr_val, nullptr);

        _rocsparse_spattern_descr csr_spattern(
            rocsparse_format_csr, m, m, nnz, &row_idvec, &col_idvec, descr);

        _rocsparse_spmat_descr csr(&csr_spattern, &val_dnvec, info);

#if 0
        _rocsparse_spmat_descr csr(rocsparse_format_csr,
                                   false,
                                   batch_count,
                                   m,
                                   m,
                                   nnz,
                                   rocsparse::get_datatype<T>(),
                                   csr_val,
                                   nullptr,
                                   batch_dist_csr_val,
                                   rocsparse::get_indextype<I>(),
                                   csr_row_ptr,
                                   nullptr,
                                   batch_dist_csr_row_ptr,
                                   rocsparse::get_indextype<J>(),
                                   csr_col_ind,
                                   nullptr,
                                   batch_dist_csr_col_ind,
                                   descr->base,
                                   descr,
                                   info);
#endif

        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::csrsv_analysis_buffer_size(handle, op, &csr, p_buffer_size_in_bytes));
        size_t buffer_size_in_bytes_solve;
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::csrsv_solve_buffer_size(handle, op, &csr, &buffer_size_in_bytes_solve));

        p_buffer_size_in_bytes[0] = std::max(p_buffer_size_in_bytes[0], buffer_size_in_bytes_solve);
        return rocsparse_status_success;
    }

    template <typename I, typename J, typename T>
    inline rocsparse_status csrsv_analysis_template(rocsparse_handle          handle,
                                                    rocsparse_operation       op,
                                                    int64_t                   m,
                                                    int64_t                   nnz,
                                                    const rocsparse_mat_descr descr,
                                                    const void*               csr_val,
                                                    const void*               csr_row_ptr,
                                                    const void*               csr_col_ind,
                                                    rocsparse_mat_info        info,
                                                    rocsparse_analysis_policy analysis,
                                                    rocsparse_solve_policy    solve,
                                                    rocsparse_csrsv_info*     p_csrsv_info,
                                                    void*                     buffer)
    {
        const int64_t batch_count            = static_cast<int64_t>(1);
        const int64_t batch_dist_csr_val     = static_cast<int64_t>(0);
        const int64_t batch_dist_csr_row_ptr = static_cast<int64_t>(0);
        const int64_t batch_dist_csr_col_ind = static_cast<int64_t>(0);

#if 0
        _rocsparse_spmat_descr csr(rocsparse_format_csr,
                                   false,
                                   batch_count,
                                   m,
                                   m,
                                   nnz,
                                   rocsparse::get_datatype<T>(),
                                   csr_val,
                                   nullptr,
                                   batch_dist_csr_val,
                                   rocsparse::get_indextype<I>(),
                                   csr_row_ptr,
                                   nullptr,
                                   batch_dist_csr_row_ptr,
                                   rocsparse::get_indextype<J>(),
                                   csr_col_ind,
                                   nullptr,
                                   batch_dist_csr_col_ind,
                                   descr->base,
                                   descr,
                                   info);
#endif

        _rocsparse_idvec_descr row_idvec(rocsparse::get_indextype<I>(),
                                         descr->base,
                                         m + 1,
                                         static_cast<int64_t>(1),
                                         csr_row_ptr,
                                         nullptr);

        _rocsparse_idvec_descr col_idvec(rocsparse::get_indextype<J>(),
                                         descr->base,
                                         nnz,
                                         static_cast<int64_t>(1),
                                         csr_col_ind,
                                         nullptr);

        _rocsparse_dnvec_descr val_dnvec(
            rocsparse::get_datatype<T>(), nnz, static_cast<int64_t>(1), csr_val, nullptr);

        _rocsparse_spattern_descr csr_spattern(
            rocsparse_format_csr, m, m, nnz, &row_idvec, &col_idvec, descr);

        _rocsparse_spmat_descr csr(&csr_spattern, &val_dnvec, info);

        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::csrsv_analysis(handle, op, &csr, analysis, solve, p_csrsv_info, buffer));
        return rocsparse_status_success;
    }

    template <typename I, typename J, typename T>
    inline rocsparse_status csrsv_solve_template(rocsparse_handle          handle,
                                                 rocsparse_operation       op,
                                                 int64_t                   m,
                                                 int64_t                   nnz,
                                                 const void*               alpha,
                                                 const rocsparse_mat_descr descr,
                                                 const void*               csr_val,
                                                 const void*               csr_row_ptr,
                                                 const void*               csr_col_ind,
                                                 rocsparse_mat_info        info,
                                                 const void*               x,
                                                 int64_t                   x_inc,
                                                 void*                     y,
                                                 rocsparse_solve_policy    policy,
                                                 rocsparse_csrsv_info      csrsv_info,
                                                 void*                     buffer)
    {
        const int64_t                batch_count            = static_cast<int64_t>(1);
        const int64_t                batch_dist_csr_val     = static_cast<int64_t>(0);
        const int64_t                batch_dist_csr_row_ptr = static_cast<int64_t>(0);
        const int64_t                batch_dist_csr_col_ind = static_cast<int64_t>(0);
        const int64_t                batch_dist_x           = static_cast<int64_t>(0);
        const int64_t                batch_dist_y           = static_cast<int64_t>(0);
        const int64_t                batch_dist_alpha       = static_cast<int64_t>(0);
        const int64_t                inc_x                  = x_inc;
        const int64_t                inc_y                  = static_cast<int64_t>(1);
        const rocsparse_batchtype    batchtype              = rocsparse_batchtype_strided;
        const rocsparse_batchstorage batchstorage           = rocsparse_batchstorage_soa;

        _rocsparse_idvec_descr row_idvec(rocsparse::get_indextype<I>(),
                                         descr->base,
                                         m + 1,
                                         static_cast<int64_t>(1),
                                         csr_row_ptr,
                                         nullptr);

        _rocsparse_idvec_descr col_idvec(rocsparse::get_indextype<J>(),
                                         descr->base,
                                         nnz,
                                         static_cast<int64_t>(1),
                                         csr_col_ind,
                                         nullptr);

        _rocsparse_dnvec_descr val_dnvec(
            rocsparse::get_datatype<T>(), nnz, x_inc, csr_val, nullptr);

        _rocsparse_spattern_descr csr_spattern(
            rocsparse_format_csr, m, m, nnz, &row_idvec, &col_idvec, descr);

        _rocsparse_spmat_descr csr(&csr_spattern, &val_dnvec, info);

#if 0

	_rocsparse_spmat_descr csr(rocsparse_format_csr,
                                   false,
                                   batch_count,
                                   m,
                                   m,
                                   nnz,
                                   rocsparse::get_datatype<T>(),
                                   csr_val,
                                   nullptr,
                                   batch_dist_csr_val,
                                   rocsparse::get_indextype<I>(),
                                   csr_row_ptr,
                                   nullptr,
                                   batch_dist_csr_row_ptr,
                                   rocsparse::get_indextype<J>(),
                                   csr_col_ind,
                                   nullptr,
                                   batch_dist_csr_col_ind,
                                   descr->base,
                                   descr,
                                   info);
#endif

        _rocsparse_dnvec_descr dnx(rocsparse::get_datatype<T>(),
                                   m,
                                   inc_x,
#if 0
				   batchtype,
				   batchstorage,
				   batch_count,
				   batch_dist_x,
#endif
                                   x,
                                   nullptr);

        _rocsparse_dnvec_descr dny(rocsparse::get_datatype<T>(),
                                   m,
                                   inc_y,
#if 0
				   batchtype,
				   batchstorage,
				   batch_count,
				   batch_dist_y,
#endif
                                   y,
                                   y);

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_solve(handle,
                                                         op,
                                                         rocsparse::get_datatype<T>(),
                                                         alpha,
                                                         batch_dist_alpha,
                                                         &csr,
                                                         &dnx,
                                                         &dny,
                                                         policy,
                                                         csrsv_info,
                                                         buffer));

        return rocsparse_status_success;
    }
}
