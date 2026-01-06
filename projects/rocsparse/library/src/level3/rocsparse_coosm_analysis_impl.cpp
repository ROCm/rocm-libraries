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
#include "rocsparse_control.hpp"
#include "rocsparse_coosm.hpp"
#include "rocsparse_csrsm.hpp"
#include "rocsparse_utility.hpp"

#include "../conversion/rocsparse_coo2csr.hpp"

template <typename I, typename T>
rocsparse_status rocsparse::coosm_analysis_core(rocsparse_handle          handle,
                                                rocsparse_operation       trans_A,
                                                rocsparse_operation       trans_B,
                                                int64_t                   m_,
                                                int64_t                   nrhs_,
                                                int64_t                   nnz,
                                                const void*               alpha_device_host_,
                                                const rocsparse_mat_descr descr,
                                                const void*               coo_val_,
                                                const void*               coo_row_ind_,
                                                const void*               coo_col_ind_,
                                                const void*               B_,
                                                int64_t                   ldb,
                                                rocsparse_mat_info        info,
                                                rocsparse_analysis_policy analysis,
                                                rocsparse_solve_policy    solve,
                                                rocsparse_csrsm_info*     p_csrsm_info,
                                                void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    const I  m                 = static_cast<I>(m_);
    const I  nrhs              = static_cast<I>(nrhs_);
    const T* alpha_device_host = reinterpret_cast<const T*>(alpha_device_host_);
    const T* coo_val           = reinterpret_cast<const T*>(coo_val_);
    const I* coo_row_ind       = reinterpret_cast<const I*>(coo_row_ind_);
    const I* coo_col_ind       = reinterpret_cast<const I*>(coo_col_ind_);
    const T* B                 = reinterpret_cast<const T*>(B_);

    const bool                choose_i32 = nnz <= std::numeric_limits<int32_t>::max();
    const rocsparse_indextype indextype
        = choose_i32 ? rocsparse_indextype_i32 : rocsparse_indextype_i64;

    rocsparse::sorted_coo2csr_info_t* sorted_coo2csr_info = info->get_sorted_coo2csr_info();
    if(sorted_coo2csr_info == nullptr)
    {
        sorted_coo2csr_info = new rocsparse::sorted_coo2csr_info_t(m, indextype, handle->stream);

        //
        // Assign it first, because if an error occurs in calculate below, then we won't have a memory leak.
        //
        info->set_sorted_coo2csr_info(sorted_coo2csr_info);

        RETURN_IF_ROCSPARSE_ERROR(sorted_coo2csr_info->calculate(
            handle, nnz, coo_row_ind, rocsparse::get_indextype<I>(), descr->base));
    }

    const I* csr_col_ind = coo_col_ind;
    const T* csr_val     = coo_val;

    if(choose_i32)
    {
        const int32_t* csr_row_ptr = (const int32_t*)sorted_coo2csr_info->get_row_ptr();
        // Call CSR analysis
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::csrsm_analysis_template<int32_t, I, T>(handle,
                                                               trans_A,
                                                               trans_B,
                                                               m,
                                                               nrhs,
                                                               static_cast<int32_t>(nnz),
                                                               alpha_device_host,
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
    }
    else
    {
        const int64_t* csr_row_ptr = (const int64_t*)sorted_coo2csr_info->get_row_ptr();
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::csrsm_analysis_template<int64_t, I, T>(handle,
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
                                                               info,
                                                               analysis,
                                                               solve,
                                                               p_csrsm_info,
                                                               temp_buffer)));
    }

    return rocsparse_status_success;
}

rocsparse_status rocsparse::coosm_analysis_quickreturn(rocsparse_handle          handle,
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
                                                       void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(m == 0 || nrhs == 0)
    {
        return rocsparse_status_success;
    }
    return rocsparse_status_continue;
}

#define INSTANTIATE(I, T)                                           \
    template rocsparse_status rocsparse::coosm_analysis_core<I, T>( \
        rocsparse_handle          handle,                           \
        rocsparse_operation       trans_A,                          \
        rocsparse_operation       trans_B,                          \
        int64_t                   m,                                \
        int64_t                   nrhs,                             \
        int64_t                   nnz,                              \
        const void*               alpha_device_host,                \
        const rocsparse_mat_descr descr,                            \
        const void*               coo_val,                          \
        const void*               coo_row_ind,                      \
        const void*               coo_col_ind,                      \
        const void*               B,                                \
        int64_t                   ldb,                              \
        rocsparse_mat_info        info,                             \
        rocsparse_analysis_policy analysis,                         \
        rocsparse_solve_policy    solve,                            \
        rocsparse_csrsm_info*     p_csrsm_info,                     \
        void*                     temp_buffer);

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
#undef INSTANTIATE
