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
rocsparse_status rocsparse::coosm_buffer_size_core(rocsparse_handle          handle,
                                                   rocsparse_operation       trans_A,
                                                   rocsparse_operation       trans_B,
                                                   int64_t                   m_,
                                                   int64_t                   nrhs_,
                                                   int64_t                   nnz,
                                                   const rocsparse_mat_descr descr,
                                                   const void*               coo_val_,
                                                   const void*               coo_row_ind_,
                                                   const void*               coo_col_ind_,
                                                   rocsparse_order           order_B,
                                                   rocsparse_mat_info        info,
                                                   rocsparse_solve_policy    policy,
                                                   size_t*                   buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    const I  m           = static_cast<I>(m_);
    const I  nrhs        = static_cast<I>(nrhs_);
    const T* coo_val     = reinterpret_cast<const T*>(coo_val_);
    const I* coo_col_ind = reinterpret_cast<const I*>(coo_col_ind_);

    if(std::is_same<I, int32_t>() && nnz < std::numeric_limits<int32_t>::max())
    {
        // Trick since it is not used in csrsm_buffer_size, otherwise we need to create a proper ptr array for nothing.
        const int32_t* ptr = (const int32_t*)0x4;
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::csrsm_buffer_size_template<int32_t, int32_t, T>(handle,
                                                                        trans_A,
                                                                        trans_B,
                                                                        m,
                                                                        nrhs,
                                                                        nnz,
                                                                        descr,
                                                                        coo_val,
                                                                        ptr,
                                                                        coo_col_ind,
                                                                        order_B,
                                                                        info,
                                                                        policy,
                                                                        buffer_size)));
    }
    else
    {
        //
        // Trick since it is not used in csrsm_buffer_size, otherwise we need to create a proper ptr array for nothing.
        //
        // Note: If this is not used, then it shouldn't be there!
        //
        const int64_t* ptr = (const int64_t*)0x4;
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::csrsm_buffer_size_template<int64_t, int64_t, T>(handle,
                                                                        trans_A,
                                                                        trans_B,
                                                                        m,
                                                                        nrhs,
                                                                        nnz,
                                                                        descr,
                                                                        coo_val,
                                                                        ptr,
                                                                        coo_col_ind,
                                                                        order_B,
                                                                        info,
                                                                        policy,
                                                                        buffer_size)));
    }

    return rocsparse_status_success;
}

rocsparse_status rocsparse::coosm_buffer_size_quickreturn(rocsparse_handle          handle,
                                                          rocsparse_operation       trans_A,
                                                          rocsparse_operation       trans_B,
                                                          int64_t                   m,
                                                          int64_t                   nrhs,
                                                          int64_t                   nnz,
                                                          const rocsparse_mat_descr descr,
                                                          const void*               coo_val,
                                                          const void*               coo_row_ind,
                                                          const void*               coo_col_ind,
                                                          rocsparse_order           order_B,
                                                          rocsparse_mat_info        info,
                                                          rocsparse_solve_policy    policy,
                                                          size_t*                   buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(m == 0 || nrhs == 0)
    {
        *buffer_size = 0;
        return rocsparse_status_success;
    }

    return rocsparse_status_continue;
}

#define INSTANTIATE(I, T)                                              \
    template rocsparse_status rocsparse::coosm_buffer_size_core<I, T>( \
        rocsparse_handle          handle,                              \
        rocsparse_operation       trans_A,                             \
        rocsparse_operation       trans_B,                             \
        int64_t                   m,                                   \
        int64_t                   nrhs,                                \
        int64_t                   nnz,                                 \
        const rocsparse_mat_descr descr,                               \
        const void*               coo_val,                             \
        const void*               coo_row_ind,                         \
        const void*               coo_col_ind,                         \
        rocsparse_order           order_B,                             \
        rocsparse_mat_info        info,                                \
        rocsparse_solve_policy    policy,                              \
        size_t*                   buffer_size)

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
#undef INSTANTIATE
