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

#include "rocsparse_csrgemm_mult.hpp"
#include "../conversion/rocsparse_identity.hpp"
#include "csrgemm_device.h"
#include "internal/extra/rocsparse_csrgemm.h"
#include "rocsparse_control.hpp"
#include "rocsparse_csrgemm.hpp"
#include "rocsparse_csrgemm_calc.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::csrgemm_mult_quickreturn(rocsparse_handle          handle,
                                                     rocsparse_operation       trans_A,
                                                     rocsparse_operation       trans_B,
                                                     int64_t                   m,
                                                     int64_t                   n,
                                                     int64_t                   k,
                                                     const void*               alpha,
                                                     const rocsparse_mat_descr descr_A,
                                                     int64_t                   nnz_A,
                                                     const void*               csr_val_A,
                                                     const void*               csr_row_ptr_A,
                                                     const void*               csr_col_ind_A,
                                                     const rocsparse_mat_descr descr_B,
                                                     int64_t                   nnz_B,
                                                     const void*               csr_val_B,
                                                     const void*               csr_row_ptr_B,
                                                     const void*               csr_col_ind_B,
                                                     const rocsparse_mat_descr descr_C,
                                                     void*                     csr_val_C,
                                                     const void*               csr_row_ptr_C,
                                                     void*                     csr_col_ind_C,
                                                     const rocsparse_mat_info  info_C,
                                                     void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(m == 0 || n == 0 || k == 0 || nnz_A == 0 || nnz_B == 0)
    {
        return rocsparse_status_success;
    }
    return rocsparse_status_continue;
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse::csrgemm_mult_core(rocsparse_handle          handle,
                                              rocsparse_operation       trans_A,
                                              rocsparse_operation       trans_B,
                                              J                         m,
                                              J                         n,
                                              J                         k,
                                              const T*                  alpha,
                                              const rocsparse_mat_descr descr_A,
                                              I                         nnz_A,
                                              const T*                  csr_val_A,
                                              const I*                  csr_row_ptr_A,
                                              const J*                  csr_col_ind_A,
                                              const rocsparse_mat_descr descr_B,
                                              I                         nnz_B,
                                              const T*                  csr_val_B,
                                              const I*                  csr_row_ptr_B,
                                              const J*                  csr_col_ind_B,
                                              const rocsparse_mat_descr descr_C,
                                              T*                        csr_val_C,
                                              const I*                  csr_row_ptr_C,
                                              J*                        csr_col_ind_C,
                                              const rocsparse_mat_info  info_C,
                                              void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    const bool mul = info_C->csrgemm_info->mul;
    const bool add = info_C->csrgemm_info->add;
    if(mul == true && add == false)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgemm_calc_template(handle,
                                                                   trans_A,
                                                                   trans_B,
                                                                   m,
                                                                   n,
                                                                   k,
                                                                   alpha,
                                                                   descr_A,
                                                                   nnz_A,
                                                                   csr_val_A,
                                                                   csr_row_ptr_A,
                                                                   csr_col_ind_A,
                                                                   descr_B,
                                                                   nnz_B,
                                                                   csr_val_B,
                                                                   csr_row_ptr_B,
                                                                   csr_col_ind_B,
                                                                   (const T*)nullptr,
                                                                   nullptr,
                                                                   (I)0,
                                                                   (const T*)nullptr,
                                                                   (const I*)nullptr,
                                                                   (const J*)nullptr,
                                                                   descr_C,
                                                                   csr_val_C,
                                                                   csr_row_ptr_C,
                                                                   csr_col_ind_C,
                                                                   info_C,
                                                                   temp_buffer));
        return rocsparse_status_success;
    }
    else
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error,
                                               "failed condition (mul == true && add == false)");
    }
}

#define INSTANTIATE(I, J, T)                                                                        \
    template rocsparse_status rocsparse::csrgemm_mult_core(rocsparse_handle          handle,        \
                                                           rocsparse_operation       trans_A,       \
                                                           rocsparse_operation       trans_B,       \
                                                           J                         m,             \
                                                           J                         n,             \
                                                           J                         k,             \
                                                           const T*                  alpha,         \
                                                           const rocsparse_mat_descr descr_A,       \
                                                           I                         nnz_A,         \
                                                           const T*                  csr_val_A,     \
                                                           const I*                  csr_row_ptr_A, \
                                                           const J*                  csr_col_ind_A, \
                                                           const rocsparse_mat_descr descr_B,       \
                                                           I                         nnz_B,         \
                                                           const T*                  csr_val_B,     \
                                                           const I*                  csr_row_ptr_B, \
                                                           const J*                  csr_col_ind_B, \
                                                           const rocsparse_mat_descr descr_C,       \
                                                           T*                        csr_val_C,     \
                                                           const I*                  csr_row_ptr_C, \
                                                           J*                        csr_col_ind_C, \
                                                           const rocsparse_mat_info  info_C,        \
                                                           void*                     temp_buffer)

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);

INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);

INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);

#undef INSTANTIATE
