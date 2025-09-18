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

#include "rocsparse_bsrgemm_mult.hpp"
#include "../conversion/rocsparse_identity.hpp"
#include "bsrgemm_device.h"
#include "csrgemm_device.h"
#include "internal/extra/rocsparse_bsrgemm.h"
#include "rocsparse_bsrgemm.hpp"
#include "rocsparse_bsrgemm_calc.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_csrgemm.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::bsrgemm_mult_quickreturn(rocsparse_handle          handle,
                                                     rocsparse_direction       dir,
                                                     rocsparse_operation       trans_A,
                                                     rocsparse_operation       trans_B,
                                                     int64_t                   mb,
                                                     int64_t                   nb,
                                                     int64_t                   kb,
                                                     int64_t                   block_dim,
                                                     const void*               alpha,
                                                     const rocsparse_mat_descr descr_A,
                                                     int64_t                   nnzb_A,
                                                     const void*               bsr_val_A,
                                                     const void*               bsr_row_ptr_A,
                                                     const void*               bsr_col_ind_A,
                                                     const rocsparse_mat_descr descr_B,
                                                     int64_t                   nnzb_B,
                                                     const void*               bsr_val_B,
                                                     const void*               bsr_row_ptr_B,
                                                     const void*               bsr_col_ind_B,
                                                     const rocsparse_mat_descr descr_C,
                                                     void*                     bsr_val_C,
                                                     const void*               bsr_row_ptr_C,
                                                     void*                     bsr_col_ind_C,
                                                     const rocsparse_mat_info  info_C,
                                                     void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(mb == 0 || nb == 0 || kb == 0 || nnzb_A == 0 || nnzb_B == 0)
    {
        return rocsparse_status_success;
    }
    return rocsparse_status_continue;
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse::bsrgemm_mult_core(rocsparse_handle          handle,
                                              rocsparse_direction       dir,
                                              rocsparse_operation       trans_A,
                                              rocsparse_operation       trans_B,
                                              J                         mb,
                                              J                         nb,
                                              J                         kb,
                                              J                         block_dim,
                                              const T*                  alpha,
                                              const rocsparse_mat_descr descr_A,
                                              I                         nnzb_A,
                                              const T*                  bsr_val_A,
                                              const I*                  bsr_row_ptr_A,
                                              const J*                  bsr_col_ind_A,
                                              const rocsparse_mat_descr descr_B,
                                              I                         nnzb_B,
                                              const T*                  bsr_val_B,
                                              const I*                  bsr_row_ptr_B,
                                              const J*                  bsr_col_ind_B,
                                              const rocsparse_mat_descr descr_C,
                                              T*                        bsr_val_C,
                                              const I*                  bsr_row_ptr_C,
                                              J*                        bsr_col_ind_C,
                                              const rocsparse_mat_info  info_C,
                                              void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::bsrgemm_calc_template_dispatch(handle,
                                                                        dir,
                                                                        trans_A,
                                                                        trans_B,
                                                                        mb,
                                                                        nb,
                                                                        kb,
                                                                        block_dim,
                                                                        alpha,
                                                                        descr_A,
                                                                        nnzb_A,
                                                                        bsr_val_A,
                                                                        bsr_row_ptr_A,
                                                                        bsr_col_ind_A,
                                                                        descr_B,
                                                                        nnzb_B,
                                                                        bsr_val_B,
                                                                        bsr_row_ptr_B,
                                                                        bsr_col_ind_B,
                                                                        (const T*)nullptr,
                                                                        nullptr,
                                                                        (I)0,
                                                                        (const T*)nullptr,
                                                                        (const I*)nullptr,
                                                                        (const J*)nullptr,
                                                                        descr_C,
                                                                        bsr_val_C,
                                                                        bsr_row_ptr_C,
                                                                        bsr_col_ind_C,
                                                                        info_C,
                                                                        temp_buffer));
    return rocsparse_status_success;
}

#define INSTANTIATE(I, J, T)                                                                        \
    template rocsparse_status rocsparse::bsrgemm_mult_core(rocsparse_handle          handle,        \
                                                           rocsparse_direction       dir,           \
                                                           rocsparse_operation       trans_A,       \
                                                           rocsparse_operation       trans_B,       \
                                                           J                         mb,            \
                                                           J                         nb,            \
                                                           J                         kb,            \
                                                           J                         block_dim,     \
                                                           const T*                  alpha,         \
                                                           const rocsparse_mat_descr descr_A,       \
                                                           I                         nnzb_A,        \
                                                           const T*                  bsr_val_A,     \
                                                           const I*                  bsr_row_ptr_A, \
                                                           const J*                  bsr_col_ind_A, \
                                                           const rocsparse_mat_descr descr_B,       \
                                                           I                         nnzb_B,        \
                                                           const T*                  bsr_val_B,     \
                                                           const I*                  bsr_row_ptr_B, \
                                                           const J*                  bsr_col_ind_B, \
                                                           const rocsparse_mat_descr descr_C,       \
                                                           T*                        bsr_val_C,     \
                                                           const I*                  bsr_row_ptr_C, \
                                                           J*                        bsr_col_ind_C, \
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
