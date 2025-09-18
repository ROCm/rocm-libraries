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

#include "../conversion/rocsparse_identity.hpp"
#include "csrgemm_device.h"
#include "internal/extra/rocsparse_csrgemm.h"
#include "rocsparse_control.hpp"
#include "rocsparse_csrgemm.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_csrgemm_scal.hpp"

rocsparse_status rocsparse::csrgemm_scal_buffer_size_quickreturn(rocsparse_handle          handle,
                                                                 int64_t                   m,
                                                                 int64_t                   n,
                                                                 const void*               beta,
                                                                 const rocsparse_mat_descr descr_D,
                                                                 int64_t                   nnz_D,
                                                                 const void*        csr_row_ptr_D,
                                                                 const void*        csr_col_ind_D,
                                                                 rocsparse_mat_info info_C,
                                                                 size_t*            buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(m == 0 || nnz_D == 0)
    {
        *buffer_size                         = 0;
        info_C->csrgemm_info->buffer_size    = buffer_size[0];
        info_C->csrgemm_info->is_initialized = true;
        return rocsparse_status_success;
    }

    return rocsparse_status_continue;
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse::csrgemm_scal_buffer_size_core(rocsparse_handle          handle,
                                                          J                         m,
                                                          J                         n,
                                                          const T*                  beta,
                                                          const rocsparse_mat_descr descr_D,
                                                          I                         nnz_D,
                                                          const I*                  csr_row_ptr_D,
                                                          const J*                  csr_col_ind_D,
                                                          rocsparse_mat_info        info_C,
                                                          size_t*                   buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    *buffer_size                         = 0;
    info_C->csrgemm_info->buffer_size    = buffer_size[0];
    info_C->csrgemm_info->is_initialized = true;
    return rocsparse_status_success;
}

#define INSTANTIATE(I, J, T)                                            \
    template rocsparse_status rocsparse::csrgemm_scal_buffer_size_core( \
        rocsparse_handle          handle,                               \
        J                         m,                                    \
        J                         n,                                    \
        const T*                  beta,                                 \
        const rocsparse_mat_descr descr_D,                              \
        I                         nnz_D,                                \
        const I*                  csr_row_ptr_D,                        \
        const J*                  csr_col_ind_D,                        \
        rocsparse_mat_info        info_C,                               \
        size_t*                   buffer_size)

INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);

INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);

INSTANTIATE(int32_t, int64_t, float);
INSTANTIATE(int32_t, int64_t, double);
INSTANTIATE(int32_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int64_t, rocsparse_double_complex);

INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);

#undef INSTANTIATE
