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

#include "rocsparse_bsrgemm_scal.hpp"
#include "../conversion/rocsparse_identity.hpp"
#include "bsrgemm_device.h"
#include "csrgemm_device.h"
#include "internal/extra/rocsparse_bsrgemm.h"
#include "rocsparse_bsrgemm.hpp"
#include "rocsparse_bsrgemm_calc.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_csrgemm.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void bsrgemm_copy_scale(I size,
                            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                            const T* __restrict__ in,
                            T* __restrict__ out,
                            bool is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);
        rocsparse::bsrgemm_copy_scale_device<BLOCKSIZE>(size, beta, in, out);
    }
}

rocsparse_status rocsparse::bsrgemm_scal_quickreturn(rocsparse_handle          handle,
                                                     int64_t                   mb,
                                                     int64_t                   nb,
                                                     int64_t                   block_dim,
                                                     const void*               beta,
                                                     const rocsparse_mat_descr descr_D,
                                                     int64_t                   nnzb_D,
                                                     const void*               bsr_val_D,
                                                     const void*               bsr_row_ptr_D,
                                                     const void*               bsr_col_ind_D,
                                                     const rocsparse_mat_descr descr_C,
                                                     void*                     bsr_val_C,
                                                     const void*               bsr_row_ptr_C,
                                                     void*                     bsr_col_ind_C,
                                                     const rocsparse_mat_info  info_C,
                                                     void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(mb == 0 || nb == 0 || nnzb_D == 0)
    {
        return rocsparse_status_success;
    }
    return rocsparse_status_continue;
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse::bsrgemm_scal_core(rocsparse_handle          handle,
                                              J                         mb,
                                              J                         nb,
                                              J                         block_dim,
                                              const T*                  beta,
                                              const rocsparse_mat_descr descr_D,
                                              I                         nnzb_D,
                                              const T*                  bsr_val_D,
                                              const I*                  bsr_row_ptr_D,
                                              const J*                  bsr_col_ind_D,
                                              const rocsparse_mat_descr descr_C,
                                              T*                        bsr_val_C,
                                              const I*                  bsr_row_ptr_C,
                                              J*                        bsr_col_ind_C,
                                              const rocsparse_mat_info  info_C,
                                              void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Stream
    hipStream_t stream = handle->stream;

#define BSRGEMM_DIM 1024
    // Copy column entries, if D != C
    if(bsr_col_ind_C != bsr_col_ind_D)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::bsrgemm_copy<BSRGEMM_DIM>),
                                           dim3((nnzb_D - 1) / BSRGEMM_DIM + 1),
                                           dim3(BSRGEMM_DIM),
                                           0,
                                           stream,
                                           nnzb_D,
                                           bsr_col_ind_D,
                                           bsr_col_ind_C,
                                           descr_D->base,
                                           descr_C->base);
    }

    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::bsrgemm_copy_scale<BSRGEMM_DIM>),
                                       dim3((block_dim * block_dim * nnzb_D - 1) / BSRGEMM_DIM + 1),
                                       dim3(BSRGEMM_DIM),
                                       0,
                                       stream,
                                       block_dim * block_dim * nnzb_D,
                                       ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta),
                                       bsr_val_D,
                                       bsr_val_C,
                                       handle->pointer_mode == rocsparse_pointer_mode_host);
#undef BSRGEMM_DIM

    return rocsparse_status_success;
}

#define INSTANTIATE(I, J, T)                                                                        \
    template rocsparse_status rocsparse::bsrgemm_scal_core(rocsparse_handle          handle,        \
                                                           J                         mb,            \
                                                           J                         nb,            \
                                                           J                         block_dim,     \
                                                           const T*                  beta,          \
                                                           const rocsparse_mat_descr descr_D,       \
                                                           I                         nnzb_D,        \
                                                           const T*                  bsr_val_D,     \
                                                           const I*                  bsr_row_ptr_D, \
                                                           const J*                  bsr_col_ind_D, \
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
