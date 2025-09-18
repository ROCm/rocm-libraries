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

#include "rocsparse_csrgemm_scal.hpp"
#include "../conversion/rocsparse_identity.hpp"
#include "csrgemm_device.h"
#include "internal/extra/rocsparse_csrgemm.h"
#include "rocsparse_control.hpp"
#include "rocsparse_csrgemm.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::csrgemm_scal_quickreturn(rocsparse_handle          handle,
                                                     int64_t                   m,
                                                     int64_t                   n,
                                                     const void*               beta,
                                                     const rocsparse_mat_descr descr_D,
                                                     int64_t                   nnz_D,
                                                     const void*               csr_val_D,
                                                     const void*               csr_row_ptr_D,
                                                     const void*               csr_col_ind_D,
                                                     const rocsparse_mat_descr descr_C,
                                                     void*                     csr_val_C,
                                                     const void*               csr_row_ptr_C,
                                                     void*                     csr_col_ind_C,
                                                     const rocsparse_mat_info  info_C,
                                                     void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(m == 0 || n == 0 || nnz_D == 0)
    {
        return rocsparse_status_success;
    }

    return rocsparse_status_continue;
}

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgemm_copy_scale(I size,
                            ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                            const T* __restrict__ in,
                            T* __restrict__ out,
                            bool is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        rocsparse::csrgemm_copy_scale_device<BLOCKSIZE>(size, alpha, in, out);
    }
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse::csrgemm_scal_core(rocsparse_handle          handle,
                                              J                         m,
                                              J                         n,
                                              const T*                  beta,
                                              const rocsparse_mat_descr descr_D,
                                              I                         nnz_D,
                                              const T*                  csr_val_D,
                                              const I*                  csr_row_ptr_D,
                                              const J*                  csr_col_ind_D,
                                              const rocsparse_mat_descr descr_C,
                                              T*                        csr_val_C,
                                              const I*                  csr_row_ptr_C,
                                              J*                        csr_col_ind_C,
                                              const rocsparse_mat_info  info_C,
                                              void*                     temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Stream
    hipStream_t stream = handle->stream;

#define CSRGEMM_DIM 1024
    dim3 csrgemm_blocks((nnz_D - 1) / CSRGEMM_DIM + 1);
    dim3 csrgemm_threads(CSRGEMM_DIM);

    // Copy column entries, if D != C
    if(csr_col_ind_C != csr_col_ind_D)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrgemm_copy<CSRGEMM_DIM>),
                                           csrgemm_blocks,
                                           csrgemm_threads,
                                           0,
                                           stream,
                                           nnz_D,
                                           csr_col_ind_D,
                                           csr_col_ind_C,
                                           descr_D->base,
                                           descr_C->base);
    }

    // Scale the matrix
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csrgemm_copy_scale<CSRGEMM_DIM>),
                                       csrgemm_blocks,
                                       csrgemm_threads,
                                       0,
                                       stream,
                                       nnz_D,
                                       ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta),
                                       csr_val_D,
                                       csr_val_C,
                                       handle->pointer_mode == rocsparse_pointer_mode_host);
#undef CSRGEMM_DIM

    return rocsparse_status_success;
}

#define INSTANTIATE(I, J, T)                                                                        \
    template rocsparse_status rocsparse::csrgemm_scal_core(rocsparse_handle          handle,        \
                                                           J                         m,             \
                                                           J                         n,             \
                                                           const T*                  beta,          \
                                                           const rocsparse_mat_descr descr_D,       \
                                                           I                         nnz_D,         \
                                                           const T*                  csr_val_D,     \
                                                           const I*                  csr_row_ptr_D, \
                                                           const J*                  csr_col_ind_D, \
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

INSTANTIATE(int32_t, int64_t, float);
INSTANTIATE(int32_t, int64_t, double);
INSTANTIATE(int32_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int64_t, rocsparse_double_complex);

INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);

#undef INSTANTIATE
