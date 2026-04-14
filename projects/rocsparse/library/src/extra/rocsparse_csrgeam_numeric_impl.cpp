/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_common.h"
#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_csrgeam_numeric.hpp"

#include "csrgeam_numeric_device.h"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename I, typename J, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgeam_fill_numeric_multipass_kernel(int64_t m,
                                               int64_t n,
                                               ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                               const I* __restrict__ csr_row_ptr_A,
                                               const J* __restrict__ csr_col_ind_A,
                                               const T* __restrict__ csr_val_A,
                                               ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                               const I* __restrict__ csr_row_ptr_B,
                                               const J* __restrict__ csr_col_ind_B,
                                               const T* __restrict__ csr_val_B,
                                               const I* __restrict__ csr_row_ptr_C,
                                               T* __restrict__ csr_val_C,
                                               rocsparse_index_base idx_base_A,
                                               rocsparse_index_base idx_base_B,
                                               rocsparse_index_base idx_base_C,
                                               bool                 alpha_mul,
                                               bool                 beta_mul,
                                               bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(alpha_mul, alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET_IF(beta_mul, beta);
        rocsparse::csrgeam_fill_numeric_multipass_device<BLOCKSIZE, WFSIZE>(m,
                                                                            n,
                                                                            alpha,
                                                                            csr_row_ptr_A,
                                                                            csr_col_ind_A,
                                                                            csr_val_A,
                                                                            beta,
                                                                            csr_row_ptr_B,
                                                                            csr_col_ind_B,
                                                                            csr_val_B,
                                                                            csr_row_ptr_C,
                                                                            csr_val_C,
                                                                            idx_base_A,
                                                                            idx_base_B,
                                                                            idx_base_C);
    }
}

namespace rocsparse
{
    static rocsparse_status csrgeam_numeric_quickreturn(rocsparse_handle          handle,
                                                        rocsparse_operation       trans_A,
                                                        rocsparse_operation       trans_B,
                                                        int64_t                   m,
                                                        int64_t                   n,
                                                        const void*               alpha_device_host,
                                                        const rocsparse_mat_descr descr_A,
                                                        int64_t                   nnz_A,
                                                        const void*               csr_val_A,
                                                        const void*               csr_row_ptr_A,
                                                        const void*               csr_col_ind_A,
                                                        const void*               beta_device_host,
                                                        const rocsparse_mat_descr descr_B,
                                                        int64_t                   nnz_B,
                                                        const void*               csr_val_B,
                                                        const void*               csr_row_ptr_B,
                                                        const void*               csr_col_ind_B,
                                                        const rocsparse_mat_descr descr_C,
                                                        void*                     csr_val_C,
                                                        const void*               csr_row_ptr_C,
                                                        const void*               csr_col_ind_C,
                                                        void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(m == 0 || n == 0 || (nnz_A == 0 && nnz_B == 0))
        {
            return rocsparse_status_success;
        }

        return rocsparse_status_continue;
    }

    template <typename I, typename J, typename T>
    static rocsparse_status csrgeam_numeric_dispatch(rocsparse_handle          handle,
                                                     rocsparse_operation       trans_A,
                                                     rocsparse_operation       trans_B,
                                                     int64_t                   m,
                                                     int64_t                   n,
                                                     const T*                  alpha_device_host,
                                                     const rocsparse_mat_descr descr_A,
                                                     int64_t                   nnz_A,
                                                     const T*                  csr_val_A,
                                                     const I*                  csr_row_ptr_A,
                                                     const J*                  csr_col_ind_A,
                                                     const T*                  beta_device_host,
                                                     const rocsparse_mat_descr descr_B,
                                                     int64_t                   nnz_B,
                                                     const T*                  csr_val_B,
                                                     const I*                  csr_row_ptr_B,
                                                     const J*                  csr_col_ind_B,
                                                     const rocsparse_mat_descr descr_C,
                                                     T*                        csr_val_C,
                                                     const I*                  csr_row_ptr_C,
                                                     const J*                  csr_col_ind_C)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Stream
        hipStream_t stream = handle->stream;

        // Pointer mode device
#define CSRGEAM_DIM 256
        if(handle->wavefront_size == 32)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::csrgeam_fill_numeric_multipass_kernel<CSRGEAM_DIM, 32>),
                dim3((m - 1) / (CSRGEAM_DIM / 32) + 1),
                dim3(CSRGEAM_DIM),
                0,
                stream,
                m,
                n,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, alpha_device_host),
                csr_row_ptr_A,
                csr_col_ind_A,
                csr_val_A,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, beta_device_host),
                csr_row_ptr_B,
                csr_col_ind_B,
                csr_val_B,
                csr_row_ptr_C,
                csr_val_C,
                descr_A->base,
                descr_B->base,
                descr_C->base,
                (alpha_device_host != nullptr),
                (beta_device_host != nullptr),
                handle->pointer_mode == rocsparse_pointer_mode_host);
        }
        else
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::csrgeam_fill_numeric_multipass_kernel<CSRGEAM_DIM, 64>),
                dim3((m - 1) / (CSRGEAM_DIM / 64) + 1),
                dim3(CSRGEAM_DIM),
                0,
                stream,
                m,
                n,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, alpha_device_host),
                csr_row_ptr_A,
                csr_col_ind_A,
                csr_val_A,
                ROCSPARSE_DEVICE_HOST_SCALAR_PERMISSIVE_ARGS(handle, beta_device_host),
                csr_row_ptr_B,
                csr_col_ind_B,
                csr_val_B,
                csr_row_ptr_C,
                csr_val_C,
                descr_A->base,
                descr_B->base,
                descr_C->base,
                (alpha_device_host != nullptr),
                (beta_device_host != nullptr),
                handle->pointer_mode == rocsparse_pointer_mode_host);
        }
#undef CSRGEAM_DIM

        return rocsparse_status_success;
    }

    template <typename I, typename J, typename T>
    static rocsparse_status csrgeam_numeric_core(rocsparse_handle          handle,
                                                 rocsparse_operation       trans_A,
                                                 rocsparse_operation       trans_B,
                                                 int64_t                   m,
                                                 int64_t                   n,
                                                 const T*                  alpha_device_host,
                                                 const rocsparse_mat_descr descr_A,
                                                 int64_t                   nnz_A,
                                                 const T*                  csr_val_A,
                                                 const I*                  csr_row_ptr_A,
                                                 const J*                  csr_col_ind_A,
                                                 const T*                  beta_device_host,
                                                 const rocsparse_mat_descr descr_B,
                                                 int64_t                   nnz_B,
                                                 const T*                  csr_val_B,
                                                 const I*                  csr_row_ptr_B,
                                                 const J*                  csr_col_ind_B,
                                                 const rocsparse_mat_descr descr_C,
                                                 T*                        csr_val_C,
                                                 const I*                  csr_row_ptr_C,
                                                 const J*                  csr_col_ind_C,
                                                 void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgeam_numeric_dispatch(handle,
                                                                      trans_A,
                                                                      trans_B,
                                                                      m,
                                                                      n,
                                                                      alpha_device_host,
                                                                      descr_A,
                                                                      nnz_A,
                                                                      csr_val_A,
                                                                      csr_row_ptr_A,
                                                                      csr_col_ind_A,
                                                                      beta_device_host,
                                                                      descr_B,
                                                                      nnz_B,
                                                                      csr_val_B,
                                                                      csr_row_ptr_B,
                                                                      csr_col_ind_B,
                                                                      descr_C,
                                                                      csr_val_C,
                                                                      csr_row_ptr_C,
                                                                      csr_col_ind_C));
        return rocsparse_status_success;
    }
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse::csrgeam_numeric_template(rocsparse_handle          handle,
                                                     rocsparse_operation       trans_A,
                                                     rocsparse_operation       trans_B,
                                                     int64_t                   m,
                                                     int64_t                   n,
                                                     const void*               alpha_device_host,
                                                     const rocsparse_mat_descr descr_A,
                                                     int64_t                   nnz_A,
                                                     const void*               csr_val_A,
                                                     const void*               csr_row_ptr_A,
                                                     const void*               csr_col_ind_A,
                                                     const void*               beta_device_host,
                                                     const rocsparse_mat_descr descr_B,
                                                     int64_t                   nnz_B,
                                                     const void*               csr_val_B,
                                                     const void*               csr_row_ptr_B,
                                                     const void*               csr_col_ind_B,
                                                     const rocsparse_mat_descr descr_C,
                                                     void*                     csr_val_C,
                                                     const void*               csr_row_ptr_C,
                                                     const void*               csr_col_ind_C,
                                                     void*                     temp_buffer)
{
    const rocsparse_status status = rocsparse::csrgeam_numeric_quickreturn(handle,
                                                                           trans_A,
                                                                           trans_B,
                                                                           m,
                                                                           n,
                                                                           alpha_device_host,
                                                                           descr_A,
                                                                           nnz_A,
                                                                           csr_val_A,
                                                                           csr_row_ptr_A,
                                                                           csr_col_ind_A,
                                                                           beta_device_host,
                                                                           descr_B,
                                                                           nnz_B,
                                                                           csr_val_B,
                                                                           csr_row_ptr_B,
                                                                           csr_col_ind_B,
                                                                           descr_C,
                                                                           csr_val_C,
                                                                           csr_row_ptr_C,
                                                                           csr_col_ind_C,
                                                                           temp_buffer);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgeam_numeric_core(handle,
                                                              trans_A,
                                                              trans_B,
                                                              m,
                                                              n,
                                                              (const T*)alpha_device_host,
                                                              descr_A,
                                                              nnz_A,
                                                              (const T*)csr_val_A,
                                                              (const I*)csr_row_ptr_A,
                                                              (const J*)csr_col_ind_A,
                                                              (const T*)beta_device_host,
                                                              descr_B,
                                                              nnz_B,
                                                              (const T*)csr_val_B,
                                                              (const I*)csr_row_ptr_B,
                                                              (const J*)csr_col_ind_B,
                                                              descr_C,
                                                              (T*)csr_val_C,
                                                              (const I*)csr_row_ptr_C,
                                                              (const J*)csr_col_ind_C,
                                                              temp_buffer));
    return rocsparse_status_success;
}

#define INSTANTIATE(T, I, J)                                                \
    template rocsparse_status rocsparse::csrgeam_numeric_template<T, I, J>( \
        rocsparse_handle          handle,                                   \
        rocsparse_operation       trans_A,                                  \
        rocsparse_operation       trans_B,                                  \
        int64_t                   m,                                        \
        int64_t                   n,                                        \
        const void*               alpha_device_host,                        \
        const rocsparse_mat_descr descr_A,                                  \
        int64_t                   nnz_A,                                    \
        const void*               csr_val_A,                                \
        const void*               csr_row_ptr_A,                            \
        const void*               csr_col_ind_A,                            \
        const void*               beta_device_host,                         \
        const rocsparse_mat_descr descr_B,                                  \
        int64_t                   nnz_B,                                    \
        const void*               csr_val_B,                                \
        const void*               csr_row_ptr_B,                            \
        const void*               csr_col_ind_B,                            \
        const rocsparse_mat_descr descr_C,                                  \
        void*                     csr_val_C,                                \
        const void*               csr_row_ptr_C,                            \
        const void*               csr_col_ind_C,                            \
        void*                     temp_buffer);

INSTANTIATE(float, int32_t, int32_t);
INSTANTIATE(float, int64_t, int32_t);
INSTANTIATE(float, int64_t, int64_t);
INSTANTIATE(double, int32_t, int32_t);
INSTANTIATE(double, int64_t, int32_t);
INSTANTIATE(double, int64_t, int64_t);
INSTANTIATE(rocsparse_float_complex, int32_t, int32_t);
INSTANTIATE(rocsparse_float_complex, int64_t, int32_t);
INSTANTIATE(rocsparse_float_complex, int64_t, int64_t);
INSTANTIATE(rocsparse_double_complex, int32_t, int32_t);
INSTANTIATE(rocsparse_double_complex, int64_t, int32_t);
INSTANTIATE(rocsparse_double_complex, int64_t, int64_t);
#undef INSTANTIATE
