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

#include "../conversion/rocsparse_convert_array.hpp"
#include "internal/extra/rocsparse_csrgeam.h"
#include "rocsparse_common.h"
#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_csrgeam.hpp"
#include "rocsparse_utility.hpp"

#include "csrgeam_device.h"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename I, typename J, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void csrgeam_fill_multipass_kernel(int64_t m,
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
                                       J* __restrict__ csr_col_ind_C,
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
        rocsparse::csrgeam_fill_multipass_device<BLOCKSIZE, WFSIZE>(m,
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
                                                                    csr_col_ind_C,
                                                                    csr_val_C,
                                                                    idx_base_A,
                                                                    idx_base_B,
                                                                    idx_base_C);
    }

    template <typename I, typename J, typename T>
    static rocsparse_status csrgeam_dispatch(rocsparse_handle          handle,
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
                                             J*                        csr_col_ind_C)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Stream
        hipStream_t stream = handle->stream;

        // Pointer mode device
#define CSRGEAM_DIM 256
        if(handle->wavefront_size == 32)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::csrgeam_fill_multipass_kernel<CSRGEAM_DIM, 32>),
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
                csr_col_ind_C,
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
                (rocsparse::csrgeam_fill_multipass_kernel<CSRGEAM_DIM, 64>),
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
                csr_col_ind_C,
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
    static rocsparse_status csrgeam_core(rocsparse_handle          handle,
                                         rocsparse_operation       trans_A,
                                         rocsparse_operation       trans_B,
                                         int64_t                   m,
                                         int64_t                   n,
                                         const T*                  alpha,
                                         const rocsparse_mat_descr descr_A,
                                         int64_t                   nnz_A,
                                         const T*                  csr_val_A,
                                         const I*                  csr_row_ptr_A,
                                         const J*                  csr_col_ind_A,
                                         const T*                  beta,
                                         const rocsparse_mat_descr descr_B,
                                         int64_t                   nnz_B,
                                         const T*                  csr_val_B,
                                         const I*                  csr_row_ptr_B,
                                         const J*                  csr_col_ind_B,
                                         const rocsparse_mat_descr descr_C,
                                         T*                        csr_val_C,
                                         const I*                  csr_row_ptr_C,
                                         J*                        csr_col_ind_C,
                                         void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgeam_dispatch(handle,
                                                              trans_A,
                                                              trans_B,
                                                              m,
                                                              n,
                                                              alpha,
                                                              descr_A,
                                                              nnz_A,
                                                              csr_val_A,
                                                              csr_row_ptr_A,
                                                              csr_col_ind_A,
                                                              beta,
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

    template <typename I>
    static rocsparse_status csrgeam_quickreturn(rocsparse_handle          handle,
                                                rocsparse_operation       trans_A,
                                                rocsparse_operation       trans_B,
                                                int64_t                   m,
                                                int64_t                   n,
                                                const void*               alpha,
                                                const rocsparse_mat_descr descr_A,
                                                int64_t                   nnz_A,
                                                const void*               csr_val_A,
                                                const void*               csr_row_ptr_A,
                                                const void*               csr_col_ind_A,
                                                const void*               beta,
                                                const rocsparse_mat_descr descr_B,
                                                int64_t                   nnz_B,
                                                const void*               csr_val_B,
                                                const void*               csr_row_ptr_B,
                                                const void*               csr_col_ind_B,
                                                const rocsparse_mat_descr descr_C,
                                                void*                     csr_val_C,
                                                const I*                  csr_row_ptr_C,
                                                void*                     csr_col_ind_C,
                                                void*                     temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(m == 0 || n == 0 || (nnz_A == 0 && nnz_B == 0))
        {
            return rocsparse_status_success;
        }

        return rocsparse_status_continue;
    }

    template <typename I>
    static rocsparse_status csrgeam_checkarg(rocsparse_handle          handle, //0
                                             int64_t                   m, //1
                                             int64_t                   n, //2
                                             const void*               alpha, //3
                                             const rocsparse_mat_descr descr_A, //4
                                             int64_t                   nnz_A, //5
                                             const void*               csr_val_A, //6
                                             const void*               csr_row_ptr_A, //7
                                             const void*               csr_col_ind_A, //8
                                             const void*               beta, //9
                                             const rocsparse_mat_descr descr_B, //10
                                             int64_t                   nnz_B, //11
                                             const void*               csr_val_B, //12
                                             const void*               csr_row_ptr_B, //13
                                             const void*               csr_col_ind_B, //14
                                             const rocsparse_mat_descr descr_C, //15
                                             void*                     csr_val_C, //16
                                             const I*                  csr_row_ptr_C, //17
                                             void*                     csr_col_ind_C) //18
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_POINTER(3, alpha);
        ROCSPARSE_CHECKARG_POINTER(9, beta);
        ROCSPARSE_CHECKARG_POINTER(4, descr_A);
        ROCSPARSE_CHECKARG_POINTER(10, descr_B);
        ROCSPARSE_CHECKARG_POINTER(15, descr_C);

        ROCSPARSE_CHECKARG(4,
                           descr_A,
                           (descr_A->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(10,
                           descr_B,
                           (descr_B->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(15,
                           descr_C,
                           (descr_C->type != rocsparse_matrix_type_general),
                           rocsparse_status_not_implemented);

        ROCSPARSE_CHECKARG(4,
                           descr_A,
                           (descr_A->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);
        ROCSPARSE_CHECKARG(10,
                           descr_B,
                           (descr_B->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);
        ROCSPARSE_CHECKARG(15,
                           descr_C,
                           (descr_C->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);

        ROCSPARSE_CHECKARG_SIZE(1, m);
        ROCSPARSE_CHECKARG_SIZE(2, n);
        ROCSPARSE_CHECKARG_SIZE(5, nnz_A);
        ROCSPARSE_CHECKARG_SIZE(11, nnz_B);

        void*                  temp_buffer = nullptr;
        const rocsparse_status status      = rocsparse::csrgeam_quickreturn(handle,
                                                                       rocsparse_operation_none,
                                                                       rocsparse_operation_none,
                                                                       m,
                                                                       n,
                                                                       alpha,
                                                                       descr_A,
                                                                       nnz_A,
                                                                       csr_val_A,
                                                                       csr_row_ptr_A,
                                                                       csr_col_ind_A,
                                                                       beta,
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

        ROCSPARSE_CHECKARG_ARRAY(7, m, csr_row_ptr_A);
        ROCSPARSE_CHECKARG_ARRAY(13, m, csr_row_ptr_B);
        ROCSPARSE_CHECKARG_ARRAY(17, m, csr_row_ptr_C);

        ROCSPARSE_CHECKARG_ARRAY(6, nnz_A, csr_val_A);
        ROCSPARSE_CHECKARG_ARRAY(8, nnz_A, csr_col_ind_A);

        ROCSPARSE_CHECKARG_ARRAY(12, nnz_B, csr_val_B);
        ROCSPARSE_CHECKARG_ARRAY(14, nnz_B, csr_col_ind_B);

        if(csr_col_ind_C == nullptr || csr_val_C == nullptr)
        {
            I start = 0;
            I end   = 0;
            if(csr_row_ptr_C != nullptr)
            {
                RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                    &end, &csr_row_ptr_C[m], sizeof(I), hipMemcpyDeviceToHost, handle->stream));
                RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                    &start, &csr_row_ptr_C[0], sizeof(I), hipMemcpyDeviceToHost, handle->stream));
                RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
            }
            const I nnz_C = (end - start);
            ROCSPARSE_CHECKARG_ARRAY(16, nnz_C, csr_val_C);
            ROCSPARSE_CHECKARG_ARRAY(18, nnz_C, csr_col_ind_C);
        }

        return rocsparse_status_continue;
    }

    template <typename T>
    static rocsparse_status csrgeam_impl(rocsparse_handle          handle,
                                         rocsparse_int             m,
                                         rocsparse_int             n,
                                         const T*                  alpha,
                                         const rocsparse_mat_descr descr_A,
                                         rocsparse_int             nnz_A,
                                         const T*                  csr_val_A,
                                         const rocsparse_int*      csr_row_ptr_A,
                                         const rocsparse_int*      csr_col_ind_A,
                                         const T*                  beta,
                                         const rocsparse_mat_descr descr_B,
                                         rocsparse_int             nnz_B,
                                         const T*                  csr_val_B,
                                         const rocsparse_int*      csr_row_ptr_B,
                                         const rocsparse_int*      csr_col_ind_B,
                                         const rocsparse_mat_descr descr_C,
                                         T*                        csr_val_C,
                                         const rocsparse_int*      csr_row_ptr_C,
                                         rocsparse_int*            csr_col_ind_C)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse::log_trace("rocsparse_Xcsrgeam",
                             handle,
                             m,
                             n,
                             (const void*&)alpha,
                             (const void*&)descr_A,
                             nnz_A,
                             (const void*&)csr_val_A,
                             (const void*&)csr_row_ptr_A,
                             (const void*&)csr_col_ind_A,
                             (const void*&)beta,
                             (const void*&)descr_B,
                             nnz_B,
                             (const void*&)csr_val_B,
                             (const void*&)csr_row_ptr_B,
                             (const void*&)csr_col_ind_B,
                             (const void*&)descr_C,
                             (const void*&)csr_val_C,
                             (const void*&)csr_row_ptr_C,
                             (const void*&)csr_col_ind_C);

        const rocsparse_status status = rocsparse::csrgeam_checkarg(handle,
                                                                    m,
                                                                    n,
                                                                    alpha,
                                                                    descr_A,
                                                                    nnz_A,
                                                                    csr_val_A,
                                                                    csr_row_ptr_A,
                                                                    csr_col_ind_A,
                                                                    beta,
                                                                    descr_B,
                                                                    nnz_B,
                                                                    csr_val_B,
                                                                    csr_row_ptr_B,
                                                                    csr_col_ind_B,
                                                                    descr_C,
                                                                    csr_val_C,
                                                                    csr_row_ptr_C,
                                                                    csr_col_ind_C);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        void* temp_buffer = nullptr;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgeam_core(handle,
                                                          rocsparse_operation_none,
                                                          rocsparse_operation_none,
                                                          m,
                                                          n,
                                                          alpha,
                                                          descr_A,
                                                          nnz_A,
                                                          csr_val_A,
                                                          csr_row_ptr_A,
                                                          csr_col_ind_A,
                                                          beta,
                                                          descr_B,
                                                          nnz_B,
                                                          csr_val_B,
                                                          csr_row_ptr_B,
                                                          csr_col_ind_B,
                                                          descr_C,
                                                          csr_val_C,
                                                          csr_row_ptr_C,
                                                          csr_col_ind_C,
                                                          temp_buffer));
        return rocsparse_status_success;
    }
}

template <typename T, typename I, typename J>
rocsparse_status rocsparse::csrgeam_template(rocsparse_handle          handle,
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
                                             void*                     csr_col_ind_C,
                                             void*                     temp_buffer)
{
    const rocsparse_status status = rocsparse::csrgeam_quickreturn(handle,
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

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgeam_core(handle,
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
                                                      (J*)csr_col_ind_C,
                                                      temp_buffer));
    return rocsparse_status_success;
}

#define INSTANTIATE(T, I, J)                                        \
    template rocsparse_status rocsparse::csrgeam_template<T, I, J>( \
        rocsparse_handle          handle,                           \
        rocsparse_operation       trans_A,                          \
        rocsparse_operation       trans_B,                          \
        int64_t                   m,                                \
        int64_t                   n,                                \
        const void*               alpha_device_host,                \
        const rocsparse_mat_descr descr_A,                          \
        int64_t                   nnz_A,                            \
        const void*               csr_val_A,                        \
        const void*               csr_row_ptr_A,                    \
        const void*               csr_col_ind_A,                    \
        const void*               beta_device_host,                 \
        const rocsparse_mat_descr descr_B,                          \
        int64_t                   nnz_B,                            \
        const void*               csr_val_B,                        \
        const void*               csr_row_ptr_B,                    \
        const void*               csr_col_ind_B,                    \
        const rocsparse_mat_descr descr_C,                          \
        void*                     csr_val_C,                        \
        const void*               csr_row_ptr_C,                    \
        void*                     csr_col_ind_C,                    \
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

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#define C_IMPL(NAME, TYPE)                                                    \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,        \
                                     rocsparse_int             m,             \
                                     rocsparse_int             n,             \
                                     const TYPE*               alpha,         \
                                     const rocsparse_mat_descr descr_A,       \
                                     rocsparse_int             nnz_A,         \
                                     const TYPE*               csr_val_A,     \
                                     const rocsparse_int*      csr_row_ptr_A, \
                                     const rocsparse_int*      csr_col_ind_A, \
                                     const TYPE*               beta,          \
                                     const rocsparse_mat_descr descr_B,       \
                                     rocsparse_int             nnz_B,         \
                                     const TYPE*               csr_val_B,     \
                                     const rocsparse_int*      csr_row_ptr_B, \
                                     const rocsparse_int*      csr_col_ind_B, \
                                     const rocsparse_mat_descr descr_C,       \
                                     TYPE*                     csr_val_C,     \
                                     const rocsparse_int*      csr_row_ptr_C, \
                                     rocsparse_int*            csr_col_ind_C) \
    try                                                                       \
    {                                                                         \
        ROCSPARSE_ROUTINE_TRACE;                                              \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrgeam_impl(handle,             \
                                                          m,                  \
                                                          n,                  \
                                                          alpha,              \
                                                          descr_A,            \
                                                          nnz_A,              \
                                                          csr_val_A,          \
                                                          csr_row_ptr_A,      \
                                                          csr_col_ind_A,      \
                                                          beta,               \
                                                          descr_B,            \
                                                          nnz_B,              \
                                                          csr_val_B,          \
                                                          csr_row_ptr_B,      \
                                                          csr_col_ind_B,      \
                                                          descr_C,            \
                                                          csr_val_C,          \
                                                          csr_row_ptr_C,      \
                                                          csr_col_ind_C));    \
        return rocsparse_status_success;                                      \
    }                                                                         \
    catch(...)                                                                \
    {                                                                         \
        RETURN_ROCSPARSE_EXCEPTION();                                         \
    }

C_IMPL(rocsparse_scsrgeam, float);
C_IMPL(rocsparse_dcsrgeam, double);
C_IMPL(rocsparse_ccsrgeam, rocsparse_float_complex);
C_IMPL(rocsparse_zcsrgeam, rocsparse_double_complex);

#undef C_IMPL
