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

#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

#include "coomm/atomic/kernel_declarations.h"
#include "coomm_device_atomic.h"

namespace rocsparse
{
#define LAUNCH_COOMMNN_ATOMIC_MAIN_KERNEL(COOMMNN_DIM, WF_SIZE, LOOPS, TRANSB)    \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                           \
        (rocsparse::coommnn_atomic_main<COOMMNN_DIM, WF_SIZE, LOOPS, TRANSB, T>), \
        dim3((nnz - 1) / COOMMNN_DIM + 1, batch_count_C),                         \
        dim3(COOMMNN_DIM),                                                        \
        0,                                                                        \
        handle->stream,                                                           \
        conj_A,                                                                   \
        conj_B,                                                                   \
        main,                                                                     \
        nnz,                                                                      \
        n,                                                                        \
        batch_stride_A,                                                           \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),             \
        coo_row_ind,                                                              \
        coo_col_ind,                                                              \
        coo_val,                                                                  \
        dense_B,                                                                  \
        ldb,                                                                      \
        batch_stride_B,                                                           \
        dense_C,                                                                  \
        ldc,                                                                      \
        batch_stride_C,                                                           \
        order_C,                                                                  \
        descr->base,                                                              \
        handle->pointer_mode == rocsparse_pointer_mode_host);

#define LAUNCH_COOMMNN_ATOMIC_REMAINDER_KERNEL(COOMMNN_DIM, WF_SIZE, TRANSB)    \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                         \
        (rocsparse::coommnn_atomic_remainder<COOMMNN_DIM, WF_SIZE, TRANSB, T>), \
        dim3((nnz - 1) / COOMMNN_DIM + 1, batch_count_C),                       \
        dim3(COOMMNN_DIM),                                                      \
        0,                                                                      \
        handle->stream,                                                         \
        conj_A,                                                                 \
        conj_B,                                                                 \
        main,                                                                   \
        n,                                                                      \
        nnz,                                                                    \
        batch_stride_A,                                                         \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),           \
        coo_row_ind,                                                            \
        coo_col_ind,                                                            \
        coo_val,                                                                \
        dense_B,                                                                \
        ldb,                                                                    \
        batch_stride_B,                                                         \
        dense_C,                                                                \
        ldc,                                                                    \
        batch_stride_C,                                                         \
        order_C,                                                                \
        descr->base,                                                            \
        handle->pointer_mode == rocsparse_pointer_mode_host);

    template <uint32_t BLOCKSIZE,
              uint32_t WF_SIZE,
              bool     TRANSB,
              typename T,
              typename I,
              typename A,
              typename B,
              typename C>
    static rocsparse_status coommnn_atomic_dispatch(rocsparse_handle          handle,
                                                    bool                      conj_A,
                                                    bool                      conj_B,
                                                    I                         m,
                                                    I                         n,
                                                    I                         k,
                                                    int64_t                   nnz,
                                                    I                         batch_count_A,
                                                    int64_t                   batch_stride_A,
                                                    const T*                  alpha_device_host,
                                                    const rocsparse_mat_descr descr,
                                                    const A*                  coo_val,
                                                    const I*                  coo_row_ind,
                                                    const I*                  coo_col_ind,
                                                    const B*                  dense_B,
                                                    int64_t                   ldb,
                                                    I                         batch_count_B,
                                                    int64_t                   batch_stride_B,
                                                    const T*                  beta_device_host,
                                                    C*                        dense_C,
                                                    int64_t                   ldc,
                                                    I                         batch_count_C,
                                                    int64_t                   batch_stride_C,
                                                    rocsparse_order           order_C)
    {
        ROCSPARSE_ROUTINE_TRACE;

        I main      = 0;
        I remainder = n;

        if(n >= 256)
        {
            remainder = n % 256;
            main      = n - remainder;
            LAUNCH_COOMMNN_ATOMIC_MAIN_KERNEL(BLOCKSIZE, WF_SIZE, (256 / WF_SIZE), TRANSB);
        }
        else if(n >= 192)
        {
            remainder = n % 192;
            main      = n - remainder;
            LAUNCH_COOMMNN_ATOMIC_MAIN_KERNEL(BLOCKSIZE, WF_SIZE, (192 / WF_SIZE), TRANSB);
        }
        else if(n >= 128)
        {
            remainder = n % 128;
            main      = n - remainder;
            LAUNCH_COOMMNN_ATOMIC_MAIN_KERNEL(BLOCKSIZE, WF_SIZE, (128 / WF_SIZE), TRANSB);
        }
        else if(n >= 64)
        {
            remainder = n % 64;
            main      = n - remainder;
            LAUNCH_COOMMNN_ATOMIC_MAIN_KERNEL(BLOCKSIZE, WF_SIZE, (64 / WF_SIZE), TRANSB);
        }

        if(remainder > 0)
        {
            if(remainder <= 1)
            {
                LAUNCH_COOMMNN_ATOMIC_REMAINDER_KERNEL(BLOCKSIZE, 1, TRANSB);
            }
            else if(remainder <= 2)
            {
                LAUNCH_COOMMNN_ATOMIC_REMAINDER_KERNEL(BLOCKSIZE, 2, TRANSB);
            }
            else if(remainder <= 4)
            {
                LAUNCH_COOMMNN_ATOMIC_REMAINDER_KERNEL(BLOCKSIZE, 4, TRANSB);
            }
            else if(remainder <= 8)
            {
                LAUNCH_COOMMNN_ATOMIC_REMAINDER_KERNEL(BLOCKSIZE, 8, TRANSB);
            }
            else if(remainder <= 16)
            {
                LAUNCH_COOMMNN_ATOMIC_REMAINDER_KERNEL(BLOCKSIZE, 16, TRANSB);
            }
            else if(remainder <= 32 || WF_SIZE == 32)
            {
                LAUNCH_COOMMNN_ATOMIC_REMAINDER_KERNEL(BLOCKSIZE, 32, TRANSB);
            }
            else
            {
                LAUNCH_COOMMNN_ATOMIC_REMAINDER_KERNEL(BLOCKSIZE, 64, TRANSB);
            }
        }

        return rocsparse_status_success;
    }

    template <typename T, typename I, typename A, typename B, typename C>
    rocsparse_status coomm_template_atomic(rocsparse_handle          handle,
                                           rocsparse_operation       trans_A,
                                           rocsparse_operation       trans_B,
                                           I                         m,
                                           I                         n,
                                           I                         k,
                                           int64_t                   nnz,
                                           I                         batch_count_A,
                                           int64_t                   batch_stride_A,
                                           const T*                  alpha_device_host,
                                           const rocsparse_mat_descr descr,
                                           const A*                  coo_val,
                                           const I*                  coo_row_ind,
                                           const I*                  coo_col_ind,
                                           const B*                  dense_B,
                                           int64_t                   ldb,
                                           I                         batch_count_B,
                                           int64_t                   batch_stride_B,
                                           rocsparse_order           order_B,
                                           const T*                  beta_device_host,
                                           C*                        dense_C,
                                           int64_t                   ldc,
                                           I                         batch_count_C,
                                           int64_t                   batch_stride_C,
                                           rocsparse_order           order_C)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const bool conj_A = (trans_A == rocsparse_operation_conjugate_transpose);
        const bool conj_B = (trans_B == rocsparse_operation_conjugate_transpose);

        // Run different coomm kernels
        if(trans_A == rocsparse_operation_none)
        {
            if((order_B == rocsparse_order_column && trans_B == rocsparse_operation_none)
               || (order_B == rocsparse_order_row && trans_B == rocsparse_operation_transpose)
               || (order_B == rocsparse_order_row
                   && trans_B == rocsparse_operation_conjugate_transpose))
            {
                if(handle->wavefront_size == 32)
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::coommnn_atomic_dispatch<256, 32, false, T>(handle,
                                                                               conj_A,
                                                                               conj_B,
                                                                               m,
                                                                               n,
                                                                               k,
                                                                               nnz,
                                                                               batch_count_A,
                                                                               batch_stride_A,
                                                                               alpha_device_host,
                                                                               descr,
                                                                               coo_val,
                                                                               coo_row_ind,
                                                                               coo_col_ind,
                                                                               dense_B,
                                                                               ldb,
                                                                               batch_count_B,
                                                                               batch_stride_B,
                                                                               beta_device_host,
                                                                               dense_C,
                                                                               ldc,
                                                                               batch_count_C,
                                                                               batch_stride_C,
                                                                               order_C)));
                    return rocsparse_status_success;
                }
                else if(handle->wavefront_size == 64)
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::coommnn_atomic_dispatch<256, 64, false, T>(handle,
                                                                               conj_A,
                                                                               conj_B,
                                                                               m,
                                                                               n,
                                                                               k,
                                                                               nnz,
                                                                               batch_count_A,
                                                                               batch_stride_A,
                                                                               alpha_device_host,
                                                                               descr,
                                                                               coo_val,
                                                                               coo_row_ind,
                                                                               coo_col_ind,
                                                                               dense_B,
                                                                               ldb,
                                                                               batch_count_B,
                                                                               batch_stride_B,
                                                                               beta_device_host,
                                                                               dense_C,
                                                                               ldc,
                                                                               batch_count_C,
                                                                               batch_stride_C,
                                                                               order_C)));
                    return rocsparse_status_success;
                }
                else
                {
                    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
                }
            }
            else if((order_B == rocsparse_order_column
                     && trans_B == rocsparse_operation_conjugate_transpose)
                    || (order_B == rocsparse_order_column
                        && trans_B == rocsparse_operation_transpose)
                    || (order_B == rocsparse_order_row && trans_B == rocsparse_operation_none))
            {
                if(handle->wavefront_size == 32)
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::coommnn_atomic_dispatch<256, 32, true, T>(handle,
                                                                              conj_A,
                                                                              conj_B,
                                                                              m,
                                                                              n,
                                                                              k,
                                                                              nnz,
                                                                              batch_count_A,
                                                                              batch_stride_A,
                                                                              alpha_device_host,
                                                                              descr,
                                                                              coo_val,
                                                                              coo_row_ind,
                                                                              coo_col_ind,
                                                                              dense_B,
                                                                              ldb,
                                                                              batch_count_B,
                                                                              batch_stride_B,
                                                                              beta_device_host,
                                                                              dense_C,
                                                                              ldc,
                                                                              batch_count_C,
                                                                              batch_stride_C,
                                                                              order_C)));
                    return rocsparse_status_success;
                }
                else if(handle->wavefront_size == 64)
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        (rocsparse::coommnn_atomic_dispatch<256, 64, true, T>(handle,
                                                                              conj_A,
                                                                              conj_B,
                                                                              m,
                                                                              n,
                                                                              k,
                                                                              nnz,
                                                                              batch_count_A,
                                                                              batch_stride_A,
                                                                              alpha_device_host,
                                                                              descr,
                                                                              coo_val,
                                                                              coo_row_ind,
                                                                              coo_col_ind,
                                                                              dense_B,
                                                                              ldb,
                                                                              batch_count_B,
                                                                              batch_stride_B,
                                                                              beta_device_host,
                                                                              dense_C,
                                                                              ldc,
                                                                              batch_count_C,
                                                                              batch_stride_C,
                                                                              order_C)));
                    return rocsparse_status_success;
                }
                else
                {
                    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
                }
            }
        }
        else
        {
            if((order_B == rocsparse_order_column && trans_B == rocsparse_operation_none)
               || (order_B == rocsparse_order_row && trans_B == rocsparse_operation_transpose)
               || (order_B == rocsparse_order_row
                   && trans_B == rocsparse_operation_conjugate_transpose))
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::coommtn_atomic_main<256, false, T>),
                    dim3((nnz - 1) / 256 + 1, n, batch_count_C),
                    dim3(256),
                    0,
                    handle->stream,
                    conj_A,
                    conj_B,
                    nnz,
                    n,
                    batch_stride_A,
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                    coo_row_ind,
                    coo_col_ind,
                    coo_val,
                    dense_B,
                    ldb,
                    batch_stride_B,
                    dense_C,
                    ldc,
                    batch_stride_C,
                    order_C,
                    descr->base,
                    handle->pointer_mode == rocsparse_pointer_mode_host);
            }
            else if((order_B == rocsparse_order_column
                     && trans_B == rocsparse_operation_conjugate_transpose)
                    || (order_B == rocsparse_order_column
                        && trans_B == rocsparse_operation_transpose)
                    || (order_B == rocsparse_order_row && trans_B == rocsparse_operation_none))
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::coommtn_atomic_main<256, true, T>),
                    dim3((nnz - 1) / 256 + 1, n, batch_count_C),
                    dim3(256),
                    0,
                    handle->stream,
                    conj_A,
                    conj_B,
                    nnz,
                    n,
                    batch_stride_A,
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                    coo_row_ind,
                    coo_col_ind,
                    coo_val,
                    dense_B,
                    ldb,
                    batch_stride_B,
                    dense_C,
                    ldc,
                    batch_stride_C,
                    order_C,
                    descr->base,
                    handle->pointer_mode == rocsparse_pointer_mode_host);
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
        }
        return rocsparse_status_success;
    }
}

#define INSTANTIATE(TTYPE, ITYPE, ATYPE, BTYPE, CTYPE)                                             \
    template rocsparse_status rocsparse::coomm_template_atomic(rocsparse_handle    handle,         \
                                                               rocsparse_operation trans_A,        \
                                                               rocsparse_operation trans_B,        \
                                                               ITYPE               m,              \
                                                               ITYPE               n,              \
                                                               ITYPE               k,              \
                                                               int64_t             nnz,            \
                                                               ITYPE               batch_count_A,  \
                                                               int64_t             batch_stride_A, \
                                                               const TTYPE* alpha_device_host,     \
                                                               const rocsparse_mat_descr descr,    \
                                                               const ATYPE*              coo_val,  \
                                                               const ITYPE*    coo_row_ind,        \
                                                               const ITYPE*    coo_col_ind,        \
                                                               const BTYPE*    dense_B,            \
                                                               int64_t         ldb,                \
                                                               ITYPE           batch_count_B,      \
                                                               int64_t         batch_stride_B,     \
                                                               rocsparse_order order_B,            \
                                                               const TTYPE*    beta_device_host,   \
                                                               CTYPE*          dense_C,            \
                                                               int64_t         ldc,                \
                                                               ITYPE           batch_count_C,      \
                                                               int64_t         batch_stride_C,     \
                                                               rocsparse_order order_C);

// Uniform precisions
INSTANTIATE(float, int32_t, float, float, float);
INSTANTIATE(float, int64_t, float, float, float);
INSTANTIATE(double, int32_t, double, double, double);
INSTANTIATE(double, int64_t, double, double, double);
INSTANTIATE(rocsparse_float_complex,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);

// Mixed Precisions
INSTANTIATE(float, int32_t, _Float16, _Float16, float);
INSTANTIATE(float, int64_t, _Float16, _Float16, float);
INSTANTIATE(float, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(float, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(int32_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE(int32_t, int64_t, int8_t, int8_t, int32_t);
INSTANTIATE(float, int32_t, int8_t, int8_t, float);
INSTANTIATE(float, int64_t, int8_t, int8_t, float);

#undef INSTANTIATE
