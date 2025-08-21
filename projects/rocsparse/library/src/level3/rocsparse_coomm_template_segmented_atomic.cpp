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

#include "coomm/segmented_atomic/kernel_declarations.h"
#include "coomm_device_segmented_atomic.h"

namespace rocsparse
{
#define LAUNCH_COOMMNN_SEGMENTED_ATOMIC_MAIN_KERNEL(WF_SIZE, LOOPS, COLS, NT) \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                       \
        (rocsparse::coommnn_segmented_atomic<WF_SIZE, LOOPS, COLS, NT, T>),   \
        dim3(nblocks, (main - 1) / COLS + 1, batch_count_C),                  \
        dim3(WF_SIZE),                                                        \
        0,                                                                    \
        stream,                                                               \
        trans_B,                                                              \
        nnz,                                                                  \
        (I)0,                                                                 \
        batch_stride_A,                                                       \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),         \
        coo_row_ind,                                                          \
        coo_col_ind,                                                          \
        coo_val,                                                              \
        dense_B,                                                              \
        ldb,                                                                  \
        batch_stride_B,                                                       \
        dense_C,                                                              \
        ldc,                                                                  \
        batch_stride_C,                                                       \
        order_C,                                                              \
        descr->base,                                                          \
        handle->pointer_mode == rocsparse_pointer_mode_host)

#define LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(WF_SIZE, LOOPS, COLS, NT) \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                            \
        (rocsparse::coommnn_segmented_atomic<WF_SIZE, LOOPS, COLS, NT, T>),        \
        dim3(nblocks, 1, batch_count_C),                                           \
        dim3(WF_SIZE),                                                             \
        0,                                                                         \
        stream,                                                                    \
        trans_B,                                                                   \
        nnz,                                                                       \
        main,                                                                      \
        batch_stride_A,                                                            \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),              \
        coo_row_ind,                                                               \
        coo_col_ind,                                                               \
        coo_val,                                                                   \
        dense_B,                                                                   \
        ldb,                                                                       \
        batch_stride_B,                                                            \
        dense_C,                                                                   \
        ldc,                                                                       \
        batch_stride_C,                                                            \
        order_C,                                                                   \
        descr->base,                                                               \
        handle->pointer_mode == rocsparse_pointer_mode_host)

    template <typename T, typename I, typename A, typename B, typename C>
    rocsparse_status coomm_template_segmented_atomic(rocsparse_handle          handle,
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

        // Stream
        hipStream_t stream = handle->stream;

        // Run different coomm kernels
        if(trans_A == rocsparse_operation_none)
        {
            if((order_B == rocsparse_order_column && trans_B == rocsparse_operation_none)
               || (order_B == rocsparse_order_row && trans_B == rocsparse_operation_transpose)
               || (order_B == rocsparse_order_row
                   && trans_B == rocsparse_operation_conjugate_transpose))
            {
                I main = 0;
                I remainder;

                if(handle->wavefront_size == 32)
                {
                    static constexpr I nloops  = 16;
                    const I            nblocks = (nnz - 1) / (32 * nloops) + 1;

                    if(n >= 8)
                    {
                        remainder = n % 8;
                        main      = n - remainder;

                        LAUNCH_COOMMNN_SEGMENTED_ATOMIC_MAIN_KERNEL(32, 16, 8, false);
                    }
                    else if(n >= 4)
                    {
                        remainder = n % 4;
                        main      = n - remainder;

                        LAUNCH_COOMMNN_SEGMENTED_ATOMIC_MAIN_KERNEL(32, 16, 4, false);
                    }
                    else
                    {
                        remainder = n;
                    }

                    if(remainder > 0)
                    {
                        if(remainder == 1)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(32, 16, 1, false);
                        }
                        else if(remainder == 2)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(32, 16, 2, false);
                        }
                        else if(remainder == 3)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(32, 16, 3, false);
                        }
                        else if(remainder == 4)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(32, 16, 4, false);
                        }
                        else if(remainder == 5)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(32, 16, 5, false);
                        }
                        else if(remainder == 6)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(32, 16, 6, false);
                        }
                        else if(remainder == 7)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(32, 16, 7, false);
                        }
                    }
                }
                else if(handle->wavefront_size == 64)
                {
                    static constexpr I nloops  = 16;
                    I                  nblocks = (nnz - 1) / (64 * nloops) + 1;

                    if(n >= 8)
                    {
                        remainder = n % 8;
                        main      = n - remainder;

                        LAUNCH_COOMMNN_SEGMENTED_ATOMIC_MAIN_KERNEL(64, 16, 8, false);
                    }
                    else if(n >= 4)
                    {
                        remainder = n % 4;
                        main      = n - remainder;

                        LAUNCH_COOMMNN_SEGMENTED_ATOMIC_MAIN_KERNEL(64, 16, 4, false);
                    }
                    else
                    {
                        remainder = n;
                    }

                    if(remainder > 0)
                    {
                        if(remainder == 1)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(64, 16, 1, false);
                        }
                        else if(remainder == 2)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(64, 16, 2, false);
                        }
                        else if(remainder == 3)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(64, 16, 3, false);
                        }
                        else if(remainder == 4)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(64, 16, 4, false);
                        }
                        else if(remainder == 5)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(64, 16, 5, false);
                        }
                        else if(remainder == 6)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(64, 16, 6, false);
                        }
                        else if(remainder == 7)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(64, 16, 7, false);
                        }
                    }
                }
            }
            else if((order_B == rocsparse_order_column
                     && trans_B == rocsparse_operation_conjugate_transpose)
                    || (order_B == rocsparse_order_column
                        && trans_B == rocsparse_operation_transpose)
                    || (order_B == rocsparse_order_row && trans_B == rocsparse_operation_none))
            {
                I main = 0;
                I remainder;

                if(handle->wavefront_size == 32)
                {
                    static constexpr I nloops  = 16;
                    I                  nblocks = (nnz - 1) / (32 * nloops) + 1;

                    if(n >= 8)
                    {
                        remainder = n % 8;
                        main      = n - remainder;

                        LAUNCH_COOMMNN_SEGMENTED_ATOMIC_MAIN_KERNEL(32, 16, 8, true);
                    }
                    else if(n >= 4)
                    {
                        remainder = n % 4;
                        main      = n - remainder;

                        LAUNCH_COOMMNN_SEGMENTED_ATOMIC_MAIN_KERNEL(32, 16, 4, true);
                    }
                    else
                    {
                        remainder = n;
                    }

                    if(remainder > 0)
                    {
                        if(remainder == 1)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(32, 16, 1, true);
                        }
                        else if(remainder == 2)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(32, 16, 2, true);
                        }
                        else if(remainder == 3)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(32, 16, 3, true);
                        }
                        else if(remainder == 4)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(32, 16, 4, true);
                        }
                        else if(remainder == 5)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(32, 16, 5, true);
                        }
                        else if(remainder == 6)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(32, 16, 6, true);
                        }
                        else if(remainder == 7)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(32, 16, 7, true);
                        }
                    }
                }
                else if(handle->wavefront_size == 64)
                {
                    static constexpr I nloops  = 16;
                    I                  nblocks = (nnz - 1) / (64 * nloops) + 1;

                    if(n >= 8)
                    {
                        remainder = n % 8;
                        main      = n - remainder;

                        LAUNCH_COOMMNN_SEGMENTED_ATOMIC_MAIN_KERNEL(64, 16, 8, true);
                    }
                    else if(n >= 4)
                    {
                        remainder = n % 4;
                        main      = n - remainder;

                        LAUNCH_COOMMNN_SEGMENTED_ATOMIC_MAIN_KERNEL(64, 16, 4, true);
                    }
                    else
                    {
                        remainder = n;
                    }

                    if(remainder > 0)
                    {
                        if(remainder == 1)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(64, 16, 1, true);
                        }
                        else if(remainder == 2)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(64, 16, 2, true);
                        }
                        else if(remainder == 3)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(64, 16, 3, true);
                        }
                        else if(remainder == 4)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(64, 16, 4, true);
                        }
                        else if(remainder == 5)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(64, 16, 5, true);
                        }
                        else if(remainder == 6)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(64, 16, 6, true);
                        }
                        else if(remainder == 7)
                        {
                            LAUNCH_COOMMNN_SEGMENTED_ATOMIC_REMAINDER_KERNEL(64, 16, 7, true);
                        }
                    }
                }
            }
#undef COOMMN_DIM
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        return rocsparse_status_success;
    }
}

#define INSTANTIATE(TTYPE, ITYPE, ATYPE, BTYPE, CTYPE)                    \
    template rocsparse_status rocsparse::coomm_template_segmented_atomic( \
        rocsparse_handle          handle,                                 \
        rocsparse_operation       trans_A,                                \
        rocsparse_operation       trans_B,                                \
        ITYPE                     m,                                      \
        ITYPE                     n,                                      \
        ITYPE                     k,                                      \
        int64_t                   nnz,                                    \
        ITYPE                     batch_count_A,                          \
        int64_t                   batch_stride_A,                         \
        const TTYPE*              alpha_device_host,                      \
        const rocsparse_mat_descr descr,                                  \
        const ATYPE*              coo_val,                                \
        const ITYPE*              coo_row_ind,                            \
        const ITYPE*              coo_col_ind,                            \
        const BTYPE*              dense_B,                                \
        int64_t                   ldb,                                    \
        ITYPE                     batch_count_B,                          \
        int64_t                   batch_stride_B,                         \
        rocsparse_order           order_B,                                \
        const TTYPE*              beta_device_host,                       \
        CTYPE*                    dense_C,                                \
        int64_t                   ldc,                                    \
        ITYPE                     batch_count_C,                          \
        int64_t                   batch_stride_C,                         \
        rocsparse_order           order_C);

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
