/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "bsrmm_device_large_ext.h"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <uint32_t BSR_BLOCK_DIM,
              uint32_t BLK_SIZE_Y,
              uint32_t UNROLL_SIZE_Y,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C,
              typename T>
    ROCSPARSE_KERNEL(BSR_BLOCK_DIM* BLK_SIZE_Y)
    void bsrmm_large_blockdim_kernel_ext(bool                nn,
                                         rocsparse_direction direction,
                                         J                   mb,
                                         J                   n,
                                         int64_t             offsets_batch_stride_A,
                                         int64_t             columns_values_batch_stride_A,
                                         ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                         const I* __restrict__ bsr_row_ptr,
                                         const J* __restrict__ bsr_col_ind,
                                         const A* __restrict__ bsr_val,
                                         J block_dim,
                                         const B* __restrict__ dense_B,
                                         int64_t ldb,
                                         int64_t batch_stride_B,
                                         ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                         C* __restrict__ dense_C,
                                         int64_t              ldc,
                                         int64_t              batch_stride_C,
                                         rocsparse_order      order_C,
                                         rocsparse_index_base idx_base,
                                         bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);

        if(alpha == 0 && beta == 1)
        {
            return;
        }

        rocsparse::bsrmm_large_blockdim_device_ext<BSR_BLOCK_DIM, BLK_SIZE_Y, UNROLL_SIZE_Y>(
            nn,
            direction,
            mb,
            n,
            offsets_batch_stride_A,
            columns_values_batch_stride_A,
            alpha,
            bsr_row_ptr,
            bsr_col_ind,
            bsr_val,
            block_dim,
            dense_B,
            ldb,
            batch_stride_B,
            beta,
            dense_C,
            ldc,
            batch_stride_C,
            order_C,
            idx_base);
    }

    typedef enum
    {
        large_config_ext_4_16 = 1,
        large_config_ext_8_8,
        large_config_ext_16_16,
        large_config_ext_32_32
    } enum_large_config_ext;

    //
    // Select which tuned kernel to apply.
    //
    template <typename J>
    static enum_large_config_ext get_large_config_ext(J block_dim, J n)
    {
        if(block_dim <= 4)
        {
            return large_config_ext_4_16;
        }
        else if(block_dim <= 8)
        {
            return large_config_ext_8_8;
        }
        else if(block_dim <= 16)
        {
            return large_config_ext_16_16;
        }
        else
        {
            rocsparse_host_assert(block_dim <= 32, "Wrong logical dispatch.");
            return large_config_ext_32_32;
        }
    }

    template <typename T, typename I, typename J, typename A, typename B, typename C>
    rocsparse_status bsrmm_template_large_ext(bool                nn,
                                              rocsparse_handle    handle,
                                              rocsparse_direction dir,
                                              rocsparse_operation trans_A,
                                              rocsparse_operation trans_B,
                                              J                   mb,
                                              J                   n,
                                              J                   kb,
                                              I                   nnzb,
                                              J                   batch_count_A,
                                              int64_t             offsets_batch_stride_A,
                                              int64_t             columns_values_batch_stride_A,
                                              const T*            alpha,
                                              const rocsparse_mat_descr descr,
                                              const A*                  bsr_val,
                                              const I*                  bsr_row_ptr,
                                              const J*                  bsr_col_ind,
                                              J                         block_dim,
                                              const B*                  dense_B,
                                              int64_t                   ldb,
                                              J                         batch_count_B,
                                              int64_t                   batch_stride_B,
                                              rocsparse_order           order_B,
                                              const T*                  beta,
                                              C*                        dense_C,
                                              int64_t                   ldc,
                                              J                         batch_count_C,
                                              int64_t                   batch_stride_C,
                                              rocsparse_order           order_C)
    {
        ROCSPARSE_ROUTINE_TRACE;

        hipStream_t stream = handle->stream;
        rocsparse_host_assert(block_dim <= 32, "This function is designed for block_dim <= 32.");

#define LAUNCH_LARGE_KERNEL(M_, N_, K_)                                                          \
    const dim3 bsrmm_blocks((mb - 1) / 1 + 1, (n - 1) / (N_ * K_) + 1);                          \
    const dim3 bsrmm_threads(M_, N_);                                                            \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::bsrmm_large_blockdim_kernel_ext<M_, N_, K_>), \
                                       bsrmm_blocks,                                             \
                                       bsrmm_threads,                                            \
                                       0,                                                        \
                                       stream,                                                   \
                                       nn,                                                       \
                                       dir,                                                      \
                                       mb,                                                       \
                                       n,                                                        \
                                       offsets_batch_stride_A,                                   \
                                       columns_values_batch_stride_A,                            \
                                       ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha),         \
                                       bsr_row_ptr,                                              \
                                       bsr_col_ind,                                              \
                                       bsr_val,                                                  \
                                       block_dim,                                                \
                                       dense_B,                                                  \
                                       ldb,                                                      \
                                       batch_stride_B,                                           \
                                       ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta),          \
                                       dense_C,                                                  \
                                       ldc,                                                      \
                                       batch_stride_C,                                           \
                                       order_C,                                                  \
                                       descr->base,                                              \
                                       handle->pointer_mode == rocsparse_pointer_mode_host)

        //
        // Select which tuned kernel to apply.
        //
        const enum_large_config_ext config = rocsparse::get_large_config_ext(block_dim, n);
        switch(config)
        {
#define DEFINE_CASE(i, j, k, l)       \
    case i:                           \
    {                                 \
        LAUNCH_LARGE_KERNEL(j, k, l); \
        break;                        \
    }

            DEFINE_CASE(large_config_ext_8_8, 8, 8, 2);
            DEFINE_CASE(large_config_ext_4_16, 4, 16, 2);
            DEFINE_CASE(large_config_ext_16_16, 16, 16, 2);
            DEFINE_CASE(large_config_ext_32_32, 32, 32, 2);

#undef DEFINE_CASE
        }

#undef LAUNCH_LARGE_KERNEL
        return rocsparse_status_success;
    }
}

#define INSTANTIATE(T, I, J, A, B, C)                              \
    template rocsparse_status rocsparse::bsrmm_template_large_ext( \
        bool                      nn,                              \
        rocsparse_handle          handle,                          \
        rocsparse_direction       dir,                             \
        rocsparse_operation       trans_A,                         \
        rocsparse_operation       trans_B,                         \
        J                         mb,                              \
        J                         n,                               \
        J                         kb,                              \
        I                         nnzb,                            \
        J                         batch_count_A,                   \
        int64_t                   offsets_batch_stride_A,          \
        int64_t                   columns_values_batch_stride_A,   \
        const T*                  alpha,                           \
        const rocsparse_mat_descr descr,                           \
        const A*                  bsr_val,                         \
        const I*                  bsr_row_ptr,                     \
        const J*                  bsr_col_ind,                     \
        J                         block_dim,                       \
        const B*                  dense_B,                         \
        int64_t                   ldb,                             \
        J                         batch_count_B,                   \
        int64_t                   batch_stride_B,                  \
        rocsparse_order           order_B,                         \
        const T*                  beta,                            \
        C*                        dense_C,                         \
        int64_t                   ldc,                             \
        J                         batch_count_C,                   \
        int64_t                   batch_stride_C,                  \
        rocsparse_order           order_C)

// Uniform precisions
INSTANTIATE(float, int32_t, int32_t, float, float, float);
INSTANTIATE(float, int64_t, int32_t, float, float, float);
INSTANTIATE(float, int64_t, int64_t, float, float, float);
INSTANTIATE(double, int32_t, int32_t, double, double, double);
INSTANTIATE(double, int64_t, int32_t, double, double, double);
INSTANTIATE(double, int64_t, int64_t, double, double, double);
INSTANTIATE(rocsparse_float_complex,
            int32_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int32_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_float_complex,
            int64_t,
            int64_t,
            rocsparse_float_complex,
            rocsparse_float_complex,
            rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex,
            int32_t,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int32_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);
INSTANTIATE(rocsparse_double_complex,
            int64_t,
            int64_t,
            rocsparse_double_complex,
            rocsparse_double_complex,
            rocsparse_double_complex);

// Mixed Precisions
INSTANTIATE(float, int32_t, int32_t, _Float16, _Float16, float);
INSTANTIATE(float, int64_t, int32_t, _Float16, _Float16, float);
INSTANTIATE(float, int64_t, int64_t, _Float16, _Float16, float);
INSTANTIATE(float, int32_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(float, int64_t, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(float, int64_t, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(int32_t, int32_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE(int32_t, int64_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE(int32_t, int64_t, int64_t, int8_t, int8_t, int32_t);
INSTANTIATE(float, int32_t, int32_t, int8_t, int8_t, float);
INSTANTIATE(float, int64_t, int32_t, int8_t, int8_t, float);
INSTANTIATE(float, int64_t, int64_t, int8_t, int8_t, float);

#undef INSTANTIATE
