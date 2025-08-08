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

#include "bsrmm_device_general.h"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <uint32_t BSR_BLOCK_DIM,
              uint32_t BLK_SIZE_Y,
              typename I,
              typename J,
              typename A,
              typename B,
              typename C,
              typename T>
    ROCSPARSE_KERNEL(BSR_BLOCK_DIM* BLK_SIZE_Y)
    void bsrmm_general_blockdim_kernel(bool                nn,
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

        rocsparse::bsrmm_general_blockdim_device<BSR_BLOCK_DIM, BLK_SIZE_Y>(
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

    template <typename T, typename I, typename J, typename A, typename B, typename C>
    rocsparse_status bsrmm_template_general(bool                      nn,
                                            rocsparse_handle          handle,
                                            rocsparse_direction       dir,
                                            rocsparse_operation       trans_A,
                                            rocsparse_operation       trans_B,
                                            J                         mb,
                                            J                         n,
                                            J                         kb,
                                            I                         nnzb,
                                            J                         batch_count_A,
                                            int64_t                   offsets_batch_stride_A,
                                            int64_t                   columns_values_batch_stride_A,
                                            const T*                  alpha,
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
        rocsparse_host_assert(block_dim > 32, "This function is designed for block_dim > 32.");

        const dim3 bsrmm_blocks((mb - 1) / 1 + 1, (n - 1) / 32 + 1);
        const dim3 bsrmm_threads(32, 32, 1);
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::bsrmm_general_blockdim_kernel<32, 32>),
                                           bsrmm_blocks,
                                           bsrmm_threads,
                                           0,
                                           stream,
                                           nn,
                                           dir,
                                           mb,
                                           n,
                                           offsets_batch_stride_A,
                                           columns_values_batch_stride_A,
                                           ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha),
                                           bsr_row_ptr,
                                           bsr_col_ind,
                                           bsr_val,
                                           block_dim,
                                           dense_B,
                                           ldb,
                                           batch_stride_B,
                                           ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta),
                                           dense_C,
                                           ldc,
                                           batch_stride_C,
                                           order_C,
                                           descr->base,
                                           handle->pointer_mode == rocsparse_pointer_mode_host);

        return rocsparse_status_success;
    }
}

#define INSTANTIATE(TTYPE, ITYPE, JTYPE, ATYPE, BTYPE, CTYPE)    \
    template rocsparse_status rocsparse::bsrmm_template_general( \
        bool                      nn,                            \
        rocsparse_handle          handle,                        \
        rocsparse_direction       dir,                           \
        rocsparse_operation       trans_A,                       \
        rocsparse_operation       trans_B,                       \
        JTYPE                     mb,                            \
        JTYPE                     n,                             \
        JTYPE                     kb,                            \
        ITYPE                     nnzb,                          \
        JTYPE                     batch_count_A,                 \
        int64_t                   offsets_batch_stride_A,        \
        int64_t                   columns_values_batch_stride_A, \
        const TTYPE*              alpha,                         \
        const rocsparse_mat_descr descr,                         \
        const ATYPE*              bsr_val,                       \
        const ITYPE*              bsr_row_ptr,                   \
        const JTYPE*              bsr_col_ind,                   \
        JTYPE                     block_dim,                     \
        const BTYPE*              dense_B,                       \
        int64_t                   ldb,                           \
        JTYPE                     batch_count_B,                 \
        int64_t                   batch_stride_B,                \
        rocsparse_order           order_B,                       \
        const TTYPE*              beta,                          \
        CTYPE*                    dense_C,                       \
        int64_t                   ldc,                           \
        JTYPE                     batch_count_C,                 \
        int64_t                   batch_stride_C,                \
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
