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

#include "bellmm_device_general.h"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <rocsparse_int BELL_BLOCK_DIM,
              rocsparse_int BLK_SIZE_Y,
              typename T,
              typename I,
              typename A,
              typename B,
              typename C>
    ROCSPARSE_KERNEL(BELL_BLOCK_DIM* BLK_SIZE_Y)
    void bellmm_general_blockdim_kernel(rocsparse_operation trans_A,
                                        rocsparse_operation trans_B,
                                        rocsparse_direction dir_A,
                                        I                   Mb,
                                        I                   N,
                                        ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                        I bell_cols,
                                        I block_dim,
                                        const I* __restrict__ bell_col_ind,
                                        const A* __restrict__ bell_val,
                                        const B* __restrict__ dense_B,
                                        int64_t         ldb,
                                        rocsparse_order order_B,
                                        ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                        C* __restrict__ dense_C,
                                        int64_t              ldc,
                                        rocsparse_order      order_C,
                                        rocsparse_index_base idx_base,
                                        bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);

        if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
        {
            return;
        }

        rocsparse::bellmm_general_blockdim_device<BELL_BLOCK_DIM, BLK_SIZE_Y, T>(trans_A,
                                                                                 trans_B,
                                                                                 dir_A,
                                                                                 Mb,
                                                                                 N,
                                                                                 alpha,
                                                                                 bell_cols,
                                                                                 block_dim,
                                                                                 bell_col_ind,
                                                                                 bell_val,
                                                                                 dense_B,
                                                                                 ldb,
                                                                                 order_B,
                                                                                 beta,
                                                                                 dense_C,
                                                                                 ldc,
                                                                                 order_C,
                                                                                 idx_base);
    }

    template <typename T, typename I, typename A, typename B, typename C>
    rocsparse_status bellmm_template_general(rocsparse_handle          handle,
                                             rocsparse_operation       trans_A,
                                             rocsparse_operation       trans_B,
                                             rocsparse_direction       dir_A,
                                             I                         mb,
                                             I                         n,
                                             I                         kb,
                                             I                         bell_cols,
                                             I                         block_dim,
                                             const T*                  alpha,
                                             const rocsparse_mat_descr descr,
                                             const I*                  bell_col_ind,
                                             const A*                  bell_val,
                                             const B*                  dense_B,
                                             int64_t                   ldb,
                                             rocsparse_order           order_B,
                                             const T*                  beta,
                                             C*                        dense_C,
                                             int64_t                   ldc,
                                             rocsparse_order           order_C)
    {
        ROCSPARSE_ROUTINE_TRACE;

        hipStream_t stream = handle->stream;
        dim3        bellmm_blocks((mb - 1) / 1 + 1, (n - 1) / 32 + 1);
        dim3        bellmm_threads(32, 32, 1);

        if(trans_A != rocsparse_operation_none)
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_not_implemented,
                "This function is designed for trans_A = rocsparse_operation_none.");
        }

        //
        // What happends if A needs to be transposed?
        //
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::bellmm_general_blockdim_kernel<32, 32, T>),
                                           bellmm_blocks,
                                           bellmm_threads,
                                           0,
                                           stream,
                                           trans_A,
                                           trans_B,
                                           dir_A,
                                           mb,
                                           n,
                                           ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha),
                                           bell_cols,
                                           block_dim,
                                           bell_col_ind,
                                           bell_val,
                                           dense_B,
                                           ldb,
                                           order_B,
                                           ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta),
                                           dense_C,
                                           ldc,
                                           order_C,
                                           descr->base,
                                           handle->pointer_mode == rocsparse_pointer_mode_host);

        return rocsparse_status_success;
    }
}

#define INSTANTIATE(TTYPE, ITYPE, ATYPE, BTYPE, CTYPE)                                               \
    template rocsparse_status rocsparse::bellmm_template_general(rocsparse_handle    handle,         \
                                                                 rocsparse_operation trans_A,        \
                                                                 rocsparse_operation trans_B,        \
                                                                 rocsparse_direction dir_A,          \
                                                                 ITYPE               mb,             \
                                                                 ITYPE               n,              \
                                                                 ITYPE               kb,             \
                                                                 ITYPE               bell_cols,      \
                                                                 ITYPE               bell_block_dim, \
                                                                 const TTYPE*        alpha,          \
                                                                 const rocsparse_mat_descr descr,    \
                                                                 const ITYPE*    bell_col_ind,       \
                                                                 const ATYPE*    bell_val,           \
                                                                 const BTYPE*    dense_B,            \
                                                                 int64_t         ldb,                \
                                                                 rocsparse_order order_B,            \
                                                                 const TTYPE*    beta,               \
                                                                 CTYPE*          dense_C,            \
                                                                 int64_t         ldc,                \
                                                                 rocsparse_order order_C)

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

// Mixed precisions
INSTANTIATE(int32_t, int32_t, int32_t, int32_t, int32_t);
INSTANTIATE(int32_t, int64_t, int32_t, int32_t, int32_t);
INSTANTIATE(float, int32_t, _Float16, _Float16, float);
INSTANTIATE(float, int64_t, _Float16, _Float16, float);
INSTANTIATE(float, int32_t, _Float16, _Float16, _Float16);
INSTANTIATE(float, int64_t, _Float16, _Float16, _Float16);
INSTANTIATE(float, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(float, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, float);
INSTANTIATE(float, int32_t, rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16);
INSTANTIATE(float, int64_t, rocsparse_bfloat16, rocsparse_bfloat16, rocsparse_bfloat16);
INSTANTIATE(int32_t, int32_t, int8_t, int8_t, int32_t);
INSTANTIATE(int32_t, int64_t, int8_t, int8_t, int32_t);
INSTANTIATE(float, int32_t, int8_t, int8_t, float);
INSTANTIATE(float, int64_t, int8_t, int8_t, float);

#undef INSTANTIATE
