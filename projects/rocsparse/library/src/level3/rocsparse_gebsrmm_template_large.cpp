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

#include "rocsparse_utility.hpp"

#include "gebsrmm_device_large.h"
#include "rocsparse_csrmm.hpp"

template <rocsparse_int BSR_BLOCK_DIM, rocsparse_int BLK_SIZE_Y, typename T>
ROCSPARSE_KERNEL(BSR_BLOCK_DIM* BLK_SIZE_Y)
void gebsrmm_large_blockdim_kernel(rocsparse_direction direction,
                                   rocsparse_operation trans_B,
                                   rocsparse_int       mb,
                                   rocsparse_int       n,
                                   ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                   const rocsparse_int* __restrict__ bsr_row_ptr,
                                   const rocsparse_int* __restrict__ bsr_col_ind,
                                   const T* __restrict__ bsr_val,
                                   rocsparse_int row_block_dim,
                                   rocsparse_int col_block_dim,
                                   const T* __restrict__ B,
                                   int64_t ldb,
                                   ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                   T* __restrict__ C,
                                   int64_t              ldc,
                                   rocsparse_index_base idx_base,
                                   bool                 is_host_mode)
{
    ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
    ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);

    if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
    {
        return;
    }

    rocsparse::gebsrmm_large_blockdim_device<BSR_BLOCK_DIM, BLK_SIZE_Y>(direction,
                                                                        trans_B,
                                                                        mb,
                                                                        n,
                                                                        alpha,
                                                                        bsr_row_ptr,
                                                                        bsr_col_ind,
                                                                        bsr_val,
                                                                        row_block_dim,
                                                                        col_block_dim,
                                                                        B,
                                                                        ldb,
                                                                        beta,
                                                                        C,
                                                                        ldc,
                                                                        idx_base);
}

typedef enum
{
    large_config_4_16 = 1,
    large_config_8_32,
    large_config_8_16,
    large_config_16_16,
    large_config_32_32
} enum_large_config;

//
// Select which tuned kernel to apply.
//
static enum_large_config get_large_config(rocsparse_int block_dim, rocsparse_int n)
{
    enum_large_config config;
    if(block_dim <= 4)
    {
        config = large_config_4_16;
    }
    else if(block_dim <= 8)
    {
        if(n <= 16)
        {
            config = large_config_8_16;
        }
        else
        {
            config = large_config_8_32;
        }
    }
    else if(block_dim <= 16)
    {
        config = large_config_16_16;
    }
    else
    {
        rocsparse_host_assert(block_dim <= 32, "Wrong logical dispatch.");
        config = large_config_32_32;
    }
    return config;
}

namespace rocsparse
{
    template <typename T>
    rocsparse_status gebsrmm_template_large(rocsparse_handle          handle,
                                            rocsparse_direction       dir,
                                            rocsparse_operation       trans_A,
                                            rocsparse_operation       trans_B,
                                            rocsparse_int             mb,
                                            rocsparse_int             n,
                                            rocsparse_int             kb,
                                            rocsparse_int             nnzb,
                                            const T*                  alpha,
                                            const rocsparse_mat_descr descr,
                                            const T*                  bsr_val,
                                            const rocsparse_int*      bsr_row_ptr,
                                            const rocsparse_int*      bsr_col_ind,
                                            rocsparse_int             row_block_dim,
                                            rocsparse_int             col_block_dim,
                                            const T*                  B,
                                            int64_t                   ldb,
                                            const T*                  beta,
                                            T*                        C,
                                            int64_t                   ldc)
    {
        ROCSPARSE_ROUTINE_TRACE;

        hipStream_t stream = handle->stream;
        rocsparse_host_assert(row_block_dim <= 32,
                              "This function is designed for row_block_dim <= 32.");

#define LAUNCH_LARGE_KERNEL(M_, N_)                                                        \
    dim3 gebsrmm_blocks((mb - 1) / 1 + 1, (n - 1) / N_ + 1);                               \
    dim3 gebsrmm_threads(M_, N_);                                                          \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gebsrmm_large_blockdim_kernel<M_, N_>), \
                                       gebsrmm_blocks,                                     \
                                       gebsrmm_threads,                                    \
                                       0,                                                  \
                                       stream,                                             \
                                       dir,                                                \
                                       trans_B,                                            \
                                       mb,                                                 \
                                       n,                                                  \
                                       ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha),   \
                                       bsr_row_ptr,                                        \
                                       bsr_col_ind,                                        \
                                       bsr_val,                                            \
                                       row_block_dim,                                      \
                                       col_block_dim,                                      \
                                       B,                                                  \
                                       ldb,                                                \
                                       ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta),    \
                                       C,                                                  \
                                       ldc,                                                \
                                       descr->base,                                        \
                                       handle->pointer_mode == rocsparse_pointer_mode_host)

        //
        // Select which tuned kernel to apply.
        // where the block max size is the maximum block dimension.
        //
        const enum_large_config config
            = rocsparse::get_large_config(rocsparse::max(row_block_dim, col_block_dim), n);
        switch(config)
        {
#define DEFINE_CASE(i, j, k)       \
    case i:                        \
    {                              \
        LAUNCH_LARGE_KERNEL(j, k); \
        break;                     \
    }
            DEFINE_CASE(large_config_8_32, 8, 32);
            DEFINE_CASE(large_config_4_16, 4, 16);
            DEFINE_CASE(large_config_8_16, 8, 16);
            DEFINE_CASE(large_config_16_16, 16, 16);
            DEFINE_CASE(large_config_32_32, 32, 32);
#undef DEFINE_CASE
        }

#undef LAUNCH_LARGE_KERNEL
        return rocsparse_status_success;
    }
}

#define INSTANTIATE(T)                                                                              \
                                                                                                    \
    template rocsparse_status rocsparse::gebsrmm_template_large(rocsparse_handle          handle,   \
                                                                rocsparse_direction       dir,      \
                                                                rocsparse_operation       trans_A,  \
                                                                rocsparse_operation       trans_B,  \
                                                                rocsparse_int             mb,       \
                                                                rocsparse_int             n,        \
                                                                rocsparse_int             kb,       \
                                                                rocsparse_int             nnzb,     \
                                                                const T*                  alpha,    \
                                                                const rocsparse_mat_descr descr,    \
                                                                const T*                  bsr_val,  \
                                                                const rocsparse_int* bsr_row_ptr,   \
                                                                const rocsparse_int* bsr_col_ind,   \
                                                                rocsparse_int        row_block_dim, \
                                                                rocsparse_int        col_block_dim, \
                                                                const T*             B,             \
                                                                int64_t              ldb,           \
                                                                const T*             beta,          \
                                                                T*                   C,             \
                                                                int64_t              ldc)

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

#undef INSTANTIATE
