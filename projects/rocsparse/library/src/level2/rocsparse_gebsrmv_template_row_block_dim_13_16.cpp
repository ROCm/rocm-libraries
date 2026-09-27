/* ************************************************************************
 * Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "gebsrmv_device.h"

#include "rocsparse_handle.hpp"

#include "rocsparse_utility.hpp"

#include <hip/hip_runtime.h>

namespace rocsparse
{
// One block per block row, clamped to the device's grid.x limit. With
// BUILD_ROCSPARSE_ILP64=ON `mb` is an int64_t, so handing it to dim3 unclamped
// narrows it to unsigned int and silently drops most of the matrix. The kernels
// grid-stride over the block rows, so an undersized grid still covers [0, mb).
// Replace with rocsparse::get_grid_size(mb, handle->properties.maxGridSize[0])
// once PR #11512 lands.
#define LAUNCH_GEBSRMV_GENERAL_KERNEL(BLOCKSIZE, WFSIZE)                               \
    THROW_IF_HIPLAUNCHKERNELGGL_ERROR(                                                 \
        (gebsrmvn_general_kernel<BLOCKSIZE, WFSIZE>),                                  \
        dim3(rocsparse::min(static_cast<int64_t>(mb),                                  \
                            static_cast<int64_t>(handle->properties.maxGridSize[0]))), \
        dim3(BLOCKSIZE),                                                               \
        0,                                                                             \
        handle->stream,                                                                \
        mb,                                                                            \
        dir,                                                                           \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),                  \
        bsr_row_ptr,                                                                   \
        bsr_col_ind,                                                                   \
        bsr_val,                                                                       \
        row_block_dim,                                                                 \
        col_block_dim,                                                                 \
        x,                                                                             \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),                   \
        y,                                                                             \
        base,                                                                          \
        handle->pointer_mode == rocsparse_pointer_mode_host)

#define LAUNCH_GEBSRMV_MXN_16_KERNEL(BLOCKSIZE, ROWBSRDIM, COLBSRDIM)                  \
    THROW_IF_HIPLAUNCHKERNELGGL_ERROR(                                                 \
        (gebsrmvn_mxn_16_kernel<BLOCKSIZE, ROWBSRDIM, COLBSRDIM>),                     \
        dim3(rocsparse::min(static_cast<int64_t>(mb),                                  \
                            static_cast<int64_t>(handle->properties.maxGridSize[0]))), \
        dim3(BLOCKSIZE),                                                               \
        0,                                                                             \
        handle->stream,                                                                \
        mb,                                                                            \
        dir,                                                                           \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),                  \
        bsr_row_ptr,                                                                   \
        bsr_col_ind,                                                                   \
        bsr_val,                                                                       \
        row_block_dim,                                                                 \
        col_block_dim,                                                                 \
        x,                                                                             \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),                   \
        y,                                                                             \
        base,                                                                          \
        handle->pointer_mode == rocsparse_pointer_mode_host)

    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gebsrmvn_general_kernel(rocsparse_int       mb,
                                 rocsparse_direction dir,
                                 ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                 const rocsparse_int* __restrict__ bsr_row_ptr,
                                 const rocsparse_int* __restrict__ bsr_col_ind,
                                 const T* __restrict__ bsr_val,
                                 rocsparse_int row_block_dim,
                                 rocsparse_int col_block_dim,
                                 const T* __restrict__ x,
                                 ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                 T* __restrict__ y,
                                 rocsparse_index_base idx_base,
                                 bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);

        if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
        {
            return;
        }

        // Grid-stride over the block rows: grid.x is clamped against
        // maxGridSize[0], so one grid sweep only covers hipGridDim_x of them. The
        // bound is block uniform -- it uses only hipBlockIdx_x, hipGridDim_x and the
        // kernel argument mb, never hipThreadIdx_x and never a value read from
        // memory. gebsrmvn_general_device holds no __shared__ state and contains no
        // barrier, so iterations need no fence between them.
        for(int64_t row = hipBlockIdx_x; row < mb; row += hipGridDim_x)
        {
            rocsparse::gebsrmvn_general_device<BLOCKSIZE, WFSIZE>(static_cast<rocsparse_int>(row),
                                                                  dir,
                                                                  alpha,
                                                                  bsr_row_ptr,
                                                                  bsr_col_ind,
                                                                  bsr_val,
                                                                  row_block_dim,
                                                                  col_block_dim,
                                                                  x,
                                                                  beta,
                                                                  y,
                                                                  idx_base);
        }
    }

    template <uint32_t BLOCKSIZE, uint32_t ROWBSRDIM, uint32_t COLBSRDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gebsrmvn_mxn_16_kernel(rocsparse_int       mb,
                                rocsparse_direction dir,
                                ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                const rocsparse_int* __restrict__ bsr_row_ptr,
                                const rocsparse_int* __restrict__ bsr_col_ind,
                                const T* __restrict__ bsr_val,
                                rocsparse_int row_block_dim,
                                rocsparse_int col_block_dim,
                                const T* __restrict__ x,
                                ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                                T* __restrict__ y,
                                rocsparse_index_base idx_base,
                                bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);

        if(alpha == static_cast<T>(0) && beta == static_cast<T>(1))
        {
            return;
        }

        // Grid-stride over the block rows: grid.x is clamped against
        // maxGridSize[0], so one grid sweep only covers hipGridDim_x of them. The
        // bound is block uniform -- it uses only hipBlockIdx_x, hipGridDim_x and the
        // kernel argument mb -- which matters here because the device function
        // reduces through a __shared__ sdata array behind __syncthreads(): a bound
        // that varied within the block would diverge at those barriers.
        for(int64_t row = hipBlockIdx_x; row < mb; row += hipGridDim_x)
        {
            rocsparse::gebsrmvn_mxn_16_device<BLOCKSIZE, ROWBSRDIM, COLBSRDIM>(
                static_cast<rocsparse_int>(row),
                mb,
                dir,
                alpha,
                bsr_row_ptr,
                bsr_col_ind,
                bsr_val,
                x,
                beta,
                y,
                idx_base);

            // The device function leaves its last reads of sdata unfenced against
            // the next iteration's first store. Fence them here.
            __syncthreads();
        }
    }

    template <typename T>
    void launch_gebsrmvn_row_block_dim_13_16(rocsparse_handle     handle,
                                             rocsparse_direction  dir,
                                             rocsparse_int        mb,
                                             rocsparse_int        nnzb,
                                             const T*             alpha_device_host,
                                             const rocsparse_int* bsr_row_ptr,
                                             const rocsparse_int* bsr_col_ind,
                                             const T*             bsr_val,
                                             rocsparse_int        row_block_dim,
                                             rocsparse_int        col_block_dim,
                                             const T*             x,
                                             const T*             beta_device_host,
                                             T*                   y,
                                             rocsparse_index_base base)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(row_block_dim == 13)
        {
            if(col_block_dim == 1)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(52, 13, 1);
            }
            else if(col_block_dim == 2)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(52, 13, 2);
            }
            else if(col_block_dim == 3)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(39, 13, 3);
            }
            else if(col_block_dim == 4)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(52, 13, 4);
            }
            else if(col_block_dim == 5)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(65, 13, 5);
            }
            else if(col_block_dim == 6)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(78, 13, 6);
            }
            else if(col_block_dim == 7)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(91, 13, 7);
            }
            else if(col_block_dim == 8)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(104, 13, 8);
            }
            else if(col_block_dim <= 16)
            {
                LAUNCH_GEBSRMV_GENERAL_KERNEL(16 * 16, 16);
            }
            else
            {
                LAUNCH_GEBSRMV_GENERAL_KERNEL(32 * 16, 32);
            }
        }
        else if(row_block_dim == 14)
        {
            if(col_block_dim == 1)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(56, 14, 1);
            }
            else if(col_block_dim == 2)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(56, 14, 2);
            }
            else if(col_block_dim == 3)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(42, 14, 3);
            }
            else if(col_block_dim == 4)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(56, 14, 4);
            }
            else if(col_block_dim == 5)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(70, 14, 5);
            }
            else if(col_block_dim == 6)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(84, 14, 6);
            }
            else if(col_block_dim == 7)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(98, 14, 7);
            }
            else if(col_block_dim == 8)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(112, 14, 8);
            }
            else if(col_block_dim <= 16)
            {
                LAUNCH_GEBSRMV_GENERAL_KERNEL(16 * 16, 16);
            }
            else
            {
                LAUNCH_GEBSRMV_GENERAL_KERNEL(32 * 16, 32);
            }
        }
        else if(row_block_dim == 15)
        {
            if(col_block_dim == 1)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(60, 15, 1);
            }
            else if(col_block_dim == 2)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(60, 15, 2);
            }
            else if(col_block_dim == 3)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(45, 15, 3);
            }
            else if(col_block_dim == 4)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(60, 15, 4);
            }
            else if(col_block_dim == 5)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(75, 15, 5);
            }
            else if(col_block_dim == 6)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(90, 15, 6);
            }
            else if(col_block_dim == 7)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(105, 15, 7);
            }
            else if(col_block_dim == 8)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(120, 15, 8);
            }
            else if(col_block_dim <= 16)
            {
                LAUNCH_GEBSRMV_GENERAL_KERNEL(16 * 16, 16);
            }
            else
            {
                LAUNCH_GEBSRMV_GENERAL_KERNEL(32 * 16, 32);
            }
        }
        else if(row_block_dim == 16)
        {
            if(col_block_dim == 1)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(64, 16, 1);
            }
            else if(col_block_dim == 2)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(64, 16, 2);
            }
            else if(col_block_dim == 3)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(48, 16, 3);
            }
            else if(col_block_dim == 4)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(64, 16, 4);
            }
            else if(col_block_dim == 5)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(80, 16, 5);
            }
            else if(col_block_dim == 6)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(96, 16, 6);
            }
            else if(col_block_dim == 7)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(112, 16, 7);
            }
            else if(col_block_dim == 8)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(128, 16, 8);
            }
            else if(col_block_dim <= 16)
            {
                // BECAUSE HOOKED BY BSRMV
                // LCOV_EXCL_START
                LAUNCH_GEBSRMV_GENERAL_KERNEL(16 * 16, 16);
                // LCOV_EXCL_STOP
            }
            else
            {
                LAUNCH_GEBSRMV_GENERAL_KERNEL(32 * 16, 32);
            }
        }
    }

    template <typename T>
    rocsparse_status gebsrmv_template_row_block_dim_13_16(rocsparse_handle          handle,
                                                          rocsparse_direction       dir,
                                                          rocsparse_operation       trans,
                                                          rocsparse_int             mb,
                                                          rocsparse_int             nb,
                                                          rocsparse_int             nnzb,
                                                          const T*                  alpha,
                                                          const rocsparse_mat_descr descr,
                                                          const T*                  bsr_val,
                                                          const rocsparse_int*      bsr_row_ptr,
                                                          const rocsparse_int*      bsr_col_ind,
                                                          rocsparse_int             row_block_dim,
                                                          rocsparse_int             col_block_dim,
                                                          const T*                  x,
                                                          const T*                  beta,
                                                          T*                        y)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse_host_assert(
            row_block_dim >= 13 && row_block_dim <= 16,
            "This function is designed for row_block_dim >= 13 and row_block_dim <= 16.");

        if(trans == rocsparse_operation_none)
        {
            launch_gebsrmvn_row_block_dim_13_16(handle,
                                                dir,
                                                mb,
                                                nnzb,
                                                alpha,
                                                bsr_row_ptr,
                                                bsr_col_ind,
                                                bsr_val,
                                                row_block_dim,
                                                col_block_dim,
                                                x,
                                                beta,
                                                y,
                                                descr->base);
        }
        else
        {
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            // LCOV_EXCL_STOP
        }

        return rocsparse_status_success;
    }
}

#define INSTANTIATE(T)                                                         \
                                                                               \
    template rocsparse_status rocsparse::gebsrmv_template_row_block_dim_13_16( \
        rocsparse_handle          handle,                                      \
        rocsparse_direction       dir,                                         \
        rocsparse_operation       trans,                                       \
        rocsparse_int             mb,                                          \
        rocsparse_int             nb,                                          \
        rocsparse_int             nnzb,                                        \
        const T*                  alpha,                                       \
        const rocsparse_mat_descr descr,                                       \
        const T*                  bsr_val,                                     \
        const rocsparse_int*      bsr_row_ptr,                                 \
        const rocsparse_int*      bsr_col_ind,                                 \
        rocsparse_int             row_block_dim,                               \
        rocsparse_int             col_block_dim,                               \
        const T*                  x,                                           \
        const T*                  beta,                                        \
        T*                        y)

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

#undef INSTANTIATE
