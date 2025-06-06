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

#include "gebsrmv_device.h"

#include "rocsparse_handle.hpp"

#include "rocsparse_utility.hpp"

#include <hip/hip_runtime.h>

namespace rocsparse
{
#define LAUNCH_GEBSRMV_GENERAL_KERNEL(BLOCKSIZE, WFSIZE)              \
    THROW_IF_HIPLAUNCHKERNELGGL_ERROR(                                \
        (gebsrmvn_general_kernel<BLOCKSIZE, WFSIZE>),                 \
        dim3(mb),                                                     \
        dim3(BLOCKSIZE),                                              \
        0,                                                            \
        handle->stream,                                               \
        mb,                                                           \
        dir,                                                          \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host), \
        bsr_row_ptr,                                                  \
        bsr_col_ind,                                                  \
        bsr_val,                                                      \
        row_block_dim,                                                \
        col_block_dim,                                                \
        x,                                                            \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),  \
        y,                                                            \
        base,                                                         \
        handle->pointer_mode == rocsparse_pointer_mode_host);

#define LAUNCH_GEBSRMV_MXN_KERNEL(BLOCKSIZE, ROWBSRDIM, COLBSRDIM)    \
    THROW_IF_HIPLAUNCHKERNELGGL_ERROR(                                \
        (gebsrmvn_mxn_kernel<BLOCKSIZE, ROWBSRDIM, COLBSRDIM>),       \
        dim3(mb),                                                     \
        dim3(BLOCKSIZE),                                              \
        0,                                                            \
        handle->stream,                                               \
        mb,                                                           \
        dir,                                                          \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host), \
        bsr_row_ptr,                                                  \
        bsr_col_ind,                                                  \
        bsr_val,                                                      \
        row_block_dim,                                                \
        col_block_dim,                                                \
        x,                                                            \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),  \
        y,                                                            \
        base,                                                         \
        handle->pointer_mode == rocsparse_pointer_mode_host);

#define LAUNCH_GEBSRMV_MXN_16_KERNEL(BLOCKSIZE, ROWBSRDIM, COLBSRDIM) \
    THROW_IF_HIPLAUNCHKERNELGGL_ERROR(                                \
        (gebsrmvn_mxn_16_kernel<BLOCKSIZE, ROWBSRDIM, COLBSRDIM>),    \
        dim3(mb),                                                     \
        dim3(BLOCKSIZE),                                              \
        0,                                                            \
        handle->stream,                                               \
        mb,                                                           \
        dir,                                                          \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host), \
        bsr_row_ptr,                                                  \
        bsr_col_ind,                                                  \
        bsr_val,                                                      \
        row_block_dim,                                                \
        col_block_dim,                                                \
        x,                                                            \
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),  \
        y,                                                            \
        base,                                                         \
        handle->pointer_mode == rocsparse_pointer_mode_host);

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

        rocsparse::gebsrmvn_general_device<BLOCKSIZE, WFSIZE>(dir,
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

    template <uint32_t BLOCKSIZE, uint32_t ROWBSRDIM, uint32_t COLBSRDIM, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gebsrmvn_mxn_kernel(rocsparse_int       mb,
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

        rocsparse::gebsrmvn_mxn_device<BLOCKSIZE, ROWBSRDIM, COLBSRDIM>(
            mb, dir, alpha, bsr_row_ptr, bsr_col_ind, bsr_val, x, beta, y, idx_base);
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

        rocsparse::gebsrmvn_mxn_16_device<BLOCKSIZE, ROWBSRDIM, COLBSRDIM>(
            mb, dir, alpha, bsr_row_ptr, bsr_col_ind, bsr_val, x, beta, y, idx_base);
    }

    template <typename T>
    void launch_gebsrmvn_row_block_dim_5_8(rocsparse_handle     handle,
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

        if(row_block_dim == 5)
        {
            if(col_block_dim == 1)
            {
                LAUNCH_GEBSRMV_MXN_KERNEL(60, 5, 1);
            }
            else if(col_block_dim == 2)
            {
                LAUNCH_GEBSRMV_MXN_KERNEL(60, 5, 2);
            }
            else if(col_block_dim == 3)
            {
                LAUNCH_GEBSRMV_MXN_KERNEL(60, 5, 3);
            }
            else if(col_block_dim == 4)
            {
                LAUNCH_GEBSRMV_MXN_KERNEL(60, 5, 4);
            }
            else if(col_block_dim == 5)
            {
                // BECAUSE HOOKED BY BSRMV
                // LCOV_EXCL_START
                LAUNCH_GEBSRMV_MXN_KERNEL(50, 5, 5);
                // LCOV_EXCL_STOP
            }
            else if(col_block_dim == 6)
            {
                LAUNCH_GEBSRMV_MXN_KERNEL(60, 5, 6);
            }
            else if(col_block_dim == 7)
            {
                LAUNCH_GEBSRMV_MXN_KERNEL(35, 5, 7);
            }
            else if(col_block_dim == 8)
            {
                LAUNCH_GEBSRMV_MXN_KERNEL(40, 5, 8);
            }
            else if(col_block_dim == 9)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(90, 5, 9);
            }
            else if(col_block_dim == 10)
            {
                LAUNCH_GEBSRMV_MXN_16_KERNEL(100, 5, 10);
            }
            else if(col_block_dim <= 16)
            {
                LAUNCH_GEBSRMV_GENERAL_KERNEL(16 * 8, 16);
            }
            else
            {
                LAUNCH_GEBSRMV_GENERAL_KERNEL(32 * 8, 32);
            }
        }
        else if(row_block_dim == 6)
        {
            if(col_block_dim == 1)
            {
                LAUNCH_GEBSRMV_MXN_KERNEL(60, 6, 1);
            }
            else if(col_block_dim == 2)
            {
                LAUNCH_GEBSRMV_MXN_KERNEL(60, 6, 2);
            }
            else if(col_block_dim == 3)
            {
                LAUNCH_GEBSRMV_MXN_KERNEL(54, 6, 3);
            }
            else if(col_block_dim == 4)
            {
                LAUNCH_GEBSRMV_MXN_KERNEL(48, 6, 4);
            }
            else if(col_block_dim == 5)
            {
                LAUNCH_GEBSRMV_MXN_KERNEL(60, 6, 5);
            }
            else if(col_block_dim == 6)
            {
                // BECAUSE HOOKED BY BSRMV
                // LCOV_EXCL_START
                LAUNCH_GEBSRMV_MXN_KERNEL(36, 6, 6);
                // LCOV_EXCL_STOP
            }
            else if(col_block_dim == 7)
            {
                LAUNCH_GEBSRMV_MXN_KERNEL(42, 6, 7);
            }
            else if(col_block_dim == 8)
            {
                LAUNCH_GEBSRMV_MXN_KERNEL(48, 6, 8);
            }
            else if(col_block_dim <= 16)
            {
                LAUNCH_GEBSRMV_GENERAL_KERNEL(16 * 8, 16);
            }
            else
            {
                LAUNCH_GEBSRMV_GENERAL_KERNEL(32 * 8, 32);
            }
        }
        else if(row_block_dim == 7)
        {
            if(col_block_dim == 1)
            {
                LAUNCH_GEBSRMV_MXN_KERNEL(63, 7, 1);
            }
            else if(col_block_dim == 2)
            {
                LAUNCH_GEBSRMV_MXN_KERNEL(56, 7, 2);
            }
            else if(col_block_dim == 3)
            {
                LAUNCH_GEBSRMV_MXN_KERNEL(63, 7, 3);
            }
            else if(col_block_dim == 4)
            {
                LAUNCH_GEBSRMV_MXN_KERNEL(56, 7, 4);
            }
            else if(col_block_dim == 5)
            {
                LAUNCH_GEBSRMV_MXN_KERNEL(35, 7, 5);
            }
            else if(col_block_dim == 6)
            {
                LAUNCH_GEBSRMV_MXN_KERNEL(42, 7, 6);
            }
            else if(col_block_dim == 7)
            {
                // BECAUSE HOOKED BY BSRMV
                // LCOV_EXCL_START
                LAUNCH_GEBSRMV_MXN_KERNEL(49, 7, 7);
                // LCOV_EXCL_STOP
            }
            else if(col_block_dim == 8)
            {
                LAUNCH_GEBSRMV_MXN_KERNEL(56, 7, 8);
            }
            else if(col_block_dim <= 16)
            {
                LAUNCH_GEBSRMV_GENERAL_KERNEL(16 * 8, 16);
            }
            else
            {
                LAUNCH_GEBSRMV_GENERAL_KERNEL(32 * 8, 32);
            }
        }
        else if(row_block_dim == 8)
        {
            if(col_block_dim == 1)
            {
                LAUNCH_GEBSRMV_MXN_KERNEL(64, 8, 1);
            }
            else if(col_block_dim == 2)
            {
                LAUNCH_GEBSRMV_MXN_KERNEL(64, 8, 2);
            }
            else if(col_block_dim == 3)
            {
                LAUNCH_GEBSRMV_MXN_KERNEL(48, 8, 3);
            }
            else if(col_block_dim == 4)
            {
                LAUNCH_GEBSRMV_MXN_KERNEL(64, 8, 4);
            }
            else if(col_block_dim == 5)
            {
                LAUNCH_GEBSRMV_MXN_KERNEL(40, 8, 5);
            }
            else if(col_block_dim == 6)
            {
                LAUNCH_GEBSRMV_MXN_KERNEL(48, 8, 6);
            }
            else if(col_block_dim == 7)
            {
                LAUNCH_GEBSRMV_MXN_KERNEL(56, 8, 7);
            }
            else if(col_block_dim == 8)
            {
                // BECAUSE HOOKED BY BSRMV
                // LCOV_EXCL_START
                LAUNCH_GEBSRMV_MXN_KERNEL(64, 8, 8);
                // LCOV_EXCL_STOP
            }
            else if(col_block_dim <= 16)
            {
                LAUNCH_GEBSRMV_GENERAL_KERNEL(16 * 8, 16);
            }
            else
            {
                LAUNCH_GEBSRMV_GENERAL_KERNEL(32 * 8, 32);
            }
        }
    }

    template <typename T>
    rocsparse_status gebsrmv_template_row_block_dim_5_8(rocsparse_handle          handle,
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
            row_block_dim >= 5 && row_block_dim <= 8,
            "This function is designed for row_block_dim >= 5 and row_block_dim <= 8.");

        if(trans == rocsparse_operation_none)
        {
            launch_gebsrmvn_row_block_dim_5_8(handle,
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
            // TODO
            // LCOV_EXCL_START
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            // LCOV_EXCL_STOP
        }

        return rocsparse_status_success;
    }
}

#define INSTANTIATE(T)                                                       \
                                                                             \
    template rocsparse_status rocsparse::gebsrmv_template_row_block_dim_5_8( \
        rocsparse_handle          handle,                                    \
        rocsparse_direction       dir,                                       \
        rocsparse_operation       trans,                                     \
        rocsparse_int             mb,                                        \
        rocsparse_int             nb,                                        \
        rocsparse_int             nnzb,                                      \
        const T*                  alpha,                                     \
        const rocsparse_mat_descr descr,                                     \
        const T*                  bsr_val,                                   \
        const rocsparse_int*      bsr_row_ptr,                               \
        const rocsparse_int*      bsr_col_ind,                               \
        rocsparse_int             row_block_dim,                             \
        rocsparse_int             col_block_dim,                             \
        const T*                  x,                                         \
        const T*                  beta,                                      \
        T*                        y)

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);

#undef INSTANTIATE
