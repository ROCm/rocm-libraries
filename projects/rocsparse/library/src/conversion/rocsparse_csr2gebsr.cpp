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

#include "internal/conversion/rocsparse_csr2gebsr.h"
#include "internal/conversion/rocsparse_csr2bsr.h"
#include "internal/conversion/rocsparse_csr2coo.h"

#include "rocsparse_control.hpp"
#include "rocsparse_csr2bsr.hpp"
#include "rocsparse_csr2gebsr.hpp"
#include "rocsparse_utility.hpp"

#include "csr2gebsr_device.h"
#include "rocsparse_common.h"
#include "rocsparse_primitives.hpp"

namespace rocsparse
{
#define launch_csr2gebsr_block_per_row_multipass_kernel(block_size, row_blockdim, col_blockdim) \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                                         \
        (rocsparse::                                                                            \
             csr2gebsr_block_per_row_multipass_kernel<block_size, row_blockdim, col_blockdim>), \
        mb,                                                                                     \
        block_size,                                                                             \
        0,                                                                                      \
        stream,                                                                                 \
        dir,                                                                                    \
        m,                                                                                      \
        n,                                                                                      \
        mb,                                                                                     \
        nb,                                                                                     \
        row_block_dim,                                                                          \
        col_block_dim,                                                                          \
        csr_descr->base,                                                                        \
        csr_val,                                                                                \
        csr_row_ptr,                                                                            \
        csr_col_ind,                                                                            \
        bsr_descr->base,                                                                        \
        bsr_val,                                                                                \
        bsr_row_ptr,                                                                            \
        bsr_col_ind);

#define launch_csr2gebsr_wavefront_per_row_multipass_kernel(                   \
    block_size, row_blockdim, col_blockdim, wf_size)                           \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                        \
        (rocsparse::csr2gebsr_wavefront_per_row_multipass_kernel<block_size,   \
                                                                 row_blockdim, \
                                                                 col_blockdim, \
                                                                 wf_size>),    \
        dim3((mb - 1) / (block_size / wf_size) + 1),                           \
        block_size,                                                            \
        0,                                                                     \
        stream,                                                                \
        dir,                                                                   \
        m,                                                                     \
        n,                                                                     \
        mb,                                                                    \
        nb,                                                                    \
        row_block_dim,                                                         \
        col_block_dim,                                                         \
        csr_descr->base,                                                       \
        csr_val,                                                               \
        csr_row_ptr,                                                           \
        csr_col_ind,                                                           \
        bsr_descr->base,                                                       \
        bsr_val,                                                               \
        bsr_row_ptr,                                                           \
        bsr_col_ind);

    template <typename T,
              typename std::enable_if<std::is_same<T, rocsparse_double_complex>::value, int>::type
              = 0>
    static inline rocsparse_status csr2gebsr_64_64_launcher(rocsparse_handle          handle,
                                                            rocsparse_direction       dir,
                                                            rocsparse_int             m,
                                                            rocsparse_int             n,
                                                            rocsparse_int             mb,
                                                            rocsparse_int             nb,
                                                            const rocsparse_mat_descr csr_descr,
                                                            const T*                  csr_val,
                                                            const rocsparse_int*      csr_row_ptr,
                                                            const rocsparse_int*      csr_col_ind,
                                                            const rocsparse_mat_descr bsr_descr,
                                                            T*                        bsr_val,
                                                            rocsparse_int*            bsr_row_ptr,
                                                            rocsparse_int*            bsr_col_ind,
                                                            rocsparse_int             row_block_dim,
                                                            rocsparse_int             col_block_dim)
    {
        ROCSPARSE_ROUTINE_TRACE;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
    }

    template <
        typename T,
        typename std::enable_if<std::is_same<T, float>::value || std::is_same<T, double>::value
                                    || std::is_same<T, rocsparse_float_complex>::value,
                                int>::type
        = 0>
    static inline rocsparse_status csr2gebsr_64_64_launcher(rocsparse_handle          handle,
                                                            rocsparse_direction       dir,
                                                            rocsparse_int             m,
                                                            rocsparse_int             n,
                                                            rocsparse_int             mb,
                                                            rocsparse_int             nb,
                                                            const rocsparse_mat_descr csr_descr,
                                                            const T*                  csr_val,
                                                            const rocsparse_int*      csr_row_ptr,
                                                            const rocsparse_int*      csr_col_ind,
                                                            const rocsparse_mat_descr bsr_descr,
                                                            T*                        bsr_val,
                                                            rocsparse_int*            bsr_row_ptr,
                                                            rocsparse_int*            bsr_col_ind,
                                                            rocsparse_int             row_block_dim,
                                                            rocsparse_int             col_block_dim)
    {
        ROCSPARSE_ROUTINE_TRACE;

        hipStream_t stream = handle->stream;

        launch_csr2gebsr_block_per_row_multipass_kernel(256, 64, 64);

        return rocsparse_status_success;
    }
}

template <typename T>
rocsparse_status rocsparse::csr2gebsr_buffer_size_template(rocsparse_handle          handle, //0
                                                           rocsparse_direction       dir, //1
                                                           rocsparse_int             m, //2
                                                           rocsparse_int             n, //3
                                                           const rocsparse_mat_descr csr_descr, //4
                                                           const T*                  csr_val, //5
                                                           const rocsparse_int* csr_row_ptr, //6
                                                           const rocsparse_int* csr_col_ind, //7
                                                           rocsparse_int        row_block_dim, //8
                                                           rocsparse_int        col_block_dim, //9
                                                           size_t*              buffer_size) //10
{
    ROCSPARSE_ROUTINE_TRACE;

    //
    // Logging
    //
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xcsr2gebsr_buffer_size"),
                         dir,
                         m,
                         n,
                         csr_descr,
                         (const void*&)csr_val,
                         (const void*&)csr_row_ptr,
                         (const void*&)csr_col_ind,
                         row_block_dim,
                         col_block_dim,
                         (const void*&)buffer_size);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, dir);

    ROCSPARSE_CHECKARG_SIZE(2, m);
    ROCSPARSE_CHECKARG_SIZE(3, n);

    ROCSPARSE_CHECKARG_POINTER(4, csr_descr);
    ROCSPARSE_CHECKARG(4,
                       csr_descr,
                       (csr_descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    ROCSPARSE_CHECKARG_ARRAY(6, m, csr_row_ptr);
    if(csr_val == nullptr || csr_col_ind == nullptr)
    {
        rocsparse_int start = 0;
        rocsparse_int end   = 0;
        if(csr_row_ptr != nullptr)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&end,
                                               &csr_row_ptr[m],
                                               sizeof(rocsparse_int),
                                               hipMemcpyDeviceToHost,
                                               handle->stream));
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&start,
                                               &csr_row_ptr[0],
                                               sizeof(rocsparse_int),
                                               hipMemcpyDeviceToHost,
                                               handle->stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
        }
        const rocsparse_int nnz = (end - start);
        ROCSPARSE_CHECKARG_ARRAY(5, nnz, csr_val);
        ROCSPARSE_CHECKARG_ARRAY(7, nnz, csr_col_ind);
    }

    ROCSPARSE_CHECKARG_SIZE(8, row_block_dim);
    ROCSPARSE_CHECKARG(8, row_block_dim, (row_block_dim == 0), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_SIZE(9, col_block_dim);
    ROCSPARSE_CHECKARG(9, col_block_dim, (col_block_dim == 0), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_POINTER(10, buffer_size);

    //
    // Quick return if possible
    //
    if(m == 0 || n == 0)
    {
        *buffer_size = 0;
        return rocsparse_status_success;
    }

    *buffer_size = 0;
    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse::csr2gebsr_template(rocsparse_handle          handle, //0
                                               rocsparse_direction       dir, //1
                                               rocsparse_int             m, //2
                                               rocsparse_int             n, //3
                                               const rocsparse_mat_descr csr_descr, //4
                                               const T*                  csr_val, //5
                                               const rocsparse_int*      csr_row_ptr, //6
                                               const rocsparse_int*      csr_col_ind, //7
                                               const rocsparse_mat_descr bsr_descr, //8
                                               T*                        bsr_val, //9
                                               rocsparse_int*            bsr_row_ptr, //10
                                               rocsparse_int*            bsr_col_ind, //11
                                               rocsparse_int             row_block_dim, //12
                                               rocsparse_int             col_block_dim, //13
                                               void*                     temp_buffer) //14
{
    ROCSPARSE_ROUTINE_TRACE;

    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xcsr2gebsr"),
                         dir,
                         m,
                         n,
                         csr_descr,
                         (const void*&)csr_val,
                         (const void*&)csr_row_ptr,
                         (const void*&)csr_col_ind,
                         bsr_descr,
                         (const void*&)bsr_val,
                         (const void*&)bsr_row_ptr,
                         (const void*&)bsr_col_ind,
                         row_block_dim,
                         col_block_dim,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, dir);
    ROCSPARSE_CHECKARG_SIZE(2, m);
    ROCSPARSE_CHECKARG_SIZE(3, n);
    ROCSPARSE_CHECKARG_POINTER(4, csr_descr);
    ROCSPARSE_CHECKARG(4,
                       csr_descr,
                       csr_descr->storage_mode != rocsparse_storage_mode_sorted,
                       rocsparse_status_requires_sorted_storage);

    ROCSPARSE_CHECKARG_ARRAY(6, m, csr_row_ptr);
    if(csr_val == nullptr || csr_col_ind == nullptr)
    {
        rocsparse_int start = 0;
        rocsparse_int end   = 0;
        if(csr_row_ptr != nullptr)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&end,
                                               &csr_row_ptr[m],
                                               sizeof(rocsparse_int),
                                               hipMemcpyDeviceToHost,
                                               handle->stream));
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&start,
                                               &csr_row_ptr[0],
                                               sizeof(rocsparse_int),
                                               hipMemcpyDeviceToHost,
                                               handle->stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
        }
        const rocsparse_int nnz = (end - start);
        ROCSPARSE_CHECKARG_ARRAY(5, nnz, csr_val);
        ROCSPARSE_CHECKARG_ARRAY(7, nnz, csr_col_ind);
    }

    ROCSPARSE_CHECKARG_POINTER(8, bsr_descr);
    ROCSPARSE_CHECKARG(8,
                       bsr_descr,
                       bsr_descr->storage_mode != rocsparse_storage_mode_sorted,
                       rocsparse_status_requires_sorted_storage);
    ROCSPARSE_CHECKARG_ARRAY(10, m, bsr_row_ptr);
    ROCSPARSE_CHECKARG_SIZE(12, row_block_dim);
    ROCSPARSE_CHECKARG(12, row_block_dim, (row_block_dim == 0), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_SIZE(13, col_block_dim);
    ROCSPARSE_CHECKARG(13, col_block_dim, (col_block_dim == 0), rocsparse_status_invalid_size);

    //
    // Quick return if possible
    //
    if(m == 0 || n == 0)
    {
        return rocsparse_status_success;
    }

    hipStream_t         stream = handle->stream;
    const rocsparse_int mb     = (m + row_block_dim - 1) / row_block_dim;
    const rocsparse_int nb     = (n + col_block_dim - 1) / col_block_dim;

    rocsparse_int start = 0;
    rocsparse_int end   = 0;
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        &end, &bsr_row_ptr[mb], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        &start, &bsr_row_ptr[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
    const rocsparse_int nnzb = (end - start);

    ROCSPARSE_CHECKARG_ARRAY(9, nnzb, bsr_val);
    ROCSPARSE_CHECKARG_ARRAY(11, nnzb, bsr_col_ind);

    if(row_block_dim == col_block_dim)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2bsr_template(handle,
                                                              dir,
                                                              m,
                                                              n,
                                                              csr_descr,
                                                              csr_val,
                                                              csr_row_ptr,
                                                              csr_col_ind,
                                                              row_block_dim,
                                                              bsr_descr,
                                                              bsr_val,
                                                              bsr_row_ptr,
                                                              bsr_col_ind));
        return rocsparse_status_success;
    }

    //
    // Set bsr val to zero.
    //
    RETURN_IF_HIP_ERROR(hipMemsetAsync(
        bsr_val, 0, sizeof(T) * nnzb * row_block_dim * col_block_dim, handle->stream));

    if(row_block_dim == 1)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csr2gebsr_kernel_bm1<256>),
                                           dim3(m / 256 + 1),
                                           dim3(256),
                                           0,
                                           stream,
                                           m,
                                           n,
                                           mb,
                                           nb,
                                           csr_descr->base,
                                           csr_val,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           dir,
                                           bsr_descr->base,
                                           bsr_val,
                                           bsr_row_ptr,
                                           bsr_col_ind,
                                           row_block_dim,
                                           col_block_dim);

        return rocsparse_status_success;
    }

    // Common case where BSR block dimension is small
    // row_block_dim <= 64 && col_block_dim <= 64
    if(row_block_dim <= 2 && col_block_dim <= 64)
    {
        if(col_block_dim <= 2)
        {
            launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 2, 2, 4);
        }
        else if(col_block_dim <= 4)
        {
            launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 2, 4, 8);
        }
        else if(col_block_dim <= 8)
        {
            launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 2, 8, 16);
        }
        else if(col_block_dim <= 16)
        {
            launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 2, 16, 32);
        }
        else if(col_block_dim <= 32)
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 2, 32, 64);
            }
            else
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 2, 32, 32);
            }
        }
        else
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 2, 64, 64);
            }
            else
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 2, 64, 32);
            }
        }
    }
    else if(row_block_dim <= 4 && col_block_dim <= 64)
    {
        if(col_block_dim <= 2)
        {
            launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 4, 2, 8);
        }
        else if(col_block_dim <= 4)
        {
            launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 4, 4, 16);
        }
        else if(col_block_dim <= 8)
        {
            launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 4, 8, 32);
        }
        else if(col_block_dim <= 16)
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 4, 16, 64);
            }
            else
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 4, 16, 32);
            }
        }
        else if(col_block_dim <= 32)
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 4, 32, 64);
            }
            else
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 4, 32, 32);
            }
        }
        else
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 4, 64, 64);
            }
            else
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 4, 64, 32);
            }
        }
    }
    else if(row_block_dim <= 8 && col_block_dim <= 64)
    {
        if(col_block_dim <= 2)
        {
            launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 8, 2, 16);
        }
        else if(col_block_dim <= 4)
        {
            launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 8, 4, 32);
        }
        else if(col_block_dim <= 8)
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 8, 8, 64);
            }
            else
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 8, 8, 32);
            }
        }
        else if(col_block_dim <= 16)
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 8, 16, 64);
            }
            else
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 8, 16, 32);
            }
        }
        else if(col_block_dim <= 32)
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 8, 32, 64);
            }
            else
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 8, 32, 32);
            }
        }
        else
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 8, 64, 64);
            }
            else
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(128, 8, 64, 32);
            }
        }
    }
    else if(row_block_dim <= 16 && col_block_dim <= 64)
    {
        if(col_block_dim <= 2)
        {
            launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 16, 2, 32);
        }
        else if(col_block_dim <= 4)
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 16, 4, 64);
            }
            else
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 16, 4, 32);
            }
        }
        else if(col_block_dim <= 8)
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 16, 8, 64);
            }
            else
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 16, 8, 32);
            }
        }
        else if(col_block_dim <= 16)
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 16, 16, 64);
            }
            else
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 16, 16, 32);
            }
        }
        else if(col_block_dim <= 32)
        {
            launch_csr2gebsr_block_per_row_multipass_kernel(128, 16, 32);
        }
        else
        {
            launch_csr2gebsr_block_per_row_multipass_kernel(128, 16, 64);
        }
    }
    else if(row_block_dim <= 32 && col_block_dim <= 64)
    {
        if(col_block_dim <= 2)
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 32, 2, 64);
            }
            else
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 32, 2, 32);
            }
        }
        else if(col_block_dim <= 4)
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 32, 4, 64);
            }
            else
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 32, 4, 32);
            }
        }
        else if(col_block_dim <= 8)
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 32, 8, 64);
            }
            else
            {
                launch_csr2gebsr_wavefront_per_row_multipass_kernel(256, 32, 8, 32);
            }
        }
        else if(col_block_dim <= 16)
        {
            launch_csr2gebsr_block_per_row_multipass_kernel(256, 32, 16);
        }
        else if(col_block_dim <= 32)
        {
            launch_csr2gebsr_block_per_row_multipass_kernel(256, 32, 32);
        }
        else
        {
            launch_csr2gebsr_block_per_row_multipass_kernel(256, 32, 64);
        }
    }
    else if(row_block_dim <= 64 && col_block_dim <= 32)
    {
        if(col_block_dim <= 2)
        {
            launch_csr2gebsr_block_per_row_multipass_kernel(128, 64, 2);
        }
        else if(col_block_dim <= 4)
        {
            launch_csr2gebsr_block_per_row_multipass_kernel(256, 64, 4);
        }
        else if(col_block_dim <= 8)
        {
            launch_csr2gebsr_block_per_row_multipass_kernel(256, 64, 8);
        }
        else if(col_block_dim <= 16)
        {
            launch_csr2gebsr_block_per_row_multipass_kernel(256, 64, 16);
        }
        else
        {
            launch_csr2gebsr_block_per_row_multipass_kernel(256, 64, 32);
        }
    }
    else if(row_block_dim <= 64 && col_block_dim <= 64
            && !std::is_same<T, rocsparse_double_complex>())
    {
        rocsparse::csr2gebsr_64_64_launcher(handle,
                                            dir,
                                            m,
                                            n,
                                            mb,
                                            nb,
                                            csr_descr,
                                            csr_val,
                                            csr_row_ptr,
                                            csr_col_ind,
                                            bsr_descr,
                                            bsr_val,
                                            bsr_row_ptr,
                                            bsr_col_ind,
                                            row_block_dim,
                                            col_block_dim);
    }
    else
    {
        // Use a blocksize of 32 to handle each block row
        constexpr rocsparse_int block_size       = 32;
        rocsparse_int           rows_per_segment = (row_block_dim + block_size - 1) / block_size;

        size_t buffer_size = 0;
        buffer_size += sizeof(rocsparse_int)
                       * ((size_t(mb) * block_size * 3 * rows_per_segment - 1) / 256 + 1) * 256;
        buffer_size
            += sizeof(T) * ((size_t(mb) * block_size * rows_per_segment - 1) / 256 + 1) * 256;

        bool  temp_alloc       = false;
        void* temp_storage_ptr = nullptr;
        if(handle->buffer_size >= buffer_size)
        {
            temp_storage_ptr = handle->buffer;
            temp_alloc       = false;
        }
        else
        {
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMallocAsync(&temp_storage_ptr, buffer_size, handle->stream));
            temp_alloc = true;
        }

        char*          ptr   = reinterpret_cast<char*>(temp_storage_ptr);
        rocsparse_int* temp1 = reinterpret_cast<rocsparse_int*>(ptr);
        ptr += sizeof(rocsparse_int)
               * ((size_t(mb) * block_size * 3 * rows_per_segment - 1) / 256 + 1) * 256;
        T* temp2 = reinterpret_cast<T*>(ptr);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csr2gebsr_65_inf_kernel<block_size>),
                                           dim3(mb),
                                           dim3(block_size),
                                           0,
                                           handle->stream,
                                           dir,
                                           m,
                                           n,
                                           mb,
                                           nb,
                                           row_block_dim,
                                           col_block_dim,
                                           rows_per_segment,
                                           csr_descr->base,
                                           csr_val,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           bsr_descr->base,
                                           bsr_val,
                                           bsr_row_ptr,
                                           bsr_col_ind,
                                           temp1,
                                           temp2);

        if(temp_alloc)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(temp_storage_ptr, handle->stream));
        }
    }

    return rocsparse_status_success;
}

#define launch_csr2gebsr_nnz_block_per_row_multipass_kernel(block_size, blockdim)        \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                                  \
        (rocsparse::csr2gebsr_nnz_block_per_row_multipass_kernel<block_size, blockdim>), \
        dim3(mb),                                                                        \
        dim3(block_size),                                                                \
        0,                                                                               \
        handle->stream,                                                                  \
        m,                                                                               \
        n,                                                                               \
        mb,                                                                              \
        nb,                                                                              \
        row_block_dim,                                                                   \
        col_block_dim,                                                                   \
        csr_descr->base,                                                                 \
        csr_row_ptr,                                                                     \
        csr_col_ind,                                                                     \
        bsr_descr->base,                                                                 \
        bsr_row_ptr);

#define launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(block_size, row_blockdim, wf_size) \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                                            \
        (rocsparse::                                                                               \
             csr2gebsr_nnz_wavefront_per_row_multipass_kernel<block_size, row_blockdim, wf_size>), \
        dim3((mb - 1) / (block_size / wf_size) + 1),                                               \
        dim3(block_size),                                                                          \
        0,                                                                                         \
        handle->stream,                                                                            \
        m,                                                                                         \
        n,                                                                                         \
        mb,                                                                                        \
        nb,                                                                                        \
        row_block_dim,                                                                             \
        col_block_dim,                                                                             \
        csr_descr->base,                                                                           \
        csr_row_ptr,                                                                               \
        csr_col_ind,                                                                               \
        bsr_descr->base,                                                                           \
        bsr_row_ptr);

extern "C" rocsparse_status rocsparse_csr2gebsr_nnz(rocsparse_handle          handle, //0
                                                    rocsparse_direction       dir, //1
                                                    rocsparse_int             m, //2
                                                    rocsparse_int             n, //3
                                                    const rocsparse_mat_descr csr_descr, //4
                                                    const rocsparse_int*      csr_row_ptr, //5
                                                    const rocsparse_int*      csr_col_ind, //6
                                                    const rocsparse_mat_descr bsr_descr, //7
                                                    rocsparse_int*            bsr_row_ptr, //8
                                                    rocsparse_int             row_block_dim, //9
                                                    rocsparse_int             col_block_dim, //10
                                                    rocsparse_int*            bsr_nnz_devhost, //11
                                                    void*                     temp_buffer) //12
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_csr2gebsr_nnz",
                         dir,
                         m,
                         n,
                         csr_descr,
                         (const void*&)csr_row_ptr,
                         (const void*&)csr_col_ind,
                         bsr_descr,
                         (const void*&)bsr_row_ptr,
                         row_block_dim,
                         col_block_dim,
                         (const void*&)bsr_nnz_devhost,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, dir);
    ROCSPARSE_CHECKARG_SIZE(2, m);
    ROCSPARSE_CHECKARG_SIZE(3, n);
    ROCSPARSE_CHECKARG_POINTER(4, csr_descr);
    ROCSPARSE_CHECKARG(4,
                       csr_descr,
                       csr_descr->storage_mode != rocsparse_storage_mode_sorted,
                       rocsparse_status_requires_sorted_storage);
    ROCSPARSE_CHECKARG_ARRAY(5, m, csr_row_ptr);
    ROCSPARSE_CHECKARG_POINTER(7, bsr_descr);
    ROCSPARSE_CHECKARG(7,
                       bsr_descr,
                       bsr_descr->storage_mode != rocsparse_storage_mode_sorted,
                       rocsparse_status_requires_sorted_storage);
    ROCSPARSE_CHECKARG_ARRAY(8, m, bsr_row_ptr);
    ROCSPARSE_CHECKARG_SIZE(9, row_block_dim);
    ROCSPARSE_CHECKARG(9, row_block_dim, (row_block_dim == 0), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_SIZE(10, col_block_dim);
    ROCSPARSE_CHECKARG(10, col_block_dim, (col_block_dim == 0), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_POINTER(11, bsr_nnz_devhost);

    const rocsparse_int mb = (m + row_block_dim - 1) / row_block_dim;
    const rocsparse_int nb = (n + col_block_dim - 1) / col_block_dim;

    if(m == 0 || n == 0)
    {
        if(bsr_nnz_devhost != nullptr)
        {
            if(handle->pointer_mode == rocsparse_pointer_mode_device)
            {
                RETURN_IF_HIP_ERROR(
                    hipMemsetAsync(bsr_nnz_devhost, 0, sizeof(rocsparse_int), handle->stream));
            }
            else
            {
                *bsr_nnz_devhost = 0;
            }
        }

        if(bsr_row_ptr != nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::valset(
                handle, mb + 1, static_cast<rocsparse_int>(bsr_descr->base), bsr_row_ptr));
        }

        return rocsparse_status_success;
    }

    //
    // Check pointer arguments
    //
    if(csr_col_ind == nullptr)
    {
        rocsparse_int start = 0;
        rocsparse_int end   = 0;

        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &end, &csr_row_ptr[m], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &start, &csr_row_ptr[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

        const rocsparse_int nnz = (end - start);
        ROCSPARSE_CHECKARG_ARRAY(6, nnz, csr_col_ind);
    }

    if(row_block_dim == col_block_dim)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_csr2bsr_nnz(handle,
                                                        dir,
                                                        m,
                                                        n,
                                                        csr_descr,
                                                        csr_row_ptr,
                                                        csr_col_ind,
                                                        row_block_dim,
                                                        bsr_descr,
                                                        bsr_row_ptr,
                                                        bsr_nnz_devhost));
        return rocsparse_status_success;
    }

    if(row_block_dim == 1)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(rocsparse::csr2gebsr_nnz_kernel_bm1<256>,
                                           dim3((m / 256 + 1)),
                                           dim3(256),
                                           0,
                                           handle->stream,
                                           m,
                                           csr_descr->base,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           bsr_descr->base,
                                           bsr_row_ptr,
                                           col_block_dim);

        // Perform inclusive scan on bsr row pointer array
        size_t temp_storage_size_bytes;
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::primitives::inclusive_scan_buffer_size<rocsparse_int, rocsparse_int>(
                handle, mb + 1, &temp_storage_size_bytes)));

        bool  temp_alloc       = false;
        void* temp_storage_ptr = nullptr;
        if(handle->buffer_size >= temp_storage_size_bytes)
        {
            temp_storage_ptr = handle->buffer;
            temp_alloc       = false;
        }
        else
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(
                &temp_storage_ptr, temp_storage_size_bytes, handle->stream));
            temp_alloc = true;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::inclusive_scan(
            handle, bsr_row_ptr, bsr_row_ptr, mb + 1, temp_storage_size_bytes, temp_storage_ptr));

        if(temp_alloc)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(temp_storage_ptr, handle->stream));
        }
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(rocsparse::csr2gebsr_nnz_compute_nnz_total_kernel<1>,
                                               dim3(1),
                                               dim3(1),
                                               0,
                                               handle->stream,
                                               mb,
                                               bsr_row_ptr,
                                               bsr_nnz_devhost);
        }
        else
        {
            rocsparse_int hstart = 0;
            rocsparse_int hend   = 0;
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&hend,
                                               &bsr_row_ptr[mb],
                                               sizeof(rocsparse_int),
                                               hipMemcpyDeviceToHost,
                                               handle->stream));
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&hstart,
                                               &bsr_row_ptr[0],
                                               sizeof(rocsparse_int),
                                               hipMemcpyDeviceToHost,
                                               handle->stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

            *bsr_nnz_devhost = hend - hstart;
        }

        return rocsparse_status_success;
    }

    if(row_block_dim <= 2)
    {
        if(col_block_dim <= 2)
        {
            launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(256, 2, 4);
        }
        else if(col_block_dim <= 4)
        {
            launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(256, 2, 8);
        }
        else if(col_block_dim <= 8)
        {
            launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(256, 2, 16);
        }
        else if(col_block_dim <= 16)
        {
            launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(256, 2, 32);
        }
        else
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(256, 2, 64);
            }
            else
            {
                launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(256, 2, 32);
            }
        }
    }
    else if(row_block_dim <= 4)
    {
        if(col_block_dim <= 2)
        {
            launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(256, 4, 8);
        }
        else if(col_block_dim <= 4)
        {
            launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(256, 4, 16);
        }
        else if(col_block_dim <= 8)
        {
            launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(256, 4, 32);
        }
        else
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(256, 4, 64);
            }
            else
            {
                launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(256, 4, 32);
            }
        }
    }
    else if(row_block_dim <= 8)
    {
        if(col_block_dim <= 2)
        {
            launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(256, 8, 16);
        }
        else if(col_block_dim <= 4)
        {
            launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(256, 8, 32);
        }
        else
        {
            if(handle->wavefront_size == 64)
            {
                launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(256, 8, 64);
            }
            else
            {
                launch_csr2gebsr_nnz_wavefront_per_row_multipass_kernel(256, 8, 32);
            }
        }
    }
    else if(row_block_dim <= 16)
    {
        launch_csr2gebsr_nnz_block_per_row_multipass_kernel(256, 16);
    }
    else if(row_block_dim <= 32)
    {
        launch_csr2gebsr_nnz_block_per_row_multipass_kernel(256, 32);
    }
    else if(row_block_dim <= 64)
    {
        launch_csr2gebsr_nnz_block_per_row_multipass_kernel(256, 64);
    }
    else
    {
        // Use a blocksize of 32 to handle each block row
        constexpr rocsparse_int block_size       = 32;
        rocsparse_int           rows_per_segment = (row_block_dim + block_size - 1) / block_size;

        size_t buffer_size = sizeof(rocsparse_int)
                             * ((size_t(mb) * block_size * 2 * rows_per_segment - 1) / 256 + 1)
                             * 256;

        bool  temp_alloc       = false;
        void* temp_storage_ptr = nullptr;
        if(handle->buffer_size >= buffer_size)
        {
            temp_storage_ptr = handle->buffer;
            temp_alloc       = false;
        }
        else
        {
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMallocAsync(&temp_storage_ptr, buffer_size, handle->stream));
            temp_alloc = true;
        }

        rocsparse_int* temp1 = reinterpret_cast<rocsparse_int*>(temp_storage_ptr);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::csr2gebsr_nnz_65_inf_kernel<block_size>),
                                           dim3(mb),
                                           dim3(block_size),
                                           0,
                                           handle->stream,
                                           m,
                                           n,
                                           mb,
                                           nb,
                                           row_block_dim,
                                           col_block_dim,
                                           rows_per_segment,
                                           csr_descr->base,
                                           csr_row_ptr,
                                           csr_col_ind,
                                           bsr_descr->base,
                                           bsr_row_ptr,
                                           temp1);

        if(temp_alloc)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(temp_storage_ptr, handle->stream));
        }
    }

    // Perform inclusive scan on bsr row pointer array
    size_t temp_storage_size_bytes;
    RETURN_IF_ROCSPARSE_ERROR(
        (rocsparse::primitives::inclusive_scan_buffer_size<rocsparse_int, rocsparse_int>(
            handle, mb + 1, &temp_storage_size_bytes)));

    bool  temp_alloc       = false;
    void* temp_storage_ptr = nullptr;
    if(handle->buffer_size >= temp_storage_size_bytes)
    {
        temp_storage_ptr = handle->buffer;
        temp_alloc       = false;
    }
    else
    {
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMallocAsync(&temp_storage_ptr, temp_storage_size_bytes, handle->stream));
        temp_alloc = true;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::inclusive_scan(
        handle, bsr_row_ptr, bsr_row_ptr, mb + 1, temp_storage_size_bytes, temp_storage_ptr));

    if(temp_alloc)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(temp_storage_ptr, handle->stream));
    }

    // Compute bsr_nnz
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(rocsparse::csr2gebsr_nnz_compute_nnz_total_kernel<1>,
                                           dim3(1),
                                           dim3(1),
                                           0,
                                           handle->stream,
                                           mb,
                                           bsr_row_ptr,
                                           bsr_nnz_devhost);
    }
    else
    {
        rocsparse_int hstart = 0;
        rocsparse_int hend   = 0;
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &hend, &bsr_row_ptr[mb], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(&hstart,
                                           &bsr_row_ptr[0],
                                           sizeof(rocsparse_int),
                                           hipMemcpyDeviceToHost,
                                           handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

        *bsr_nnz_devhost = hend - hstart;
    }

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

//
// C INTERFACE
//
#define C_IMPL(NAME, TYPE)                                                                 \
    rocsparse_status NAME##_buffer_size(rocsparse_handle          handle,                  \
                                        rocsparse_direction       dir,                     \
                                        rocsparse_int             m,                       \
                                        rocsparse_int             n,                       \
                                        const rocsparse_mat_descr csr_descr,               \
                                        const TYPE*               csr_val,                 \
                                        const rocsparse_int*      csr_row_ptr,             \
                                        const rocsparse_int*      csr_col_ind,             \
                                        rocsparse_int             row_block_dim,           \
                                        rocsparse_int             col_block_dim,           \
                                        size_t*                   buffer_size)             \
    try                                                                                    \
    {                                                                                      \
        ROCSPARSE_ROUTINE_TRACE;                                                           \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2gebsr_buffer_size_template(handle,        \
                                                                            dir,           \
                                                                            m,             \
                                                                            n,             \
                                                                            csr_descr,     \
                                                                            csr_val,       \
                                                                            csr_row_ptr,   \
                                                                            csr_col_ind,   \
                                                                            row_block_dim, \
                                                                            col_block_dim, \
                                                                            buffer_size)); \
        return rocsparse_status_success;                                                   \
    }                                                                                      \
    catch(...)                                                                             \
    {                                                                                      \
        RETURN_ROCSPARSE_EXCEPTION();                                                      \
    }                                                                                      \
                                                                                           \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,                     \
                                     rocsparse_direction       dir,                        \
                                     rocsparse_int             m,                          \
                                     rocsparse_int             n,                          \
                                     const rocsparse_mat_descr csr_descr,                  \
                                     const TYPE*               csr_val,                    \
                                     const rocsparse_int*      csr_row_ptr,                \
                                     const rocsparse_int*      csr_col_ind,                \
                                     const rocsparse_mat_descr bsr_descr,                  \
                                     TYPE*                     bsr_val,                    \
                                     rocsparse_int*            bsr_row_ptr,                \
                                     rocsparse_int*            bsr_col_ind,                \
                                     rocsparse_int             row_block_dim,              \
                                     rocsparse_int             col_block_dim,              \
                                     void*                     temp_buffer)                \
                                                                                           \
    try                                                                                    \
    {                                                                                      \
        ROCSPARSE_ROUTINE_TRACE;                                                           \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2gebsr_template(handle,                    \
                                                                dir,                       \
                                                                m,                         \
                                                                n,                         \
                                                                csr_descr,                 \
                                                                csr_val,                   \
                                                                csr_row_ptr,               \
                                                                csr_col_ind,               \
                                                                bsr_descr,                 \
                                                                bsr_val,                   \
                                                                bsr_row_ptr,               \
                                                                bsr_col_ind,               \
                                                                row_block_dim,             \
                                                                col_block_dim,             \
                                                                temp_buffer));             \
        return rocsparse_status_success;                                                   \
    }                                                                                      \
    catch(...)                                                                             \
    {                                                                                      \
        RETURN_ROCSPARSE_EXCEPTION();                                                      \
    }

C_IMPL(rocsparse_scsr2gebsr, float);
C_IMPL(rocsparse_dcsr2gebsr, double);
C_IMPL(rocsparse_ccsr2gebsr, rocsparse_float_complex);
C_IMPL(rocsparse_zcsr2gebsr, rocsparse_double_complex);
#undef C_IMPL
