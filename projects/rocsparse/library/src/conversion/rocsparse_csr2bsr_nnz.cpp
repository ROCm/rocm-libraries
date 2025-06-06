/*! \file */
/* ************************************************************************
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/conversion/rocsparse_csr2bsr.h"
#include "rocsparse_common.h"
#include "rocsparse_control.hpp"
#include "rocsparse_csr2bsr.hpp"
#include "rocsparse_utility.hpp"

#include "csr2bsr_nnz_device.h"

#include "rocsparse_primitives.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
#define launch_csr2bsr_nnz_wavefront_per_row_multipass_kernel(blocksize, wfsize, blockdim)        \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                                           \
        (rocsparse::csr2bsr_nnz_wavefront_per_row_multipass_kernel<blocksize, wfsize, blockdim>), \
        dim3((mb - 1) / (blocksize / wfsize) + 1),                                                \
        dim3(blocksize),                                                                          \
        0,                                                                                        \
        handle->stream,                                                                           \
        m,                                                                                        \
        n,                                                                                        \
        mb,                                                                                       \
        nb,                                                                                       \
        block_dim,                                                                                \
        csr_descr->base,                                                                          \
        csr_row_ptr,                                                                              \
        csr_col_ind,                                                                              \
        bsr_descr->base,                                                                          \
        bsr_row_ptr);

#define launch_csr2bsr_nnz_block_per_row_multipass_kernel(blocksize, blockdim)        \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                               \
        (rocsparse::csr2bsr_nnz_block_per_row_multipass_kernel<blocksize, blockdim>), \
        dim3(mb),                                                                     \
        dim3(blocksize),                                                              \
        0,                                                                            \
        handle->stream,                                                               \
        m,                                                                            \
        n,                                                                            \
        mb,                                                                           \
        nb,                                                                           \
        block_dim,                                                                    \
        csr_descr->base,                                                              \
        csr_row_ptr,                                                                  \
        csr_col_ind,                                                                  \
        bsr_descr->base,                                                              \
        bsr_row_ptr);

template <typename I, typename J>
rocsparse_status rocsparse::csr2bsr_nnz_quickreturn(rocsparse_handle          handle,
                                                    rocsparse_direction       dir,
                                                    J                         m,
                                                    J                         n,
                                                    const rocsparse_mat_descr csr_descr,
                                                    const I*                  csr_row_ptr,
                                                    const J*                  csr_col_ind,
                                                    J                         block_dim,
                                                    const rocsparse_mat_descr bsr_descr,
                                                    I*                        bsr_row_ptr,
                                                    I*                        bsr_nnz)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(m == 0 || n == 0)
    {
        if(bsr_nnz != nullptr)
        {
            if(handle->pointer_mode == rocsparse_pointer_mode_device)
            {
                RETURN_IF_HIP_ERROR(hipMemsetAsync(bsr_nnz, 0, sizeof(I), handle->stream));
            }
            else
            {
                *bsr_nnz = 0;
            }
        }

        if(bsr_row_ptr != nullptr)
        {
            const J mb = (m + block_dim - 1) / block_dim;
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::valset(handle, mb + 1, static_cast<I>(bsr_descr->base), bsr_row_ptr));
        }

        return rocsparse_status_success;
    }

    return rocsparse_status_continue;
}

namespace rocsparse
{
    template <typename I, typename J>
    static rocsparse_status csr2bsr_nnz_checkarg(rocsparse_handle          handle, //0
                                                 rocsparse_direction       dir, //1
                                                 J                         m, //2
                                                 J                         n, //3
                                                 const rocsparse_mat_descr csr_descr, //4
                                                 const I*                  csr_row_ptr, //5
                                                 const J*                  csr_col_ind, //6
                                                 J                         block_dim, //7
                                                 const rocsparse_mat_descr bsr_descr, //8
                                                 I*                        bsr_row_ptr, //9
                                                 I*                        bsr_nnz) //10
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_ENUM(1, dir);
        ROCSPARSE_CHECKARG_SIZE(2, m);
        ROCSPARSE_CHECKARG_SIZE(3, n);
        ROCSPARSE_CHECKARG_SIZE(7, block_dim);
        ROCSPARSE_CHECKARG(7, block_dim, (block_dim == 0), rocsparse_status_invalid_size);

        const rocsparse_status status = rocsparse::csr2bsr_nnz_quickreturn(handle,
                                                                           dir,
                                                                           m,
                                                                           n,
                                                                           csr_descr,
                                                                           csr_row_ptr,
                                                                           csr_col_ind,
                                                                           block_dim,
                                                                           bsr_descr,
                                                                           bsr_row_ptr,
                                                                           bsr_nnz);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_POINTER(4, csr_descr);
        ROCSPARSE_CHECKARG(4,
                           csr_descr,
                           (csr_descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);
        ROCSPARSE_CHECKARG_POINTER(8, bsr_descr);
        ROCSPARSE_CHECKARG(8,
                           bsr_descr,
                           (bsr_descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);

        const J mb = (m + block_dim - 1) / block_dim;

        ROCSPARSE_CHECKARG_ARRAY(5, m, csr_row_ptr);
        ROCSPARSE_CHECKARG_ARRAY(9, mb, bsr_row_ptr);
        ROCSPARSE_CHECKARG_POINTER(10, bsr_nnz);

        if(csr_col_ind == nullptr)
        {
            I start = 0;
            I end   = 0;

            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                &end, &csr_row_ptr[m], sizeof(I), hipMemcpyDeviceToHost, handle->stream));
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                &start, &csr_row_ptr[0], sizeof(I), hipMemcpyDeviceToHost, handle->stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

            const I nnz = (end - start);
            ROCSPARSE_CHECKARG_ARRAY(6, nnz, csr_col_ind);
        }
        return rocsparse_status_continue;
    }
}

template <typename I, typename J>
rocsparse_status rocsparse::csr2bsr_nnz_core(rocsparse_handle          handle,
                                             rocsparse_direction       dir,
                                             J                         m,
                                             J                         n,
                                             const rocsparse_mat_descr csr_descr,
                                             const I*                  csr_row_ptr,
                                             const J*                  csr_col_ind,
                                             J                         block_dim,
                                             const rocsparse_mat_descr bsr_descr,
                                             I*                        bsr_row_ptr,
                                             I*                        bsr_nnz)
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, dir);
    ROCSPARSE_CHECKARG_SIZE(2, m);
    ROCSPARSE_CHECKARG_SIZE(3, n);
    ROCSPARSE_CHECKARG_SIZE(7, block_dim);
    ROCSPARSE_CHECKARG(7, block_dim, (block_dim == 0), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_POINTER(4, csr_descr);
    ROCSPARSE_CHECKARG(4,
                       csr_descr,
                       (csr_descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);
    ROCSPARSE_CHECKARG_POINTER(8, bsr_descr);
    ROCSPARSE_CHECKARG(8,
                       bsr_descr,
                       (bsr_descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    const J mb = (m + block_dim - 1) / block_dim;
    const J nb = (n + block_dim - 1) / block_dim;

    // Quick return if possible
    if(m == 0 || n == 0)
    {
        if(bsr_nnz != nullptr)
        {
            if(handle->pointer_mode == rocsparse_pointer_mode_device)
            {
                RETURN_IF_HIP_ERROR(hipMemsetAsync(bsr_nnz, 0, sizeof(I), handle->stream));
            }
            else
            {
                *bsr_nnz = 0;
            }
        }

        if(bsr_row_ptr != nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::valset(handle, mb + 1, static_cast<I>(bsr_descr->base), bsr_row_ptr));
        }

        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_ARRAY(5, m, csr_row_ptr);
    ROCSPARSE_CHECKARG_ARRAY(9, mb, bsr_row_ptr);
    ROCSPARSE_CHECKARG_POINTER(10, bsr_nnz);

    if(csr_col_ind == nullptr)
    {
        I start = 0;
        I end   = 0;

        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &end, &csr_row_ptr[m], sizeof(I), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &start, &csr_row_ptr[0], sizeof(I), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

        const I nnz = (end - start);

        ROCSPARSE_CHECKARG_ARRAY(6, nnz, csr_col_ind);
    }

    // If block dimension is one then BSR is equal to CSR
    if(block_dim == 1)
    {
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((csr2bsr_nnz_block_dim_equals_one_kernel<256, I, J>),
                                               dim3(m / 256 + 1),
                                               dim3(256),
                                               0,
                                               handle->stream,
                                               m,
                                               csr_descr->base,
                                               csr_row_ptr,
                                               bsr_descr->base,
                                               bsr_row_ptr,
                                               bsr_nnz);
        }
        else
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((csr2bsr_nnz_block_dim_equals_one_kernel<256, I, J>),
                                               dim3(m / 256 + 1),
                                               dim3(256),
                                               0,
                                               handle->stream,
                                               m,
                                               csr_descr->base,
                                               csr_row_ptr,
                                               bsr_descr->base,
                                               bsr_row_ptr);

            I start = 0;
            I end   = 0;
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                &end, &bsr_row_ptr[mb], sizeof(I), hipMemcpyDeviceToHost, handle->stream));
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                &start, &bsr_row_ptr[0], sizeof(I), hipMemcpyDeviceToHost, handle->stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

            *bsr_nnz = end - start;
        }

        return rocsparse_status_success;
    }

    if(block_dim <= 4)
    {
        launch_csr2bsr_nnz_wavefront_per_row_multipass_kernel(256, 16, 4);
    }
    else if(block_dim <= 8)
    {
        if(handle->wavefront_size == 64)
        {
            launch_csr2bsr_nnz_wavefront_per_row_multipass_kernel(256, 64, 8);
        }
        else
        {
            launch_csr2bsr_nnz_wavefront_per_row_multipass_kernel(256, 32, 8);
        }
    }
    else if(block_dim <= 16)
    {
        if(handle->wavefront_size == 64)
        {
            launch_csr2bsr_nnz_wavefront_per_row_multipass_kernel(256, 64, 16);
        }
        else
        {
            launch_csr2bsr_nnz_wavefront_per_row_multipass_kernel(256, 32, 16);
        }
    }
    else if(block_dim <= 32)
    {
        launch_csr2bsr_nnz_block_per_row_multipass_kernel(256, 32);
    }
    else if(block_dim <= 64)
    {
        launch_csr2bsr_nnz_block_per_row_multipass_kernel(256, 64);
    }
    else
    {
        // Use a blocksize of 32 to handle each block row
        constexpr J block_size       = 32;
        J           rows_per_segment = (block_dim + block_size - 1) / block_size;

        size_t buffer_size
            = sizeof(I) * ((size_t(mb) * block_size * 2 * rows_per_segment - 1) / 256 + 1) * 256;

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

        I* temp1 = reinterpret_cast<I*>(temp_storage_ptr);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((csr2bsr_nnz_65_inf_kernel<block_size, I, J>),
                                           dim3(mb),
                                           dim3(block_size),
                                           0,
                                           handle->stream,
                                           m,
                                           n,
                                           mb,
                                           nb,
                                           block_dim,
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
    RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::inclusive_scan_buffer_size<I, I>(
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
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(csr2bsr_nnz_compute_nnz_total_kernel<1>,
                                           dim3(1),
                                           dim3(1),
                                           0,
                                           handle->stream,
                                           mb,
                                           bsr_row_ptr,
                                           bsr_nnz);
    }
    else
    {
        I start = 0;
        I end   = 0;
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &end, &bsr_row_ptr[mb], sizeof(I), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &start, &bsr_row_ptr[0], sizeof(I), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

        *bsr_nnz = end - start;
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(I, J)                                                                        \
    template rocsparse_status rocsparse::csr2bsr_nnz_core(rocsparse_handle          handle,      \
                                                          rocsparse_direction       dir,         \
                                                          J                         m,           \
                                                          J                         n,           \
                                                          const rocsparse_mat_descr csr_descr,   \
                                                          const I*                  csr_row_ptr, \
                                                          const J*                  csr_col_ind, \
                                                          J                         block_dim,   \
                                                          const rocsparse_mat_descr bsr_descr,   \
                                                          I*                        bsr_row_ptr, \
                                                          I*                        bsr_nnz);                           \
    template rocsparse_status rocsparse::csr2bsr_nnz_quickreturn(                                \
        rocsparse_handle          handle,                                                        \
        rocsparse_direction       dir,                                                           \
        J                         m,                                                             \
        J                         n,                                                             \
        const rocsparse_mat_descr csr_descr,                                                     \
        const I*                  csr_row_ptr,                                                   \
        const J*                  csr_col_ind,                                                   \
        J                         block_dim,                                                     \
        const rocsparse_mat_descr bsr_descr,                                                     \
        I*                        bsr_row_ptr,                                                   \
        I*                        bsr_nnz)

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int64_t, int32_t);
INSTANTIATE(int32_t, int64_t);
INSTANTIATE(int64_t, int64_t);
#undef INSTANTIATE

namespace rocsparse
{
    template <typename... P>
    static rocsparse_status csr2bsr_nnz_impl(P&&... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse::log_trace("rocsparse_csr2bsr_nnz", p...);
        const rocsparse_status status = rocsparse::csr2bsr_nnz_checkarg(p...);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2bsr_nnz_core(p...));
        return rocsparse_status_success;
    }
}

extern "C" rocsparse_status rocsparse_csr2bsr_nnz(rocsparse_handle          handle,
                                                  rocsparse_direction       dir,
                                                  rocsparse_int             m,
                                                  rocsparse_int             n,
                                                  const rocsparse_mat_descr csr_descr,
                                                  const rocsparse_int*      csr_row_ptr,
                                                  const rocsparse_int*      csr_col_ind,
                                                  rocsparse_int             block_dim,
                                                  const rocsparse_mat_descr bsr_descr,
                                                  rocsparse_int*            bsr_row_ptr,
                                                  rocsparse_int*            bsr_nnz)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2bsr_nnz_impl(handle,
                                                          dir,
                                                          m,
                                                          n,
                                                          csr_descr,
                                                          csr_row_ptr,
                                                          csr_col_ind,
                                                          block_dim,
                                                          bsr_descr,
                                                          bsr_row_ptr,
                                                          bsr_nnz));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
