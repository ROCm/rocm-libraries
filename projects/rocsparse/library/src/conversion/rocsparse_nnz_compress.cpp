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

#include "internal/conversion/rocsparse_nnz_compress.h"
#include "rocsparse_control.hpp"
#include "rocsparse_nnz_compress.hpp"
#include "rocsparse_utility.hpp"

#include "nnz_compress_device.h"
#include "rocsparse_primitives.hpp"

namespace rocsparse
{
    template <rocsparse_int BLOCK_SIZE,
              rocsparse_int SEGMENTS_PER_BLOCK,
              rocsparse_int SEGMENT_SIZE,
              typename T>
    ROCSPARSE_KERNEL(BLOCK_SIZE)
    void nnz_compress_kernel(rocsparse_int        m,
                             rocsparse_index_base idx_base_A,
                             const T* __restrict__ csr_val_A,
                             const rocsparse_int* __restrict__ csr_row_ptr_A,
                             rocsparse_int* __restrict__ nnz_per_row,
                             T threshold)
    {
        rocsparse::nnz_compress_device<BLOCK_SIZE, SEGMENTS_PER_BLOCK, SEGMENT_SIZE>(
            m, idx_base_A, csr_val_A, csr_row_ptr_A, nnz_per_row, threshold);
    }
}

template <typename T>
rocsparse_status rocsparse::nnz_compress_template(rocsparse_handle          handle, //0
                                                  rocsparse_int             m, //1
                                                  const rocsparse_mat_descr descr_A, //2
                                                  const T*                  csr_val_A, //3
                                                  const rocsparse_int*      csr_row_ptr_A, //4
                                                  rocsparse_int*            nnz_per_row, //5
                                                  rocsparse_int*            nnz_C, //6
                                                  T                         tol) //7
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xnnz_compress"),
                         m,
                         descr_A,
                         (const void*&)csr_val_A,
                         (const void*&)csr_row_ptr_A,
                         (const void*&)nnz_per_row,
                         (const void*&)nnz_C,
                         tol);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_POINTER(2, descr_A);
    ROCSPARSE_CHECKARG(2,
                       descr_A,
                       (descr_A->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    ROCSPARSE_CHECKARG(
        7, tol, (std::real(tol) < std::real(static_cast<T>(0))), rocsparse_status_invalid_value);

    // Quick return if possible
    if(m == 0)
    {
        if(nnz_C != nullptr)
        {
            if(handle->pointer_mode == rocsparse_pointer_mode_device)
            {
                RETURN_IF_HIP_ERROR(
                    hipMemsetAsync(nnz_C, 0, sizeof(rocsparse_int), handle->stream));
            }
            else
            {
                *nnz_C = 0;
            }
        }

        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_ARRAY(4, m, csr_row_ptr_A);
    ROCSPARSE_CHECKARG_POINTER(5, nnz_per_row);
    ROCSPARSE_CHECKARG_POINTER(6, nnz_C);

    // Find number of non-zeros in input CSR matrix on host
    rocsparse_int nnz_A, nnz_A_0;
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        &nnz_A, &csr_row_ptr_A[m], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        &nnz_A_0, &csr_row_ptr_A[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

    nnz_A -= nnz_A_0;
    if(nnz_A < 0)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_size);
    }

    // CSR values array can be nullptr if nnz_A is zero
    ROCSPARSE_CHECKARG_ARRAY(3, nnz_A, csr_val_A);

    // Stream
    hipStream_t stream = handle->stream;

    constexpr rocsparse_int block_size = 1024;

    // Mean number of elements per row in the input CSR matrix
    rocsparse_int mean_nnz_per_row = nnz_A / m;

    // A wavefront is divided into segments of size 2, 4, 8, 16, or 32 threads (or 64 in the case of 64
    // thread wavefronts) depending on the mean number of elements per CSR matrix row. Each row in the
    // matrix is then handled by a single segment.
    if(mean_nnz_per_row < 4)
    {
        constexpr rocsparse_int segments_per_block = block_size / 2;
        rocsparse_int           grid_size = (m + segments_per_block - 1) / segments_per_block;

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::nnz_compress_kernel<block_size, segments_per_block, 2>),
            dim3(grid_size),
            dim3(block_size),
            0,
            stream,
            m,
            descr_A->base,
            csr_val_A,
            csr_row_ptr_A,
            nnz_per_row,
            tol);
    }
    else if(mean_nnz_per_row < 8)
    {
        constexpr rocsparse_int segments_per_block = block_size / 4;
        rocsparse_int           grid_size = (m + segments_per_block - 1) / segments_per_block;

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::nnz_compress_kernel<block_size, segments_per_block, 4>),
            dim3(grid_size),
            dim3(block_size),
            0,
            stream,
            m,
            descr_A->base,
            csr_val_A,
            csr_row_ptr_A,
            nnz_per_row,
            tol);
    }
    else if(mean_nnz_per_row < 16)
    {
        constexpr rocsparse_int segments_per_block = block_size / 8;
        rocsparse_int           grid_size = (m + segments_per_block - 1) / segments_per_block;

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::nnz_compress_kernel<block_size, segments_per_block, 8>),
            dim3(grid_size),
            dim3(block_size),
            0,
            stream,
            m,
            descr_A->base,
            csr_val_A,
            csr_row_ptr_A,
            nnz_per_row,
            tol);
    }
    else if(mean_nnz_per_row < 32)
    {
        constexpr rocsparse_int segments_per_block = block_size / 16;
        rocsparse_int           grid_size = (m + segments_per_block - 1) / segments_per_block;

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::nnz_compress_kernel<block_size, segments_per_block, 16>),
            dim3(grid_size),
            dim3(block_size),
            0,
            stream,
            m,
            descr_A->base,
            csr_val_A,
            csr_row_ptr_A,
            nnz_per_row,
            tol);
    }
    else if(mean_nnz_per_row < 64 || handle->wavefront_size == 32)
    {
        constexpr rocsparse_int segments_per_block = block_size / 32;
        rocsparse_int           grid_size = (m + segments_per_block - 1) / segments_per_block;

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::nnz_compress_kernel<block_size, segments_per_block, 32>),
            dim3(grid_size),
            dim3(block_size),
            0,
            stream,
            m,
            descr_A->base,
            csr_val_A,
            csr_row_ptr_A,
            nnz_per_row,
            tol);
    }
    else
    {
        constexpr rocsparse_int segments_per_block = block_size / 64;
        rocsparse_int           grid_size = (m + segments_per_block - 1) / segments_per_block;

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::nnz_compress_kernel<block_size, segments_per_block, 64>),
            dim3(grid_size),
            dim3(block_size),
            0,
            stream,
            m,
            descr_A->base,
            csr_val_A,
            csr_row_ptr_A,
            nnz_per_row,
            tol);
    }

    rocsparse_int* dnnz_C;
    if(handle->pointer_mode == rocsparse_pointer_mode_host)
    {
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMallocAsync(&dnnz_C, sizeof(rocsparse_int), handle->stream));
    }
    else
    {
        dnnz_C = nnz_C;
    }

    // Perform inclusive scan on csr row pointer array
    size_t temp_storage_size_bytes;
    RETURN_IF_ROCSPARSE_ERROR(
        (rocsparse::primitives::find_sum_buffer_size<rocsparse_int, rocsparse_int>(
            handle, m, &temp_storage_size_bytes)));

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

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::find_sum(
        handle, nnz_per_row, dnnz_C, m, temp_storage_size_bytes, temp_storage_ptr));

    if(handle->pointer_mode == rocsparse_pointer_mode_host)
    {
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            nnz_C, dnnz_C, sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(dnnz_C, handle->stream));
    }

    if(temp_alloc)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(temp_storage_ptr, handle->stream));
    }

    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_snnz_compress(rocsparse_handle          handle,
                                                    rocsparse_int             m,
                                                    const rocsparse_mat_descr descr_A,
                                                    const float*              csr_val_A,
                                                    const rocsparse_int*      csr_row_ptr_A,
                                                    rocsparse_int*            nnz_per_row,
                                                    rocsparse_int*            nnz_C,
                                                    float                     tol)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::nnz_compress_template(
        handle, m, descr_A, csr_val_A, csr_row_ptr_A, nnz_per_row, nnz_C, tol));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_dnnz_compress(rocsparse_handle          handle,
                                                    rocsparse_int             m,
                                                    const rocsparse_mat_descr descr_A,
                                                    const double*             csr_val_A,
                                                    const rocsparse_int*      csr_row_ptr_A,
                                                    rocsparse_int*            nnz_per_row,
                                                    rocsparse_int*            nnz_C,
                                                    double                    tol)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::nnz_compress_template(
        handle, m, descr_A, csr_val_A, csr_row_ptr_A, nnz_per_row, nnz_C, tol));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_cnnz_compress(rocsparse_handle               handle,
                                                    rocsparse_int                  m,
                                                    const rocsparse_mat_descr      descr_A,
                                                    const rocsparse_float_complex* csr_val_A,
                                                    const rocsparse_int*           csr_row_ptr_A,
                                                    rocsparse_int*                 nnz_per_row,
                                                    rocsparse_int*                 nnz_C,
                                                    rocsparse_float_complex        tol)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::nnz_compress_template(
        handle, m, descr_A, csr_val_A, csr_row_ptr_A, nnz_per_row, nnz_C, tol));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_znnz_compress(rocsparse_handle                handle,
                                                    rocsparse_int                   m,
                                                    const rocsparse_mat_descr       descr_A,
                                                    const rocsparse_double_complex* csr_val_A,
                                                    const rocsparse_int*            csr_row_ptr_A,
                                                    rocsparse_int*                  nnz_per_row,
                                                    rocsparse_int*                  nnz_C,
                                                    rocsparse_double_complex        tol)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::nnz_compress_template(
        handle, m, descr_A, csr_val_A, csr_row_ptr_A, nnz_per_row, nnz_C, tol));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
