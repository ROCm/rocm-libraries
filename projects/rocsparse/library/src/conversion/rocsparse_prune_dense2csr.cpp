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

#include "internal/conversion/rocsparse_prune_dense2csr.h"
#include "rocsparse_control.hpp"
#include "rocsparse_prune_dense2csr.hpp"
#include "rocsparse_utility.hpp"

#include "csr2csr_compress_device.h"
#include "prune_dense2csr_device.h"
#include "rocsparse_common.h"
#include "rocsparse_primitives.hpp"

namespace rocsparse
{
    template <rocsparse_int DIM_X, rocsparse_int DIM_Y, typename T>
    ROCSPARSE_KERNEL(DIM_X* DIM_Y)
    void prune_dense2csr_nnz_kernel(rocsparse_int m,
                                    rocsparse_int n,
                                    const T* __restrict__ A,
                                    int64_t lda,
                                    ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, threshold),
                                    rocsparse_int* __restrict__ nnz_per_rows,
                                    bool is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(threshold);
        rocsparse::prune_dense2csr_nnz_device<DIM_X, DIM_Y>(m, n, A, lda, threshold, nnz_per_rows);
    }

    template <rocsparse_int NUMROWS_PER_BLOCK, rocsparse_int WF_SIZE, typename T>
    ROCSPARSE_KERNEL(WF_SIZE* NUMROWS_PER_BLOCK)
    void prune_dense2csr_kernel(rocsparse_index_base base,
                                rocsparse_int        m,
                                rocsparse_int        n,
                                const T* __restrict__ A,
                                int64_t lda,
                                ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, threshold),
                                T* __restrict__ csr_val,
                                const rocsparse_int* __restrict__ csr_row_ptr,
                                rocsparse_int* __restrict__ csr_col_ind,
                                bool is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(threshold);
        rocsparse::prune_dense2csr_device<NUMROWS_PER_BLOCK, WF_SIZE>(
            base, m, n, A, lda, threshold, csr_val, csr_row_ptr, csr_col_ind);
    }
}

template <typename T>
rocsparse_status
    rocsparse::prune_dense2csr_buffer_size_template(rocsparse_handle          handle, //0
                                                    rocsparse_int             m, //1
                                                    rocsparse_int             n, //2
                                                    const T*                  A, //3
                                                    int64_t                   lda, //4
                                                    const T*                  threshold, //5
                                                    const rocsparse_mat_descr descr, //6
                                                    const T*                  csr_val, //7
                                                    const rocsparse_int*      csr_row_ptr, //8
                                                    const rocsparse_int*      csr_col_ind, //9
                                                    size_t*                   buffer_size) //10
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xprune_dense2csr_buffer_size"),
                         m,
                         n,
                         (const void*&)A,
                         lda,
                         (const void*&)threshold,
                         descr,
                         (const void*&)csr_val,
                         (const void*&)csr_row_ptr,
                         (const void*&)csr_col_ind,
                         (const void*&)buffer_size);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_ARRAY(3, size_t(m) * n, A);
    ROCSPARSE_CHECKARG(4, lda, (lda < m), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_POINTER(5, threshold);
    ROCSPARSE_CHECKARG_POINTER(6, descr);
    ROCSPARSE_CHECKARG(6,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);
    ROCSPARSE_CHECKARG_ARRAY(8, m, csr_row_ptr);
    ROCSPARSE_CHECKARG_POINTER(10, buffer_size);

    *buffer_size = 0;

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse::prune_dense2csr_nnz_template(rocsparse_handle          handle, //0
                                                         rocsparse_int             m, //1
                                                         rocsparse_int             n, //2
                                                         const T*                  A, //3
                                                         int64_t                   lda, //4
                                                         const T*                  threshold, //5
                                                         const rocsparse_mat_descr descr, //6
                                                         rocsparse_int*            csr_row_ptr, //7
                                                         rocsparse_int* nnz_total_dev_host_ptr, //8
                                                         void*          temp_buffer) //9
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xprune_dense2csr_nnz"),
                         m,
                         n,
                         (const void*&)A,
                         lda,
                         (const void*&)threshold,
                         descr,
                         (const void*&)csr_row_ptr,
                         (const void*&)nnz_total_dev_host_ptr,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_ARRAY(3, size_t(m) * n, A);

    ROCSPARSE_CHECKARG(4, lda, (lda < m), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_POINTER(5, threshold);

    ROCSPARSE_CHECKARG_POINTER(6, descr);
    ROCSPARSE_CHECKARG(6,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    ROCSPARSE_CHECKARG_ARRAY(7, m, csr_row_ptr);

    hipStream_t stream = handle->stream;

    // Quick return if possible
    if(m == 0 || n == 0)
    {
        if(nnz_total_dev_host_ptr != nullptr && csr_row_ptr != nullptr)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::valset(
                handle, m + 1, static_cast<rocsparse_int>(descr->base), csr_row_ptr));

            if(handle->pointer_mode == rocsparse_pointer_mode_device)
            {
                RETURN_IF_HIP_ERROR(hipMemsetAsync(
                    nnz_total_dev_host_ptr, 0, sizeof(rocsparse_int), handle->stream));
            }
            else
            {
                *nnz_total_dev_host_ptr = 0;
            }
        }

        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_POINTER(8, nnz_total_dev_host_ptr);
    // ROCSPARSE_CHECKARG_POINTER(9, temp_buffer);

    static constexpr int NNZ_DIM_X = 64;
    static constexpr int NNZ_DIM_Y = 16;
    rocsparse_int        blocks    = (m - 1) / (NNZ_DIM_X * 4) + 1;

    dim3 grid(blocks);
    dim3 threads(NNZ_DIM_X, NNZ_DIM_Y);

    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
        (rocsparse::prune_dense2csr_nnz_kernel<NNZ_DIM_X, NNZ_DIM_Y, T>),
        grid,
        threads,
        0,
        stream,
        m,
        n,
        A,
        lda,
        ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, threshold),
        &csr_row_ptr[1],
        handle->pointer_mode == rocsparse_pointer_mode_host);

    // Compute csr_row_ptr with the right index base.
    rocsparse_int first_value = descr->base;
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        csr_row_ptr, &first_value, sizeof(rocsparse_int), hipMemcpyHostToDevice, handle->stream));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

    // Obtain rocprim buffer size
    size_t temp_storage_bytes = 0;
    RETURN_IF_ROCSPARSE_ERROR(
        (rocsparse::primitives::inclusive_scan_buffer_size<rocsparse_int, rocsparse_int>(
            handle, m + 1, &temp_storage_bytes)));

    // Get rocprim buffer
    bool  d_temp_alloc;
    void* d_temp_storage;

    // Device buffer should be sufficient for rocprim in most cases
    if(handle->buffer_size >= temp_storage_bytes)
    {
        d_temp_storage = handle->buffer;
        d_temp_alloc   = false;
    }
    else
    {
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMallocAsync(&d_temp_storage, temp_storage_bytes, handle->stream));
        d_temp_alloc = true;
    }

    // Perform actual inclusive sum
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::inclusive_scan(
        handle, csr_row_ptr, csr_row_ptr, m + 1, temp_storage_bytes, d_temp_storage));

    // Free rocprim buffer, if allocated
    if(d_temp_alloc == true)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(d_temp_storage, handle->stream));
    }

    // Extract nnz_total_dev_host_ptr
    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(rocsparse::nnz_total_device_kernel,
                                           dim3(1),
                                           dim3(1),
                                           0,
                                           stream,
                                           m,
                                           csr_row_ptr,
                                           nnz_total_dev_host_ptr);
    }
    else
    {
        rocsparse_int start, end;
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &start, &csr_row_ptr[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &end, &csr_row_ptr[m], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

        *nnz_total_dev_host_ptr = end - start;
    }

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse::prune_dense2csr_template(rocsparse_handle          handle, //0
                                                     rocsparse_int             m, //1
                                                     rocsparse_int             n, //2
                                                     const T*                  A, //3
                                                     int64_t                   lda, //4
                                                     const T*                  threshold, //5
                                                     const rocsparse_mat_descr descr, //6
                                                     T*                        csr_val, //7
                                                     const rocsparse_int*      csr_row_ptr, //8
                                                     rocsparse_int*            csr_col_ind, //9
                                                     void*                     temp_buffer) //10
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xprune_dense2csr"),
                         m,
                         n,
                         (const void*&)A,
                         lda,
                         (const void*&)threshold,
                         descr,
                         (const void*&)csr_val,
                         (const void*&)csr_row_ptr,
                         (const void*&)csr_col_ind,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);

    ROCSPARSE_CHECKARG(4, lda, (lda < m), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_POINTER(6, descr);
    ROCSPARSE_CHECKARG(6,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    ROCSPARSE_CHECKARG_ARRAY(8, m, csr_row_ptr);

    if(m == 0 || n == 0)
    {
        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_POINTER(3, A);
    ROCSPARSE_CHECKARG_POINTER(5, threshold);
    // ROCSPARSE_CHECKARG_POINTER(10, temp_buffer);

    if(csr_val == nullptr || csr_col_ind == nullptr)
    {
        rocsparse_int start = 0;
        rocsparse_int end   = 0;

        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &end, &csr_row_ptr[m], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &start, &csr_row_ptr[0], sizeof(rocsparse_int), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

        const rocsparse_int nnz = (end - start);
        ROCSPARSE_CHECKARG_ARRAY(7, nnz, csr_val);
        ROCSPARSE_CHECKARG_ARRAY(9, nnz, csr_col_ind);
    }

    // Stream
    hipStream_t stream = handle->stream;

    static constexpr rocsparse_int data_ratio = sizeof(T) / sizeof(float);

    if(handle->wavefront_size == 32)
    {
        static constexpr rocsparse_int WF_SIZE         = 32;
        static constexpr rocsparse_int NROWS_PER_BLOCK = 16 / (data_ratio > 0 ? data_ratio : 1);
        dim3                           blocks((m - 1) / NROWS_PER_BLOCK + 1);
        dim3                           threads(WF_SIZE * NROWS_PER_BLOCK);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::prune_dense2csr_kernel<NROWS_PER_BLOCK, WF_SIZE, T>),
            blocks,
            threads,
            0,
            stream,
            descr->base,
            m,
            n,
            A,
            lda,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, threshold),
            csr_val,
            csr_row_ptr,
            csr_col_ind,
            handle->pointer_mode == rocsparse_pointer_mode_host);
    }
    else
    {
        static constexpr rocsparse_int WF_SIZE         = 64;
        static constexpr rocsparse_int NROWS_PER_BLOCK = 16 / (data_ratio > 0 ? data_ratio : 1);
        dim3                           blocks((m - 1) / NROWS_PER_BLOCK + 1);
        dim3                           threads(WF_SIZE * NROWS_PER_BLOCK);
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::prune_dense2csr_kernel<NROWS_PER_BLOCK, WF_SIZE, T>),
            blocks,
            threads,
            0,
            stream,
            descr->base,
            m,
            n,
            A,
            lda,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, threshold),
            csr_val,
            csr_row_ptr,
            csr_col_ind,
            handle->pointer_mode == rocsparse_pointer_mode_host);
    }

    return rocsparse_status_success;
}

/*
* ===========================================================================
*    C wrapper
* ===========================================================================
*/

extern "C" rocsparse_status rocsparse_sprune_dense2csr_buffer_size(rocsparse_handle handle,
                                                                   rocsparse_int    m,
                                                                   rocsparse_int    n,
                                                                   const float*     A,
                                                                   rocsparse_int    lda,
                                                                   const float*     threshold,
                                                                   const rocsparse_mat_descr descr,
                                                                   const float*         csr_val,
                                                                   const rocsparse_int* csr_row_ptr,
                                                                   const rocsparse_int* csr_col_ind,
                                                                   size_t*              buffer_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::prune_dense2csr_buffer_size_template(
        handle, m, n, A, lda, threshold, descr, csr_val, csr_row_ptr, csr_col_ind, buffer_size));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_dprune_dense2csr_buffer_size(rocsparse_handle handle,
                                                                   rocsparse_int    m,
                                                                   rocsparse_int    n,
                                                                   const double*    A,
                                                                   rocsparse_int    lda,
                                                                   const double*    threshold,
                                                                   const rocsparse_mat_descr descr,
                                                                   const double*        csr_val,
                                                                   const rocsparse_int* csr_row_ptr,
                                                                   const rocsparse_int* csr_col_ind,
                                                                   size_t*              buffer_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::prune_dense2csr_buffer_size_template(
        handle, m, n, A, lda, threshold, descr, csr_val, csr_row_ptr, csr_col_ind, buffer_size));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_sprune_dense2csr_nnz(rocsparse_handle          handle,
                                                           rocsparse_int             m,
                                                           rocsparse_int             n,
                                                           const float*              A,
                                                           rocsparse_int             lda,
                                                           const float*              threshold,
                                                           const rocsparse_mat_descr descr,
                                                           rocsparse_int*            csr_row_ptr,
                                                           rocsparse_int* nnz_total_dev_host_ptr,
                                                           void*          temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::prune_dense2csr_nnz_template(
        handle, m, n, A, lda, threshold, descr, csr_row_ptr, nnz_total_dev_host_ptr, temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_dprune_dense2csr_nnz(rocsparse_handle          handle,
                                                           rocsparse_int             m,
                                                           rocsparse_int             n,
                                                           const double*             A,
                                                           rocsparse_int             lda,
                                                           const double*             threshold,
                                                           const rocsparse_mat_descr descr,
                                                           rocsparse_int*            csr_row_ptr,
                                                           rocsparse_int* nnz_total_dev_host_ptr,
                                                           void*          temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::prune_dense2csr_nnz_template(
        handle, m, n, A, lda, threshold, descr, csr_row_ptr, nnz_total_dev_host_ptr, temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_sprune_dense2csr(rocsparse_handle          handle,
                                                       rocsparse_int             m,
                                                       rocsparse_int             n,
                                                       const float*              A,
                                                       rocsparse_int             lda,
                                                       const float*              threshold,
                                                       const rocsparse_mat_descr descr,
                                                       float*                    csr_val,
                                                       const rocsparse_int*      csr_row_ptr,
                                                       rocsparse_int*            csr_col_ind,
                                                       void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::prune_dense2csr_template(
        handle, m, n, A, lda, threshold, descr, csr_val, csr_row_ptr, csr_col_ind, temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_dprune_dense2csr(rocsparse_handle          handle,
                                                       rocsparse_int             m,
                                                       rocsparse_int             n,
                                                       const double*             A,
                                                       rocsparse_int             lda,
                                                       const double*             threshold,
                                                       const rocsparse_mat_descr descr,
                                                       double*                   csr_val,
                                                       const rocsparse_int*      csr_row_ptr,
                                                       rocsparse_int*            csr_col_ind,
                                                       void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::prune_dense2csr_template(
        handle, m, n, A, lda, threshold, descr, csr_val, csr_row_ptr, csr_col_ind, temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
