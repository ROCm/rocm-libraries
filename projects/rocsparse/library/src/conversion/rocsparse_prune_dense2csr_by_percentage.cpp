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

#include "internal/conversion/rocsparse_prune_dense2csr_by_percentage.h"
#include "rocsparse_control.hpp"
#include "rocsparse_prune_dense2csr_by_percentage.hpp"
#include "rocsparse_utility.hpp"

#include "csr2csr_compress_device.h"
#include "prune_dense2csr_by_percentage_device.h"
#include "prune_dense2csr_device.h"
#include "rocsparse_common.h"
#include "rocsparse_primitives.hpp"

namespace rocsparse
{
    template <rocsparse_int DIM_X, rocsparse_int DIM_Y, typename T>
    ROCSPARSE_KERNEL(DIM_X* DIM_Y)
    void prune_dense2csr_nnz_kernel2(rocsparse_int m,
                                     rocsparse_int n,
                                     const T* __restrict__ A,
                                     int64_t  lda,
                                     const T* threshold,
                                     rocsparse_int* __restrict__ nnz_per_rows)
    {
        rocsparse::prune_dense2csr_nnz_device<DIM_X, DIM_Y>(m, n, A, lda, *threshold, nnz_per_rows);
    }

    template <rocsparse_int NUMROWS_PER_BLOCK, rocsparse_int WF_SIZE, typename T>
    ROCSPARSE_KERNEL(WF_SIZE* NUMROWS_PER_BLOCK)
    void prune_dense2csr_kernel2_device_pointer(rocsparse_index_base base,
                                                rocsparse_int        m,
                                                rocsparse_int        n,
                                                const T* __restrict__ A,
                                                int64_t  lda,
                                                const T* threshold,
                                                T* __restrict__ csr_val,
                                                const rocsparse_int* __restrict__ csr_row_ptr,
                                                rocsparse_int* __restrict__ csr_col_ind)
    {
        rocsparse::prune_dense2csr_device<NUMROWS_PER_BLOCK, WF_SIZE>(
            base, m, n, A, lda, *threshold, csr_val, csr_row_ptr, csr_col_ind);
    }
}

template <typename T>
rocsparse_status rocsparse::prune_dense2csr_by_percentage_buffer_size_template(
    rocsparse_handle          handle, //0
    rocsparse_int             m, //1
    rocsparse_int             n, //2
    const T*                  A, //3
    int64_t                   lda, //4
    T                         percentage, //5
    const rocsparse_mat_descr descr, //6
    const T*                  csr_val, //7
    const rocsparse_int*      csr_row_ptr, //8
    const rocsparse_int*      csr_col_ind, //9
    rocsparse_mat_info        info, //10
    size_t*                   buffer_size) //11
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(
        handle,
        rocsparse::replaceX<T>("rocsparse_Xprune_dense2csr_by_percentage_buffer_size"),
        m,
        n,
        (const void*&)A,
        lda,
        percentage,
        descr,
        (const void*&)csr_val,
        (const void*&)csr_row_ptr,
        (const void*&)csr_col_ind,
        info,
        (const void*&)buffer_size);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_ARRAY(3, size_t(m) * n, A);

    ROCSPARSE_CHECKARG(4, lda, (lda < m), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG(5,
                       percentage,
                       (percentage < static_cast<T>(0.0) || percentage > static_cast<T>(100.0)),
                       rocsparse_status_invalid_value);

    ROCSPARSE_CHECKARG_POINTER(6, descr);
    ROCSPARSE_CHECKARG(6,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    // ROCSPARSE_CHECKARG_POINTER(7, csr_val);
    ROCSPARSE_CHECKARG_ARRAY(8, m, csr_row_ptr);
    // ROCSPARSE_CHECKARG_POINTER(9, csr_col_ind);

    ROCSPARSE_CHECKARG_POINTER(10, info);
    ROCSPARSE_CHECKARG_POINTER(11, buffer_size);

    *buffer_size = sizeof(T) * 2 * m * n;
    return rocsparse_status_success;
}

template <typename T>
rocsparse_status
    rocsparse::prune_dense2csr_nnz_by_percentage_template(rocsparse_handle          handle, //0
                                                          rocsparse_int             m, //1
                                                          rocsparse_int             n, //2
                                                          const T*                  A, //3
                                                          int64_t                   lda, //4
                                                          T                         percentage, //5
                                                          const rocsparse_mat_descr descr, //6
                                                          rocsparse_int*            csr_row_ptr, //7
                                                          rocsparse_int* nnz_total_dev_host_ptr, //8
                                                          rocsparse_mat_info info, //9
                                                          void*              temp_buffer) //10
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xprune_dense2csr_nnz_by_percentage"),
                         m,
                         n,
                         (const void*&)A,
                         lda,
                         (const void*&)percentage,
                         descr,
                         (const void*&)csr_row_ptr,
                         (const void*&)nnz_total_dev_host_ptr,
                         info,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_ARRAY(3, size_t(m) * n, A);

    ROCSPARSE_CHECKARG(4, lda, (lda < m), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG(5,
                       percentage,
                       (percentage < static_cast<T>(0.0) || percentage > static_cast<T>(100.0)),
                       rocsparse_status_invalid_value);

    ROCSPARSE_CHECKARG_POINTER(6, descr);
    ROCSPARSE_CHECKARG(6,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    ROCSPARSE_CHECKARG_ARRAY(7, m, csr_row_ptr);
    ROCSPARSE_CHECKARG_POINTER(9, info);

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
    ROCSPARSE_CHECKARG_POINTER(10, temp_buffer);

    const rocsparse_int nnz_A = m * n;

    rocsparse_int pos = rocsparse::ceil(nnz_A * (percentage / 100)) - 1;
    pos               = rocsparse::min(pos, nnz_A - 1);
    pos               = rocsparse::max(pos, static_cast<rocsparse_int>(0));

    T* output = reinterpret_cast<T*>(temp_buffer);

    // Compute absolute value of A and store in first half of output array
    {
        dim3 grid((nnz_A - 1) / 256 + 1);
        dim3 threads(256);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (abs_kernel<256, T>), grid, threads, 0, stream, m, n, A, lda, output);
    }

    // Determine amount of temporary storage needed for rocprim sort and inclusive scan and allocate if necessary
    size_t temp_storage_size_bytes_sort = 0;
    size_t temp_storage_size_bytes_scan = 0;

    uint32_t startbit = 0;
    uint32_t endbit   = 8 * sizeof(T);
    RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::radix_sort_keys_buffer_size<T>(
        handle, nnz_A, startbit, endbit, &temp_storage_size_bytes_sort)));

    RETURN_IF_ROCSPARSE_ERROR(
        (rocsparse::primitives::inclusive_scan_buffer_size<rocsparse_int, rocsparse_int>(
            handle, m + 1, &temp_storage_size_bytes_scan)));

    size_t temp_storage_size_bytes
        = rocsparse::max(temp_storage_size_bytes_sort, temp_storage_size_bytes_scan);

    // Device buffer should be sufficient for rocprim in most cases
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

    // perform sort on first half of output array and store result in second half of output array
    rocsparse::primitives::double_buffer<T> keys(output, output + nnz_A);
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::radix_sort_keys(
        handle, keys, nnz_A, startbit, endbit, temp_storage_size_bytes_sort, temp_storage_ptr));

    const T* d_threshold = keys.current() + pos;

    static constexpr int NNZ_DIM_X = 64;
    static constexpr int NNZ_DIM_Y = 16;

    {
        dim3 grid((m - 1) / (NNZ_DIM_X * 4) + 1);
        dim3 threads(NNZ_DIM_X, NNZ_DIM_Y);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::prune_dense2csr_nnz_kernel2<NNZ_DIM_X, NNZ_DIM_Y>),
            grid,
            threads,
            0,
            stream,
            m,
            n,
            A,
            lda,
            d_threshold,
            &csr_row_ptr[1]);
    }

    // Store threshold at first entry in output array
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(output, d_threshold, sizeof(T), hipMemcpyDeviceToDevice, handle->stream));

    // Compute csr_row_ptr with the right index base.
    rocsparse_int first_value = descr->base;
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        csr_row_ptr, &first_value, sizeof(rocsparse_int), hipMemcpyHostToDevice, handle->stream));
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

    // Perform actual inclusive sum
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::inclusive_scan(
        handle, csr_row_ptr, csr_row_ptr, m + 1, temp_storage_size_bytes_scan, temp_storage_ptr));

    // Free rocprim buffer, if allocated
    if(temp_alloc == true)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(temp_storage_ptr, handle->stream));
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
rocsparse_status
    rocsparse::prune_dense2csr_by_percentage_template(rocsparse_handle          handle, //0
                                                      rocsparse_int             m, //1
                                                      rocsparse_int             n, //2
                                                      const T*                  A, //3
                                                      int64_t                   lda, //4
                                                      T                         percentage, //5
                                                      const rocsparse_mat_descr descr, //6
                                                      T*                        csr_val, //7
                                                      const rocsparse_int*      csr_row_ptr, //8
                                                      rocsparse_int*            csr_col_ind, //9
                                                      rocsparse_mat_info        info, //10
                                                      void*                     temp_buffer) //11
{
    ROCSPARSE_ROUTINE_TRACE;

    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xprune_dense2csr_by_percentage"),
                         m,
                         n,
                         (const void*&)A,
                         lda,
                         percentage,
                         descr,
                         (const void*&)csr_val,
                         (const void*&)csr_row_ptr,
                         (const void*&)csr_col_ind,
                         info,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);

    ROCSPARSE_CHECKARG(4, lda, (lda < m), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG(5,
                       percentage,
                       (percentage < static_cast<T>(0.0) || percentage > static_cast<T>(100.0)),
                       rocsparse_status_invalid_value);

    ROCSPARSE_CHECKARG_POINTER(6, descr);
    ROCSPARSE_CHECKARG(6,
                       descr,
                       (descr->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);

    ROCSPARSE_CHECKARG_ARRAY(8, m, csr_row_ptr);

    // Quick return if possible
    if(m == 0 || n == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    ROCSPARSE_CHECKARG_POINTER(3, A);
    ROCSPARSE_CHECKARG_POINTER(10, info);
    ROCSPARSE_CHECKARG_POINTER(11, temp_buffer);

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

    T* d_threshold = &(reinterpret_cast<T*>(temp_buffer))[0];

    static constexpr rocsparse_int data_ratio = sizeof(T) / sizeof(float);

    if(handle->wavefront_size == 32)
    {
        static constexpr rocsparse_int WF_SIZE         = 32;
        static constexpr rocsparse_int NROWS_PER_BLOCK = 16 / (data_ratio > 0 ? data_ratio : 1);
        dim3                           blocks((m - 1) / NROWS_PER_BLOCK + 1);
        dim3                           threads(WF_SIZE * NROWS_PER_BLOCK);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::prune_dense2csr_kernel2_device_pointer<NROWS_PER_BLOCK, WF_SIZE, T>),
            blocks,
            threads,
            0,
            stream,
            descr->base,
            m,
            n,
            A,
            lda,
            d_threshold,
            csr_val,
            csr_row_ptr,
            csr_col_ind);
    }
    else
    {
        static constexpr rocsparse_int WF_SIZE         = 64;
        static constexpr rocsparse_int NROWS_PER_BLOCK = 16 / (data_ratio > 0 ? data_ratio : 1);
        dim3                           blocks((m - 1) / NROWS_PER_BLOCK + 1);
        dim3                           threads(WF_SIZE * NROWS_PER_BLOCK);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::prune_dense2csr_kernel2_device_pointer<NROWS_PER_BLOCK, WF_SIZE, T>),
            blocks,
            threads,
            0,
            stream,
            descr->base,
            m,
            n,
            A,
            lda,
            d_threshold,
            csr_val,
            csr_row_ptr,
            csr_col_ind);
    }

    return rocsparse_status_success;
}

/*
* ===========================================================================
*    C wrapper
* ===========================================================================
*/

extern "C" rocsparse_status
    rocsparse_sprune_dense2csr_by_percentage_buffer_size(rocsparse_handle          handle,
                                                         rocsparse_int             m,
                                                         rocsparse_int             n,
                                                         const float*              A,
                                                         rocsparse_int             lda,
                                                         float                     percentage,
                                                         const rocsparse_mat_descr descr,
                                                         const float*              csr_val,
                                                         const rocsparse_int*      csr_row_ptr,
                                                         const rocsparse_int*      csr_col_ind,
                                                         rocsparse_mat_info        info,
                                                         size_t*                   buffer_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::prune_dense2csr_by_percentage_buffer_size_template(handle,
                                                                      m,
                                                                      n,
                                                                      A,
                                                                      lda,
                                                                      percentage,
                                                                      descr,
                                                                      csr_val,
                                                                      csr_row_ptr,
                                                                      csr_col_ind,
                                                                      info,
                                                                      buffer_size));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status
    rocsparse_dprune_dense2csr_by_percentage_buffer_size(rocsparse_handle          handle,
                                                         rocsparse_int             m,
                                                         rocsparse_int             n,
                                                         const double*             A,
                                                         rocsparse_int             lda,
                                                         double                    percentage,
                                                         const rocsparse_mat_descr descr,
                                                         const double*             csr_val,
                                                         const rocsparse_int*      csr_row_ptr,
                                                         const rocsparse_int*      csr_col_ind,
                                                         rocsparse_mat_info        info,
                                                         size_t*                   buffer_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::prune_dense2csr_by_percentage_buffer_size_template(handle,
                                                                      m,
                                                                      n,
                                                                      A,
                                                                      lda,
                                                                      percentage,
                                                                      descr,
                                                                      csr_val,
                                                                      csr_row_ptr,
                                                                      csr_col_ind,
                                                                      info,
                                                                      buffer_size));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status
    rocsparse_sprune_dense2csr_nnz_by_percentage(rocsparse_handle          handle,
                                                 rocsparse_int             m,
                                                 rocsparse_int             n,
                                                 const float*              A,
                                                 rocsparse_int             lda,
                                                 float                     percentage,
                                                 const rocsparse_mat_descr descr,
                                                 rocsparse_int*            csr_row_ptr,
                                                 rocsparse_int*            nnz_total_dev_host_ptr,
                                                 rocsparse_mat_info        info,
                                                 void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::prune_dense2csr_nnz_by_percentage_template(handle,
                                                              m,
                                                              n,
                                                              A,
                                                              lda,
                                                              percentage,
                                                              descr,
                                                              csr_row_ptr,
                                                              nnz_total_dev_host_ptr,
                                                              info,
                                                              temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status
    rocsparse_dprune_dense2csr_nnz_by_percentage(rocsparse_handle          handle,
                                                 rocsparse_int             m,
                                                 rocsparse_int             n,
                                                 const double*             A,
                                                 rocsparse_int             lda,
                                                 double                    percentage,
                                                 const rocsparse_mat_descr descr,
                                                 rocsparse_int*            csr_row_ptr,
                                                 rocsparse_int*            nnz_total_dev_host_ptr,
                                                 rocsparse_mat_info        info,
                                                 void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::prune_dense2csr_nnz_by_percentage_template(handle,
                                                              m,
                                                              n,
                                                              A,
                                                              lda,
                                                              percentage,
                                                              descr,
                                                              csr_row_ptr,
                                                              nnz_total_dev_host_ptr,
                                                              info,
                                                              temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status
    rocsparse_sprune_dense2csr_by_percentage(rocsparse_handle          handle,
                                             rocsparse_int             m,
                                             rocsparse_int             n,
                                             const float*              A,
                                             rocsparse_int             lda,
                                             float                     percentage,
                                             const rocsparse_mat_descr descr,
                                             float*                    csr_val,
                                             const rocsparse_int*      csr_row_ptr,
                                             rocsparse_int*            csr_col_ind,
                                             rocsparse_mat_info        info,
                                             void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::prune_dense2csr_by_percentage_template(handle,
                                                                                m,
                                                                                n,
                                                                                A,
                                                                                lda,
                                                                                percentage,
                                                                                descr,
                                                                                csr_val,
                                                                                csr_row_ptr,
                                                                                csr_col_ind,
                                                                                info,
                                                                                temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status
    rocsparse_dprune_dense2csr_by_percentage(rocsparse_handle          handle,
                                             rocsparse_int             m,
                                             rocsparse_int             n,
                                             const double*             A,
                                             rocsparse_int             lda,
                                             double                    percentage,
                                             const rocsparse_mat_descr descr,
                                             double*                   csr_val,
                                             const rocsparse_int*      csr_row_ptr,
                                             rocsparse_int*            csr_col_ind,
                                             rocsparse_mat_info        info,
                                             void*                     temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::prune_dense2csr_by_percentage_template(handle,
                                                                                m,
                                                                                n,
                                                                                A,
                                                                                lda,
                                                                                percentage,
                                                                                descr,
                                                                                csr_val,
                                                                                csr_row_ptr,
                                                                                csr_col_ind,
                                                                                info,
                                                                                temp_buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
