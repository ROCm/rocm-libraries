/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/conversion/rocsparse_coosort.h"
#include "internal/conversion/rocsparse_inverse_permutation.h"

#include "rocsparse_utility.hpp"

#include "../level1/rocsparse_gthr.hpp"
#include "coosort_device.h"
#include "rocsparse_control.hpp"
#include "rocsparse_coosort.hpp"
#include "rocsparse_gcreate_identity_permutation.hpp"
#include "rocsparse_identity.hpp"
#include "rocsparse_primitives.hpp"

namespace rocsparse
{
    template <typename J>
    static rocsparse_status determine_rocprim_buffer_size(rocsparse_handle handle,
                                                          J                m,
                                                          J                n,
                                                          J                nnz,
                                                          const J*         coo_row_ind,
                                                          const J*         coo_col_ind,
                                                          size_t*          buffer_size)
    {
        ROCSPARSE_ROUTINE_TRACE;

        uint32_t startbit = 0;
        uint32_t endbit   = rocsparse::clz(m);

        // Determine max buffer size
        size_t size;
        *buffer_size = 0;

        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::radix_sort_pairs_buffer_size<J, J>(
            handle, nnz, startbit, endbit, &size)));

        *buffer_size = rocsparse::max(size, *buffer_size);

        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::primitives::run_length_encode_buffer_size<J>(handle, nnz, &size));
        *buffer_size = rocsparse::max(size, *buffer_size);

        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::exclusive_scan_buffer_size<J, J>(
            handle, static_cast<J>(0), m + 1, &size)));
        *buffer_size = rocsparse::max(size, *buffer_size);

        endbit = rocsparse::clz(n);

        size_t size1;
        size_t size2;
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::primitives::segmented_radix_sort_pairs_buffer_size<J, J, J>(
                handle, nnz, m, startbit, endbit, &size1)));
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::primitives::segmented_radix_sort_keys_buffer_size<J, J>(
                handle, nnz, m, startbit, endbit, &size2)));

        *buffer_size = rocsparse::max(rocsparse::max(size1, size2), *buffer_size);

        return rocsparse_status_success;
    }
}

template <typename J>
rocsparse_status rocsparse::coosort_buffer_size_template(rocsparse_handle handle,
                                                         int64_t          m,
                                                         int64_t          n,
                                                         int64_t          nnz,
                                                         const void*      coo_row_ind,
                                                         const void*      coo_col_ind,
                                                         size_t*          buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_coosort_buffer_size",
                         m,
                         n,
                         nnz,
                         (const void*&)coo_row_ind,
                         (const void*&)coo_col_ind,
                         (const void*&)buffer_size);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG_ARRAY(4, nnz, coo_row_ind);
    ROCSPARSE_CHECKARG_ARRAY(5, nnz, coo_col_ind);
    ROCSPARSE_CHECKARG_POINTER(6, buffer_size);

    // Quick return if possible
    if(m == 0 || n == 0 || nnz == 0)
    {
        *buffer_size = 0;
        return rocsparse_status_success;
    }

    const J* coo_row_ind_ = reinterpret_cast<const J*>(coo_row_ind);
    const J* coo_col_ind_ = reinterpret_cast<const J*>(coo_col_ind);

    // Determine rocprim buffer size when coosort is by row
    size_t buffer_size_by_row;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::determine_rocprim_buffer_size<J>(handle,
                                                                          static_cast<J>(m),
                                                                          static_cast<J>(n),
                                                                          static_cast<J>(nnz),
                                                                          coo_row_ind_,
                                                                          coo_col_ind_,
                                                                          &buffer_size_by_row));

    // Determine rocprim buffer size when coosort is by column
    size_t buffer_size_by_col;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::determine_rocprim_buffer_size<J>(handle,
                                                                          static_cast<J>(n),
                                                                          static_cast<J>(m),
                                                                          static_cast<J>(nnz),
                                                                          coo_col_ind_,
                                                                          coo_row_ind_,
                                                                          &buffer_size_by_col));

    // Use the maximum buffer size chosen between sorting by row or by column
    *buffer_size
        = rocsparse::align_size<char>(rocsparse::max(buffer_size_by_row, buffer_size_by_col));

    // rocPRIM does not support in-place sorting, so we need additional buffer
    // for all temporary arrays

    // rows buffer
    *buffer_size += rocsparse::align_size<J>(nnz);
    // columns buffer
    *buffer_size += rocsparse::align_size<J>(nnz);
    // perm buffer
    *buffer_size += rocsparse::align_size<J>(nnz);
    // segment buffer, which holds max(m, n) + 1 segment offsets after the exclusive scan
    *buffer_size += rocsparse::align_size<J>(rocsparse::max(m, n) + 1);

    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_coosort_buffer_size(rocsparse_handle     handle,
                                                          rocsparse_int        m,
                                                          rocsparse_int        n,
                                                          rocsparse_int        nnz,
                                                          const rocsparse_int* coo_row_ind,
                                                          const rocsparse_int* coo_col_ind,
                                                          size_t*              buffer_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    return rocsparse::coosort_buffer_size_template<rocsparse_int>(
        handle, m, n, nnz, coo_row_ind, coo_col_ind, buffer_size);
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

namespace rocsparse
{
    template <typename J>
    static rocsparse_status coosort_by_row_quickreturn(rocsparse_handle handle,
                                                       J                m,
                                                       J                n,
                                                       J                nnz,
                                                       J*               coo_row_ind,
                                                       J*               coo_col_ind,
                                                       J*               perm,
                                                       void*            temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Quick return if possible
        if(m == 0 || n == 0 || nnz == 0)
        {
            return rocsparse_status_success;
        }
        return rocsparse_status_continue;
    }

    template <typename J>
    static rocsparse_status coosort_by_row_checkarg(rocsparse_handle handle,
                                                    J                m,
                                                    J                n,
                                                    J                nnz,
                                                    J*               coo_row_ind,
                                                    J*               coo_col_ind,
                                                    J*               perm,
                                                    void*            temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_SIZE(1, m);
        ROCSPARSE_CHECKARG_SIZE(2, n);
        ROCSPARSE_CHECKARG_SIZE(3, nnz);
        ROCSPARSE_CHECKARG_ARRAY(4, nnz, coo_row_ind);
        ROCSPARSE_CHECKARG_ARRAY(5, nnz, coo_col_ind);
        ROCSPARSE_CHECKARG_ARRAY(7, nnz, temp_buffer);

        const rocsparse_status status = rocsparse::coosort_by_row_quickreturn(
            handle, m, n, nnz, coo_row_ind, coo_col_ind, perm, temp_buffer);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }
        return rocsparse_status_continue;
    }
}

template <typename J>
rocsparse_status rocsparse::coosort_by_row_template(rocsparse_handle handle,
                                                    int64_t          m,
                                                    int64_t          n,
                                                    int64_t          nnz,
                                                    void*            coo_row_ind,
                                                    void*            coo_col_ind,
                                                    void*            perm,
                                                    void*            temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for valid handle
    if(handle == nullptr)
    {
        return rocsparse_status_invalid_handle;
    }

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_coosort_by_row",
                         m,
                         n,
                         nnz,
                         (const void*&)coo_row_ind,
                         (const void*&)coo_col_ind,
                         (const void*&)perm,
                         (const void*&)temp_buffer);

    // Check sizes
    if(m < 0 || n < 0 || nnz < 0)
    {
        return rocsparse_status_invalid_size;
    }

    // Quick return if possible
    if(m == 0 || n == 0 || nnz == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    if(coo_row_ind == nullptr || coo_col_ind == nullptr || temp_buffer == nullptr)
    {
        return rocsparse_status_invalid_pointer;
    }

    J* row_ind = reinterpret_cast<J*>(coo_row_ind);
    J* col_ind = reinterpret_cast<J*>(coo_col_ind);
    J* perm_   = reinterpret_cast<J*>(perm);

    // Stream
    hipStream_t stream = handle->stream;

    uint32_t startbit = 0;
    uint32_t endbit   = rocsparse::clz(static_cast<rocsparse_int>(m));

    // Temporary buffer entry points
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    // Permutation vector given
    J* work1 = reinterpret_cast<J*>(ptr);
    ptr += rocsparse::align_size<J>(nnz);

    J* work2 = reinterpret_cast<J*>(ptr);
    ptr += rocsparse::align_size<J>(nnz);

    J* work3 = reinterpret_cast<J*>(ptr);
    ptr += rocsparse::align_size<J>(nnz);

    J* work4 = reinterpret_cast<J*>(ptr);
    ptr += rocsparse::align_size<J>(rocsparse::max(m, n) + 1);

    // Temporary rocprim buffer
    size_t size        = 0;
    void*  tmp_rocprim = reinterpret_cast<void*>(ptr);

    if(perm != nullptr)
    {
        // Create identitiy permutation to keep track of reorderings
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::create_identity_permutation_template<J>(handle, static_cast<J>(nnz), work1));

        // Sort by rows and store permutation
        rocsparse::primitives::double_buffer<J> keys(row_ind, work3);
        rocsparse::primitives::double_buffer<J> vals(work1, work2);

        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::radix_sort_pairs_buffer_size<J, J>(
            handle, nnz, startbit, endbit, &size)));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::radix_sort_pairs(
            handle, keys, vals, nnz, startbit, endbit, size, tmp_rocprim));

        J* output  = keys.current();
        J* mapping = vals.current();
        J* alt_map = vals.alternate();

        // Copy sorted rows, if stored in buffer
        if(output != row_ind)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(
                row_ind, output, sizeof(J) * nnz, hipMemcpyDeviceToDevice, stream));
        }

        // Obtain segments for segmented sort by columns
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::primitives::run_length_encode_buffer_size<J>(handle, nnz, &size));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::run_length_encode(
            handle, row_ind, work3 + 1, work4, work3, nnz, size, tmp_rocprim));

        J nsegm;
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMemcpyAsync(&nsegm, work3, sizeof(J), hipMemcpyDeviceToHost, stream));

        // Wait for host transfer to finish
        RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(stream));

        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::exclusive_scan_buffer_size<J, J>(
            handle, static_cast<J>(0), nsegm + 1, &size)));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::exclusive_scan(
            handle, work4, work4, static_cast<J>(0), nsegm + 1, size, tmp_rocprim));

// Reorder columns
#define COOSORT_DIM 512
        dim3 coosort_blocks((nnz - 1) / COOSORT_DIM + 1);
        dim3 coosort_threads(COOSORT_DIM);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::coosort_permute_kernel<COOSORT_DIM>),
                                           coosort_blocks,
                                           coosort_threads,
                                           0,
                                           stream,
                                           static_cast<J>(nnz),
                                           col_ind,
                                           mapping,
                                           work3);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::coosort_permute_kernel<COOSORT_DIM>),
                                           coosort_blocks,
                                           coosort_threads,
                                           0,
                                           stream,
                                           static_cast<J>(nnz),
                                           perm_,
                                           mapping,
                                           alt_map);
#undef COOSORT_DIM

        // Sort columns per row
        endbit = rocsparse::clz(static_cast<rocsparse_int>(n));

        rocsparse::primitives::double_buffer<J> keys2(work3, col_ind);
        rocsparse::primitives::double_buffer<J> vals2(alt_map, perm_);

        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::primitives::segmented_radix_sort_pairs_buffer_size<J, J, J>(
                handle, nnz, nsegm, startbit, endbit, &size)));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::segmented_radix_sort_pairs(handle,
                                                                                    keys2,
                                                                                    vals2,
                                                                                    nnz,
                                                                                    nsegm,
                                                                                    work4,
                                                                                    work4 + 1,
                                                                                    startbit,
                                                                                    endbit,
                                                                                    size,
                                                                                    tmp_rocprim));

        output  = keys2.current();
        mapping = vals2.current();

        // Copy sorted columns, if stored in buffer
        if(output != col_ind)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(
                col_ind, output, sizeof(J) * nnz, hipMemcpyDeviceToDevice, stream));
        }

        // Copy reordered permutation, if stored in buffer
        if(mapping != perm_)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(
                perm_, mapping, sizeof(J) * nnz, hipMemcpyDeviceToDevice, stream));
        }
    }
    else
    {
        // No permutation vector given

        // Sort by rows and permute columns
        rocsparse::primitives::double_buffer<J> keys(row_ind, work3);
        rocsparse::primitives::double_buffer<J> vals(col_ind, work2);

        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::radix_sort_pairs_buffer_size<J, J>(
            handle, nnz, startbit, endbit, &size)));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::radix_sort_pairs(
            handle, keys, vals, nnz, startbit, endbit, size, tmp_rocprim));
        J* output = keys.current();

        // Copy sorted rows, if stored in buffer
        if(output != row_ind)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(
                row_ind, output, sizeof(J) * nnz, hipMemcpyDeviceToDevice, stream));
        }

        // Obtain segments for segmented sort by columns
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::primitives::run_length_encode_buffer_size<J>(handle, nnz, &size));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::run_length_encode(
            handle, row_ind, work3 + 1, work4, work3, nnz, size, tmp_rocprim));

        J nsegm;
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMemcpyAsync(&nsegm, work3, sizeof(J), hipMemcpyDeviceToHost, stream));

        // Wait for host transfer to finish
        RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(stream));

        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::exclusive_scan_buffer_size<J, J>(
            handle, static_cast<J>(0), nsegm + 1, &size)));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::exclusive_scan(
            handle, work4, work4, static_cast<J>(0), nsegm + 1, size, tmp_rocprim));

        // Sort columns per row
        endbit = rocsparse::clz(static_cast<rocsparse_int>(n));

        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::primitives::segmented_radix_sort_keys_buffer_size<J, J>(
                handle, nnz, nsegm, startbit, endbit, &size)));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::segmented_radix_sort_keys(
            handle, vals, nnz, nsegm, work4, work4 + 1, startbit, endbit, size, tmp_rocprim));

        output = vals.current();

        // Copy sorted columns, if stored in buffer
        if(output != col_ind)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(
                col_ind, output, sizeof(J) * nnz, hipMemcpyDeviceToDevice, stream));
        }
    }

    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_coosort_by_row(rocsparse_handle handle,
                                                     rocsparse_int    m,
                                                     rocsparse_int    n,
                                                     rocsparse_int    nnz,
                                                     rocsparse_int*   coo_row_ind,
                                                     rocsparse_int*   coo_col_ind,
                                                     rocsparse_int*   perm,
                                                     void*            temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_coosort_by_row",
                         m,
                         n,
                         nnz,
                         (const void*&)coo_row_ind,
                         (const void*&)coo_col_ind,
                         (const void*&)perm,
                         (const void*&)temp_buffer);

    const rocsparse_status status = rocsparse::coosort_by_row_checkarg(
        handle, m, n, nnz, coo_row_ind, coo_col_ind, perm, temp_buffer);

    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosort_by_row_template<rocsparse_int>(
        handle, m, n, nnz, coo_row_ind, coo_col_ind, perm, temp_buffer));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

namespace rocsparse
{
    template <typename J>
    static rocsparse_status coosort_by_column_quickreturn(rocsparse_handle handle,
                                                          J                m,
                                                          J                n,
                                                          J                nnz,
                                                          J*               coo_row_ind,
                                                          J*               coo_col_ind,
                                                          J*               perm,
                                                          void*            temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Quick return if possible
        if(m == 0 || n == 0 || nnz == 0)
        {
            return rocsparse_status_success;
        }
        return rocsparse_status_continue;
    }

    template <typename J>
    static rocsparse_status coosort_by_column_checkarg(rocsparse_handle handle,
                                                       J                m,
                                                       J                n,
                                                       J                nnz,
                                                       J*               coo_row_ind,
                                                       J*               coo_col_ind,
                                                       J*               perm,
                                                       void*            temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_SIZE(1, m);
        ROCSPARSE_CHECKARG_SIZE(2, n);
        ROCSPARSE_CHECKARG_SIZE(3, nnz);
        ROCSPARSE_CHECKARG_ARRAY(4, nnz, coo_row_ind);
        ROCSPARSE_CHECKARG_ARRAY(5, nnz, coo_col_ind);
        ROCSPARSE_CHECKARG_ARRAY(7, nnz, temp_buffer);

        const rocsparse_status status = rocsparse::coosort_by_column_quickreturn(
            handle, m, n, nnz, coo_row_ind, coo_col_ind, perm, temp_buffer);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        return rocsparse_status_continue;
    }
}

template <typename J>
rocsparse_status rocsparse::coosort_by_column_template(rocsparse_handle handle,
                                                       int64_t          m,
                                                       int64_t          n,
                                                       int64_t          nnz,
                                                       void*            coo_row_ind,
                                                       void*            coo_col_ind,
                                                       void*            perm,
                                                       void*            temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosort_by_row_template<J>(
        handle, n, m, nnz, coo_col_ind, coo_row_ind, perm, temp_buffer));
    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_coosort_by_column(rocsparse_handle handle,
                                                        rocsparse_int    m,
                                                        rocsparse_int    n,
                                                        rocsparse_int    nnz,
                                                        rocsparse_int*   coo_row_ind,
                                                        rocsparse_int*   coo_col_ind,
                                                        rocsparse_int*   perm,
                                                        void*            temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_coosort_by_column",
                         m,
                         n,
                         nnz,
                         (const void*&)coo_row_ind,
                         (const void*&)coo_col_ind,
                         (const void*&)perm,
                         (const void*&)temp_buffer);

    const rocsparse_status status = rocsparse::coosort_by_column_checkarg(
        handle, m, n, nnz, coo_row_ind, coo_col_ind, perm, temp_buffer);

    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosort_by_column_template<rocsparse_int>(
        handle, m, n, nnz, coo_row_ind, coo_col_ind, perm, temp_buffer));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

#define INSTANTIATE(J)                                                                               \
    template rocsparse_status rocsparse::coosort_buffer_size_template<J>(rocsparse_handle handle,    \
                                                                         int64_t          m,         \
                                                                         int64_t          n,         \
                                                                         int64_t          nnz,       \
                                                                         const void* coo_row_ind,    \
                                                                         const void* coo_col_ind,    \
                                                                         size_t*     buffer_size);       \
    template rocsparse_status rocsparse::coosort_by_row_template<J>(rocsparse_handle handle,         \
                                                                    int64_t          m,              \
                                                                    int64_t          n,              \
                                                                    int64_t          nnz,            \
                                                                    void*            coo_row_ind,    \
                                                                    void*            coo_col_ind,    \
                                                                    void*            perm,           \
                                                                    void*            temp_buffer);              \
                                                                                                     \
    template rocsparse_status rocsparse::coosort_by_column_template<J>(rocsparse_handle handle,      \
                                                                       int64_t          m,           \
                                                                       int64_t          n,           \
                                                                       int64_t          nnz,         \
                                                                       void*            coo_row_ind, \
                                                                       void*            coo_col_ind, \
                                                                       void*            perm,        \
                                                                       void*            temp_buffer)
INSTANTIATE(int32_t);
INSTANTIATE(int64_t);
#undef INSTANTIATE

namespace rocsparse
{
    // The buffer starts with the permutation array, which tracks where each entry moves
    // while the indices are sorted, followed by scratch space shared by the index sort
    // and the value permutation.
    static size_t coosort_perm_size(int64_t nnz, rocsparse_indextype perm_indextype)
    {
        return rocsparse::align_size<char>(rocsparse::indextype_sizeof(perm_indextype) * nnz);
    }

    // The indices are copied and the values gathered without any type conversion.
    static rocsparse_status coosort_check_types(rocsparse_indextype coo_row_indextype_A,
                                                rocsparse_indextype coo_col_indextype_A,
                                                rocsparse_datatype  coo_val_datatype_A,
                                                rocsparse_indextype coo_row_indextype_B,
                                                rocsparse_indextype coo_col_indextype_B,
                                                rocsparse_datatype  coo_val_datatype_B)
    {
        if(coo_row_indextype_B != coo_row_indextype_A || coo_col_indextype_B != coo_col_indextype_A)
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_invalid_value,
                "the index types of the output matrix must match the index types of the input "
                "matrix");
        }
        if(coo_val_datatype_B != coo_val_datatype_A)
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_invalid_value,
                "the data type of the output matrix must match the data type of the input matrix");
        }
        return rocsparse_status_success;
    }
}

rocsparse_status rocsparse::coosort_buffer_size(rocsparse_handle      handle,
                                                rocsparse_coosort_alg alg,
                                                rocsparse_direction   dir,
                                                int64_t               m,
                                                int64_t               n,
                                                int64_t               nnz,
                                                rocsparse_indextype   coo_row_indextype_A,
                                                const void*           coo_row_ind_A,
                                                rocsparse_indextype   coo_col_indextype_A,
                                                const void*           coo_col_ind_A,
                                                rocsparse_datatype    coo_val_datatype_A,
                                                const void*           coo_val_A,
                                                rocsparse_indextype   coo_row_indextype_B,
                                                const void*           coo_row_ind_B,
                                                rocsparse_indextype   coo_col_indextype_B,
                                                const void*           coo_col_ind_B,
                                                rocsparse_datatype    coo_val_datatype_B,
                                                const void*           coo_val_B,
                                                size_t*               buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosort_check_types(coo_row_indextype_A,
                                                             coo_col_indextype_A,
                                                             coo_val_datatype_A,
                                                             coo_row_indextype_B,
                                                             coo_col_indextype_B,
                                                             coo_val_datatype_B));

    auto f = rocsparse::coosort_buffer_size_template<int32_t>;
    if(coo_row_indextype_B == rocsparse_indextype_i32)
    {
        f = rocsparse::coosort_buffer_size_template<int32_t>;
    }
    else if(coo_row_indextype_B == rocsparse_indextype_i64)
    {
        f = rocsparse::coosort_buffer_size_template<int64_t>;
    }
    else
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            rocsparse_status_invalid_value,
            "rocsparse::coosort_buffer_size failed from dispatching");
    }

    size_t sort_buffer_size = 0;
    RETURN_IF_ROCSPARSE_ERROR(
        f(handle, m, n, nnz, coo_row_ind_B, coo_col_ind_B, &sort_buffer_size));

    // Values sorted in place are gathered into scratch space first, since the gather cannot
    // write over its own input.
    const size_t gather_buffer_size
        = (coo_val_B == coo_val_A)
              ? rocsparse::align_size<char>(rocsparse::datatype_sizeof(coo_val_datatype_B) * nnz)
              : 0;

    *buffer_size = rocsparse::coosort_perm_size(nnz, coo_row_indextype_B)
                   + rocsparse::max(sort_buffer_size, gather_buffer_size);

    return rocsparse_status_success;
}

rocsparse_status rocsparse::coosort(rocsparse_handle      handle,
                                    rocsparse_coosort_alg alg,
                                    rocsparse_direction   dir,
                                    int64_t               m,
                                    int64_t               n,
                                    int64_t               nnz,
                                    int64_t               batch_count_A,
                                    int64_t               batch_stride_A,
                                    rocsparse_indextype   coo_row_indextype_A,
                                    const void*           coo_row_ind_A,
                                    rocsparse_indextype   coo_col_indextype_A,
                                    const void*           coo_col_ind_A,
                                    rocsparse_datatype    coo_val_datatype_A,
                                    const void*           coo_val_A,
                                    int64_t               batch_count_B,
                                    int64_t               batch_stride_B,
                                    rocsparse_indextype   coo_row_indextype_B,
                                    void*                 coo_row_ind_B,
                                    rocsparse_indextype   coo_col_indextype_B,
                                    void*                 coo_col_ind_B,
                                    rocsparse_datatype    coo_val_datatype_B,
                                    void*                 coo_val_B,
                                    void*                 temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosort_check_types(coo_row_indextype_A,
                                                             coo_col_indextype_A,
                                                             coo_val_datatype_A,
                                                             coo_row_indextype_B,
                                                             coo_col_indextype_B,
                                                             coo_val_datatype_B));

    if(batch_count_B != batch_count_A)
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            rocsparse_status_invalid_value,
            "the batch count of the output matrix must match the batch count of the input matrix");
    }

    auto f = rocsparse::coosort_by_row_template<int32_t>;
    if(dir == rocsparse_direction_row && coo_row_indextype_B == rocsparse_indextype_i32)
    {
        f = rocsparse::coosort_by_row_template<int32_t>;
    }
    else if(dir == rocsparse_direction_row && coo_row_indextype_B == rocsparse_indextype_i64)
    {
        f = rocsparse::coosort_by_row_template<int64_t>;
    }
    else if(dir == rocsparse_direction_column && coo_col_indextype_B == rocsparse_indextype_i32)
    {
        f = rocsparse::coosort_by_column_template<int32_t>;
    }
    else if(dir == rocsparse_direction_column && coo_col_indextype_B == rocsparse_indextype_i64)
    {
        f = rocsparse::coosort_by_column_template<int64_t>;
    }
    else
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                               "rocsparse::coosort failed from dispatching");
    }

    const rocsparse_indextype perm_indextype = coo_row_indextype_B;

    const size_t row_size_A = rocsparse::indextype_sizeof(coo_row_indextype_A);
    const size_t col_size_A = rocsparse::indextype_sizeof(coo_col_indextype_A);
    const size_t val_size_A = rocsparse::datatype_sizeof(coo_val_datatype_A);
    const size_t row_size_B = rocsparse::indextype_sizeof(coo_row_indextype_B);
    const size_t col_size_B = rocsparse::indextype_sizeof(coo_col_indextype_B);
    const size_t val_size_B = rocsparse::datatype_sizeof(coo_val_datatype_B);

    void* perm = temp_buffer;
    void* sort_buffer
        = reinterpret_cast<char*>(temp_buffer) + rocsparse::coosort_perm_size(nnz, perm_indextype);

    // The batches run one after the other on the handle stream, so they share temp_buffer.
    for(int64_t batch = 0; batch < batch_count_A; ++batch)
    {
        const int64_t offset_A = batch * batch_stride_A;
        const int64_t offset_B = batch * batch_stride_B;

        const void* row_ind_A
            = reinterpret_cast<const char*>(coo_row_ind_A) + offset_A * row_size_A;
        const void* col_ind_A
            = reinterpret_cast<const char*>(coo_col_ind_A) + offset_A * col_size_A;
        const void* val_A     = reinterpret_cast<const char*>(coo_val_A) + offset_A * val_size_A;
        void*       row_ind_B = reinterpret_cast<char*>(coo_row_ind_B) + offset_B * row_size_B;
        void*       col_ind_B = reinterpret_cast<char*>(coo_col_ind_B) + offset_B * col_size_B;
        void*       val_B     = reinterpret_cast<char*>(coo_val_B) + offset_B * val_size_B;

        // The index sort works in place, so the indices of A are first copied into B.
        if(row_ind_B != row_ind_A)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                row_ind_B, row_ind_A, row_size_A * nnz, hipMemcpyDeviceToDevice, handle->stream));
        }
        if(col_ind_B != col_ind_A)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                col_ind_B, col_ind_A, col_size_A * nnz, hipMemcpyDeviceToDevice, handle->stream));
        }

        // The index sort applies its reordering to perm, so it must start as the identity.
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::gcreate_identity_permutation(handle, nnz, perm_indextype, perm));

        RETURN_IF_ROCSPARSE_ERROR(f(handle, m, n, nnz, row_ind_B, col_ind_B, perm, sort_buffer));

        // The gather cannot write over its own input, so in place values go through scratch.
        const bool in_place_val = (val_B == val_A);
        void*      sorted_val   = in_place_val ? sort_buffer : val_B;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gthr(handle,
                                                  nnz,
                                                  coo_val_datatype_A,
                                                  val_A,
                                                  coo_val_datatype_B,
                                                  sorted_val,
                                                  perm_indextype,
                                                  perm,
                                                  rocsparse_index_base_zero));

        if(in_place_val)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                val_B, sorted_val, val_size_B * nnz, hipMemcpyDeviceToDevice, handle->stream));
        }
    }

    return rocsparse_status_success;
}
