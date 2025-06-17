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

#include "internal/conversion/rocsparse_csr2csr_compress.h"
#include "rocsparse_control.hpp"
#include "rocsparse_csr2csr_compress.hpp"
#include "rocsparse_utility.hpp"

#include "csr2csr_compress_device.h"
#include "rocsparse_primitives.hpp"

template <typename T>
rocsparse_status rocsparse::csr2csr_compress_template(rocsparse_handle          handle, //0
                                                      rocsparse_int             m, //1
                                                      rocsparse_int             n, //2
                                                      const rocsparse_mat_descr descr_A, //3
                                                      const T*                  csr_val_A, //4
                                                      const rocsparse_int*      csr_row_ptr_A, //5
                                                      const rocsparse_int*      csr_col_ind_A, //6
                                                      rocsparse_int             nnz_A, //7
                                                      const rocsparse_int*      nnz_per_row, //8
                                                      T*                        csr_val_C, //9
                                                      rocsparse_int*            csr_row_ptr_C, //10
                                                      rocsparse_int*            csr_col_ind_C, //11
                                                      T                         tol) //12
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xcsr2csr_compress"),
                         m,
                         n,
                         descr_A,
                         (const void*&)csr_val_A,
                         (const void*&)csr_row_ptr_A,
                         (const void*&)csr_col_ind_A,
                         nnz_A,
                         (const void*&)nnz_per_row,
                         (const void*&)csr_val_C,
                         (const void*&)csr_row_ptr_C,
                         (const void*&)csr_col_ind_C,
                         tol);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(3, descr_A);
    ROCSPARSE_CHECKARG(3,
                       descr_A,
                       (descr_A->storage_mode != rocsparse_storage_mode_sorted),
                       rocsparse_status_requires_sorted_storage);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_ARRAY(4, nnz_A, csr_val_A);
    ROCSPARSE_CHECKARG_ARRAY(5, m, csr_row_ptr_A);
    ROCSPARSE_CHECKARG_ARRAY(6, nnz_A, csr_col_ind_A);
    ROCSPARSE_CHECKARG_SIZE(7, nnz_A);
    ROCSPARSE_CHECKARG_ARRAY(8, m, nnz_per_row);
    ROCSPARSE_CHECKARG_ARRAY(10, m, csr_row_ptr_C);
    ROCSPARSE_CHECKARG(
        12, tol, (std::real(tol) < std::real(static_cast<T>(0))), rocsparse_status_invalid_value);

    if(m == 0 || n == 0)
    {
        return rocsparse_status_success;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Compute required temporary storage buffer size
    size_t nwarps                   = (nnz_A - 1) / handle->wavefront_size + 1;
    size_t temp_storage_size_bytes1 = sizeof(uint32_t) * (nwarps / 256 + 1) * 256;

    // Compute buffer size for inclusive scan on csr_row_ptr_C
    size_t temp_storage_size_bytes2;
    RETURN_IF_ROCSPARSE_ERROR(
        (rocsparse::primitives::inclusive_scan_buffer_size<rocsparse_int, rocsparse_int>(
            handle, m + 1, &temp_storage_size_bytes2)));
    temp_storage_size_bytes2 = ((temp_storage_size_bytes2 - 1) / 256 + 1) * 256;

    // Compute buffer size for inclusive scan on warp_start
    size_t temp_storage_size_bytes3;
    RETURN_IF_ROCSPARSE_ERROR(
        (rocsparse::primitives::inclusive_scan_buffer_size<uint32_t, uint32_t>(
            handle, nwarps + 1, &temp_storage_size_bytes3)));
    temp_storage_size_bytes3 = ((temp_storage_size_bytes3 - 1) / 256 + 1) * 256;

    size_t temp_storage_size_bytes
        = temp_storage_size_bytes1 + temp_storage_size_bytes2 + temp_storage_size_bytes3;

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

    char*     ptr        = reinterpret_cast<char*>(temp_storage_ptr);
    uint32_t* warp_start = reinterpret_cast<uint32_t*>(ptr);
    ptr += temp_storage_size_bytes1;
    void* temp_storage_buffer2 = ptr;
    ptr += temp_storage_size_bytes2;
    void* temp_storage_buffer3 = ptr;
    ptr += temp_storage_size_bytes3;

    // Copy nnz_per_row to csr_row_ptr_C array
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::fill_row_ptr_device<1024>),
                                       dim3((m - 1) / 1024 + 1),
                                       dim3(1024),
                                       0,
                                       stream,
                                       m,
                                       descr_A->base,
                                       nnz_per_row,
                                       csr_row_ptr_C);

    // Perform inclusive scan on csr row pointer array
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::inclusive_scan(handle,
                                                                    csr_row_ptr_C,
                                                                    csr_row_ptr_C,
                                                                    m + 1,
                                                                    temp_storage_size_bytes2,
                                                                    temp_storage_buffer2));

    if(csr_val_C == nullptr || csr_col_ind_C == nullptr)
    {
        rocsparse_int start = 0;
        rocsparse_int end   = 0;
        if(csr_row_ptr_C != nullptr)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&end,
                                               &csr_row_ptr_C[m],
                                               sizeof(rocsparse_int),
                                               hipMemcpyDeviceToHost,
                                               handle->stream));
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(&start,
                                               &csr_row_ptr_C[0],
                                               sizeof(rocsparse_int),
                                               hipMemcpyDeviceToHost,
                                               handle->stream));
        }
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));

        const rocsparse_int nnz_C = (end - start);
        ROCSPARSE_CHECKARG_ARRAY(9, nnz_C, csr_val_C);
        ROCSPARSE_CHECKARG_ARRAY(11, nnz_C, csr_col_ind_C);
    }

#define LOOPS 2
    if(handle->wavefront_size == 32)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csr2csr_compress_fill_warp_start_device<256, 32, LOOPS>),
            dim3((nnz_A - 1) / (256 * LOOPS) + 1),
            dim3(256),
            0,
            stream,
            nnz_A,
            csr_val_A,
            warp_start,
            tol);
    }
    else if(handle->wavefront_size == 64)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csr2csr_compress_fill_warp_start_device<256, 64, LOOPS>),
            dim3((nnz_A - 1) / (256 * LOOPS) + 1),
            dim3(256),
            0,
            stream,
            nnz_A,
            csr_val_A,
            warp_start,
            tol);
    }

    // Perform inclusive scan on warp start array
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::inclusive_scan(handle,
                                                                    warp_start,
                                                                    warp_start,
                                                                    nwarps + 1,
                                                                    temp_storage_size_bytes3,
                                                                    temp_storage_buffer3));

    if(handle->wavefront_size == 32)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csr2csr_compress_use_warp_start_device<256, 32, LOOPS>),
            dim3((nnz_A - 1) / (256 * LOOPS) + 1),
            dim3(256),
            0,
            stream,
            nnz_A,
            descr_A->base,
            csr_val_A,
            csr_col_ind_A,
            descr_A->base,
            csr_val_C,
            csr_col_ind_C,
            warp_start,
            tol);
    }
    else if(handle->wavefront_size == 64)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::csr2csr_compress_use_warp_start_device<256, 64, LOOPS>),
            dim3((nnz_A - 1) / (256 * LOOPS) + 1),
            dim3(256),
            0,
            stream,
            nnz_A,
            descr_A->base,
            csr_val_A,
            csr_col_ind_A,
            descr_A->base,
            csr_val_C,
            csr_col_ind_C,
            warp_start,
            tol);
    }
#undef LOOPS

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

#define C_IMPL(NAME, TYPE)                                                            \
    extern "C" rocsparse_status NAME(rocsparse_handle          handle,                \
                                     rocsparse_int             m,                     \
                                     rocsparse_int             n,                     \
                                     const rocsparse_mat_descr descr_A,               \
                                     const TYPE*               csr_val_A,             \
                                     const rocsparse_int*      csr_row_ptr_A,         \
                                     const rocsparse_int*      csr_col_ind_A,         \
                                     rocsparse_int             nnz_A,                 \
                                     const rocsparse_int*      nnz_per_row,           \
                                     TYPE*                     csr_val_C,             \
                                     rocsparse_int*            csr_row_ptr_C,         \
                                     rocsparse_int*            csr_col_ind_C,         \
                                     TYPE                      tol)                   \
    try                                                                               \
    {                                                                                 \
        ROCSPARSE_ROUTINE_TRACE;                                                      \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::csr2csr_compress_template(handle,        \
                                                                       m,             \
                                                                       n,             \
                                                                       descr_A,       \
                                                                       csr_val_A,     \
                                                                       csr_row_ptr_A, \
                                                                       csr_col_ind_A, \
                                                                       nnz_A,         \
                                                                       nnz_per_row,   \
                                                                       csr_val_C,     \
                                                                       csr_row_ptr_C, \
                                                                       csr_col_ind_C, \
                                                                       tol));         \
        return rocsparse_status_success;                                              \
    }                                                                                 \
    catch(...)                                                                        \
    {                                                                                 \
        RETURN_ROCSPARSE_EXCEPTION();                                                 \
    }

C_IMPL(rocsparse_scsr2csr_compress, float);
C_IMPL(rocsparse_dcsr2csr_compress, double);
C_IMPL(rocsparse_ccsr2csr_compress, rocsparse_float_complex);
C_IMPL(rocsparse_zcsr2csr_compress, rocsparse_double_complex);
#undef C_IMPL
