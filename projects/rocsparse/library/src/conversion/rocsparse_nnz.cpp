/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/conversion/rocsparse_nnz.h"
#include "rocsparse_control.hpp"
#include "rocsparse_nnz.hpp"
#include "rocsparse_nnz_impl.hpp"
#include "rocsparse_utility.hpp"

#include "nnz_device.h"
#include "rocsparse_primitives.hpp"

namespace rocsparse
{
    template <typename I, typename J, typename T>
    static rocsparse_status nnz_kernel_row(rocsparse_handle handle,
                                           J                m,
                                           J                n,
                                           const T*         A,
                                           int64_t          ld,
                                           rocsparse_order  order,
                                           I*               nnz_per_rows)
    {
        ROCSPARSE_ROUTINE_TRACE;

        hipStream_t stream = handle->stream;

        static constexpr rocsparse_int NNZ_DIM_X = 64;
        static constexpr rocsparse_int NNZ_DIM_Y = 16;
        rocsparse_int                  blocks    = (m - 1) / (NNZ_DIM_X * 4) + 1;
        if(std::is_same<T, rocsparse_double_complex>{})
            blocks = (m - 1) / (NNZ_DIM_X) + 1;
        dim3 k_grid(blocks);
        dim3 k_threads(NNZ_DIM_X, NNZ_DIM_Y);
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::nnz_kernel_row<NNZ_DIM_X, NNZ_DIM_Y>),
                                           k_grid,
                                           k_threads,
                                           0,
                                           stream,
                                           order,
                                           m,
                                           n,
                                           A,
                                           ld,
                                           nnz_per_rows);

        return rocsparse_status_success;
    }

    template <typename I, typename J, typename T>
    static rocsparse_status nnz_kernel_col(rocsparse_handle handle,
                                           J                m,
                                           J                n,
                                           const T*         A,
                                           int64_t          ld,
                                           rocsparse_order  order,
                                           I*               nnz_per_columns)
    {
        ROCSPARSE_ROUTINE_TRACE;

        hipStream_t stream = handle->stream;

        static constexpr rocsparse_int NB = 256;
        dim3                           kernel_blocks(n);
        dim3                           kernel_threads(NB);
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::nnz_kernel_col<NB>),
                                           kernel_blocks,
                                           kernel_threads,
                                           0,
                                           stream,
                                           order,
                                           m,
                                           n,
                                           A,
                                           ld,
                                           nnz_per_columns);

        return rocsparse_status_success;
    }
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse::nnz_template(rocsparse_handle    handle,
                                         rocsparse_direction dir,
                                         rocsparse_order     order,
                                         J                   m,
                                         J                   n,
                                         const T*            A,
                                         int64_t             ld,
                                         I*                  nnz_per_row_columns)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(m == 0 || n == 0)
    {
        return rocsparse_status_success;
    }

    switch(dir)
    {
    case rocsparse_direction_row:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::nnz_kernel_row(handle, m, n, A, ld, order, nnz_per_row_columns));
        return rocsparse_status_success;
        break;
    }

    case rocsparse_direction_column:
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::nnz_kernel_col(handle, m, n, A, ld, order, nnz_per_row_columns));
        return rocsparse_status_success;
        break;
    }
    }

    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

namespace rocsparse
{
    template <typename I, typename J, typename T>
    rocsparse_status nnz_checkarg(rocsparse_handle          handle,
                                  rocsparse_direction       dir,
                                  J                         m,
                                  J                         n,
                                  const rocsparse_mat_descr descr,
                                  const T*                  A,
                                  int64_t                   ld,
                                  I*                        nnz_per_row_columns,
                                  I*                        nnz_total_dev_host_ptr,
                                  rocsparse_order           order)
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_ENUM(1, dir);
        ROCSPARSE_CHECKARG_SIZE(2, m);
        ROCSPARSE_CHECKARG_SIZE(3, n);
        ROCSPARSE_CHECKARG(
            6, ld, (ld < (order == rocsparse_order_column ? m : n)), rocsparse_status_invalid_size);

        //
        // Quick return if possible, before checking for invalid pointers.
        //
        if(m == 0 || n == 0)
        {
            if(nnz_per_row_columns != nullptr)
            {
                switch(dir)
                {
                case rocsparse_direction_row:
                {
                    RETURN_IF_HIP_ERROR(
                        hipMemsetAsync(nnz_per_row_columns, 0, sizeof(I) * m, handle->stream));
                }
                case rocsparse_direction_column:
                {
                    RETURN_IF_HIP_ERROR(
                        hipMemsetAsync(nnz_per_row_columns, 0, sizeof(I) * n, handle->stream));
                }
                }
            }

            if(nnz_total_dev_host_ptr != nullptr)
            {
                rocsparse_pointer_mode mode;
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_get_pointer_mode(handle, &mode));
                if(rocsparse_pointer_mode_device == mode)
                {
                    RETURN_IF_HIP_ERROR(
                        hipMemsetAsync(nnz_total_dev_host_ptr, 0, sizeof(I), handle->stream));
                }
                else
                {
                    *nnz_total_dev_host_ptr = 0;
                }
            }

            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_POINTER(4, descr);
        ROCSPARSE_CHECKARG(4,
                           descr,
                           (rocsparse_matrix_type_general != descr->type),
                           rocsparse_status_not_implemented);
        ROCSPARSE_CHECKARG(4,
                           descr,
                           (descr->storage_mode != rocsparse_storage_mode_sorted),
                           rocsparse_status_requires_sorted_storage);

        ROCSPARSE_CHECKARG_POINTER(5, A);
        ROCSPARSE_CHECKARG_POINTER(7, nnz_per_row_columns);
        ROCSPARSE_CHECKARG_POINTER(8, nnz_total_dev_host_ptr);
        return rocsparse_status_continue;
    }
}

template <typename I, typename J, typename T>
rocsparse_status rocsparse::nnz_impl(rocsparse_handle          handle,
                                     rocsparse_direction       dir,
                                     rocsparse_order           order,
                                     J                         m,
                                     J                         n,
                                     const rocsparse_mat_descr descr,
                                     const T*                  A,
                                     int64_t                   ld,
                                     I*                        nnz_per_row_columns,
                                     I*                        nnz_total_dev_host_ptr)
{
    ROCSPARSE_ROUTINE_TRACE;

    //
    // Loggings
    //
    rocsparse::log_trace(handle,
                         "rocsparse_nnz",
                         dir,
                         order,
                         m,
                         n,
                         descr,
                         (const void*&)A,
                         ld,
                         (const void*&)nnz_per_row_columns,
                         (const void*&)nnz_total_dev_host_ptr);

    const rocsparse_status status = rocsparse::nnz_checkarg(
        handle, dir, m, n, descr, A, ld, nnz_per_row_columns, nnz_total_dev_host_ptr, order);
    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    //
    // Count.
    //
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::nnz_template(handle, dir, order, m, n, A, ld, nnz_per_row_columns));

    //
    // Compute the total number of non-zeros.
    //
    {
        J      mn = dir == rocsparse_direction_row ? m : n;
        size_t temp_storage_size_bytes;
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::find_sum_buffer_size<I, I>(
            handle, mn, &temp_storage_size_bytes)));
        temp_storage_size_bytes += sizeof(I);
        bool  temp_alloc       = false;
        void* temp_storage_ptr = nullptr;

        //
        // Device buffer should be sufficient for rocprim in most cases
        //
        I* d_nnz;
        if(handle->buffer_size >= temp_storage_size_bytes)
        {
            d_nnz            = (I*)handle->buffer;
            temp_storage_ptr = d_nnz + 1;
            temp_alloc       = false;
        }
        else
        {
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMallocAsync(&d_nnz, temp_storage_size_bytes, handle->stream));
            temp_storage_ptr = d_nnz + 1;
            temp_alloc       = true;
        }

        //
        // Perform reduce
        //
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::find_sum(
            handle, nnz_per_row_columns, d_nnz, mn, temp_storage_size_bytes, temp_storage_ptr));

        //
        // Extract nnz
        //
        if(handle->pointer_mode == rocsparse_pointer_mode_device)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                nnz_total_dev_host_ptr, d_nnz, sizeof(I), hipMemcpyDeviceToDevice, handle->stream));
        }
        else
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                nnz_total_dev_host_ptr, d_nnz, sizeof(I), hipMemcpyDeviceToHost, handle->stream));
            RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
        }

        //
        // Free rocprim buffer, if allocated
        //
        if(temp_alloc)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(d_nnz, handle->stream));
        }
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, JTYPE, TTYPE)                                \
    template rocsparse_status rocsparse::nnz_impl<ITYPE, JTYPE, TTYPE>( \
        rocsparse_handle          handle,                               \
        rocsparse_direction       dir,                                  \
        rocsparse_order           order,                                \
        JTYPE                     m,                                    \
        JTYPE                     n,                                    \
        const rocsparse_mat_descr descr,                                \
        const TTYPE*              A,                                    \
        int64_t                   ld,                                   \
        ITYPE*                    nnz_per_row_columns,                  \
        ITYPE*                    nnz_total_dev_host_ptr);

INSTANTIATE(int32_t, int32_t, _Float16);
INSTANTIATE(int32_t, int32_t, rocsparse_bfloat16);
INSTANTIATE(int32_t, int32_t, float);
INSTANTIATE(int32_t, int32_t, double);
INSTANTIATE(int32_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_bfloat16);
INSTANTIATE(int64_t, int32_t, _Float16);
INSTANTIATE(int64_t, int32_t, float);
INSTANTIATE(int64_t, int32_t, double);
INSTANTIATE(int64_t, int32_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int32_t, rocsparse_double_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_bfloat16);
INSTANTIATE(int64_t, int64_t, _Float16);
INSTANTIATE(int64_t, int64_t, float);
INSTANTIATE(int64_t, int64_t, double);
INSTANTIATE(int64_t, int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, int64_t, rocsparse_double_complex);
#undef INSTANTIATE

extern "C" {

//
// Definition of the C-implementation.
//
#define CAPI_IMPL(name_, type_)                                                     \
    rocsparse_status name_(rocsparse_handle          handle,                        \
                           rocsparse_direction       dir,                           \
                           rocsparse_int             m,                             \
                           rocsparse_int             n,                             \
                           const rocsparse_mat_descr descr,                         \
                           const type_*              A,                             \
                           rocsparse_int             ld,                            \
                           rocsparse_int*            nnz_per_row_columns,           \
                           rocsparse_int*            nnz_total_dev_host_ptr)        \
    {                                                                               \
        try                                                                         \
        {                                                                           \
            ROCSPARSE_ROUTINE_TRACE;                                                \
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::nnz_impl(handle,                   \
                                                          dir,                      \
                                                          rocsparse_order_column,   \
                                                          m,                        \
                                                          n,                        \
                                                          descr,                    \
                                                          A,                        \
                                                          ld,                       \
                                                          nnz_per_row_columns,      \
                                                          nnz_total_dev_host_ptr)); \
            return rocsparse_status_success;                                        \
        }                                                                           \
        catch(...)                                                                  \
        {                                                                           \
            RETURN_ROCSPARSE_EXCEPTION();                                           \
        }                                                                           \
    }

//
// C-implementations.
//
CAPI_IMPL(rocsparse_snnz, float);
CAPI_IMPL(rocsparse_dnnz, double);
CAPI_IMPL(rocsparse_cnnz, rocsparse_float_complex);
CAPI_IMPL(rocsparse_znnz, rocsparse_double_complex);

//
// Undefine the macro CAPI_IMPL.
//
#undef CAPI_IMPL
}
