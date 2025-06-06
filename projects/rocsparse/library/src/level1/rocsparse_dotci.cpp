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

#include "internal/level1/rocsparse_dotci.h"
#include "dotci_device.h"
#include "rocsparse_control.hpp"
#include "rocsparse_dotci.hpp"
#include "rocsparse_utility.hpp"

template <typename T, typename I, typename X, typename Y>
rocsparse_status rocsparse::dotci_template(rocsparse_handle     handle,
                                           I                    nnz,
                                           const X*             x_val,
                                           const I*             x_ind,
                                           const Y*             y,
                                           T*                   result,
                                           rocsparse_index_base idx_base)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for valid handle
    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xdotci"),
                         nnz,
                         (const void*&)x_val,
                         (const void*&)x_ind,
                         (const void*&)y,
                         LOG_TRACE_SCALAR_VALUE(handle, result),
                         idx_base);

    // Check index base
    ROCSPARSE_CHECKARG_ENUM(6, idx_base);

    // Check size
    ROCSPARSE_CHECKARG_SIZE(1, nnz);

    // Quick return if possible
    if(nnz == 0)
    {
        if(result != nullptr)
        {
            if(handle->pointer_mode == rocsparse_pointer_mode_device)
            {
                RETURN_IF_HIP_ERROR(hipMemsetAsync(result, 0, sizeof(T), handle->stream));
            }
            else
            {
                *result = static_cast<T>(0);
            }

            return rocsparse_status_success;
        }
    }

    // Check pointer arguments
    ROCSPARSE_CHECKARG_POINTER(2, x_val);
    ROCSPARSE_CHECKARG_POINTER(3, x_ind);
    ROCSPARSE_CHECKARG_POINTER(4, y);
    ROCSPARSE_CHECKARG_POINTER(5, result);

    // Stream
    hipStream_t stream = handle->stream;

#define DOTCI_DIM 256
    // Get workspace from handle device buffer
    T* workspace = reinterpret_cast<T*>(handle->buffer);

    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::dotci_kernel_part1<DOTCI_DIM>),
                                       dim3(DOTCI_DIM),
                                       dim3(DOTCI_DIM),
                                       0,
                                       stream,
                                       nnz,
                                       x_val,
                                       x_ind,
                                       y,
                                       workspace,
                                       idx_base);

    if(handle->pointer_mode == rocsparse_pointer_mode_device)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::dotci_kernel_part2<DOTCI_DIM>),
                                           dim3(1),
                                           dim3(DOTCI_DIM),
                                           0,
                                           stream,
                                           workspace,
                                           result);
    }
    else
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::dotci_kernel_part2<DOTCI_DIM>),
                                           dim3(1),
                                           dim3(DOTCI_DIM),
                                           0,
                                           stream,
                                           workspace,
                                           (T*)nullptr);

        RETURN_IF_HIP_ERROR(
            hipMemcpyAsync(result, workspace, sizeof(T), hipMemcpyDeviceToHost, stream));
    }
#undef DOTCI_DIM

    return rocsparse_status_success;
}

#define INSTANTIATE(TTYPE, ITYPE)                                                    \
    template rocsparse_status rocsparse::dotci_template(rocsparse_handle     handle, \
                                                        ITYPE                nnz,    \
                                                        const TTYPE*         x_val,  \
                                                        const ITYPE*         x_ind,  \
                                                        const TTYPE*         y,      \
                                                        TTYPE*               result, \
                                                        rocsparse_index_base idx_base)

INSTANTIATE(rocsparse_float_complex, int32_t);
INSTANTIATE(rocsparse_float_complex, int64_t);
INSTANTIATE(rocsparse_double_complex, int32_t);
INSTANTIATE(rocsparse_double_complex, int64_t);
#undef INSTANTIATE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_cdotci(rocsparse_handle               handle,
                                             rocsparse_int                  nnz,
                                             const rocsparse_float_complex* x_val,
                                             const rocsparse_int*           x_ind,
                                             const rocsparse_float_complex* y,
                                             rocsparse_float_complex*       result,
                                             rocsparse_index_base           idx_base)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::dotci_template(handle, nnz, x_val, x_ind, y, result, idx_base));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_zdotci(rocsparse_handle                handle,
                                             rocsparse_int                   nnz,
                                             const rocsparse_double_complex* x_val,
                                             const rocsparse_int*            x_ind,
                                             const rocsparse_double_complex* y,
                                             rocsparse_double_complex*       result,
                                             rocsparse_index_base            idx_base)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::dotci_template(handle, nnz, x_val, x_ind, y, result, idx_base));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
