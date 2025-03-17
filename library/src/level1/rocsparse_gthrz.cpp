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
#include "internal/level1/rocsparse_gthrz.h"
#include "rocsparse_gthrz.hpp"

#include "gthrz_device.h"

template <typename T>
rocsparse_status rocsparse::gthrz_template(rocsparse_handle     handle,
                                           rocsparse_int        nnz,
                                           T*                   y,
                                           T*                   x_val,
                                           const rocsparse_int* x_ind,
                                           rocsparse_index_base idx_base)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for valid handle
    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xgthrz"),
                         nnz,
                         (const void*&)y,
                         (const void*&)x_val,
                         (const void*&)x_ind,
                         idx_base);

    // Check index base
    ROCSPARSE_CHECKARG_SIZE(1, nnz);
    ROCSPARSE_CHECKARG_ARRAY(2, nnz, y);
    ROCSPARSE_CHECKARG_ARRAY(3, nnz, x_val);
    ROCSPARSE_CHECKARG_ARRAY(4, nnz, x_ind);
    ROCSPARSE_CHECKARG_ENUM(5, idx_base);

    // Quick return
    if(nnz == 0)
    {
        return rocsparse_status_success;
    }

    // Stream
    hipStream_t stream = handle->stream;

#define GTHRZ_DIM 512
    dim3 gthrz_blocks((nnz - 1) / GTHRZ_DIM + 1);
    dim3 gthrz_threads(GTHRZ_DIM);

    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gthrz_kernel<GTHRZ_DIM>),
                                       gthrz_blocks,
                                       gthrz_threads,
                                       0,
                                       stream,
                                       nnz,
                                       y,
                                       x_val,
                                       x_ind,
                                       idx_base);
#undef GTHRZ_DIM
    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_sgthrz(rocsparse_handle     handle,
                                             rocsparse_int        nnz,
                                             float*               y,
                                             float*               x_val,
                                             const rocsparse_int* x_ind,
                                             rocsparse_index_base idx_base)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gthrz_template(handle, nnz, y, x_val, x_ind, idx_base));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_dgthrz(rocsparse_handle     handle,
                                             rocsparse_int        nnz,
                                             double*              y,
                                             double*              x_val,
                                             const rocsparse_int* x_ind,
                                             rocsparse_index_base idx_base)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gthrz_template(handle, nnz, y, x_val, x_ind, idx_base));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_cgthrz(rocsparse_handle         handle,
                                             rocsparse_int            nnz,
                                             rocsparse_float_complex* y,
                                             rocsparse_float_complex* x_val,
                                             const rocsparse_int*     x_ind,
                                             rocsparse_index_base     idx_base)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gthrz_template(handle, nnz, y, x_val, x_ind, idx_base));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_zgthrz(rocsparse_handle          handle,
                                             rocsparse_int             nnz,
                                             rocsparse_double_complex* y,
                                             rocsparse_double_complex* x_val,
                                             const rocsparse_int*      x_ind,
                                             rocsparse_index_base      idx_base)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gthrz_template(handle, nnz, y, x_val, x_ind, idx_base));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
