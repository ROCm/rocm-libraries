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

#include "internal/level1/rocsparse_axpyi.h"
#include "axpyi_device.h"
#include "rocsparse_axpyi.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, typename T, typename I, typename X, typename Y>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void axpyi_kernel(I nnz,
                      ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                      const X* __restrict__ x_val,
                      const I* __restrict__ x_ind,
                      Y*                   y,
                      rocsparse_index_base idx_base,
                      bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        if(alpha != static_cast<T>(0))
        {
            rocsparse::axpyi_device<BLOCKSIZE>(nnz, alpha, x_val, x_ind, y, idx_base);
        }
    }
}

template <typename T, typename I, typename X, typename Y>
rocsparse_status rocsparse::axpyi_template(rocsparse_handle     handle,
                                           I                    nnz,
                                           const T*             alpha,
                                           const X*             x_val,
                                           const I*             x_ind,
                                           Y*                   y,
                                           rocsparse_index_base idx_base)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for valid handle
    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xaxpyi"),
                         nnz,
                         LOG_TRACE_SCALAR_VALUE(handle, alpha),
                         (const void*&)x_val,
                         (const void*&)x_ind,
                         (const void*&)y);

    // Check index base
    ROCSPARSE_CHECKARG_ENUM(6, idx_base);

    // Check size
    ROCSPARSE_CHECKARG_SIZE(1, nnz);

    // Quick return if possible
    if(nnz == 0)
    {
        return rocsparse_status_success;
    }

    // Check pointer arguments
    ROCSPARSE_CHECKARG_POINTER(2, alpha);

    if(handle->pointer_mode == rocsparse_pointer_mode_host && *alpha == static_cast<T>(0))
    {
        return rocsparse_status_success;
    }

    ROCSPARSE_CHECKARG_POINTER(3, x_val);
    ROCSPARSE_CHECKARG_POINTER(4, x_ind);
    ROCSPARSE_CHECKARG_POINTER(5, y);

    // Stream
    hipStream_t stream = handle->stream;

#define AXPYI_DIM 256
    dim3 axpyi_blocks((nnz - 1) / AXPYI_DIM + 1);
    dim3 axpyi_threads(AXPYI_DIM);

    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::axpyi_kernel<AXPYI_DIM>),
                                       axpyi_blocks,
                                       axpyi_threads,
                                       0,
                                       stream,
                                       nnz,
                                       ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha),
                                       x_val,
                                       x_ind,
                                       y,
                                       idx_base,
                                       handle->pointer_mode == rocsparse_pointer_mode_host);
#undef AXPYI_DIM
    return rocsparse_status_success;
}

#define INSTANTIATE(TTYPE, ITYPE, XTYPE, YTYPE)                                      \
    template rocsparse_status rocsparse::axpyi_template<TTYPE, ITYPE, XTYPE, YTYPE>( \
        rocsparse_handle     handle,                                                 \
        ITYPE                nnz,                                                    \
        const TTYPE*         alpha,                                                  \
        const XTYPE*         x_val,                                                  \
        const ITYPE*         x_ind,                                                  \
        YTYPE*               y,                                                      \
        rocsparse_index_base idx_base);

INSTANTIATE(float, int32_t, rocsparse_bfloat16, rocsparse_bfloat16);
INSTANTIATE(float, int32_t, _Float16, _Float16);
INSTANTIATE(float, int32_t, float, float);
INSTANTIATE(double, int32_t, double, double);
INSTANTIATE(rocsparse_float_complex, int32_t, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex, int32_t, rocsparse_double_complex, rocsparse_double_complex);
INSTANTIATE(float, int64_t, rocsparse_bfloat16, rocsparse_bfloat16);
INSTANTIATE(float, int64_t, _Float16, _Float16);
INSTANTIATE(float, int64_t, float, float);
INSTANTIATE(double, int64_t, double, double);
INSTANTIATE(rocsparse_float_complex, int64_t, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex, int64_t, rocsparse_double_complex, rocsparse_double_complex);
#undef INSTANTIATE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_saxpyi(rocsparse_handle     handle,
                                             rocsparse_int        nnz,
                                             const float*         alpha,
                                             const float*         x_val,
                                             const rocsparse_int* x_ind,
                                             float*               y,
                                             rocsparse_index_base idx_base)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::axpyi_template(handle, nnz, alpha, x_val, x_ind, y, idx_base));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_daxpyi(rocsparse_handle     handle,
                                             rocsparse_int        nnz,
                                             const double*        alpha,
                                             const double*        x_val,
                                             const rocsparse_int* x_ind,
                                             double*              y,
                                             rocsparse_index_base idx_base)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::axpyi_template(handle, nnz, alpha, x_val, x_ind, y, idx_base));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_caxpyi(rocsparse_handle               handle,
                                             rocsparse_int                  nnz,
                                             const rocsparse_float_complex* alpha,
                                             const rocsparse_float_complex* x_val,
                                             const rocsparse_int*           x_ind,
                                             rocsparse_float_complex*       y,
                                             rocsparse_index_base           idx_base)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::axpyi_template(handle, nnz, alpha, x_val, x_ind, y, idx_base));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_zaxpyi(rocsparse_handle                handle,
                                             rocsparse_int                   nnz,
                                             const rocsparse_double_complex* alpha,
                                             const rocsparse_double_complex* x_val,
                                             const rocsparse_int*            x_ind,
                                             rocsparse_double_complex*       y,
                                             rocsparse_index_base            idx_base)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::axpyi_template(handle, nnz, alpha, x_val, x_ind, y, idx_base));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
