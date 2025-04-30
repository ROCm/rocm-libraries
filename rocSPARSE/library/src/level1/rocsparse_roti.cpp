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

#include "internal/level1/rocsparse_roti.h"
#include "rocsparse_roti.hpp"
#include "roti_device.h"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void roti_kernel(I nnz,
                     T* __restrict__ x_val,
                     const I* __restrict__ x_ind,
                     T* __restrict__ y,
                     ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, c),
                     ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, s),
                     rocsparse_index_base idx_base,
                     bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(c);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(s);
        if(c == static_cast<T>(1) && s == static_cast<T>(0))
        {
            return;
        }
        rocsparse::roti_device<BLOCKSIZE>(nnz, x_val, x_ind, y, c, s, idx_base);
    }
}

template <typename I, typename T>
rocsparse_status rocsparse::roti_template(rocsparse_handle     handle, //0
                                          I                    nnz, //1
                                          T*                   x_val, //2
                                          const I*             x_ind, //3
                                          T*                   y, //4
                                          const T*             c, //5
                                          const T*             s, //6
                                          rocsparse_index_base idx_base) //7
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for valid handle
    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Logging // TODO bench logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xroti"),
                         nnz,
                         (const void*&)x_val,
                         (const void*&)x_ind,
                         (const void*&)y,
                         LOG_TRACE_SCALAR_VALUE(handle, c),
                         LOG_TRACE_SCALAR_VALUE(handle, s),
                         idx_base);

    // Check index base
    ROCSPARSE_CHECKARG_SIZE(1, nnz);
    ROCSPARSE_CHECKARG_ARRAY(2, nnz, x_val);
    ROCSPARSE_CHECKARG_ARRAY(3, nnz, x_ind);
    ROCSPARSE_CHECKARG_ARRAY(4, nnz, y);
    ROCSPARSE_CHECKARG_POINTER(5, c);
    ROCSPARSE_CHECKARG_POINTER(6, s);
    ROCSPARSE_CHECKARG_ENUM(7, idx_base);

    // Quick return if possible
    if(nnz == 0)
    {
        return rocsparse_status_success;
    }

    // Stream
    hipStream_t stream = handle->stream;

#define ROTI_DIM 512
    dim3 roti_blocks((nnz - 1) / ROTI_DIM + 1);
    dim3 roti_threads(ROTI_DIM);

    const bool on_host = (handle->pointer_mode == rocsparse_pointer_mode_host);
    if(on_host && (*c == static_cast<T>(1) && *s == static_cast<T>(0)))
    {
        return rocsparse_status_success;
    }

    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::roti_kernel<ROTI_DIM>),
                                       roti_blocks,
                                       roti_threads,
                                       0,
                                       stream,
                                       nnz,
                                       x_val,
                                       x_ind,
                                       y,
                                       ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, c),
                                       ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, s),
                                       idx_base,
                                       handle->pointer_mode == rocsparse_pointer_mode_host);
#undef ROTI_DIM
    return rocsparse_status_success;
}

#define INSTANTIATE(I, T)                                                           \
    template rocsparse_status rocsparse::roti_template(rocsparse_handle     handle, \
                                                       I                    nnz,    \
                                                       T*                   x_val,  \
                                                       const I*             x_ind,  \
                                                       T*                   y,      \
                                                       const T*             c,      \
                                                       const T*             s,      \
                                                       rocsparse_index_base idx_base)

INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_double_complex);

INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_double_complex);

#undef INSTANTIATE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_sroti(rocsparse_handle     handle,
                                            rocsparse_int        nnz,
                                            float*               x_val,
                                            const rocsparse_int* x_ind,
                                            float*               y,
                                            const float*         c,
                                            const float*         s,
                                            rocsparse_index_base idx_base)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::roti_template(handle, nnz, x_val, x_ind, y, c, s, idx_base));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

extern "C" rocsparse_status rocsparse_droti(rocsparse_handle     handle,
                                            rocsparse_int        nnz,
                                            double*              x_val,
                                            const rocsparse_int* x_ind,
                                            double*              y,
                                            const double*        c,
                                            const double*        s,
                                            rocsparse_index_base idx_base)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::roti_template(handle, nnz, x_val, x_ind, y, c, s, idx_base));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
