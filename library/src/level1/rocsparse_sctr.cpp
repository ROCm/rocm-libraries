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

#include "internal/level1/rocsparse_sctr.h"
#include "rocsparse_sctr.hpp"

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

#include "sctr_device.h"

template <typename I, typename T>
rocsparse_status rocsparse::sctr_template(rocsparse_handle     handle,
                                          I                    nnz,
                                          const T*             x_val,
                                          const I*             x_ind,
                                          T*                   y,
                                          rocsparse_index_base idx_base)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Check for valid handle
    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xsctr"),
                         nnz,
                         (const void*&)x_val,
                         (const void*&)x_ind,
                         (const void*&)y,
                         idx_base);

    ROCSPARSE_CHECKARG_SIZE(1, nnz);
    ROCSPARSE_CHECKARG_ARRAY(2, nnz, x_val);
    ROCSPARSE_CHECKARG_ARRAY(3, nnz, x_ind);
    ROCSPARSE_CHECKARG_ARRAY(4, nnz, y);
    ROCSPARSE_CHECKARG_ENUM(5, idx_base);

    // Quick return if possible
    if(nnz == 0)
    {
        return rocsparse_status_success;
    }

    // Stream
    hipStream_t stream = handle->stream;

#define SCTR_DIM 512
    dim3 sctr_blocks((nnz - 1) / SCTR_DIM + 1);
    dim3 sctr_threads(SCTR_DIM);

    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::sctr_kernel<SCTR_DIM>),
                                       sctr_blocks,
                                       sctr_threads,
                                       0,
                                       stream,
                                       nnz,
                                       x_val,
                                       x_ind,
                                       y,
                                       idx_base);
#undef SCTR_DIM
    return rocsparse_status_success;
}

extern "C" rocsparse_status rocsparse_ssctr(rocsparse_handle     handle,
                                            rocsparse_int        nnz,
                                            const float*         x_val,
                                            const rocsparse_int* x_ind,
                                            float*               y,
                                            rocsparse_index_base idx_base)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sctr_template(handle, nnz, x_val, x_ind, y, idx_base));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status rocsparse_dsctr(rocsparse_handle     handle,
                                            rocsparse_int        nnz,
                                            const double*        x_val,
                                            const rocsparse_int* x_ind,
                                            double*              y,
                                            rocsparse_index_base idx_base)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sctr_template(handle, nnz, x_val, x_ind, y, idx_base));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status rocsparse_csctr(rocsparse_handle               handle,
                                            rocsparse_int                  nnz,
                                            const rocsparse_float_complex* x_val,
                                            const rocsparse_int*           x_ind,
                                            rocsparse_float_complex*       y,
                                            rocsparse_index_base           idx_base)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sctr_template(handle, nnz, x_val, x_ind, y, idx_base));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status rocsparse_zsctr(rocsparse_handle                handle,
                                            rocsparse_int                   nnz,
                                            const rocsparse_double_complex* x_val,
                                            const rocsparse_int*            x_ind,
                                            rocsparse_double_complex*       y,
                                            rocsparse_index_base            idx_base)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sctr_template(handle, nnz, x_val, x_ind, y, idx_base));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

extern "C" rocsparse_status rocsparse_isctr(rocsparse_handle     handle,
                                            rocsparse_int        nnz,
                                            const rocsparse_int* x_val,
                                            const rocsparse_int* x_ind,
                                            rocsparse_int*       y,
                                            rocsparse_index_base idx_base)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sctr_template(handle, nnz, x_val, x_ind, y, idx_base));
    return rocsparse_status_success;
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}

#define INSTANTIATE(I, T)                                                           \
    template rocsparse_status rocsparse::sctr_template(rocsparse_handle     handle, \
                                                       I                    nnz,    \
                                                       const T*             x_val,  \
                                                       const I*             x_ind,  \
                                                       T*                   y,      \
                                                       rocsparse_index_base idx_base)

INSTANTIATE(int32_t, int8_t);
INSTANTIATE(int32_t, _Float16);
INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_double_complex);

INSTANTIATE(int64_t, int8_t);
INSTANTIATE(int64_t, _Float16);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_double_complex);

#undef INSTANTIATE
