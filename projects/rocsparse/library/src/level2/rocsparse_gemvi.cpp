/*! \file */
/* ************************************************************************
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/level2/rocsparse_gemvi.h"
#include "rocsparse_common.h"
#include "rocsparse_gemvi.hpp"

#include "gemvi_device.h"
#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, uint32_t WFSIZE, typename I, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void gemvi_kernel(I m,
                      I n,
                      ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                      const T* __restrict__ A,
                      int64_t lda,
                      I       nnz,
                      const T* __restrict__ x_val,
                      const I* __restrict__ x_ind,
                      ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                      T* __restrict__ y,
                      rocsparse_index_base idx_base,
                      bool                 is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);

        if(alpha != static_cast<T>(0) || beta != static_cast<T>(1))
        {
            rocsparse::gemvi_device<BLOCKSIZE, WFSIZE>(
                m, n, alpha, A, lda, nnz, x_val, x_ind, beta, y, idx_base);
        }
    }

    template <typename I, typename T>
    rocsparse_status gemvi_dispatch(rocsparse_handle     handle,
                                    rocsparse_operation  trans,
                                    I                    m,
                                    I                    n,
                                    const T*             alpha_device_host,
                                    const T*             A,
                                    int64_t              lda,
                                    I                    nnz,
                                    const T*             x_val,
                                    const I*             x_ind,
                                    const T*             beta_device_host,
                                    T*                   y,
                                    rocsparse_index_base idx_base,
                                    void*                temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

#define GEMVI_DIM 1024
        // If nnz is zero, only compute beta * y
        if(nnz == 0)
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::scale_array(handle, m, beta_device_host, y));

            return rocsparse_status_success;
        }

        if(trans == rocsparse_operation_none)
        {
            if(handle->wavefront_size == 32)
            {
                dim3 gemvi_blocks((m - 1) / 32 + 1);
                dim3 gemvi_threads(GEMVI_DIM);

                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::gemvi_kernel<GEMVI_DIM, 32>),
                    gemvi_blocks,
                    gemvi_threads,
                    0,
                    handle->stream,
                    m,
                    n,
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                    A,
                    lda,
                    nnz,
                    x_val,
                    x_ind,
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
                    y,
                    idx_base,
                    handle->pointer_mode == rocsparse_pointer_mode_host);
            }
            else
            {
                rocsparse_host_assert(handle->wavefront_size == 64,
                                      "Wrong wavefront size dispatch.");

                dim3 gemvi_blocks((m - 1) / 64 + 1);
                dim3 gemvi_threads(GEMVI_DIM);

                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                    (rocsparse::gemvi_kernel<GEMVI_DIM, 64>),
                    gemvi_blocks,
                    gemvi_threads,
                    0,
                    handle->stream,
                    m,
                    n,
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, alpha_device_host),
                    A,
                    lda,
                    nnz,
                    x_val,
                    x_ind,
                    ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
                    y,
                    idx_base,
                    handle->pointer_mode == rocsparse_pointer_mode_host);
            }
#undef GEMVI_DIM
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }

        return rocsparse_status_success;
    }

    template <typename I, typename T>
    rocsparse_status gemvi_template(rocsparse_handle     handle, //0
                                    rocsparse_operation  trans, //1
                                    I                    m, //2
                                    I                    n, //3
                                    const T*             alpha_device_host, //4
                                    const T*             A, //5
                                    int64_t              lda, //6
                                    I                    nnz, //7
                                    const T*             x_val, //8
                                    const I*             x_ind, //9
                                    const T*             beta_device_host, //10
                                    T*                   y, //11
                                    rocsparse_index_base idx_base, //12
                                    void*                temp_buffer) //13
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Check for valid handle
        ROCSPARSE_CHECKARG_HANDLE(0, handle);

        // Logging
        rocsparse::log_trace(handle,
                             rocsparse::replaceX<T>("rocsparse_Xgemvi"),
                             trans,
                             m,
                             n,
                             LOG_TRACE_SCALAR_VALUE(handle, alpha_device_host),
                             (const void*&)A,
                             lda,
                             nnz,
                             (const void*&)x_val,
                             (const void*&)x_ind,
                             LOG_TRACE_SCALAR_VALUE(handle, beta_device_host),
                             (const void*&)y,
                             idx_base,
                             (const void*&)temp_buffer);

        // Check operation mode
        ROCSPARSE_CHECKARG_ENUM(1, trans);

        // Check index base
        ROCSPARSE_CHECKARG_ENUM(12, idx_base);

        // Check sizes
        ROCSPARSE_CHECKARG_SIZE(2, m);
        ROCSPARSE_CHECKARG_SIZE(3, n);
        ROCSPARSE_CHECKARG_SIZE(7, nnz);

        // nnz of sparse vector cannot exceed its size
        ROCSPARSE_CHECKARG(7, nnz, (nnz > n), rocsparse_status_invalid_size);

        // Check leading dimension
        ROCSPARSE_CHECKARG(6,
                           lda,
                           ((lda < m) && (trans == rocsparse_operation_none)),
                           rocsparse_status_invalid_size);
        ROCSPARSE_CHECKARG(6,
                           lda,
                           ((lda < n) && (trans != rocsparse_operation_none)),
                           rocsparse_status_invalid_size);

        // Quick return if possible
        if(m == 0)
        {
            return rocsparse_status_success;
        }

        ROCSPARSE_CHECKARG_POINTER(4, alpha_device_host);

        // Check invalid pointers
        if(m > 0 && n > 0 && nnz > 0)
        {
            ROCSPARSE_CHECKARG_POINTER(5, A);
            ROCSPARSE_CHECKARG_POINTER(8, x_val);
            ROCSPARSE_CHECKARG_POINTER(9, x_ind);
            // Allow temp_buffer to be nullptr
        }

        ROCSPARSE_CHECKARG_POINTER(10, beta_device_host);
        ROCSPARSE_CHECKARG_POINTER(11, y);

        // Quick return if there is no work to do - alpha can be (valid) nullptr!
        if(handle->pointer_mode == rocsparse_pointer_mode_host)
        {
            if(alpha_device_host == nullptr && *beta_device_host == static_cast<T>(1))
            {
                return rocsparse_status_success;
            }

            if(alpha_device_host != nullptr)
            {
                if(*alpha_device_host == static_cast<T>(0)
                   && *beta_device_host == static_cast<T>(1))
                {
                    return rocsparse_status_success;
                }
            }
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gemvi_dispatch(handle,
                                                            trans,
                                                            m,
                                                            n,
                                                            alpha_device_host,
                                                            A,
                                                            lda,
                                                            nnz,
                                                            x_val,
                                                            x_ind,
                                                            beta_device_host,
                                                            y,
                                                            idx_base,
                                                            temp_buffer));
        return rocsparse_status_success;
    }
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

// Definition of the C-implementation

// rocsparse_xgemvi_buffer_size
#define CAPI_IMPL(name_, type_)                             \
    rocsparse_status name_(rocsparse_handle    handle,      \
                           rocsparse_operation trans,       \
                           rocsparse_int       m,           \
                           rocsparse_int       n,           \
                           rocsparse_int       nnz,         \
                           size_t*             buffer_size) \
    try                                                     \
    {                                                       \
        ROCSPARSE_ROUTINE_TRACE;                            \
        *buffer_size = 0;                                   \
        return rocsparse_status_success;                    \
    }                                                       \
    catch(...)                                              \
    {                                                       \
        RETURN_ROCSPARSE_EXCEPTION();                       \
    }

// C-implementations
CAPI_IMPL(rocsparse_sgemvi_buffer_size, float);
CAPI_IMPL(rocsparse_dgemvi_buffer_size, double);
CAPI_IMPL(rocsparse_cgemvi_buffer_size, rocsparse_float_complex);
CAPI_IMPL(rocsparse_zgemvi_buffer_size, rocsparse_double_complex);

// Undefine the CAPI_IMPL macro
#undef CAPI_IMPL

// rocsparse_xgemvi
#define CAPI_IMPL(name_, type_)                                                \
    rocsparse_status name_(rocsparse_handle     handle,                        \
                           rocsparse_operation  trans,                         \
                           rocsparse_int        m,                             \
                           rocsparse_int        n,                             \
                           const type_*         alpha,                         \
                           const type_*         A,                             \
                           rocsparse_int        lda,                           \
                           rocsparse_int        nnz,                           \
                           const type_*         x_val,                         \
                           const rocsparse_int* x_ind,                         \
                           const type_*         beta,                          \
                           type_*               y,                             \
                           rocsparse_index_base idx_base,                      \
                           void*                temp_buffer)                   \
    {                                                                          \
        try                                                                    \
        {                                                                      \
            ROCSPARSE_ROUTINE_TRACE;                                           \
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::gemvi_template(handle,        \
                                                                trans,         \
                                                                m,             \
                                                                n,             \
                                                                alpha,         \
                                                                A,             \
                                                                lda,           \
                                                                nnz,           \
                                                                x_val,         \
                                                                x_ind,         \
                                                                beta,          \
                                                                y,             \
                                                                idx_base,      \
                                                                temp_buffer)); \
            return rocsparse_status_success;                                   \
        }                                                                      \
        catch(...)                                                             \
        {                                                                      \
            RETURN_ROCSPARSE_EXCEPTION();                                      \
        }                                                                      \
    }

// C-implementations
CAPI_IMPL(rocsparse_sgemvi, float);
CAPI_IMPL(rocsparse_dgemvi, double);
CAPI_IMPL(rocsparse_cgemvi, rocsparse_float_complex);
CAPI_IMPL(rocsparse_zgemvi, rocsparse_double_complex);

// Undefine the CAPI_IMPL macro
#undef CAPI_IMPL
}
