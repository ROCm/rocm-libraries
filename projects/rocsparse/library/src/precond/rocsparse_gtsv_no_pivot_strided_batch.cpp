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

#include "rocsparse_gtsv_no_pivot_strided_batch.hpp"
#include "internal/precond/rocsparse_gtsv.h"

#include "gtsv_nopivot_strided_batch_device.h"

#define LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_POW2_STAGE1(T, block_size, stride, iter) \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                                \
        (rocsparse::gtsv_nopivot_strided_batch_pcr_pow2_stage1_kernel<block_size>),    \
        dim3(((m - 1) / block_size + 1), batch_count, 1),                              \
        dim3(block_size, 1, 1),                                                        \
        0,                                                                             \
        handle->stream,                                                                \
        stride,                                                                        \
        m,                                                                             \
        batch_count,                                                                   \
        ((iter == 0) ? batch_stride : m),                                              \
        ((iter == 0) ? dl : (((iter & 1) == 0) ? da0 : da1)),                          \
        ((iter == 0) ? d : (((iter & 1) == 0) ? db0 : db1)),                           \
        ((iter == 0) ? du : (((iter & 1) == 0) ? dc0 : dc1)),                          \
        ((iter == 0) ? x : (((iter & 1) == 0) ? drhs0 : drhs1)),                       \
        (((iter & 1) == 0) ? da1 : da0),                                               \
        (((iter & 1) == 0) ? db1 : db0),                                               \
        (((iter & 1) == 0) ? dc1 : dc0),                                               \
        (((iter & 1) == 0) ? drhs1 : drhs0));

#define LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_CR_POW2_STAGE2(T, block_size, iter)      \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                            \
        (rocsparse::gtsv_nopivot_strided_batch_cr_pow2_stage2_kernel<block_size>), \
        dim3(subsystem_count, batch_count, 1),                                     \
        dim3(block_size),                                                          \
        0,                                                                         \
        handle->stream,                                                            \
        m,                                                                         \
        batch_count,                                                               \
        batch_stride,                                                              \
        (((iter & 1) != 0) ? da1 : da0),                                           \
        (((iter & 1) != 0) ? db1 : db0),                                           \
        (((iter & 1) != 0) ? dc1 : dc0),                                           \
        (((iter & 1) != 0) ? drhs1 : drhs0),                                       \
        x);

#define LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_CR_POW2_SHARED(T, block_size)            \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                            \
        (rocsparse::gtsv_nopivot_strided_batch_cr_pow2_shared_kernel<block_size>), \
        dim3(batch_count),                                                         \
        dim3(block_size),                                                          \
        0,                                                                         \
        handle->stream,                                                            \
        m,                                                                         \
        batch_count,                                                               \
        batch_stride,                                                              \
        dl,                                                                        \
        d,                                                                         \
        du,                                                                        \
        x);

#define LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_POW2_SHARED(T, block_size)            \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                             \
        (rocsparse::gtsv_nopivot_strided_batch_pcr_pow2_shared_kernel<block_size>), \
        dim3(batch_count),                                                          \
        dim3(block_size),                                                           \
        0,                                                                          \
        handle->stream,                                                             \
        m,                                                                          \
        batch_count,                                                                \
        batch_stride,                                                               \
        dl,                                                                         \
        d,                                                                          \
        du,                                                                         \
        x);

#define LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_CRPCR_POW2_SHARED(T, block_size, pcr_size)            \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                                         \
        (rocsparse::gtsv_nopivot_strided_batch_crpcr_pow2_shared_kernel<block_size, pcr_size>), \
        dim3(batch_count),                                                                      \
        dim3(block_size),                                                                       \
        0,                                                                                      \
        handle->stream,                                                                         \
        m,                                                                                      \
        batch_count,                                                                            \
        batch_stride,                                                                           \
        dl,                                                                                     \
        d,                                                                                      \
        du,                                                                                     \
        x);

#define LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_SHARED(T, block_size)            \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                        \
        (rocsparse::gtsv_nopivot_strided_batch_pcr_shared_kernel<block_size>), \
        dim3(batch_count),                                                     \
        dim3(block_size),                                                      \
        0,                                                                     \
        handle->stream,                                                        \
        m,                                                                     \
        batch_count,                                                           \
        batch_stride,                                                          \
        dl,                                                                    \
        d,                                                                     \
        du,                                                                    \
        x);

#define LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_STAGE1(T, block_size, stride, iter) \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                           \
        (rocsparse::gtsv_nopivot_strided_batch_pcr_stage1_kernel<block_size>),    \
        dim3(((m - 1) / block_size + 1), batch_count, 1),                         \
        dim3(block_size),                                                         \
        0,                                                                        \
        handle->stream,                                                           \
        stride,                                                                   \
        m,                                                                        \
        batch_count,                                                              \
        ((iter == 0) ? batch_stride : m),                                         \
        ((iter == 0) ? dl : (((iter & 1) == 0) ? da0 : da1)),                     \
        ((iter == 0) ? d : (((iter & 1) == 0) ? db0 : db1)),                      \
        ((iter == 0) ? du : (((iter & 1) == 0) ? dc0 : dc1)),                     \
        ((iter == 0) ? x : (((iter & 1) == 0) ? drhs0 : drhs1)),                  \
        (((iter & 1) == 0) ? da1 : da0),                                          \
        (((iter & 1) == 0) ? db1 : db0),                                          \
        (((iter & 1) == 0) ? dc1 : dc0),                                          \
        (((iter & 1) == 0) ? drhs1 : drhs0));

#define LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_STAGE2(T, block_size, iter)      \
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(                                        \
        (rocsparse::gtsv_nopivot_strided_batch_pcr_stage2_kernel<block_size>), \
        dim3(subsystem_count, batch_count, 1),                                 \
        dim3(block_size),                                                      \
        0,                                                                     \
        handle->stream,                                                        \
        m,                                                                     \
        batch_count,                                                           \
        batch_stride,                                                          \
        (((iter & 1) != 0) ? da1 : da0),                                       \
        (((iter & 1) != 0) ? db1 : db0),                                       \
        (((iter & 1) != 0) ? dc1 : dc0),                                       \
        (((iter & 1) != 0) ? drhs1 : drhs0),                                   \
        x);

template <typename T>
rocsparse_status
    rocsparse::gtsv_no_pivot_strided_batch_buffer_size_template(rocsparse_handle handle,
                                                                rocsparse_int    m,
                                                                const T*         dl,
                                                                const T*         d,
                                                                const T*         du,
                                                                const T*         x,
                                                                rocsparse_int    batch_count,
                                                                rocsparse_int    batch_stride,
                                                                size_t*          buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    rocsparse::log_trace(
        handle,
        rocsparse::replaceX<T>("rocsparse_Xgtsv_no_pivot_strided_batch_buffer_size"),
        m,
        (const void*&)dl,
        (const void*&)d,
        (const void*&)du,
        (const void*&)x,
        batch_count,
        batch_stride,
        (const void*&)buffer_size);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG(1, m, (m <= 1), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG(7, batch_stride, (batch_stride < m), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_SIZE(6, batch_count);

    ROCSPARSE_CHECKARG_ARRAY(2, batch_count, dl);
    ROCSPARSE_CHECKARG_ARRAY(3, batch_count, d);
    ROCSPARSE_CHECKARG_ARRAY(4, batch_count, du);
    ROCSPARSE_CHECKARG_ARRAY(5, batch_count, x);
    ROCSPARSE_CHECKARG_POINTER(8, buffer_size);

    // Quick return if possible
    if(batch_count == 0)
    {
        *buffer_size = 0;
        return rocsparse_status_success;
    }

    if(m <= 512)
    {
        *buffer_size = 0;
    }
    else
    {
        *buffer_size = 0;

        *buffer_size += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256; // da0
        *buffer_size += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256; // da1
        *buffer_size += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256; // db0
        *buffer_size += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256; // db1
        *buffer_size += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256; // dc0
        *buffer_size += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256; // dc1
        *buffer_size += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256; // drhs0
        *buffer_size += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256; // drhs1
    }

    return rocsparse_status_success;
}

namespace rocsparse
{
    template <typename T>
    rocsparse_status gtsv_no_pivot_strided_batch_small_template(rocsparse_handle handle,
                                                                rocsparse_int    m,
                                                                const T*         dl,
                                                                const T*         d,
                                                                const T*         du,
                                                                T*               x,
                                                                rocsparse_int    batch_count,
                                                                rocsparse_int    batch_stride,
                                                                void*            temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse_host_assert(m <= 512, "This function is designed for m <= 512.");

        // Run special algorithm if m is power of 2
        if((m & (m - 1)) == 0)
        {
            if(m == 2)
            {
                LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_POW2_SHARED(T, 2);
            }
            else if(m == 4)
            {
                LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_POW2_SHARED(T, 4);
            }
            else if(m == 8)
            {
                LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_POW2_SHARED(T, 8);
            }
            else if(m == 16)
            {
                LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_POW2_SHARED(T, 16);
            }
            else if(m == 32)
            {
                LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_POW2_SHARED(T, 32);
            }
            else if(m == 64)
            {
                LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_POW2_SHARED(T, 64);
            }
            else if(m == 128)
            {
                LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_CRPCR_POW2_SHARED(T, 64, 64);
            }
            else if(m == 256)
            {
                LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_CRPCR_POW2_SHARED(T, 128, 64);
            }
            else if(m == 512)
            {
                LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_CRPCR_POW2_SHARED(T, 256, 64);
            }
        }
        else
        {
            if(m <= 4)
            {
                LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_SHARED(T, 4);
            }
            else if(m <= 8)
            {
                LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_SHARED(T, 8);
            }
            else if(m <= 16)
            {
                LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_SHARED(T, 16);
            }
            else if(m <= 32)
            {
                LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_SHARED(T, 32);
            }
            else if(m <= 64)
            {
                LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_SHARED(T, 64);
            }
            else if(m <= 128)
            {
                LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_SHARED(T, 128);
            }
            else if(m <= 256)
            {
                LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_SHARED(T, 256);
            }
            else if(m <= 512)
            {
                LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_SHARED(T, 512);
            }
        }

        return rocsparse_status_success;
    }

    template <typename T>
    rocsparse_status gtsv_no_pivot_strided_batch_large_template(rocsparse_handle handle,
                                                                rocsparse_int    m,
                                                                const T*         dl,
                                                                const T*         d,
                                                                const T*         du,
                                                                T*               x,
                                                                rocsparse_int    batch_count,
                                                                rocsparse_int    batch_stride,
                                                                void*            temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        rocsparse_host_assert(m > 512, "This function is designed for m > 512.");

        char* ptr = reinterpret_cast<char*>(temp_buffer);
        T*    da0 = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;
        T* da1 = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;
        T* db0 = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;
        T* db1 = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;
        T* dc0 = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;
        T* dc1 = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;
        T* drhs0 = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;
        T* drhs1 = reinterpret_cast<T*>(ptr);
        // ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;

        RETURN_IF_HIP_ERROR(hipMemsetAsync(
            da0, 0, ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemsetAsync(
            da1, 0, ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemsetAsync(
            db0, 0, ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemsetAsync(
            db1, 0, ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemsetAsync(
            dc0, 0, ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemsetAsync(
            dc1, 0, ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemsetAsync(
            drhs0, 0, ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemsetAsync(
            drhs1, 0, ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256, handle->stream));

        // Run special algorithm if m is power of 2
        if((m & (m - 1)) == 0)
        {
            // Stage1: Break large tridiagonal system into multiple smaller systems
            // using parallel cyclic reduction so that each sub system is of size 512.
            rocsparse_int iter = static_cast<rocsparse_int>(rocsparse::log2(m))
                                 - static_cast<rocsparse_int>(rocsparse::log2(512));

            rocsparse_int stride = 1;
            for(rocsparse_int i = 0; i < iter; i++)
            {
                LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_POW2_STAGE1(T, 256, stride, i);

                stride *= 2;
            }

            // Stage2: Solve the many systems from stage1 in parallel using cyclic reduction.
            rocsparse_int subsystem_count = 1 << iter;

            LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_CR_POW2_STAGE2(T, 256, iter);
        }
        else
        {
            // Stage1: Break large tridiagonal system into multiple smaller systems
            // using parallel cyclic reduction so that each sub system is of size 512 or less.
            rocsparse_int iter = static_cast<rocsparse_int>(rocsparse::log2(m))
                                 - static_cast<rocsparse_int>(rocsparse::log2(512)) + 1;

            rocsparse_int stride = 1;
            for(rocsparse_int i = 0; i < iter; i++)
            {
                LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_STAGE1(T, 256, stride, i);

                stride *= 2;
            }

            // Stage2: Solve the many systems from stage1 in parallel using cyclic reduction.
            rocsparse_int subsystem_count = 1 << iter;

            LAUNCH_GTSV_NOPIVOT_STRIDED_BATCH_PCR_STAGE2(T, 512, iter);
        }

        return rocsparse_status_success;
    }
}

template <typename T>
rocsparse_status rocsparse::gtsv_no_pivot_strided_batch_template(rocsparse_handle handle,
                                                                 rocsparse_int    m,
                                                                 const T*         dl,
                                                                 const T*         d,
                                                                 const T*         du,
                                                                 T*               x,
                                                                 rocsparse_int    batch_count,
                                                                 rocsparse_int    batch_stride,
                                                                 void*            temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xgtsv_no_pivot_strided_batch"),
                         m,
                         (const void*&)dl,
                         (const void*&)d,
                         (const void*&)du,
                         (const void*&)x,
                         batch_count,
                         batch_stride,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG(1, m, (m <= 1), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG(7, batch_stride, (batch_stride < m), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_SIZE(6, batch_count);

    ROCSPARSE_CHECKARG_ARRAY(2, batch_count, dl);
    ROCSPARSE_CHECKARG_ARRAY(3, batch_count, d);
    ROCSPARSE_CHECKARG_ARRAY(4, batch_count, du);
    ROCSPARSE_CHECKARG_ARRAY(5, batch_count, x);
    ROCSPARSE_CHECKARG(
        8, temp_buffer, (m > 512 && temp_buffer == nullptr), rocsparse_status_invalid_pointer);

    if(batch_count == 0)
    {
        return rocsparse_status_success;
    }

    // If m is small we can solve the systems entirely in shared memory
    if(m <= 512)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gtsv_no_pivot_strided_batch_small_template(
            handle, m, dl, d, du, x, batch_count, batch_stride, temp_buffer));
        return rocsparse_status_success;
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gtsv_no_pivot_strided_batch_large_template(
        handle, m, dl, d, du, x, batch_count, batch_stride, temp_buffer));
    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
#define C_IMPL(NAME, TYPE)                                                                     \
    extern "C" rocsparse_status NAME(rocsparse_handle handle,                                  \
                                     rocsparse_int    m,                                       \
                                     const TYPE*      dl,                                      \
                                     const TYPE*      d,                                       \
                                     const TYPE*      du,                                      \
                                     const TYPE*      x,                                       \
                                     rocsparse_int    batch_count,                             \
                                     rocsparse_int    batch_stride,                            \
                                     size_t*          buffer_size)                             \
    try                                                                                        \
    {                                                                                          \
        ROCSPARSE_ROUTINE_TRACE;                                                               \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gtsv_no_pivot_strided_batch_buffer_size_template( \
            handle, m, dl, d, du, x, batch_count, batch_stride, buffer_size));                 \
        return rocsparse_status_success;                                                       \
    }                                                                                          \
    catch(...)                                                                                 \
    {                                                                                          \
        RETURN_ROCSPARSE_EXCEPTION();                                                          \
    }

C_IMPL(rocsparse_sgtsv_no_pivot_strided_batch_buffer_size, float);
C_IMPL(rocsparse_dgtsv_no_pivot_strided_batch_buffer_size, double);
C_IMPL(rocsparse_cgtsv_no_pivot_strided_batch_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zgtsv_no_pivot_strided_batch_buffer_size, rocsparse_double_complex);

#undef C_IMPL

#define C_IMPL(NAME, TYPE)                                                         \
    extern "C" rocsparse_status NAME(rocsparse_handle handle,                      \
                                     rocsparse_int    m,                           \
                                     const TYPE*      dl,                          \
                                     const TYPE*      d,                           \
                                     const TYPE*      du,                          \
                                     TYPE*            x,                           \
                                     rocsparse_int    batch_count,                 \
                                     rocsparse_int    batch_stride,                \
                                     void*            temp_buffer)                 \
    try                                                                            \
    {                                                                              \
        ROCSPARSE_ROUTINE_TRACE;                                                   \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gtsv_no_pivot_strided_batch_template( \
            handle, m, dl, d, du, x, batch_count, batch_stride, temp_buffer));     \
        return rocsparse_status_success;                                           \
    }                                                                              \
    catch(...)                                                                     \
    {                                                                              \
        RETURN_ROCSPARSE_EXCEPTION();                                              \
    }

C_IMPL(rocsparse_sgtsv_no_pivot_strided_batch, float);
C_IMPL(rocsparse_dgtsv_no_pivot_strided_batch, double);
C_IMPL(rocsparse_cgtsv_no_pivot_strided_batch, rocsparse_float_complex);
C_IMPL(rocsparse_zgtsv_no_pivot_strided_batch, rocsparse_double_complex);

#undef C_IMPL
