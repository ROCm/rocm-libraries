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

#include "rocsparse_gpsv_interleaved_batch.hpp"
#include "internal/precond/rocsparse_gpsv.h"

#include "gpsv_interleaved_batch_device.h"

template <>
inline bool rocsparse::enum_utils::is_invalid(rocsparse_gpsv_interleaved_alg value_)
{
    switch(value_)
    {
    case rocsparse_gpsv_interleaved_alg_default:
    case rocsparse_gpsv_interleaved_alg_qr:
    {
        return false;
    }
    }
    return true;
};

template <typename T>
rocsparse_status
    rocsparse::gpsv_interleaved_batch_buffer_size_template(rocsparse_handle               handle,
                                                           rocsparse_gpsv_interleaved_alg alg,
                                                           rocsparse_int                  m,
                                                           const T*                       ds,
                                                           const T*                       dl,
                                                           const T*                       d,
                                                           const T*                       du,
                                                           const T*                       dw,
                                                           const T*                       x,
                                                           rocsparse_int batch_count,
                                                           rocsparse_int batch_stride,
                                                           size_t*       buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xgpsv_interleaved_batch_buffer_size"),
                         alg,
                         m,
                         (const void*&)ds,
                         (const void*&)dl,
                         (const void*&)d,
                         (const void*&)du,
                         (const void*&)dw,
                         (const void*&)x,
                         batch_count,
                         batch_stride,
                         (const void*&)buffer_size);

    ROCSPARSE_CHECKARG_ENUM(1, alg);
    ROCSPARSE_CHECKARG_SIZE(2, m);
    ROCSPARSE_CHECKARG(2, m, (m < 3), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_ARRAY(3, batch_count, ds);
    ROCSPARSE_CHECKARG_ARRAY(4, batch_count, dl);
    ROCSPARSE_CHECKARG_ARRAY(5, batch_count, d);
    ROCSPARSE_CHECKARG_ARRAY(6, batch_count, du);
    ROCSPARSE_CHECKARG_ARRAY(7, batch_count, dw);
    ROCSPARSE_CHECKARG_ARRAY(8, batch_count, x);
    ROCSPARSE_CHECKARG_SIZE(9, batch_count);
    ROCSPARSE_CHECKARG_SIZE(10, batch_stride);
    ROCSPARSE_CHECKARG(
        10, batch_stride, (batch_stride < batch_count), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_POINTER(11, buffer_size);

    // Quick return if possible
    if(batch_count == 0)
    {
        *buffer_size = 0;
        return rocsparse_status_success;
    }

    *buffer_size = 0;

    if(std::is_same<T, float>() || std::is_same<T, double>())
    {
        *buffer_size += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256; // dt1
        *buffer_size += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256; // dt2
        *buffer_size += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256; // dB
    }
    else
    {
        *buffer_size += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256; // r3
        *buffer_size += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256; // r4
    }

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse::gpsv_interleaved_batch_template(rocsparse_handle               handle,
                                                            rocsparse_gpsv_interleaved_alg alg,
                                                            rocsparse_int                  m,
                                                            T*                             ds,
                                                            T*                             dl,
                                                            T*                             d,
                                                            T*                             du,
                                                            T*                             dw,
                                                            T*                             x,
                                                            rocsparse_int batch_count,
                                                            rocsparse_int batch_stride,
                                                            void*         temp_buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);

    // Logging
    rocsparse::log_trace(handle,
                         rocsparse::replaceX<T>("rocsparse_Xgpsv_interleaved_batch"),
                         alg,
                         m,
                         (const void*&)ds,
                         (const void*&)dl,
                         (const void*&)d,
                         (const void*&)du,
                         (const void*&)dw,
                         (const void*&)x,
                         batch_count,
                         batch_stride,
                         (const void*&)temp_buffer);

    ROCSPARSE_CHECKARG_ENUM(1, alg);
    ROCSPARSE_CHECKARG_SIZE(2, m);
    ROCSPARSE_CHECKARG(2, m, (m < 3), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_ARRAY(3, batch_count, ds);
    ROCSPARSE_CHECKARG_ARRAY(4, batch_count, dl);
    ROCSPARSE_CHECKARG_ARRAY(5, batch_count, d);
    ROCSPARSE_CHECKARG_ARRAY(6, batch_count, du);
    ROCSPARSE_CHECKARG_ARRAY(7, batch_count, dw);
    ROCSPARSE_CHECKARG_ARRAY(8, batch_count, x);
    ROCSPARSE_CHECKARG_SIZE(9, batch_count);
    ROCSPARSE_CHECKARG_SIZE(10, batch_stride);
    ROCSPARSE_CHECKARG(
        10, batch_stride, (batch_stride < batch_count), rocsparse_status_invalid_size);
    ROCSPARSE_CHECKARG_ARRAY(11, batch_count, temp_buffer);

    // Quick return if possible
    if(batch_count == 0)
    {
        return rocsparse_status_success;
    }

    // Stream
    hipStream_t stream = handle->stream;

    // Buffer
    char* ptr = reinterpret_cast<char*>(temp_buffer);

    if(std::is_same<T, float>() || std::is_same<T, double>())
    {
        T* dt1 = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;

        T* dt2 = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;

        T* B = reinterpret_cast<T*>(ptr);

        // Initialize buffers with zero
        RETURN_IF_HIP_ERROR(hipMemsetAsync(dt1, 0, sizeof(T) * m * batch_count, stream));
        RETURN_IF_HIP_ERROR(hipMemsetAsync(dt2, 0, sizeof(T) * m * batch_count, stream));

#define GPSV_DIM 256
        dim3 gpsv_blocks((batch_count - 1) / GPSV_DIM + 1);
        dim3 gpsv_threads(GPSV_DIM);

        // Copy strided B into buffer
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::gpsv_strided_gather<GPSV_DIM>),
                                           gpsv_blocks,
                                           gpsv_threads,
                                           0,
                                           stream,
                                           m,
                                           batch_count,
                                           batch_stride,
                                           x,
                                           B);

        // Launch kernel
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::gpsv_interleaved_batch_householder_qr_kernel<GPSV_DIM>),
            gpsv_blocks,
            gpsv_threads,
            0,
            stream,
            m,
            batch_count,
            batch_stride,
            ds,
            dl,
            d,
            du,
            dw,
            x,
            dt1,
            dt2,
            B);
#undef GPSV_DIM
    }
    else
    {
        T* r3 = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;
        T* r4 = reinterpret_cast<T*>(ptr);
        ptr += ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256;

        RETURN_IF_HIP_ERROR(hipMemsetAsync(
            r3, 0, ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256, handle->stream));
        RETURN_IF_HIP_ERROR(hipMemsetAsync(
            r4, 0, ((sizeof(T) * m * batch_count - 1) / 256 + 1) * 256, handle->stream));

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::gpsv_interleaved_batch_givens_qr_kernel<128>),
            dim3(((batch_count - 1) / 128 + 1), 1, 1),
            dim3(128, 1, 1),
            0,
            handle->stream,
            m,
            batch_count,
            batch_stride,
            ds,
            dl,
            d,
            du,
            dw,
            r3,
            r4,
            x);
    }

    return rocsparse_status_success;
}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
#define C_IMPL(NAME, TYPE)                                                                  \
    extern "C" rocsparse_status NAME(rocsparse_handle               handle,                 \
                                     rocsparse_gpsv_interleaved_alg alg,                    \
                                     rocsparse_int                  m,                      \
                                     const TYPE*                    ds,                     \
                                     const TYPE*                    dl,                     \
                                     const TYPE*                    d,                      \
                                     const TYPE*                    du,                     \
                                     const TYPE*                    dw,                     \
                                     const TYPE*                    x,                      \
                                     rocsparse_int                  batch_count,            \
                                     rocsparse_int                  batch_stride,           \
                                     size_t*                        buffer_size)            \
    try                                                                                     \
    {                                                                                       \
        ROCSPARSE_ROUTINE_TRACE;                                                            \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gpsv_interleaved_batch_buffer_size_template(   \
            handle, alg, m, ds, dl, d, du, dw, x, batch_count, batch_stride, buffer_size)); \
        return rocsparse_status_success;                                                    \
    }                                                                                       \
    catch(...)                                                                              \
    {                                                                                       \
        RETURN_ROCSPARSE_EXCEPTION();                                                       \
    }

C_IMPL(rocsparse_sgpsv_interleaved_batch_buffer_size, float);
C_IMPL(rocsparse_dgpsv_interleaved_batch_buffer_size, double);
C_IMPL(rocsparse_cgpsv_interleaved_batch_buffer_size, rocsparse_float_complex);
C_IMPL(rocsparse_zgpsv_interleaved_batch_buffer_size, rocsparse_double_complex);
#undef C_IMPL

#define C_IMPL(NAME, TYPE)                                                                  \
    extern "C" rocsparse_status NAME(rocsparse_handle               handle,                 \
                                     rocsparse_gpsv_interleaved_alg alg,                    \
                                     rocsparse_int                  m,                      \
                                     TYPE*                          ds,                     \
                                     TYPE*                          dl,                     \
                                     TYPE*                          d,                      \
                                     TYPE*                          du,                     \
                                     TYPE*                          dw,                     \
                                     TYPE*                          x,                      \
                                     rocsparse_int                  batch_count,            \
                                     rocsparse_int                  batch_stride,           \
                                     void*                          temp_buffer)            \
    try                                                                                     \
    {                                                                                       \
        ROCSPARSE_ROUTINE_TRACE;                                                            \
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gpsv_interleaved_batch_template(               \
            handle, alg, m, ds, dl, d, du, dw, x, batch_count, batch_stride, temp_buffer)); \
        return rocsparse_status_success;                                                    \
    }                                                                                       \
    catch(...)                                                                              \
    {                                                                                       \
        RETURN_ROCSPARSE_EXCEPTION();                                                       \
    }

C_IMPL(rocsparse_sgpsv_interleaved_batch, float);
C_IMPL(rocsparse_dgpsv_interleaved_batch, double);
C_IMPL(rocsparse_cgpsv_interleaved_batch, rocsparse_float_complex);
C_IMPL(rocsparse_zgpsv_interleaved_batch, rocsparse_double_complex);
#undef C_IMPL
