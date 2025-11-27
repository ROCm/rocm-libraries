/*! \file */
/* ************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_common.h"
#include "rocsparse_common.hpp"
#include "rocsparse_utility.hpp"

#include <hip/hip_runtime.h>

namespace rocsparse
{

    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCSPARSE_DEVICE_ILF void valset_2d_device(
        I m, I n, int64_t ld, T value, T* __restrict__ array, rocsparse_order order)
    {
        I gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        if(gid >= m * n)
        {
            return;
        }

        I wid = (order == rocsparse_order_column) ? gid / m : gid / n;
        I lid = (order == rocsparse_order_column) ? gid % m : gid % n;

        array[lid + ld * wid] = value;
    }

    template <uint32_t BLOCKSIZE, typename I, typename A, typename T>
    ROCSPARSE_DEVICE_ILF void scale_device(I length, T scalar, A* __restrict__ array)
    {
        const I gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        if(gid >= length)
        {
            return;
        }

        if(scalar == static_cast<T>(0))
        {
            array[gid] = static_cast<A>(0);
        }
        else
        {
            array[gid] *= scalar;
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename X, typename Y, typename T>
    ROCSPARSE_DEVICE_ILF void axpby_device(
        I length, T alpha, const X* __restrict__ x_array, T beta, Y* __restrict__ y_array)
    {
        const I gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        if(gid >= length)
        {
            return;
        }

        T tmp = static_cast<T>(0);
        if(beta != static_cast<T>(0))
        {
            tmp = fma(beta, static_cast<T>(y_array[gid]), tmp);
        }
        if(alpha != static_cast<T>(0))
        {
            tmp = fma(alpha, static_cast<T>(x_array[gid]), tmp);
        }
        y_array[gid] = static_cast<Y>(tmp);
    }

    template <uint32_t BLOCKSIZE, typename I, typename X, typename Y, typename T>
    ROCSPARSE_DEVICE_ILF void axpby_batched_device(I               length,
                                                   rocsparse_int   num_extra,
                                                   const T*        gamma_values,
                                                   const X* const* x_arrays,
                                                   T               beta,
                                                   Y* __restrict__ y_array)
    {
        const I gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        if(gid >= length)
        {
            return;
        }

        T tmp = static_cast<T>(0);
        if(beta != static_cast<T>(0))
        {
            tmp = fma(beta, static_cast<T>(y_array[gid]), tmp);
        }

        // Add contributions from all extra vectors, each with its own gamma
        for(rocsparse_int i = 0; i < num_extra; ++i)
        {
            T gamma = gamma_values[i]; // gamma is a scalar value from the array
            if(gamma != static_cast<T>(0))
            {
                tmp = rocsparse::fma<T>(gamma, rocsparse::nontemporal_load(x_arrays[i] + gid), tmp);
            }
        }

        y_array[gid] = static_cast<Y>(tmp);
    }

    template <uint32_t BLOCKSIZE, typename I, typename A, typename T>
    ROCSPARSE_DEVICE_ILF void scale_2d_device(
        I m, I n, int64_t ld, int64_t stride, T value, A* __restrict__ array, rocsparse_order order)
    {
        I gid   = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;
        I batch = hipBlockIdx_y;

        if(gid >= m * n)
        {
            return;
        }

        I wid = (order == rocsparse_order_column) ? gid / m : gid / n;
        I lid = (order == rocsparse_order_column) ? gid % m : gid % n;

        if(value == static_cast<T>(0))
        {
            array[lid + ld * wid + stride * batch] = static_cast<A>(0);
        }
        else
        {
            array[lid + ld * wid + stride * batch] *= value;
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_DEVICE_ILF void copy_device(int64_t length,
                                          const I* __restrict__ in,
                                          J* __restrict__ out,
                                          rocsparse_index_base idx_base_in,
                                          rocsparse_index_base idx_base_out)
    {
        const uint32_t gid = BLOCKSIZE * hipBlockIdx_x + hipThreadIdx_x;

        if(gid < length)
        {
            out[gid] = in[gid] - idx_base_in + idx_base_out;
        }
    }

    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_DEVICE_ILF void copy_and_scale_device(int64_t length,
                                                    const T* __restrict__ in,
                                                    T* __restrict__ out,
                                                    T scalar)
    {
        const uint32_t gid = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;

        if(gid >= length)
        {
            return;
        }

        if(scalar == static_cast<T>(0))
        {
            out[gid] = static_cast<T>(0);
        }
        else
        {
            out[gid] = in[gid] * scalar;
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void valset_2d_kernel(I m, I n, int64_t ld, T value, T* array, rocsparse_order order)
    {
        rocsparse::valset_2d_device<BLOCKSIZE>(m, n, ld, value, array, order);
    }

    template <uint32_t BLOCKSIZE, typename I, typename A, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void scale_kernel(I length,
                      ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, scalar),
                      A* __restrict__ array,
                      bool is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(scalar);
        if(scalar != static_cast<T>(1))
        {
            rocsparse::scale_device<BLOCKSIZE>(length, scalar, array);
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename X, typename Y, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void axpby_kernel(I length,
                      ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                      const X* __restrict__ x_array,
                      ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                      Y* __restrict__ y_array,
                      bool is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(alpha);
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);

        rocsparse::axpby_device<BLOCKSIZE>(length, alpha, x_array, beta, y_array);
    }

    template <uint32_t BLOCKSIZE, typename I, typename X, typename Y, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void axpby_batched_kernel(I               length,
                              rocsparse_int   num_extra,
                              const T*        gamma_values,
                              const X* const* x_arrays,
                              ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, beta),
                              Y* __restrict__ y_array,
                              bool is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(beta);

        rocsparse::axpby_batched_device<BLOCKSIZE>(
            length, num_extra, gamma_values, x_arrays, beta, y_array);
    }

    // Helper kernel to copy gamma values from device pointers
    template <typename T>
    ROCSPARSE_KERNEL(256)
    void gather_gamma_values_kernel(rocsparse_int num_extra, const T** gamma_ptrs, T* gamma_values)
    {
        rocsparse_int idx = hipBlockIdx_x * 256 + hipThreadIdx_x;
        if(idx < num_extra)
        {
            if(gamma_ptrs[idx] != nullptr)
            {
                gamma_values[idx] = gamma_ptrs[idx][0];
            }
            else
            {
                gamma_values[idx] = static_cast<T>(0);
            }
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename A, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void scale_2d_kernel(I       m,
                         I       n,
                         int64_t ld,
                         int64_t stride,
                         ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, scalar),
                         A* __restrict__ array,
                         rocsparse_order order,
                         bool            is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(scalar);

        if(scalar != static_cast<T>(1))
        {
            rocsparse::scale_2d_device<BLOCKSIZE>(m, n, ld, stride, scalar, array, order);
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void copy_kernel(int64_t              length,
                     const I*             in,
                     J*                   out,
                     rocsparse_index_base idx_base_in,
                     rocsparse_index_base idx_base_out)
    {
        rocsparse::copy_device<BLOCKSIZE>(length, in, out, idx_base_in, idx_base_out);
    }

    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void copy_and_scale_kernel(int64_t  length,
                               const T* in,
                               T*       out,
                               ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, scalar),
                               bool is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(scalar);
        rocsparse::copy_and_scale_device<BLOCKSIZE>(length, in, out, scalar);
    }
}

template <typename I, typename T>
rocsparse_status rocsparse::valset_2d(
    rocsparse_handle handle, I m, I n, int64_t ld, T value, T* array, rocsparse_order order)
{
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::valset_2d_kernel<256>),
                                       dim3((int64_t(m) * n - 1) / 256 + 1),
                                       dim3(256),
                                       0,
                                       handle->stream,
                                       m,
                                       n,
                                       ld,
                                       value,
                                       array,
                                       order);

    return rocsparse_status_success;
}

template <typename I, typename A, typename T>
rocsparse_status
    rocsparse::scale_array(rocsparse_handle handle, I length, const T* scalar_device_host, A* array)
{
    if(length > 0)
    {
        const bool on_host = handle->pointer_mode == rocsparse_pointer_mode_host;
        if(on_host && *scalar_device_host == 0)
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(array, 0, sizeof(A) * length, handle->stream));
        }
        else if((on_host && *scalar_device_host != 1) || on_host == false)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::scale_kernel<256>),
                dim3((length - 1) / 256 + 1),
                dim3(256),
                0,
                handle->stream,
                length,
                ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, scalar_device_host),
                array,
                handle->pointer_mode == rocsparse_pointer_mode_host);
        }
    }
    return rocsparse_status_success;
}

template <typename I, typename X, typename Y, typename T>
rocsparse_status rocsparse::axpby_array_batched(rocsparse_handle handle,
                                                I                length,
                                                rocsparse_int    num_extra,
                                                const T*         gamma_device_host,
                                                const X**        x_arrays,
                                                const T*         beta_device_host,
                                                Y*               y_array)
{
    if(length > 0 && num_extra > 0)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::axpby_batched_kernel<256>),
            dim3((length - 1) / 256 + 1),
            dim3(256),
            0,
            handle->stream,
            length,
            num_extra,
            gamma_device_host,
            x_arrays,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, beta_device_host),
            y_array,
            handle->pointer_mode == rocsparse_pointer_mode_host);
    }
    else if(length > 0 && num_extra == 0)
    {
        // If no extra vectors, just scale y by beta
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::scale_array(handle, length, beta_device_host, y_array));
    }
    return rocsparse_status_success;
}

template <typename I, typename A, typename T>
rocsparse_status rocsparse::scale_2d_array(rocsparse_handle handle,
                                           I                m,
                                           I                n,
                                           int64_t          ld,
                                           int64_t          batch_count,
                                           int64_t          stride,
                                           const T*         scalar_device_host,
                                           A*               array,
                                           rocsparse_order  order)
{
    const bool on_host = handle->pointer_mode == rocsparse_pointer_mode_host;
    if((on_host && *scalar_device_host != 1) || on_host == false)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::scale_2d_kernel<256>),
            dim3((int64_t(m) * n - 1) / 256 + 1, batch_count),
            dim3(256),
            0,
            handle->stream,
            m,
            n,
            ld,
            stride,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, scalar_device_host),
            array,
            order,
            handle->pointer_mode == rocsparse_pointer_mode_host);
    }
    return rocsparse_status_success;
}

template <typename I, typename J>
rocsparse_status rocsparse::copy(rocsparse_handle     handle,
                                 int64_t              length,
                                 const I*             in,
                                 J*                   out,
                                 rocsparse_index_base idx_base_in,
                                 rocsparse_index_base idx_base_out)
{
    if(length > 0)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::copy_kernel<256>),
                                           dim3((length - 1) / 256 + 1),
                                           dim3(256),
                                           0,
                                           handle->stream,
                                           length,
                                           in,
                                           out,
                                           idx_base_in,
                                           idx_base_out);
    }

    return rocsparse_status_success;
}

template <typename T>
rocsparse_status rocsparse::copy_and_scale(
    rocsparse_handle handle, int64_t length, const T* in, T* out, const T* scalar_device_host)
{
    if(length > 0)
    {
        const bool on_host = handle->pointer_mode == rocsparse_pointer_mode_host;
        if(on_host && *scalar_device_host == 0)
        {
            RETURN_IF_HIP_ERROR(hipMemsetAsync(out, 0, sizeof(T) * length, handle->stream));
        }
        else if(on_host && *scalar_device_host == 1)
        {
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                out, in, sizeof(T) * length, hipMemcpyDeviceToDevice, handle->stream));
        }
        else
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::copy_and_scale_kernel<256>),
                dim3((length - 1) / 256 + 1),
                dim3(256),
                0,
                handle->stream,
                length,
                in,
                out,
                ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, scalar_device_host),
                handle->pointer_mode == rocsparse_pointer_mode_host);
        }
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(ITYPE, TTYPE)                                           \
    template rocsparse_status rocsparse::valset_2d(rocsparse_handle handle, \
                                                   ITYPE            m,      \
                                                   ITYPE            n,      \
                                                   int64_t          ld,     \
                                                   TTYPE            value,  \
                                                   TTYPE*           array,  \
                                                   rocsparse_order  order);

INSTANTIATE(int32_t, _Float16);
INSTANTIATE(int32_t, rocsparse_bfloat16);
INSTANTIATE(int32_t, float);
INSTANTIATE(int32_t, double);
INSTANTIATE(int32_t, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex);

INSTANTIATE(int64_t, _Float16);
INSTANTIATE(int64_t, rocsparse_bfloat16);
INSTANTIATE(int64_t, float);
INSTANTIATE(int64_t, double);
INSTANTIATE(int64_t, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex);
#undef INSTANTIATE

#define INSTANTIATE(ITYPE, ATYPE, TTYPE)                                                          \
    template rocsparse_status rocsparse::scale_array(                                             \
        rocsparse_handle handle, ITYPE length, const TTYPE* scalar_device_host, ATYPE* array);    \
    template rocsparse_status rocsparse::axpby_array_batched(rocsparse_handle handle,             \
                                                             ITYPE            length,             \
                                                             rocsparse_int    num_extra,          \
                                                             const TTYPE*     gamma_device_array, \
                                                             const ATYPE**    x_arrays,           \
                                                             const TTYPE*     beta_device_host,   \
                                                             ATYPE*           y_array);

INSTANTIATE(int32_t, rocsparse_bfloat16, float);
INSTANTIATE(int32_t, _Float16, float);
INSTANTIATE(int32_t, int32_t, int32_t);
INSTANTIATE(int32_t, float, float);
INSTANTIATE(int32_t, double, double);
INSTANTIATE(int32_t, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex, rocsparse_double_complex);

INSTANTIATE(int64_t, rocsparse_bfloat16, float);
INSTANTIATE(int64_t, _Float16, float);
INSTANTIATE(int64_t, int32_t, int32_t);
INSTANTIATE(int64_t, float, float);
INSTANTIATE(int64_t, double, double);
INSTANTIATE(int64_t, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex, rocsparse_double_complex);
#undef INSTANTIATE

#define INSTANTIATE(I, A, T)                                                                 \
    template rocsparse_status rocsparse::scale_2d_array(rocsparse_handle handle,             \
                                                        I                m,                  \
                                                        I                n,                  \
                                                        int64_t          ld,                 \
                                                        int64_t          batch_count,        \
                                                        int64_t          stride,             \
                                                        const T*         scalar_device_host, \
                                                        A*               array,              \
                                                        rocsparse_order  order);

INSTANTIATE(int32_t, _Float16, _Float16);
INSTANTIATE(int32_t, int32_t, int32_t);
INSTANTIATE(int32_t, float, float);
INSTANTIATE(int32_t, double, double);
INSTANTIATE(int32_t, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex, rocsparse_double_complex);

INSTANTIATE(int64_t, _Float16, _Float16);
INSTANTIATE(int64_t, int32_t, int32_t);
INSTANTIATE(int64_t, float, float);
INSTANTIATE(int64_t, double, double);
INSTANTIATE(int64_t, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex, rocsparse_double_complex);
#undef INSTANTIATE

#define INSTANTIATE(I, J)                                                       \
    template rocsparse_status rocsparse::copy(rocsparse_handle     handle,      \
                                              int64_t              length,      \
                                              const I*             in,          \
                                              J*                   out,         \
                                              rocsparse_index_base idx_base_in, \
                                              rocsparse_index_base idx_base_out);

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int32_t, int64_t);
INSTANTIATE(int64_t, int32_t);
INSTANTIATE(int64_t, int64_t);

#undef INSTANTIATE

#define INSTANTIATE(T)                                                           \
    template rocsparse_status rocsparse::copy_and_scale(rocsparse_handle handle, \
                                                        int64_t          length, \
                                                        const T*         in,     \
                                                        T*               out,    \
                                                        const T*         scalar_device_host);

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(rocsparse_float_complex);
INSTANTIATE(rocsparse_double_complex);
#undef INSTANTIATE
