/*! \file */
/* ************************************************************************
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

namespace
{
    // Number of BLOCKSIZE-thread blocks needed to cover nelm elements, clamped to the
    // device grid.x limit. Every kernel launched with one of these grids grid-strides
    // over the full element count, so a clamped grid still covers all of it.
    //
    // Deliberately local: AISPARSE-696 (PR #11512) adds rocsparse::ceil_div() and
    // rocsparse::get_grid_size() to rocsparse_common.h, but it has not merged and
    // rocsparse_common.h/.hpp are a live conflict zone (AISPARSE-677/678/696). Once
    // #11512 lands the body below is the one-line
    //   return rocsparse::get_grid_size(rocsparse::ceil_div(nelm, BLOCKSIZE),
    //                                   handle->properties.maxGridSize[0]);
    template <uint32_t BLOCKSIZE>
    int64_t grid_size_x(rocsparse_handle handle, int64_t nelm)
    {
        return rocsparse::min((nelm - 1) / BLOCKSIZE + 1,
                              static_cast<int64_t>(handle->properties.maxGridSize[0]));
    }
}

namespace rocsparse
{
    //
    // AISPARSE-699. Two invariants hold for every kernel below.
    //
    // 1. A dense element count is the product of two index-typed extents, so it must
    //    be formed in int64_t. The int32_t product overflows at m = n = 46341, which
    //    is 8.6 GB in single precision and therefore reachable on one MI300X. The
    //    host already sized the grid from the 64-bit product, but the kernels
    //    recomputed the same product in the template index type I for their bounds
    //    check, so the guard rejected every thread of an otherwise correct grid and
    //    the matrix was left untouched.
    //
    // 2. The flattened element index has to be 64-bit too. Widening the bound alone
    //    achieves nothing: hipBlockIdx_x * BLOCKSIZE is unsigned-int arithmetic that
    //    wraps at 2^32 whatever I is, and narrowing the result into a 32-bit I wraps
    //    sooner still.
    //
    // The grid-stride loops follow the AISPARSE-666 idiom: the loop lives in the
    // __global__ wrapper and passes the element index down, leaving each __device__
    // function a straight-line single-element body. Every trip count depends only on
    // hipBlockIdx_x, hipGridDim_x, BLOCKSIZE and the element count, all block
    // uniform, so all threads of a block run the same number of iterations.
    //

    template <typename I, typename T>
    ROCSPARSE_DEVICE_ILF void valset_2d_device(
        int64_t gid, I m, I n, int64_t ld, T value, T* __restrict__ array, rocsparse_order order)
    {
        const int64_t wid = (order == rocsparse_order_column) ? gid / m : gid / n;
        const int64_t lid = (order == rocsparse_order_column) ? gid % m : gid % n;

        array[lid + ld * wid] = value;
    }

    template <typename A, typename T>
    ROCSPARSE_DEVICE_ILF void scale_device(int64_t gid, T scalar, A* __restrict__ array)
    {
        if(scalar == static_cast<T>(0))
        {
            array[gid] = static_cast<A>(0);
        }
        else
        {
            array[gid] *= scalar;
        }
    }

    template <typename X, typename Y, typename T>
    ROCSPARSE_DEVICE_ILF void axpby_device(
        int64_t gid, T alpha, const X* __restrict__ x_array, T beta, Y* __restrict__ y_array)
    {
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

    template <typename X, typename Y, typename T>
    ROCSPARSE_DEVICE_ILF void axpby_batched_device(int64_t         gid,
                                                   rocsparse_int   num_extra,
                                                   const T*        gamma_values,
                                                   const X* const* x_arrays,
                                                   T               beta,
                                                   Y* __restrict__ y_array)
    {
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

    template <typename I, typename A, typename T>
    ROCSPARSE_DEVICE_ILF void scale_2d_device(
        int64_t gid, I m, I n, int64_t ld, T value, A* __restrict__ array, rocsparse_order order)
    {
        const int64_t wid = (order == rocsparse_order_column) ? gid / m : gid / n;
        const int64_t lid = (order == rocsparse_order_column) ? gid % m : gid % n;

        if(value == static_cast<T>(0))
        {
            array[lid + ld * wid] = static_cast<A>(0);
        }
        else
        {
            array[lid + ld * wid] *= value;
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void valset_2d_kernel(I m, I n, int64_t ld, T value, T* array, rocsparse_order order)
    {
        // Character for character the expression the host used to size grid.x.
        const int64_t nelm = static_cast<int64_t>(m) * n;

        for(int64_t base = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE; base < nelm;
            base += static_cast<int64_t>(hipGridDim_x) * BLOCKSIZE)
        {
            const int64_t gid = base + hipThreadIdx_x;
            if(gid < nelm)
            {
                rocsparse::valset_2d_device(gid, m, n, ld, value, array, order);
            }
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename A, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void scale_kernel(I length,
                      ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, scalar),
                      A* __restrict__ array,
                      bool is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(scalar);
        if(scalar == static_cast<T>(1))
        {
            return;
        }

        // length is a single extent rather than a product, so it cannot overflow the
        // way m * n can. It is widened so that the index arithmetic cannot wrap and
        // so that this kernel shares one indexing idiom with the 2-D ones.
        const int64_t len = static_cast<int64_t>(length);

        for(int64_t base = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE; base < len;
            base += static_cast<int64_t>(hipGridDim_x) * BLOCKSIZE)
        {
            const int64_t gid = base + hipThreadIdx_x;
            if(gid < len)
            {
                rocsparse::scale_device(gid, scalar, array);
            }
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

        // See scale_kernel above for why length is widened.
        const int64_t len = static_cast<int64_t>(length);

        for(int64_t base = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE; base < len;
            base += static_cast<int64_t>(hipGridDim_x) * BLOCKSIZE)
        {
            const int64_t gid = base + hipThreadIdx_x;
            if(gid < len)
            {
                rocsparse::axpby_device(gid, alpha, x_array, beta, y_array);
            }
        }
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

        // See scale_kernel above for why length is widened.
        const int64_t len = static_cast<int64_t>(length);

        for(int64_t base = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE; base < len;
            base += static_cast<int64_t>(hipGridDim_x) * BLOCKSIZE)
        {
            const int64_t gid = base + hipThreadIdx_x;
            if(gid < len)
            {
                rocsparse::axpby_batched_device(
                    gid, num_extra, gamma_values, x_arrays, beta, y_array);
            }
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename A, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void scale_2d_kernel(I       m,
                         I       n,
                         int64_t ld,
                         int64_t batch_count,
                         int64_t stride,
                         ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, scalar),
                         A* __restrict__ array,
                         rocsparse_order order,
                         bool            is_host_mode)
    {
        ROCSPARSE_DEVICE_HOST_SCALAR_GET(scalar);
        if(scalar == static_cast<T>(1))
        {
            return;
        }

        // Character for character the expression the host used to size grid.x.
        const int64_t nelm = static_cast<int64_t>(m) * n;

        for(int64_t batch = hipBlockIdx_y; batch < batch_count; batch += hipGridDim_y)
        {
            A* batch_array = rocsparse::load_pointer(array, batch, stride);

            for(int64_t base = static_cast<int64_t>(hipBlockIdx_x) * BLOCKSIZE; base < nelm;
                base += static_cast<int64_t>(hipGridDim_x) * BLOCKSIZE)
            {
                const int64_t gid = base + hipThreadIdx_x;
                if(gid < nelm)
                {
                    rocsparse::scale_2d_device(gid, m, n, ld, scalar, batch_array, order);
                }
            }
        }
    }
}

template <typename I, typename T>
rocsparse_status rocsparse::valset_2d(
    rocsparse_handle handle, I m, I n, int64_t ld, T value, T* array, rocsparse_order order)
{
    // 64-bit element count, shared with the kernel bound (AISPARSE-699).
    const int64_t nelm = static_cast<int64_t>(m) * n;

    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::valset_2d_kernel<256>),
                                       dim3(grid_size_x<256>(handle, nelm)),
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
rocsparse_status rocsparse::scale_array(rocsparse_handle       handle,
                                        I                      length,
                                        rocsparse_pointer_mode pointer_mode,
                                        const T*               scalar_device_host,
                                        A*                     array)
{
    if(length > 0)
    {
        const bool on_host = pointer_mode == rocsparse_pointer_mode_host;
        if(on_host && *scalar_device_host == static_cast<T>(0))
        {
            RETURN_IF_HIP_ERROR(
                rocsparse_hipMemsetAsync(array, 0, sizeof(A) * length, handle->stream));
        }
        else if((on_host && *scalar_device_host != static_cast<T>(1)) || on_host == false)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::scale_kernel<256>),
                dim3(grid_size_x<256>(handle, static_cast<int64_t>(length))),
                dim3(256),
                0,
                handle->stream,
                length,
                ROCSPARSE_SCALAR_HOST_DEVICE_ARGUMENT(pointer_mode, scalar_device_host),
                array,
                on_host);
        }
    }
    return rocsparse_status_success;
}

template <typename I, typename A, typename T>
rocsparse_status
    rocsparse::scale_array(rocsparse_handle handle, I length, const T* scalar_device_host, A* array)
{
    return rocsparse::scale_array(handle, length, handle->pointer_mode, scalar_device_host, array);
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
            dim3(grid_size_x<256>(handle, static_cast<int64_t>(length))),
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
        // 64-bit element count, shared with the kernel bound (AISPARSE-699).
        const int64_t nelm = static_cast<int64_t>(m) * n;

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::scale_2d_kernel<256>),
            dim3(grid_size_x<256>(handle, nelm), (batch_count > 65535) ? 65535 : batch_count),
            dim3(256),
            0,
            handle->stream,
            m,
            n,
            ld,
            batch_count,
            stride,
            ROCSPARSE_DEVICE_HOST_SCALAR_ARGS(handle, scalar_device_host),
            array,
            order,
            handle->pointer_mode == rocsparse_pointer_mode_host);
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
    template rocsparse_status rocsparse::scale_array(rocsparse_handle       handle,               \
                                                     ITYPE                  length,               \
                                                     rocsparse_pointer_mode pointer_mode,         \
                                                     const TTYPE*           scalar_device_host,   \
                                                     ATYPE*                 array);                               \
    template rocsparse_status rocsparse::axpby_array_batched(rocsparse_handle handle,             \
                                                             ITYPE            length,             \
                                                             rocsparse_int    num_extra,          \
                                                             const TTYPE*     gamma_device_array, \
                                                             const ATYPE**    x_arrays,           \
                                                             const TTYPE*     beta_device_host,   \
                                                             ATYPE*           y_array);

INSTANTIATE(int32_t, rocsparse_bfloat16, float);
INSTANTIATE(int32_t, _Float16, float);
INSTANTIATE(int32_t, float, _Float16);
INSTANTIATE(int32_t, float, rocsparse_bfloat16);
INSTANTIATE(int32_t, int32_t, int32_t);
INSTANTIATE(int32_t, float, float);
INSTANTIATE(int32_t, double, double);
INSTANTIATE(int32_t, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex, rocsparse_double_complex);

INSTANTIATE(int64_t, rocsparse_bfloat16, float);
INSTANTIATE(int64_t, _Float16, float);
INSTANTIATE(int64_t, float, _Float16);
INSTANTIATE(int64_t, float, rocsparse_bfloat16);
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
INSTANTIATE(int32_t, _Float16, float);
INSTANTIATE(int32_t, rocsparse_bfloat16, rocsparse_bfloat16);
INSTANTIATE(int32_t, rocsparse_bfloat16, float);
INSTANTIATE(int32_t, int32_t, int32_t);
INSTANTIATE(int32_t, float, float);
INSTANTIATE(int32_t, double, double);
INSTANTIATE(int32_t, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(int32_t, rocsparse_double_complex, rocsparse_double_complex);

INSTANTIATE(int64_t, _Float16, _Float16);
INSTANTIATE(int64_t, _Float16, float);
INSTANTIATE(int64_t, rocsparse_bfloat16, rocsparse_bfloat16);
INSTANTIATE(int64_t, rocsparse_bfloat16, float);
INSTANTIATE(int64_t, int32_t, int32_t);
INSTANTIATE(int64_t, float, float);
INSTANTIATE(int64_t, double, double);
INSTANTIATE(int64_t, rocsparse_float_complex, rocsparse_float_complex);
INSTANTIATE(int64_t, rocsparse_double_complex, rocsparse_double_complex);
#undef INSTANTIATE
