/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once

#include "check_numerics_matrix.hpp"
#include "check_numerics_vector.hpp"
#include "device_macros.hpp"
#include "handle.hpp"
#include "rocblas_ger.hpp"

template <rocblas_int DIM_X,
          rocblas_int DIM_Y,
          rocblas_int WIN,
          bool        CONJ,
          typename T,
          typename V,
          typename U,
          typename W>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_ger_kernel(rocblas_int    m,
                   rocblas_int    n,
                   V              alpha_device_host,
                   rocblas_stride stride_alpha,
                   const U __restrict__ xa,
                   rocblas_stride shiftx,
                   int64_t        incx,
                   rocblas_stride stridex,
                   const U __restrict__ ya,
                   rocblas_stride shifty,
                   int64_t        incy,
                   rocblas_stride stridey,
                   W __restrict__ Aa,
                   rocblas_stride shifta,
                   size_t         lda,
                   rocblas_stride strideA,
                   rocblas_int    batch_count)
{
    __shared__ T xdata[DIM_X];
    __shared__ T ydata[DIM_Y * WIN];

    int num_blocksx = (m - 1) / DIM_X + 1;
    int blkx        = blockIdx.x % num_blocksx;
    int blky        = blockIdx.x / num_blocksx;
    int tx          = blkx * DIM_X + threadIdx.x;
    int ty          = blky * DIM_Y + threadIdx.y;
    ty *= WIN;

    // shared data base index
    int tyi = threadIdx.y * WIN;

    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        auto alpha = load_scalar(alpha_device_host, batch, stride_alpha);
        if(!alpha)
        {
#if DEVICE_GRID_YZ_16BIT
            continue; //iterate to the next batch in the for loop rather than return.
#else
        return;
#endif
        }

        const T* __restrict__ x = load_ptr_batch(xa, batch, shiftx, stridex);
        const T* __restrict__ y = load_ptr_batch(ya, batch, shifty, stridey);

        T* __restrict__ A = load_ptr_batch(Aa, batch, shifta, strideA);

        if(threadIdx.y == 0)
        {
            xdata[threadIdx.x] = tx < m ? x[tx * int64_t(incx)] : 0;
        }

        if(threadIdx.x < WIN)
        {
            ydata[tyi + threadIdx.x]
                = (ty + threadIdx.x < n) ? y[(ty + threadIdx.x) * int64_t(incy)] : 0;
        }

        __syncthreads();

        if(tx < m)
        {
            T x_value = alpha * xdata[threadIdx.x];

            for(int i = 0; i < WIN; i++)
            {
                int yi = ty + i;
                if(yi < n)
                    A[tx + size_t(lda) * yi]
                        += x_value * (CONJ ? conj(ydata[tyi + i]) : ydata[tyi + i]);
            }
        }
#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

//optimized kernel for SGER for gfx942
template <rocblas_int DIM_X, typename T, typename V, typename U, typename W>
ROCBLAS_KERNEL(DIM_X)
rocblas_sger_gfx942_kernel(rocblas_int    m,
                           rocblas_int    n,
                           V              alpha_device_host,
                           rocblas_stride stride_alpha,
                           const U        xa,
                           rocblas_stride shiftx,
                           int64_t        incx,
                           rocblas_stride stridex,
                           const U        ya,
                           rocblas_stride shifty,
                           int64_t        incy,
                           rocblas_stride stridey,
                           W              Aa,
                           rocblas_stride shifta,
                           int64_t        lda,
                           rocblas_stride strideA)
{
// gfx942 kernels
#if defined(__gfx942__)

    rocblas_int tx  = (blockIdx.x * DIM_X + threadIdx.x) * 2;
    rocblas_int col = blockIdx.y;

    auto alpha = load_scalar(alpha_device_host, blockIdx.z, stride_alpha);

    if(!alpha)
        return;

    const T* __restrict__ x = load_ptr_batch(xa, blockIdx.z, shiftx, stridex);
    const T* __restrict__ y = load_ptr_batch(ya, blockIdx.z, shifty, stridey);

    T* __restrict__ A = load_ptr_batch(Aa, blockIdx.z, shifta, strideA);

    const T reg_y = y[col * incy] * alpha;

    // Load two consecutive elements of vector x and matrix A
    const T x_1 = (tx < m) ? x[tx * incx] : 0;
    const T x_2 = ((tx + 1) < m) ? x[(tx + 1) * incx] : 0;

    T res_A_1 = (tx < m) ? A[tx + col * lda] : 0;
    T res_A_2 = ((tx + 1) < m) ? A[(tx + 1) + col * lda] : 0;

    //scalar-vector-vector product and add the result to the last element of  'A'.
    if((m & 1) != 0 && (tx + 1) == m)
    {
        res_A_1 += reg_y * x_1;

        A[tx + col * lda] = res_A_1;
    }

    //scalar-vector-vector product and add the result to the matrix 'A'.
    if((tx + 1) < m)
    {
        res_A_1 += reg_y * x_1;
        res_A_2 += reg_y * x_2;

        A[tx + col * lda]       = res_A_1;
        A[(tx + 1) + col * lda] = res_A_2;
    }
#endif
}

//optimized kernel for SGER
template <rocblas_int DIM_X, typename T, typename V, typename U, typename W>
ROCBLAS_KERNEL(DIM_X)
rocblas_sger_kernel(rocblas_int    m,
                    rocblas_int    n,
                    V              alpha_device_host,
                    rocblas_stride stride_alpha,
                    const U __restrict__ xa,
                    rocblas_stride shiftx,
                    int64_t        incx,
                    rocblas_stride stridex,
                    const U __restrict__ ya,
                    rocblas_stride shifty,
                    int64_t        incy,
                    rocblas_stride stridey,
                    W __restrict__ Aa,
                    rocblas_stride shifta,
                    size_t         lda,
                    rocblas_stride strideA,
                    rocblas_int    batch_count)
{
    rocblas_int tx  = threadIdx.x;
    rocblas_int col = blockIdx.x;

    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        auto alpha = load_scalar(alpha_device_host, batch, stride_alpha);

        if(!alpha)
        {
#if DEVICE_GRID_YZ_16BIT
            continue; //iterate to the next batch in the for loop rather than return.
#else
        return;
#endif
        }

        const T* __restrict__ x = load_ptr_batch(xa, batch, shiftx, stridex);
        const T* __restrict__ y = load_ptr_batch(ya, batch, shifty, stridey);

        T* __restrict__ A = load_ptr_batch(Aa, batch, shifta, strideA);

        if(tx < m)
            A += tx;

        //Each blockIdx.x takes care of the computation of each column of matrix 'A'
        A += col * size_t(lda);

        const T res_y = y[col * (int64_t)incy] * alpha;

        //scalar-vector-vector product and add the result to a Hermitian matrix 'A'.
        //If m > DIM_X, then the threads are reused and the multiplied values will be accumalated to Hermitian matrix 'A'.

        for(rocblas_int i = 0; tx + i < m; i += DIM_X)
        {
            A[i] += res_y * x[(tx + i) * int64_t(incx)];
        }
#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

//optimized double buffered load kernel for GER
template <bool        CONJ,
          rocblas_int DIM_X,
          rocblas_int DIM_Y,
          rocblas_int elements_per_thread,
          typename T,
          typename TStruct,
          typename U,
          typename W>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_ger_double_buffered_kernel(bool           host_ptr_mode,
                                   rocblas_int    m,
                                   rocblas_int    n,
                                   TStruct        alpha_device_host,
                                   rocblas_stride stride_alpha,
                                   const U __restrict__ xa,
                                   rocblas_stride shiftx,
                                   int64_t        incx,
                                   rocblas_stride stridex,
                                   const U __restrict__ ya,
                                   rocblas_stride shifty,
                                   int64_t        incy,
                                   rocblas_stride stridey,
                                   W __restrict__ Aa,
                                   rocblas_stride shifta,
                                   size_t         lda,
                                   rocblas_stride strideA,
                                   rocblas_int    batch_count)
{

    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int bx  = blockIdx.x;
    const int by  = blockIdx.y;
    const int td  = (DIM_X * ty) + tx;
    const int tx_ = td % (DIM_X / 2);
    const int ty_ = td / (DIM_X / 2);

    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        auto alpha              = host_ptr_mode ? alpha_device_host.value
                                                : load_scalar(alpha_device_host.ptr, batch, stride_alpha);
        const T* __restrict__ x = load_ptr_batch(xa, batch, shiftx, stridex);
        const T* __restrict__ y = load_ptr_batch(ya, batch, shifty, stridey);

        T* __restrict__ A = load_ptr_batch(Aa, batch, shifta, strideA);

        if(!alpha)
        {
#if DEVICE_GRID_YZ_16BIT
            continue; //iterate to the next batch in the for loop rather than return.
#else
        return;
#endif
        }

        T x_reg_upper                     = 0.0;
        T x_reg_lower                     = 0.0;
        T areg_upper[elements_per_thread] = {0.0};
        T areg_lower[elements_per_thread] = {0.0};
        T y_reg[elements_per_thread]      = {0.0};

        // Advance 'A'
        A += DIM_X * bx;
        A += by * DIM_X * size_t(lda);

        // Advance 'x'
        x += (bx * DIM_X) * int64_t(incx);

        // Advance 'y'
        y += (by * DIM_X) * int64_t(incy);

        const size_t j = ty_ * elements_per_thread * size_t(lda) + tx_;

        x_reg_upper = x[tx_ * int64_t(incx)] * alpha;
        x_reg_lower = x[((DIM_X / 2) + tx_) * int64_t(incx)] * alpha;

// read upper
#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
            areg_upper[k] = A[j + k * size_t(lda)];

// read lower
#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
        {
            areg_lower[k] = A[(DIM_X / 2) + j + k * size_t(lda)];
            y_reg[k]      = y[(ty_ * elements_per_thread + k) * int64_t(incy)];
        }

// compute upper
#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
            areg_upper[k] += x_reg_upper * (CONJ ? conj(y_reg[k]) : y_reg[k]);

// store upper
#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
            A[j + k * size_t(lda)] = areg_upper[k];

// compute lower
#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
            areg_lower[k] += x_reg_lower * (CONJ ? conj(y_reg[k]) : y_reg[k]);

// store lower
#pragma unroll
        for(int k = 0; k < elements_per_thread; k++)
            A[(DIM_X / 2) + j + k * size_t(lda)] = areg_lower[k];

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <bool CONJ, typename T, typename U, typename V, typename W>
rocblas_status rocblas_internal_ger_launcher(rocblas_handle handle,
                                             rocblas_int    m,
                                             rocblas_int    n,
                                             const V*       alpha,
                                             rocblas_stride stride_alpha,
                                             const U*       x,
                                             rocblas_stride offsetx,
                                             int64_t        incx,
                                             rocblas_stride stridex,
                                             const U*       y,
                                             rocblas_stride offsety,
                                             int64_t        incy,
                                             rocblas_stride stridey,
                                             W*             A,
                                             rocblas_stride offsetA,
                                             int64_t        lda,
                                             rocblas_stride strideA,
                                             rocblas_int    batch_count)
{
    // Quick return if possible. Not Argument error
    if(!m || !n || !batch_count)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->get_stream();

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    auto shiftx = incx < 0 ? offsetx - ptrdiff_t(incx) * (m - 1) : offsetx;
    auto shifty = incy < 0 ? offsety - ptrdiff_t(incy) * (n - 1) : offsety;

    //Identifying the precision to have an appropriate optimization
    static constexpr bool is_float         = std::is_same_v<T, float>;
    static constexpr bool is_double        = std::is_same_v<T, double>;
    static constexpr bool is_complex_float = std::is_same_v<T, rocblas_float_complex>;

    int batches = handle->getBatchGridDim((int)batch_count);

    bool is_gfx90a = handle->getArch() == 910 ? true : false;
    bool is_gfx942 = handle->getArch() == 942 ? true : false;

#define ger_KARGS(alpha_)                                                                  \
    ger_grid, ger_threads, 0, rocblas_stream, m, n, alpha_, stride_alpha, x, shiftx, incx, \
        stridex, y, shifty, incy, stridey, A, offsetA, lda, strideA, batch_count

#define ger_gfx942_KARGS(alpha_)                                                           \
    ger_grid, ger_threads, 0, rocblas_stream, m, n, alpha_, stride_alpha, x, shiftx, incx, \
        stridex, y, shifty, incy, stridey, A, offsetA, lda, strideA

    //optimized double buffered loads kernel for float, double and float_complex precisions in gfx90a
    if(is_gfx90a && (m > 2000) && (m == n)
       && ((m % 64 == 0 && (is_double || is_complex_float)) || ((m % 128 == 0) && is_float)))
    {
        //The following rocblas_ger_double_buffered_kernel is only valid for the multiples of DIM_X
        static constexpr int DIM_X               = is_float ? 128 : 64;
        static constexpr int DIM_Y               = is_float ? 8 : 16;
        static constexpr int elements_per_thread = DIM_X / (2 * DIM_Y);

        const int block_x = m / DIM_X;
        const int block_y = n / DIM_X;
        dim3      ger_threads(DIM_X, DIM_Y);
        dim3      ger_grid(block_x, block_y, batches);

        bool host_ptr_mode = handle->pointer_mode == rocblas_pointer_mode_host;
        rocblas_internal_val_ptr<V> alpha_device_host(host_ptr_mode, alpha);

        ROCBLAS_LAUNCH_KERNEL(
            (rocblas_ger_double_buffered_kernel<CONJ, DIM_X, DIM_Y, elements_per_thread, T>),
            ger_grid,
            ger_threads,
            0,
            rocblas_stream,
            host_ptr_mode,
            m,
            n,
            alpha_device_host,
            stride_alpha,
            x,
            shiftx,
            incx,
            stridex,
            y,
            shifty,
            incy,
            stridey,
            A,
            offsetA,
            lda,
            strideA,
            batch_count);
    }
    else if(is_float && m > 1024)
    {
        if(is_gfx942)
        {
            static constexpr int DIM_X   = 256;
            rocblas_int          blocksX = (m - 1) / (DIM_X * 2) + 1;
            dim3                 ger_grid(blocksX, n, batch_count);
            dim3                 ger_threads(DIM_X);

            if(handle->pointer_mode == rocblas_pointer_mode_device)
            {
                ROCBLAS_LAUNCH_KERNEL((rocblas_sger_gfx942_kernel<DIM_X, T>),
                                      ger_gfx942_KARGS(alpha));
            }
            else
            {
                ROCBLAS_LAUNCH_KERNEL((rocblas_sger_gfx942_kernel<DIM_X, T>),
                                      ger_gfx942_KARGS(*alpha));
            }
        }
        else
        {
            static constexpr int DIM_X = 1024;
            dim3                 ger_grid(n, 1, batches);
            dim3                 ger_threads(DIM_X);

            if(handle->pointer_mode == rocblas_pointer_mode_device)
            {
                ROCBLAS_LAUNCH_KERNEL((rocblas_sger_kernel<DIM_X, T>), ger_KARGS(alpha));
            }
            else
            {
                ROCBLAS_LAUNCH_KERNEL((rocblas_sger_kernel<DIM_X, T>), ger_KARGS(*alpha));
            }
        }
    }
    else
    {
        static constexpr int DIM_X   = 32;
        static constexpr int DIM_Y   = 32;
        static constexpr int WIN     = 2; // work item number of elements to process
        rocblas_int          blocksX = (m - 1) / DIM_X + 1;
        rocblas_int          blocksY = (n - 1) / (DIM_Y * WIN) + 1; // WIN columns/work item
        blocksX *= blocksY;

        dim3 ger_grid(blocksX, 1, batches);
        dim3 ger_threads(DIM_X, DIM_Y);

        if(handle->pointer_mode == rocblas_pointer_mode_device)
        {
            ROCBLAS_LAUNCH_KERNEL((rocblas_ger_kernel<DIM_X, DIM_Y, WIN, CONJ, T>),
                                  ger_KARGS(alpha));
        }
        else
        {
            ROCBLAS_LAUNCH_KERNEL((rocblas_ger_kernel<DIM_X, DIM_Y, WIN, CONJ, T>),
                                  ger_KARGS(*alpha));
        }
    }
#undef ger_KARGS
    return rocblas_status_success;
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_ger_template(rocblas_handle handle,
                                  rocblas_int    m,
                                  rocblas_int    n,
                                  const T*       alpha,
                                  rocblas_stride stride_alpha,
                                  const T*       x,
                                  rocblas_stride offsetx,
                                  rocblas_int    incx,
                                  rocblas_stride stridex,
                                  const T*       y,
                                  rocblas_stride offsety,
                                  rocblas_int    incy,
                                  rocblas_stride stridey,
                                  T*             A,
                                  rocblas_stride offsetA,
                                  rocblas_int    lda,
                                  rocblas_stride strideA,
                                  rocblas_int    batch_count)
{
    return rocblas_internal_ger_launcher<false, T>(handle,
                                                   m,
                                                   n,
                                                   alpha,
                                                   stride_alpha,
                                                   x,
                                                   offsetx,
                                                   incx,
                                                   stridex,
                                                   y,
                                                   offsety,
                                                   incy,
                                                   stridey,
                                                   A,
                                                   offsetA,
                                                   lda,
                                                   strideA,
                                                   batch_count);
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_gerc_template(rocblas_handle handle,
                                   rocblas_int    m,
                                   rocblas_int    n,
                                   const T*       alpha,
                                   rocblas_stride stride_alpha,
                                   const T*       x,
                                   rocblas_stride offsetx,
                                   rocblas_int    incx,
                                   rocblas_stride stridex,
                                   const T*       y,
                                   rocblas_stride offsety,
                                   rocblas_int    incy,
                                   rocblas_stride stridey,
                                   T*             A,
                                   rocblas_stride offsetA,
                                   rocblas_int    lda,
                                   rocblas_stride strideA,
                                   rocblas_int    batch_count)
{
    return rocblas_internal_ger_launcher<true, T>(handle,
                                                  m,
                                                  n,
                                                  alpha,
                                                  stride_alpha,
                                                  x,
                                                  offsetx,
                                                  incx,
                                                  stridex,
                                                  y,
                                                  offsety,
                                                  incy,
                                                  stridey,
                                                  A,
                                                  offsetA,
                                                  lda,
                                                  strideA,
                                                  batch_count);
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_ger_batched_template(rocblas_handle  handle,
                                          rocblas_int     m,
                                          rocblas_int     n,
                                          const T*        alpha,
                                          rocblas_stride  stride_alpha,
                                          const T* const* x,
                                          rocblas_stride  offsetx,
                                          rocblas_int     incx,
                                          rocblas_stride  stridex,
                                          const T* const* y,
                                          rocblas_stride  offsety,
                                          rocblas_int     incy,
                                          rocblas_stride  stridey,
                                          T* const*       A,
                                          rocblas_stride  offsetA,
                                          rocblas_int     lda,
                                          rocblas_stride  strideA,
                                          rocblas_int     batch_count)
{
    return rocblas_internal_ger_launcher<false, T>(handle,
                                                   m,
                                                   n,
                                                   alpha,
                                                   stride_alpha,
                                                   x,
                                                   offsetx,
                                                   incx,
                                                   stridex,
                                                   y,
                                                   offsety,
                                                   incy,
                                                   stridey,
                                                   A,
                                                   offsetA,
                                                   lda,
                                                   strideA,
                                                   batch_count);
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_gerc_batched_template(rocblas_handle  handle,
                                           rocblas_int     m,
                                           rocblas_int     n,
                                           const T*        alpha,
                                           rocblas_stride  stride_alpha,
                                           const T* const* x,
                                           rocblas_stride  offsetx,
                                           rocblas_int     incx,
                                           rocblas_stride  stridex,
                                           const T* const* y,
                                           rocblas_stride  offsety,
                                           rocblas_int     incy,
                                           rocblas_stride  stridey,
                                           T* const*       A,
                                           rocblas_stride  offsetA,
                                           rocblas_int     lda,
                                           rocblas_stride  strideA,
                                           rocblas_int     batch_count)
{
    return rocblas_internal_ger_launcher<true, T>(handle,
                                                  m,
                                                  n,
                                                  alpha,
                                                  stride_alpha,
                                                  x,
                                                  offsetx,
                                                  incx,
                                                  stridex,
                                                  y,
                                                  offsety,
                                                  incy,
                                                  stridey,
                                                  A,
                                                  offsetA,
                                                  lda,
                                                  strideA,
                                                  batch_count);
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *ger*.cpp

#ifdef INSTANTIATE_GER_TEMPLATE
#error INSTANTIATE_GER_TEMPLATE already defined
#endif

#define INSTANTIATE_GER_TEMPLATE(API_INT_, T_)                                                  \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status rocblas_internal_ger_template<T_>( \
        rocblas_handle handle,                                                                  \
        API_INT_       m,                                                                       \
        API_INT_       n,                                                                       \
        const T_*      alpha,                                                                   \
        rocblas_stride stride_alpha,                                                            \
        const T_*      x,                                                                       \
        rocblas_stride offsetx,                                                                 \
        API_INT_       incx,                                                                    \
        rocblas_stride stridex,                                                                 \
        const T_*      y,                                                                       \
        rocblas_stride offsety,                                                                 \
        API_INT_       incy,                                                                    \
        rocblas_stride stridey,                                                                 \
        T_*            A,                                                                       \
        rocblas_stride offsetA,                                                                 \
        API_INT_       lda,                                                                     \
        rocblas_stride strideA,                                                                 \
        API_INT_       batch_count);

#ifdef INSTANTIATE_GERC_TEMPLATE
#error INSTANTIATE_GERC_TEMPLATE already defined
#endif

#define INSTANTIATE_GERC_TEMPLATE(API_INT_, T_)                                                  \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status rocblas_internal_gerc_template<T_>( \
        rocblas_handle handle,                                                                   \
        API_INT_       m,                                                                        \
        API_INT_       n,                                                                        \
        const T_*      alpha,                                                                    \
        rocblas_stride stride_alpha,                                                             \
        const T_*      x,                                                                        \
        rocblas_stride offsetx,                                                                  \
        API_INT_       incx,                                                                     \
        rocblas_stride stridex,                                                                  \
        const T_*      y,                                                                        \
        rocblas_stride offsety,                                                                  \
        API_INT_       incy,                                                                     \
        rocblas_stride stridey,                                                                  \
        T_*            A,                                                                        \
        rocblas_stride offsetA,                                                                  \
        API_INT_       lda,                                                                      \
        rocblas_stride strideA,                                                                  \
        API_INT_       batch_count);

#ifdef INSTANTIATE_GER_BATCHED_TEMPLATE
#error INSTANTIATE_GER_BATCHED_TEMPLATE already defined
#endif

#define INSTANTIATE_GER_BATCHED_TEMPLATE(API_INT_, T_)                           \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                     \
        rocblas_internal_ger_batched_template<T_>(rocblas_handle   handle,       \
                                                  API_INT_         m,            \
                                                  API_INT_         n,            \
                                                  const T_*        alpha,        \
                                                  rocblas_stride   stride_alpha, \
                                                  const T_* const* x,            \
                                                  rocblas_stride   offsetx,      \
                                                  API_INT_         incx,         \
                                                  rocblas_stride   stridex,      \
                                                  const T_* const* y,            \
                                                  rocblas_stride   offsety,      \
                                                  API_INT_         incy,         \
                                                  rocblas_stride   stridey,      \
                                                  T_* const*       A,            \
                                                  rocblas_stride   offsetA,      \
                                                  API_INT_         lda,          \
                                                  rocblas_stride   strideA,      \
                                                  API_INT_         batch_count);

#ifdef INSTANTIATE_GERC_BATCHED_TEMPLATE
#error INSTANTIATE_GERC_BATCHED_TEMPLATE already defined
#endif

#define INSTANTIATE_GERC_BATCHED_TEMPLATE(API_INT_, T_)                           \
    template ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status                      \
        rocblas_internal_gerc_batched_template<T_>(rocblas_handle   handle,       \
                                                   API_INT_         m,            \
                                                   API_INT_         n,            \
                                                   const T_*        alpha,        \
                                                   rocblas_stride   stride_alpha, \
                                                   const T_* const* x,            \
                                                   rocblas_stride   offsetx,      \
                                                   API_INT_         incx,         \
                                                   rocblas_stride   stridex,      \
                                                   const T_* const* y,            \
                                                   rocblas_stride   offsety,      \
                                                   API_INT_         incy,         \
                                                   rocblas_stride   stridey,      \
                                                   T_* const*       A,            \
                                                   rocblas_stride   offsetA,      \
                                                   API_INT_         lda,          \
                                                   rocblas_stride   strideA,      \
                                                   API_INT_         batch_count);
