/* ************************************************************************
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "check_numerics_vector.hpp"
#include "device_macros.hpp"
#include "handle.hpp"
#include "rocblas_sbmv.hpp"

/**
  *  create partial sums for each ty.
  */
template <bool UPPER, rocblas_int DIM_Y, typename T>
inline __device__ T rocblas_sbmv_kernel_helper(rocblas_int ty,
                                               rocblas_int ind,
                                               rocblas_int n,
                                               rocblas_int k,
                                               const T* __restrict__ A,
                                               int64_t lda,
                                               const T* __restrict__ x,
                                               int64_t incx)
{
    T           res_A = 0.0;
    rocblas_int col;

    // Since the column is consistent, we iterate up diagonally in banded format
    // ty defines the column of banded matrix
    for(col = ty; col < n; col += DIM_Y)
    {
        // We have to convert ind to banded matrix row
        rocblas_int row = UPPER ? ind + (k - col) : ind - col;

        if(ind < n)
        {
            if((ind <= col && UPPER) || (ind >= col && !UPPER))
            {
                // in upper/lower triangular part
                if(row <= k && row >= 0)
                {
                    res_A += A[row + col * size_t(lda)] * x[col * incx];
                }
            }
            else
            {
                // in the opposite triangle, get value at transposed position
                rocblas_int trans_row = col;
                rocblas_int trans_col = ind;
                trans_row             = UPPER ? trans_row + (k - trans_col) : trans_row - trans_col;
                if(trans_row <= k && trans_row >= 0)
                {
                    res_A += A[trans_row + trans_col * size_t(lda)] * x[col * incx];
                }
            }
        }
    }
    return res_A;
}

/**
  *  Computes y := alpha*A*x + beta*y where A is a symmetric matrix.
  *  If uplo == upper, the strictly lower part of A is not referenced,
  *  if uplo == lower, the strictly upper part of A is not referenced.
  */
template <bool UPPER, rocblas_int DIM_X, rocblas_int DIM_Y, typename T>
inline __device__ void rocblas_sbmv_kernel_calc(rocblas_int n,
                                                rocblas_int k,
                                                T           alpha,
                                                const T* __restrict__ A,
                                                int64_t lda,
                                                const T* __restrict__ x,
                                                int64_t incx,
                                                T       beta,
                                                T* __restrict__ y,
                                                int64_t incy)
{
    rocblas_int thread_id = threadIdx.x + threadIdx.y * blockDim.x;

    if(!alpha)
    {
        rocblas_int ind = blockIdx.x * DIM_X + thread_id;
        if(thread_id < DIM_X && ind < n)
        {
            y[ind * incy] = beta ? (beta * y[ind * incy]) : 0;
        }
        return;
    }

    // threads are all configurated locally
    rocblas_int tx = thread_id % DIM_X;
    rocblas_int ty = thread_id / DIM_X;

    rocblas_int ind = blockIdx.x * DIM_X + tx;

    __shared__ T sdata[DIM_X * DIM_Y];

    T res_A;

    res_A = rocblas_sbmv_kernel_helper<UPPER, DIM_Y>(ty, ind, n, k, A, lda, x, incx);

    sdata[tx + ty * DIM_X] = res_A;

    __syncthreads();

    ind = blockIdx.x * DIM_X + thread_id;
    if(thread_id < DIM_X && ind < n)
    {
        for(rocblas_int i = 1; i < DIM_Y; i++)
            sdata[thread_id] += sdata[thread_id + DIM_X * i];

        y[ind * incy]
            = beta ? (alpha * sdata[thread_id]) + (beta * y[ind * incy]) : alpha * sdata[thread_id];
    }
}

/**
  *  U is either: const T* OR T
  *  V is either: const T* OR const T* const*
  *  W is either:       T* OR       T* const*
  */
template <bool UPPER, rocblas_int DIM_X, rocblas_int DIM_Y, typename U, typename V, typename W>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_sbmv_kernel(rocblas_int    n,
                    rocblas_int    k,
                    U              alpha_device_host,
                    rocblas_stride stride_alpha,
                    V              Aa,
                    rocblas_stride shifta,
                    int64_t        lda,
                    rocblas_stride stride_A,
                    V              xa,
                    rocblas_stride shiftx,
                    int64_t        incx,
                    rocblas_stride stride_x,
                    U              beta_device_host,
                    rocblas_stride stride_beta,
                    W              ya,
                    rocblas_stride shifty,
                    int64_t        incy,
                    rocblas_stride stride_y,
                    rocblas_int    batch_count)
{
    rocblas_int num_threads = blockDim.x * blockDim.y * blockDim.z;
    if(DIM_X * DIM_Y != num_threads)
        return; // need to launch exactly the same number of threads as template parameters indicate

    uint32_t batch = blockIdx.z;

#if DEVICE_GRID_YZ_16BIT
    for(; batch < batch_count; batch += c_YZ_grid_launch_limit)
    {
#endif

        auto alpha = load_scalar(alpha_device_host, batch, stride_alpha);
        auto beta  = load_scalar(beta_device_host, batch, stride_beta);
        if(!alpha && beta == 1)
        {
#if DEVICE_GRID_YZ_16BIT
            continue; //iterate to the next batch in the for loop rather than return.
#else
        return;
#endif
        }

        const auto* A = cond_load_ptr_batch(alpha, Aa, batch, shifta, stride_A);
        const auto* x = cond_load_ptr_batch(alpha, xa, batch, shiftx, stride_x);

        auto* y = load_ptr_batch(ya, batch, shifty, stride_y);

        rocblas_sbmv_kernel_calc<UPPER, DIM_X, DIM_Y>(n, k, alpha, A, lda, x, incx, beta, y, incy);

#if DEVICE_GRID_YZ_16BIT
    }
#endif
}

template <typename T, typename TScal, typename TConstPtr, typename TPtr>
rocblas_status rocblas_internal_sbmv_launcher(rocblas_handle handle,
                                              rocblas_fill   uplo,
                                              rocblas_int    n,
                                              rocblas_int    k,
                                              TScal          alpha,
                                              rocblas_stride stride_alpha,
                                              TConstPtr      A,
                                              rocblas_stride offset_A,
                                              int64_t        lda,
                                              rocblas_stride stride_A,
                                              TConstPtr      x,
                                              rocblas_stride offset_x,
                                              int64_t        incx,
                                              rocblas_stride stride_x,
                                              TScal          beta,
                                              rocblas_stride stride_beta,
                                              TPtr           y,
                                              rocblas_stride offset_y,
                                              int64_t        incy,
                                              rocblas_stride stride_y,
                                              rocblas_int    batch_count)
{
    //quick return
    if(!n || !batch_count)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->get_stream();

    // in case of negative inc shift pointer to end of data for negative indexing tid*inc
    auto shiftx = incx < 0 ? offset_x - incx * (n - 1) : offset_x;
    auto shifty = incy < 0 ? offset_y - incy * (n - 1) : offset_y;

    int batches = handle->getBatchGridDim((int)batch_count);

    static constexpr int sbmv_DIM_X = 64;
    static constexpr int sbmv_DIM_Y = 16;
    rocblas_int          blocks     = (n - 1) / (sbmv_DIM_X) + 1;
    dim3                 grid(blocks, 1, batches);
    dim3                 threads(sbmv_DIM_X, sbmv_DIM_Y);

    if(handle->pointer_mode == rocblas_pointer_mode_device)
    {
        if(uplo == rocblas_fill_upper)
        {
            ROCBLAS_LAUNCH_KERNEL((rocblas_sbmv_kernel<true, sbmv_DIM_X, sbmv_DIM_Y>),
                                  grid,
                                  threads,
                                  0,
                                  rocblas_stream,
                                  n,
                                  k,
                                  alpha,
                                  stride_alpha,
                                  A,
                                  offset_A,
                                  lda,
                                  stride_A,
                                  x,
                                  shiftx,
                                  incx,
                                  stride_x,
                                  beta,
                                  stride_beta,
                                  y,
                                  shifty,
                                  incy,
                                  stride_y,
                                  batch_count);
        }
        else
        {
            ROCBLAS_LAUNCH_KERNEL((rocblas_sbmv_kernel<false, sbmv_DIM_X, sbmv_DIM_Y>),
                                  grid,
                                  threads,
                                  0,
                                  rocblas_stream,
                                  n,
                                  k,
                                  alpha,
                                  stride_alpha,
                                  A,
                                  offset_A,
                                  lda,
                                  stride_A,
                                  x,
                                  shiftx,
                                  incx,
                                  stride_x,
                                  beta,
                                  stride_beta,
                                  y,
                                  shifty,
                                  incy,
                                  stride_y,
                                  batch_count);
        }
    }
    else
    {
        // quick return only for non-batched
        if(batch_count == 1 && !*alpha && *beta == 1)
            return rocblas_status_success;

        if(uplo == rocblas_fill_upper)
        {
            ROCBLAS_LAUNCH_KERNEL((rocblas_sbmv_kernel<true, sbmv_DIM_X, sbmv_DIM_Y>),
                                  grid,
                                  threads,
                                  0,
                                  rocblas_stream,
                                  n,
                                  k,
                                  *alpha,
                                  stride_alpha,
                                  A,
                                  offset_A,
                                  lda,
                                  stride_A,
                                  x,
                                  shiftx,
                                  incx,
                                  stride_x,
                                  *beta,
                                  stride_beta,
                                  y,
                                  shifty,
                                  incy,
                                  stride_y,
                                  batch_count);
        }
        else
        {
            ROCBLAS_LAUNCH_KERNEL((rocblas_sbmv_kernel<false, sbmv_DIM_X, sbmv_DIM_Y>),
                                  grid,
                                  threads,
                                  0,
                                  rocblas_stream,
                                  n,
                                  k,
                                  *alpha,
                                  stride_alpha,
                                  A,
                                  offset_A,
                                  lda,
                                  stride_A,
                                  x,
                                  shiftx,
                                  incx,
                                  stride_x,
                                  *beta,
                                  stride_beta,
                                  y,
                                  shifty,
                                  incy,
                                  stride_y,
                                  batch_count);
        }
    }

    return rocblas_status_success;
}

//TODO :-Add rocblas_check_numerics_sb_matrix_template for checking Matrix `A` which is a Symmetric Band Matrix
template <typename T, typename U>
rocblas_status rocblas_sbmv_check_numerics(const char*    function_name,
                                           rocblas_handle handle,
                                           int64_t        n,
                                           T              A,
                                           rocblas_stride offset_a,
                                           int64_t        lda,
                                           rocblas_stride stride_a,
                                           T              x,
                                           rocblas_stride offset_x,
                                           int64_t        inc_x,
                                           rocblas_stride stride_x,
                                           U              y,
                                           rocblas_stride offset_y,
                                           int64_t        inc_y,
                                           rocblas_stride stride_y,
                                           int64_t        batch_count,
                                           const int      check_numerics,
                                           bool           is_input)
{
    rocblas_status check_numerics_status
        = rocblas_internal_check_numerics_vector_template(function_name,
                                                          handle,
                                                          n,
                                                          x,
                                                          offset_x,
                                                          inc_x,
                                                          stride_x,
                                                          batch_count,
                                                          check_numerics,
                                                          is_input);
    if(check_numerics_status != rocblas_status_success)
        return check_numerics_status;

    check_numerics_status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                            handle,
                                                                            n,
                                                                            y,
                                                                            offset_y,
                                                                            inc_y,
                                                                            stride_y,
                                                                            batch_count,
                                                                            check_numerics,
                                                                            is_input);

    return check_numerics_status;
}

// Instantiations below will need to be manually updated to match any change in
// template parameters in the files *sbmv*.cpp

// clang-format off

#ifdef INSTANTIATE_SBMV_LAUNCHER
#error INSTANTIATE_SBMV_LAUNCHER already defined
#endif

#define INSTANTIATE_SBMV_LAUNCHER(T_, TScal_, TConstPtr_, TPtr_)                  \
template rocblas_status rocblas_internal_sbmv_launcher<T_, TScal_, TConstPtr_, TPtr_>      \
                                    (rocblas_handle handle,        \
                                     rocblas_fill   uplo,          \
                                     rocblas_int    n,             \
                                     rocblas_int    k,             \
                                     TScal_       alpha,        \
                                     rocblas_stride stride_alpha,  \
                                     TConstPtr_       A,            \
                                     rocblas_stride offset_A,       \
                                     int64_t    lda,           \
                                     rocblas_stride stride_A,       \
                                     TConstPtr_       x,            \
                                     rocblas_stride offset_x,       \
                                     int64_t    incx,          \
                                     rocblas_stride stride_x,       \
                                     TScal_       beta,         \
                                     rocblas_stride stride_beta,   \
                                     TPtr_             y,            \
                                     rocblas_stride offset_y,       \
                                     int64_t    incy,          \
                                     rocblas_stride stride_y,       \
                                     rocblas_int    batch_count);

INSTANTIATE_SBMV_LAUNCHER(float, float const*, float const*, float*)
INSTANTIATE_SBMV_LAUNCHER(double, double const*, double const*, double*)
INSTANTIATE_SBMV_LAUNCHER(float, float const*, float const* const*, float* const*)
INSTANTIATE_SBMV_LAUNCHER(double, double const*, double const* const*, double* const*)

#undef INSTANTIATE_SBMV_LAUNCHER

#ifdef INSTANTIATE_SBMV_NUMERICS
#error INSTANTIATE_SBMV_NUMERICS already defined
#endif

#define INSTANTIATE_SBMV_NUMERICS(T_, U_)                                 \
template rocblas_status rocblas_sbmv_check_numerics<T_, U_>               \
                                          (const char*    function_name,  \
                                           rocblas_handle handle,         \
                                           int64_t    n,              \
                                           T_              A,             \
                                           rocblas_stride    offset_a,       \
                                           int64_t    lda,            \
                                           rocblas_stride stride_a,       \
                                           T_              x,             \
                                           rocblas_stride    offset_x,       \
                                           int64_t    inc_x,          \
                                           rocblas_stride stride_x,       \
                                           U_              y,             \
                                           rocblas_stride    offset_y,       \
                                           int64_t    inc_y,          \
                                           rocblas_stride stride_y,       \
                                           int64_t    batch_count,    \
                                           const int      check_numerics, \
                                           bool           is_input);

INSTANTIATE_SBMV_NUMERICS(float const*, float*)
INSTANTIATE_SBMV_NUMERICS(double const*, double*)
INSTANTIATE_SBMV_NUMERICS(float const* const*, float* const*)
INSTANTIATE_SBMV_NUMERICS(double const* const*, double* const*)

#undef INSTANTIATE_SBMV_NUMERICS

// clang-format on
