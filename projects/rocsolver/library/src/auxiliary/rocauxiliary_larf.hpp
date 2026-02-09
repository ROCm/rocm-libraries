/*****************************************************************************
 * Derived from the BSD3-licensed
 * LAPACK routine (version 3.7.0) --
 *     Univ. of Tennessee, Univ. of California Berkeley,
 *     Univ. of Colorado Denver and NAG Ltd..
 *     December 2016
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"
#include "rocsolver_run_specialized_kernels.hpp"
#include <hip/hip_cooperative_groups.h>

ROCSOLVER_BEGIN_NAMESPACE

/*
*   LARF kernel for the left side case. Each work group of NB_X threads
*   operates on a column of matrix A. (m + NB_X / warpSize) * sizeof(T)
*   bytes of LDS memory is required. Grid dimensions = dim3(1, n, batch count)
*   and block dimensions = dim3(NB_X).
*/
template <int NB_X, typename T, typename I, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(NB_X) larf_left_kernel(const I m,
                                                               const I n,
                                                               U xx,
                                                               const rocblas_stride shiftX,
                                                               const I incX,
                                                               const rocblas_stride strideX,
                                                               const T* tauA,
                                                               const rocblas_stride strideP,
                                                               U AA,
                                                               const rocblas_stride shiftA,
                                                               const I lda,
                                                               const rocblas_stride strideA)
{
    I bid = blockIdx.z;
    I tx = threadIdx.x;
    I col = blockIdx.y;

    // select batch instance
    T* x = load_ptr_batch<T>(xx, bid, shiftX, strideX);
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    const T* tau = tauA + bid * strideP;

    A += col * size_t(lda);

    I start = (incX > 0 ? 0 : (m - 1) * -incX);

    T res = 0;

    extern __shared__ double smem[];
    T* sdata = reinterpret_cast<T*>(smem);
    T* xs = sdata + (NB_X / warpSize);

    for(I i = tx; i < m; i += NB_X)
        xs[i] = x[start + i * size_t(incX)];

    //
    // GEMV
    //
    for(I i = tx; i < m; i += NB_X)
        res += conj(A[i]) * xs[i];

    // reduction
    res += shift_left(res, 1);
    res += shift_left(res, 2);
    res += shift_left(res, 4);
    res += shift_left(res, 8);
    res += shift_left(res, 16);
    if(warpSize > 32)
        res += shift_left(res, 32);
    if(tx % warpSize == 0)
        sdata[tx / warpSize] = res;
    __syncthreads();
    if(tx == 0)
    {
        for(I k = 1; k < NB_X / warpSize; k++)
            res += sdata[k];

        sdata[0] = res;
    }
    __syncthreads();

    //
    // GER
    //
    res = -tau[0] * conj(sdata[0]);
    for(I i = tx; i < m; i += NB_X)
        A[i] += res * xs[i];
}

/*
*   LARF kernel for the right side case. Each work group of NB_X threads
*   operates on a row of matrix A. (n + NB_X / warpSize) * sizeof(T)
*   bytes of LDS memory is required. Grid dimensions = dim3(1, m, batch count)
*   and block dimensions = dim3(NB_X).
*/
template <int NB_X, typename T, typename I, typename U>
ROCSOLVER_KERNEL void __launch_bounds__(NB_X) larf_right_kernel(const I m,
                                                                const I n,
                                                                U xx,
                                                                const rocblas_stride shiftX,
                                                                const I incX,
                                                                const rocblas_stride strideX,
                                                                const T* tauA,
                                                                const rocblas_stride strideP,
                                                                U AA,
                                                                const rocblas_stride shiftA,
                                                                const I lda,
                                                                const rocblas_stride strideA)
{
    I bid = blockIdx.z;
    I tx = threadIdx.x;
    I row = blockIdx.y;

    // select batch instance
    T* x = load_ptr_batch<T>(xx, bid, shiftX, strideX);
    T* A = load_ptr_batch<T>(AA, bid, shiftA, strideA);
    const T* tau = tauA + bid * strideP;

    A += row;

    I start = (incX > 0 ? 0 : (n - 1) * -incX);

    T res = 0;

    extern __shared__ double smem[];
    T* sdata = reinterpret_cast<T*>(smem);
    T* xs = sdata + (NB_X / warpSize);

    for(I j = tx; j < n; j += NB_X)
        xs[j] = x[start + j * size_t(incX)];

    //
    // GEMV
    //
    for(I j = tx; j < n; j += NB_X)
        res += A[j * size_t(lda)] * xs[j];

    // reduction
    res += shift_left(res, 1);
    res += shift_left(res, 2);
    res += shift_left(res, 4);
    res += shift_left(res, 8);
    res += shift_left(res, 16);
    if(warpSize > 32)
        res += shift_left(res, 32);
    if(tx % warpSize == 0)
        sdata[tx / warpSize] = res;
    __syncthreads();
    if(tx == 0)
    {
        for(I k = 1; k < NB_X / warpSize; k++)
            res += sdata[k];

        sdata[0] = res;
    }
    __syncthreads();

    //
    // GER
    //
    res = -tau[0] * sdata[0];
    for(I j = tx; j < n; j += NB_X)
        A[j * size_t(lda)] += res * conj(xs[j]);
}

template <bool BATCHED, typename T, typename I>
void rocsolver_larf_getMemorySize(const rocblas_side side,
                                  const I m,
                                  const I n,
                                  const I batch_count,
                                  size_t* size_scalars,
                                  size_t* size_Abyx,
                                  size_t* size_workArr)
{
    // if quick return no workspace needed
    if(n == 0 || m == 0 || !batch_count)
    {
        *size_scalars = 0;
        *size_Abyx = 0;
        *size_workArr = 0;
        return;
    }

    // size of scalars (constants)
    *size_scalars = sizeof(T) * 3;

    // size of array of pointers to workspace
    if(BATCHED)
        *size_workArr = sizeof(T*) * batch_count;
    else
        *size_workArr = 0;

    // if small size no workspace needed
    bool ssker_left
        = (side == rocblas_side_left && m <= LARF_SSKER_MAX_DIM && n <= LARF_SSKER_MIN_DIM);
    bool ssker_right
        = (side == rocblas_side_right && m <= LARF_SSKER_MIN_DIM && n <= LARF_SSKER_MAX_DIM);
    if(ssker_left || ssker_right)
    {
        *size_Abyx = 0;
        return;
    }

    // size of temporary result in Householder matrix generation
    if(side == rocblas_side_left)
        *size_Abyx = n;
    else if(side == rocblas_side_right)
        *size_Abyx = m;
    else
        *size_Abyx = std::max(m, n);
    *size_Abyx *= sizeof(T) * batch_count;
}

template <typename T, typename I, typename U>
rocblas_status rocsolver_larf_argCheck(rocblas_handle handle,
                                       const rocblas_side side,
                                       const I m,
                                       const I n,
                                       const I lda,
                                       const I incx,
                                       T x,
                                       T A,
                                       U alpha)
{
    // order is important for unit tests:

    // 1. invalid/non-supported values
    if(side != rocblas_side_left && side != rocblas_side_right)
        return rocblas_status_invalid_value;
    bool left = (side == rocblas_side_left);

    // 2. invalid size
    if(n < 0 || m < 0 || lda < m || !incx)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // 3. invalid pointers
    if((m && n && !A) || (left && m && (!alpha || !x)) || (!left && n && (!alpha || !x)))
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

template <typename T, typename I, typename U, bool COMPLEX = rocblas_is_complex<T>>
rocblas_status rocsolver_larf_template(rocblas_handle handle,
                                       const rocblas_side side,
                                       const I m,
                                       const I n,
                                       U x,
                                       const rocblas_stride shiftx,
                                       const I incx,
                                       const rocblas_stride stridex,
                                       const T* alpha,
                                       const rocblas_stride stridep,
                                       U A,
                                       const rocblas_stride shiftA,
                                       const I lda,
                                       const rocblas_stride stridea,
                                       const I batch_count,
                                       T* scalars,
                                       T* Abyx,
                                       T** workArr)
{
    ROCSOLVER_ENTER("larf", "side:", side, "m:", m, "n:", n, "shiftX:", shiftx, "incx:", incx,
                    "shiftA:", shiftA, "lda:", lda, "bc:", batch_count);

    // quick return
    if(n == 0 || m == 0 || !batch_count)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // if n is small, use small-size kernel
    bool ssker_left
        = (side == rocblas_side_left && m <= LARF_SSKER_MAX_DIM && n <= LARF_SSKER_MIN_DIM);
    bool ssker_right
        = (side == rocblas_side_right && m <= LARF_SSKER_MIN_DIM && n <= LARF_SSKER_MAX_DIM);
    if(ssker_left || ssker_right)
    {
        return larf_run_small(handle, side, m, n, x, shiftx, incx, stridex, alpha, stridep, A,
                              shiftA, lda, stridea, batch_count);
    }

    // get device prop
    const hipDeviceProp_t* props = rocblas_internal_get_device_prop(handle);

    // determine side
    bool leftside = (side == rocblas_side_left);

    static constexpr int NB = 1024;
    const int lds_size = leftside ? (m + (NB / props->warpSize)) * sizeof(T)
                                  : (n + (NB / props->warpSize)) * sizeof(T);

    if(lds_size <= props->sharedMemPerBlock)
    {
        // Launch larf kernel if tune parameters are met.
        if(leftside && (n <= 1024 || m >= 2048))
        {
            ROCSOLVER_LAUNCH_KERNEL((larf_left_kernel<NB>), dim3(1, n, batch_count), dim3(NB),
                                    lds_size, stream, m, n, x, shiftx, incx, stridex, alpha,
                                    stridep, A, shiftA, lda, stridea);
            return rocblas_status_success;
        }
        // TODO: investigate right side tuning.
        else if(!leftside && (m <= 1024 || n >= 2048))
        {
            ROCSOLVER_LAUNCH_KERNEL((larf_right_kernel<NB>), dim3(1, m, batch_count), dim3(NB),
                                    lds_size, stream, m, n, x, shiftx, incx, stridex, alpha,
                                    stridep, A, shiftA, lda, stridea);
            return rocblas_status_success;
        }
    }

    // everything must be executed with scalars on the device
    rocblas_pointer_mode old_mode;
    rocblas_get_pointer_mode(handle, &old_mode);
    rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device);

    // determine order of H
    I order = m;
    rocblas_operation trans = rocblas_operation_none;
    if(leftside)
    {
        trans = COMPLEX ? rocblas_operation_conjugate_transpose : rocblas_operation_transpose;
        order = n;
    }

    // **** FOR NOW, IT DOES NOT DETERMINE "NON-ZERO" DIMENSIONS
    //      OF A AND X, AS THIS WOULD REQUIRE SYNCHRONIZATION WITH GPU.
    //      IT WILL WORK ON THE ENTIRE MATRIX/VECTOR REGARDLESS OF
    //      ZERO ENTRIES ****

    // compute the matrix vector product  (W=-A'*X or W=-A*X)
    rocblasCall_gemv<T>(handle, trans, m, n, cast2constType<T>(scalars), 0, A, shiftA, lda, stridea,
                        x, shiftx, incx, stridex, cast2constType<T>(scalars + 1), 0, Abyx, 0, 1,
                        order, batch_count, workArr);

    // compute the rank-1 update  (A + tau*X*W'  or A + tau*W*X')
    if(leftside)
    {
        rocblasCall_ger<COMPLEX, T, I>(handle, m, n, alpha, stridep, x, shiftx, incx, stridex, Abyx,
                                       0, 1, order, A, shiftA, lda, stridea, batch_count, workArr);
    }
    else
    {
        rocblasCall_ger<COMPLEX, T, I>(handle, m, n, alpha, stridep, Abyx, 0, 1, order, x, shiftx,
                                       incx, stridex, A, shiftA, lda, stridea, batch_count, workArr);
    }

    rocblas_set_pointer_mode(handle, old_mode);
    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
