/* **************************************************************************
 * Copyright (C) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lapack_host_functions.hpp"
#include "rocblas.hpp"
#include "rocsolver_run_specialized_kernels.hpp"

ROCSOLVER_BEGIN_NAMESPACE

/** Call this kernel with min(max_blocks,'batch_count') groups in z, and enough
    groups in x and y to cover all the 'm' rows and 'n' columns of C. **/
template <typename T, typename I, typename V, typename U1, typename U2, typename U3>
ROCSOLVER_KERNEL void ger_kernel(I const m,
                                 I const n,
                                 V alpha,
                                 rocblas_stride const stridea,
                                 U1 xx,
                                 rocblas_stride const shiftX,
                                 I const incx,
                                 rocblas_stride const strideX,
                                 U2 yy,
                                 rocblas_stride const shiftY,
                                 I const incy,
                                 rocblas_stride const strideY,
                                 U3 AA,
                                 rocblas_stride const shiftA,
                                 I const inca,
                                 I const lda,
                                 rocblas_stride const strideA,
                                 I const batch_count)
{
    // indices
    I bid = hipBlockIdx_z;
    I i = hipBlockIdx_x * static_cast<I>(hipBlockDim_x) + hipThreadIdx_x;
    I j = hipBlockIdx_y * static_cast<I>(hipBlockDim_y) + hipThreadIdx_y;

    I const bid_start = hipBlockIdx_z;
    I const bid_inc = hipGridDim_z;
    I const i_start = hipBlockIdx_x * static_cast<I>(hipBlockDim_x) + hipThreadIdx_x;
    I const i_inc = hipBlockDim_x * hipGridDim_x;
    I const j_start = hipBlockIdx_y * static_cast<I>(hipBlockDim_y) + hipThreadIdx_y;
    I const j_inc = hipBlockDim_y * hipGridDim_y;

    for(I bid = bid_start; bid < batch_count; bid += bid_inc)
    {
        // batch instance
        T const a = load_scalar(alpha, bid, stridea);
        T* const A = load_ptr_batch(AA, bid, shiftA, strideA);
        T* const x = load_ptr_batch(xx, bid, shiftX, strideX);
        T* const y = load_ptr_batch(yy, bid, shiftY, strideY);

        for(I j = j_start; j < n; j += j_inc)
        {
            for(I i = i_start; i < m; i += i_inc)
            {
                A[i * inca + j * lda] += a * x[i * incx] * y[j * incy];
            }
        }

    } // end for bid
}

/*************************************************************
    Launchers of specialized kernels
*************************************************************/

template <bool CONJ, typename T, typename I, typename U>
rocblas_status rocsolver_ger(rocblas_handle handle,
                             I m,
                             I n,
                             const T* alpha,
                             rocblas_stride stridea,
                             U x,
                             rocblas_stride shiftX,
                             I incx,
                             rocblas_stride strideX,
                             U y,
                             rocblas_stride shiftY,
                             I incy,
                             rocblas_stride strideY,
                             U A,
                             rocblas_stride shiftA,
                             I inca,
                             I lda,
                             rocblas_stride strideA,
                             I batch_count,
                             T** work)
{
    ROCSOLVER_ENTER("ger", "m:", m, "n:", n, "shiftX:", shiftX, "incx:", incx, "shiftY:", shiftY,
                    "incy:", incy, "shiftA:", shiftA, "inca:", inca, "lda:", lda, "bc:", batch_count);

    if(m == 0 || n == 0 || batch_count == 0)
        return rocblas_status_success;

    if(inca == 1)
        return rocblasCall_ger<CONJ, T>(handle, m, n, alpha, stridea, x, shiftX, incx, strideX, y,
                                        shiftY, incy, strideY, A, shiftA, lda, strideA, batch_count,
                                        work);

    // TODO: add interleaved support for conjugation
    if(CONJ)
        THROW_IF_ROCBLAS_ERROR(rocblas_status_not_implemented);

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    rocblas_pointer_mode pmode;
    rocblas_get_pointer_mode(handle, &pmode);

    // launch specialized kernel
    I max_blocks = get_nblocks_yz(handle);
    I blocksx = (m - 1) / BS2 + 1;
    I blocksy = (n - 1) / BS2 + 1;
    dim3 grid(blocksx, std::min(max_blocks, blocksy), std::min(max_blocks, batch_count));
    dim3 threads(BS2, BS2, 1);
    if(pmode == rocblas_pointer_mode_device)
    {
        ROCSOLVER_LAUNCH_KERNEL((ger_kernel<T>), grid, threads, 0, stream, m, n, alpha, stridea, x,
                                shiftX, incx, strideX, y, shiftY, incy, strideY, A, shiftA, inca,
                                lda, strideA, batch_count);
    }
    else
    {
        ROCSOLVER_LAUNCH_KERNEL((ger_kernel<T>), grid, threads, 0, stream, m, n, *alpha, stridea, x,
                                shiftX, incx, strideX, y, shiftY, incy, strideY, A, shiftA, inca,
                                lda, strideA, batch_count);
    }

    return rocblas_status_success;
}

/*************************************************************
    Non-interleaved wrappers
*************************************************************/

template <bool CONJ, typename T, typename I, typename U>
inline rocblas_status rocsolver_ger(rocblas_handle handle,
                                    I const m,
                                    I const n,
                                    const T* alpha,
                                    rocblas_stride const stridea,
                                    U x,
                                    rocblas_stride const shiftX,
                                    I const incx,
                                    rocblas_stride const strideX,
                                    U y,
                                    rocblas_stride const shiftY,
                                    I const incy,
                                    rocblas_stride const strideY,
                                    U A,
                                    rocblas_stride const shiftA,
                                    I const lda,
                                    rocblas_stride const strideA,
                                    I const batch_count,
                                    T** const work)
{
    return rocsolver_ger<CONJ, T, I>(handle, m, n, alpha, stridea, x, shiftX, incx, strideX, y,
                                     shiftY, incy, strideY, A, shiftA, 1, lda, strideA, batch_count,
                                     work);
}

/*************************************************************
    Instantiation macros
*************************************************************/

#define INSTANTIATE_GER(CONJ, T, I, U)                                                             \
    template rocblas_status rocsolver_ger<CONJ, T, I, U>(                                          \
        rocblas_handle handle, I m, I n, const T* alpha, rocblas_stride stridea, U x,              \
        rocblas_stride shiftX, I incx, rocblas_stride strideX, U y, rocblas_stride shiftY, I incy, \
        rocblas_stride strideY, U A, rocblas_stride shiftA, I lda, rocblas_stride strideA,         \
        I batch_count, T** work)

ROCSOLVER_END_NAMESPACE
