/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *
 * ************************************************************************/

#include "hipblas.h"
#include "hipblas.hpp"

#ifndef WIN32
#include "hipblas_fortran.hpp"
#else
#include "hipblas_no_fortran.hpp"
#endif

#include <typeinfo>

// This file's purpose is now only for casting hipblasComplex -> hipComplex when necessary.
// When we finish transitioning to hipComplex, this file can be deleted.

/*
 * ===========================================================================
 *    level 1 BLAS
 * ===========================================================================
 */

#ifdef HIPBLAS_V2
// axpy
hipblasStatus_t hipblasCaxpyCast(hipblasHandle_t       handle,
                                 int                   n,
                                 const hipblasComplex* alpha,
                                 const hipblasComplex* x,
                                 int                   incx,
                                 hipblasComplex*       y,
                                 int                   incy)
{
    return hipblasCaxpy(
        handle, n, (const hipComplex*)alpha, (const hipComplex*)x, incx, (hipComplex*)y, incy);
}

hipblasStatus_t hipblasZaxpyCast(hipblasHandle_t             handle,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 hipblasDoubleComplex*       y,
                                 int                         incy)
{
    return hipblasZaxpy(handle,
                        n,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)x,
                        incx,
                        (hipDoubleComplex*)y,
                        incy);
}

hipblasStatus_t hipblasCaxpyCast_64(hipblasHandle_t       handle,
                                    int64_t               n,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* x,
                                    int64_t               incx,
                                    hipblasComplex*       y,
                                    int64_t               incy)
{
    return hipblasCaxpy_64(
        handle, n, (const hipComplex*)alpha, (const hipComplex*)x, incx, (hipComplex*)y, incy);
}

hipblasStatus_t hipblasZaxpyCast_64(hipblasHandle_t             handle,
                                    int64_t                     n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* x,
                                    int64_t                     incx,
                                    hipblasDoubleComplex*       y,
                                    int64_t                     incy)
{
    return hipblasZaxpy_64(handle,
                           n,
                           (const hipDoubleComplex*)alpha,
                           (const hipDoubleComplex*)x,
                           incx,
                           (hipDoubleComplex*)y,
                           incy);
}

// axpy_batched
hipblasStatus_t hipblasCaxpyBatchedCast(hipblasHandle_t             handle,
                                        int                         n,
                                        const hipblasComplex*       alpha,
                                        const hipblasComplex* const x[],
                                        int                         incx,
                                        hipblasComplex* const       y[],
                                        int                         incy,
                                        int                         batch_count)
{
    return hipblasCaxpyBatched(handle,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)x,
                               incx,
                               (hipComplex* const*)y,
                               incy,
                               batch_count);
}

hipblasStatus_t hipblasZaxpyBatchedCast(hipblasHandle_t                   handle,
                                        int                               n,
                                        const hipblasDoubleComplex*       alpha,
                                        const hipblasDoubleComplex* const x[],
                                        int                               incx,
                                        hipblasDoubleComplex* const       y[],
                                        int                               incy,
                                        int                               batch_count)
{
    return hipblasZaxpyBatched(handle,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (hipDoubleComplex* const*)y,
                               incy,
                               batch_count);
}

hipblasStatus_t hipblasCaxpyBatchedCast_64(hipblasHandle_t             handle,
                                           int64_t                     n,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const x[],
                                           int64_t                     incx,
                                           hipblasComplex* const       y[],
                                           int64_t                     incy,
                                           int64_t                     batch_count)
{
    return hipblasCaxpyBatched_64(handle,
                                  n,
                                  (const hipComplex*)alpha,
                                  (const hipComplex* const*)x,
                                  incx,
                                  (hipComplex* const*)y,
                                  incy,
                                  batch_count);
}

hipblasStatus_t hipblasZaxpyBatchedCast_64(hipblasHandle_t                   handle,
                                           int64_t                           n,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const x[],
                                           int64_t                           incx,
                                           hipblasDoubleComplex* const       y[],
                                           int64_t                           incy,
                                           int64_t                           batch_count)
{
    return hipblasZaxpyBatched_64(handle,
                                  n,
                                  (const hipDoubleComplex*)alpha,
                                  (const hipDoubleComplex* const*)x,
                                  incx,
                                  (hipDoubleComplex* const*)y,
                                  incy,
                                  batch_count);
}

// axpy_strided_batched
hipblasStatus_t hipblasCaxpyStridedBatchedCast(hipblasHandle_t       handle,
                                               int                   n,
                                               const hipblasComplex* alpha,
                                               const hipblasComplex* x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               hipblasComplex*       y,
                                               int                   incy,
                                               hipblasStride         stridey,
                                               int                   batch_count)
{
    return hipblasCaxpyStridedBatched(handle,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)x,
                                      incx,
                                      stridex,
                                      (hipComplex*)y,
                                      incy,
                                      stridey,
                                      batch_count);
}

hipblasStatus_t hipblasZaxpyStridedBatchedCast(hipblasHandle_t             handle,
                                               int                         n,
                                               const hipblasDoubleComplex* alpha,
                                               const hipblasDoubleComplex* x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               hipblasDoubleComplex*       y,
                                               int                         incy,
                                               hipblasStride               stridey,
                                               int                         batch_count)
{
    return hipblasZaxpyStridedBatched(handle,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      (hipDoubleComplex*)y,
                                      incy,
                                      stridey,
                                      batch_count);
}

hipblasStatus_t hipblasCaxpyStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  int64_t               n,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* x,
                                                  int64_t               incx,
                                                  hipblasStride         stridex,
                                                  hipblasComplex*       y,
                                                  int64_t               incy,
                                                  hipblasStride         stridey,
                                                  int64_t               batch_count)
{
    return hipblasCaxpyStridedBatched_64(handle,
                                         n,
                                         (const hipComplex*)alpha,
                                         (const hipComplex*)x,
                                         incx,
                                         stridex,
                                         (hipComplex*)y,
                                         incy,
                                         stridey,
                                         batch_count);
}

hipblasStatus_t hipblasZaxpyStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  int64_t                     n,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* x,
                                                  int64_t                     incx,
                                                  hipblasStride               stridex,
                                                  hipblasDoubleComplex*       y,
                                                  int64_t                     incy,
                                                  hipblasStride               stridey,
                                                  int64_t                     batch_count)
{
    return hipblasZaxpyStridedBatched_64(handle,
                                         n,
                                         (const hipDoubleComplex*)alpha,
                                         (const hipDoubleComplex*)x,
                                         incx,
                                         stridex,
                                         (hipDoubleComplex*)y,
                                         incy,
                                         stridey,
                                         batch_count);
}

// swap
hipblasStatus_t hipblasCswapCast(
    hipblasHandle_t handle, int n, hipblasComplex* x, int incx, hipblasComplex* y, int incy)
{
    return hipblasCswap(handle, n, (hipComplex*)x, incx, (hipComplex*)y, incy);
}

hipblasStatus_t hipblasZswapCast(hipblasHandle_t       handle,
                                 int                   n,
                                 hipblasDoubleComplex* x,
                                 int                   incx,
                                 hipblasDoubleComplex* y,
                                 int                   incy)
{
    return hipblasZswap(handle, n, (hipDoubleComplex*)x, incx, (hipDoubleComplex*)y, incy);
}

// swap_64
hipblasStatus_t hipblasCswapCast_64(hipblasHandle_t handle,
                                    int64_t         n,
                                    hipblasComplex* x,
                                    int64_t         incx,
                                    hipblasComplex* y,
                                    int64_t         incy)
{
    return hipblasCswap_64(handle, n, (hipComplex*)x, incx, (hipComplex*)y, incy);
}

hipblasStatus_t hipblasZswapCast_64(hipblasHandle_t       handle,
                                    int64_t               n,
                                    hipblasDoubleComplex* x,
                                    int64_t               incx,
                                    hipblasDoubleComplex* y,
                                    int64_t               incy)
{
    return hipblasZswap_64(handle, n, (hipDoubleComplex*)x, incx, (hipDoubleComplex*)y, incy);
}

// swap_batched
hipblasStatus_t hipblasCswapBatchedCast(hipblasHandle_t       handle,
                                        int                   n,
                                        hipblasComplex* const x[],
                                        int                   incx,
                                        hipblasComplex* const y[],
                                        int                   incy,
                                        int                   batch_count)
{
    return hipblasCswapBatched(
        handle, n, (hipComplex* const*)x, incx, (hipComplex* const*)y, incy, batch_count);
}

hipblasStatus_t hipblasZswapBatchedCast(hipblasHandle_t             handle,
                                        int                         n,
                                        hipblasDoubleComplex* const x[],
                                        int                         incx,
                                        hipblasDoubleComplex* const y[],
                                        int                         incy,
                                        int                         batch_count)
{
    return hipblasZswapBatched(handle,
                               n,
                               (hipDoubleComplex* const*)x,
                               incx,
                               (hipDoubleComplex* const*)y,
                               incy,
                               batch_count);
}

// swap_batched_64
hipblasStatus_t hipblasCswapBatchedCast_64(hipblasHandle_t       handle,
                                           int64_t               n,
                                           hipblasComplex* const x[],
                                           int64_t               incx,
                                           hipblasComplex* const y[],
                                           int64_t               incy,
                                           int64_t               batch_count)
{
    return hipblasCswapBatched_64(
        handle, n, (hipComplex* const*)x, incx, (hipComplex* const*)y, incy, batch_count);
}

hipblasStatus_t hipblasZswapBatchedCast_64(hipblasHandle_t             handle,
                                           int64_t                     n,
                                           hipblasDoubleComplex* const x[],
                                           int64_t                     incx,
                                           hipblasDoubleComplex* const y[],
                                           int64_t                     incy,
                                           int64_t                     batch_count)
{
    return hipblasZswapBatched_64(handle,
                                  n,
                                  (hipDoubleComplex* const*)x,
                                  incx,
                                  (hipDoubleComplex* const*)y,
                                  incy,
                                  batch_count);
}

// swap_strided_batched
hipblasStatus_t hipblasCswapStridedBatchedCast(hipblasHandle_t handle,
                                               int             n,
                                               hipblasComplex* x,
                                               int             incx,
                                               hipblasStride   stridex,
                                               hipblasComplex* y,
                                               int             incy,
                                               hipblasStride   stridey,
                                               int             batch_count)
{
    return hipblasCswapStridedBatched(
        handle, n, (hipComplex*)x, incx, stridex, (hipComplex*)y, incy, stridey, batch_count);
}

hipblasStatus_t hipblasZswapStridedBatchedCast(hipblasHandle_t       handle,
                                               int                   n,
                                               hipblasDoubleComplex* x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               hipblasDoubleComplex* y,
                                               int                   incy,
                                               hipblasStride         stridey,
                                               int                   batch_count)
{
    return hipblasZswapStridedBatched(handle,
                                      n,
                                      (hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      (hipDoubleComplex*)y,
                                      incy,
                                      stridey,
                                      batch_count);
}

// swap_strided_batched_64
hipblasStatus_t hipblasCswapStridedBatchedCast_64(hipblasHandle_t handle,
                                                  int64_t         n,
                                                  hipblasComplex* x,
                                                  int64_t         incx,
                                                  hipblasStride   stridex,
                                                  hipblasComplex* y,
                                                  int64_t         incy,
                                                  hipblasStride   stridey,
                                                  int64_t         batch_count)
{
    return hipblasCswapStridedBatched_64(
        handle, n, (hipComplex*)x, incx, stridex, (hipComplex*)y, incy, stridey, batch_count);
}

hipblasStatus_t hipblasZswapStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  int64_t               n,
                                                  hipblasDoubleComplex* x,
                                                  int64_t               incx,
                                                  hipblasStride         stridex,
                                                  hipblasDoubleComplex* y,
                                                  int64_t               incy,
                                                  hipblasStride         stridey,
                                                  int64_t               batch_count)
{
    return hipblasZswapStridedBatched_64(handle,
                                         n,
                                         (hipDoubleComplex*)x,
                                         incx,
                                         stridex,
                                         (hipDoubleComplex*)y,
                                         incy,
                                         stridey,
                                         batch_count);
}

// copy
hipblasStatus_t hipblasCcopyCast(
    hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, hipblasComplex* y, int incy)
{
    return hipblasCcopy(handle, n, (const hipComplex*)x, incx, (hipComplex*)y, incy);
}

hipblasStatus_t hipblasZcopyCast(hipblasHandle_t             handle,
                                 int                         n,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 hipblasDoubleComplex*       y,
                                 int                         incy)
{
    return hipblasZcopy(handle, n, (const hipDoubleComplex*)x, incx, (hipDoubleComplex*)y, incy);
}

hipblasStatus_t hipblasCcopyCast_64(hipblasHandle_t       handle,
                                    int64_t               n,
                                    const hipblasComplex* x,
                                    int64_t               incx,
                                    hipblasComplex*       y,
                                    int64_t               incy)
{
    return hipblasCcopy_64(handle, n, (const hipComplex*)x, incx, (hipComplex*)y, incy);
}

hipblasStatus_t hipblasZcopyCast_64(hipblasHandle_t             handle,
                                    int64_t                     n,
                                    const hipblasDoubleComplex* x,
                                    int64_t                     incx,
                                    hipblasDoubleComplex*       y,
                                    int64_t                     incy)
{
    return hipblasZcopy_64(handle, n, (const hipDoubleComplex*)x, incx, (hipDoubleComplex*)y, incy);
}

// batched
hipblasStatus_t hipblasCcopyBatchedCast(hipblasHandle_t             handle,
                                        int                         n,
                                        const hipblasComplex* const x[],
                                        int                         incx,
                                        hipblasComplex* const       y[],
                                        int                         incy,
                                        int                         batch_count)
{
    return hipblasCcopyBatched(
        handle, n, (const hipComplex* const*)x, incx, (hipComplex* const*)y, incy, batch_count);
}

hipblasStatus_t hipblasZcopyBatchedCast(hipblasHandle_t                   handle,
                                        int                               n,
                                        const hipblasDoubleComplex* const x[],
                                        int                               incx,
                                        hipblasDoubleComplex* const       y[],
                                        int                               incy,
                                        int                               batch_count)
{
    return hipblasZcopyBatched(handle,
                               n,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (hipDoubleComplex* const*)y,
                               incy,
                               batch_count);
}

hipblasStatus_t hipblasCcopyBatchedCast_64(hipblasHandle_t             handle,
                                           int64_t                     n,
                                           const hipblasComplex* const x[],
                                           int64_t                     incx,
                                           hipblasComplex* const       y[],
                                           int64_t                     incy,
                                           int64_t                     batch_count)
{
    return hipblasCcopyBatched_64(
        handle, n, (const hipComplex* const*)x, incx, (hipComplex* const*)y, incy, batch_count);
}

hipblasStatus_t hipblasZcopyBatchedCast_64(hipblasHandle_t                   handle,
                                           int64_t                           n,
                                           const hipblasDoubleComplex* const x[],
                                           int64_t                           incx,
                                           hipblasDoubleComplex* const       y[],
                                           int64_t                           incy,
                                           int64_t                           batch_count)
{
    return hipblasZcopyBatched_64(handle,
                                  n,
                                  (const hipDoubleComplex* const*)x,
                                  incx,
                                  (hipDoubleComplex* const*)y,
                                  incy,
                                  batch_count);
}

// strided_batched
hipblasStatus_t hipblasCcopyStridedBatchedCast(hipblasHandle_t       handle,
                                               int                   n,
                                               const hipblasComplex* x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               hipblasComplex*       y,
                                               int                   incy,
                                               hipblasStride         stridey,
                                               int                   batch_count)
{
    return hipblasCcopyStridedBatched(
        handle, n, (const hipComplex*)x, incx, stridex, (hipComplex*)y, incy, stridey, batch_count);
}

hipblasStatus_t hipblasZcopyStridedBatchedCast(hipblasHandle_t             handle,
                                               int                         n,
                                               const hipblasDoubleComplex* x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               hipblasDoubleComplex*       y,
                                               int                         incy,
                                               hipblasStride               stridey,
                                               int                         batch_count)
{
    return hipblasZcopyStridedBatched(handle,
                                      n,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      (hipDoubleComplex*)y,
                                      incy,
                                      stridey,
                                      batch_count);
}

hipblasStatus_t hipblasCcopyStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  int64_t               n,
                                                  const hipblasComplex* x,
                                                  int64_t               incx,
                                                  hipblasStride         stridex,
                                                  hipblasComplex*       y,
                                                  int64_t               incy,
                                                  hipblasStride         stridey,
                                                  int64_t               batch_count)
{
    return hipblasCcopyStridedBatched_64(
        handle, n, (const hipComplex*)x, incx, stridex, (hipComplex*)y, incy, stridey, batch_count);
}

hipblasStatus_t hipblasZcopyStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  int64_t                     n,
                                                  const hipblasDoubleComplex* x,
                                                  int64_t                     incx,
                                                  hipblasStride               stridex,
                                                  hipblasDoubleComplex*       y,
                                                  int64_t                     incy,
                                                  hipblasStride               stridey,
                                                  int64_t                     batch_count)
{
    return hipblasZcopyStridedBatched_64(handle,
                                         n,
                                         (const hipDoubleComplex*)x,
                                         incx,
                                         stridex,
                                         (hipDoubleComplex*)y,
                                         incy,
                                         stridey,
                                         batch_count);
}

// dot
hipblasStatus_t hipblasCdotuCast(hipblasHandle_t       handle,
                                 int                   n,
                                 const hipblasComplex* x,
                                 int                   incx,
                                 const hipblasComplex* y,
                                 int                   incy,
                                 hipblasComplex*       result)
{
    return hipblasCdotu(
        handle, n, (const hipComplex*)x, incx, (const hipComplex*)y, incy, (hipComplex*)result);
}

hipblasStatus_t hipblasZdotuCast(hipblasHandle_t             handle,
                                 int                         n,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* y,
                                 int                         incy,
                                 hipblasDoubleComplex*       result)
{
    return hipblasZdotu(handle,
                        n,
                        (const hipDoubleComplex*)x,
                        incx,
                        (const hipDoubleComplex*)y,
                        incy,
                        (hipDoubleComplex*)result);
}

hipblasStatus_t hipblasCdotcCast(hipblasHandle_t       handle,
                                 int                   n,
                                 const hipblasComplex* x,
                                 int                   incx,
                                 const hipblasComplex* y,
                                 int                   incy,
                                 hipblasComplex*       result)
{
    return hipblasCdotc(
        handle, n, (const hipComplex*)x, incx, (const hipComplex*)y, incy, (hipComplex*)result);
}

hipblasStatus_t hipblasZdotcCast(hipblasHandle_t             handle,
                                 int                         n,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* y,
                                 int                         incy,
                                 hipblasDoubleComplex*       result)
{
    return hipblasZdotc(handle,
                        n,
                        (const hipDoubleComplex*)x,
                        incx,
                        (const hipDoubleComplex*)y,
                        incy,
                        (hipDoubleComplex*)result);
}

hipblasStatus_t hipblasCdotuCast_64(hipblasHandle_t       handle,
                                    int64_t               n,
                                    const hipblasComplex* x,
                                    int64_t               incx,
                                    const hipblasComplex* y,
                                    int64_t               incy,
                                    hipblasComplex*       result)
{
    return hipblasCdotu_64(
        handle, n, (const hipComplex*)x, incx, (const hipComplex*)y, incy, (hipComplex*)result);
}

hipblasStatus_t hipblasZdotuCast_64(hipblasHandle_t             handle,
                                    int64_t                     n,
                                    const hipblasDoubleComplex* x,
                                    int64_t                     incx,
                                    const hipblasDoubleComplex* y,
                                    int64_t                     incy,
                                    hipblasDoubleComplex*       result)
{
    return hipblasZdotu_64(handle,
                           n,
                           (const hipDoubleComplex*)x,
                           incx,
                           (const hipDoubleComplex*)y,
                           incy,
                           (hipDoubleComplex*)result);
}

hipblasStatus_t hipblasCdotcCast_64(hipblasHandle_t       handle,
                                    int64_t               n,
                                    const hipblasComplex* x,
                                    int64_t               incx,
                                    const hipblasComplex* y,
                                    int64_t               incy,
                                    hipblasComplex*       result)
{
    return hipblasCdotc_64(
        handle, n, (const hipComplex*)x, incx, (const hipComplex*)y, incy, (hipComplex*)result);
}

hipblasStatus_t hipblasZdotcCast_64(hipblasHandle_t             handle,
                                    int64_t                     n,
                                    const hipblasDoubleComplex* x,
                                    int64_t                     incx,
                                    const hipblasDoubleComplex* y,
                                    int64_t                     incy,
                                    hipblasDoubleComplex*       result)
{
    return hipblasZdotc_64(handle,
                           n,
                           (const hipDoubleComplex*)x,
                           incx,
                           (const hipDoubleComplex*)y,
                           incy,
                           (hipDoubleComplex*)result);
}

// dot_batched
hipblasStatus_t hipblasCdotuBatchedCast(hipblasHandle_t             handle,
                                        int                         n,
                                        const hipblasComplex* const x[],
                                        int                         incx,
                                        const hipblasComplex* const y[],
                                        int                         incy,
                                        int                         batch_count,
                                        hipblasComplex*             result)
{
    return hipblasCdotuBatched(handle,
                               n,
                               (const hipComplex* const*)x,
                               incx,
                               (const hipComplex* const*)y,
                               incy,
                               batch_count,
                               (hipComplex*)result);
}

hipblasStatus_t hipblasCdotcBatchedCast(hipblasHandle_t             handle,
                                        int                         n,
                                        const hipblasComplex* const x[],
                                        int                         incx,
                                        const hipblasComplex* const y[],
                                        int                         incy,
                                        int                         batch_count,
                                        hipblasComplex*             result)
{
    return hipblasCdotcBatched(handle,
                               n,
                               (const hipComplex* const*)x,
                               incx,
                               (const hipComplex* const*)y,
                               incy,
                               batch_count,
                               (hipComplex*)result);
}

hipblasStatus_t hipblasZdotuBatchedCast(hipblasHandle_t                   handle,
                                        int                               n,
                                        const hipblasDoubleComplex* const x[],
                                        int                               incx,
                                        const hipblasDoubleComplex* const y[],
                                        int                               incy,
                                        int                               batch_count,
                                        hipblasDoubleComplex*             result)
{
    return hipblasZdotuBatched(handle,
                               n,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (const hipDoubleComplex* const*)y,
                               incy,
                               batch_count,
                               (hipDoubleComplex*)result);
}

hipblasStatus_t hipblasZdotcBatchedCast(hipblasHandle_t                   handle,
                                        int                               n,
                                        const hipblasDoubleComplex* const x[],
                                        int                               incx,
                                        const hipblasDoubleComplex* const y[],
                                        int                               incy,
                                        int                               batch_count,
                                        hipblasDoubleComplex*             result)
{
    return hipblasZdotcBatched(handle,
                               n,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (const hipDoubleComplex* const*)y,
                               incy,
                               batch_count,
                               (hipDoubleComplex*)result);
}

hipblasStatus_t hipblasCdotuBatchedCast_64(hipblasHandle_t             handle,
                                           int64_t                     n,
                                           const hipblasComplex* const x[],
                                           int64_t                     incx,
                                           const hipblasComplex* const y[],
                                           int64_t                     incy,
                                           int64_t                     batch_count,
                                           hipblasComplex*             result)
{
    return hipblasCdotuBatched_64(handle,
                                  n,
                                  (const hipComplex* const*)x,
                                  incx,
                                  (const hipComplex* const*)y,
                                  incy,
                                  batch_count,
                                  (hipComplex*)result);
}

hipblasStatus_t hipblasCdotcBatchedCast_64(hipblasHandle_t             handle,
                                           int64_t                     n,
                                           const hipblasComplex* const x[],
                                           int64_t                     incx,
                                           const hipblasComplex* const y[],
                                           int64_t                     incy,
                                           int64_t                     batch_count,
                                           hipblasComplex*             result)
{
    return hipblasCdotcBatched_64(handle,
                                  n,
                                  (const hipComplex* const*)x,
                                  incx,
                                  (const hipComplex* const*)y,
                                  incy,
                                  batch_count,
                                  (hipComplex*)result);
}

hipblasStatus_t hipblasZdotuBatchedCast_64(hipblasHandle_t                   handle,
                                           int64_t                           n,
                                           const hipblasDoubleComplex* const x[],
                                           int64_t                           incx,
                                           const hipblasDoubleComplex* const y[],
                                           int64_t                           incy,
                                           int64_t                           batch_count,
                                           hipblasDoubleComplex*             result)
{
    return hipblasZdotuBatched_64(handle,
                                  n,
                                  (const hipDoubleComplex* const*)x,
                                  incx,
                                  (const hipDoubleComplex* const*)y,
                                  incy,
                                  batch_count,
                                  (hipDoubleComplex*)result);
}

hipblasStatus_t hipblasZdotcBatchedCast_64(hipblasHandle_t                   handle,
                                           int64_t                           n,
                                           const hipblasDoubleComplex* const x[],
                                           int64_t                           incx,
                                           const hipblasDoubleComplex* const y[],
                                           int64_t                           incy,
                                           int64_t                           batch_count,
                                           hipblasDoubleComplex*             result)
{
    return hipblasZdotcBatched_64(handle,
                                  n,
                                  (const hipDoubleComplex* const*)x,
                                  incx,
                                  (const hipDoubleComplex* const*)y,
                                  incy,
                                  batch_count,
                                  (hipDoubleComplex*)result);
}

// dot_strided_batched
hipblasStatus_t hipblasCdotuStridedBatchedCast(hipblasHandle_t       handle,
                                               int                   n,
                                               const hipblasComplex* x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               const hipblasComplex* y,
                                               int                   incy,
                                               hipblasStride         stridey,
                                               int                   batch_count,
                                               hipblasComplex*       result)
{
    return hipblasCdotuStridedBatched(handle,
                                      n,
                                      (const hipComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipComplex*)y,
                                      incy,
                                      stridey,
                                      batch_count,
                                      (hipComplex*)result);
}

hipblasStatus_t hipblasCdotcStridedBatchedCast(hipblasHandle_t       handle,
                                               int                   n,
                                               const hipblasComplex* x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               const hipblasComplex* y,
                                               int                   incy,
                                               hipblasStride         stridey,
                                               int                   batch_count,
                                               hipblasComplex*       result)
{
    return hipblasCdotcStridedBatched(handle,
                                      n,
                                      (const hipComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipComplex*)y,
                                      incy,
                                      stridey,
                                      batch_count,
                                      (hipComplex*)result);
}

hipblasStatus_t hipblasZdotuStridedBatchedCast(hipblasHandle_t             handle,
                                               int                         n,
                                               const hipblasDoubleComplex* x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               const hipblasDoubleComplex* y,
                                               int                         incy,
                                               hipblasStride               stridey,
                                               int                         batch_count,
                                               hipblasDoubleComplex*       result)
{
    return hipblasZdotuStridedBatched(handle,
                                      n,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipDoubleComplex*)y,
                                      incy,
                                      stridey,
                                      batch_count,
                                      (hipDoubleComplex*)result);
}

hipblasStatus_t hipblasZdotcStridedBatchedCast(hipblasHandle_t             handle,
                                               int                         n,
                                               const hipblasDoubleComplex* x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               const hipblasDoubleComplex* y,
                                               int                         incy,
                                               hipblasStride               stridey,
                                               int                         batch_count,
                                               hipblasDoubleComplex*       result)
{
    return hipblasZdotcStridedBatched(handle,
                                      n,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipDoubleComplex*)y,
                                      incy,
                                      stridey,
                                      batch_count,
                                      (hipDoubleComplex*)result);
}

hipblasStatus_t hipblasCdotuStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  int64_t               n,
                                                  const hipblasComplex* x,
                                                  int64_t               incx,
                                                  hipblasStride         stridex,
                                                  const hipblasComplex* y,
                                                  int64_t               incy,
                                                  hipblasStride         stridey,
                                                  int64_t               batch_count,
                                                  hipblasComplex*       result)
{
    return hipblasCdotuStridedBatched_64(handle,
                                         n,
                                         (const hipComplex*)x,
                                         incx,
                                         stridex,
                                         (const hipComplex*)y,
                                         incy,
                                         stridey,
                                         batch_count,
                                         (hipComplex*)result);
}

hipblasStatus_t hipblasCdotcStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  int64_t               n,
                                                  const hipblasComplex* x,
                                                  int64_t               incx,
                                                  hipblasStride         stridex,
                                                  const hipblasComplex* y,
                                                  int64_t               incy,
                                                  hipblasStride         stridey,
                                                  int64_t               batch_count,
                                                  hipblasComplex*       result)
{
    return hipblasCdotcStridedBatched_64(handle,
                                         n,
                                         (const hipComplex*)x,
                                         incx,
                                         stridex,
                                         (const hipComplex*)y,
                                         incy,
                                         stridey,
                                         batch_count,
                                         (hipComplex*)result);
}

hipblasStatus_t hipblasZdotuStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  int64_t                     n,
                                                  const hipblasDoubleComplex* x,
                                                  int64_t                     incx,
                                                  hipblasStride               stridex,
                                                  const hipblasDoubleComplex* y,
                                                  int64_t                     incy,
                                                  hipblasStride               stridey,
                                                  int64_t                     batch_count,
                                                  hipblasDoubleComplex*       result)
{
    return hipblasZdotuStridedBatched_64(handle,
                                         n,
                                         (const hipDoubleComplex*)x,
                                         incx,
                                         stridex,
                                         (const hipDoubleComplex*)y,
                                         incy,
                                         stridey,
                                         batch_count,
                                         (hipDoubleComplex*)result);
}

hipblasStatus_t hipblasZdotcStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  int64_t                     n,
                                                  const hipblasDoubleComplex* x,
                                                  int64_t                     incx,
                                                  hipblasStride               stridex,
                                                  const hipblasDoubleComplex* y,
                                                  int64_t                     incy,
                                                  hipblasStride               stridey,
                                                  int64_t                     batch_count,
                                                  hipblasDoubleComplex*       result)
{
    return hipblasZdotcStridedBatched_64(handle,
                                         n,
                                         (const hipDoubleComplex*)x,
                                         incx,
                                         stridex,
                                         (const hipDoubleComplex*)y,
                                         incy,
                                         stridey,
                                         batch_count,
                                         (hipDoubleComplex*)result);
}

// asum
hipblasStatus_t hipblasScasumCast(
    hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, float* result)
{
    return hipblasScasum(handle, n, (const hipComplex*)x, incx, result);
}

hipblasStatus_t hipblasDzasumCast(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, double* result)
{
    return hipblasDzasum(handle, n, (const hipDoubleComplex*)x, incx, result);
}

hipblasStatus_t hipblasScasumCast_64(
    hipblasHandle_t handle, int64_t n, const hipblasComplex* x, int64_t incx, float* result)
{
    return hipblasScasum_64(handle, n, (const hipComplex*)x, incx, result);
}

hipblasStatus_t hipblasDzasumCast_64(
    hipblasHandle_t handle, int64_t n, const hipblasDoubleComplex* x, int64_t incx, double* result)
{
    return hipblasDzasum_64(handle, n, (const hipDoubleComplex*)x, incx, result);
}

// asum_batched
hipblasStatus_t hipblasScasumBatchedCast(hipblasHandle_t             handle,
                                         int                         n,
                                         const hipblasComplex* const x[],
                                         int                         incx,
                                         int                         batch_count,
                                         float*                      result)
{
    return hipblasScasumBatched(handle, n, (const hipComplex* const*)x, incx, batch_count, result);
}

hipblasStatus_t hipblasDzasumBatchedCast(hipblasHandle_t                   handle,
                                         int                               n,
                                         const hipblasDoubleComplex* const x[],
                                         int                               incx,
                                         int                               batch_count,
                                         double*                           result)
{
    return hipblasDzasumBatched(
        handle, n, (const hipDoubleComplex* const*)x, incx, batch_count, result);
}

hipblasStatus_t hipblasScasumBatchedCast_64(hipblasHandle_t             handle,
                                            int64_t                     n,
                                            const hipblasComplex* const x[],
                                            int64_t                     incx,
                                            int64_t                     batch_count,
                                            float*                      result)
{
    return hipblasScasumBatched_64(
        handle, n, (const hipComplex* const*)x, incx, batch_count, result);
}

hipblasStatus_t hipblasDzasumBatchedCast_64(hipblasHandle_t                   handle,
                                            int64_t                           n,
                                            const hipblasDoubleComplex* const x[],
                                            int64_t                           incx,
                                            int64_t                           batch_count,
                                            double*                           result)
{
    return hipblasDzasumBatched_64(
        handle, n, (const hipDoubleComplex* const*)x, incx, batch_count, result);
}

// asum_strided_batched
hipblasStatus_t hipblasScasumStridedBatchedCast(hipblasHandle_t       handle,
                                                int                   n,
                                                const hipblasComplex* x,
                                                int                   incx,
                                                hipblasStride         stridex,
                                                int                   batch_count,
                                                float*                result)
{
    return hipblasScasumStridedBatched(
        handle, n, (const hipComplex*)x, incx, stridex, batch_count, result);
}

hipblasStatus_t hipblasDzasumStridedBatchedCast(hipblasHandle_t             handle,
                                                int                         n,
                                                const hipblasDoubleComplex* x,
                                                int                         incx,
                                                hipblasStride               stridex,
                                                int                         batch_count,
                                                double*                     result)
{
    return hipblasDzasumStridedBatched(
        handle, n, (const hipDoubleComplex*)x, incx, stridex, batch_count, result);
}

hipblasStatus_t hipblasScasumStridedBatchedCast_64(hipblasHandle_t       handle,
                                                   int64_t               n,
                                                   const hipblasComplex* x,
                                                   int64_t               incx,
                                                   hipblasStride         stridex,
                                                   int64_t               batch_count,
                                                   float*                result)
{
    return hipblasScasumStridedBatched_64(
        handle, n, (const hipComplex*)x, incx, stridex, batch_count, result);
}

hipblasStatus_t hipblasDzasumStridedBatchedCast_64(hipblasHandle_t             handle,
                                                   int64_t                     n,
                                                   const hipblasDoubleComplex* x,
                                                   int64_t                     incx,
                                                   hipblasStride               stridex,
                                                   int64_t                     batch_count,
                                                   double*                     result)
{
    return hipblasDzasumStridedBatched_64(
        handle, n, (const hipDoubleComplex*)x, incx, stridex, batch_count, result);
}

// nrm2
hipblasStatus_t hipblasScnrm2Cast(
    hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, float* result)
{
    return hipblasScnrm2(handle, n, (const hipComplex*)x, incx, result);
}

hipblasStatus_t hipblasDznrm2Cast(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, double* result)
{
    return hipblasDznrm2(handle, n, (const hipDoubleComplex*)x, incx, result);
}

hipblasStatus_t hipblasScnrm2Cast_64(
    hipblasHandle_t handle, int64_t n, const hipblasComplex* x, int64_t incx, float* result)
{
    return hipblasScnrm2_64(handle, n, (const hipComplex*)x, incx, result);
}

hipblasStatus_t hipblasDznrm2Cast_64(
    hipblasHandle_t handle, int64_t n, const hipblasDoubleComplex* x, int64_t incx, double* result)
{
    return hipblasDznrm2_64(handle, n, (const hipDoubleComplex*)x, incx, result);
}

// nrm2_batched
hipblasStatus_t hipblasScnrm2BatchedCast(hipblasHandle_t             handle,
                                         int                         n,
                                         const hipblasComplex* const x[],
                                         int                         incx,
                                         int                         batch_count,
                                         float*                      result)
{
    return hipblasScnrm2Batched(handle, n, (const hipComplex* const*)x, incx, batch_count, result);
}

hipblasStatus_t hipblasDznrm2BatchedCast(hipblasHandle_t                   handle,
                                         int                               n,
                                         const hipblasDoubleComplex* const x[],
                                         int                               incx,
                                         int                               batch_count,
                                         double*                           result)
{
    return hipblasDznrm2Batched(
        handle, n, (const hipDoubleComplex* const*)x, incx, batch_count, result);
}

hipblasStatus_t hipblasScnrm2BatchedCast_64(hipblasHandle_t             handle,
                                            int64_t                     n,
                                            const hipblasComplex* const x[],
                                            int64_t                     incx,
                                            int64_t                     batch_count,
                                            float*                      result)
{
    return hipblasScnrm2Batched_64(
        handle, n, (const hipComplex* const*)x, incx, batch_count, result);
}

hipblasStatus_t hipblasDznrm2BatchedCast_64(hipblasHandle_t                   handle,
                                            int64_t                           n,
                                            const hipblasDoubleComplex* const x[],
                                            int64_t                           incx,
                                            int64_t                           batch_count,
                                            double*                           result)
{
    return hipblasDznrm2Batched_64(
        handle, n, (const hipDoubleComplex* const*)x, incx, batch_count, result);
}

// nrm2_strided_batched
hipblasStatus_t hipblasScnrm2StridedBatchedCast(hipblasHandle_t       handle,
                                                int                   n,
                                                const hipblasComplex* x,
                                                int                   incx,
                                                hipblasStride         stridex,
                                                int                   batch_count,
                                                float*                result)
{
    return hipblasScnrm2StridedBatched(
        handle, n, (const hipComplex*)x, incx, stridex, batch_count, result);
}

hipblasStatus_t hipblasDznrm2StridedBatchedCast(hipblasHandle_t             handle,
                                                int                         n,
                                                const hipblasDoubleComplex* x,
                                                int                         incx,
                                                hipblasStride               stridex,
                                                int                         batch_count,
                                                double*                     result)
{
    return hipblasDznrm2StridedBatched(
        handle, n, (const hipDoubleComplex*)x, incx, stridex, batch_count, result);
}

hipblasStatus_t hipblasScnrm2StridedBatchedCast_64(hipblasHandle_t       handle,
                                                   int64_t               n,
                                                   const hipblasComplex* x,
                                                   int64_t               incx,
                                                   hipblasStride         stridex,
                                                   int64_t               batch_count,
                                                   float*                result)
{
    return hipblasScnrm2StridedBatched_64(
        handle, n, (const hipComplex*)x, incx, stridex, batch_count, result);
}

hipblasStatus_t hipblasDznrm2StridedBatchedCast_64(hipblasHandle_t             handle,
                                                   int64_t                     n,
                                                   const hipblasDoubleComplex* x,
                                                   int64_t                     incx,
                                                   hipblasStride               stridex,
                                                   int64_t                     batch_count,
                                                   double*                     result)
{
    return hipblasDznrm2StridedBatched_64(
        handle, n, (const hipDoubleComplex*)x, incx, stridex, batch_count, result);
}

// rot
hipblasStatus_t hipblasCrotCast(hipblasHandle_t       handle,
                                int                   n,
                                hipblasComplex*       x,
                                int                   incx,
                                hipblasComplex*       y,
                                int                   incy,
                                const float*          c,
                                const hipblasComplex* s)
{
    return hipblasCrot(
        handle, n, (hipComplex*)x, incx, (hipComplex*)y, incy, c, (const hipComplex*)s);
}

hipblasStatus_t hipblasCsrotCast(hipblasHandle_t handle,
                                 int             n,
                                 hipblasComplex* x,
                                 int             incx,
                                 hipblasComplex* y,
                                 int             incy,
                                 const float*    c,
                                 const float*    s)
{
    return hipblasCsrot(handle, n, (hipComplex*)x, incx, (hipComplex*)y, incy, c, s);
}

hipblasStatus_t hipblasZrotCast(hipblasHandle_t             handle,
                                int                         n,
                                hipblasDoubleComplex*       x,
                                int                         incx,
                                hipblasDoubleComplex*       y,
                                int                         incy,
                                const double*               c,
                                const hipblasDoubleComplex* s)
{
    return hipblasZrot(handle,
                       n,
                       (hipDoubleComplex*)x,
                       incx,
                       (hipDoubleComplex*)y,
                       incy,
                       c,
                       (const hipDoubleComplex*)s);
}

hipblasStatus_t hipblasZdrotCast(hipblasHandle_t       handle,
                                 int                   n,
                                 hipblasDoubleComplex* x,
                                 int                   incx,
                                 hipblasDoubleComplex* y,
                                 int                   incy,
                                 const double*         c,
                                 const double*         s)
{
    return hipblasZdrot(handle, n, (hipDoubleComplex*)x, incx, (hipDoubleComplex*)y, incy, c, s);
}

hipblasStatus_t hipblasCrotCast_64(hipblasHandle_t       handle,
                                   int64_t               n,
                                   hipblasComplex*       x,
                                   int64_t               incx,
                                   hipblasComplex*       y,
                                   int64_t               incy,
                                   const float*          c,
                                   const hipblasComplex* s)
{
    return hipblasCrot_64(
        handle, n, (hipComplex*)x, incx, (hipComplex*)y, incy, c, (const hipComplex*)s);
}

hipblasStatus_t hipblasCsrotCast_64(hipblasHandle_t handle,
                                    int64_t         n,
                                    hipblasComplex* x,
                                    int64_t         incx,
                                    hipblasComplex* y,
                                    int64_t         incy,
                                    const float*    c,
                                    const float*    s)
{
    return hipblasCsrot_64(handle, n, (hipComplex*)x, incx, (hipComplex*)y, incy, c, s);
}

hipblasStatus_t hipblasZrotCast_64(hipblasHandle_t             handle,
                                   int64_t                     n,
                                   hipblasDoubleComplex*       x,
                                   int64_t                     incx,
                                   hipblasDoubleComplex*       y,
                                   int64_t                     incy,
                                   const double*               c,
                                   const hipblasDoubleComplex* s)
{
    return hipblasZrot_64(handle,
                          n,
                          (hipDoubleComplex*)x,
                          incx,
                          (hipDoubleComplex*)y,
                          incy,
                          c,
                          (const hipDoubleComplex*)s);
}

hipblasStatus_t hipblasZdrotCast_64(hipblasHandle_t       handle,
                                    int64_t               n,
                                    hipblasDoubleComplex* x,
                                    int64_t               incx,
                                    hipblasDoubleComplex* y,
                                    int64_t               incy,
                                    const double*         c,
                                    const double*         s)
{
    return hipblasZdrot_64(handle, n, (hipDoubleComplex*)x, incx, (hipDoubleComplex*)y, incy, c, s);
}

// rot_batched
hipblasStatus_t hipblasCrotBatchedCast(hipblasHandle_t       handle,
                                       int                   n,
                                       hipblasComplex* const x[],
                                       int                   incx,
                                       hipblasComplex* const y[],
                                       int                   incy,
                                       const float*          c,
                                       const hipblasComplex* s,
                                       int                   batch_count)
{
    return hipblasCrotBatched(handle,
                              n,
                              (hipComplex* const*)x,
                              incx,
                              (hipComplex* const*)y,
                              incy,
                              c,
                              (const hipComplex*)s,
                              batch_count);
}

hipblasStatus_t hipblasCsrotBatchedCast(hipblasHandle_t       handle,
                                        int                   n,
                                        hipblasComplex* const x[],
                                        int                   incx,
                                        hipblasComplex* const y[],
                                        int                   incy,
                                        const float*          c,
                                        const float*          s,
                                        int                   batch_count)
{
    return hipblasCsrotBatched(
        handle, n, (hipComplex* const*)x, incx, (hipComplex* const*)y, incy, c, s, batch_count);
}

hipblasStatus_t hipblasZrotBatchedCast(hipblasHandle_t             handle,
                                       int                         n,
                                       hipblasDoubleComplex* const x[],
                                       int                         incx,
                                       hipblasDoubleComplex* const y[],
                                       int                         incy,
                                       const double*               c,
                                       const hipblasDoubleComplex* s,
                                       int                         batch_count)
{
    return hipblasZrotBatched(handle,
                              n,
                              (hipDoubleComplex* const*)x,
                              incx,
                              (hipDoubleComplex* const*)y,
                              incy,
                              c,
                              (const hipDoubleComplex*)s,
                              batch_count);
}

hipblasStatus_t hipblasZdrotBatchedCast(hipblasHandle_t             handle,
                                        int                         n,
                                        hipblasDoubleComplex* const x[],
                                        int                         incx,
                                        hipblasDoubleComplex* const y[],
                                        int                         incy,
                                        const double*               c,
                                        const double*               s,
                                        int                         batch_count)
{
    return hipblasZdrotBatched(handle,
                               n,
                               (hipDoubleComplex* const*)x,
                               incx,
                               (hipDoubleComplex* const*)y,
                               incy,
                               c,
                               s,
                               batch_count);
}

hipblasStatus_t hipblasCrotBatchedCast_64(hipblasHandle_t       handle,
                                          int64_t               n,
                                          hipblasComplex* const x[],
                                          int64_t               incx,
                                          hipblasComplex* const y[],
                                          int64_t               incy,
                                          const float*          c,
                                          const hipblasComplex* s,
                                          int64_t               batch_count)
{
    return hipblasCrotBatched_64(handle,
                                 n,
                                 (hipComplex* const*)x,
                                 incx,
                                 (hipComplex* const*)y,
                                 incy,
                                 c,
                                 (const hipComplex*)s,
                                 batch_count);
}

hipblasStatus_t hipblasCsrotBatchedCast_64(hipblasHandle_t       handle,
                                           int64_t               n,
                                           hipblasComplex* const x[],
                                           int64_t               incx,
                                           hipblasComplex* const y[],
                                           int64_t               incy,
                                           const float*          c,
                                           const float*          s,
                                           int64_t               batch_count)
{
    return hipblasCsrotBatched_64(
        handle, n, (hipComplex* const*)x, incx, (hipComplex* const*)y, incy, c, s, batch_count);
}

hipblasStatus_t hipblasZrotBatchedCast_64(hipblasHandle_t             handle,
                                          int64_t                     n,
                                          hipblasDoubleComplex* const x[],
                                          int64_t                     incx,
                                          hipblasDoubleComplex* const y[],
                                          int64_t                     incy,
                                          const double*               c,
                                          const hipblasDoubleComplex* s,
                                          int64_t                     batch_count)
{
    return hipblasZrotBatched_64(handle,
                                 n,
                                 (hipDoubleComplex* const*)x,
                                 incx,
                                 (hipDoubleComplex* const*)y,
                                 incy,
                                 c,
                                 (const hipDoubleComplex*)s,
                                 batch_count);
}

hipblasStatus_t hipblasZdrotBatchedCast_64(hipblasHandle_t             handle,
                                           int64_t                     n,
                                           hipblasDoubleComplex* const x[],
                                           int64_t                     incx,
                                           hipblasDoubleComplex* const y[],
                                           int64_t                     incy,
                                           const double*               c,
                                           const double*               s,
                                           int64_t                     batch_count)
{
    return hipblasZdrotBatched_64(handle,
                                  n,
                                  (hipDoubleComplex* const*)x,
                                  incx,
                                  (hipDoubleComplex* const*)y,
                                  incy,
                                  c,
                                  s,
                                  batch_count);
}

// rot_strided_batched
hipblasStatus_t hipblasCrotStridedBatchedCast(hipblasHandle_t       handle,
                                              int                   n,
                                              hipblasComplex*       x,
                                              int                   incx,
                                              hipblasStride         stridex,
                                              hipblasComplex*       y,
                                              int                   incy,
                                              hipblasStride         stridey,
                                              const float*          c,
                                              const hipblasComplex* s,
                                              int                   batch_count)
{
    return hipblasCrotStridedBatched(handle,
                                     n,
                                     (hipComplex*)x,
                                     incx,
                                     stridex,
                                     (hipComplex*)y,
                                     incy,
                                     stridey,
                                     c,
                                     (const hipComplex*)s,
                                     batch_count);
}

hipblasStatus_t hipblasCsrotStridedBatchedCast(hipblasHandle_t handle,
                                               int             n,
                                               hipblasComplex* x,
                                               int             incx,
                                               hipblasStride   stridex,
                                               hipblasComplex* y,
                                               int             incy,
                                               hipblasStride   stridey,
                                               const float*    c,
                                               const float*    s,
                                               int             batch_count)
{
    return hipblasCsrotStridedBatched(
        handle, n, (hipComplex*)x, incx, stridex, (hipComplex*)y, incy, stridey, c, s, batch_count);
}

hipblasStatus_t hipblasZrotStridedBatchedCast(hipblasHandle_t             handle,
                                              int                         n,
                                              hipblasDoubleComplex*       x,
                                              int                         incx,
                                              hipblasStride               stridex,
                                              hipblasDoubleComplex*       y,
                                              int                         incy,
                                              hipblasStride               stridey,
                                              const double*               c,
                                              const hipblasDoubleComplex* s,
                                              int                         batch_count)
{
    return hipblasZrotStridedBatched(handle,
                                     n,
                                     (hipDoubleComplex*)x,
                                     incx,
                                     stridex,
                                     (hipDoubleComplex*)y,
                                     incy,
                                     stridey,
                                     c,
                                     (const hipDoubleComplex*)s,
                                     batch_count);
}

hipblasStatus_t hipblasZdrotStridedBatchedCast(hipblasHandle_t       handle,
                                               int                   n,
                                               hipblasDoubleComplex* x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               hipblasDoubleComplex* y,
                                               int                   incy,
                                               hipblasStride         stridey,
                                               const double*         c,
                                               const double*         s,
                                               int                   batch_count)
{
    return hipblasZdrotStridedBatched(handle,
                                      n,
                                      (hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      (hipDoubleComplex*)y,
                                      incy,
                                      stridey,
                                      c,
                                      s,
                                      batch_count);
}

hipblasStatus_t hipblasCrotStridedBatchedCast_64(hipblasHandle_t       handle,
                                                 int64_t               n,
                                                 hipblasComplex*       x,
                                                 int64_t               incx,
                                                 hipblasStride         stridex,
                                                 hipblasComplex*       y,
                                                 int64_t               incy,
                                                 hipblasStride         stridey,
                                                 const float*          c,
                                                 const hipblasComplex* s,
                                                 int64_t               batch_count)
{
    return hipblasCrotStridedBatched_64(handle,
                                        n,
                                        (hipComplex*)x,
                                        incx,
                                        stridex,
                                        (hipComplex*)y,
                                        incy,
                                        stridey,
                                        c,
                                        (const hipComplex*)s,
                                        batch_count);
}

hipblasStatus_t hipblasCsrotStridedBatchedCast_64(hipblasHandle_t handle,
                                                  int64_t         n,
                                                  hipblasComplex* x,
                                                  int64_t         incx,
                                                  hipblasStride   stridex,
                                                  hipblasComplex* y,
                                                  int64_t         incy,
                                                  hipblasStride   stridey,
                                                  const float*    c,
                                                  const float*    s,
                                                  int64_t         batch_count)
{
    return hipblasCsrotStridedBatched_64(
        handle, n, (hipComplex*)x, incx, stridex, (hipComplex*)y, incy, stridey, c, s, batch_count);
}

hipblasStatus_t hipblasZrotStridedBatchedCast_64(hipblasHandle_t             handle,
                                                 int64_t                     n,
                                                 hipblasDoubleComplex*       x,
                                                 int64_t                     incx,
                                                 hipblasStride               stridex,
                                                 hipblasDoubleComplex*       y,
                                                 int64_t                     incy,
                                                 hipblasStride               stridey,
                                                 const double*               c,
                                                 const hipblasDoubleComplex* s,
                                                 int64_t                     batch_count)
{
    return hipblasZrotStridedBatched_64(handle,
                                        n,
                                        (hipDoubleComplex*)x,
                                        incx,
                                        stridex,
                                        (hipDoubleComplex*)y,
                                        incy,
                                        stridey,
                                        c,
                                        (const hipDoubleComplex*)s,
                                        batch_count);
}

hipblasStatus_t hipblasZdrotStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  int64_t               n,
                                                  hipblasDoubleComplex* x,
                                                  int64_t               incx,
                                                  hipblasStride         stridex,
                                                  hipblasDoubleComplex* y,
                                                  int64_t               incy,
                                                  hipblasStride         stridey,
                                                  const double*         c,
                                                  const double*         s,
                                                  int64_t               batch_count)
{
    return hipblasZdrotStridedBatched_64(handle,
                                         n,
                                         (hipDoubleComplex*)x,
                                         incx,
                                         stridex,
                                         (hipDoubleComplex*)y,
                                         incy,
                                         stridey,
                                         c,
                                         s,
                                         batch_count);
}

// rotg
hipblasStatus_t hipblasCrotgCast(
    hipblasHandle_t handle, hipblasComplex* a, hipblasComplex* b, float* c, hipblasComplex* s)
{
    return hipblasCrotg(handle, (hipComplex*)a, (hipComplex*)b, c, (hipComplex*)s);
}

hipblasStatus_t hipblasZrotgCast(hipblasHandle_t       handle,
                                 hipblasDoubleComplex* a,
                                 hipblasDoubleComplex* b,
                                 double*               c,
                                 hipblasDoubleComplex* s)
{
    return hipblasZrotg(
        handle, (hipDoubleComplex*)a, (hipDoubleComplex*)b, c, (hipDoubleComplex*)s);
}

hipblasStatus_t hipblasCrotgCast_64(
    hipblasHandle_t handle, hipblasComplex* a, hipblasComplex* b, float* c, hipblasComplex* s)
{
    return hipblasCrotg_64(handle, (hipComplex*)a, (hipComplex*)b, c, (hipComplex*)s);
}

hipblasStatus_t hipblasZrotgCast_64(hipblasHandle_t       handle,
                                    hipblasDoubleComplex* a,
                                    hipblasDoubleComplex* b,
                                    double*               c,
                                    hipblasDoubleComplex* s)
{
    return hipblasZrotg_64(
        handle, (hipDoubleComplex*)a, (hipDoubleComplex*)b, c, (hipDoubleComplex*)s);
}

// rotg_batched
hipblasStatus_t hipblasCrotgBatchedCast(hipblasHandle_t       handle,
                                        hipblasComplex* const a[],
                                        hipblasComplex* const b[],
                                        float* const          c[],
                                        hipblasComplex* const s[],
                                        int                   batch_count)
{
    return hipblasCrotgBatched(handle,
                               (hipComplex* const*)a,
                               (hipComplex* const*)b,
                               c,
                               (hipComplex* const*)s,
                               batch_count);
}

hipblasStatus_t hipblasZrotgBatchedCast(hipblasHandle_t             handle,
                                        hipblasDoubleComplex* const a[],
                                        hipblasDoubleComplex* const b[],
                                        double* const               c[],
                                        hipblasDoubleComplex* const s[],
                                        int                         batch_count)
{
    return hipblasZrotgBatched(handle,
                               (hipDoubleComplex* const*)a,
                               (hipDoubleComplex* const*)b,
                               c,
                               (hipDoubleComplex* const*)s,
                               batch_count);
}

hipblasStatus_t hipblasCrotgBatchedCast_64(hipblasHandle_t       handle,
                                           hipblasComplex* const a[],
                                           hipblasComplex* const b[],
                                           float* const          c[],
                                           hipblasComplex* const s[],
                                           int64_t               batch_count)
{
    return hipblasCrotgBatched_64(handle,
                                  (hipComplex* const*)a,
                                  (hipComplex* const*)b,
                                  c,
                                  (hipComplex* const*)s,
                                  batch_count);
}

hipblasStatus_t hipblasZrotgBatchedCast_64(hipblasHandle_t             handle,
                                           hipblasDoubleComplex* const a[],
                                           hipblasDoubleComplex* const b[],
                                           double* const               c[],
                                           hipblasDoubleComplex* const s[],
                                           int64_t                     batch_count)
{
    return hipblasZrotgBatched_64(handle,
                                  (hipDoubleComplex* const*)a,
                                  (hipDoubleComplex* const*)b,
                                  c,
                                  (hipDoubleComplex* const*)s,
                                  batch_count);
}

// rotg_strided_batched
hipblasStatus_t hipblasCrotgStridedBatchedCast(hipblasHandle_t handle,
                                               hipblasComplex* a,
                                               hipblasStride   stridea,
                                               hipblasComplex* b,
                                               hipblasStride   strideb,
                                               float*          c,
                                               hipblasStride   stridec,
                                               hipblasComplex* s,
                                               hipblasStride   strides,
                                               int             batch_count)
{
    return hipblasCrotgStridedBatched(handle,
                                      (hipComplex*)a,
                                      stridea,
                                      (hipComplex*)b,
                                      strideb,
                                      c,
                                      stridec,
                                      (hipComplex*)s,
                                      strides,
                                      batch_count);
}

hipblasStatus_t hipblasZrotgStridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasDoubleComplex* a,
                                               hipblasStride         stridea,
                                               hipblasDoubleComplex* b,
                                               hipblasStride         strideb,
                                               double*               c,
                                               hipblasStride         stridec,
                                               hipblasDoubleComplex* s,
                                               hipblasStride         strides,
                                               int                   batch_count)
{
    return hipblasZrotgStridedBatched(handle,
                                      (hipDoubleComplex*)a,
                                      stridea,
                                      (hipDoubleComplex*)b,
                                      strideb,
                                      c,
                                      stridec,
                                      (hipDoubleComplex*)s,
                                      strides,
                                      batch_count);
}

hipblasStatus_t hipblasCrotgStridedBatchedCast_64(hipblasHandle_t handle,
                                                  hipblasComplex* a,
                                                  hipblasStride   stridea,
                                                  hipblasComplex* b,
                                                  hipblasStride   strideb,
                                                  float*          c,
                                                  hipblasStride   stridec,
                                                  hipblasComplex* s,
                                                  hipblasStride   strides,
                                                  int64_t         batch_count)
{
    return hipblasCrotgStridedBatched_64(handle,
                                         (hipComplex*)a,
                                         stridea,
                                         (hipComplex*)b,
                                         strideb,
                                         c,
                                         stridec,
                                         (hipComplex*)s,
                                         strides,
                                         batch_count);
}

hipblasStatus_t hipblasZrotgStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  hipblasDoubleComplex* a,
                                                  hipblasStride         stridea,
                                                  hipblasDoubleComplex* b,
                                                  hipblasStride         strideb,
                                                  double*               c,
                                                  hipblasStride         stridec,
                                                  hipblasDoubleComplex* s,
                                                  hipblasStride         strides,
                                                  int64_t               batch_count)
{
    return hipblasZrotgStridedBatched_64(handle,
                                         (hipDoubleComplex*)a,
                                         stridea,
                                         (hipDoubleComplex*)b,
                                         strideb,
                                         c,
                                         stridec,
                                         (hipDoubleComplex*)s,
                                         strides,
                                         batch_count);
}

// rotm, rotmg - no complex versions

// amax
hipblasStatus_t
    hipblasIcamaxCast(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result)
{
    return hipblasIcamax(handle, n, (const hipComplex*)x, incx, result);
}

hipblasStatus_t hipblasIzamaxCast(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result)
{
    return hipblasIzamax(handle, n, (const hipDoubleComplex*)x, incx, result);
}

hipblasStatus_t hipblasIcamaxCast_64(
    hipblasHandle_t handle, int64_t n, const hipblasComplex* x, int64_t incx, int64_t* result)
{
    return hipblasIcamax_64(handle, n, (const hipComplex*)x, incx, result);
}

hipblasStatus_t hipblasIzamaxCast_64(
    hipblasHandle_t handle, int64_t n, const hipblasDoubleComplex* x, int64_t incx, int64_t* result)
{
    return hipblasIzamax_64(handle, n, (const hipDoubleComplex*)x, incx, result);
}

// amax_batched
hipblasStatus_t hipblasIcamaxBatchedCast(hipblasHandle_t             handle,
                                         int                         n,
                                         const hipblasComplex* const x[],
                                         int                         incx,
                                         int                         batch_count,
                                         int*                        result)
{
    return hipblasIcamaxBatched(handle, n, (const hipComplex* const*)x, incx, batch_count, result);
}

hipblasStatus_t hipblasIzamaxBatchedCast(hipblasHandle_t                   handle,
                                         int                               n,
                                         const hipblasDoubleComplex* const x[],
                                         int                               incx,
                                         int                               batch_count,
                                         int*                              result)
{
    return hipblasIzamaxBatched(
        handle, n, (const hipDoubleComplex* const*)x, incx, batch_count, result);
}

hipblasStatus_t hipblasIcamaxBatchedCast_64(hipblasHandle_t             handle,
                                            int64_t                     n,
                                            const hipblasComplex* const x[],
                                            int64_t                     incx,
                                            int64_t                     batch_count,
                                            int64_t*                    result)
{
    return hipblasIcamaxBatched_64(
        handle, n, (const hipComplex* const*)x, incx, batch_count, result);
}

hipblasStatus_t hipblasIzamaxBatchedCast_64(hipblasHandle_t                   handle,
                                            int64_t                           n,
                                            const hipblasDoubleComplex* const x[],
                                            int64_t                           incx,
                                            int64_t                           batch_count,
                                            int64_t*                          result)
{
    return hipblasIzamaxBatched_64(
        handle, n, (const hipDoubleComplex* const*)x, incx, batch_count, result);
}

// amax_strided_batched
hipblasStatus_t hipblasIcamaxStridedBatchedCast(hipblasHandle_t       handle,
                                                int                   n,
                                                const hipblasComplex* x,
                                                int                   incx,
                                                hipblasStride         stridex,
                                                int                   batch_count,
                                                int*                  result)
{
    return hipblasIcamaxStridedBatched(
        handle, n, (const hipComplex*)x, incx, stridex, batch_count, result);
}

hipblasStatus_t hipblasIzamaxStridedBatchedCast(hipblasHandle_t             handle,
                                                int                         n,
                                                const hipblasDoubleComplex* x,
                                                int                         incx,
                                                hipblasStride               stridex,
                                                int                         batch_count,
                                                int*                        result)
{
    return hipblasIzamaxStridedBatched(
        handle, n, (const hipDoubleComplex*)x, incx, stridex, batch_count, result);
}

hipblasStatus_t hipblasIcamaxStridedBatchedCast_64(hipblasHandle_t       handle,
                                                   int64_t               n,
                                                   const hipblasComplex* x,
                                                   int64_t               incx,
                                                   hipblasStride         stridex,
                                                   int64_t               batch_count,
                                                   int64_t*              result)
{
    return hipblasIcamaxStridedBatched_64(
        handle, n, (const hipComplex*)x, incx, stridex, batch_count, result);
}

hipblasStatus_t hipblasIzamaxStridedBatchedCast_64(hipblasHandle_t             handle,
                                                   int64_t                     n,
                                                   const hipblasDoubleComplex* x,
                                                   int64_t                     incx,
                                                   hipblasStride               stridex,
                                                   int64_t                     batch_count,
                                                   int64_t*                    result)
{
    return hipblasIzamaxStridedBatched_64(
        handle, n, (const hipDoubleComplex*)x, incx, stridex, batch_count, result);
}

// amin
hipblasStatus_t
    hipblasIcaminCast(hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result)
{
    return hipblasIcamin(handle, n, (const hipComplex*)x, incx, result);
}

hipblasStatus_t hipblasIzaminCast(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result)
{
    return hipblasIzamin(handle, n, (const hipDoubleComplex*)x, incx, result);
}

hipblasStatus_t hipblasIcaminCast_64(
    hipblasHandle_t handle, int64_t n, const hipblasComplex* x, int64_t incx, int64_t* result)
{
    return hipblasIcamin_64(handle, n, (const hipComplex*)x, incx, result);
}

hipblasStatus_t hipblasIzaminCast_64(
    hipblasHandle_t handle, int64_t n, const hipblasDoubleComplex* x, int64_t incx, int64_t* result)
{
    return hipblasIzamin_64(handle, n, (const hipDoubleComplex*)x, incx, result);
}

// amin_batched
hipblasStatus_t hipblasIcaminBatchedCast(hipblasHandle_t             handle,
                                         int                         n,
                                         const hipblasComplex* const x[],
                                         int                         incx,
                                         int                         batch_count,
                                         int*                        result)
{
    return hipblasIcaminBatched(handle, n, (const hipComplex* const*)x, incx, batch_count, result);
}

hipblasStatus_t hipblasIzaminBatchedCast(hipblasHandle_t                   handle,
                                         int                               n,
                                         const hipblasDoubleComplex* const x[],
                                         int                               incx,
                                         int                               batch_count,
                                         int*                              result)
{
    return hipblasIzaminBatched(
        handle, n, (const hipDoubleComplex* const*)x, incx, batch_count, result);
}

hipblasStatus_t hipblasIcaminBatchedCast_64(hipblasHandle_t             handle,
                                            int64_t                     n,
                                            const hipblasComplex* const x[],
                                            int64_t                     incx,
                                            int64_t                     batch_count,
                                            int64_t*                    result)
{
    return hipblasIcaminBatched_64(
        handle, n, (const hipComplex* const*)x, incx, batch_count, result);
}

hipblasStatus_t hipblasIzaminBatchedCast_64(hipblasHandle_t                   handle,
                                            int64_t                           n,
                                            const hipblasDoubleComplex* const x[],
                                            int64_t                           incx,
                                            int64_t                           batch_count,
                                            int64_t*                          result)
{
    return hipblasIzaminBatched_64(
        handle, n, (const hipDoubleComplex* const*)x, incx, batch_count, result);
}

// amin_strided_batched
hipblasStatus_t hipblasIcaminStridedBatchedCast(hipblasHandle_t       handle,
                                                int                   n,
                                                const hipblasComplex* x,
                                                int                   incx,
                                                hipblasStride         stridex,
                                                int                   batch_count,
                                                int*                  result)
{
    return hipblasIcaminStridedBatched(
        handle, n, (const hipComplex*)x, incx, stridex, batch_count, result);
}

hipblasStatus_t hipblasIzaminStridedBatchedCast(hipblasHandle_t             handle,
                                                int                         n,
                                                const hipblasDoubleComplex* x,
                                                int                         incx,
                                                hipblasStride               stridex,
                                                int                         batch_count,
                                                int*                        result)
{
    return hipblasIzaminStridedBatched(
        handle, n, (const hipDoubleComplex*)x, incx, stridex, batch_count, result);
}

hipblasStatus_t hipblasIcaminStridedBatchedCast_64(hipblasHandle_t       handle,
                                                   int64_t               n,
                                                   const hipblasComplex* x,
                                                   int64_t               incx,
                                                   hipblasStride         stridex,
                                                   int64_t               batch_count,
                                                   int64_t*              result)
{
    return hipblasIcaminStridedBatched_64(
        handle, n, (const hipComplex*)x, incx, stridex, batch_count, result);
}

hipblasStatus_t hipblasIzaminStridedBatchedCast_64(hipblasHandle_t             handle,
                                                   int64_t                     n,
                                                   const hipblasDoubleComplex* x,
                                                   int64_t                     incx,
                                                   hipblasStride               stridex,
                                                   int64_t                     batch_count,
                                                   int64_t*                    result)
{
    return hipblasIzaminStridedBatched_64(
        handle, n, (const hipDoubleComplex*)x, incx, stridex, batch_count, result);
}

// scal
hipblasStatus_t hipblasCscalCast(
    hipblasHandle_t handle, int n, const hipblasComplex* alpha, hipblasComplex* x, int incx)
{
    return hipblasCscal(handle, n, (const hipComplex*)alpha, (hipComplex*)x, incx);
}

hipblasStatus_t hipblasCsscalCast(
    hipblasHandle_t handle, int n, const float* alpha, hipblasComplex* x, int incx)
{
    return hipblasCsscal(handle, n, alpha, (hipComplex*)x, incx);
}

hipblasStatus_t hipblasZscalCast(hipblasHandle_t             handle,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 hipblasDoubleComplex*       x,
                                 int                         incx)
{
    return hipblasZscal(handle, n, (const hipDoubleComplex*)alpha, (hipDoubleComplex*)x, incx);
}

hipblasStatus_t hipblasZdscalCast(
    hipblasHandle_t handle, int n, const double* alpha, hipblasDoubleComplex* x, int incx)
{
    return hipblasZdscal(handle, n, alpha, (hipDoubleComplex*)x, incx);
}

// scal_64
hipblasStatus_t hipblasCscalCast_64(
    hipblasHandle_t handle, int64_t n, const hipblasComplex* alpha, hipblasComplex* x, int64_t incx)
{
    return hipblasCscal_64(handle, n, (const hipComplex*)alpha, (hipComplex*)x, incx);
}

hipblasStatus_t hipblasCsscalCast_64(
    hipblasHandle_t handle, int64_t n, const float* alpha, hipblasComplex* x, int64_t incx)
{
    return hipblasCsscal_64(handle, n, alpha, (hipComplex*)x, incx);
}

hipblasStatus_t hipblasZscalCast_64(hipblasHandle_t             handle,
                                    int64_t                     n,
                                    const hipblasDoubleComplex* alpha,
                                    hipblasDoubleComplex*       x,
                                    int64_t                     incx)
{
    return hipblasZscal_64(handle, n, (const hipDoubleComplex*)alpha, (hipDoubleComplex*)x, incx);
}

hipblasStatus_t hipblasZdscalCast_64(
    hipblasHandle_t handle, int64_t n, const double* alpha, hipblasDoubleComplex* x, int64_t incx)
{
    return hipblasZdscal_64(handle, n, alpha, (hipDoubleComplex*)x, incx);
}

// batched
hipblasStatus_t hipblasCscalBatchedCast(hipblasHandle_t       handle,
                                        int                   n,
                                        const hipblasComplex* alpha,
                                        hipblasComplex* const x[],
                                        int                   incx,
                                        int                   batch_count)
{
    return hipblasCscalBatched(
        handle, n, (const hipComplex*)alpha, (hipComplex* const*)x, incx, batch_count);
}

hipblasStatus_t hipblasCsscalBatchedCast(hipblasHandle_t       handle,
                                         int                   n,
                                         const float*          alpha,
                                         hipblasComplex* const x[],
                                         int                   incx,
                                         int                   batch_count)
{
    return hipblasCsscalBatched(handle, n, alpha, (hipComplex* const*)x, incx, batch_count);
}

hipblasStatus_t hipblasZscalBatchedCast(hipblasHandle_t             handle,
                                        int                         n,
                                        const hipblasDoubleComplex* alpha,
                                        hipblasDoubleComplex* const x[],
                                        int                         incx,
                                        int                         batch_count)
{
    return hipblasZscalBatched(
        handle, n, (const hipDoubleComplex*)alpha, (hipDoubleComplex* const*)x, incx, batch_count);
}

hipblasStatus_t hipblasZdscalBatchedCast(hipblasHandle_t             handle,
                                         int                         n,
                                         const double*               alpha,
                                         hipblasDoubleComplex* const x[],
                                         int                         incx,
                                         int                         batch_count)
{
    return hipblasZdscalBatched(handle, n, alpha, (hipDoubleComplex* const*)x, incx, batch_count);
}

// batched_64
hipblasStatus_t hipblasCscalBatchedCast_64(hipblasHandle_t       handle,
                                           int64_t               n,
                                           const hipblasComplex* alpha,
                                           hipblasComplex* const x[],
                                           int64_t               incx,
                                           int64_t               batch_count)
{
    return hipblasCscalBatched_64(
        handle, n, (const hipComplex*)alpha, (hipComplex* const*)x, incx, batch_count);
}

hipblasStatus_t hipblasCsscalBatchedCast_64(hipblasHandle_t       handle,
                                            int64_t               n,
                                            const float*          alpha,
                                            hipblasComplex* const x[],
                                            int64_t               incx,
                                            int64_t               batch_count)
{
    return hipblasCsscalBatched_64(handle, n, alpha, (hipComplex* const*)x, incx, batch_count);
}

hipblasStatus_t hipblasZscalBatchedCast_64(hipblasHandle_t             handle,
                                           int64_t                     n,
                                           const hipblasDoubleComplex* alpha,
                                           hipblasDoubleComplex* const x[],
                                           int64_t                     incx,
                                           int64_t                     batch_count)
{
    return hipblasZscalBatched_64(
        handle, n, (const hipDoubleComplex*)alpha, (hipDoubleComplex* const*)x, incx, batch_count);
}

hipblasStatus_t hipblasZdscalBatchedCast_64(hipblasHandle_t             handle,
                                            int64_t                     n,
                                            const double*               alpha,
                                            hipblasDoubleComplex* const x[],
                                            int64_t                     incx,
                                            int64_t                     batch_count)
{
    return hipblasZdscalBatched_64(
        handle, n, alpha, (hipDoubleComplex* const*)x, incx, batch_count);
}

// strided_batched
hipblasStatus_t hipblasCscalStridedBatchedCast(hipblasHandle_t       handle,
                                               int                   n,
                                               const hipblasComplex* alpha,
                                               hipblasComplex*       x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               int                   batch_count)
{
    return hipblasCscalStridedBatched(
        handle, n, (const hipComplex*)alpha, (hipComplex*)x, incx, stridex, batch_count);
}

hipblasStatus_t hipblasCsscalStridedBatchedCast(hipblasHandle_t handle,
                                                int             n,
                                                const float*    alpha,
                                                hipblasComplex* x,
                                                int             incx,
                                                hipblasStride   stridex,
                                                int             batch_count)
{
    return hipblasCsscalStridedBatched(
        handle, n, alpha, (hipComplex*)x, incx, stridex, batch_count);
}

hipblasStatus_t hipblasZscalStridedBatchedCast(hipblasHandle_t             handle,
                                               int                         n,
                                               const hipblasDoubleComplex* alpha,
                                               hipblasDoubleComplex*       x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               int                         batch_count)
{
    return hipblasZscalStridedBatched(handle,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      batch_count);
}

hipblasStatus_t hipblasZdscalStridedBatchedCast(hipblasHandle_t       handle,
                                                int                   n,
                                                const double*         alpha,
                                                hipblasDoubleComplex* x,
                                                int                   incx,
                                                hipblasStride         stridex,
                                                int                   batch_count)
{
    return hipblasZdscalStridedBatched(
        handle, n, alpha, (hipDoubleComplex*)x, incx, stridex, batch_count);
}

// strided_batched_64
hipblasStatus_t hipblasCscalStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  int64_t               n,
                                                  const hipblasComplex* alpha,
                                                  hipblasComplex*       x,
                                                  int64_t               incx,
                                                  hipblasStride         stridex,
                                                  int64_t               batch_count)
{
    return hipblasCscalStridedBatched_64(
        handle, n, (const hipComplex*)alpha, (hipComplex*)x, incx, stridex, batch_count);
}

hipblasStatus_t hipblasCsscalStridedBatchedCast_64(hipblasHandle_t handle,
                                                   int64_t         n,
                                                   const float*    alpha,
                                                   hipblasComplex* x,
                                                   int64_t         incx,
                                                   hipblasStride   stridex,
                                                   int64_t         batch_count)
{
    return hipblasCsscalStridedBatched_64(
        handle, n, alpha, (hipComplex*)x, incx, stridex, batch_count);
}

hipblasStatus_t hipblasZscalStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  int64_t                     n,
                                                  const hipblasDoubleComplex* alpha,
                                                  hipblasDoubleComplex*       x,
                                                  int64_t                     incx,
                                                  hipblasStride               stridex,
                                                  int64_t                     batch_count)
{
    return hipblasZscalStridedBatched_64(handle,
                                         n,
                                         (const hipDoubleComplex*)alpha,
                                         (hipDoubleComplex*)x,
                                         incx,
                                         stridex,
                                         batch_count);
}

hipblasStatus_t hipblasZdscalStridedBatchedCast_64(hipblasHandle_t       handle,
                                                   int64_t               n,
                                                   const double*         alpha,
                                                   hipblasDoubleComplex* x,
                                                   int64_t               incx,
                                                   hipblasStride         stridex,
                                                   int64_t               batch_count)
{
    return hipblasZdscalStridedBatched_64(
        handle, n, alpha, (hipDoubleComplex*)x, incx, stridex, batch_count);
}

/*
 * ===========================================================================
 *    level 2 BLAS
 * ===========================================================================
 */

// gbmv
hipblasStatus_t hipblasCgbmvCast(hipblasHandle_t       handle,
                                 hipblasOperation_t    transA,
                                 int                   m,
                                 int                   n,
                                 int                   kl,
                                 int                   ku,
                                 const hipblasComplex* alpha,
                                 const hipblasComplex* A,
                                 int                   lda,
                                 const hipblasComplex* x,
                                 int                   incx,
                                 const hipblasComplex* beta,
                                 hipblasComplex*       y,
                                 int                   incy)
{
    return hipblasCgbmv(handle,
                        transA,
                        m,
                        n,
                        kl,
                        ku,
                        (const hipComplex*)alpha,
                        (const hipComplex*)A,
                        lda,
                        (const hipComplex*)x,
                        incx,
                        (const hipComplex*)beta,
                        (hipComplex*)y,
                        incy);
}

hipblasStatus_t hipblasZgbmvCast(hipblasHandle_t             handle,
                                 hipblasOperation_t          transA,
                                 int                         m,
                                 int                         n,
                                 int                         kl,
                                 int                         ku,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* A,
                                 int                         lda,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* beta,
                                 hipblasDoubleComplex*       y,
                                 int                         incy)
{
    return hipblasZgbmv(handle,
                        transA,
                        m,
                        n,
                        kl,
                        ku,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)A,
                        lda,
                        (const hipDoubleComplex*)x,
                        incx,
                        (const hipDoubleComplex*)beta,
                        (hipDoubleComplex*)y,
                        incy);
}

// gbmv_64
hipblasStatus_t hipblasCgbmvCast_64(hipblasHandle_t       handle,
                                    hipblasOperation_t    transA,
                                    int64_t               m,
                                    int64_t               n,
                                    int64_t               kl,
                                    int64_t               ku,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* A,
                                    int64_t               lda,
                                    const hipblasComplex* x,
                                    int64_t               incx,
                                    const hipblasComplex* beta,
                                    hipblasComplex*       y,
                                    int64_t               incy)
{
    return hipblasCgbmv_64(handle,
                           transA,
                           m,
                           n,
                           kl,
                           ku,
                           (const hipComplex*)alpha,
                           (const hipComplex*)A,
                           lda,
                           (const hipComplex*)x,
                           incx,
                           (const hipComplex*)beta,
                           (hipComplex*)y,
                           incy);
}

hipblasStatus_t hipblasZgbmvCast_64(hipblasHandle_t             handle,
                                    hipblasOperation_t          transA,
                                    int64_t                     m,
                                    int64_t                     n,
                                    int64_t                     kl,
                                    int64_t                     ku,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* A,
                                    int64_t                     lda,
                                    const hipblasDoubleComplex* x,
                                    int64_t                     incx,
                                    const hipblasDoubleComplex* beta,
                                    hipblasDoubleComplex*       y,
                                    int64_t                     incy)
{
    return hipblasZgbmv_64(handle,
                           transA,
                           m,
                           n,
                           kl,
                           ku,
                           (const hipDoubleComplex*)alpha,
                           (const hipDoubleComplex*)A,
                           lda,
                           (const hipDoubleComplex*)x,
                           incx,
                           (const hipDoubleComplex*)beta,
                           (hipDoubleComplex*)y,
                           incy);
}

// gbmv_batched
hipblasStatus_t hipblasCgbmvBatchedCast(hipblasHandle_t             handle,
                                        hipblasOperation_t          transA,
                                        int                         m,
                                        int                         n,
                                        int                         kl,
                                        int                         ku,
                                        const hipblasComplex*       alpha,
                                        const hipblasComplex* const A[],
                                        int                         lda,
                                        const hipblasComplex* const x[],
                                        int                         incx,
                                        const hipblasComplex*       beta,
                                        hipblasComplex* const       y[],
                                        int                         incy,
                                        int                         batch_count)
{
    return hipblasCgbmvBatched(handle,
                               transA,
                               m,
                               n,
                               kl,
                               ku,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)A,
                               lda,
                               (const hipComplex* const*)x,
                               incx,
                               (const hipComplex*)beta,
                               (hipComplex* const*)y,
                               incy,
                               batch_count);
}

hipblasStatus_t hipblasZgbmvBatchedCast(hipblasHandle_t                   handle,
                                        hipblasOperation_t                transA,
                                        int                               m,
                                        int                               n,
                                        int                               kl,
                                        int                               ku,
                                        const hipblasDoubleComplex*       alpha,
                                        const hipblasDoubleComplex* const A[],
                                        int                               lda,
                                        const hipblasDoubleComplex* const x[],
                                        int                               incx,
                                        const hipblasDoubleComplex*       beta,
                                        hipblasDoubleComplex* const       y[],
                                        int                               incy,
                                        int                               batch_count)
{
    return hipblasZgbmvBatched(handle,
                               transA,
                               m,
                               n,
                               kl,
                               ku,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (const hipDoubleComplex*)beta,
                               (hipDoubleComplex* const*)y,
                               incy,
                               batch_count);
}

// gbmv_batched_64
hipblasStatus_t hipblasCgbmvBatchedCast_64(hipblasHandle_t             handle,
                                           hipblasOperation_t          transA,
                                           int64_t                     m,
                                           int64_t                     n,
                                           int64_t                     kl,
                                           int64_t                     ku,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const A[],
                                           int64_t                     lda,
                                           const hipblasComplex* const x[],
                                           int64_t                     incx,
                                           const hipblasComplex*       beta,
                                           hipblasComplex* const       y[],
                                           int64_t                     incy,
                                           int64_t                     batch_count)
{
    return hipblasCgbmvBatched_64(handle,
                                  transA,
                                  m,
                                  n,
                                  kl,
                                  ku,
                                  (const hipComplex*)alpha,
                                  (const hipComplex* const*)A,
                                  lda,
                                  (const hipComplex* const*)x,
                                  incx,
                                  (const hipComplex*)beta,
                                  (hipComplex* const*)y,
                                  incy,
                                  batch_count);
}

hipblasStatus_t hipblasZgbmvBatchedCast_64(hipblasHandle_t                   handle,
                                           hipblasOperation_t                transA,
                                           int64_t                           m,
                                           int64_t                           n,
                                           int64_t                           kl,
                                           int64_t                           ku,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const A[],
                                           int64_t                           lda,
                                           const hipblasDoubleComplex* const x[],
                                           int64_t                           incx,
                                           const hipblasDoubleComplex*       beta,
                                           hipblasDoubleComplex* const       y[],
                                           int64_t                           incy,
                                           int64_t                           batch_count)
{
    return hipblasZgbmvBatched_64(handle,
                                  transA,
                                  m,
                                  n,
                                  kl,
                                  ku,
                                  (const hipDoubleComplex*)alpha,
                                  (const hipDoubleComplex* const*)A,
                                  lda,
                                  (const hipDoubleComplex* const*)x,
                                  incx,
                                  (const hipDoubleComplex*)beta,
                                  (hipDoubleComplex* const*)y,
                                  incy,
                                  batch_count);
}

// gbmv_strided_batched
hipblasStatus_t hipblasCgbmvStridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasOperation_t    transA,
                                               int                   m,
                                               int                   n,
                                               int                   kl,
                                               int                   ku,
                                               const hipblasComplex* alpha,
                                               const hipblasComplex* A,
                                               int                   lda,
                                               hipblasStride         stride_a,
                                               const hipblasComplex* x,
                                               int                   incx,
                                               hipblasStride         stride_x,
                                               const hipblasComplex* beta,
                                               hipblasComplex*       y,
                                               int                   incy,
                                               hipblasStride         stride_y,
                                               int                   batch_count)
{
    return hipblasCgbmvStridedBatched(handle,
                                      transA,
                                      m,
                                      n,
                                      kl,
                                      ku,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)A,
                                      lda,
                                      stride_a,
                                      (const hipComplex*)x,
                                      incx,
                                      stride_x,
                                      (const hipComplex*)beta,
                                      (hipComplex*)y,
                                      incy,
                                      stride_y,
                                      batch_count);
}

hipblasStatus_t hipblasZgbmvStridedBatchedCast(hipblasHandle_t             handle,
                                               hipblasOperation_t          transA,
                                               int                         m,
                                               int                         n,
                                               int                         kl,
                                               int                         ku,
                                               const hipblasDoubleComplex* alpha,
                                               const hipblasDoubleComplex* A,
                                               int                         lda,
                                               hipblasStride               stride_a,
                                               const hipblasDoubleComplex* x,
                                               int                         incx,
                                               hipblasStride               stride_x,
                                               const hipblasDoubleComplex* beta,
                                               hipblasDoubleComplex*       y,
                                               int                         incy,
                                               hipblasStride               stride_y,
                                               int                         batch_count)
{
    return hipblasZgbmvStridedBatched(handle,
                                      transA,
                                      m,
                                      n,
                                      kl,
                                      ku,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      stride_a,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stride_x,
                                      (const hipDoubleComplex*)beta,
                                      (hipDoubleComplex*)y,
                                      incy,
                                      stride_y,
                                      batch_count);
}

// gbmv_strided_batched_64
hipblasStatus_t hipblasCgbmvStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  hipblasOperation_t    transA,
                                                  int64_t               m,
                                                  int64_t               n,
                                                  int64_t               kl,
                                                  int64_t               ku,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* A,
                                                  int64_t               lda,
                                                  hipblasStride         stride_a,
                                                  const hipblasComplex* x,
                                                  int64_t               incx,
                                                  hipblasStride         stride_x,
                                                  const hipblasComplex* beta,
                                                  hipblasComplex*       y,
                                                  int64_t               incy,
                                                  hipblasStride         stride_y,
                                                  int64_t               batch_count)
{
    return hipblasCgbmvStridedBatched_64(handle,
                                         transA,
                                         m,
                                         n,
                                         kl,
                                         ku,
                                         (const hipComplex*)alpha,
                                         (const hipComplex*)A,
                                         lda,
                                         stride_a,
                                         (const hipComplex*)x,
                                         incx,
                                         stride_x,
                                         (const hipComplex*)beta,
                                         (hipComplex*)y,
                                         incy,
                                         stride_y,
                                         batch_count);
}

hipblasStatus_t hipblasZgbmvStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  hipblasOperation_t          transA,
                                                  int64_t                     m,
                                                  int64_t                     n,
                                                  int64_t                     kl,
                                                  int64_t                     ku,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* A,
                                                  int64_t                     lda,
                                                  hipblasStride               stride_a,
                                                  const hipblasDoubleComplex* x,
                                                  int64_t                     incx,
                                                  hipblasStride               stride_x,
                                                  const hipblasDoubleComplex* beta,
                                                  hipblasDoubleComplex*       y,
                                                  int64_t                     incy,
                                                  hipblasStride               stride_y,
                                                  int64_t                     batch_count)
{
    return hipblasZgbmvStridedBatched_64(handle,
                                         transA,
                                         m,
                                         n,
                                         kl,
                                         ku,
                                         (const hipDoubleComplex*)alpha,
                                         (const hipDoubleComplex*)A,
                                         lda,
                                         stride_a,
                                         (const hipDoubleComplex*)x,
                                         incx,
                                         stride_x,
                                         (const hipDoubleComplex*)beta,
                                         (hipDoubleComplex*)y,
                                         incy,
                                         stride_y,
                                         batch_count);
}

// gemv
hipblasStatus_t hipblasCgemvCast(hipblasHandle_t       handle,
                                 hipblasOperation_t    transA,
                                 int                   m,
                                 int                   n,
                                 const hipblasComplex* alpha,
                                 const hipblasComplex* A,
                                 int                   lda,
                                 const hipblasComplex* x,
                                 int                   incx,
                                 const hipblasComplex* beta,
                                 hipblasComplex*       y,
                                 int                   incy)
{
    return hipblasCgemv(handle,
                        transA,
                        m,
                        n,
                        (const hipComplex*)alpha,
                        (const hipComplex*)A,
                        lda,
                        (const hipComplex*)x,
                        incx,
                        (const hipComplex*)beta,
                        (hipComplex*)y,
                        incy);
}

hipblasStatus_t hipblasZgemvCast(hipblasHandle_t             handle,
                                 hipblasOperation_t          transA,
                                 int                         m,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* A,
                                 int                         lda,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* beta,
                                 hipblasDoubleComplex*       y,
                                 int                         incy)
{
    return hipblasZgemv(handle,
                        transA,
                        m,
                        n,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)A,
                        lda,
                        (const hipDoubleComplex*)x,
                        incx,
                        (const hipDoubleComplex*)beta,
                        (hipDoubleComplex*)y,
                        incy);
}

// gemv_64
hipblasStatus_t hipblasCgemvCast_64(hipblasHandle_t       handle,
                                    hipblasOperation_t    transA,
                                    int64_t               m,
                                    int64_t               n,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* A,
                                    int64_t               lda,
                                    const hipblasComplex* x,
                                    int64_t               incx,
                                    const hipblasComplex* beta,
                                    hipblasComplex*       y,
                                    int64_t               incy)
{
    return hipblasCgemv_64(handle,
                           transA,
                           m,
                           n,
                           (const hipComplex*)alpha,
                           (const hipComplex*)A,
                           lda,
                           (const hipComplex*)x,
                           incx,
                           (const hipComplex*)beta,
                           (hipComplex*)y,
                           incy);
}

hipblasStatus_t hipblasZgemvCast_64(hipblasHandle_t             handle,
                                    hipblasOperation_t          transA,
                                    int64_t                     m,
                                    int64_t                     n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* A,
                                    int64_t                     lda,
                                    const hipblasDoubleComplex* x,
                                    int64_t                     incx,
                                    const hipblasDoubleComplex* beta,
                                    hipblasDoubleComplex*       y,
                                    int64_t                     incy)
{
    return hipblasZgemv_64(handle,
                           transA,
                           m,
                           n,
                           (const hipDoubleComplex*)alpha,
                           (const hipDoubleComplex*)A,
                           lda,
                           (const hipDoubleComplex*)x,
                           incx,
                           (const hipDoubleComplex*)beta,
                           (hipDoubleComplex*)y,
                           incy);
}

// gemv_batched
hipblasStatus_t hipblasCgemvBatchedCast(hipblasHandle_t             handle,
                                        hipblasOperation_t          transA,
                                        int                         m,
                                        int                         n,
                                        const hipblasComplex*       alpha,
                                        const hipblasComplex* const A[],
                                        int                         lda,
                                        const hipblasComplex* const x[],
                                        int                         incx,
                                        const hipblasComplex*       beta,
                                        hipblasComplex* const       y[],
                                        int                         incy,
                                        int                         batch_count)
{
    return hipblasCgemvBatched(handle,
                               transA,
                               m,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)A,
                               lda,
                               (const hipComplex* const*)x,
                               incx,
                               (const hipComplex*)beta,
                               (hipComplex* const*)y,
                               incy,
                               batch_count);
}

hipblasStatus_t hipblasZgemvBatchedCast(hipblasHandle_t                   handle,
                                        hipblasOperation_t                transA,
                                        int                               m,
                                        int                               n,
                                        const hipblasDoubleComplex*       alpha,
                                        const hipblasDoubleComplex* const A[],
                                        int                               lda,
                                        const hipblasDoubleComplex* const x[],
                                        int                               incx,
                                        const hipblasDoubleComplex*       beta,
                                        hipblasDoubleComplex* const       y[],
                                        int                               incy,
                                        int                               batch_count)
{
    return hipblasZgemvBatched(handle,
                               transA,
                               m,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (const hipDoubleComplex*)beta,
                               (hipDoubleComplex* const*)y,
                               incy,
                               batch_count);
}

// gemv_batched_64
hipblasStatus_t hipblasCgemvBatchedCast_64(hipblasHandle_t             handle,
                                           hipblasOperation_t          transA,
                                           int64_t                     m,
                                           int64_t                     n,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const A[],
                                           int64_t                     lda,
                                           const hipblasComplex* const x[],
                                           int64_t                     incx,
                                           const hipblasComplex*       beta,
                                           hipblasComplex* const       y[],
                                           int64_t                     incy,
                                           int64_t                     batch_count)
{
    return hipblasCgemvBatched_64(handle,
                                  transA,
                                  m,
                                  n,
                                  (const hipComplex*)alpha,
                                  (const hipComplex* const*)A,
                                  lda,
                                  (const hipComplex* const*)x,
                                  incx,
                                  (const hipComplex*)beta,
                                  (hipComplex* const*)y,
                                  incy,
                                  batch_count);
}

hipblasStatus_t hipblasZgemvBatchedCast_64(hipblasHandle_t                   handle,
                                           hipblasOperation_t                transA,
                                           int64_t                           m,
                                           int64_t                           n,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const A[],
                                           int64_t                           lda,
                                           const hipblasDoubleComplex* const x[],
                                           int64_t                           incx,
                                           const hipblasDoubleComplex*       beta,
                                           hipblasDoubleComplex* const       y[],
                                           int64_t                           incy,
                                           int64_t                           batch_count)
{
    return hipblasZgemvBatched_64(handle,
                                  transA,
                                  m,
                                  n,
                                  (const hipDoubleComplex*)alpha,
                                  (const hipDoubleComplex* const*)A,
                                  lda,
                                  (const hipDoubleComplex* const*)x,
                                  incx,
                                  (const hipDoubleComplex*)beta,
                                  (hipDoubleComplex* const*)y,
                                  incy,
                                  batch_count);
}

// gemv_strided_batched
hipblasStatus_t hipblasCgemvStridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasOperation_t    transA,
                                               int                   m,
                                               int                   n,
                                               const hipblasComplex* alpha,
                                               const hipblasComplex* A,
                                               int                   lda,
                                               hipblasStride         strideA,
                                               const hipblasComplex* x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               const hipblasComplex* beta,
                                               hipblasComplex*       y,
                                               int                   incy,
                                               hipblasStride         stridey,
                                               int                   batch_count)
{
    return hipblasCgemvStridedBatched(handle,
                                      transA,
                                      m,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipComplex*)beta,
                                      (hipComplex*)y,
                                      incy,
                                      stridey,
                                      batch_count);
}

hipblasStatus_t hipblasZgemvStridedBatchedCast(hipblasHandle_t             handle,
                                               hipblasOperation_t          transA,
                                               int                         m,
                                               int                         n,
                                               const hipblasDoubleComplex* alpha,
                                               const hipblasDoubleComplex* A,
                                               int                         lda,
                                               hipblasStride               strideA,
                                               const hipblasDoubleComplex* x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               const hipblasDoubleComplex* beta,
                                               hipblasDoubleComplex*       y,
                                               int                         incy,
                                               hipblasStride               stridey,
                                               int                         batch_count)
{
    return hipblasZgemvStridedBatched(handle,
                                      transA,
                                      m,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipDoubleComplex*)beta,
                                      (hipDoubleComplex*)y,
                                      incy,
                                      stridey,
                                      batch_count);
}

// gemv_strided_batched_64
hipblasStatus_t hipblasCgemvStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  hipblasOperation_t    transA,
                                                  int64_t               m,
                                                  int64_t               n,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* A,
                                                  int64_t               lda,
                                                  hipblasStride         strideA,
                                                  const hipblasComplex* x,
                                                  int64_t               incx,
                                                  hipblasStride         stridex,
                                                  const hipblasComplex* beta,
                                                  hipblasComplex*       y,
                                                  int64_t               incy,
                                                  hipblasStride         stridey,
                                                  int64_t               batch_count)
{
    return hipblasCgemvStridedBatched_64(handle,
                                         transA,
                                         m,
                                         n,
                                         (const hipComplex*)alpha,
                                         (const hipComplex*)A,
                                         lda,
                                         strideA,
                                         (const hipComplex*)x,
                                         incx,
                                         stridex,
                                         (const hipComplex*)beta,
                                         (hipComplex*)y,
                                         incy,
                                         stridey,
                                         batch_count);
}

hipblasStatus_t hipblasZgemvStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  hipblasOperation_t          transA,
                                                  int64_t                     m,
                                                  int64_t                     n,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* A,
                                                  int64_t                     lda,
                                                  hipblasStride               strideA,
                                                  const hipblasDoubleComplex* x,
                                                  int64_t                     incx,
                                                  hipblasStride               stridex,
                                                  const hipblasDoubleComplex* beta,
                                                  hipblasDoubleComplex*       y,
                                                  int64_t                     incy,
                                                  hipblasStride               stridey,
                                                  int64_t                     batch_count)
{
    return hipblasZgemvStridedBatched_64(handle,
                                         transA,
                                         m,
                                         n,
                                         (const hipDoubleComplex*)alpha,
                                         (const hipDoubleComplex*)A,
                                         lda,
                                         strideA,
                                         (const hipDoubleComplex*)x,
                                         incx,
                                         stridex,
                                         (const hipDoubleComplex*)beta,
                                         (hipDoubleComplex*)y,
                                         incy,
                                         stridey,
                                         batch_count);
}

// ger
hipblasStatus_t hipblasCgeruCast(hipblasHandle_t       handle,
                                 int                   m,
                                 int                   n,
                                 const hipblasComplex* alpha,
                                 const hipblasComplex* x,
                                 int                   incx,
                                 const hipblasComplex* y,
                                 int                   incy,
                                 hipblasComplex*       A,
                                 int                   lda)
{
    return hipblasCgeru(handle,
                        m,
                        n,
                        (const hipComplex*)alpha,
                        (const hipComplex*)x,
                        incx,
                        (const hipComplex*)y,
                        incy,
                        (hipComplex*)A,
                        lda);
}

hipblasStatus_t hipblasCgercCast(hipblasHandle_t       handle,
                                 int                   m,
                                 int                   n,
                                 const hipblasComplex* alpha,
                                 const hipblasComplex* x,
                                 int                   incx,
                                 const hipblasComplex* y,
                                 int                   incy,
                                 hipblasComplex*       A,
                                 int                   lda)
{
    return hipblasCgerc(handle,
                        m,
                        n,
                        (const hipComplex*)alpha,
                        (const hipComplex*)x,
                        incx,
                        (const hipComplex*)y,
                        incy,
                        (hipComplex*)A,
                        lda);
}

hipblasStatus_t hipblasZgeruCast(hipblasHandle_t             handle,
                                 int                         m,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* y,
                                 int                         incy,
                                 hipblasDoubleComplex*       A,
                                 int                         lda)
{
    return hipblasZgeru(handle,
                        m,
                        n,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)x,
                        incx,
                        (const hipDoubleComplex*)y,
                        incy,
                        (hipDoubleComplex*)A,
                        lda);
}

hipblasStatus_t hipblasZgercCast(hipblasHandle_t             handle,
                                 int                         m,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* y,
                                 int                         incy,
                                 hipblasDoubleComplex*       A,
                                 int                         lda)
{
    return hipblasZgerc(handle,
                        m,
                        n,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)x,
                        incx,
                        (const hipDoubleComplex*)y,
                        incy,
                        (hipDoubleComplex*)A,
                        lda);
}

// ger_64
hipblasStatus_t hipblasCgeruCast_64(hipblasHandle_t       handle,
                                    int64_t               m,
                                    int64_t               n,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* x,
                                    int64_t               incx,
                                    const hipblasComplex* y,
                                    int64_t               incy,
                                    hipblasComplex*       A,
                                    int64_t               lda)
{
    return hipblasCgeru_64(handle,
                           m,
                           n,
                           (const hipComplex*)alpha,
                           (const hipComplex*)x,
                           incx,
                           (const hipComplex*)y,
                           incy,
                           (hipComplex*)A,
                           lda);
}

hipblasStatus_t hipblasCgercCast_64(hipblasHandle_t       handle,
                                    int64_t               m,
                                    int64_t               n,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* x,
                                    int64_t               incx,
                                    const hipblasComplex* y,
                                    int64_t               incy,
                                    hipblasComplex*       A,
                                    int64_t               lda)
{
    return hipblasCgerc_64(handle,
                           m,
                           n,
                           (const hipComplex*)alpha,
                           (const hipComplex*)x,
                           incx,
                           (const hipComplex*)y,
                           incy,
                           (hipComplex*)A,
                           lda);
}

hipblasStatus_t hipblasZgeruCast_64(hipblasHandle_t             handle,
                                    int64_t                     m,
                                    int64_t                     n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* x,
                                    int64_t                     incx,
                                    const hipblasDoubleComplex* y,
                                    int64_t                     incy,
                                    hipblasDoubleComplex*       A,
                                    int64_t                     lda)
{
    return hipblasZgeru_64(handle,
                           m,
                           n,
                           (const hipDoubleComplex*)alpha,
                           (const hipDoubleComplex*)x,
                           incx,
                           (const hipDoubleComplex*)y,
                           incy,
                           (hipDoubleComplex*)A,
                           lda);
}

hipblasStatus_t hipblasZgercCast_64(hipblasHandle_t             handle,
                                    int64_t                     m,
                                    int64_t                     n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* x,
                                    int64_t                     incx,
                                    const hipblasDoubleComplex* y,
                                    int64_t                     incy,
                                    hipblasDoubleComplex*       A,
                                    int64_t                     lda)
{
    return hipblasZgerc_64(handle,
                           m,
                           n,
                           (const hipDoubleComplex*)alpha,
                           (const hipDoubleComplex*)x,
                           incx,
                           (const hipDoubleComplex*)y,
                           incy,
                           (hipDoubleComplex*)A,
                           lda);
}

// ger_batched
hipblasStatus_t hipblasCgeruBatchedCast(hipblasHandle_t             handle,
                                        int                         m,
                                        int                         n,
                                        const hipblasComplex*       alpha,
                                        const hipblasComplex* const x[],
                                        int                         incx,
                                        const hipblasComplex* const y[],
                                        int                         incy,
                                        hipblasComplex* const       A[],
                                        int                         lda,
                                        int                         batch_count)
{
    return hipblasCgeruBatched(handle,
                               m,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)x,
                               incx,
                               (const hipComplex* const*)y,
                               incy,
                               (hipComplex* const*)A,
                               lda,
                               batch_count);
}

hipblasStatus_t hipblasCgercBatchedCast(hipblasHandle_t             handle,
                                        int                         m,
                                        int                         n,
                                        const hipblasComplex*       alpha,
                                        const hipblasComplex* const x[],
                                        int                         incx,
                                        const hipblasComplex* const y[],
                                        int                         incy,
                                        hipblasComplex* const       A[],
                                        int                         lda,
                                        int                         batch_count)
{
    return hipblasCgercBatched(handle,
                               m,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)x,
                               incx,
                               (const hipComplex* const*)y,
                               incy,
                               (hipComplex* const*)A,
                               lda,
                               batch_count);
}

hipblasStatus_t hipblasZgeruBatchedCast(hipblasHandle_t                   handle,
                                        int                               m,
                                        int                               n,
                                        const hipblasDoubleComplex*       alpha,
                                        const hipblasDoubleComplex* const x[],
                                        int                               incx,
                                        const hipblasDoubleComplex* const y[],
                                        int                               incy,
                                        hipblasDoubleComplex* const       A[],
                                        int                               lda,
                                        int                               batch_count)
{
    return hipblasZgeruBatched(handle,
                               m,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (const hipDoubleComplex* const*)y,
                               incy,
                               (hipDoubleComplex* const*)A,
                               lda,
                               batch_count);
}

hipblasStatus_t hipblasZgercBatchedCast(hipblasHandle_t                   handle,
                                        int                               m,
                                        int                               n,
                                        const hipblasDoubleComplex*       alpha,
                                        const hipblasDoubleComplex* const x[],
                                        int                               incx,
                                        const hipblasDoubleComplex* const y[],
                                        int                               incy,
                                        hipblasDoubleComplex* const       A[],
                                        int                               lda,
                                        int                               batch_count)
{
    return hipblasZgercBatched(handle,
                               m,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (const hipDoubleComplex* const*)y,
                               incy,
                               (hipDoubleComplex* const*)A,
                               lda,
                               batch_count);
}

// ger_batched_64
hipblasStatus_t hipblasCgeruBatchedCast_64(hipblasHandle_t             handle,
                                           int64_t                     m,
                                           int64_t                     n,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const x[],
                                           int64_t                     incx,
                                           const hipblasComplex* const y[],
                                           int64_t                     incy,
                                           hipblasComplex* const       A[],
                                           int64_t                     lda,
                                           int64_t                     batch_count)
{
    return hipblasCgeruBatched_64(handle,
                                  m,
                                  n,
                                  (const hipComplex*)alpha,
                                  (const hipComplex* const*)x,
                                  incx,
                                  (const hipComplex* const*)y,
                                  incy,
                                  (hipComplex* const*)A,
                                  lda,
                                  batch_count);
}

hipblasStatus_t hipblasCgercBatchedCast_64(hipblasHandle_t             handle,
                                           int64_t                     m,
                                           int64_t                     n,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const x[],
                                           int64_t                     incx,
                                           const hipblasComplex* const y[],
                                           int64_t                     incy,
                                           hipblasComplex* const       A[],
                                           int64_t                     lda,
                                           int64_t                     batch_count)
{
    return hipblasCgercBatched_64(handle,
                                  m,
                                  n,
                                  (const hipComplex*)alpha,
                                  (const hipComplex* const*)x,
                                  incx,
                                  (const hipComplex* const*)y,
                                  incy,
                                  (hipComplex* const*)A,
                                  lda,
                                  batch_count);
}

hipblasStatus_t hipblasZgeruBatchedCast_64(hipblasHandle_t                   handle,
                                           int64_t                           m,
                                           int64_t                           n,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const x[],
                                           int64_t                           incx,
                                           const hipblasDoubleComplex* const y[],
                                           int64_t                           incy,
                                           hipblasDoubleComplex* const       A[],
                                           int64_t                           lda,
                                           int64_t                           batch_count)
{
    return hipblasZgeruBatched_64(handle,
                                  m,
                                  n,
                                  (const hipDoubleComplex*)alpha,
                                  (const hipDoubleComplex* const*)x,
                                  incx,
                                  (const hipDoubleComplex* const*)y,
                                  incy,
                                  (hipDoubleComplex* const*)A,
                                  lda,
                                  batch_count);
}

hipblasStatus_t hipblasZgercBatchedCast_64(hipblasHandle_t                   handle,
                                           int64_t                           m,
                                           int64_t                           n,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const x[],
                                           int64_t                           incx,
                                           const hipblasDoubleComplex* const y[],
                                           int64_t                           incy,
                                           hipblasDoubleComplex* const       A[],
                                           int64_t                           lda,
                                           int64_t                           batch_count)
{
    return hipblasZgercBatched_64(handle,
                                  m,
                                  n,
                                  (const hipDoubleComplex*)alpha,
                                  (const hipDoubleComplex* const*)x,
                                  incx,
                                  (const hipDoubleComplex* const*)y,
                                  incy,
                                  (hipDoubleComplex* const*)A,
                                  lda,
                                  batch_count);
}

// ger_strided_batched
hipblasStatus_t hipblasCgeruStridedBatchedCast(hipblasHandle_t       handle,
                                               int                   m,
                                               int                   n,
                                               const hipblasComplex* alpha,
                                               const hipblasComplex* x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               const hipblasComplex* y,
                                               int                   incy,
                                               hipblasStride         stridey,
                                               hipblasComplex*       A,
                                               int                   lda,
                                               hipblasStride         strideA,
                                               int                   batch_count)
{
    return hipblasCgeruStridedBatched(handle,
                                      m,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipComplex*)y,
                                      incy,
                                      stridey,
                                      (hipComplex*)A,
                                      lda,
                                      strideA,
                                      batch_count);
}

hipblasStatus_t hipblasCgercStridedBatchedCast(hipblasHandle_t       handle,
                                               int                   m,
                                               int                   n,
                                               const hipblasComplex* alpha,
                                               const hipblasComplex* x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               const hipblasComplex* y,
                                               int                   incy,
                                               hipblasStride         stridey,
                                               hipblasComplex*       A,
                                               int                   lda,
                                               hipblasStride         strideA,
                                               int                   batch_count)
{
    return hipblasCgercStridedBatched(handle,
                                      m,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipComplex*)y,
                                      incy,
                                      stridey,
                                      (hipComplex*)A,
                                      lda,
                                      strideA,
                                      batch_count);
}

hipblasStatus_t hipblasZgeruStridedBatchedCast(hipblasHandle_t             handle,
                                               int                         m,
                                               int                         n,
                                               const hipblasDoubleComplex* alpha,
                                               const hipblasDoubleComplex* x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               const hipblasDoubleComplex* y,
                                               int                         incy,
                                               hipblasStride               stridey,
                                               hipblasDoubleComplex*       A,
                                               int                         lda,
                                               hipblasStride               strideA,
                                               int                         batch_count)
{
    return hipblasZgeruStridedBatched(handle,
                                      m,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipDoubleComplex*)y,
                                      incy,
                                      stridey,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      batch_count);
}

hipblasStatus_t hipblasZgercStridedBatchedCast(hipblasHandle_t             handle,
                                               int                         m,
                                               int                         n,
                                               const hipblasDoubleComplex* alpha,
                                               const hipblasDoubleComplex* x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               const hipblasDoubleComplex* y,
                                               int                         incy,
                                               hipblasStride               stridey,
                                               hipblasDoubleComplex*       A,
                                               int                         lda,
                                               hipblasStride               strideA,
                                               int                         batch_count)
{
    return hipblasZgercStridedBatched(handle,
                                      m,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipDoubleComplex*)y,
                                      incy,
                                      stridey,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      batch_count);
}

// ger_strided_batched_64
hipblasStatus_t hipblasCgeruStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  int64_t               m,
                                                  int64_t               n,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* x,
                                                  int64_t               incx,
                                                  hipblasStride         stridex,
                                                  const hipblasComplex* y,
                                                  int64_t               incy,
                                                  hipblasStride         stridey,
                                                  hipblasComplex*       A,
                                                  int64_t               lda,
                                                  hipblasStride         strideA,
                                                  int64_t               batch_count)
{
    return hipblasCgeruStridedBatched_64(handle,
                                         m,
                                         n,
                                         (const hipComplex*)alpha,
                                         (const hipComplex*)x,
                                         incx,
                                         stridex,
                                         (const hipComplex*)y,
                                         incy,
                                         stridey,
                                         (hipComplex*)A,
                                         lda,
                                         strideA,
                                         batch_count);
}

hipblasStatus_t hipblasCgercStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  int64_t               m,
                                                  int64_t               n,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* x,
                                                  int64_t               incx,
                                                  hipblasStride         stridex,
                                                  const hipblasComplex* y,
                                                  int64_t               incy,
                                                  hipblasStride         stridey,
                                                  hipblasComplex*       A,
                                                  int64_t               lda,
                                                  hipblasStride         strideA,
                                                  int64_t               batch_count)
{
    return hipblasCgercStridedBatched_64(handle,
                                         m,
                                         n,
                                         (const hipComplex*)alpha,
                                         (const hipComplex*)x,
                                         incx,
                                         stridex,
                                         (const hipComplex*)y,
                                         incy,
                                         stridey,
                                         (hipComplex*)A,
                                         lda,
                                         strideA,
                                         batch_count);
}

hipblasStatus_t hipblasZgeruStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  int64_t                     m,
                                                  int64_t                     n,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* x,
                                                  int64_t                     incx,
                                                  hipblasStride               stridex,
                                                  const hipblasDoubleComplex* y,
                                                  int64_t                     incy,
                                                  hipblasStride               stridey,
                                                  hipblasDoubleComplex*       A,
                                                  int64_t                     lda,
                                                  hipblasStride               strideA,
                                                  int64_t                     batch_count)
{
    return hipblasZgeruStridedBatched_64(handle,
                                         m,
                                         n,
                                         (const hipDoubleComplex*)alpha,
                                         (const hipDoubleComplex*)x,
                                         incx,
                                         stridex,
                                         (const hipDoubleComplex*)y,
                                         incy,
                                         stridey,
                                         (hipDoubleComplex*)A,
                                         lda,
                                         strideA,
                                         batch_count);
}

hipblasStatus_t hipblasZgercStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  int64_t                     m,
                                                  int64_t                     n,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* x,
                                                  int64_t                     incx,
                                                  hipblasStride               stridex,
                                                  const hipblasDoubleComplex* y,
                                                  int64_t                     incy,
                                                  hipblasStride               stridey,
                                                  hipblasDoubleComplex*       A,
                                                  int64_t                     lda,
                                                  hipblasStride               strideA,
                                                  int64_t                     batch_count)
{
    return hipblasZgercStridedBatched_64(handle,
                                         m,
                                         n,
                                         (const hipDoubleComplex*)alpha,
                                         (const hipDoubleComplex*)x,
                                         incx,
                                         stridex,
                                         (const hipDoubleComplex*)y,
                                         incy,
                                         stridey,
                                         (hipDoubleComplex*)A,
                                         lda,
                                         strideA,
                                         batch_count);
}

// hbmv
hipblasStatus_t hipblasChbmvCast(hipblasHandle_t       handle,
                                 hipblasFillMode_t     uplo,
                                 int                   n,
                                 int                   k,
                                 const hipblasComplex* alpha,
                                 const hipblasComplex* A,
                                 int                   lda,
                                 const hipblasComplex* x,
                                 int                   incx,
                                 const hipblasComplex* beta,
                                 hipblasComplex*       y,
                                 int                   incy)
{
    return hipblasChbmv(handle,
                        uplo,
                        n,
                        k,
                        (const hipComplex*)alpha,
                        (const hipComplex*)A,
                        lda,
                        (const hipComplex*)x,
                        incx,
                        (const hipComplex*)beta,
                        (hipComplex*)y,
                        incy);
}

hipblasStatus_t hipblasZhbmvCast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 int                         n,
                                 int                         k,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* A,
                                 int                         lda,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* beta,
                                 hipblasDoubleComplex*       y,
                                 int                         incy)
{
    return hipblasZhbmv(handle,
                        uplo,
                        n,
                        k,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)A,
                        lda,
                        (const hipDoubleComplex*)x,
                        incx,
                        (const hipDoubleComplex*)beta,
                        (hipDoubleComplex*)y,
                        incy);
}

// hbmv_64
hipblasStatus_t hipblasChbmvCast_64(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    int64_t               n,
                                    int64_t               k,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* A,
                                    int64_t               lda,
                                    const hipblasComplex* x,
                                    int64_t               incx,
                                    const hipblasComplex* beta,
                                    hipblasComplex*       y,
                                    int64_t               incy)
{
    return hipblasChbmv_64(handle,
                           uplo,
                           n,
                           k,
                           (const hipComplex*)alpha,
                           (const hipComplex*)A,
                           lda,
                           (const hipComplex*)x,
                           incx,
                           (const hipComplex*)beta,
                           (hipComplex*)y,
                           incy);
}

hipblasStatus_t hipblasZhbmvCast_64(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int64_t                     n,
                                    int64_t                     k,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* A,
                                    int64_t                     lda,
                                    const hipblasDoubleComplex* x,
                                    int64_t                     incx,
                                    const hipblasDoubleComplex* beta,
                                    hipblasDoubleComplex*       y,
                                    int64_t                     incy)
{
    return hipblasZhbmv_64(handle,
                           uplo,
                           n,
                           k,
                           (const hipDoubleComplex*)alpha,
                           (const hipDoubleComplex*)A,
                           lda,
                           (const hipDoubleComplex*)x,
                           incx,
                           (const hipDoubleComplex*)beta,
                           (hipDoubleComplex*)y,
                           incy);
}

// hbmv_batched
hipblasStatus_t hipblasChbmvBatchedCast(hipblasHandle_t             handle,
                                        hipblasFillMode_t           uplo,
                                        int                         n,
                                        int                         k,
                                        const hipblasComplex*       alpha,
                                        const hipblasComplex* const A[],
                                        int                         lda,
                                        const hipblasComplex* const x[],
                                        int                         incx,
                                        const hipblasComplex*       beta,
                                        hipblasComplex* const       y[],
                                        int                         incy,
                                        int                         batchCount)
{
    return hipblasChbmvBatched(handle,
                               uplo,
                               n,
                               k,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)A,
                               lda,
                               (const hipComplex* const*)x,
                               incx,
                               (const hipComplex*)beta,
                               (hipComplex* const*)y,
                               incy,
                               batchCount);
}

hipblasStatus_t hipblasZhbmvBatchedCast(hipblasHandle_t                   handle,
                                        hipblasFillMode_t                 uplo,
                                        int                               n,
                                        int                               k,
                                        const hipblasDoubleComplex*       alpha,
                                        const hipblasDoubleComplex* const A[],
                                        int                               lda,
                                        const hipblasDoubleComplex* const x[],
                                        int                               incx,
                                        const hipblasDoubleComplex*       beta,
                                        hipblasDoubleComplex* const       y[],
                                        int                               incy,
                                        int                               batchCount)
{
    return hipblasZhbmvBatched(handle,
                               uplo,
                               n,
                               k,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (const hipDoubleComplex*)beta,
                               (hipDoubleComplex* const*)y,
                               incy,
                               batchCount);
}

// hbmv_batched_64
hipblasStatus_t hipblasChbmvBatchedCast_64(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int64_t                     n,
                                           int64_t                     k,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const A[],
                                           int64_t                     lda,
                                           const hipblasComplex* const x[],
                                           int64_t                     incx,
                                           const hipblasComplex*       beta,
                                           hipblasComplex* const       y[],
                                           int64_t                     incy,
                                           int64_t                     batchCount)
{
    return hipblasChbmvBatched_64(handle,
                                  uplo,
                                  n,
                                  k,
                                  (const hipComplex*)alpha,
                                  (const hipComplex* const*)A,
                                  lda,
                                  (const hipComplex* const*)x,
                                  incx,
                                  (const hipComplex*)beta,
                                  (hipComplex* const*)y,
                                  incy,
                                  batchCount);
}

hipblasStatus_t hipblasZhbmvBatchedCast_64(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           int64_t                           n,
                                           int64_t                           k,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const A[],
                                           int64_t                           lda,
                                           const hipblasDoubleComplex* const x[],
                                           int64_t                           incx,
                                           const hipblasDoubleComplex*       beta,
                                           hipblasDoubleComplex* const       y[],
                                           int64_t                           incy,
                                           int64_t                           batchCount)
{
    return hipblasZhbmvBatched_64(handle,
                                  uplo,
                                  n,
                                  k,
                                  (const hipDoubleComplex*)alpha,
                                  (const hipDoubleComplex* const*)A,
                                  lda,
                                  (const hipDoubleComplex* const*)x,
                                  incx,
                                  (const hipDoubleComplex*)beta,
                                  (hipDoubleComplex* const*)y,
                                  incy,
                                  batchCount);
}

// hbmv_strided_batched
hipblasStatus_t hipblasChbmvStridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasFillMode_t     uplo,
                                               int                   n,
                                               int                   k,
                                               const hipblasComplex* alpha,
                                               const hipblasComplex* A,
                                               int                   lda,
                                               hipblasStride         strideA,
                                               const hipblasComplex* x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               const hipblasComplex* beta,
                                               hipblasComplex*       y,
                                               int                   incy,
                                               hipblasStride         stridey,
                                               int                   batchCount)
{
    return hipblasChbmvStridedBatched(handle,
                                      uplo,
                                      n,
                                      k,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipComplex*)beta,
                                      (hipComplex*)y,
                                      incy,
                                      stridey,
                                      batchCount);
}

hipblasStatus_t hipblasZhbmvStridedBatchedCast(hipblasHandle_t             handle,
                                               hipblasFillMode_t           uplo,
                                               int                         n,
                                               int                         k,
                                               const hipblasDoubleComplex* alpha,
                                               const hipblasDoubleComplex* A,
                                               int                         lda,
                                               hipblasStride               strideA,
                                               const hipblasDoubleComplex* x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               const hipblasDoubleComplex* beta,
                                               hipblasDoubleComplex*       y,
                                               int                         incy,
                                               hipblasStride               stridey,
                                               int                         batchCount)
{
    return hipblasZhbmvStridedBatched(handle,
                                      uplo,
                                      n,
                                      k,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipDoubleComplex*)beta,
                                      (hipDoubleComplex*)y,
                                      incy,
                                      stridey,
                                      batchCount);
}

// hbmv_strided_batched_64
hipblasStatus_t hipblasChbmvStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  int64_t               n,
                                                  int64_t               k,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* A,
                                                  int64_t               lda,
                                                  hipblasStride         strideA,
                                                  const hipblasComplex* x,
                                                  int64_t               incx,
                                                  hipblasStride         stridex,
                                                  const hipblasComplex* beta,
                                                  hipblasComplex*       y,
                                                  int64_t               incy,
                                                  hipblasStride         stridey,
                                                  int64_t               batchCount)
{
    return hipblasChbmvStridedBatched_64(handle,
                                         uplo,
                                         n,
                                         k,
                                         (const hipComplex*)alpha,
                                         (const hipComplex*)A,
                                         lda,
                                         strideA,
                                         (const hipComplex*)x,
                                         incx,
                                         stridex,
                                         (const hipComplex*)beta,
                                         (hipComplex*)y,
                                         incy,
                                         stridey,
                                         batchCount);
}

hipblasStatus_t hipblasZhbmvStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  int64_t                     n,
                                                  int64_t                     k,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* A,
                                                  int64_t                     lda,
                                                  hipblasStride               strideA,
                                                  const hipblasDoubleComplex* x,
                                                  int64_t                     incx,
                                                  hipblasStride               stridex,
                                                  const hipblasDoubleComplex* beta,
                                                  hipblasDoubleComplex*       y,
                                                  int64_t                     incy,
                                                  hipblasStride               stridey,
                                                  int64_t                     batchCount)
{
    return hipblasZhbmvStridedBatched_64(handle,
                                         uplo,
                                         n,
                                         k,
                                         (const hipDoubleComplex*)alpha,
                                         (const hipDoubleComplex*)A,
                                         lda,
                                         strideA,
                                         (const hipDoubleComplex*)x,
                                         incx,
                                         stridex,
                                         (const hipDoubleComplex*)beta,
                                         (hipDoubleComplex*)y,
                                         incy,
                                         stridey,
                                         batchCount);
}

// hemv
hipblasStatus_t hipblasChemvCast(hipblasHandle_t       handle,
                                 hipblasFillMode_t     uplo,
                                 int                   n,
                                 const hipblasComplex* alpha,
                                 const hipblasComplex* A,
                                 int                   lda,
                                 const hipblasComplex* x,
                                 int                   incx,
                                 const hipblasComplex* beta,
                                 hipblasComplex*       y,
                                 int                   incy)
{
    return hipblasChemv(handle,
                        uplo,
                        n,
                        (const hipComplex*)alpha,
                        (const hipComplex*)A,
                        lda,
                        (const hipComplex*)x,
                        incx,
                        (const hipComplex*)beta,
                        (hipComplex*)y,
                        incy);
}

hipblasStatus_t hipblasZhemvCast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* A,
                                 int                         lda,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* beta,
                                 hipblasDoubleComplex*       y,
                                 int                         incy)
{
    return hipblasZhemv(handle,
                        uplo,
                        n,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)A,
                        lda,
                        (const hipDoubleComplex*)x,
                        incx,
                        (const hipDoubleComplex*)beta,
                        (hipDoubleComplex*)y,
                        incy);
}

// hemv_64
hipblasStatus_t hipblasChemvCast_64(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    int64_t               n,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* A,
                                    int64_t               lda,
                                    const hipblasComplex* x,
                                    int64_t               incx,
                                    const hipblasComplex* beta,
                                    hipblasComplex*       y,
                                    int64_t               incy)
{
    return hipblasChemv_64(handle,
                           uplo,
                           n,
                           (const hipComplex*)alpha,
                           (const hipComplex*)A,
                           lda,
                           (const hipComplex*)x,
                           incx,
                           (const hipComplex*)beta,
                           (hipComplex*)y,
                           incy);
}

hipblasStatus_t hipblasZhemvCast_64(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int64_t                     n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* A,
                                    int64_t                     lda,
                                    const hipblasDoubleComplex* x,
                                    int64_t                     incx,
                                    const hipblasDoubleComplex* beta,
                                    hipblasDoubleComplex*       y,
                                    int64_t                     incy)
{
    return hipblasZhemv_64(handle,
                           uplo,
                           n,
                           (const hipDoubleComplex*)alpha,
                           (const hipDoubleComplex*)A,
                           lda,
                           (const hipDoubleComplex*)x,
                           incx,
                           (const hipDoubleComplex*)beta,
                           (hipDoubleComplex*)y,
                           incy);
}

// hemv_batched
hipblasStatus_t hipblasChemvBatchedCast(hipblasHandle_t             handle,
                                        hipblasFillMode_t           uplo,
                                        int                         n,
                                        const hipblasComplex*       alpha,
                                        const hipblasComplex* const A[],
                                        int                         lda,
                                        const hipblasComplex* const x[],
                                        int                         incx,
                                        const hipblasComplex*       beta,
                                        hipblasComplex* const       y[],
                                        int                         incy,
                                        int                         batch_count)
{
    return hipblasChemvBatched(handle,
                               uplo,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)A,
                               lda,
                               (const hipComplex* const*)x,
                               incx,
                               (const hipComplex*)beta,
                               (hipComplex* const*)y,
                               incy,
                               batch_count);
}

hipblasStatus_t hipblasZhemvBatchedCast(hipblasHandle_t                   handle,
                                        hipblasFillMode_t                 uplo,
                                        int                               n,
                                        const hipblasDoubleComplex*       alpha,
                                        const hipblasDoubleComplex* const A[],
                                        int                               lda,
                                        const hipblasDoubleComplex* const x[],
                                        int                               incx,
                                        const hipblasDoubleComplex*       beta,
                                        hipblasDoubleComplex* const       y[],
                                        int                               incy,
                                        int                               batch_count)
{
    return hipblasZhemvBatched(handle,
                               uplo,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (const hipDoubleComplex*)beta,
                               (hipDoubleComplex* const*)y,
                               incy,
                               batch_count);
}

// hemv_batched_64
hipblasStatus_t hipblasChemvBatchedCast_64(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int64_t                     n,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const A[],
                                           int64_t                     lda,
                                           const hipblasComplex* const x[],
                                           int64_t                     incx,
                                           const hipblasComplex*       beta,
                                           hipblasComplex* const       y[],
                                           int64_t                     incy,
                                           int64_t                     batch_count)
{
    return hipblasChemvBatched_64(handle,
                                  uplo,
                                  n,
                                  (const hipComplex*)alpha,
                                  (const hipComplex* const*)A,
                                  lda,
                                  (const hipComplex* const*)x,
                                  incx,
                                  (const hipComplex*)beta,
                                  (hipComplex* const*)y,
                                  incy,
                                  batch_count);
}

hipblasStatus_t hipblasZhemvBatchedCast_64(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           int64_t                           n,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const A[],
                                           int64_t                           lda,
                                           const hipblasDoubleComplex* const x[],
                                           int64_t                           incx,
                                           const hipblasDoubleComplex*       beta,
                                           hipblasDoubleComplex* const       y[],
                                           int64_t                           incy,
                                           int64_t                           batch_count)
{
    return hipblasZhemvBatched_64(handle,
                                  uplo,
                                  n,
                                  (const hipDoubleComplex*)alpha,
                                  (const hipDoubleComplex* const*)A,
                                  lda,
                                  (const hipDoubleComplex* const*)x,
                                  incx,
                                  (const hipDoubleComplex*)beta,
                                  (hipDoubleComplex* const*)y,
                                  incy,
                                  batch_count);
}

// hemv_strided_batched
hipblasStatus_t hipblasChemvStridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasFillMode_t     uplo,
                                               int                   n,
                                               const hipblasComplex* alpha,
                                               const hipblasComplex* A,
                                               int                   lda,
                                               hipblasStride         stride_a,
                                               const hipblasComplex* x,
                                               int                   incx,
                                               hipblasStride         stride_x,
                                               const hipblasComplex* beta,
                                               hipblasComplex*       y,
                                               int                   incy,
                                               hipblasStride         stride_y,
                                               int                   batch_count)
{
    return hipblasChemvStridedBatched(handle,
                                      uplo,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)A,
                                      lda,
                                      stride_a,
                                      (const hipComplex*)x,
                                      incx,
                                      stride_x,
                                      (const hipComplex*)beta,
                                      (hipComplex*)y,
                                      incy,
                                      stride_y,
                                      batch_count);
}

hipblasStatus_t hipblasZhemvStridedBatchedCast(hipblasHandle_t             handle,
                                               hipblasFillMode_t           uplo,
                                               int                         n,
                                               const hipblasDoubleComplex* alpha,
                                               const hipblasDoubleComplex* A,
                                               int                         lda,
                                               hipblasStride               stride_a,
                                               const hipblasDoubleComplex* x,
                                               int                         incx,
                                               hipblasStride               stride_x,
                                               const hipblasDoubleComplex* beta,
                                               hipblasDoubleComplex*       y,
                                               int                         incy,
                                               hipblasStride               stride_y,
                                               int                         batch_count)
{
    return hipblasZhemvStridedBatched(handle,
                                      uplo,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      stride_a,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stride_x,
                                      (const hipDoubleComplex*)beta,
                                      (hipDoubleComplex*)y,
                                      incy,
                                      stride_y,
                                      batch_count);
}

// hemv_strided_batched_64
hipblasStatus_t hipblasChemvStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  int64_t               n,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* A,
                                                  int64_t               lda,
                                                  hipblasStride         stride_a,
                                                  const hipblasComplex* x,
                                                  int64_t               incx,
                                                  hipblasStride         stride_x,
                                                  const hipblasComplex* beta,
                                                  hipblasComplex*       y,
                                                  int64_t               incy,
                                                  hipblasStride         stride_y,
                                                  int64_t               batch_count)
{
    return hipblasChemvStridedBatched_64(handle,
                                         uplo,
                                         n,
                                         (const hipComplex*)alpha,
                                         (const hipComplex*)A,
                                         lda,
                                         stride_a,
                                         (const hipComplex*)x,
                                         incx,
                                         stride_x,
                                         (const hipComplex*)beta,
                                         (hipComplex*)y,
                                         incy,
                                         stride_y,
                                         batch_count);
}

hipblasStatus_t hipblasZhemvStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  int64_t                     n,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* A,
                                                  int64_t                     lda,
                                                  hipblasStride               stride_a,
                                                  const hipblasDoubleComplex* x,
                                                  int64_t                     incx,
                                                  hipblasStride               stride_x,
                                                  const hipblasDoubleComplex* beta,
                                                  hipblasDoubleComplex*       y,
                                                  int64_t                     incy,
                                                  hipblasStride               stride_y,
                                                  int64_t                     batch_count)
{
    return hipblasZhemvStridedBatched_64(handle,
                                         uplo,
                                         n,
                                         (const hipDoubleComplex*)alpha,
                                         (const hipDoubleComplex*)A,
                                         lda,
                                         stride_a,
                                         (const hipDoubleComplex*)x,
                                         incx,
                                         stride_x,
                                         (const hipDoubleComplex*)beta,
                                         (hipDoubleComplex*)y,
                                         incy,
                                         stride_y,
                                         batch_count);
}

// her
hipblasStatus_t hipblasCherCast(hipblasHandle_t       handle,
                                hipblasFillMode_t     uplo,
                                int                   n,
                                const float*          alpha,
                                const hipblasComplex* x,
                                int                   incx,
                                hipblasComplex*       A,
                                int                   lda)
{
    return hipblasCher(handle, uplo, n, alpha, (const hipComplex*)x, incx, (hipComplex*)A, lda);
}

hipblasStatus_t hipblasZherCast(hipblasHandle_t             handle,
                                hipblasFillMode_t           uplo,
                                int                         n,
                                const double*               alpha,
                                const hipblasDoubleComplex* x,
                                int                         incx,
                                hipblasDoubleComplex*       A,
                                int                         lda)
{
    return hipblasZher(
        handle, uplo, n, alpha, (const hipDoubleComplex*)x, incx, (hipDoubleComplex*)A, lda);
}

// her_batched
hipblasStatus_t hipblasCherBatchedCast(hipblasHandle_t             handle,
                                       hipblasFillMode_t           uplo,
                                       int                         n,
                                       const float*                alpha,
                                       const hipblasComplex* const x[],
                                       int                         incx,
                                       hipblasComplex* const       A[],
                                       int                         lda,
                                       int                         batchCount)
{
    return hipblasCherBatched(handle,
                              uplo,
                              n,
                              alpha,
                              (const hipComplex* const*)x,
                              incx,
                              (hipComplex* const*)A,
                              lda,
                              batchCount);
}

hipblasStatus_t hipblasZherBatchedCast(hipblasHandle_t                   handle,
                                       hipblasFillMode_t                 uplo,
                                       int                               n,
                                       const double*                     alpha,
                                       const hipblasDoubleComplex* const x[],
                                       int                               incx,
                                       hipblasDoubleComplex* const       A[],
                                       int                               lda,
                                       int                               batchCount)
{
    return hipblasZherBatched(handle,
                              uplo,
                              n,
                              alpha,
                              (const hipDoubleComplex* const*)x,
                              incx,
                              (hipDoubleComplex* const*)A,
                              lda,
                              batchCount);
}

// her_strided_batched
hipblasStatus_t hipblasCherStridedBatchedCast(hipblasHandle_t       handle,
                                              hipblasFillMode_t     uplo,
                                              int                   n,
                                              const float*          alpha,
                                              const hipblasComplex* x,
                                              int                   incx,
                                              hipblasStride         stridex,
                                              hipblasComplex*       A,
                                              int                   lda,
                                              hipblasStride         strideA,
                                              int                   batchCount)
{
    return hipblasCherStridedBatched(handle,
                                     uplo,
                                     n,
                                     alpha,
                                     (const hipComplex*)x,
                                     incx,
                                     stridex,
                                     (hipComplex*)A,
                                     lda,
                                     strideA,
                                     batchCount);
}

hipblasStatus_t hipblasZherStridedBatchedCast(hipblasHandle_t             handle,
                                              hipblasFillMode_t           uplo,
                                              int                         n,
                                              const double*               alpha,
                                              const hipblasDoubleComplex* x,
                                              int                         incx,
                                              hipblasStride               stridex,
                                              hipblasDoubleComplex*       A,
                                              int                         lda,
                                              hipblasStride               strideA,
                                              int                         batchCount)
{
    return hipblasZherStridedBatched(handle,
                                     uplo,
                                     n,
                                     alpha,
                                     (const hipDoubleComplex*)x,
                                     incx,
                                     stridex,
                                     (hipDoubleComplex*)A,
                                     lda,
                                     strideA,
                                     batchCount);
}

// her_64
hipblasStatus_t hipblasCherCast_64(hipblasHandle_t       handle,
                                   hipblasFillMode_t     uplo,
                                   int64_t               n,
                                   const float*          alpha,
                                   const hipblasComplex* x,
                                   int64_t               incx,
                                   hipblasComplex*       A,
                                   int64_t               lda)
{
    return hipblasCher_64(handle, uplo, n, alpha, (const hipComplex*)x, incx, (hipComplex*)A, lda);
}

hipblasStatus_t hipblasZherCast_64(hipblasHandle_t             handle,
                                   hipblasFillMode_t           uplo,
                                   int64_t                     n,
                                   const double*               alpha,
                                   const hipblasDoubleComplex* x,
                                   int64_t                     incx,
                                   hipblasDoubleComplex*       A,
                                   int64_t                     lda)
{
    return hipblasZher_64(
        handle, uplo, n, alpha, (const hipDoubleComplex*)x, incx, (hipDoubleComplex*)A, lda);
}

// her_batched_64
hipblasStatus_t hipblasCherBatchedCast_64(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          int64_t                     n,
                                          const float*                alpha,
                                          const hipblasComplex* const x[],
                                          int64_t                     incx,
                                          hipblasComplex* const       A[],
                                          int64_t                     lda,
                                          int64_t                     batchCount)
{
    return hipblasCherBatched_64(handle,
                                 uplo,
                                 n,
                                 alpha,
                                 (const hipComplex* const*)x,
                                 incx,
                                 (hipComplex* const*)A,
                                 lda,
                                 batchCount);
}

hipblasStatus_t hipblasZherBatchedCast_64(hipblasHandle_t                   handle,
                                          hipblasFillMode_t                 uplo,
                                          int64_t                           n,
                                          const double*                     alpha,
                                          const hipblasDoubleComplex* const x[],
                                          int64_t                           incx,
                                          hipblasDoubleComplex* const       A[],
                                          int64_t                           lda,
                                          int64_t                           batchCount)
{
    return hipblasZherBatched_64(handle,
                                 uplo,
                                 n,
                                 alpha,
                                 (const hipDoubleComplex* const*)x,
                                 incx,
                                 (hipDoubleComplex* const*)A,
                                 lda,
                                 batchCount);
}

// her_strided_batched_64
hipblasStatus_t hipblasCherStridedBatchedCast_64(hipblasHandle_t       handle,
                                                 hipblasFillMode_t     uplo,
                                                 int64_t               n,
                                                 const float*          alpha,
                                                 const hipblasComplex* x,
                                                 int64_t               incx,
                                                 hipblasStride         stridex,
                                                 hipblasComplex*       A,
                                                 int64_t               lda,
                                                 hipblasStride         strideA,
                                                 int64_t               batchCount)
{
    return hipblasCherStridedBatched_64(handle,
                                        uplo,
                                        n,
                                        alpha,
                                        (const hipComplex*)x,
                                        incx,
                                        stridex,
                                        (hipComplex*)A,
                                        lda,
                                        strideA,
                                        batchCount);
}

hipblasStatus_t hipblasZherStridedBatchedCast_64(hipblasHandle_t             handle,
                                                 hipblasFillMode_t           uplo,
                                                 int64_t                     n,
                                                 const double*               alpha,
                                                 const hipblasDoubleComplex* x,
                                                 int64_t                     incx,
                                                 hipblasStride               stridex,
                                                 hipblasDoubleComplex*       A,
                                                 int64_t                     lda,
                                                 hipblasStride               strideA,
                                                 int64_t                     batchCount)
{
    return hipblasZherStridedBatched_64(handle,
                                        uplo,
                                        n,
                                        alpha,
                                        (const hipDoubleComplex*)x,
                                        incx,
                                        stridex,
                                        (hipDoubleComplex*)A,
                                        lda,
                                        strideA,
                                        batchCount);
}

// her2
hipblasStatus_t hipblasCher2Cast(hipblasHandle_t       handle,
                                 hipblasFillMode_t     uplo,
                                 int                   n,
                                 const hipblasComplex* alpha,
                                 const hipblasComplex* x,
                                 int                   incx,
                                 const hipblasComplex* y,
                                 int                   incy,
                                 hipblasComplex*       A,
                                 int                   lda)
{
    return hipblasCher2(handle,
                        uplo,
                        n,
                        (const hipComplex*)alpha,
                        (const hipComplex*)x,
                        incx,
                        (const hipComplex*)y,
                        incy,
                        (hipComplex*)A,
                        lda);
}

hipblasStatus_t hipblasZher2Cast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* y,
                                 int                         incy,
                                 hipblasDoubleComplex*       A,
                                 int                         lda)
{
    return hipblasZher2(handle,
                        uplo,
                        n,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)x,
                        incx,
                        (const hipDoubleComplex*)y,
                        incy,
                        (hipDoubleComplex*)A,
                        lda);
}

// her2_batched
hipblasStatus_t hipblasCher2BatchedCast(hipblasHandle_t             handle,
                                        hipblasFillMode_t           uplo,
                                        int                         n,
                                        const hipblasComplex*       alpha,
                                        const hipblasComplex* const x[],
                                        int                         incx,
                                        const hipblasComplex* const y[],
                                        int                         incy,
                                        hipblasComplex* const       A[],
                                        int                         lda,
                                        int                         batchCount)
{
    return hipblasCher2Batched(handle,
                               uplo,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)x,
                               incx,
                               (const hipComplex* const*)y,
                               incy,
                               (hipComplex* const*)A,
                               lda,
                               batchCount);
}

hipblasStatus_t hipblasZher2BatchedCast(hipblasHandle_t                   handle,
                                        hipblasFillMode_t                 uplo,
                                        int                               n,
                                        const hipblasDoubleComplex*       alpha,
                                        const hipblasDoubleComplex* const x[],
                                        int                               incx,
                                        const hipblasDoubleComplex* const y[],
                                        int                               incy,
                                        hipblasDoubleComplex* const       A[],
                                        int                               lda,
                                        int                               batchCount)
{
    return hipblasZher2Batched(handle,
                               uplo,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (const hipDoubleComplex* const*)y,
                               incy,
                               (hipDoubleComplex* const*)A,
                               lda,
                               batchCount);
}

// her2_strided_batched
hipblasStatus_t hipblasCher2StridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasFillMode_t     uplo,
                                               int                   n,
                                               const hipblasComplex* alpha,
                                               const hipblasComplex* x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               const hipblasComplex* y,
                                               int                   incy,
                                               hipblasStride         stridey,
                                               hipblasComplex*       A,
                                               int                   lda,
                                               hipblasStride         strideA,
                                               int                   batchCount)
{
    return hipblasCher2StridedBatched(handle,
                                      uplo,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipComplex*)y,
                                      incy,
                                      stridey,
                                      (hipComplex*)A,
                                      lda,
                                      strideA,
                                      batchCount);
}

hipblasStatus_t hipblasZher2StridedBatchedCast(hipblasHandle_t             handle,
                                               hipblasFillMode_t           uplo,
                                               int                         n,
                                               const hipblasDoubleComplex* alpha,
                                               const hipblasDoubleComplex* x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               const hipblasDoubleComplex* y,
                                               int                         incy,
                                               hipblasStride               stridey,
                                               hipblasDoubleComplex*       A,
                                               int                         lda,
                                               hipblasStride               strideA,
                                               int                         batchCount)
{
    return hipblasZher2StridedBatched(handle,
                                      uplo,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipDoubleComplex*)y,
                                      incy,
                                      stridey,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      batchCount);
}

// her2_64
hipblasStatus_t hipblasCher2Cast_64(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    int64_t               n,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* x,
                                    int64_t               incx,
                                    const hipblasComplex* y,
                                    int64_t               incy,
                                    hipblasComplex*       A,
                                    int64_t               lda)
{
    return hipblasCher2_64(handle,
                           uplo,
                           n,
                           (const hipComplex*)alpha,
                           (const hipComplex*)x,
                           incx,
                           (const hipComplex*)y,
                           incy,
                           (hipComplex*)A,
                           lda);
}

hipblasStatus_t hipblasZher2Cast_64(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int64_t                     n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* x,
                                    int64_t                     incx,
                                    const hipblasDoubleComplex* y,
                                    int64_t                     incy,
                                    hipblasDoubleComplex*       A,
                                    int64_t                     lda)
{
    return hipblasZher2_64(handle,
                           uplo,
                           n,
                           (const hipDoubleComplex*)alpha,
                           (const hipDoubleComplex*)x,
                           incx,
                           (const hipDoubleComplex*)y,
                           incy,
                           (hipDoubleComplex*)A,
                           lda);
}

// her2_batched_64
hipblasStatus_t hipblasCher2BatchedCast_64(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int64_t                     n,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const x[],
                                           int64_t                     incx,
                                           const hipblasComplex* const y[],
                                           int64_t                     incy,
                                           hipblasComplex* const       A[],
                                           int64_t                     lda,
                                           int64_t                     batchCount)
{
    return hipblasCher2Batched_64(handle,
                                  uplo,
                                  n,
                                  (const hipComplex*)alpha,
                                  (const hipComplex* const*)x,
                                  incx,
                                  (const hipComplex* const*)y,
                                  incy,
                                  (hipComplex* const*)A,
                                  lda,
                                  batchCount);
}

hipblasStatus_t hipblasZher2BatchedCast_64(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           int64_t                           n,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const x[],
                                           int64_t                           incx,
                                           const hipblasDoubleComplex* const y[],
                                           int64_t                           incy,
                                           hipblasDoubleComplex* const       A[],
                                           int64_t                           lda,
                                           int64_t                           batchCount)
{
    return hipblasZher2Batched_64(handle,
                                  uplo,
                                  n,
                                  (const hipDoubleComplex*)alpha,
                                  (const hipDoubleComplex* const*)x,
                                  incx,
                                  (const hipDoubleComplex* const*)y,
                                  incy,
                                  (hipDoubleComplex* const*)A,
                                  lda,
                                  batchCount);
}

// her2_strided_batched_64
hipblasStatus_t hipblasCher2StridedBatchedCast_64(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  int64_t               n,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* x,
                                                  int64_t               incx,
                                                  hipblasStride         stridex,
                                                  const hipblasComplex* y,
                                                  int64_t               incy,
                                                  hipblasStride         stridey,
                                                  hipblasComplex*       A,
                                                  int64_t               lda,
                                                  hipblasStride         strideA,
                                                  int64_t               batchCount)
{
    return hipblasCher2StridedBatched_64(handle,
                                         uplo,
                                         n,
                                         (const hipComplex*)alpha,
                                         (const hipComplex*)x,
                                         incx,
                                         stridex,
                                         (const hipComplex*)y,
                                         incy,
                                         stridey,
                                         (hipComplex*)A,
                                         lda,
                                         strideA,
                                         batchCount);
}

hipblasStatus_t hipblasZher2StridedBatchedCast_64(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  int64_t                     n,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* x,
                                                  int64_t                     incx,
                                                  hipblasStride               stridex,
                                                  const hipblasDoubleComplex* y,
                                                  int64_t                     incy,
                                                  hipblasStride               stridey,
                                                  hipblasDoubleComplex*       A,
                                                  int64_t                     lda,
                                                  hipblasStride               strideA,
                                                  int64_t                     batchCount)
{
    return hipblasZher2StridedBatched_64(handle,
                                         uplo,
                                         n,
                                         (const hipDoubleComplex*)alpha,
                                         (const hipDoubleComplex*)x,
                                         incx,
                                         stridex,
                                         (const hipDoubleComplex*)y,
                                         incy,
                                         stridey,
                                         (hipDoubleComplex*)A,
                                         lda,
                                         strideA,
                                         batchCount);
}

// hpmv
hipblasStatus_t hipblasChpmvCast(hipblasHandle_t       handle,
                                 hipblasFillMode_t     uplo,
                                 int                   n,
                                 const hipblasComplex* alpha,
                                 const hipblasComplex* AP,
                                 const hipblasComplex* x,
                                 int                   incx,
                                 const hipblasComplex* beta,
                                 hipblasComplex*       y,
                                 int                   incy)
{
    return hipblasChpmv(handle,
                        uplo,
                        n,
                        (const hipComplex*)alpha,
                        (const hipComplex*)AP,
                        (const hipComplex*)x,
                        incx,
                        (const hipComplex*)beta,
                        (hipComplex*)y,
                        incy);
}

hipblasStatus_t hipblasZhpmvCast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* AP,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* beta,
                                 hipblasDoubleComplex*       y,
                                 int                         incy)
{
    return hipblasZhpmv(handle,
                        uplo,
                        n,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)AP,
                        (const hipDoubleComplex*)x,
                        incx,
                        (const hipDoubleComplex*)beta,
                        (hipDoubleComplex*)y,
                        incy);
}

// hpmv_64
hipblasStatus_t hipblasChpmvCast_64(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    int64_t               n,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* AP,
                                    const hipblasComplex* x,
                                    int64_t               incx,
                                    const hipblasComplex* beta,
                                    hipblasComplex*       y,
                                    int64_t               incy)
{
    return hipblasChpmv_64(handle,
                           uplo,
                           n,
                           (const hipComplex*)alpha,
                           (const hipComplex*)AP,
                           (const hipComplex*)x,
                           incx,
                           (const hipComplex*)beta,
                           (hipComplex*)y,
                           incy);
}

hipblasStatus_t hipblasZhpmvCast_64(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int64_t                     n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* AP,
                                    const hipblasDoubleComplex* x,
                                    int64_t                     incx,
                                    const hipblasDoubleComplex* beta,
                                    hipblasDoubleComplex*       y,
                                    int64_t                     incy)
{
    return hipblasZhpmv_64(handle,
                           uplo,
                           n,
                           (const hipDoubleComplex*)alpha,
                           (const hipDoubleComplex*)AP,
                           (const hipDoubleComplex*)x,
                           incx,
                           (const hipDoubleComplex*)beta,
                           (hipDoubleComplex*)y,
                           incy);
}

// hpmv_batched
hipblasStatus_t hipblasChpmvBatchedCast(hipblasHandle_t             handle,
                                        hipblasFillMode_t           uplo,
                                        int                         n,
                                        const hipblasComplex*       alpha,
                                        const hipblasComplex* const AP[],
                                        const hipblasComplex* const x[],
                                        int                         incx,
                                        const hipblasComplex*       beta,
                                        hipblasComplex* const       y[],
                                        int                         incy,
                                        int                         batchCount)
{
    return hipblasChpmvBatched(handle,
                               uplo,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)AP,
                               (const hipComplex* const*)x,
                               incx,
                               (const hipComplex*)beta,
                               (hipComplex* const*)y,
                               incy,
                               batchCount);
}

hipblasStatus_t hipblasZhpmvBatchedCast(hipblasHandle_t                   handle,
                                        hipblasFillMode_t                 uplo,
                                        int                               n,
                                        const hipblasDoubleComplex*       alpha,
                                        const hipblasDoubleComplex* const AP[],
                                        const hipblasDoubleComplex* const x[],
                                        int                               incx,
                                        const hipblasDoubleComplex*       beta,
                                        hipblasDoubleComplex* const       y[],
                                        int                               incy,
                                        int                               batchCount)
{
    return hipblasZhpmvBatched(handle,
                               uplo,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)AP,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (const hipDoubleComplex*)beta,
                               (hipDoubleComplex* const*)y,
                               incy,
                               batchCount);
}

// hpmv_batched_64
hipblasStatus_t hipblasChpmvBatchedCast_64(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int64_t                     n,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const AP[],
                                           const hipblasComplex* const x[],
                                           int64_t                     incx,
                                           const hipblasComplex*       beta,
                                           hipblasComplex* const       y[],
                                           int64_t                     incy,
                                           int64_t                     batchCount)
{
    return hipblasChpmvBatched_64(handle,
                                  uplo,
                                  n,
                                  (const hipComplex*)alpha,
                                  (const hipComplex* const*)AP,
                                  (const hipComplex* const*)x,
                                  incx,
                                  (const hipComplex*)beta,
                                  (hipComplex* const*)y,
                                  incy,
                                  batchCount);
}

hipblasStatus_t hipblasZhpmvBatchedCast_64(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           int64_t                           n,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const AP[],
                                           const hipblasDoubleComplex* const x[],
                                           int64_t                           incx,
                                           const hipblasDoubleComplex*       beta,
                                           hipblasDoubleComplex* const       y[],
                                           int64_t                           incy,
                                           int64_t                           batchCount)
{
    return hipblasZhpmvBatched_64(handle,
                                  uplo,
                                  n,
                                  (const hipDoubleComplex*)alpha,
                                  (const hipDoubleComplex* const*)AP,
                                  (const hipDoubleComplex* const*)x,
                                  incx,
                                  (const hipDoubleComplex*)beta,
                                  (hipDoubleComplex* const*)y,
                                  incy,
                                  batchCount);
}

// hpmv_strided_batched
hipblasStatus_t hipblasChpmvStridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasFillMode_t     uplo,
                                               int                   n,
                                               const hipblasComplex* alpha,
                                               const hipblasComplex* AP,
                                               hipblasStride         strideAP,
                                               const hipblasComplex* x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               const hipblasComplex* beta,
                                               hipblasComplex*       y,
                                               int                   incy,
                                               hipblasStride         stridey,
                                               int                   batchCount)
{
    return hipblasChpmvStridedBatched(handle,
                                      uplo,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)AP,
                                      strideAP,
                                      (const hipComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipComplex*)beta,
                                      (hipComplex*)y,
                                      incy,
                                      stridey,
                                      batchCount);
}

hipblasStatus_t hipblasZhpmvStridedBatchedCast(hipblasHandle_t             handle,
                                               hipblasFillMode_t           uplo,
                                               int                         n,
                                               const hipblasDoubleComplex* alpha,
                                               const hipblasDoubleComplex* AP,
                                               hipblasStride               strideAP,
                                               const hipblasDoubleComplex* x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               const hipblasDoubleComplex* beta,
                                               hipblasDoubleComplex*       y,
                                               int                         incy,
                                               hipblasStride               stridey,
                                               int                         batchCount)
{
    return hipblasZhpmvStridedBatched(handle,
                                      uplo,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)AP,
                                      strideAP,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipDoubleComplex*)beta,
                                      (hipDoubleComplex*)y,
                                      incy,
                                      stridey,
                                      batchCount);
}

// hpmv_strided_batched_64
hipblasStatus_t hipblasChpmvStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  int64_t               n,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* AP,
                                                  hipblasStride         strideAP,
                                                  const hipblasComplex* x,
                                                  int64_t               incx,
                                                  hipblasStride         stridex,
                                                  const hipblasComplex* beta,
                                                  hipblasComplex*       y,
                                                  int64_t               incy,
                                                  hipblasStride         stridey,
                                                  int64_t               batchCount)
{
    return hipblasChpmvStridedBatched_64(handle,
                                         uplo,
                                         n,
                                         (const hipComplex*)alpha,
                                         (const hipComplex*)AP,
                                         strideAP,
                                         (const hipComplex*)x,
                                         incx,
                                         stridex,
                                         (const hipComplex*)beta,
                                         (hipComplex*)y,
                                         incy,
                                         stridey,
                                         batchCount);
}

hipblasStatus_t hipblasZhpmvStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  int64_t                     n,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* AP,
                                                  hipblasStride               strideAP,
                                                  const hipblasDoubleComplex* x,
                                                  int64_t                     incx,
                                                  hipblasStride               stridex,
                                                  const hipblasDoubleComplex* beta,
                                                  hipblasDoubleComplex*       y,
                                                  int64_t                     incy,
                                                  hipblasStride               stridey,
                                                  int64_t                     batchCount)
{
    return hipblasZhpmvStridedBatched_64(handle,
                                         uplo,
                                         n,
                                         (const hipDoubleComplex*)alpha,
                                         (const hipDoubleComplex*)AP,
                                         strideAP,
                                         (const hipDoubleComplex*)x,
                                         incx,
                                         stridex,
                                         (const hipDoubleComplex*)beta,
                                         (hipDoubleComplex*)y,
                                         incy,
                                         stridey,
                                         batchCount);
}

// hpr
hipblasStatus_t hipblasChprCast(hipblasHandle_t       handle,
                                hipblasFillMode_t     uplo,
                                int                   n,
                                const float*          alpha,
                                const hipblasComplex* x,
                                int                   incx,
                                hipblasComplex*       AP)
{
    return hipblasChpr(handle, uplo, n, alpha, (const hipComplex*)x, incx, (hipComplex*)AP);
}

hipblasStatus_t hipblasZhprCast(hipblasHandle_t             handle,
                                hipblasFillMode_t           uplo,
                                int                         n,
                                const double*               alpha,
                                const hipblasDoubleComplex* x,
                                int                         incx,
                                hipblasDoubleComplex*       AP)
{
    return hipblasZhpr(
        handle, uplo, n, alpha, (const hipDoubleComplex*)x, incx, (hipDoubleComplex*)AP);
}

// hpr_64
hipblasStatus_t hipblasChprCast_64(hipblasHandle_t       handle,
                                   hipblasFillMode_t     uplo,
                                   int64_t               n,
                                   const float*          alpha,
                                   const hipblasComplex* x,
                                   int64_t               incx,
                                   hipblasComplex*       AP)
{
    return hipblasChpr_64(handle, uplo, n, alpha, (const hipComplex*)x, incx, (hipComplex*)AP);
}

hipblasStatus_t hipblasZhprCast_64(hipblasHandle_t             handle,
                                   hipblasFillMode_t           uplo,
                                   int64_t                     n,
                                   const double*               alpha,
                                   const hipblasDoubleComplex* x,
                                   int64_t                     incx,
                                   hipblasDoubleComplex*       AP)
{
    return hipblasZhpr_64(
        handle, uplo, n, alpha, (const hipDoubleComplex*)x, incx, (hipDoubleComplex*)AP);
}

// hpr_batched
hipblasStatus_t hipblasChprBatchedCast(hipblasHandle_t             handle,
                                       hipblasFillMode_t           uplo,
                                       int                         n,
                                       const float*                alpha,
                                       const hipblasComplex* const x[],
                                       int                         incx,
                                       hipblasComplex* const       AP[],
                                       int                         batchCount)
{
    return hipblasChprBatched(handle,
                              uplo,
                              n,
                              alpha,
                              (const hipComplex* const*)x,
                              incx,
                              (hipComplex* const*)AP,
                              batchCount);
}

hipblasStatus_t hipblasZhprBatchedCast(hipblasHandle_t                   handle,
                                       hipblasFillMode_t                 uplo,
                                       int                               n,
                                       const double*                     alpha,
                                       const hipblasDoubleComplex* const x[],
                                       int                               incx,
                                       hipblasDoubleComplex* const       AP[],
                                       int                               batchCount)
{
    return hipblasZhprBatched(handle,
                              uplo,
                              n,
                              alpha,
                              (const hipDoubleComplex* const*)x,
                              incx,
                              (hipDoubleComplex* const*)AP,
                              batchCount);
}

// hpr_batched_64
hipblasStatus_t hipblasChprBatchedCast_64(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          int64_t                     n,
                                          const float*                alpha,
                                          const hipblasComplex* const x[],
                                          int64_t                     incx,
                                          hipblasComplex* const       AP[],
                                          int64_t                     batchCount)
{
    return hipblasChprBatched_64(handle,
                                 uplo,
                                 n,
                                 alpha,
                                 (const hipComplex* const*)x,
                                 incx,
                                 (hipComplex* const*)AP,
                                 batchCount);
}

hipblasStatus_t hipblasZhprBatchedCast_64(hipblasHandle_t                   handle,
                                          hipblasFillMode_t                 uplo,
                                          int64_t                           n,
                                          const double*                     alpha,
                                          const hipblasDoubleComplex* const x[],
                                          int64_t                           incx,
                                          hipblasDoubleComplex* const       AP[],
                                          int64_t                           batchCount)
{
    return hipblasZhprBatched_64(handle,
                                 uplo,
                                 n,
                                 alpha,
                                 (const hipDoubleComplex* const*)x,
                                 incx,
                                 (hipDoubleComplex* const*)AP,
                                 batchCount);
}

// hpr_strided_batched
hipblasStatus_t hipblasChprStridedBatchedCast(hipblasHandle_t       handle,
                                              hipblasFillMode_t     uplo,
                                              int                   n,
                                              const float*          alpha,
                                              const hipblasComplex* x,
                                              int                   incx,
                                              hipblasStride         stridex,
                                              hipblasComplex*       AP,
                                              hipblasStride         strideAP,
                                              int                   batchCount)
{
    return hipblasChprStridedBatched(handle,
                                     uplo,
                                     n,
                                     alpha,
                                     (const hipComplex*)x,
                                     incx,
                                     stridex,
                                     (hipComplex*)AP,
                                     strideAP,
                                     batchCount);
}

hipblasStatus_t hipblasZhprStridedBatchedCast(hipblasHandle_t             handle,
                                              hipblasFillMode_t           uplo,
                                              int                         n,
                                              const double*               alpha,
                                              const hipblasDoubleComplex* x,
                                              int                         incx,
                                              hipblasStride               stridex,
                                              hipblasDoubleComplex*       AP,
                                              hipblasStride               strideAP,
                                              int                         batchCount)
{
    return hipblasZhprStridedBatched(handle,
                                     uplo,
                                     n,
                                     alpha,
                                     (const hipDoubleComplex*)x,
                                     incx,
                                     stridex,
                                     (hipDoubleComplex*)AP,
                                     strideAP,
                                     batchCount);
}

// hpr_strided_batched_64
hipblasStatus_t hipblasChprStridedBatchedCast_64(hipblasHandle_t       handle,
                                                 hipblasFillMode_t     uplo,
                                                 int64_t               n,
                                                 const float*          alpha,
                                                 const hipblasComplex* x,
                                                 int64_t               incx,
                                                 hipblasStride         stridex,
                                                 hipblasComplex*       AP,
                                                 hipblasStride         strideAP,
                                                 int64_t               batchCount)
{
    return hipblasChprStridedBatched_64(handle,
                                        uplo,
                                        n,
                                        alpha,
                                        (const hipComplex*)x,
                                        incx,
                                        stridex,
                                        (hipComplex*)AP,
                                        strideAP,
                                        batchCount);
}

hipblasStatus_t hipblasZhprStridedBatchedCast_64(hipblasHandle_t             handle,
                                                 hipblasFillMode_t           uplo,
                                                 int64_t                     n,
                                                 const double*               alpha,
                                                 const hipblasDoubleComplex* x,
                                                 int64_t                     incx,
                                                 hipblasStride               stridex,
                                                 hipblasDoubleComplex*       AP,
                                                 hipblasStride               strideAP,
                                                 int64_t                     batchCount)
{
    return hipblasZhprStridedBatched_64(handle,
                                        uplo,
                                        n,
                                        alpha,
                                        (const hipDoubleComplex*)x,
                                        incx,
                                        stridex,
                                        (hipDoubleComplex*)AP,
                                        strideAP,
                                        batchCount);
}

// hpr2
hipblasStatus_t hipblasChpr2Cast(hipblasHandle_t       handle,
                                 hipblasFillMode_t     uplo,
                                 int                   n,
                                 const hipblasComplex* alpha,
                                 const hipblasComplex* x,
                                 int                   incx,
                                 const hipblasComplex* y,
                                 int                   incy,
                                 hipblasComplex*       AP)
{
    return hipblasChpr2(handle,
                        uplo,
                        n,
                        (const hipComplex*)alpha,
                        (const hipComplex*)x,
                        incx,
                        (const hipComplex*)y,
                        incy,
                        (hipComplex*)AP);
}

hipblasStatus_t hipblasZhpr2Cast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* y,
                                 int                         incy,
                                 hipblasDoubleComplex*       AP)
{
    return hipblasZhpr2(handle,
                        uplo,
                        n,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)x,
                        incx,
                        (const hipDoubleComplex*)y,
                        incy,
                        (hipDoubleComplex*)AP);
}

// hpr2
hipblasStatus_t hipblasChpr2Cast_64(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    int64_t               n,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* x,
                                    int64_t               incx,
                                    const hipblasComplex* y,
                                    int64_t               incy,
                                    hipblasComplex*       AP)
{
    return hipblasChpr2_64(handle,
                           uplo,
                           n,
                           (const hipComplex*)alpha,
                           (const hipComplex*)x,
                           incx,
                           (const hipComplex*)y,
                           incy,
                           (hipComplex*)AP);
}

hipblasStatus_t hipblasZhpr2Cast_64(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int64_t                     n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* x,
                                    int64_t                     incx,
                                    const hipblasDoubleComplex* y,
                                    int64_t                     incy,
                                    hipblasDoubleComplex*       AP)
{
    return hipblasZhpr2_64(handle,
                           uplo,
                           n,
                           (const hipDoubleComplex*)alpha,
                           (const hipDoubleComplex*)x,
                           incx,
                           (const hipDoubleComplex*)y,
                           incy,
                           (hipDoubleComplex*)AP);
}

// hpr2_batched
hipblasStatus_t hipblasChpr2BatchedCast(hipblasHandle_t             handle,
                                        hipblasFillMode_t           uplo,
                                        int                         n,
                                        const hipblasComplex*       alpha,
                                        const hipblasComplex* const x[],
                                        int                         incx,
                                        const hipblasComplex* const y[],
                                        int                         incy,
                                        hipblasComplex* const       AP[],
                                        int                         batchCount)
{
    return hipblasChpr2Batched(handle,
                               uplo,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)x,
                               incx,
                               (const hipComplex* const*)y,
                               incy,
                               (hipComplex* const*)AP,
                               batchCount);
}

hipblasStatus_t hipblasZhpr2BatchedCast(hipblasHandle_t                   handle,
                                        hipblasFillMode_t                 uplo,
                                        int                               n,
                                        const hipblasDoubleComplex*       alpha,
                                        const hipblasDoubleComplex* const x[],
                                        int                               incx,
                                        const hipblasDoubleComplex* const y[],
                                        int                               incy,
                                        hipblasDoubleComplex* const       AP[],
                                        int                               batchCount)
{
    return hipblasZhpr2Batched(handle,
                               uplo,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (const hipDoubleComplex* const*)y,
                               incy,
                               (hipDoubleComplex* const*)AP,
                               batchCount);
}

// hpr2_batched_64
hipblasStatus_t hipblasChpr2BatchedCast_64(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int64_t                     n,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const x[],
                                           int64_t                     incx,
                                           const hipblasComplex* const y[],
                                           int64_t                     incy,
                                           hipblasComplex* const       AP[],
                                           int64_t                     batchCount)
{
    return hipblasChpr2Batched_64(handle,
                                  uplo,
                                  n,
                                  (const hipComplex*)alpha,
                                  (const hipComplex* const*)x,
                                  incx,
                                  (const hipComplex* const*)y,
                                  incy,
                                  (hipComplex* const*)AP,
                                  batchCount);
}

hipblasStatus_t hipblasZhpr2BatchedCast_64(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           int64_t                           n,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const x[],
                                           int64_t                           incx,
                                           const hipblasDoubleComplex* const y[],
                                           int64_t                           incy,
                                           hipblasDoubleComplex* const       AP[],
                                           int64_t                           batchCount)
{
    return hipblasZhpr2Batched_64(handle,
                                  uplo,
                                  n,
                                  (const hipDoubleComplex*)alpha,
                                  (const hipDoubleComplex* const*)x,
                                  incx,
                                  (const hipDoubleComplex* const*)y,
                                  incy,
                                  (hipDoubleComplex* const*)AP,
                                  batchCount);
}

// hpr2_strided_batched
hipblasStatus_t hipblasChpr2StridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasFillMode_t     uplo,
                                               int                   n,
                                               const hipblasComplex* alpha,
                                               const hipblasComplex* x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               const hipblasComplex* y,
                                               int                   incy,
                                               hipblasStride         stridey,
                                               hipblasComplex*       AP,
                                               hipblasStride         strideAP,
                                               int                   batchCount)
{
    return hipblasChpr2StridedBatched(handle,
                                      uplo,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipComplex*)y,
                                      incy,
                                      stridey,
                                      (hipComplex*)AP,
                                      strideAP,
                                      batchCount);
}

hipblasStatus_t hipblasZhpr2StridedBatchedCast(hipblasHandle_t             handle,
                                               hipblasFillMode_t           uplo,
                                               int                         n,
                                               const hipblasDoubleComplex* alpha,
                                               const hipblasDoubleComplex* x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               const hipblasDoubleComplex* y,
                                               int                         incy,
                                               hipblasStride               stridey,
                                               hipblasDoubleComplex*       AP,
                                               hipblasStride               strideAP,
                                               int                         batchCount)
{
    return hipblasZhpr2StridedBatched(handle,
                                      uplo,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipDoubleComplex*)y,
                                      incy,
                                      stridey,
                                      (hipDoubleComplex*)AP,
                                      strideAP,
                                      batchCount);
}

// hpr2_strided_batched_64
hipblasStatus_t hipblasChpr2StridedBatchedCast_64(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  int64_t               n,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* x,
                                                  int64_t               incx,
                                                  hipblasStride         stridex,
                                                  const hipblasComplex* y,
                                                  int64_t               incy,
                                                  hipblasStride         stridey,
                                                  hipblasComplex*       AP,
                                                  hipblasStride         strideAP,
                                                  int64_t               batchCount)
{
    return hipblasChpr2StridedBatched_64(handle,
                                         uplo,
                                         n,
                                         (const hipComplex*)alpha,
                                         (const hipComplex*)x,
                                         incx,
                                         stridex,
                                         (const hipComplex*)y,
                                         incy,
                                         stridey,
                                         (hipComplex*)AP,
                                         strideAP,
                                         batchCount);
}

hipblasStatus_t hipblasZhpr2StridedBatchedCast_64(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  int64_t                     n,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* x,
                                                  int64_t                     incx,
                                                  hipblasStride               stridex,
                                                  const hipblasDoubleComplex* y,
                                                  int64_t                     incy,
                                                  hipblasStride               stridey,
                                                  hipblasDoubleComplex*       AP,
                                                  hipblasStride               strideAP,
                                                  int64_t                     batchCount)
{
    return hipblasZhpr2StridedBatched_64(handle,
                                         uplo,
                                         n,
                                         (const hipDoubleComplex*)alpha,
                                         (const hipDoubleComplex*)x,
                                         incx,
                                         stridex,
                                         (const hipDoubleComplex*)y,
                                         incy,
                                         stridey,
                                         (hipDoubleComplex*)AP,
                                         strideAP,
                                         batchCount);
}

// sbmv, spmv, spr2 no complex versions

// spr
hipblasStatus_t hipblasCsprCast(hipblasHandle_t       handle,
                                hipblasFillMode_t     uplo,
                                int                   n,
                                const hipblasComplex* alpha,
                                const hipblasComplex* x,
                                int                   incx,
                                hipblasComplex*       AP)
{
    return hipblasCspr(
        handle, uplo, n, (const hipComplex*)alpha, (const hipComplex*)x, incx, (hipComplex*)AP);
}

hipblasStatus_t hipblasZsprCast(hipblasHandle_t             handle,
                                hipblasFillMode_t           uplo,
                                int                         n,
                                const hipblasDoubleComplex* alpha,
                                const hipblasDoubleComplex* x,
                                int                         incx,
                                hipblasDoubleComplex*       AP)
{
    return hipblasZspr(handle,
                       uplo,
                       n,
                       (const hipDoubleComplex*)alpha,
                       (const hipDoubleComplex*)x,
                       incx,
                       (hipDoubleComplex*)AP);
}

// spr_64
hipblasStatus_t hipblasCsprCast_64(hipblasHandle_t       handle,
                                   hipblasFillMode_t     uplo,
                                   int64_t               n,
                                   const hipblasComplex* alpha,
                                   const hipblasComplex* x,
                                   int64_t               incx,
                                   hipblasComplex*       AP)
{
    return hipblasCspr_64(
        handle, uplo, n, (const hipComplex*)alpha, (const hipComplex*)x, incx, (hipComplex*)AP);
}

hipblasStatus_t hipblasZsprCast_64(hipblasHandle_t             handle,
                                   hipblasFillMode_t           uplo,
                                   int64_t                     n,
                                   const hipblasDoubleComplex* alpha,
                                   const hipblasDoubleComplex* x,
                                   int64_t                     incx,
                                   hipblasDoubleComplex*       AP)
{
    return hipblasZspr_64(handle,
                          uplo,
                          n,
                          (const hipDoubleComplex*)alpha,
                          (const hipDoubleComplex*)x,
                          incx,
                          (hipDoubleComplex*)AP);
}

// spr_batched
hipblasStatus_t hipblasCsprBatchedCast(hipblasHandle_t             handle,
                                       hipblasFillMode_t           uplo,
                                       int                         n,
                                       const hipblasComplex*       alpha,
                                       const hipblasComplex* const x[],
                                       int                         incx,
                                       hipblasComplex* const       AP[],
                                       int                         batchCount)
{
    return hipblasCsprBatched(handle,
                              uplo,
                              n,
                              (const hipComplex*)alpha,
                              (const hipComplex* const*)x,
                              incx,
                              (hipComplex* const*)AP,
                              batchCount);
}

hipblasStatus_t hipblasZsprBatchedCast(hipblasHandle_t                   handle,
                                       hipblasFillMode_t                 uplo,
                                       int                               n,
                                       const hipblasDoubleComplex*       alpha,
                                       const hipblasDoubleComplex* const x[],
                                       int                               incx,
                                       hipblasDoubleComplex* const       AP[],
                                       int                               batchCount)
{
    return hipblasZsprBatched(handle,
                              uplo,
                              n,
                              (const hipDoubleComplex*)alpha,
                              (const hipDoubleComplex* const*)x,
                              incx,
                              (hipDoubleComplex* const*)AP,
                              batchCount);
}

// spr_batched_64
hipblasStatus_t hipblasCsprBatchedCast_64(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          int64_t                     n,
                                          const hipblasComplex*       alpha,
                                          const hipblasComplex* const x[],
                                          int64_t                     incx,
                                          hipblasComplex* const       AP[],
                                          int64_t                     batchCount)
{
    return hipblasCsprBatched_64(handle,
                                 uplo,
                                 n,
                                 (const hipComplex*)alpha,
                                 (const hipComplex* const*)x,
                                 incx,
                                 (hipComplex* const*)AP,
                                 batchCount);
}

hipblasStatus_t hipblasZsprBatchedCast_64(hipblasHandle_t                   handle,
                                          hipblasFillMode_t                 uplo,
                                          int64_t                           n,
                                          const hipblasDoubleComplex*       alpha,
                                          const hipblasDoubleComplex* const x[],
                                          int64_t                           incx,
                                          hipblasDoubleComplex* const       AP[],
                                          int64_t                           batchCount)
{
    return hipblasZsprBatched_64(handle,
                                 uplo,
                                 n,
                                 (const hipDoubleComplex*)alpha,
                                 (const hipDoubleComplex* const*)x,
                                 incx,
                                 (hipDoubleComplex* const*)AP,
                                 batchCount);
}

// spr_strided_batched
hipblasStatus_t hipblasCsprStridedBatchedCast(hipblasHandle_t       handle,
                                              hipblasFillMode_t     uplo,
                                              int                   n,
                                              const hipblasComplex* alpha,
                                              const hipblasComplex* x,
                                              int                   incx,
                                              hipblasStride         stridex,
                                              hipblasComplex*       AP,
                                              hipblasStride         strideAP,
                                              int                   batchCount)
{
    return hipblasCsprStridedBatched(handle,
                                     uplo,
                                     n,
                                     (const hipComplex*)alpha,
                                     (const hipComplex*)x,
                                     incx,
                                     stridex,
                                     (hipComplex*)AP,
                                     strideAP,
                                     batchCount);
}

hipblasStatus_t hipblasZsprStridedBatchedCast(hipblasHandle_t             handle,
                                              hipblasFillMode_t           uplo,
                                              int                         n,
                                              const hipblasDoubleComplex* alpha,
                                              const hipblasDoubleComplex* x,
                                              int                         incx,
                                              hipblasStride               stridex,
                                              hipblasDoubleComplex*       AP,
                                              hipblasStride               strideAP,
                                              int                         batchCount)
{
    return hipblasZsprStridedBatched(handle,
                                     uplo,
                                     n,
                                     (const hipDoubleComplex*)alpha,
                                     (const hipDoubleComplex*)x,
                                     incx,
                                     stridex,
                                     (hipDoubleComplex*)AP,
                                     strideAP,
                                     batchCount);
}

// spr_strided_batched_64
hipblasStatus_t hipblasCsprStridedBatchedCast_64(hipblasHandle_t       handle,
                                                 hipblasFillMode_t     uplo,
                                                 int64_t               n,
                                                 const hipblasComplex* alpha,
                                                 const hipblasComplex* x,
                                                 int64_t               incx,
                                                 hipblasStride         stridex,
                                                 hipblasComplex*       AP,
                                                 hipblasStride         strideAP,
                                                 int64_t               batchCount)
{
    return hipblasCsprStridedBatched_64(handle,
                                        uplo,
                                        n,
                                        (const hipComplex*)alpha,
                                        (const hipComplex*)x,
                                        incx,
                                        stridex,
                                        (hipComplex*)AP,
                                        strideAP,
                                        batchCount);
}

hipblasStatus_t hipblasZsprStridedBatchedCast_64(hipblasHandle_t             handle,
                                                 hipblasFillMode_t           uplo,
                                                 int64_t                     n,
                                                 const hipblasDoubleComplex* alpha,
                                                 const hipblasDoubleComplex* x,
                                                 int64_t                     incx,
                                                 hipblasStride               stridex,
                                                 hipblasDoubleComplex*       AP,
                                                 hipblasStride               strideAP,
                                                 int64_t                     batchCount)
{
    return hipblasZsprStridedBatched_64(handle,
                                        uplo,
                                        n,
                                        (const hipDoubleComplex*)alpha,
                                        (const hipDoubleComplex*)x,
                                        incx,
                                        stridex,
                                        (hipDoubleComplex*)AP,
                                        strideAP,
                                        batchCount);
}

// symv
hipblasStatus_t hipblasCsymvCast(hipblasHandle_t       handle,
                                 hipblasFillMode_t     uplo,
                                 int                   n,
                                 const hipblasComplex* alpha,
                                 const hipblasComplex* A,
                                 int                   lda,
                                 const hipblasComplex* x,
                                 int                   incx,
                                 const hipblasComplex* beta,
                                 hipblasComplex*       y,
                                 int                   incy)
{
    return hipblasCsymv(handle,
                        uplo,
                        n,
                        (const hipComplex*)alpha,
                        (const hipComplex*)A,
                        lda,
                        (const hipComplex*)x,
                        incx,
                        (const hipComplex*)beta,
                        (hipComplex*)y,
                        incy);
}

hipblasStatus_t hipblasZsymvCast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* A,
                                 int                         lda,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* beta,
                                 hipblasDoubleComplex*       y,
                                 int                         incy)
{
    return hipblasZsymv(handle,
                        uplo,
                        n,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)A,
                        lda,
                        (const hipDoubleComplex*)x,
                        incx,
                        (const hipDoubleComplex*)beta,
                        (hipDoubleComplex*)y,
                        incy);
}

// symv_64
hipblasStatus_t hipblasCsymvCast_64(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    int64_t               n,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* A,
                                    int64_t               lda,
                                    const hipblasComplex* x,
                                    int64_t               incx,
                                    const hipblasComplex* beta,
                                    hipblasComplex*       y,
                                    int64_t               incy)
{
    return hipblasCsymv_64(handle,
                           uplo,
                           n,
                           (const hipComplex*)alpha,
                           (const hipComplex*)A,
                           lda,
                           (const hipComplex*)x,
                           incx,
                           (const hipComplex*)beta,
                           (hipComplex*)y,
                           incy);
}

hipblasStatus_t hipblasZsymvCast_64(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int64_t                     n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* A,
                                    int64_t                     lda,
                                    const hipblasDoubleComplex* x,
                                    int64_t                     incx,
                                    const hipblasDoubleComplex* beta,
                                    hipblasDoubleComplex*       y,
                                    int64_t                     incy)
{
    return hipblasZsymv_64(handle,
                           uplo,
                           n,
                           (const hipDoubleComplex*)alpha,
                           (const hipDoubleComplex*)A,
                           lda,
                           (const hipDoubleComplex*)x,
                           incx,
                           (const hipDoubleComplex*)beta,
                           (hipDoubleComplex*)y,
                           incy);
}

// symv_batched
hipblasStatus_t hipblasCsymvBatchedCast(hipblasHandle_t             handle,
                                        hipblasFillMode_t           uplo,
                                        int                         n,
                                        const hipblasComplex*       alpha,
                                        const hipblasComplex* const A[],
                                        int                         lda,
                                        const hipblasComplex* const x[],
                                        int                         incx,
                                        const hipblasComplex*       beta,
                                        hipblasComplex* const       y[],
                                        int                         incy,
                                        int                         batchCount)
{
    return hipblasCsymvBatched(handle,
                               uplo,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)A,
                               lda,
                               (const hipComplex* const*)x,
                               incx,
                               (const hipComplex*)beta,
                               (hipComplex* const*)y,
                               incy,
                               batchCount);
}

hipblasStatus_t hipblasZsymvBatchedCast(hipblasHandle_t                   handle,
                                        hipblasFillMode_t                 uplo,
                                        int                               n,
                                        const hipblasDoubleComplex*       alpha,
                                        const hipblasDoubleComplex* const A[],
                                        int                               lda,
                                        const hipblasDoubleComplex* const x[],
                                        int                               incx,
                                        const hipblasDoubleComplex*       beta,
                                        hipblasDoubleComplex* const       y[],
                                        int                               incy,
                                        int                               batchCount)
{
    return hipblasZsymvBatched(handle,
                               uplo,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (const hipDoubleComplex*)beta,
                               (hipDoubleComplex* const*)y,
                               incy,
                               batchCount);
}

// symv_batched_64
hipblasStatus_t hipblasCsymvBatchedCast_64(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int64_t                     n,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const A[],
                                           int64_t                     lda,
                                           const hipblasComplex* const x[],
                                           int64_t                     incx,
                                           const hipblasComplex*       beta,
                                           hipblasComplex* const       y[],
                                           int64_t                     incy,
                                           int64_t                     batchCount)
{
    return hipblasCsymvBatched_64(handle,
                                  uplo,
                                  n,
                                  (const hipComplex*)alpha,
                                  (const hipComplex* const*)A,
                                  lda,
                                  (const hipComplex* const*)x,
                                  incx,
                                  (const hipComplex*)beta,
                                  (hipComplex* const*)y,
                                  incy,
                                  batchCount);
}

hipblasStatus_t hipblasZsymvBatchedCast_64(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           int64_t                           n,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const A[],
                                           int64_t                           lda,
                                           const hipblasDoubleComplex* const x[],
                                           int64_t                           incx,
                                           const hipblasDoubleComplex*       beta,
                                           hipblasDoubleComplex* const       y[],
                                           int64_t                           incy,
                                           int64_t                           batchCount)
{
    return hipblasZsymvBatched_64(handle,
                                  uplo,
                                  n,
                                  (const hipDoubleComplex*)alpha,
                                  (const hipDoubleComplex* const*)A,
                                  lda,
                                  (const hipDoubleComplex* const*)x,
                                  incx,
                                  (const hipDoubleComplex*)beta,
                                  (hipDoubleComplex* const*)y,
                                  incy,
                                  batchCount);
}

// symv_strided_batched
hipblasStatus_t hipblasCsymvStridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasFillMode_t     uplo,
                                               int                   n,
                                               const hipblasComplex* alpha,
                                               const hipblasComplex* A,
                                               int                   lda,
                                               hipblasStride         strideA,
                                               const hipblasComplex* x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               const hipblasComplex* beta,
                                               hipblasComplex*       y,
                                               int                   incy,
                                               hipblasStride         stridey,
                                               int                   batchCount)
{
    return hipblasCsymvStridedBatched(handle,
                                      uplo,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipComplex*)beta,
                                      (hipComplex*)y,
                                      incy,
                                      stridey,
                                      batchCount);
}

hipblasStatus_t hipblasZsymvStridedBatchedCast(hipblasHandle_t             handle,
                                               hipblasFillMode_t           uplo,
                                               int                         n,
                                               const hipblasDoubleComplex* alpha,
                                               const hipblasDoubleComplex* A,
                                               int                         lda,
                                               hipblasStride               strideA,
                                               const hipblasDoubleComplex* x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               const hipblasDoubleComplex* beta,
                                               hipblasDoubleComplex*       y,
                                               int                         incy,
                                               hipblasStride               stridey,
                                               int                         batchCount)
{
    return hipblasZsymvStridedBatched(handle,
                                      uplo,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipDoubleComplex*)beta,
                                      (hipDoubleComplex*)y,
                                      incy,
                                      stridey,
                                      batchCount);
}

// symv_strided_batched_64
hipblasStatus_t hipblasCsymvStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  int64_t               n,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* A,
                                                  int64_t               lda,
                                                  hipblasStride         strideA,
                                                  const hipblasComplex* x,
                                                  int64_t               incx,
                                                  hipblasStride         stridex,
                                                  const hipblasComplex* beta,
                                                  hipblasComplex*       y,
                                                  int64_t               incy,
                                                  hipblasStride         stridey,
                                                  int64_t               batchCount)
{
    return hipblasCsymvStridedBatched_64(handle,
                                         uplo,
                                         n,
                                         (const hipComplex*)alpha,
                                         (const hipComplex*)A,
                                         lda,
                                         strideA,
                                         (const hipComplex*)x,
                                         incx,
                                         stridex,
                                         (const hipComplex*)beta,
                                         (hipComplex*)y,
                                         incy,
                                         stridey,
                                         batchCount);
}

hipblasStatus_t hipblasZsymvStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  int64_t                     n,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* A,
                                                  int64_t                     lda,
                                                  hipblasStride               strideA,
                                                  const hipblasDoubleComplex* x,
                                                  int64_t                     incx,
                                                  hipblasStride               stridex,
                                                  const hipblasDoubleComplex* beta,
                                                  hipblasDoubleComplex*       y,
                                                  int64_t                     incy,
                                                  hipblasStride               stridey,
                                                  int64_t                     batchCount)
{
    return hipblasZsymvStridedBatched_64(handle,
                                         uplo,
                                         n,
                                         (const hipDoubleComplex*)alpha,
                                         (const hipDoubleComplex*)A,
                                         lda,
                                         strideA,
                                         (const hipDoubleComplex*)x,
                                         incx,
                                         stridex,
                                         (const hipDoubleComplex*)beta,
                                         (hipDoubleComplex*)y,
                                         incy,
                                         stridey,
                                         batchCount);
}

// syr
hipblasStatus_t hipblasCsyrCast(hipblasHandle_t       handle,
                                hipblasFillMode_t     uplo,
                                int                   n,
                                const hipblasComplex* alpha,
                                const hipblasComplex* x,
                                int                   incx,
                                hipblasComplex*       A,
                                int                   lda)
{
    return hipblasCsyr(
        handle, uplo, n, (const hipComplex*)alpha, (const hipComplex*)x, incx, (hipComplex*)A, lda);
}

hipblasStatus_t hipblasZsyrCast(hipblasHandle_t             handle,
                                hipblasFillMode_t           uplo,
                                int                         n,
                                const hipblasDoubleComplex* alpha,
                                const hipblasDoubleComplex* x,
                                int                         incx,
                                hipblasDoubleComplex*       A,
                                int                         lda)
{
    return hipblasZsyr(handle,
                       uplo,
                       n,
                       (const hipDoubleComplex*)alpha,
                       (const hipDoubleComplex*)x,
                       incx,
                       (hipDoubleComplex*)A,
                       lda);
}

// syr_64
hipblasStatus_t hipblasCsyrCast_64(hipblasHandle_t       handle,
                                   hipblasFillMode_t     uplo,
                                   int64_t               n,
                                   const hipblasComplex* alpha,
                                   const hipblasComplex* x,
                                   int64_t               incx,
                                   hipblasComplex*       A,
                                   int64_t               lda)
{
    return hipblasCsyr_64(
        handle, uplo, n, (const hipComplex*)alpha, (const hipComplex*)x, incx, (hipComplex*)A, lda);
}

hipblasStatus_t hipblasZsyrCast_64(hipblasHandle_t             handle,
                                   hipblasFillMode_t           uplo,
                                   int64_t                     n,
                                   const hipblasDoubleComplex* alpha,
                                   const hipblasDoubleComplex* x,
                                   int64_t                     incx,
                                   hipblasDoubleComplex*       A,
                                   int64_t                     lda)
{
    return hipblasZsyr_64(handle,
                          uplo,
                          n,
                          (const hipDoubleComplex*)alpha,
                          (const hipDoubleComplex*)x,
                          incx,
                          (hipDoubleComplex*)A,
                          lda);
}

// syr_batched
hipblasStatus_t hipblasCsyrBatchedCast(hipblasHandle_t             handle,
                                       hipblasFillMode_t           uplo,
                                       int                         n,
                                       const hipblasComplex*       alpha,
                                       const hipblasComplex* const x[],
                                       int                         incx,
                                       hipblasComplex* const       A[],
                                       int                         lda,
                                       int                         batch_count)
{
    return hipblasCsyrBatched(handle,
                              uplo,
                              n,
                              (const hipComplex*)alpha,
                              (const hipComplex* const*)x,
                              incx,
                              (hipComplex* const*)A,
                              lda,
                              batch_count);
}

hipblasStatus_t hipblasZsyrBatchedCast(hipblasHandle_t                   handle,
                                       hipblasFillMode_t                 uplo,
                                       int                               n,
                                       const hipblasDoubleComplex*       alpha,
                                       const hipblasDoubleComplex* const x[],
                                       int                               incx,
                                       hipblasDoubleComplex* const       A[],
                                       int                               lda,
                                       int                               batch_count)
{
    return hipblasZsyrBatched(handle,
                              uplo,
                              n,
                              (const hipDoubleComplex*)alpha,
                              (const hipDoubleComplex* const*)x,
                              incx,
                              (hipDoubleComplex* const*)A,
                              lda,
                              batch_count);
}

// syr_batched_64
hipblasStatus_t hipblasCsyrBatchedCast_64(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          int64_t                     n,
                                          const hipblasComplex*       alpha,
                                          const hipblasComplex* const x[],
                                          int64_t                     incx,
                                          hipblasComplex* const       A[],
                                          int64_t                     lda,
                                          int64_t                     batch_count)
{
    return hipblasCsyrBatched_64(handle,
                                 uplo,
                                 n,
                                 (const hipComplex*)alpha,
                                 (const hipComplex* const*)x,
                                 incx,
                                 (hipComplex* const*)A,
                                 lda,
                                 batch_count);
}

hipblasStatus_t hipblasZsyrBatchedCast_64(hipblasHandle_t                   handle,
                                          hipblasFillMode_t                 uplo,
                                          int64_t                           n,
                                          const hipblasDoubleComplex*       alpha,
                                          const hipblasDoubleComplex* const x[],
                                          int64_t                           incx,
                                          hipblasDoubleComplex* const       A[],
                                          int64_t                           lda,
                                          int64_t                           batch_count)
{
    return hipblasZsyrBatched_64(handle,
                                 uplo,
                                 n,
                                 (const hipDoubleComplex*)alpha,
                                 (const hipDoubleComplex* const*)x,
                                 incx,
                                 (hipDoubleComplex* const*)A,
                                 lda,
                                 batch_count);
}

// syr_strided_batched
hipblasStatus_t hipblasCsyrStridedBatchedCast(hipblasHandle_t       handle,
                                              hipblasFillMode_t     uplo,
                                              int                   n,
                                              const hipblasComplex* alpha,
                                              const hipblasComplex* x,
                                              int                   incx,
                                              hipblasStride         stridex,
                                              hipblasComplex*       A,
                                              int                   lda,
                                              hipblasStride         strideA,
                                              int                   batch_count)
{
    return hipblasCsyrStridedBatched(handle,
                                     uplo,
                                     n,
                                     (const hipComplex*)alpha,
                                     (const hipComplex*)x,
                                     incx,
                                     stridex,
                                     (hipComplex*)A,
                                     lda,
                                     strideA,
                                     batch_count);
}

hipblasStatus_t hipblasZsyrStridedBatchedCast(hipblasHandle_t             handle,
                                              hipblasFillMode_t           uplo,
                                              int                         n,
                                              const hipblasDoubleComplex* alpha,
                                              const hipblasDoubleComplex* x,
                                              int                         incx,
                                              hipblasStride               stridex,
                                              hipblasDoubleComplex*       A,
                                              int                         lda,
                                              hipblasStride               strideA,
                                              int                         batch_count)
{
    return hipblasZsyrStridedBatched(handle,
                                     uplo,
                                     n,
                                     (const hipDoubleComplex*)alpha,
                                     (const hipDoubleComplex*)x,
                                     incx,
                                     stridex,
                                     (hipDoubleComplex*)A,
                                     lda,
                                     strideA,
                                     batch_count);
}

// syr_strided_batched_64
hipblasStatus_t hipblasCsyrStridedBatchedCast_64(hipblasHandle_t       handle,
                                                 hipblasFillMode_t     uplo,
                                                 int64_t               n,
                                                 const hipblasComplex* alpha,
                                                 const hipblasComplex* x,
                                                 int64_t               incx,
                                                 hipblasStride         stridex,
                                                 hipblasComplex*       A,
                                                 int64_t               lda,
                                                 hipblasStride         strideA,
                                                 int64_t               batch_count)
{
    return hipblasCsyrStridedBatched_64(handle,
                                        uplo,
                                        n,
                                        (const hipComplex*)alpha,
                                        (const hipComplex*)x,
                                        incx,
                                        stridex,
                                        (hipComplex*)A,
                                        lda,
                                        strideA,
                                        batch_count);
}

hipblasStatus_t hipblasZsyrStridedBatchedCast_64(hipblasHandle_t             handle,
                                                 hipblasFillMode_t           uplo,
                                                 int64_t                     n,
                                                 const hipblasDoubleComplex* alpha,
                                                 const hipblasDoubleComplex* x,
                                                 int64_t                     incx,
                                                 hipblasStride               stridex,
                                                 hipblasDoubleComplex*       A,
                                                 int64_t                     lda,
                                                 hipblasStride               strideA,
                                                 int64_t                     batch_count)
{
    return hipblasZsyrStridedBatched_64(handle,
                                        uplo,
                                        n,
                                        (const hipDoubleComplex*)alpha,
                                        (const hipDoubleComplex*)x,
                                        incx,
                                        stridex,
                                        (hipDoubleComplex*)A,
                                        lda,
                                        strideA,
                                        batch_count);
}

// syr2
hipblasStatus_t hipblasCsyr2Cast(hipblasHandle_t       handle,
                                 hipblasFillMode_t     uplo,
                                 int                   n,
                                 const hipblasComplex* alpha,
                                 const hipblasComplex* x,
                                 int                   incx,
                                 const hipblasComplex* y,
                                 int                   incy,
                                 hipblasComplex*       A,
                                 int                   lda)
{
    return hipblasCsyr2(handle,
                        uplo,
                        n,
                        (const hipComplex*)alpha,
                        (const hipComplex*)x,
                        incx,
                        (const hipComplex*)y,
                        incy,
                        (hipComplex*)A,
                        lda);
}

hipblasStatus_t hipblasZsyr2Cast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* y,
                                 int                         incy,
                                 hipblasDoubleComplex*       A,
                                 int                         lda)
{
    return hipblasZsyr2(handle,
                        uplo,
                        n,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)x,
                        incx,
                        (const hipDoubleComplex*)y,
                        incy,
                        (hipDoubleComplex*)A,
                        lda);
}

// syr2
hipblasStatus_t hipblasCsyr2Cast_64(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    int64_t               n,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* x,
                                    int64_t               incx,
                                    const hipblasComplex* y,
                                    int64_t               incy,
                                    hipblasComplex*       A,
                                    int64_t               lda)
{
    return hipblasCsyr2_64(handle,
                           uplo,
                           n,
                           (const hipComplex*)alpha,
                           (const hipComplex*)x,
                           incx,
                           (const hipComplex*)y,
                           incy,
                           (hipComplex*)A,
                           lda);
}

hipblasStatus_t hipblasZsyr2Cast_64(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int64_t                     n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* x,
                                    int64_t                     incx,
                                    const hipblasDoubleComplex* y,
                                    int64_t                     incy,
                                    hipblasDoubleComplex*       A,
                                    int64_t                     lda)
{
    return hipblasZsyr2_64(handle,
                           uplo,
                           n,
                           (const hipDoubleComplex*)alpha,
                           (const hipDoubleComplex*)x,
                           incx,
                           (const hipDoubleComplex*)y,
                           incy,
                           (hipDoubleComplex*)A,
                           lda);
}

// syr2_batched
hipblasStatus_t hipblasCsyr2BatchedCast(hipblasHandle_t             handle,
                                        hipblasFillMode_t           uplo,
                                        int                         n,
                                        const hipblasComplex*       alpha,
                                        const hipblasComplex* const x[],
                                        int                         incx,
                                        const hipblasComplex* const y[],
                                        int                         incy,
                                        hipblasComplex* const       A[],
                                        int                         lda,
                                        int                         batchCount)
{
    return hipblasCsyr2Batched(handle,
                               uplo,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)x,
                               incx,
                               (const hipComplex* const*)y,
                               incy,
                               (hipComplex* const*)A,
                               lda,
                               batchCount);
}

hipblasStatus_t hipblasZsyr2BatchedCast(hipblasHandle_t                   handle,
                                        hipblasFillMode_t                 uplo,
                                        int                               n,
                                        const hipblasDoubleComplex*       alpha,
                                        const hipblasDoubleComplex* const x[],
                                        int                               incx,
                                        const hipblasDoubleComplex* const y[],
                                        int                               incy,
                                        hipblasDoubleComplex* const       A[],
                                        int                               lda,
                                        int                               batchCount)
{
    return hipblasZsyr2Batched(handle,
                               uplo,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (const hipDoubleComplex* const*)y,
                               incy,
                               (hipDoubleComplex* const*)A,
                               lda,
                               batchCount);
}

// syr2_batched_64
hipblasStatus_t hipblasCsyr2BatchedCast_64(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int64_t                     n,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const x[],
                                           int64_t                     incx,
                                           const hipblasComplex* const y[],
                                           int64_t                     incy,
                                           hipblasComplex* const       A[],
                                           int64_t                     lda,
                                           int64_t                     batchCount)
{
    return hipblasCsyr2Batched_64(handle,
                                  uplo,
                                  n,
                                  (const hipComplex*)alpha,
                                  (const hipComplex* const*)x,
                                  incx,
                                  (const hipComplex* const*)y,
                                  incy,
                                  (hipComplex* const*)A,
                                  lda,
                                  batchCount);
}

hipblasStatus_t hipblasZsyr2BatchedCast_64(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           int64_t                           n,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const x[],
                                           int64_t                           incx,
                                           const hipblasDoubleComplex* const y[],
                                           int64_t                           incy,
                                           hipblasDoubleComplex* const       A[],
                                           int64_t                           lda,
                                           int64_t                           batchCount)
{
    return hipblasZsyr2Batched_64(handle,
                                  uplo,
                                  n,
                                  (const hipDoubleComplex*)alpha,
                                  (const hipDoubleComplex* const*)x,
                                  incx,
                                  (const hipDoubleComplex* const*)y,
                                  incy,
                                  (hipDoubleComplex* const*)A,
                                  lda,
                                  batchCount);
}

// syr2_strided_batched
hipblasStatus_t hipblasCsyr2StridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasFillMode_t     uplo,
                                               int                   n,
                                               const hipblasComplex* alpha,
                                               const hipblasComplex* x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               const hipblasComplex* y,
                                               int                   incy,
                                               hipblasStride         stridey,
                                               hipblasComplex*       A,
                                               int                   lda,
                                               hipblasStride         strideA,
                                               int                   batchCount)
{
    return hipblasCsyr2StridedBatched(handle,
                                      uplo,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipComplex*)y,
                                      incy,
                                      stridey,
                                      (hipComplex*)A,
                                      lda,
                                      strideA,
                                      batchCount);
}

hipblasStatus_t hipblasZsyr2StridedBatchedCast(hipblasHandle_t             handle,
                                               hipblasFillMode_t           uplo,
                                               int                         n,
                                               const hipblasDoubleComplex* alpha,
                                               const hipblasDoubleComplex* x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               const hipblasDoubleComplex* y,
                                               int                         incy,
                                               hipblasStride               stridey,
                                               hipblasDoubleComplex*       A,
                                               int                         lda,
                                               hipblasStride               strideA,
                                               int                         batchCount)
{
    return hipblasZsyr2StridedBatched(handle,
                                      uplo,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      (const hipDoubleComplex*)y,
                                      incy,
                                      stridey,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      batchCount);
}

// syr2_strided_batched_64
hipblasStatus_t hipblasCsyr2StridedBatchedCast_64(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  int64_t               n,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* x,
                                                  int64_t               incx,
                                                  hipblasStride         stridex,
                                                  const hipblasComplex* y,
                                                  int64_t               incy,
                                                  hipblasStride         stridey,
                                                  hipblasComplex*       A,
                                                  int64_t               lda,
                                                  hipblasStride         strideA,
                                                  int64_t               batchCount)
{
    return hipblasCsyr2StridedBatched_64(handle,
                                         uplo,
                                         n,
                                         (const hipComplex*)alpha,
                                         (const hipComplex*)x,
                                         incx,
                                         stridex,
                                         (const hipComplex*)y,
                                         incy,
                                         stridey,
                                         (hipComplex*)A,
                                         lda,
                                         strideA,
                                         batchCount);
}

hipblasStatus_t hipblasZsyr2StridedBatchedCast_64(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  int64_t                     n,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* x,
                                                  int64_t                     incx,
                                                  hipblasStride               stridex,
                                                  const hipblasDoubleComplex* y,
                                                  int64_t                     incy,
                                                  hipblasStride               stridey,
                                                  hipblasDoubleComplex*       A,
                                                  int64_t                     lda,
                                                  hipblasStride               strideA,
                                                  int64_t                     batchCount)
{
    return hipblasZsyr2StridedBatched_64(handle,
                                         uplo,
                                         n,
                                         (const hipDoubleComplex*)alpha,
                                         (const hipDoubleComplex*)x,
                                         incx,
                                         stridex,
                                         (const hipDoubleComplex*)y,
                                         incy,
                                         stridey,
                                         (hipDoubleComplex*)A,
                                         lda,
                                         strideA,
                                         batchCount);
}

// trsv
hipblasStatus_t hipblasCtrsvCast(hipblasHandle_t       handle,
                                 hipblasFillMode_t     uplo,
                                 hipblasOperation_t    transA,
                                 hipblasDiagType_t     diag,
                                 int                   m,
                                 const hipblasComplex* A,
                                 int                   lda,
                                 hipblasComplex*       x,
                                 int                   incx)
{
    return hipblasCtrsv(
        handle, uplo, transA, diag, m, (const hipComplex*)A, lda, (hipComplex*)x, incx);
}

hipblasStatus_t hipblasZtrsvCast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 hipblasOperation_t          transA,
                                 hipblasDiagType_t           diag,
                                 int                         m,
                                 const hipblasDoubleComplex* A,
                                 int                         lda,
                                 hipblasDoubleComplex*       x,
                                 int                         incx)
{
    return hipblasZtrsv(
        handle, uplo, transA, diag, m, (const hipDoubleComplex*)A, lda, (hipDoubleComplex*)x, incx);
}

// trsv_64
hipblasStatus_t hipblasCtrsvCast_64(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    hipblasOperation_t    transA,
                                    hipblasDiagType_t     diag,
                                    int64_t               m,
                                    const hipblasComplex* A,
                                    int64_t               lda,
                                    hipblasComplex*       x,
                                    int64_t               incx)
{
    return hipblasCtrsv_64(
        handle, uplo, transA, diag, m, (const hipComplex*)A, lda, (hipComplex*)x, incx);
}

hipblasStatus_t hipblasZtrsvCast_64(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int64_t                     m,
                                    const hipblasDoubleComplex* A,
                                    int64_t                     lda,
                                    hipblasDoubleComplex*       x,
                                    int64_t                     incx)
{
    return hipblasZtrsv_64(
        handle, uplo, transA, diag, m, (const hipDoubleComplex*)A, lda, (hipDoubleComplex*)x, incx);
}

// trsv_batched
hipblasStatus_t hipblasCtrsvBatchedCast(hipblasHandle_t             handle,
                                        hipblasFillMode_t           uplo,
                                        hipblasOperation_t          transA,
                                        hipblasDiagType_t           diag,
                                        int                         m,
                                        const hipblasComplex* const A[],
                                        int                         lda,
                                        hipblasComplex* const       x[],
                                        int                         incx,
                                        int                         batch_count)
{
    return hipblasCtrsvBatched(handle,
                               uplo,
                               transA,
                               diag,
                               m,
                               (const hipComplex* const*)A,
                               lda,
                               (hipComplex* const*)x,
                               incx,
                               batch_count);
}

hipblasStatus_t hipblasZtrsvBatchedCast(hipblasHandle_t                   handle,
                                        hipblasFillMode_t                 uplo,
                                        hipblasOperation_t                transA,
                                        hipblasDiagType_t                 diag,
                                        int                               m,
                                        const hipblasDoubleComplex* const A[],
                                        int                               lda,
                                        hipblasDoubleComplex* const       x[],
                                        int                               incx,
                                        int                               batch_count)
{
    return hipblasZtrsvBatched(handle,
                               uplo,
                               transA,
                               diag,
                               m,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (hipDoubleComplex* const*)x,
                               incx,
                               batch_count);
}

// trsv_batched_64
hipblasStatus_t hipblasCtrsvBatchedCast_64(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int64_t                     m,
                                           const hipblasComplex* const A[],
                                           int64_t                     lda,
                                           hipblasComplex* const       x[],
                                           int64_t                     incx,
                                           int64_t                     batch_count)
{
    return hipblasCtrsvBatched_64(handle,
                                  uplo,
                                  transA,
                                  diag,
                                  m,
                                  (const hipComplex* const*)A,
                                  lda,
                                  (hipComplex* const*)x,
                                  incx,
                                  batch_count);
}

hipblasStatus_t hipblasZtrsvBatchedCast_64(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           hipblasOperation_t                transA,
                                           hipblasDiagType_t                 diag,
                                           int64_t                           m,
                                           const hipblasDoubleComplex* const A[],
                                           int64_t                           lda,
                                           hipblasDoubleComplex* const       x[],
                                           int64_t                           incx,
                                           int64_t                           batch_count)
{
    return hipblasZtrsvBatched_64(handle,
                                  uplo,
                                  transA,
                                  diag,
                                  m,
                                  (const hipDoubleComplex* const*)A,
                                  lda,
                                  (hipDoubleComplex* const*)x,
                                  incx,
                                  batch_count);
}

// trsv_strided_batched
hipblasStatus_t hipblasCtrsvStridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasFillMode_t     uplo,
                                               hipblasOperation_t    transA,
                                               hipblasDiagType_t     diag,
                                               int                   m,
                                               const hipblasComplex* A,
                                               int                   lda,
                                               hipblasStride         strideA,
                                               hipblasComplex*       x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               int                   batch_count)
{
    return hipblasCtrsvStridedBatched(handle,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      (const hipComplex*)A,
                                      lda,
                                      strideA,
                                      (hipComplex*)x,
                                      incx,
                                      stridex,
                                      batch_count);
}

hipblasStatus_t hipblasZtrsvStridedBatchedCast(hipblasHandle_t             handle,
                                               hipblasFillMode_t           uplo,
                                               hipblasOperation_t          transA,
                                               hipblasDiagType_t           diag,
                                               int                         m,
                                               const hipblasDoubleComplex* A,
                                               int                         lda,
                                               hipblasStride               strideA,
                                               hipblasDoubleComplex*       x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               int                         batch_count)
{
    return hipblasZtrsvStridedBatched(handle,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      (hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      batch_count);
}

// trsv_strided_batched_64
hipblasStatus_t hipblasCtrsvStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  hipblasOperation_t    transA,
                                                  hipblasDiagType_t     diag,
                                                  int64_t               m,
                                                  const hipblasComplex* A,
                                                  int64_t               lda,
                                                  hipblasStride         strideA,
                                                  hipblasComplex*       x,
                                                  int64_t               incx,
                                                  hipblasStride         stridex,
                                                  int64_t               batch_count)
{
    return hipblasCtrsvStridedBatched_64(handle,
                                         uplo,
                                         transA,
                                         diag,
                                         m,
                                         (const hipComplex*)A,
                                         lda,
                                         strideA,
                                         (hipComplex*)x,
                                         incx,
                                         stridex,
                                         batch_count);
}

hipblasStatus_t hipblasZtrsvStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  hipblasOperation_t          transA,
                                                  hipblasDiagType_t           diag,
                                                  int64_t                     m,
                                                  const hipblasDoubleComplex* A,
                                                  int64_t                     lda,
                                                  hipblasStride               strideA,
                                                  hipblasDoubleComplex*       x,
                                                  int64_t                     incx,
                                                  hipblasStride               stridex,
                                                  int64_t                     batch_count)
{
    return hipblasZtrsvStridedBatched_64(handle,
                                         uplo,
                                         transA,
                                         diag,
                                         m,
                                         (const hipDoubleComplex*)A,
                                         lda,
                                         strideA,
                                         (hipDoubleComplex*)x,
                                         incx,
                                         stridex,
                                         batch_count);
}

// tbmv
hipblasStatus_t hipblasCtbmvCast(hipblasHandle_t       handle,
                                 hipblasFillMode_t     uplo,
                                 hipblasOperation_t    transA,
                                 hipblasDiagType_t     diag,
                                 int                   m,
                                 int                   k,
                                 const hipblasComplex* A,
                                 int                   lda,
                                 hipblasComplex*       x,
                                 int                   incx)
{
    return hipblasCtbmv(
        handle, uplo, transA, diag, m, k, (const hipComplex*)A, lda, (hipComplex*)x, incx);
}

hipblasStatus_t hipblasZtbmvCast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 hipblasOperation_t          transA,
                                 hipblasDiagType_t           diag,
                                 int                         m,
                                 int                         k,
                                 const hipblasDoubleComplex* A,
                                 int                         lda,
                                 hipblasDoubleComplex*       x,
                                 int                         incx)
{
    return hipblasZtbmv(handle,
                        uplo,
                        transA,
                        diag,
                        m,
                        k,
                        (const hipDoubleComplex*)A,
                        lda,
                        (hipDoubleComplex*)x,
                        incx);
}

// tbmv_64
hipblasStatus_t hipblasCtbmvCast_64(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    hipblasOperation_t    transA,
                                    hipblasDiagType_t     diag,
                                    int64_t               m,
                                    int64_t               k,
                                    const hipblasComplex* A,
                                    int64_t               lda,
                                    hipblasComplex*       x,
                                    int64_t               incx)
{
    return hipblasCtbmv_64(
        handle, uplo, transA, diag, m, k, (const hipComplex*)A, lda, (hipComplex*)x, incx);
}

hipblasStatus_t hipblasZtbmvCast_64(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int64_t                     m,
                                    int64_t                     k,
                                    const hipblasDoubleComplex* A,
                                    int64_t                     lda,
                                    hipblasDoubleComplex*       x,
                                    int64_t                     incx)
{
    return hipblasZtbmv_64(handle,
                           uplo,
                           transA,
                           diag,
                           m,
                           k,
                           (const hipDoubleComplex*)A,
                           lda,
                           (hipDoubleComplex*)x,
                           incx);
}

// tbmv_batched
hipblasStatus_t hipblasCtbmvBatchedCast(hipblasHandle_t             handle,
                                        hipblasFillMode_t           uplo,
                                        hipblasOperation_t          transA,
                                        hipblasDiagType_t           diag,
                                        int                         m,
                                        int                         k,
                                        const hipblasComplex* const A[],
                                        int                         lda,
                                        hipblasComplex* const       x[],
                                        int                         incx,
                                        int                         batch_count)
{
    return hipblasCtbmvBatched(handle,
                               uplo,
                               transA,
                               diag,
                               m,
                               k,
                               (const hipComplex* const*)A,
                               lda,
                               (hipComplex* const*)x,
                               incx,
                               batch_count);
}

hipblasStatus_t hipblasZtbmvBatchedCast(hipblasHandle_t                   handle,
                                        hipblasFillMode_t                 uplo,
                                        hipblasOperation_t                transA,
                                        hipblasDiagType_t                 diag,
                                        int                               m,
                                        int                               k,
                                        const hipblasDoubleComplex* const A[],
                                        int                               lda,
                                        hipblasDoubleComplex* const       x[],
                                        int                               incx,
                                        int                               batch_count)
{
    return hipblasZtbmvBatched(handle,
                               uplo,
                               transA,
                               diag,
                               m,
                               k,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (hipDoubleComplex* const*)x,
                               incx,
                               batch_count);
}

// tbmv_batched_64
hipblasStatus_t hipblasCtbmvBatchedCast_64(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int64_t                     m,
                                           int64_t                     k,
                                           const hipblasComplex* const A[],
                                           int64_t                     lda,
                                           hipblasComplex* const       x[],
                                           int64_t                     incx,
                                           int64_t                     batch_count)
{
    return hipblasCtbmvBatched_64(handle,
                                  uplo,
                                  transA,
                                  diag,
                                  m,
                                  k,
                                  (const hipComplex* const*)A,
                                  lda,
                                  (hipComplex* const*)x,
                                  incx,
                                  batch_count);
}

hipblasStatus_t hipblasZtbmvBatchedCast_64(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           hipblasOperation_t                transA,
                                           hipblasDiagType_t                 diag,
                                           int64_t                           m,
                                           int64_t                           k,
                                           const hipblasDoubleComplex* const A[],
                                           int64_t                           lda,
                                           hipblasDoubleComplex* const       x[],
                                           int64_t                           incx,
                                           int64_t                           batch_count)
{
    return hipblasZtbmvBatched_64(handle,
                                  uplo,
                                  transA,
                                  diag,
                                  m,
                                  k,
                                  (const hipDoubleComplex* const*)A,
                                  lda,
                                  (hipDoubleComplex* const*)x,
                                  incx,
                                  batch_count);
}

// tbmv_strided_batched
hipblasStatus_t hipblasCtbmvStridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasFillMode_t     uplo,
                                               hipblasOperation_t    transA,
                                               hipblasDiagType_t     diag,
                                               int                   m,
                                               int                   k,
                                               const hipblasComplex* A,
                                               int                   lda,
                                               hipblasStride         stride_a,
                                               hipblasComplex*       x,
                                               int                   incx,
                                               hipblasStride         stride_x,
                                               int                   batch_count)
{
    return hipblasCtbmvStridedBatched(handle,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      k,
                                      (const hipComplex*)A,
                                      lda,
                                      stride_a,
                                      (hipComplex*)x,
                                      incx,
                                      stride_x,
                                      batch_count);
}

hipblasStatus_t hipblasZtbmvStridedBatchedCast(hipblasHandle_t             handle,
                                               hipblasFillMode_t           uplo,
                                               hipblasOperation_t          transA,
                                               hipblasDiagType_t           diag,
                                               int                         m,
                                               int                         k,
                                               const hipblasDoubleComplex* A,
                                               int                         lda,
                                               hipblasStride               stride_a,
                                               hipblasDoubleComplex*       x,
                                               int                         incx,
                                               hipblasStride               stride_x,
                                               int                         batch_count)
{
    return hipblasZtbmvStridedBatched(handle,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      k,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      stride_a,
                                      (hipDoubleComplex*)x,
                                      incx,
                                      stride_x,
                                      batch_count);
}

// tbmv_strided_batched_64
hipblasStatus_t hipblasCtbmvStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  hipblasOperation_t    transA,
                                                  hipblasDiagType_t     diag,
                                                  int64_t               m,
                                                  int64_t               k,
                                                  const hipblasComplex* A,
                                                  int64_t               lda,
                                                  hipblasStride         stride_a,
                                                  hipblasComplex*       x,
                                                  int64_t               incx,
                                                  hipblasStride         stride_x,
                                                  int64_t               batch_count)
{
    return hipblasCtbmvStridedBatched_64(handle,
                                         uplo,
                                         transA,
                                         diag,
                                         m,
                                         k,
                                         (const hipComplex*)A,
                                         lda,
                                         stride_a,
                                         (hipComplex*)x,
                                         incx,
                                         stride_x,
                                         batch_count);
}

hipblasStatus_t hipblasZtbmvStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  hipblasOperation_t          transA,
                                                  hipblasDiagType_t           diag,
                                                  int64_t                     m,
                                                  int64_t                     k,
                                                  const hipblasDoubleComplex* A,
                                                  int64_t                     lda,
                                                  hipblasStride               stride_a,
                                                  hipblasDoubleComplex*       x,
                                                  int64_t                     incx,
                                                  hipblasStride               stride_x,
                                                  int64_t                     batch_count)
{
    return hipblasZtbmvStridedBatched_64(handle,
                                         uplo,
                                         transA,
                                         diag,
                                         m,
                                         k,
                                         (const hipDoubleComplex*)A,
                                         lda,
                                         stride_a,
                                         (hipDoubleComplex*)x,
                                         incx,
                                         stride_x,
                                         batch_count);
}

// tbsv
hipblasStatus_t hipblasCtbsvCast(hipblasHandle_t       handle,
                                 hipblasFillMode_t     uplo,
                                 hipblasOperation_t    transA,
                                 hipblasDiagType_t     diag,
                                 int                   n,
                                 int                   k,
                                 const hipblasComplex* A,
                                 int                   lda,
                                 hipblasComplex*       x,
                                 int                   incx)
{
    return hipblasCtbsv(
        handle, uplo, transA, diag, n, k, (const hipComplex*)A, lda, (hipComplex*)x, incx);
}

hipblasStatus_t hipblasZtbsvCast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 hipblasOperation_t          transA,
                                 hipblasDiagType_t           diag,
                                 int                         n,
                                 int                         k,
                                 const hipblasDoubleComplex* A,
                                 int                         lda,
                                 hipblasDoubleComplex*       x,
                                 int                         incx)
{
    return hipblasZtbsv(handle,
                        uplo,
                        transA,
                        diag,
                        n,
                        k,
                        (const hipDoubleComplex*)A,
                        lda,
                        (hipDoubleComplex*)x,
                        incx);
}

// tbsv_64
hipblasStatus_t hipblasCtbsvCast_64(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    hipblasOperation_t    transA,
                                    hipblasDiagType_t     diag,
                                    int64_t               n,
                                    int64_t               k,
                                    const hipblasComplex* A,
                                    int64_t               lda,
                                    hipblasComplex*       x,
                                    int64_t               incx)
{
    return hipblasCtbsv_64(
        handle, uplo, transA, diag, n, k, (const hipComplex*)A, lda, (hipComplex*)x, incx);
}

hipblasStatus_t hipblasZtbsvCast_64(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int64_t                     n,
                                    int64_t                     k,
                                    const hipblasDoubleComplex* A,
                                    int64_t                     lda,
                                    hipblasDoubleComplex*       x,
                                    int64_t                     incx)
{
    return hipblasZtbsv_64(handle,
                           uplo,
                           transA,
                           diag,
                           n,
                           k,
                           (const hipDoubleComplex*)A,
                           lda,
                           (hipDoubleComplex*)x,
                           incx);
}

// tbsv_batched
hipblasStatus_t hipblasCtbsvBatchedCast(hipblasHandle_t             handle,
                                        hipblasFillMode_t           uplo,
                                        hipblasOperation_t          transA,
                                        hipblasDiagType_t           diag,
                                        int                         n,
                                        int                         k,
                                        const hipblasComplex* const A[],
                                        int                         lda,
                                        hipblasComplex* const       x[],
                                        int                         incx,
                                        int                         batchCount)
{
    return hipblasCtbsvBatched(handle,
                               uplo,
                               transA,
                               diag,
                               n,
                               k,
                               (const hipComplex* const*)A,
                               lda,
                               (hipComplex* const*)x,
                               incx,
                               batchCount);
}

hipblasStatus_t hipblasZtbsvBatchedCast(hipblasHandle_t                   handle,
                                        hipblasFillMode_t                 uplo,
                                        hipblasOperation_t                transA,
                                        hipblasDiagType_t                 diag,
                                        int                               n,
                                        int                               k,
                                        const hipblasDoubleComplex* const A[],
                                        int                               lda,
                                        hipblasDoubleComplex* const       x[],
                                        int                               incx,
                                        int                               batchCount)
{
    return hipblasZtbsvBatched(handle,
                               uplo,
                               transA,
                               diag,
                               n,
                               k,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (hipDoubleComplex* const*)x,
                               incx,
                               batchCount);
}

// tbsv_batched_64
hipblasStatus_t hipblasCtbsvBatchedCast_64(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int64_t                     n,
                                           int64_t                     k,
                                           const hipblasComplex* const A[],
                                           int64_t                     lda,
                                           hipblasComplex* const       x[],
                                           int64_t                     incx,
                                           int64_t                     batchCount)
{
    return hipblasCtbsvBatched_64(handle,
                                  uplo,
                                  transA,
                                  diag,
                                  n,
                                  k,
                                  (const hipComplex* const*)A,
                                  lda,
                                  (hipComplex* const*)x,
                                  incx,
                                  batchCount);
}

hipblasStatus_t hipblasZtbsvBatchedCast_64(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           hipblasOperation_t                transA,
                                           hipblasDiagType_t                 diag,
                                           int64_t                           n,
                                           int64_t                           k,
                                           const hipblasDoubleComplex* const A[],
                                           int64_t                           lda,
                                           hipblasDoubleComplex* const       x[],
                                           int64_t                           incx,
                                           int64_t                           batchCount)
{
    return hipblasZtbsvBatched_64(handle,
                                  uplo,
                                  transA,
                                  diag,
                                  n,
                                  k,
                                  (const hipDoubleComplex* const*)A,
                                  lda,
                                  (hipDoubleComplex* const*)x,
                                  incx,
                                  batchCount);
}

// tbsv_strided_batched
hipblasStatus_t hipblasCtbsvStridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasFillMode_t     uplo,
                                               hipblasOperation_t    transA,
                                               hipblasDiagType_t     diag,
                                               int                   n,
                                               int                   k,
                                               const hipblasComplex* A,
                                               int                   lda,
                                               hipblasStride         strideA,
                                               hipblasComplex*       x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               int                   batchCount)
{
    return hipblasCtbsvStridedBatched(handle,
                                      uplo,
                                      transA,
                                      diag,
                                      n,
                                      k,
                                      (const hipComplex*)A,
                                      lda,
                                      strideA,
                                      (hipComplex*)x,
                                      incx,
                                      stridex,
                                      batchCount);
}

hipblasStatus_t hipblasZtbsvStridedBatchedCast(hipblasHandle_t             handle,
                                               hipblasFillMode_t           uplo,
                                               hipblasOperation_t          transA,
                                               hipblasDiagType_t           diag,
                                               int                         n,
                                               int                         k,
                                               const hipblasDoubleComplex* A,
                                               int                         lda,
                                               hipblasStride               strideA,
                                               hipblasDoubleComplex*       x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               int                         batchCount)
{
    return hipblasZtbsvStridedBatched(handle,
                                      uplo,
                                      transA,
                                      diag,
                                      n,
                                      k,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      (hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      batchCount);
}

// tbsv_strided_batched_64
hipblasStatus_t hipblasCtbsvStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  hipblasOperation_t    transA,
                                                  hipblasDiagType_t     diag,
                                                  int64_t               n,
                                                  int64_t               k,
                                                  const hipblasComplex* A,
                                                  int64_t               lda,
                                                  hipblasStride         strideA,
                                                  hipblasComplex*       x,
                                                  int64_t               incx,
                                                  hipblasStride         stridex,
                                                  int64_t               batchCount)
{
    return hipblasCtbsvStridedBatched_64(handle,
                                         uplo,
                                         transA,
                                         diag,
                                         n,
                                         k,
                                         (const hipComplex*)A,
                                         lda,
                                         strideA,
                                         (hipComplex*)x,
                                         incx,
                                         stridex,
                                         batchCount);
}

hipblasStatus_t hipblasZtbsvStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  hipblasOperation_t          transA,
                                                  hipblasDiagType_t           diag,
                                                  int64_t                     n,
                                                  int64_t                     k,
                                                  const hipblasDoubleComplex* A,
                                                  int64_t                     lda,
                                                  hipblasStride               strideA,
                                                  hipblasDoubleComplex*       x,
                                                  int64_t                     incx,
                                                  hipblasStride               stridex,
                                                  int64_t                     batchCount)
{
    return hipblasZtbsvStridedBatched_64(handle,
                                         uplo,
                                         transA,
                                         diag,
                                         n,
                                         k,
                                         (const hipDoubleComplex*)A,
                                         lda,
                                         strideA,
                                         (hipDoubleComplex*)x,
                                         incx,
                                         stridex,
                                         batchCount);
}

// tpmv
hipblasStatus_t hipblasCtpmvCast(hipblasHandle_t       handle,
                                 hipblasFillMode_t     uplo,
                                 hipblasOperation_t    transA,
                                 hipblasDiagType_t     diag,
                                 int                   m,
                                 const hipblasComplex* AP,
                                 hipblasComplex*       x,
                                 int                   incx)
{
    return hipblasCtpmv(handle, uplo, transA, diag, m, (const hipComplex*)AP, (hipComplex*)x, incx);
}

hipblasStatus_t hipblasZtpmvCast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 hipblasOperation_t          transA,
                                 hipblasDiagType_t           diag,
                                 int                         m,
                                 const hipblasDoubleComplex* AP,
                                 hipblasDoubleComplex*       x,
                                 int                         incx)
{
    return hipblasZtpmv(
        handle, uplo, transA, diag, m, (const hipDoubleComplex*)AP, (hipDoubleComplex*)x, incx);
}

// tpmv_64
hipblasStatus_t hipblasCtpmvCast_64(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    hipblasOperation_t    transA,
                                    hipblasDiagType_t     diag,
                                    int64_t               m,
                                    const hipblasComplex* AP,
                                    hipblasComplex*       x,
                                    int64_t               incx)
{
    return hipblasCtpmv_64(
        handle, uplo, transA, diag, m, (const hipComplex*)AP, (hipComplex*)x, incx);
}

hipblasStatus_t hipblasZtpmvCast_64(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int64_t                     m,
                                    const hipblasDoubleComplex* AP,
                                    hipblasDoubleComplex*       x,
                                    int64_t                     incx)
{
    return hipblasZtpmv_64(
        handle, uplo, transA, diag, m, (const hipDoubleComplex*)AP, (hipDoubleComplex*)x, incx);
}

// tpmv_batched
hipblasStatus_t hipblasCtpmvBatchedCast(hipblasHandle_t             handle,
                                        hipblasFillMode_t           uplo,
                                        hipblasOperation_t          transA,
                                        hipblasDiagType_t           diag,
                                        int                         m,
                                        const hipblasComplex* const AP[],
                                        hipblasComplex* const       x[],
                                        int                         incx,
                                        int                         batchCount)
{
    return hipblasCtpmvBatched(handle,
                               uplo,
                               transA,
                               diag,
                               m,
                               (const hipComplex* const*)AP,
                               (hipComplex* const*)x,
                               incx,
                               batchCount);
}

hipblasStatus_t hipblasZtpmvBatchedCast(hipblasHandle_t                   handle,
                                        hipblasFillMode_t                 uplo,
                                        hipblasOperation_t                transA,
                                        hipblasDiagType_t                 diag,
                                        int                               m,
                                        const hipblasDoubleComplex* const AP[],
                                        hipblasDoubleComplex* const       x[],
                                        int                               incx,
                                        int                               batchCount)
{
    return hipblasZtpmvBatched(handle,
                               uplo,
                               transA,
                               diag,
                               m,
                               (const hipDoubleComplex* const*)AP,
                               (hipDoubleComplex* const*)x,
                               incx,
                               batchCount);
}

// tpmv_batched_64
hipblasStatus_t hipblasCtpmvBatchedCast_64(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int64_t                     m,
                                           const hipblasComplex* const AP[],
                                           hipblasComplex* const       x[],
                                           int64_t                     incx,
                                           int64_t                     batchCount)
{
    return hipblasCtpmvBatched_64(handle,
                                  uplo,
                                  transA,
                                  diag,
                                  m,
                                  (const hipComplex* const*)AP,
                                  (hipComplex* const*)x,
                                  incx,
                                  batchCount);
}

hipblasStatus_t hipblasZtpmvBatchedCast_64(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           hipblasOperation_t                transA,
                                           hipblasDiagType_t                 diag,
                                           int64_t                           m,
                                           const hipblasDoubleComplex* const AP[],
                                           hipblasDoubleComplex* const       x[],
                                           int64_t                           incx,
                                           int64_t                           batchCount)
{
    return hipblasZtpmvBatched_64(handle,
                                  uplo,
                                  transA,
                                  diag,
                                  m,
                                  (const hipDoubleComplex* const*)AP,
                                  (hipDoubleComplex* const*)x,
                                  incx,
                                  batchCount);
}

// tpmv_strided_batched
hipblasStatus_t hipblasCtpmvStridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasFillMode_t     uplo,
                                               hipblasOperation_t    transA,
                                               hipblasDiagType_t     diag,
                                               int                   m,
                                               const hipblasComplex* AP,
                                               hipblasStride         strideAP,
                                               hipblasComplex*       x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               int                   batchCount)
{
    return hipblasCtpmvStridedBatched(handle,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      (const hipComplex*)AP,
                                      strideAP,
                                      (hipComplex*)x,
                                      incx,
                                      stridex,
                                      batchCount);
}

hipblasStatus_t hipblasZtpmvStridedBatchedCast(hipblasHandle_t             handle,
                                               hipblasFillMode_t           uplo,
                                               hipblasOperation_t          transA,
                                               hipblasDiagType_t           diag,
                                               int                         m,
                                               const hipblasDoubleComplex* AP,
                                               hipblasStride               strideAP,
                                               hipblasDoubleComplex*       x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               int                         batchCount)
{
    return hipblasZtpmvStridedBatched(handle,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      (const hipDoubleComplex*)AP,
                                      strideAP,
                                      (hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      batchCount);
}

// tpmv_strided_batched_64
hipblasStatus_t hipblasCtpmvStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  hipblasOperation_t    transA,
                                                  hipblasDiagType_t     diag,
                                                  int64_t               m,
                                                  const hipblasComplex* AP,
                                                  hipblasStride         strideAP,
                                                  hipblasComplex*       x,
                                                  int64_t               incx,
                                                  hipblasStride         stridex,
                                                  int64_t               batchCount)
{
    return hipblasCtpmvStridedBatched_64(handle,
                                         uplo,
                                         transA,
                                         diag,
                                         m,
                                         (const hipComplex*)AP,
                                         strideAP,
                                         (hipComplex*)x,
                                         incx,
                                         stridex,
                                         batchCount);
}

hipblasStatus_t hipblasZtpmvStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  hipblasOperation_t          transA,
                                                  hipblasDiagType_t           diag,
                                                  int64_t                     m,
                                                  const hipblasDoubleComplex* AP,
                                                  hipblasStride               strideAP,
                                                  hipblasDoubleComplex*       x,
                                                  int64_t                     incx,
                                                  hipblasStride               stridex,
                                                  int64_t                     batchCount)
{
    return hipblasZtpmvStridedBatched_64(handle,
                                         uplo,
                                         transA,
                                         diag,
                                         m,
                                         (const hipDoubleComplex*)AP,
                                         strideAP,
                                         (hipDoubleComplex*)x,
                                         incx,
                                         stridex,
                                         batchCount);
}

// tpsv
hipblasStatus_t hipblasCtpsvCast(hipblasHandle_t       handle,
                                 hipblasFillMode_t     uplo,
                                 hipblasOperation_t    transA,
                                 hipblasDiagType_t     diag,
                                 int                   m,
                                 const hipblasComplex* AP,
                                 hipblasComplex*       x,
                                 int                   incx)
{
    return hipblasCtpsv(handle, uplo, transA, diag, m, (const hipComplex*)AP, (hipComplex*)x, incx);
}

hipblasStatus_t hipblasZtpsvCast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 hipblasOperation_t          transA,
                                 hipblasDiagType_t           diag,
                                 int                         m,
                                 const hipblasDoubleComplex* AP,
                                 hipblasDoubleComplex*       x,
                                 int                         incx)
{
    return hipblasZtpsv(
        handle, uplo, transA, diag, m, (const hipDoubleComplex*)AP, (hipDoubleComplex*)x, incx);
}

// tpsv_64
hipblasStatus_t hipblasCtpsvCast_64(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    hipblasOperation_t    transA,
                                    hipblasDiagType_t     diag,
                                    int64_t               m,
                                    const hipblasComplex* AP,
                                    hipblasComplex*       x,
                                    int64_t               incx)
{
    return hipblasCtpsv_64(
        handle, uplo, transA, diag, m, (const hipComplex*)AP, (hipComplex*)x, incx);
}

hipblasStatus_t hipblasZtpsvCast_64(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int64_t                     m,
                                    const hipblasDoubleComplex* AP,
                                    hipblasDoubleComplex*       x,
                                    int64_t                     incx)
{
    return hipblasZtpsv_64(
        handle, uplo, transA, diag, m, (const hipDoubleComplex*)AP, (hipDoubleComplex*)x, incx);
}

// tpsv_batched
hipblasStatus_t hipblasCtpsvBatchedCast(hipblasHandle_t             handle,
                                        hipblasFillMode_t           uplo,
                                        hipblasOperation_t          transA,
                                        hipblasDiagType_t           diag,
                                        int                         m,
                                        const hipblasComplex* const AP[],
                                        hipblasComplex* const       x[],
                                        int                         incx,
                                        int                         batchCount)
{
    return hipblasCtpsvBatched(handle,
                               uplo,
                               transA,
                               diag,
                               m,
                               (const hipComplex* const*)AP,
                               (hipComplex* const*)x,
                               incx,
                               batchCount);
}

hipblasStatus_t hipblasZtpsvBatchedCast(hipblasHandle_t                   handle,
                                        hipblasFillMode_t                 uplo,
                                        hipblasOperation_t                transA,
                                        hipblasDiagType_t                 diag,
                                        int                               m,
                                        const hipblasDoubleComplex* const AP[],
                                        hipblasDoubleComplex* const       x[],
                                        int                               incx,
                                        int                               batchCount)
{
    return hipblasZtpsvBatched(handle,
                               uplo,
                               transA,
                               diag,
                               m,
                               (const hipDoubleComplex* const*)AP,
                               (hipDoubleComplex* const*)x,
                               incx,
                               batchCount);
}

// tpsv_batched_64
hipblasStatus_t hipblasCtpsvBatchedCast_64(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int64_t                     m,
                                           const hipblasComplex* const AP[],
                                           hipblasComplex* const       x[],
                                           int64_t                     incx,
                                           int64_t                     batchCount)
{
    return hipblasCtpsvBatched_64(handle,
                                  uplo,
                                  transA,
                                  diag,
                                  m,
                                  (const hipComplex* const*)AP,
                                  (hipComplex* const*)x,
                                  incx,
                                  batchCount);
}

hipblasStatus_t hipblasZtpsvBatchedCast_64(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           hipblasOperation_t                transA,
                                           hipblasDiagType_t                 diag,
                                           int64_t                           m,
                                           const hipblasDoubleComplex* const AP[],
                                           hipblasDoubleComplex* const       x[],
                                           int64_t                           incx,
                                           int64_t                           batchCount)
{
    return hipblasZtpsvBatched_64(handle,
                                  uplo,
                                  transA,
                                  diag,
                                  m,
                                  (const hipDoubleComplex* const*)AP,
                                  (hipDoubleComplex* const*)x,
                                  incx,
                                  batchCount);
}

// tpsv_strided_batched
hipblasStatus_t hipblasCtpsvStridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasFillMode_t     uplo,
                                               hipblasOperation_t    transA,
                                               hipblasDiagType_t     diag,
                                               int                   m,
                                               const hipblasComplex* AP,
                                               hipblasStride         strideAP,
                                               hipblasComplex*       x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               int                   batchCount)
{
    return hipblasCtpsvStridedBatched(handle,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      (const hipComplex*)AP,
                                      strideAP,
                                      (hipComplex*)x,
                                      incx,
                                      stridex,
                                      batchCount);
}

hipblasStatus_t hipblasZtpsvStridedBatchedCast(hipblasHandle_t             handle,
                                               hipblasFillMode_t           uplo,
                                               hipblasOperation_t          transA,
                                               hipblasDiagType_t           diag,
                                               int                         m,
                                               const hipblasDoubleComplex* AP,
                                               hipblasStride               strideAP,
                                               hipblasDoubleComplex*       x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               int                         batchCount)
{
    return hipblasZtpsvStridedBatched(handle,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      (const hipDoubleComplex*)AP,
                                      strideAP,
                                      (hipDoubleComplex*)x,
                                      incx,
                                      stridex,
                                      batchCount);
}

// tpsv_strided_batched_64
hipblasStatus_t hipblasCtpsvStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  hipblasOperation_t    transA,
                                                  hipblasDiagType_t     diag,
                                                  int64_t               m,
                                                  const hipblasComplex* AP,
                                                  hipblasStride         strideAP,
                                                  hipblasComplex*       x,
                                                  int64_t               incx,
                                                  hipblasStride         stridex,
                                                  int64_t               batchCount)
{
    return hipblasCtpsvStridedBatched_64(handle,
                                         uplo,
                                         transA,
                                         diag,
                                         m,
                                         (const hipComplex*)AP,
                                         strideAP,
                                         (hipComplex*)x,
                                         incx,
                                         stridex,
                                         batchCount);
}

hipblasStatus_t hipblasZtpsvStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  hipblasOperation_t          transA,
                                                  hipblasDiagType_t           diag,
                                                  int64_t                     m,
                                                  const hipblasDoubleComplex* AP,
                                                  hipblasStride               strideAP,
                                                  hipblasDoubleComplex*       x,
                                                  int64_t                     incx,
                                                  hipblasStride               stridex,
                                                  int64_t                     batchCount)
{
    return hipblasZtpsvStridedBatched_64(handle,
                                         uplo,
                                         transA,
                                         diag,
                                         m,
                                         (const hipDoubleComplex*)AP,
                                         strideAP,
                                         (hipDoubleComplex*)x,
                                         incx,
                                         stridex,
                                         batchCount);
}

// trmv
hipblasStatus_t hipblasCtrmvCast(hipblasHandle_t       handle,
                                 hipblasFillMode_t     uplo,
                                 hipblasOperation_t    transA,
                                 hipblasDiagType_t     diag,
                                 int                   m,
                                 const hipblasComplex* A,
                                 int                   lda,
                                 hipblasComplex*       x,
                                 int                   incx)
{
    return hipblasCtrmv(
        handle, uplo, transA, diag, m, (const hipComplex*)A, lda, (hipComplex*)x, incx);
}

hipblasStatus_t hipblasZtrmvCast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 hipblasOperation_t          transA,
                                 hipblasDiagType_t           diag,
                                 int                         m,
                                 const hipblasDoubleComplex* A,
                                 int                         lda,
                                 hipblasDoubleComplex*       x,
                                 int                         incx)
{
    return hipblasZtrmv(
        handle, uplo, transA, diag, m, (const hipDoubleComplex*)A, lda, (hipDoubleComplex*)x, incx);
}

// trmv_64
hipblasStatus_t hipblasCtrmvCast_64(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    hipblasOperation_t    transA,
                                    hipblasDiagType_t     diag,
                                    int64_t               m,
                                    const hipblasComplex* A,
                                    int64_t               lda,
                                    hipblasComplex*       x,
                                    int64_t               incx)
{
    return hipblasCtrmv_64(
        handle, uplo, transA, diag, m, (const hipComplex*)A, lda, (hipComplex*)x, incx);
}

hipblasStatus_t hipblasZtrmvCast_64(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int64_t                     m,
                                    const hipblasDoubleComplex* A,
                                    int64_t                     lda,
                                    hipblasDoubleComplex*       x,
                                    int64_t                     incx)
{
    return hipblasZtrmv_64(
        handle, uplo, transA, diag, m, (const hipDoubleComplex*)A, lda, (hipDoubleComplex*)x, incx);
}

// trmv_batched
hipblasStatus_t hipblasCtrmvBatchedCast(hipblasHandle_t             handle,
                                        hipblasFillMode_t           uplo,
                                        hipblasOperation_t          transA,
                                        hipblasDiagType_t           diag,
                                        int                         m,
                                        const hipblasComplex* const A[],
                                        int                         lda,
                                        hipblasComplex* const       x[],
                                        int                         incx,
                                        int                         batch_count)
{
    return hipblasCtrmvBatched(handle,
                               uplo,
                               transA,
                               diag,
                               m,
                               (const hipComplex* const*)A,
                               lda,
                               (hipComplex* const*)x,
                               incx,
                               batch_count);
}

hipblasStatus_t hipblasZtrmvBatchedCast(hipblasHandle_t                   handle,
                                        hipblasFillMode_t                 uplo,
                                        hipblasOperation_t                transA,
                                        hipblasDiagType_t                 diag,
                                        int                               m,
                                        const hipblasDoubleComplex* const A[],
                                        int                               lda,
                                        hipblasDoubleComplex* const       x[],
                                        int                               incx,
                                        int                               batch_count)
{
    return hipblasZtrmvBatched(handle,
                               uplo,
                               transA,
                               diag,
                               m,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (hipDoubleComplex* const*)x,
                               incx,
                               batch_count);
}

// trmv_batched_64
hipblasStatus_t hipblasCtrmvBatchedCast_64(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int64_t                     m,
                                           const hipblasComplex* const A[],
                                           int64_t                     lda,
                                           hipblasComplex* const       x[],
                                           int64_t                     incx,
                                           int64_t                     batch_count)
{
    return hipblasCtrmvBatched_64(handle,
                                  uplo,
                                  transA,
                                  diag,
                                  m,
                                  (const hipComplex* const*)A,
                                  lda,
                                  (hipComplex* const*)x,
                                  incx,
                                  batch_count);
}

hipblasStatus_t hipblasZtrmvBatchedCast_64(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           hipblasOperation_t                transA,
                                           hipblasDiagType_t                 diag,
                                           int64_t                           m,
                                           const hipblasDoubleComplex* const A[],
                                           int64_t                           lda,
                                           hipblasDoubleComplex* const       x[],
                                           int64_t                           incx,
                                           int64_t                           batch_count)
{
    return hipblasZtrmvBatched_64(handle,
                                  uplo,
                                  transA,
                                  diag,
                                  m,
                                  (const hipDoubleComplex* const*)A,
                                  lda,
                                  (hipDoubleComplex* const*)x,
                                  incx,
                                  batch_count);
}

// trmv_strided_batched
hipblasStatus_t hipblasCtrmvStridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasFillMode_t     uplo,
                                               hipblasOperation_t    transA,
                                               hipblasDiagType_t     diag,
                                               int                   m,
                                               const hipblasComplex* A,
                                               int                   lda,
                                               hipblasStride         stride_a,
                                               hipblasComplex*       x,
                                               int                   incx,
                                               hipblasStride         stride_x,
                                               int                   batch_count)
{
    return hipblasCtrmvStridedBatched(handle,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      (const hipComplex*)A,
                                      lda,
                                      stride_a,
                                      (hipComplex*)x,
                                      incx,
                                      stride_x,
                                      batch_count);
}

hipblasStatus_t hipblasZtrmvStridedBatchedCast(hipblasHandle_t             handle,
                                               hipblasFillMode_t           uplo,
                                               hipblasOperation_t          transA,
                                               hipblasDiagType_t           diag,
                                               int                         m,
                                               const hipblasDoubleComplex* A,
                                               int                         lda,
                                               hipblasStride               stride_a,
                                               hipblasDoubleComplex*       x,
                                               int                         incx,
                                               hipblasStride               stride_x,
                                               int                         batch_count)
{
    return hipblasZtrmvStridedBatched(handle,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      stride_a,
                                      (hipDoubleComplex*)x,
                                      incx,
                                      stride_x,
                                      batch_count);
}

// trmv_strided_batched_64
hipblasStatus_t hipblasCtrmvStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  hipblasOperation_t    transA,
                                                  hipblasDiagType_t     diag,
                                                  int64_t               m,
                                                  const hipblasComplex* A,
                                                  int64_t               lda,
                                                  hipblasStride         stride_a,
                                                  hipblasComplex*       x,
                                                  int64_t               incx,
                                                  hipblasStride         stride_x,
                                                  int64_t               batch_count)
{
    return hipblasCtrmvStridedBatched_64(handle,
                                         uplo,
                                         transA,
                                         diag,
                                         m,
                                         (const hipComplex*)A,
                                         lda,
                                         stride_a,
                                         (hipComplex*)x,
                                         incx,
                                         stride_x,
                                         batch_count);
}

hipblasStatus_t hipblasZtrmvStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  hipblasOperation_t          transA,
                                                  hipblasDiagType_t           diag,
                                                  int64_t                     m,
                                                  const hipblasDoubleComplex* A,
                                                  int64_t                     lda,
                                                  hipblasStride               stride_a,
                                                  hipblasDoubleComplex*       x,
                                                  int64_t                     incx,
                                                  hipblasStride               stride_x,
                                                  int64_t                     batch_count)
{
    return hipblasZtrmvStridedBatched_64(handle,
                                         uplo,
                                         transA,
                                         diag,
                                         m,
                                         (const hipDoubleComplex*)A,
                                         lda,
                                         stride_a,
                                         (hipDoubleComplex*)x,
                                         incx,
                                         stride_x,
                                         batch_count);
}

/*
 * ===========================================================================
 *    level 3 BLAS
 * ===========================================================================
 */

// trtri
hipblasStatus_t hipblasCtrtriCast(hipblasHandle_t       handle,
                                  hipblasFillMode_t     uplo,
                                  hipblasDiagType_t     diag,
                                  int                   n,
                                  const hipblasComplex* A,
                                  int                   lda,
                                  hipblasComplex*       invA,
                                  int                   ldinvA)
{
    return hipblasCtrtri(
        handle, uplo, diag, n, (const hipComplex*)A, lda, (hipComplex*)invA, ldinvA);
}

hipblasStatus_t hipblasZtrtriCast(hipblasHandle_t             handle,
                                  hipblasFillMode_t           uplo,
                                  hipblasDiagType_t           diag,
                                  int                         n,
                                  const hipblasDoubleComplex* A,
                                  int                         lda,
                                  hipblasDoubleComplex*       invA,
                                  int                         ldinvA)
{
    return hipblasZtrtri(
        handle, uplo, diag, n, (const hipDoubleComplex*)A, lda, (hipDoubleComplex*)invA, ldinvA);
}

// trtri_batched
hipblasStatus_t hipblasCtrtriBatchedCast(hipblasHandle_t             handle,
                                         hipblasFillMode_t           uplo,
                                         hipblasDiagType_t           diag,
                                         int                         n,
                                         const hipblasComplex* const A[],
                                         int                         lda,
                                         hipblasComplex*             invA[],
                                         int                         ldinvA,
                                         int                         batch_count)
{
    return hipblasCtrtriBatched(handle,
                                uplo,
                                diag,
                                n,
                                (const hipComplex* const*)A,
                                lda,
                                (hipComplex**)invA,
                                ldinvA,
                                batch_count);
}

hipblasStatus_t hipblasZtrtriBatchedCast(hipblasHandle_t                   handle,
                                         hipblasFillMode_t                 uplo,
                                         hipblasDiagType_t                 diag,
                                         int                               n,
                                         const hipblasDoubleComplex* const A[],
                                         int                               lda,
                                         hipblasDoubleComplex*             invA[],
                                         int                               ldinvA,
                                         int                               batch_count)
{
    return hipblasZtrtriBatched(handle,
                                uplo,
                                diag,
                                n,
                                (const hipDoubleComplex* const*)A,
                                lda,
                                (hipDoubleComplex**)invA,
                                ldinvA,
                                batch_count);
}

// trtri_strided_batched
hipblasStatus_t hipblasCtrtriStridedBatchedCast(hipblasHandle_t       handle,
                                                hipblasFillMode_t     uplo,
                                                hipblasDiagType_t     diag,
                                                int                   n,
                                                const hipblasComplex* A,
                                                int                   lda,
                                                hipblasStride         stride_A,
                                                hipblasComplex*       invA,
                                                int                   ldinvA,
                                                hipblasStride         stride_invA,
                                                int                   batch_count)
{
    return hipblasCtrtriStridedBatched(handle,
                                       uplo,
                                       diag,
                                       n,
                                       (const hipComplex*)A,
                                       lda,
                                       stride_A,
                                       (hipComplex*)invA,
                                       ldinvA,
                                       stride_invA,
                                       batch_count);
}

hipblasStatus_t hipblasZtrtriStridedBatchedCast(hipblasHandle_t             handle,
                                                hipblasFillMode_t           uplo,
                                                hipblasDiagType_t           diag,
                                                int                         n,
                                                const hipblasDoubleComplex* A,
                                                int                         lda,
                                                hipblasStride               stride_A,
                                                hipblasDoubleComplex*       invA,
                                                int                         ldinvA,
                                                hipblasStride               stride_invA,
                                                int                         batch_count)
{
    return hipblasZtrtriStridedBatched(handle,
                                       uplo,
                                       diag,
                                       n,
                                       (const hipDoubleComplex*)A,
                                       lda,
                                       stride_A,
                                       (hipDoubleComplex*)invA,
                                       ldinvA,
                                       stride_invA,
                                       batch_count);
}

// dgmm
hipblasStatus_t hipblasCdgmmCast(hipblasHandle_t       handle,
                                 hipblasSideMode_t     side,
                                 int                   m,
                                 int                   n,
                                 const hipblasComplex* A,
                                 int                   lda,
                                 const hipblasComplex* x,
                                 int                   incx,
                                 hipblasComplex*       C,
                                 int                   ldc)
{
    return hipblasCdgmm(handle,
                        side,
                        m,
                        n,
                        (const hipComplex*)A,
                        lda,
                        (const hipComplex*)x,
                        incx,
                        (hipComplex*)C,
                        ldc);
}

hipblasStatus_t hipblasZdgmmCast(hipblasHandle_t             handle,
                                 hipblasSideMode_t           side,
                                 int                         m,
                                 int                         n,
                                 const hipblasDoubleComplex* A,
                                 int                         lda,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 hipblasDoubleComplex*       C,
                                 int                         ldc)
{
    return hipblasZdgmm(handle,
                        side,
                        m,
                        n,
                        (const hipDoubleComplex*)A,
                        lda,
                        (const hipDoubleComplex*)x,
                        incx,
                        (hipDoubleComplex*)C,
                        ldc);
}

// dgmm_batched
hipblasStatus_t hipblasCdgmmBatchedCast(hipblasHandle_t             handle,
                                        hipblasSideMode_t           side,
                                        int                         m,
                                        int                         n,
                                        const hipblasComplex* const A[],
                                        int                         lda,
                                        const hipblasComplex* const x[],
                                        int                         incx,
                                        hipblasComplex* const       C[],
                                        int                         ldc,
                                        int                         batch_count)
{
    return hipblasCdgmmBatched(handle,
                               side,
                               m,
                               n,
                               (const hipComplex* const*)A,
                               lda,
                               (const hipComplex* const*)x,
                               incx,
                               (hipComplex* const*)C,
                               ldc,
                               batch_count);
}

hipblasStatus_t hipblasZdgmmBatchedCast(hipblasHandle_t                   handle,
                                        hipblasSideMode_t                 side,
                                        int                               m,
                                        int                               n,
                                        const hipblasDoubleComplex* const A[],
                                        int                               lda,
                                        const hipblasDoubleComplex* const x[],
                                        int                               incx,
                                        hipblasDoubleComplex* const       C[],
                                        int                               ldc,
                                        int                               batch_count)
{
    return hipblasZdgmmBatched(handle,
                               side,
                               m,
                               n,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (const hipDoubleComplex* const*)x,
                               incx,
                               (hipDoubleComplex* const*)C,
                               ldc,
                               batch_count);
}

// dgmm_strided_batched
hipblasStatus_t hipblasCdgmmStridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasSideMode_t     side,
                                               int                   m,
                                               int                   n,
                                               const hipblasComplex* A,
                                               int                   lda,
                                               hipblasStride         stride_A,
                                               const hipblasComplex* x,
                                               int                   incx,
                                               hipblasStride         stride_x,
                                               hipblasComplex*       C,
                                               int                   ldc,
                                               hipblasStride         stride_C,
                                               int                   batch_count)
{
    return hipblasCdgmmStridedBatched(handle,
                                      side,
                                      m,
                                      n,
                                      (const hipComplex*)A,
                                      lda,
                                      stride_A,
                                      (const hipComplex*)x,
                                      incx,
                                      stride_x,
                                      (hipComplex*)C,
                                      ldc,
                                      stride_C,
                                      batch_count);
}

hipblasStatus_t hipblasZdgmmStridedBatchedCast(hipblasHandle_t             handle,
                                               hipblasSideMode_t           side,
                                               int                         m,
                                               int                         n,
                                               const hipblasDoubleComplex* A,
                                               int                         lda,
                                               hipblasStride               stride_A,
                                               const hipblasDoubleComplex* x,
                                               int                         incx,
                                               hipblasStride               stride_x,
                                               hipblasDoubleComplex*       C,
                                               int                         ldc,
                                               hipblasStride               stride_C,
                                               int                         batch_count)
{
    return hipblasZdgmmStridedBatched(handle,
                                      side,
                                      m,
                                      n,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      stride_A,
                                      (const hipDoubleComplex*)x,
                                      incx,
                                      stride_x,
                                      (hipDoubleComplex*)C,
                                      ldc,
                                      stride_C,
                                      batch_count);
}

// dgmm_64
hipblasStatus_t hipblasCdgmmCast_64(hipblasHandle_t       handle,
                                    hipblasSideMode_t     side,
                                    int64_t               m,
                                    int64_t               n,
                                    const hipblasComplex* A,
                                    int64_t               lda,
                                    const hipblasComplex* x,
                                    int64_t               incx,
                                    hipblasComplex*       C,
                                    int64_t               ldc)
{
    return hipblasCdgmm_64(handle,
                           side,
                           m,
                           n,
                           (const hipComplex*)A,
                           lda,
                           (const hipComplex*)x,
                           incx,
                           (hipComplex*)C,
                           ldc);
}

hipblasStatus_t hipblasZdgmmCast_64(hipblasHandle_t             handle,
                                    hipblasSideMode_t           side,
                                    int64_t                     m,
                                    int64_t                     n,
                                    const hipblasDoubleComplex* A,
                                    int64_t                     lda,
                                    const hipblasDoubleComplex* x,
                                    int64_t                     incx,
                                    hipblasDoubleComplex*       C,
                                    int64_t                     ldc)
{
    return hipblasZdgmm_64(handle,
                           side,
                           m,
                           n,
                           (const hipDoubleComplex*)A,
                           lda,
                           (const hipDoubleComplex*)x,
                           incx,
                           (hipDoubleComplex*)C,
                           ldc);
}

// dgmm_batched_64
hipblasStatus_t hipblasCdgmmBatchedCast_64(hipblasHandle_t             handle,
                                           hipblasSideMode_t           side,
                                           int64_t                     m,
                                           int64_t                     n,
                                           const hipblasComplex* const A[],
                                           int64_t                     lda,
                                           const hipblasComplex* const x[],
                                           int64_t                     incx,
                                           hipblasComplex* const       C[],
                                           int64_t                     ldc,
                                           int64_t                     batch_count)
{
    return hipblasCdgmmBatched_64(handle,
                                  side,
                                  m,
                                  n,
                                  (const hipComplex* const*)A,
                                  lda,
                                  (const hipComplex* const*)x,
                                  incx,
                                  (hipComplex* const*)C,
                                  ldc,
                                  batch_count);
}

hipblasStatus_t hipblasZdgmmBatchedCast_64(hipblasHandle_t                   handle,
                                           hipblasSideMode_t                 side,
                                           int64_t                           m,
                                           int64_t                           n,
                                           const hipblasDoubleComplex* const A[],
                                           int64_t                           lda,
                                           const hipblasDoubleComplex* const x[],
                                           int64_t                           incx,
                                           hipblasDoubleComplex* const       C[],
                                           int64_t                           ldc,
                                           int64_t                           batch_count)
{
    return hipblasZdgmmBatched_64(handle,
                                  side,
                                  m,
                                  n,
                                  (const hipDoubleComplex* const*)A,
                                  lda,
                                  (const hipDoubleComplex* const*)x,
                                  incx,
                                  (hipDoubleComplex* const*)C,
                                  ldc,
                                  batch_count);
}

// dgmm_strided_batched_64
hipblasStatus_t hipblasCdgmmStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  hipblasSideMode_t     side,
                                                  int64_t               m,
                                                  int64_t               n,
                                                  const hipblasComplex* A,
                                                  int64_t               lda,
                                                  hipblasStride         stride_A,
                                                  const hipblasComplex* x,
                                                  int64_t               incx,
                                                  hipblasStride         stride_x,
                                                  hipblasComplex*       C,
                                                  int64_t               ldc,
                                                  hipblasStride         stride_C,
                                                  int64_t               batch_count)
{
    return hipblasCdgmmStridedBatched_64(handle,
                                         side,
                                         m,
                                         n,
                                         (const hipComplex*)A,
                                         lda,
                                         stride_A,
                                         (const hipComplex*)x,
                                         incx,
                                         stride_x,
                                         (hipComplex*)C,
                                         ldc,
                                         stride_C,
                                         batch_count);
}

hipblasStatus_t hipblasZdgmmStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  hipblasSideMode_t           side,
                                                  int64_t                     m,
                                                  int64_t                     n,
                                                  const hipblasDoubleComplex* A,
                                                  int64_t                     lda,
                                                  hipblasStride               stride_A,
                                                  const hipblasDoubleComplex* x,
                                                  int64_t                     incx,
                                                  hipblasStride               stride_x,
                                                  hipblasDoubleComplex*       C,
                                                  int64_t                     ldc,
                                                  hipblasStride               stride_C,
                                                  int64_t                     batch_count)
{
    return hipblasZdgmmStridedBatched_64(handle,
                                         side,
                                         m,
                                         n,
                                         (const hipDoubleComplex*)A,
                                         lda,
                                         stride_A,
                                         (const hipDoubleComplex*)x,
                                         incx,
                                         stride_x,
                                         (hipDoubleComplex*)C,
                                         ldc,
                                         stride_C,
                                         batch_count);
}

// gemm
hipblasStatus_t hipblasCgemmCast(hipblasHandle_t       handle,
                                 hipblasOperation_t    transA,
                                 hipblasOperation_t    transB,
                                 int                   m,
                                 int                   n,
                                 int                   k,
                                 const hipblasComplex* alpha,
                                 const hipblasComplex* A,
                                 int                   lda,
                                 const hipblasComplex* B,
                                 int                   ldb,
                                 const hipblasComplex* beta,
                                 hipblasComplex*       C,
                                 int                   ldc)
{
    return hipblasCgemm(handle,
                        transA,
                        transB,
                        m,
                        n,
                        k,
                        (const hipComplex*)alpha,
                        (const hipComplex*)A,
                        lda,
                        (const hipComplex*)B,
                        ldb,
                        (const hipComplex*)beta,
                        (hipComplex*)C,
                        ldc);
}

hipblasStatus_t hipblasZgemmCast(hipblasHandle_t             handle,
                                 hipblasOperation_t          transA,
                                 hipblasOperation_t          transB,
                                 int                         m,
                                 int                         n,
                                 int                         k,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* A,
                                 int                         lda,
                                 const hipblasDoubleComplex* B,
                                 int                         ldb,
                                 const hipblasDoubleComplex* beta,
                                 hipblasDoubleComplex*       C,
                                 int                         ldc)
{
    return hipblasZgemm(handle,
                        transA,
                        transB,
                        m,
                        n,
                        k,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)A,
                        lda,
                        (const hipDoubleComplex*)B,
                        ldb,
                        (const hipDoubleComplex*)beta,
                        (hipDoubleComplex*)C,
                        ldc);
}

// gemm_batched
hipblasStatus_t hipblasCgemmBatchedCast(hipblasHandle_t             handle,
                                        hipblasOperation_t          transA,
                                        hipblasOperation_t          transB,
                                        int                         m,
                                        int                         n,
                                        int                         k,
                                        const hipblasComplex*       alpha,
                                        const hipblasComplex* const A[],
                                        int                         lda,
                                        const hipblasComplex* const B[],
                                        int                         ldb,
                                        const hipblasComplex*       beta,
                                        hipblasComplex* const       C[],
                                        int                         ldc,
                                        int                         batch_count)
{
    return hipblasCgemmBatched(handle,
                               transA,
                               transB,
                               m,
                               n,
                               k,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)A,
                               lda,
                               (const hipComplex* const*)B,
                               ldb,
                               (const hipComplex*)beta,
                               (hipComplex* const*)C,
                               ldc,
                               batch_count);
}

hipblasStatus_t hipblasZgemmBatchedCast(hipblasHandle_t                   handle,
                                        hipblasOperation_t                transA,
                                        hipblasOperation_t                transB,
                                        int                               m,
                                        int                               n,
                                        int                               k,
                                        const hipblasDoubleComplex*       alpha,
                                        const hipblasDoubleComplex* const A[],
                                        int                               lda,
                                        const hipblasDoubleComplex* const B[],
                                        int                               ldb,
                                        const hipblasDoubleComplex*       beta,
                                        hipblasDoubleComplex* const       C[],
                                        int                               ldc,
                                        int                               batch_count)
{
    return hipblasZgemmBatched(handle,
                               transA,
                               transB,
                               m,
                               n,
                               k,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (const hipDoubleComplex* const*)B,
                               ldb,
                               (const hipDoubleComplex*)beta,
                               (hipDoubleComplex* const*)C,
                               ldc,
                               batch_count);
}

// gemm_strided_batched
hipblasStatus_t hipblasCgemmStridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasOperation_t    transA,
                                               hipblasOperation_t    transB,
                                               int                   m,
                                               int                   n,
                                               int                   k,
                                               const hipblasComplex* alpha,
                                               const hipblasComplex* A,
                                               int                   lda,
                                               int                   bsa,
                                               const hipblasComplex* B,
                                               int                   ldb,
                                               int                   bsb,
                                               const hipblasComplex* beta,
                                               hipblasComplex*       C,
                                               int                   ldc,
                                               int                   bsc,
                                               int                   batch_count)
{
    return hipblasCgemmStridedBatched(handle,
                                      transA,
                                      transB,
                                      m,
                                      n,
                                      k,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)A,
                                      lda,
                                      bsa,
                                      (const hipComplex*)B,
                                      ldb,
                                      bsb,
                                      (const hipComplex*)beta,
                                      (hipComplex*)C,
                                      ldc,
                                      bsc,
                                      batch_count);
}

hipblasStatus_t hipblasZgemmStridedBatchedCast(hipblasHandle_t             handle,
                                               hipblasOperation_t          transA,
                                               hipblasOperation_t          transB,
                                               int                         m,
                                               int                         n,
                                               int                         k,
                                               const hipblasDoubleComplex* alpha,
                                               const hipblasDoubleComplex* A,
                                               int                         lda,
                                               int                         bsa,
                                               const hipblasDoubleComplex* B,
                                               int                         ldb,
                                               int                         bsb,
                                               const hipblasDoubleComplex* beta,
                                               hipblasDoubleComplex*       C,
                                               int                         ldc,
                                               int                         bsc,
                                               int                         batch_count)
{
    return hipblasZgemmStridedBatched(handle,
                                      transA,
                                      transB,
                                      m,
                                      n,
                                      k,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      bsa,
                                      (const hipDoubleComplex*)B,
                                      ldb,
                                      bsb,
                                      (const hipDoubleComplex*)beta,
                                      (hipDoubleComplex*)C,
                                      ldc,
                                      bsc,
                                      batch_count);
}

// gemm_64
hipblasStatus_t hipblasCgemmCast_64(hipblasHandle_t       handle,
                                    hipblasOperation_t    transA,
                                    hipblasOperation_t    transB,
                                    int64_t               m,
                                    int64_t               n,
                                    int64_t               k,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* A,
                                    int64_t               lda,
                                    const hipblasComplex* B,
                                    int64_t               ldb,
                                    const hipblasComplex* beta,
                                    hipblasComplex*       C,
                                    int64_t               ldc)
{
    return hipblasCgemm_64(handle,
                           transA,
                           transB,
                           m,
                           n,
                           k,
                           (const hipComplex*)alpha,
                           (const hipComplex*)A,
                           lda,
                           (const hipComplex*)B,
                           ldb,
                           (const hipComplex*)beta,
                           (hipComplex*)C,
                           ldc);
}

hipblasStatus_t hipblasZgemmCast_64(hipblasHandle_t             handle,
                                    hipblasOperation_t          transA,
                                    hipblasOperation_t          transB,
                                    int64_t                     m,
                                    int64_t                     n,
                                    int64_t                     k,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* A,
                                    int64_t                     lda,
                                    const hipblasDoubleComplex* B,
                                    int64_t                     ldb,
                                    const hipblasDoubleComplex* beta,
                                    hipblasDoubleComplex*       C,
                                    int64_t                     ldc)
{
    return hipblasZgemm_64(handle,
                           transA,
                           transB,
                           m,
                           n,
                           k,
                           (const hipDoubleComplex*)alpha,
                           (const hipDoubleComplex*)A,
                           lda,
                           (const hipDoubleComplex*)B,
                           ldb,
                           (const hipDoubleComplex*)beta,
                           (hipDoubleComplex*)C,
                           ldc);
}

// gemm_batched_64
hipblasStatus_t hipblasCgemmBatchedCast_64(hipblasHandle_t             handle,
                                           hipblasOperation_t          transA,
                                           hipblasOperation_t          transB,
                                           int64_t                     m,
                                           int64_t                     n,
                                           int64_t                     k,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const A[],
                                           int64_t                     lda,
                                           const hipblasComplex* const B[],
                                           int64_t                     ldb,
                                           const hipblasComplex*       beta,
                                           hipblasComplex* const       C[],
                                           int64_t                     ldc,
                                           int64_t                     batch_count)
{
    return hipblasCgemmBatched_64(handle,
                                  transA,
                                  transB,
                                  m,
                                  n,
                                  k,
                                  (const hipComplex*)alpha,
                                  (const hipComplex* const*)A,
                                  lda,
                                  (const hipComplex* const*)B,
                                  ldb,
                                  (const hipComplex*)beta,
                                  (hipComplex* const*)C,
                                  ldc,
                                  batch_count);
}

hipblasStatus_t hipblasZgemmBatchedCast_64(hipblasHandle_t                   handle,
                                           hipblasOperation_t                transA,
                                           hipblasOperation_t                transB,
                                           int64_t                           m,
                                           int64_t                           n,
                                           int64_t                           k,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const A[],
                                           int64_t                           lda,
                                           const hipblasDoubleComplex* const B[],
                                           int64_t                           ldb,
                                           const hipblasDoubleComplex*       beta,
                                           hipblasDoubleComplex* const       C[],
                                           int64_t                           ldc,
                                           int64_t                           batch_count)
{
    return hipblasZgemmBatched_64(handle,
                                  transA,
                                  transB,
                                  m,
                                  n,
                                  k,
                                  (const hipDoubleComplex*)alpha,
                                  (const hipDoubleComplex* const*)A,
                                  lda,
                                  (const hipDoubleComplex* const*)B,
                                  ldb,
                                  (const hipDoubleComplex*)beta,
                                  (hipDoubleComplex* const*)C,
                                  ldc,
                                  batch_count);
}

// gemm_strided_batched_64
hipblasStatus_t hipblasCgemmStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  hipblasOperation_t    transA,
                                                  hipblasOperation_t    transB,
                                                  int64_t               m,
                                                  int64_t               n,
                                                  int64_t               k,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* A,
                                                  int64_t               lda,
                                                  int64_t               bsa,
                                                  const hipblasComplex* B,
                                                  int64_t               ldb,
                                                  int64_t               bsb,
                                                  const hipblasComplex* beta,
                                                  hipblasComplex*       C,
                                                  int64_t               ldc,
                                                  int64_t               bsc,
                                                  int64_t               batch_count)
{
    return hipblasCgemmStridedBatched_64(handle,
                                         transA,
                                         transB,
                                         m,
                                         n,
                                         k,
                                         (const hipComplex*)alpha,
                                         (const hipComplex*)A,
                                         lda,
                                         bsa,
                                         (const hipComplex*)B,
                                         ldb,
                                         bsb,
                                         (const hipComplex*)beta,
                                         (hipComplex*)C,
                                         ldc,
                                         bsc,
                                         batch_count);
}

hipblasStatus_t hipblasZgemmStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  hipblasOperation_t          transA,
                                                  hipblasOperation_t          transB,
                                                  int64_t                     m,
                                                  int64_t                     n,
                                                  int64_t                     k,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* A,
                                                  int64_t                     lda,
                                                  int64_t                     bsa,
                                                  const hipblasDoubleComplex* B,
                                                  int64_t                     ldb,
                                                  int64_t                     bsb,
                                                  const hipblasDoubleComplex* beta,
                                                  hipblasDoubleComplex*       C,
                                                  int64_t                     ldc,
                                                  int64_t                     bsc,
                                                  int64_t                     batch_count)
{
    return hipblasZgemmStridedBatched_64(handle,
                                         transA,
                                         transB,
                                         m,
                                         n,
                                         k,
                                         (const hipDoubleComplex*)alpha,
                                         (const hipDoubleComplex*)A,
                                         lda,
                                         bsa,
                                         (const hipDoubleComplex*)B,
                                         ldb,
                                         bsb,
                                         (const hipDoubleComplex*)beta,
                                         (hipDoubleComplex*)C,
                                         ldc,
                                         bsc,
                                         batch_count);
}

// herk
hipblasStatus_t hipblasCherkCast(hipblasHandle_t       handle,
                                 hipblasFillMode_t     uplo,
                                 hipblasOperation_t    transA,
                                 int                   n,
                                 int                   k,
                                 const float*          alpha,
                                 const hipblasComplex* A,
                                 int                   lda,
                                 const float*          beta,
                                 hipblasComplex*       C,
                                 int                   ldc)
{
    return hipblasCherk(
        handle, uplo, transA, n, k, alpha, (const hipComplex*)A, lda, beta, (hipComplex*)C, ldc);
}

hipblasStatus_t hipblasZherkCast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 hipblasOperation_t          transA,
                                 int                         n,
                                 int                         k,
                                 const double*               alpha,
                                 const hipblasDoubleComplex* A,
                                 int                         lda,
                                 const double*               beta,
                                 hipblasDoubleComplex*       C,
                                 int                         ldc)
{
    return hipblasZherk(handle,
                        uplo,
                        transA,
                        n,
                        k,
                        alpha,
                        (const hipDoubleComplex*)A,
                        lda,
                        beta,
                        (hipDoubleComplex*)C,
                        ldc);
}

// herk_batched
hipblasStatus_t hipblasCherkBatchedCast(hipblasHandle_t             handle,
                                        hipblasFillMode_t           uplo,
                                        hipblasOperation_t          transA,
                                        int                         n,
                                        int                         k,
                                        const float*                alpha,
                                        const hipblasComplex* const A[],
                                        int                         lda,
                                        const float*                beta,
                                        hipblasComplex* const       C[],
                                        int                         ldc,
                                        int                         batchCount)
{
    return hipblasCherkBatched(handle,
                               uplo,
                               transA,
                               n,
                               k,
                               alpha,
                               (const hipComplex* const*)A,
                               lda,
                               beta,
                               (hipComplex* const*)C,
                               ldc,
                               batchCount);
}

hipblasStatus_t hipblasZherkBatchedCast(hipblasHandle_t                   handle,
                                        hipblasFillMode_t                 uplo,
                                        hipblasOperation_t                transA,
                                        int                               n,
                                        int                               k,
                                        const double*                     alpha,
                                        const hipblasDoubleComplex* const A[],
                                        int                               lda,
                                        const double*                     beta,
                                        hipblasDoubleComplex* const       C[],
                                        int                               ldc,
                                        int                               batchCount)
{
    return hipblasZherkBatched(handle,
                               uplo,
                               transA,
                               n,
                               k,
                               alpha,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               beta,
                               (hipDoubleComplex* const*)C,
                               ldc,
                               batchCount);
}

// herk_strided_batched
hipblasStatus_t hipblasCherkStridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasFillMode_t     uplo,
                                               hipblasOperation_t    transA,
                                               int                   n,
                                               int                   k,
                                               const float*          alpha,
                                               const hipblasComplex* A,
                                               int                   lda,
                                               hipblasStride         strideA,
                                               const float*          beta,
                                               hipblasComplex*       C,
                                               int                   ldc,
                                               hipblasStride         strideC,
                                               int                   batchCount)
{
    return hipblasCherkStridedBatched(handle,
                                      uplo,
                                      transA,
                                      n,
                                      k,
                                      alpha,
                                      (const hipComplex*)A,
                                      lda,
                                      strideA,
                                      beta,
                                      (hipComplex*)C,
                                      ldc,
                                      strideC,
                                      batchCount);
}

hipblasStatus_t hipblasZherkStridedBatchedCast(hipblasHandle_t             handle,
                                               hipblasFillMode_t           uplo,
                                               hipblasOperation_t          transA,
                                               int                         n,
                                               int                         k,
                                               const double*               alpha,
                                               const hipblasDoubleComplex* A,
                                               int                         lda,
                                               hipblasStride               strideA,
                                               const double*               beta,
                                               hipblasDoubleComplex*       C,
                                               int                         ldc,
                                               hipblasStride               strideC,
                                               int                         batchCount)
{
    return hipblasZherkStridedBatched(handle,
                                      uplo,
                                      transA,
                                      n,
                                      k,
                                      alpha,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      beta,
                                      (hipDoubleComplex*)C,
                                      ldc,
                                      strideC,
                                      batchCount);
}

// herk_64
hipblasStatus_t hipblasCherkCast_64(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    hipblasOperation_t    transA,
                                    int64_t               n,
                                    int64_t               k,
                                    const float*          alpha,
                                    const hipblasComplex* A,
                                    int64_t               lda,
                                    const float*          beta,
                                    hipblasComplex*       C,
                                    int64_t               ldc)
{
    return hipblasCherk_64(
        handle, uplo, transA, n, k, alpha, (const hipComplex*)A, lda, beta, (hipComplex*)C, ldc);
}

hipblasStatus_t hipblasZherkCast_64(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    int64_t                     n,
                                    int64_t                     k,
                                    const double*               alpha,
                                    const hipblasDoubleComplex* A,
                                    int64_t                     lda,
                                    const double*               beta,
                                    hipblasDoubleComplex*       C,
                                    int64_t                     ldc)
{
    return hipblasZherk_64(handle,
                           uplo,
                           transA,
                           n,
                           k,
                           alpha,
                           (const hipDoubleComplex*)A,
                           lda,
                           beta,
                           (hipDoubleComplex*)C,
                           ldc);
}

// herk_batched_64
hipblasStatus_t hipblasCherkBatchedCast_64(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           int64_t                     n,
                                           int64_t                     k,
                                           const float*                alpha,
                                           const hipblasComplex* const A[],
                                           int64_t                     lda,
                                           const float*                beta,
                                           hipblasComplex* const       C[],
                                           int64_t                     ldc,
                                           int64_t                     batchCount)
{
    return hipblasCherkBatched_64(handle,
                                  uplo,
                                  transA,
                                  n,
                                  k,
                                  alpha,
                                  (const hipComplex* const*)A,
                                  lda,
                                  beta,
                                  (hipComplex* const*)C,
                                  ldc,
                                  batchCount);
}

hipblasStatus_t hipblasZherkBatchedCast_64(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           hipblasOperation_t                transA,
                                           int64_t                           n,
                                           int64_t                           k,
                                           const double*                     alpha,
                                           const hipblasDoubleComplex* const A[],
                                           int64_t                           lda,
                                           const double*                     beta,
                                           hipblasDoubleComplex* const       C[],
                                           int64_t                           ldc,
                                           int64_t                           batchCount)
{
    return hipblasZherkBatched_64(handle,
                                  uplo,
                                  transA,
                                  n,
                                  k,
                                  alpha,
                                  (const hipDoubleComplex* const*)A,
                                  lda,
                                  beta,
                                  (hipDoubleComplex* const*)C,
                                  ldc,
                                  batchCount);
}

// herk_strided_batched_64
hipblasStatus_t hipblasCherkStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  hipblasOperation_t    transA,
                                                  int64_t               n,
                                                  int64_t               k,
                                                  const float*          alpha,
                                                  const hipblasComplex* A,
                                                  int64_t               lda,
                                                  hipblasStride         strideA,
                                                  const float*          beta,
                                                  hipblasComplex*       C,
                                                  int64_t               ldc,
                                                  hipblasStride         strideC,
                                                  int64_t               batchCount)
{
    return hipblasCherkStridedBatched_64(handle,
                                         uplo,
                                         transA,
                                         n,
                                         k,
                                         alpha,
                                         (const hipComplex*)A,
                                         lda,
                                         strideA,
                                         beta,
                                         (hipComplex*)C,
                                         ldc,
                                         strideC,
                                         batchCount);
}

hipblasStatus_t hipblasZherkStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  hipblasOperation_t          transA,
                                                  int64_t                     n,
                                                  int64_t                     k,
                                                  const double*               alpha,
                                                  const hipblasDoubleComplex* A,
                                                  int64_t                     lda,
                                                  hipblasStride               strideA,
                                                  const double*               beta,
                                                  hipblasDoubleComplex*       C,
                                                  int64_t                     ldc,
                                                  hipblasStride               strideC,
                                                  int64_t                     batchCount)
{
    return hipblasZherkStridedBatched_64(handle,
                                         uplo,
                                         transA,
                                         n,
                                         k,
                                         alpha,
                                         (const hipDoubleComplex*)A,
                                         lda,
                                         strideA,
                                         beta,
                                         (hipDoubleComplex*)C,
                                         ldc,
                                         strideC,
                                         batchCount);
}

// her2k
hipblasStatus_t hipblasCher2kCast(hipblasHandle_t       handle,
                                  hipblasFillMode_t     uplo,
                                  hipblasOperation_t    transA,
                                  int                   n,
                                  int                   k,
                                  const hipblasComplex* alpha,
                                  const hipblasComplex* A,
                                  int                   lda,
                                  const hipblasComplex* B,
                                  int                   ldb,
                                  const float*          beta,
                                  hipblasComplex*       C,
                                  int                   ldc)
{
    return hipblasCher2k(handle,
                         uplo,
                         transA,
                         n,
                         k,
                         (const hipComplex*)alpha,
                         (const hipComplex*)A,
                         lda,
                         (const hipComplex*)B,
                         ldb,
                         beta,
                         (hipComplex*)C,
                         ldc);
}

hipblasStatus_t hipblasZher2kCast(hipblasHandle_t             handle,
                                  hipblasFillMode_t           uplo,
                                  hipblasOperation_t          transA,
                                  int                         n,
                                  int                         k,
                                  const hipblasDoubleComplex* alpha,
                                  const hipblasDoubleComplex* A,
                                  int                         lda,
                                  const hipblasDoubleComplex* B,
                                  int                         ldb,
                                  const double*               beta,
                                  hipblasDoubleComplex*       C,
                                  int                         ldc)
{
    return hipblasZher2k(handle,
                         uplo,
                         transA,
                         n,
                         k,
                         (const hipDoubleComplex*)alpha,
                         (const hipDoubleComplex*)A,
                         lda,
                         (const hipDoubleComplex*)B,
                         ldb,
                         beta,
                         (hipDoubleComplex*)C,
                         ldc);
}

// her2k_batched
hipblasStatus_t hipblasCher2kBatchedCast(hipblasHandle_t             handle,
                                         hipblasFillMode_t           uplo,
                                         hipblasOperation_t          transA,
                                         int                         n,
                                         int                         k,
                                         const hipblasComplex*       alpha,
                                         const hipblasComplex* const A[],
                                         int                         lda,
                                         const hipblasComplex* const B[],
                                         int                         ldb,
                                         const float*                beta,
                                         hipblasComplex* const       C[],
                                         int                         ldc,
                                         int                         batchCount)
{
    return hipblasCher2kBatched(handle,
                                uplo,
                                transA,
                                n,
                                k,
                                (const hipComplex*)alpha,
                                (const hipComplex* const*)A,
                                lda,
                                (const hipComplex* const*)B,
                                ldb,
                                beta,
                                (hipComplex* const*)C,
                                ldc,
                                batchCount);
}

hipblasStatus_t hipblasZher2kBatchedCast(hipblasHandle_t                   handle,
                                         hipblasFillMode_t                 uplo,
                                         hipblasOperation_t                transA,
                                         int                               n,
                                         int                               k,
                                         const hipblasDoubleComplex*       alpha,
                                         const hipblasDoubleComplex* const A[],
                                         int                               lda,
                                         const hipblasDoubleComplex* const B[],
                                         int                               ldb,
                                         const double*                     beta,
                                         hipblasDoubleComplex* const       C[],
                                         int                               ldc,
                                         int                               batchCount)
{
    return hipblasZher2kBatched(handle,
                                uplo,
                                transA,
                                n,
                                k,
                                (const hipDoubleComplex*)alpha,
                                (const hipDoubleComplex* const*)A,
                                lda,
                                (const hipDoubleComplex* const*)B,
                                ldb,
                                beta,
                                (hipDoubleComplex* const*)C,
                                ldc,
                                batchCount);
}

// her2k_strided_batched
hipblasStatus_t hipblasCher2kStridedBatchedCast(hipblasHandle_t       handle,
                                                hipblasFillMode_t     uplo,
                                                hipblasOperation_t    transA,
                                                int                   n,
                                                int                   k,
                                                const hipblasComplex* alpha,
                                                const hipblasComplex* A,
                                                int                   lda,
                                                hipblasStride         strideA,
                                                const hipblasComplex* B,
                                                int                   ldb,
                                                hipblasStride         strideB,
                                                const float*          beta,
                                                hipblasComplex*       C,
                                                int                   ldc,
                                                hipblasStride         strideC,
                                                int                   batchCount)
{
    return hipblasCher2kStridedBatched(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       (const hipComplex*)alpha,
                                       (const hipComplex*)A,
                                       lda,
                                       strideA,
                                       (const hipComplex*)B,
                                       ldb,
                                       strideB,
                                       beta,
                                       (hipComplex*)C,
                                       ldc,
                                       strideC,
                                       batchCount);
}

hipblasStatus_t hipblasZher2kStridedBatchedCast(hipblasHandle_t             handle,
                                                hipblasFillMode_t           uplo,
                                                hipblasOperation_t          transA,
                                                int                         n,
                                                int                         k,
                                                const hipblasDoubleComplex* alpha,
                                                const hipblasDoubleComplex* A,
                                                int                         lda,
                                                hipblasStride               strideA,
                                                const hipblasDoubleComplex* B,
                                                int                         ldb,
                                                hipblasStride               strideB,
                                                const double*               beta,
                                                hipblasDoubleComplex*       C,
                                                int                         ldc,
                                                hipblasStride               strideC,
                                                int                         batchCount)
{
    return hipblasZher2kStridedBatched(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       (const hipDoubleComplex*)alpha,
                                       (const hipDoubleComplex*)A,
                                       lda,
                                       strideA,
                                       (const hipDoubleComplex*)B,
                                       ldb,
                                       strideB,
                                       beta,
                                       (hipDoubleComplex*)C,
                                       ldc,
                                       strideC,
                                       batchCount);
}

// her2k_64
hipblasStatus_t hipblasCher2kCast_64(hipblasHandle_t       handle,
                                     hipblasFillMode_t     uplo,
                                     hipblasOperation_t    transA,
                                     int64_t               n,
                                     int64_t               k,
                                     const hipblasComplex* alpha,
                                     const hipblasComplex* A,
                                     int64_t               lda,
                                     const hipblasComplex* B,
                                     int64_t               ldb,
                                     const float*          beta,
                                     hipblasComplex*       C,
                                     int64_t               ldc)
{
    return hipblasCher2k_64(handle,
                            uplo,
                            transA,
                            n,
                            k,
                            (const hipComplex*)alpha,
                            (const hipComplex*)A,
                            lda,
                            (const hipComplex*)B,
                            ldb,
                            beta,
                            (hipComplex*)C,
                            ldc);
}

hipblasStatus_t hipblasZher2kCast_64(hipblasHandle_t             handle,
                                     hipblasFillMode_t           uplo,
                                     hipblasOperation_t          transA,
                                     int64_t                     n,
                                     int64_t                     k,
                                     const hipblasDoubleComplex* alpha,
                                     const hipblasDoubleComplex* A,
                                     int64_t                     lda,
                                     const hipblasDoubleComplex* B,
                                     int64_t                     ldb,
                                     const double*               beta,
                                     hipblasDoubleComplex*       C,
                                     int64_t                     ldc)
{
    return hipblasZher2k_64(handle,
                            uplo,
                            transA,
                            n,
                            k,
                            (const hipDoubleComplex*)alpha,
                            (const hipDoubleComplex*)A,
                            lda,
                            (const hipDoubleComplex*)B,
                            ldb,
                            beta,
                            (hipDoubleComplex*)C,
                            ldc);
}

// her2k_batched_64
hipblasStatus_t hipblasCher2kBatchedCast_64(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            hipblasOperation_t          transA,
                                            int64_t                     n,
                                            int64_t                     k,
                                            const hipblasComplex*       alpha,
                                            const hipblasComplex* const A[],
                                            int64_t                     lda,
                                            const hipblasComplex* const B[],
                                            int64_t                     ldb,
                                            const float*                beta,
                                            hipblasComplex* const       C[],
                                            int64_t                     ldc,
                                            int64_t                     batchCount)
{
    return hipblasCher2kBatched_64(handle,
                                   uplo,
                                   transA,
                                   n,
                                   k,
                                   (const hipComplex*)alpha,
                                   (const hipComplex* const*)A,
                                   lda,
                                   (const hipComplex* const*)B,
                                   ldb,
                                   beta,
                                   (hipComplex* const*)C,
                                   ldc,
                                   batchCount);
}

hipblasStatus_t hipblasZher2kBatchedCast_64(hipblasHandle_t                   handle,
                                            hipblasFillMode_t                 uplo,
                                            hipblasOperation_t                transA,
                                            int64_t                           n,
                                            int64_t                           k,
                                            const hipblasDoubleComplex*       alpha,
                                            const hipblasDoubleComplex* const A[],
                                            int64_t                           lda,
                                            const hipblasDoubleComplex* const B[],
                                            int64_t                           ldb,
                                            const double*                     beta,
                                            hipblasDoubleComplex* const       C[],
                                            int64_t                           ldc,
                                            int64_t                           batchCount)
{
    return hipblasZher2kBatched_64(handle,
                                   uplo,
                                   transA,
                                   n,
                                   k,
                                   (const hipDoubleComplex*)alpha,
                                   (const hipDoubleComplex* const*)A,
                                   lda,
                                   (const hipDoubleComplex* const*)B,
                                   ldb,
                                   beta,
                                   (hipDoubleComplex* const*)C,
                                   ldc,
                                   batchCount);
}

// her2k_strided_batched_64
hipblasStatus_t hipblasCher2kStridedBatchedCast_64(hipblasHandle_t       handle,
                                                   hipblasFillMode_t     uplo,
                                                   hipblasOperation_t    transA,
                                                   int64_t               n,
                                                   int64_t               k,
                                                   const hipblasComplex* alpha,
                                                   const hipblasComplex* A,
                                                   int64_t               lda,
                                                   hipblasStride         strideA,
                                                   const hipblasComplex* B,
                                                   int64_t               ldb,
                                                   hipblasStride         strideB,
                                                   const float*          beta,
                                                   hipblasComplex*       C,
                                                   int64_t               ldc,
                                                   hipblasStride         strideC,
                                                   int64_t               batchCount)
{
    return hipblasCher2kStridedBatched_64(handle,
                                          uplo,
                                          transA,
                                          n,
                                          k,
                                          (const hipComplex*)alpha,
                                          (const hipComplex*)A,
                                          lda,
                                          strideA,
                                          (const hipComplex*)B,
                                          ldb,
                                          strideB,
                                          beta,
                                          (hipComplex*)C,
                                          ldc,
                                          strideC,
                                          batchCount);
}

hipblasStatus_t hipblasZher2kStridedBatchedCast_64(hipblasHandle_t             handle,
                                                   hipblasFillMode_t           uplo,
                                                   hipblasOperation_t          transA,
                                                   int64_t                     n,
                                                   int64_t                     k,
                                                   const hipblasDoubleComplex* alpha,
                                                   const hipblasDoubleComplex* A,
                                                   int64_t                     lda,
                                                   hipblasStride               strideA,
                                                   const hipblasDoubleComplex* B,
                                                   int64_t                     ldb,
                                                   hipblasStride               strideB,
                                                   const double*               beta,
                                                   hipblasDoubleComplex*       C,
                                                   int64_t                     ldc,
                                                   hipblasStride               strideC,
                                                   int64_t                     batchCount)
{
    return hipblasZher2kStridedBatched_64(handle,
                                          uplo,
                                          transA,
                                          n,
                                          k,
                                          (const hipDoubleComplex*)alpha,
                                          (const hipDoubleComplex*)A,
                                          lda,
                                          strideA,
                                          (const hipDoubleComplex*)B,
                                          ldb,
                                          strideB,
                                          beta,
                                          (hipDoubleComplex*)C,
                                          ldc,
                                          strideC,
                                          batchCount);
}

// herkx
hipblasStatus_t hipblasCherkxCast(hipblasHandle_t       handle,
                                  hipblasFillMode_t     uplo,
                                  hipblasOperation_t    transA,
                                  int                   n,
                                  int                   k,
                                  const hipblasComplex* alpha,
                                  const hipblasComplex* A,
                                  int                   lda,
                                  const hipblasComplex* B,
                                  int                   ldb,
                                  const float*          beta,
                                  hipblasComplex*       C,
                                  int                   ldc)
{
    return hipblasCherkx(handle,
                         uplo,
                         transA,
                         n,
                         k,
                         (const hipComplex*)alpha,
                         (const hipComplex*)A,
                         lda,
                         (const hipComplex*)B,
                         ldb,
                         beta,
                         (hipComplex*)C,
                         ldc);
}

hipblasStatus_t hipblasZherkxCast(hipblasHandle_t             handle,
                                  hipblasFillMode_t           uplo,
                                  hipblasOperation_t          transA,
                                  int                         n,
                                  int                         k,
                                  const hipblasDoubleComplex* alpha,
                                  const hipblasDoubleComplex* A,
                                  int                         lda,
                                  const hipblasDoubleComplex* B,
                                  int                         ldb,
                                  const double*               beta,
                                  hipblasDoubleComplex*       C,
                                  int                         ldc)
{
    return hipblasZherkx(handle,
                         uplo,
                         transA,
                         n,
                         k,
                         (const hipDoubleComplex*)alpha,
                         (const hipDoubleComplex*)A,
                         lda,
                         (const hipDoubleComplex*)B,
                         ldb,
                         beta,
                         (hipDoubleComplex*)C,
                         ldc);
}

// herkx_batched
hipblasStatus_t hipblasCherkxBatchedCast(hipblasHandle_t             handle,
                                         hipblasFillMode_t           uplo,
                                         hipblasOperation_t          transA,
                                         int                         n,
                                         int                         k,
                                         const hipblasComplex*       alpha,
                                         const hipblasComplex* const A[],
                                         int                         lda,
                                         const hipblasComplex* const B[],
                                         int                         ldb,
                                         const float*                beta,
                                         hipblasComplex* const       C[],
                                         int                         ldc,
                                         int                         batchCount)
{
    return hipblasCherkxBatched(handle,
                                uplo,
                                transA,
                                n,
                                k,
                                (const hipComplex*)alpha,
                                (const hipComplex* const*)A,
                                lda,
                                (const hipComplex* const*)B,
                                ldb,
                                beta,
                                (hipComplex* const*)C,
                                ldc,
                                batchCount);
}

hipblasStatus_t hipblasZherkxBatchedCast(hipblasHandle_t                   handle,
                                         hipblasFillMode_t                 uplo,
                                         hipblasOperation_t                transA,
                                         int                               n,
                                         int                               k,
                                         const hipblasDoubleComplex*       alpha,
                                         const hipblasDoubleComplex* const A[],
                                         int                               lda,
                                         const hipblasDoubleComplex* const B[],
                                         int                               ldb,
                                         const double*                     beta,
                                         hipblasDoubleComplex* const       C[],
                                         int                               ldc,
                                         int                               batchCount)
{
    return hipblasZherkxBatched(handle,
                                uplo,
                                transA,
                                n,
                                k,
                                (const hipDoubleComplex*)alpha,
                                (const hipDoubleComplex* const*)A,
                                lda,
                                (const hipDoubleComplex* const*)B,
                                ldb,
                                beta,
                                (hipDoubleComplex* const*)C,
                                ldc,
                                batchCount);
}

// herkx_strided_batched
hipblasStatus_t hipblasCherkxStridedBatchedCast(hipblasHandle_t       handle,
                                                hipblasFillMode_t     uplo,
                                                hipblasOperation_t    transA,
                                                int                   n,
                                                int                   k,
                                                const hipblasComplex* alpha,
                                                const hipblasComplex* A,
                                                int                   lda,
                                                hipblasStride         strideA,
                                                const hipblasComplex* B,
                                                int                   ldb,
                                                hipblasStride         strideB,
                                                const float*          beta,
                                                hipblasComplex*       C,
                                                int                   ldc,
                                                hipblasStride         strideC,
                                                int                   batchCount)
{
    return hipblasCherkxStridedBatched(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       (const hipComplex*)alpha,
                                       (const hipComplex*)A,
                                       lda,
                                       strideA,
                                       (const hipComplex*)B,
                                       ldb,
                                       strideB,
                                       beta,
                                       (hipComplex*)C,
                                       ldc,
                                       strideC,
                                       batchCount);
}

hipblasStatus_t hipblasZherkxStridedBatchedCast(hipblasHandle_t             handle,
                                                hipblasFillMode_t           uplo,
                                                hipblasOperation_t          transA,
                                                int                         n,
                                                int                         k,
                                                const hipblasDoubleComplex* alpha,
                                                const hipblasDoubleComplex* A,
                                                int                         lda,
                                                hipblasStride               strideA,
                                                const hipblasDoubleComplex* B,
                                                int                         ldb,
                                                hipblasStride               strideB,
                                                const double*               beta,
                                                hipblasDoubleComplex*       C,
                                                int                         ldc,
                                                hipblasStride               strideC,
                                                int                         batchCount)
{
    return hipblasZherkxStridedBatched(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       (const hipDoubleComplex*)alpha,
                                       (const hipDoubleComplex*)A,
                                       lda,
                                       strideA,
                                       (const hipDoubleComplex*)B,
                                       ldb,
                                       strideB,
                                       beta,
                                       (hipDoubleComplex*)C,
                                       ldc,
                                       strideC,
                                       batchCount);
}

// herkx_64
hipblasStatus_t hipblasCherkxCast_64(hipblasHandle_t       handle,
                                     hipblasFillMode_t     uplo,
                                     hipblasOperation_t    transA,
                                     int64_t               n,
                                     int64_t               k,
                                     const hipblasComplex* alpha,
                                     const hipblasComplex* A,
                                     int64_t               lda,
                                     const hipblasComplex* B,
                                     int64_t               ldb,
                                     const float*          beta,
                                     hipblasComplex*       C,
                                     int64_t               ldc)
{
    return hipblasCherkx_64(handle,
                            uplo,
                            transA,
                            n,
                            k,
                            (const hipComplex*)alpha,
                            (const hipComplex*)A,
                            lda,
                            (const hipComplex*)B,
                            ldb,
                            beta,
                            (hipComplex*)C,
                            ldc);
}

hipblasStatus_t hipblasZherkxCast_64(hipblasHandle_t             handle,
                                     hipblasFillMode_t           uplo,
                                     hipblasOperation_t          transA,
                                     int64_t                     n,
                                     int64_t                     k,
                                     const hipblasDoubleComplex* alpha,
                                     const hipblasDoubleComplex* A,
                                     int64_t                     lda,
                                     const hipblasDoubleComplex* B,
                                     int64_t                     ldb,
                                     const double*               beta,
                                     hipblasDoubleComplex*       C,
                                     int64_t                     ldc)
{
    return hipblasZherkx_64(handle,
                            uplo,
                            transA,
                            n,
                            k,
                            (const hipDoubleComplex*)alpha,
                            (const hipDoubleComplex*)A,
                            lda,
                            (const hipDoubleComplex*)B,
                            ldb,
                            beta,
                            (hipDoubleComplex*)C,
                            ldc);
}

// herkx_batched_64
hipblasStatus_t hipblasCherkxBatchedCast_64(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            hipblasOperation_t          transA,
                                            int64_t                     n,
                                            int64_t                     k,
                                            const hipblasComplex*       alpha,
                                            const hipblasComplex* const A[],
                                            int64_t                     lda,
                                            const hipblasComplex* const B[],
                                            int64_t                     ldb,
                                            const float*                beta,
                                            hipblasComplex* const       C[],
                                            int64_t                     ldc,
                                            int64_t                     batchCount)
{
    return hipblasCherkxBatched_64(handle,
                                   uplo,
                                   transA,
                                   n,
                                   k,
                                   (const hipComplex*)alpha,
                                   (const hipComplex* const*)A,
                                   lda,
                                   (const hipComplex* const*)B,
                                   ldb,
                                   beta,
                                   (hipComplex* const*)C,
                                   ldc,
                                   batchCount);
}

hipblasStatus_t hipblasZherkxBatchedCast_64(hipblasHandle_t                   handle,
                                            hipblasFillMode_t                 uplo,
                                            hipblasOperation_t                transA,
                                            int64_t                           n,
                                            int64_t                           k,
                                            const hipblasDoubleComplex*       alpha,
                                            const hipblasDoubleComplex* const A[],
                                            int64_t                           lda,
                                            const hipblasDoubleComplex* const B[],
                                            int64_t                           ldb,
                                            const double*                     beta,
                                            hipblasDoubleComplex* const       C[],
                                            int64_t                           ldc,
                                            int64_t                           batchCount)
{
    return hipblasZherkxBatched_64(handle,
                                   uplo,
                                   transA,
                                   n,
                                   k,
                                   (const hipDoubleComplex*)alpha,
                                   (const hipDoubleComplex* const*)A,
                                   lda,
                                   (const hipDoubleComplex* const*)B,
                                   ldb,
                                   beta,
                                   (hipDoubleComplex* const*)C,
                                   ldc,
                                   batchCount);
}

// herkx_strided_batched_64
hipblasStatus_t hipblasCherkxStridedBatchedCast_64(hipblasHandle_t       handle,
                                                   hipblasFillMode_t     uplo,
                                                   hipblasOperation_t    transA,
                                                   int64_t               n,
                                                   int64_t               k,
                                                   const hipblasComplex* alpha,
                                                   const hipblasComplex* A,
                                                   int64_t               lda,
                                                   hipblasStride         strideA,
                                                   const hipblasComplex* B,
                                                   int64_t               ldb,
                                                   hipblasStride         strideB,
                                                   const float*          beta,
                                                   hipblasComplex*       C,
                                                   int64_t               ldc,
                                                   hipblasStride         strideC,
                                                   int64_t               batchCount)
{
    return hipblasCherkxStridedBatched_64(handle,
                                          uplo,
                                          transA,
                                          n,
                                          k,
                                          (const hipComplex*)alpha,
                                          (const hipComplex*)A,
                                          lda,
                                          strideA,
                                          (const hipComplex*)B,
                                          ldb,
                                          strideB,
                                          beta,
                                          (hipComplex*)C,
                                          ldc,
                                          strideC,
                                          batchCount);
}

hipblasStatus_t hipblasZherkxStridedBatchedCast_64(hipblasHandle_t             handle,
                                                   hipblasFillMode_t           uplo,
                                                   hipblasOperation_t          transA,
                                                   int64_t                     n,
                                                   int64_t                     k,
                                                   const hipblasDoubleComplex* alpha,
                                                   const hipblasDoubleComplex* A,
                                                   int64_t                     lda,
                                                   hipblasStride               strideA,
                                                   const hipblasDoubleComplex* B,
                                                   int64_t                     ldb,
                                                   hipblasStride               strideB,
                                                   const double*               beta,
                                                   hipblasDoubleComplex*       C,
                                                   int64_t                     ldc,
                                                   hipblasStride               strideC,
                                                   int64_t                     batchCount)
{
    return hipblasZherkxStridedBatched_64(handle,
                                          uplo,
                                          transA,
                                          n,
                                          k,
                                          (const hipDoubleComplex*)alpha,
                                          (const hipDoubleComplex*)A,
                                          lda,
                                          strideA,
                                          (const hipDoubleComplex*)B,
                                          ldb,
                                          strideB,
                                          beta,
                                          (hipDoubleComplex*)C,
                                          ldc,
                                          strideC,
                                          batchCount);
}

// symm
hipblasStatus_t hipblasCsymmCast(hipblasHandle_t       handle,
                                 hipblasSideMode_t     side,
                                 hipblasFillMode_t     uplo,
                                 int                   m,
                                 int                   n,
                                 const hipblasComplex* alpha,
                                 const hipblasComplex* A,
                                 int                   lda,
                                 const hipblasComplex* B,
                                 int                   ldb,
                                 const hipblasComplex* beta,
                                 hipblasComplex*       C,
                                 int                   ldc)
{
    return hipblasCsymm(handle,
                        side,
                        uplo,
                        m,
                        n,
                        (const hipComplex*)alpha,
                        (const hipComplex*)A,
                        lda,
                        (const hipComplex*)B,
                        ldb,
                        (const hipComplex*)beta,
                        (hipComplex*)C,
                        ldc);
}

hipblasStatus_t hipblasZsymmCast(hipblasHandle_t             handle,
                                 hipblasSideMode_t           side,
                                 hipblasFillMode_t           uplo,
                                 int                         m,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* A,
                                 int                         lda,
                                 const hipblasDoubleComplex* B,
                                 int                         ldb,
                                 const hipblasDoubleComplex* beta,
                                 hipblasDoubleComplex*       C,
                                 int                         ldc)
{
    return hipblasZsymm(handle,
                        side,
                        uplo,
                        m,
                        n,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)A,
                        lda,
                        (const hipDoubleComplex*)B,
                        ldb,
                        (const hipDoubleComplex*)beta,
                        (hipDoubleComplex*)C,
                        ldc);
}

// symm_batched
hipblasStatus_t hipblasCsymmBatchedCast(hipblasHandle_t             handle,
                                        hipblasSideMode_t           side,
                                        hipblasFillMode_t           uplo,
                                        int                         m,
                                        int                         n,
                                        const hipblasComplex*       alpha,
                                        const hipblasComplex* const A[],
                                        int                         lda,
                                        const hipblasComplex* const B[],
                                        int                         ldb,
                                        const hipblasComplex*       beta,
                                        hipblasComplex* const       C[],
                                        int                         ldc,
                                        int                         batchCount)
{
    return hipblasCsymmBatched(handle,
                               side,
                               uplo,
                               m,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)A,
                               lda,
                               (const hipComplex* const*)B,
                               ldb,
                               (const hipComplex*)beta,
                               (hipComplex* const*)C,
                               ldc,
                               batchCount);
}

hipblasStatus_t hipblasZsymmBatchedCast(hipblasHandle_t                   handle,
                                        hipblasSideMode_t                 side,
                                        hipblasFillMode_t                 uplo,
                                        int                               m,
                                        int                               n,
                                        const hipblasDoubleComplex*       alpha,
                                        const hipblasDoubleComplex* const A[],
                                        int                               lda,
                                        const hipblasDoubleComplex* const B[],
                                        int                               ldb,
                                        const hipblasDoubleComplex*       beta,
                                        hipblasDoubleComplex* const       C[],
                                        int                               ldc,
                                        int                               batchCount)
{
    return hipblasZsymmBatched(handle,
                               side,
                               uplo,
                               m,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (const hipDoubleComplex* const*)B,
                               ldb,
                               (const hipDoubleComplex*)beta,
                               (hipDoubleComplex* const*)C,
                               ldc,
                               batchCount);
}

// symm_strided_batched
hipblasStatus_t hipblasCsymmStridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasSideMode_t     side,
                                               hipblasFillMode_t     uplo,
                                               int                   m,
                                               int                   n,
                                               const hipblasComplex* alpha,
                                               const hipblasComplex* A,
                                               int                   lda,
                                               hipblasStride         strideA,
                                               const hipblasComplex* B,
                                               int                   ldb,
                                               hipblasStride         strideB,
                                               const hipblasComplex* beta,
                                               hipblasComplex*       C,
                                               int                   ldc,
                                               hipblasStride         strideC,
                                               int                   batchCount)
{
    return hipblasCsymmStridedBatched(handle,
                                      side,
                                      uplo,
                                      m,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipComplex*)B,
                                      ldb,
                                      strideB,
                                      (const hipComplex*)beta,
                                      (hipComplex*)C,
                                      ldc,
                                      strideC,
                                      batchCount);
}

hipblasStatus_t hipblasZsymmStridedBatchedCast(hipblasHandle_t             handle,
                                               hipblasSideMode_t           side,
                                               hipblasFillMode_t           uplo,
                                               int                         m,
                                               int                         n,
                                               const hipblasDoubleComplex* alpha,
                                               const hipblasDoubleComplex* A,
                                               int                         lda,
                                               hipblasStride               strideA,
                                               const hipblasDoubleComplex* B,
                                               int                         ldb,
                                               hipblasStride               strideB,
                                               const hipblasDoubleComplex* beta,
                                               hipblasDoubleComplex*       C,
                                               int                         ldc,
                                               hipblasStride               strideC,
                                               int                         batchCount)
{
    return hipblasZsymmStridedBatched(handle,
                                      side,
                                      uplo,
                                      m,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipDoubleComplex*)B,
                                      ldb,
                                      strideB,
                                      (const hipDoubleComplex*)beta,
                                      (hipDoubleComplex*)C,
                                      ldc,
                                      strideC,
                                      batchCount);
}

// symm_64
hipblasStatus_t hipblasCsymmCast_64(hipblasHandle_t       handle,
                                    hipblasSideMode_t     side,
                                    hipblasFillMode_t     uplo,
                                    int64_t               m,
                                    int64_t               n,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* A,
                                    int64_t               lda,
                                    const hipblasComplex* B,
                                    int64_t               ldb,
                                    const hipblasComplex* beta,
                                    hipblasComplex*       C,
                                    int64_t               ldc)
{
    return hipblasCsymm_64(handle,
                           side,
                           uplo,
                           m,
                           n,
                           (const hipComplex*)alpha,
                           (const hipComplex*)A,
                           lda,
                           (const hipComplex*)B,
                           ldb,
                           (const hipComplex*)beta,
                           (hipComplex*)C,
                           ldc);
}

hipblasStatus_t hipblasZsymmCast_64(hipblasHandle_t             handle,
                                    hipblasSideMode_t           side,
                                    hipblasFillMode_t           uplo,
                                    int64_t                     m,
                                    int64_t                     n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* A,
                                    int64_t                     lda,
                                    const hipblasDoubleComplex* B,
                                    int64_t                     ldb,
                                    const hipblasDoubleComplex* beta,
                                    hipblasDoubleComplex*       C,
                                    int64_t                     ldc)
{
    return hipblasZsymm_64(handle,
                           side,
                           uplo,
                           m,
                           n,
                           (const hipDoubleComplex*)alpha,
                           (const hipDoubleComplex*)A,
                           lda,
                           (const hipDoubleComplex*)B,
                           ldb,
                           (const hipDoubleComplex*)beta,
                           (hipDoubleComplex*)C,
                           ldc);
}

// symm_batched_64
hipblasStatus_t hipblasCsymmBatchedCast_64(hipblasHandle_t             handle,
                                           hipblasSideMode_t           side,
                                           hipblasFillMode_t           uplo,
                                           int64_t                     m,
                                           int64_t                     n,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const A[],
                                           int64_t                     lda,
                                           const hipblasComplex* const B[],
                                           int64_t                     ldb,
                                           const hipblasComplex*       beta,
                                           hipblasComplex* const       C[],
                                           int64_t                     ldc,
                                           int64_t                     batchCount)
{
    return hipblasCsymmBatched_64(handle,
                                  side,
                                  uplo,
                                  m,
                                  n,
                                  (const hipComplex*)alpha,
                                  (const hipComplex* const*)A,
                                  lda,
                                  (const hipComplex* const*)B,
                                  ldb,
                                  (const hipComplex*)beta,
                                  (hipComplex* const*)C,
                                  ldc,
                                  batchCount);
}

hipblasStatus_t hipblasZsymmBatchedCast_64(hipblasHandle_t                   handle,
                                           hipblasSideMode_t                 side,
                                           hipblasFillMode_t                 uplo,
                                           int64_t                           m,
                                           int64_t                           n,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const A[],
                                           int64_t                           lda,
                                           const hipblasDoubleComplex* const B[],
                                           int64_t                           ldb,
                                           const hipblasDoubleComplex*       beta,
                                           hipblasDoubleComplex* const       C[],
                                           int64_t                           ldc,
                                           int64_t                           batchCount)
{
    return hipblasZsymmBatched_64(handle,
                                  side,
                                  uplo,
                                  m,
                                  n,
                                  (const hipDoubleComplex*)alpha,
                                  (const hipDoubleComplex* const*)A,
                                  lda,
                                  (const hipDoubleComplex* const*)B,
                                  ldb,
                                  (const hipDoubleComplex*)beta,
                                  (hipDoubleComplex* const*)C,
                                  ldc,
                                  batchCount);
}

// symm_strided_batched_64
hipblasStatus_t hipblasCsymmStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  hipblasSideMode_t     side,
                                                  hipblasFillMode_t     uplo,
                                                  int64_t               m,
                                                  int64_t               n,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* A,
                                                  int64_t               lda,
                                                  hipblasStride         strideA,
                                                  const hipblasComplex* B,
                                                  int64_t               ldb,
                                                  hipblasStride         strideB,
                                                  const hipblasComplex* beta,
                                                  hipblasComplex*       C,
                                                  int64_t               ldc,
                                                  hipblasStride         strideC,
                                                  int64_t               batchCount)
{
    return hipblasCsymmStridedBatched_64(handle,
                                         side,
                                         uplo,
                                         m,
                                         n,
                                         (const hipComplex*)alpha,
                                         (const hipComplex*)A,
                                         lda,
                                         strideA,
                                         (const hipComplex*)B,
                                         ldb,
                                         strideB,
                                         (const hipComplex*)beta,
                                         (hipComplex*)C,
                                         ldc,
                                         strideC,
                                         batchCount);
}

hipblasStatus_t hipblasZsymmStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  hipblasSideMode_t           side,
                                                  hipblasFillMode_t           uplo,
                                                  int64_t                     m,
                                                  int64_t                     n,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* A,
                                                  int64_t                     lda,
                                                  hipblasStride               strideA,
                                                  const hipblasDoubleComplex* B,
                                                  int64_t                     ldb,
                                                  hipblasStride               strideB,
                                                  const hipblasDoubleComplex* beta,
                                                  hipblasDoubleComplex*       C,
                                                  int64_t                     ldc,
                                                  hipblasStride               strideC,
                                                  int64_t                     batchCount)
{
    return hipblasZsymmStridedBatched_64(handle,
                                         side,
                                         uplo,
                                         m,
                                         n,
                                         (const hipDoubleComplex*)alpha,
                                         (const hipDoubleComplex*)A,
                                         lda,
                                         strideA,
                                         (const hipDoubleComplex*)B,
                                         ldb,
                                         strideB,
                                         (const hipDoubleComplex*)beta,
                                         (hipDoubleComplex*)C,
                                         ldc,
                                         strideC,
                                         batchCount);
}

// syrk
hipblasStatus_t hipblasCsyrkCast(hipblasHandle_t       handle,
                                 hipblasFillMode_t     uplo,
                                 hipblasOperation_t    transA,
                                 int                   n,
                                 int                   k,
                                 const hipblasComplex* alpha,
                                 const hipblasComplex* A,
                                 int                   lda,
                                 const hipblasComplex* beta,
                                 hipblasComplex*       C,
                                 int                   ldc)
{
    return hipblasCsyrk(handle,
                        uplo,
                        transA,
                        n,
                        k,
                        (const hipComplex*)alpha,
                        (const hipComplex*)A,
                        lda,
                        (const hipComplex*)beta,
                        (hipComplex*)C,
                        ldc);
}

hipblasStatus_t hipblasZsyrkCast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 hipblasOperation_t          transA,
                                 int                         n,
                                 int                         k,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* A,
                                 int                         lda,
                                 const hipblasDoubleComplex* beta,
                                 hipblasDoubleComplex*       C,
                                 int                         ldc)
{
    return hipblasZsyrk(handle,
                        uplo,
                        transA,
                        n,
                        k,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)A,
                        lda,
                        (const hipDoubleComplex*)beta,
                        (hipDoubleComplex*)C,
                        ldc);
}

// syrk_batched
hipblasStatus_t hipblasCsyrkBatchedCast(hipblasHandle_t             handle,
                                        hipblasFillMode_t           uplo,
                                        hipblasOperation_t          transA,
                                        int                         n,
                                        int                         k,
                                        const hipblasComplex*       alpha,
                                        const hipblasComplex* const A[],
                                        int                         lda,
                                        const hipblasComplex*       beta,
                                        hipblasComplex* const       C[],
                                        int                         ldc,
                                        int                         batchCount)
{
    return hipblasCsyrkBatched(handle,
                               uplo,
                               transA,
                               n,
                               k,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)A,
                               lda,
                               (const hipComplex*)beta,
                               (hipComplex* const*)C,
                               ldc,
                               batchCount);
}

hipblasStatus_t hipblasZsyrkBatchedCast(hipblasHandle_t                   handle,
                                        hipblasFillMode_t                 uplo,
                                        hipblasOperation_t                transA,
                                        int                               n,
                                        int                               k,
                                        const hipblasDoubleComplex*       alpha,
                                        const hipblasDoubleComplex* const A[],
                                        int                               lda,
                                        const hipblasDoubleComplex*       beta,
                                        hipblasDoubleComplex* const       C[],
                                        int                               ldc,
                                        int                               batchCount)
{
    return hipblasZsyrkBatched(handle,
                               uplo,
                               transA,
                               n,
                               k,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (const hipDoubleComplex*)beta,
                               (hipDoubleComplex* const*)C,
                               ldc,
                               batchCount);
}

// syrk_strided_batched
hipblasStatus_t hipblasCsyrkStridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasFillMode_t     uplo,
                                               hipblasOperation_t    transA,
                                               int                   n,
                                               int                   k,
                                               const hipblasComplex* alpha,
                                               const hipblasComplex* A,
                                               int                   lda,
                                               hipblasStride         strideA,
                                               const hipblasComplex* beta,
                                               hipblasComplex*       C,
                                               int                   ldc,
                                               hipblasStride         strideC,
                                               int                   batchCount)
{
    return hipblasCsyrkStridedBatched(handle,
                                      uplo,
                                      transA,
                                      n,
                                      k,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipComplex*)beta,
                                      (hipComplex*)C,
                                      ldc,
                                      strideC,
                                      batchCount);
}

hipblasStatus_t hipblasZsyrkStridedBatchedCast(hipblasHandle_t             handle,
                                               hipblasFillMode_t           uplo,
                                               hipblasOperation_t          transA,
                                               int                         n,
                                               int                         k,
                                               const hipblasDoubleComplex* alpha,
                                               const hipblasDoubleComplex* A,
                                               int                         lda,
                                               hipblasStride               strideA,
                                               const hipblasDoubleComplex* beta,
                                               hipblasDoubleComplex*       C,
                                               int                         ldc,
                                               hipblasStride               strideC,
                                               int                         batchCount)
{
    return hipblasZsyrkStridedBatched(handle,
                                      uplo,
                                      transA,
                                      n,
                                      k,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipDoubleComplex*)beta,
                                      (hipDoubleComplex*)C,
                                      ldc,
                                      strideC,
                                      batchCount);
}

// syrk_64
hipblasStatus_t hipblasCsyrkCast_64(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    hipblasOperation_t    transA,
                                    int64_t               n,
                                    int64_t               k,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* A,
                                    int64_t               lda,
                                    const hipblasComplex* beta,
                                    hipblasComplex*       C,
                                    int64_t               ldc)
{
    return hipblasCsyrk_64(handle,
                           uplo,
                           transA,
                           n,
                           k,
                           (const hipComplex*)alpha,
                           (const hipComplex*)A,
                           lda,
                           (const hipComplex*)beta,
                           (hipComplex*)C,
                           ldc);
}

hipblasStatus_t hipblasZsyrkCast_64(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    int64_t                     n,
                                    int64_t                     k,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* A,
                                    int64_t                     lda,
                                    const hipblasDoubleComplex* beta,
                                    hipblasDoubleComplex*       C,
                                    int64_t                     ldc)
{
    return hipblasZsyrk_64(handle,
                           uplo,
                           transA,
                           n,
                           k,
                           (const hipDoubleComplex*)alpha,
                           (const hipDoubleComplex*)A,
                           lda,
                           (const hipDoubleComplex*)beta,
                           (hipDoubleComplex*)C,
                           ldc);
}

// syrk_batched_64
hipblasStatus_t hipblasCsyrkBatchedCast_64(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           int64_t                     n,
                                           int64_t                     k,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const A[],
                                           int64_t                     lda,
                                           const hipblasComplex*       beta,
                                           hipblasComplex* const       C[],
                                           int64_t                     ldc,
                                           int64_t                     batchCount)
{
    return hipblasCsyrkBatched_64(handle,
                                  uplo,
                                  transA,
                                  n,
                                  k,
                                  (const hipComplex*)alpha,
                                  (const hipComplex* const*)A,
                                  lda,
                                  (const hipComplex*)beta,
                                  (hipComplex* const*)C,
                                  ldc,
                                  batchCount);
}

hipblasStatus_t hipblasZsyrkBatchedCast_64(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           hipblasOperation_t                transA,
                                           int64_t                           n,
                                           int64_t                           k,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const A[],
                                           int64_t                           lda,
                                           const hipblasDoubleComplex*       beta,
                                           hipblasDoubleComplex* const       C[],
                                           int64_t                           ldc,
                                           int64_t                           batchCount)
{
    return hipblasZsyrkBatched_64(handle,
                                  uplo,
                                  transA,
                                  n,
                                  k,
                                  (const hipDoubleComplex*)alpha,
                                  (const hipDoubleComplex* const*)A,
                                  lda,
                                  (const hipDoubleComplex*)beta,
                                  (hipDoubleComplex* const*)C,
                                  ldc,
                                  batchCount);
}

// syrk_strided_batched_64
hipblasStatus_t hipblasCsyrkStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  hipblasFillMode_t     uplo,
                                                  hipblasOperation_t    transA,
                                                  int64_t               n,
                                                  int64_t               k,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* A,
                                                  int64_t               lda,
                                                  hipblasStride         strideA,
                                                  const hipblasComplex* beta,
                                                  hipblasComplex*       C,
                                                  int64_t               ldc,
                                                  hipblasStride         strideC,
                                                  int64_t               batchCount)
{
    return hipblasCsyrkStridedBatched_64(handle,
                                         uplo,
                                         transA,
                                         n,
                                         k,
                                         (const hipComplex*)alpha,
                                         (const hipComplex*)A,
                                         lda,
                                         strideA,
                                         (const hipComplex*)beta,
                                         (hipComplex*)C,
                                         ldc,
                                         strideC,
                                         batchCount);
}

hipblasStatus_t hipblasZsyrkStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  hipblasFillMode_t           uplo,
                                                  hipblasOperation_t          transA,
                                                  int64_t                     n,
                                                  int64_t                     k,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* A,
                                                  int64_t                     lda,
                                                  hipblasStride               strideA,
                                                  const hipblasDoubleComplex* beta,
                                                  hipblasDoubleComplex*       C,
                                                  int64_t                     ldc,
                                                  hipblasStride               strideC,
                                                  int64_t                     batchCount)
{
    return hipblasZsyrkStridedBatched_64(handle,
                                         uplo,
                                         transA,
                                         n,
                                         k,
                                         (const hipDoubleComplex*)alpha,
                                         (const hipDoubleComplex*)A,
                                         lda,
                                         strideA,
                                         (const hipDoubleComplex*)beta,
                                         (hipDoubleComplex*)C,
                                         ldc,
                                         strideC,
                                         batchCount);
}

// syr2k
hipblasStatus_t hipblasCsyr2kCast(hipblasHandle_t       handle,
                                  hipblasFillMode_t     uplo,
                                  hipblasOperation_t    transA,
                                  int                   n,
                                  int                   k,
                                  const hipblasComplex* alpha,
                                  const hipblasComplex* A,
                                  int                   lda,
                                  const hipblasComplex* B,
                                  int                   ldb,
                                  const hipblasComplex* beta,
                                  hipblasComplex*       C,
                                  int                   ldc)
{
    return hipblasCsyr2k(handle,
                         uplo,
                         transA,
                         n,
                         k,
                         (const hipComplex*)alpha,
                         (const hipComplex*)A,
                         lda,
                         (const hipComplex*)B,
                         ldb,
                         (const hipComplex*)beta,
                         (hipComplex*)C,
                         ldc);
}

hipblasStatus_t hipblasZsyr2kCast(hipblasHandle_t             handle,
                                  hipblasFillMode_t           uplo,
                                  hipblasOperation_t          transA,
                                  int                         n,
                                  int                         k,
                                  const hipblasDoubleComplex* alpha,
                                  const hipblasDoubleComplex* A,
                                  int                         lda,
                                  const hipblasDoubleComplex* B,
                                  int                         ldb,
                                  const hipblasDoubleComplex* beta,
                                  hipblasDoubleComplex*       C,
                                  int                         ldc)
{
    return hipblasZsyr2k(handle,
                         uplo,
                         transA,
                         n,
                         k,
                         (const hipDoubleComplex*)alpha,
                         (const hipDoubleComplex*)A,
                         lda,
                         (const hipDoubleComplex*)B,
                         ldb,
                         (const hipDoubleComplex*)beta,
                         (hipDoubleComplex*)C,
                         ldc);
}

// syr2k_batched
hipblasStatus_t hipblasCsyr2kBatchedCast(hipblasHandle_t             handle,
                                         hipblasFillMode_t           uplo,
                                         hipblasOperation_t          transA,
                                         int                         n,
                                         int                         k,
                                         const hipblasComplex*       alpha,
                                         const hipblasComplex* const A[],
                                         int                         lda,
                                         const hipblasComplex* const B[],
                                         int                         ldb,
                                         const hipblasComplex*       beta,
                                         hipblasComplex* const       C[],
                                         int                         ldc,
                                         int                         batchCount)
{
    return hipblasCsyr2kBatched(handle,
                                uplo,
                                transA,
                                n,
                                k,
                                (const hipComplex*)alpha,
                                (const hipComplex* const*)A,
                                lda,
                                (const hipComplex* const*)B,
                                ldb,
                                (const hipComplex*)beta,
                                (hipComplex* const*)C,
                                ldc,
                                batchCount);
}

hipblasStatus_t hipblasZsyr2kBatchedCast(hipblasHandle_t                   handle,
                                         hipblasFillMode_t                 uplo,
                                         hipblasOperation_t                transA,
                                         int                               n,
                                         int                               k,
                                         const hipblasDoubleComplex*       alpha,
                                         const hipblasDoubleComplex* const A[],
                                         int                               lda,
                                         const hipblasDoubleComplex* const B[],
                                         int                               ldb,
                                         const hipblasDoubleComplex*       beta,
                                         hipblasDoubleComplex* const       C[],
                                         int                               ldc,
                                         int                               batchCount)
{
    return hipblasZsyr2kBatched(handle,
                                uplo,
                                transA,
                                n,
                                k,
                                (const hipDoubleComplex*)alpha,
                                (const hipDoubleComplex* const*)A,
                                lda,
                                (const hipDoubleComplex* const*)B,
                                ldb,
                                (const hipDoubleComplex*)beta,
                                (hipDoubleComplex* const*)C,
                                ldc,
                                batchCount);
}

// syr2k_strided_batched
hipblasStatus_t hipblasCsyr2kStridedBatchedCast(hipblasHandle_t       handle,
                                                hipblasFillMode_t     uplo,
                                                hipblasOperation_t    transA,
                                                int                   n,
                                                int                   k,
                                                const hipblasComplex* alpha,
                                                const hipblasComplex* A,
                                                int                   lda,
                                                hipblasStride         strideA,
                                                const hipblasComplex* B,
                                                int                   ldb,
                                                hipblasStride         strideB,
                                                const hipblasComplex* beta,
                                                hipblasComplex*       C,
                                                int                   ldc,
                                                hipblasStride         strideC,
                                                int                   batchCount)
{
    return hipblasCsyr2kStridedBatched(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       (const hipComplex*)alpha,
                                       (const hipComplex*)A,
                                       lda,
                                       strideA,
                                       (const hipComplex*)B,
                                       ldb,
                                       strideB,
                                       (const hipComplex*)beta,
                                       (hipComplex*)C,
                                       ldc,
                                       strideC,
                                       batchCount);
}

hipblasStatus_t hipblasZsyr2kStridedBatchedCast(hipblasHandle_t             handle,
                                                hipblasFillMode_t           uplo,
                                                hipblasOperation_t          transA,
                                                int                         n,
                                                int                         k,
                                                const hipblasDoubleComplex* alpha,
                                                const hipblasDoubleComplex* A,
                                                int                         lda,
                                                hipblasStride               strideA,
                                                const hipblasDoubleComplex* B,
                                                int                         ldb,
                                                hipblasStride               strideB,
                                                const hipblasDoubleComplex* beta,
                                                hipblasDoubleComplex*       C,
                                                int                         ldc,
                                                hipblasStride               strideC,
                                                int                         batchCount)
{
    return hipblasZsyr2kStridedBatched(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       (const hipDoubleComplex*)alpha,
                                       (const hipDoubleComplex*)A,
                                       lda,
                                       strideA,
                                       (const hipDoubleComplex*)B,
                                       ldb,
                                       strideB,
                                       (const hipDoubleComplex*)beta,
                                       (hipDoubleComplex*)C,
                                       ldc,
                                       strideC,
                                       batchCount);
}

// syr2k_64
hipblasStatus_t hipblasCsyr2kCast_64(hipblasHandle_t       handle,
                                     hipblasFillMode_t     uplo,
                                     hipblasOperation_t    transA,
                                     int64_t               n,
                                     int64_t               k,
                                     const hipblasComplex* alpha,
                                     const hipblasComplex* A,
                                     int64_t               lda,
                                     const hipblasComplex* B,
                                     int64_t               ldb,
                                     const hipblasComplex* beta,
                                     hipblasComplex*       C,
                                     int64_t               ldc)
{
    return hipblasCsyr2k_64(handle,
                            uplo,
                            transA,
                            n,
                            k,
                            (const hipComplex*)alpha,
                            (const hipComplex*)A,
                            lda,
                            (const hipComplex*)B,
                            ldb,
                            (const hipComplex*)beta,
                            (hipComplex*)C,
                            ldc);
}

hipblasStatus_t hipblasZsyr2kCast_64(hipblasHandle_t             handle,
                                     hipblasFillMode_t           uplo,
                                     hipblasOperation_t          transA,
                                     int64_t                     n,
                                     int64_t                     k,
                                     const hipblasDoubleComplex* alpha,
                                     const hipblasDoubleComplex* A,
                                     int64_t                     lda,
                                     const hipblasDoubleComplex* B,
                                     int64_t                     ldb,
                                     const hipblasDoubleComplex* beta,
                                     hipblasDoubleComplex*       C,
                                     int64_t                     ldc)
{
    return hipblasZsyr2k_64(handle,
                            uplo,
                            transA,
                            n,
                            k,
                            (const hipDoubleComplex*)alpha,
                            (const hipDoubleComplex*)A,
                            lda,
                            (const hipDoubleComplex*)B,
                            ldb,
                            (const hipDoubleComplex*)beta,
                            (hipDoubleComplex*)C,
                            ldc);
}

// syr2k_batched_64
hipblasStatus_t hipblasCsyr2kBatchedCast_64(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            hipblasOperation_t          transA,
                                            int64_t                     n,
                                            int64_t                     k,
                                            const hipblasComplex*       alpha,
                                            const hipblasComplex* const A[],
                                            int64_t                     lda,
                                            const hipblasComplex* const B[],
                                            int64_t                     ldb,
                                            const hipblasComplex*       beta,
                                            hipblasComplex* const       C[],
                                            int64_t                     ldc,
                                            int64_t                     batchCount)
{
    return hipblasCsyr2kBatched_64(handle,
                                   uplo,
                                   transA,
                                   n,
                                   k,
                                   (const hipComplex*)alpha,
                                   (const hipComplex* const*)A,
                                   lda,
                                   (const hipComplex* const*)B,
                                   ldb,
                                   (const hipComplex*)beta,
                                   (hipComplex* const*)C,
                                   ldc,
                                   batchCount);
}

hipblasStatus_t hipblasZsyr2kBatchedCast_64(hipblasHandle_t                   handle,
                                            hipblasFillMode_t                 uplo,
                                            hipblasOperation_t                transA,
                                            int64_t                           n,
                                            int64_t                           k,
                                            const hipblasDoubleComplex*       alpha,
                                            const hipblasDoubleComplex* const A[],
                                            int64_t                           lda,
                                            const hipblasDoubleComplex* const B[],
                                            int64_t                           ldb,
                                            const hipblasDoubleComplex*       beta,
                                            hipblasDoubleComplex* const       C[],
                                            int64_t                           ldc,
                                            int64_t                           batchCount)
{
    return hipblasZsyr2kBatched_64(handle,
                                   uplo,
                                   transA,
                                   n,
                                   k,
                                   (const hipDoubleComplex*)alpha,
                                   (const hipDoubleComplex* const*)A,
                                   lda,
                                   (const hipDoubleComplex* const*)B,
                                   ldb,
                                   (const hipDoubleComplex*)beta,
                                   (hipDoubleComplex* const*)C,
                                   ldc,
                                   batchCount);
}

// syr2k_strided_batched_64
hipblasStatus_t hipblasCsyr2kStridedBatchedCast_64(hipblasHandle_t       handle,
                                                   hipblasFillMode_t     uplo,
                                                   hipblasOperation_t    transA,
                                                   int64_t               n,
                                                   int64_t               k,
                                                   const hipblasComplex* alpha,
                                                   const hipblasComplex* A,
                                                   int64_t               lda,
                                                   hipblasStride         strideA,
                                                   const hipblasComplex* B,
                                                   int64_t               ldb,
                                                   hipblasStride         strideB,
                                                   const hipblasComplex* beta,
                                                   hipblasComplex*       C,
                                                   int64_t               ldc,
                                                   hipblasStride         strideC,
                                                   int64_t               batchCount)
{
    return hipblasCsyr2kStridedBatched_64(handle,
                                          uplo,
                                          transA,
                                          n,
                                          k,
                                          (const hipComplex*)alpha,
                                          (const hipComplex*)A,
                                          lda,
                                          strideA,
                                          (const hipComplex*)B,
                                          ldb,
                                          strideB,
                                          (const hipComplex*)beta,
                                          (hipComplex*)C,
                                          ldc,
                                          strideC,
                                          batchCount);
}

hipblasStatus_t hipblasZsyr2kStridedBatchedCast_64(hipblasHandle_t             handle,
                                                   hipblasFillMode_t           uplo,
                                                   hipblasOperation_t          transA,
                                                   int64_t                     n,
                                                   int64_t                     k,
                                                   const hipblasDoubleComplex* alpha,
                                                   const hipblasDoubleComplex* A,
                                                   int64_t                     lda,
                                                   hipblasStride               strideA,
                                                   const hipblasDoubleComplex* B,
                                                   int64_t                     ldb,
                                                   hipblasStride               strideB,
                                                   const hipblasDoubleComplex* beta,
                                                   hipblasDoubleComplex*       C,
                                                   int64_t                     ldc,
                                                   hipblasStride               strideC,
                                                   int64_t                     batchCount)
{
    return hipblasZsyr2kStridedBatched_64(handle,
                                          uplo,
                                          transA,
                                          n,
                                          k,
                                          (const hipDoubleComplex*)alpha,
                                          (const hipDoubleComplex*)A,
                                          lda,
                                          strideA,
                                          (const hipDoubleComplex*)B,
                                          ldb,
                                          strideB,
                                          (const hipDoubleComplex*)beta,
                                          (hipDoubleComplex*)C,
                                          ldc,
                                          strideC,
                                          batchCount);
}

// syrkx
hipblasStatus_t hipblasCsyrkxCast(hipblasHandle_t       handle,
                                  hipblasFillMode_t     uplo,
                                  hipblasOperation_t    transA,
                                  int                   n,
                                  int                   k,
                                  const hipblasComplex* alpha,
                                  const hipblasComplex* A,
                                  int                   lda,
                                  const hipblasComplex* B,
                                  int                   ldb,
                                  const hipblasComplex* beta,
                                  hipblasComplex*       C,
                                  int                   ldc)
{
    return hipblasCsyrkx(handle,
                         uplo,
                         transA,
                         n,
                         k,
                         (const hipComplex*)alpha,
                         (const hipComplex*)A,
                         lda,
                         (const hipComplex*)B,
                         ldb,
                         (const hipComplex*)beta,
                         (hipComplex*)C,
                         ldc);
}

hipblasStatus_t hipblasZsyrkxCast(hipblasHandle_t             handle,
                                  hipblasFillMode_t           uplo,
                                  hipblasOperation_t          transA,
                                  int                         n,
                                  int                         k,
                                  const hipblasDoubleComplex* alpha,
                                  const hipblasDoubleComplex* A,
                                  int                         lda,
                                  const hipblasDoubleComplex* B,
                                  int                         ldb,
                                  const hipblasDoubleComplex* beta,
                                  hipblasDoubleComplex*       C,
                                  int                         ldc)
{
    return hipblasZsyrkx(handle,
                         uplo,
                         transA,
                         n,
                         k,
                         (const hipDoubleComplex*)alpha,
                         (const hipDoubleComplex*)A,
                         lda,
                         (const hipDoubleComplex*)B,
                         ldb,
                         (const hipDoubleComplex*)beta,
                         (hipDoubleComplex*)C,
                         ldc);
}

// syrkx_batched
hipblasStatus_t hipblasCsyrkxBatchedCast(hipblasHandle_t             handle,
                                         hipblasFillMode_t           uplo,
                                         hipblasOperation_t          transA,
                                         int                         n,
                                         int                         k,
                                         const hipblasComplex*       alpha,
                                         const hipblasComplex* const A[],
                                         int                         lda,
                                         const hipblasComplex* const B[],
                                         int                         ldb,
                                         const hipblasComplex*       beta,
                                         hipblasComplex* const       C[],
                                         int                         ldc,
                                         int                         batchCount)
{
    return hipblasCsyrkxBatched(handle,
                                uplo,
                                transA,
                                n,
                                k,
                                (const hipComplex*)alpha,
                                (const hipComplex* const*)A,
                                lda,
                                (const hipComplex* const*)B,
                                ldb,
                                (const hipComplex*)beta,
                                (hipComplex* const*)C,
                                ldc,
                                batchCount);
}

hipblasStatus_t hipblasZsyrkxBatchedCast(hipblasHandle_t                   handle,
                                         hipblasFillMode_t                 uplo,
                                         hipblasOperation_t                transA,
                                         int                               n,
                                         int                               k,
                                         const hipblasDoubleComplex*       alpha,
                                         const hipblasDoubleComplex* const A[],
                                         int                               lda,
                                         const hipblasDoubleComplex* const B[],
                                         int                               ldb,
                                         const hipblasDoubleComplex*       beta,
                                         hipblasDoubleComplex* const       C[],
                                         int                               ldc,
                                         int                               batchCount)
{
    return hipblasZsyrkxBatched(handle,
                                uplo,
                                transA,
                                n,
                                k,
                                (const hipDoubleComplex*)alpha,
                                (const hipDoubleComplex* const*)A,
                                lda,
                                (const hipDoubleComplex* const*)B,
                                ldb,
                                (const hipDoubleComplex*)beta,
                                (hipDoubleComplex* const*)C,
                                ldc,
                                batchCount);
}

// syrkx_strided_batched
hipblasStatus_t hipblasCsyrkxStridedBatchedCast(hipblasHandle_t       handle,
                                                hipblasFillMode_t     uplo,
                                                hipblasOperation_t    transA,
                                                int                   n,
                                                int                   k,
                                                const hipblasComplex* alpha,
                                                const hipblasComplex* A,
                                                int                   lda,
                                                hipblasStride         strideA,
                                                const hipblasComplex* B,
                                                int                   ldb,
                                                hipblasStride         strideB,
                                                const hipblasComplex* beta,
                                                hipblasComplex*       C,
                                                int                   ldc,
                                                hipblasStride         strideC,
                                                int                   batchCount)
{
    return hipblasCsyrkxStridedBatched(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       (const hipComplex*)alpha,
                                       (const hipComplex*)A,
                                       lda,
                                       strideA,
                                       (const hipComplex*)B,
                                       ldb,
                                       strideB,
                                       (const hipComplex*)beta,
                                       (hipComplex*)C,
                                       ldc,
                                       strideC,
                                       batchCount);
}

hipblasStatus_t hipblasZsyrkxStridedBatchedCast(hipblasHandle_t             handle,
                                                hipblasFillMode_t           uplo,
                                                hipblasOperation_t          transA,
                                                int                         n,
                                                int                         k,
                                                const hipblasDoubleComplex* alpha,
                                                const hipblasDoubleComplex* A,
                                                int                         lda,
                                                hipblasStride               strideA,
                                                const hipblasDoubleComplex* B,
                                                int                         ldb,
                                                hipblasStride               strideB,
                                                const hipblasDoubleComplex* beta,
                                                hipblasDoubleComplex*       C,
                                                int                         ldc,
                                                hipblasStride               strideC,
                                                int                         batchCount)
{
    return hipblasZsyrkxStridedBatched(handle,
                                       uplo,
                                       transA,
                                       n,
                                       k,
                                       (const hipDoubleComplex*)alpha,
                                       (const hipDoubleComplex*)A,
                                       lda,
                                       strideA,
                                       (const hipDoubleComplex*)B,
                                       ldb,
                                       strideB,
                                       (const hipDoubleComplex*)beta,
                                       (hipDoubleComplex*)C,
                                       ldc,
                                       strideC,
                                       batchCount);
}

// syrkx_64
hipblasStatus_t hipblasCsyrkxCast_64(hipblasHandle_t       handle,
                                     hipblasFillMode_t     uplo,
                                     hipblasOperation_t    transA,
                                     int64_t               n,
                                     int64_t               k,
                                     const hipblasComplex* alpha,
                                     const hipblasComplex* A,
                                     int64_t               lda,
                                     const hipblasComplex* B,
                                     int64_t               ldb,
                                     const hipblasComplex* beta,
                                     hipblasComplex*       C,
                                     int64_t               ldc)
{
    return hipblasCsyrkx_64(handle,
                            uplo,
                            transA,
                            n,
                            k,
                            (const hipComplex*)alpha,
                            (const hipComplex*)A,
                            lda,
                            (const hipComplex*)B,
                            ldb,
                            (const hipComplex*)beta,
                            (hipComplex*)C,
                            ldc);
}

hipblasStatus_t hipblasZsyrkxCast_64(hipblasHandle_t             handle,
                                     hipblasFillMode_t           uplo,
                                     hipblasOperation_t          transA,
                                     int64_t                     n,
                                     int64_t                     k,
                                     const hipblasDoubleComplex* alpha,
                                     const hipblasDoubleComplex* A,
                                     int64_t                     lda,
                                     const hipblasDoubleComplex* B,
                                     int64_t                     ldb,
                                     const hipblasDoubleComplex* beta,
                                     hipblasDoubleComplex*       C,
                                     int64_t                     ldc)
{
    return hipblasZsyrkx_64(handle,
                            uplo,
                            transA,
                            n,
                            k,
                            (const hipDoubleComplex*)alpha,
                            (const hipDoubleComplex*)A,
                            lda,
                            (const hipDoubleComplex*)B,
                            ldb,
                            (const hipDoubleComplex*)beta,
                            (hipDoubleComplex*)C,
                            ldc);
}

// syrkx_batched_64
hipblasStatus_t hipblasCsyrkxBatchedCast_64(hipblasHandle_t             handle,
                                            hipblasFillMode_t           uplo,
                                            hipblasOperation_t          transA,
                                            int64_t                     n,
                                            int64_t                     k,
                                            const hipblasComplex*       alpha,
                                            const hipblasComplex* const A[],
                                            int64_t                     lda,
                                            const hipblasComplex* const B[],
                                            int64_t                     ldb,
                                            const hipblasComplex*       beta,
                                            hipblasComplex* const       C[],
                                            int64_t                     ldc,
                                            int64_t                     batchCount)
{
    return hipblasCsyrkxBatched_64(handle,
                                   uplo,
                                   transA,
                                   n,
                                   k,
                                   (const hipComplex*)alpha,
                                   (const hipComplex* const*)A,
                                   lda,
                                   (const hipComplex* const*)B,
                                   ldb,
                                   (const hipComplex*)beta,
                                   (hipComplex* const*)C,
                                   ldc,
                                   batchCount);
}

hipblasStatus_t hipblasZsyrkxBatchedCast_64(hipblasHandle_t                   handle,
                                            hipblasFillMode_t                 uplo,
                                            hipblasOperation_t                transA,
                                            int64_t                           n,
                                            int64_t                           k,
                                            const hipblasDoubleComplex*       alpha,
                                            const hipblasDoubleComplex* const A[],
                                            int64_t                           lda,
                                            const hipblasDoubleComplex* const B[],
                                            int64_t                           ldb,
                                            const hipblasDoubleComplex*       beta,
                                            hipblasDoubleComplex* const       C[],
                                            int64_t                           ldc,
                                            int64_t                           batchCount)
{
    return hipblasZsyrkxBatched_64(handle,
                                   uplo,
                                   transA,
                                   n,
                                   k,
                                   (const hipDoubleComplex*)alpha,
                                   (const hipDoubleComplex* const*)A,
                                   lda,
                                   (const hipDoubleComplex* const*)B,
                                   ldb,
                                   (const hipDoubleComplex*)beta,
                                   (hipDoubleComplex* const*)C,
                                   ldc,
                                   batchCount);
}

// syrkx_strided_batched_64
hipblasStatus_t hipblasCsyrkxStridedBatchedCast_64(hipblasHandle_t       handle,
                                                   hipblasFillMode_t     uplo,
                                                   hipblasOperation_t    transA,
                                                   int64_t               n,
                                                   int64_t               k,
                                                   const hipblasComplex* alpha,
                                                   const hipblasComplex* A,
                                                   int64_t               lda,
                                                   hipblasStride         strideA,
                                                   const hipblasComplex* B,
                                                   int64_t               ldb,
                                                   hipblasStride         strideB,
                                                   const hipblasComplex* beta,
                                                   hipblasComplex*       C,
                                                   int64_t               ldc,
                                                   hipblasStride         strideC,
                                                   int64_t               batchCount)
{
    return hipblasCsyrkxStridedBatched_64(handle,
                                          uplo,
                                          transA,
                                          n,
                                          k,
                                          (const hipComplex*)alpha,
                                          (const hipComplex*)A,
                                          lda,
                                          strideA,
                                          (const hipComplex*)B,
                                          ldb,
                                          strideB,
                                          (const hipComplex*)beta,
                                          (hipComplex*)C,
                                          ldc,
                                          strideC,
                                          batchCount);
}

hipblasStatus_t hipblasZsyrkxStridedBatchedCast_64(hipblasHandle_t             handle,
                                                   hipblasFillMode_t           uplo,
                                                   hipblasOperation_t          transA,
                                                   int64_t                     n,
                                                   int64_t                     k,
                                                   const hipblasDoubleComplex* alpha,
                                                   const hipblasDoubleComplex* A,
                                                   int64_t                     lda,
                                                   hipblasStride               strideA,
                                                   const hipblasDoubleComplex* B,
                                                   int64_t                     ldb,
                                                   hipblasStride               strideB,
                                                   const hipblasDoubleComplex* beta,
                                                   hipblasDoubleComplex*       C,
                                                   int64_t                     ldc,
                                                   hipblasStride               strideC,
                                                   int64_t                     batchCount)
{
    return hipblasZsyrkxStridedBatched_64(handle,
                                          uplo,
                                          transA,
                                          n,
                                          k,
                                          (const hipDoubleComplex*)alpha,
                                          (const hipDoubleComplex*)A,
                                          lda,
                                          strideA,
                                          (const hipDoubleComplex*)B,
                                          ldb,
                                          strideB,
                                          (const hipDoubleComplex*)beta,
                                          (hipDoubleComplex*)C,
                                          ldc,
                                          strideC,
                                          batchCount);
}

// hemm
hipblasStatus_t hipblasChemmCast(hipblasHandle_t       handle,
                                 hipblasSideMode_t     side,
                                 hipblasFillMode_t     uplo,
                                 int                   n,
                                 int                   k,
                                 const hipblasComplex* alpha,
                                 const hipblasComplex* A,
                                 int                   lda,
                                 const hipblasComplex* B,
                                 int                   ldb,
                                 const hipblasComplex* beta,
                                 hipblasComplex*       C,
                                 int                   ldc)
{
    return hipblasChemm(handle,
                        side,
                        uplo,
                        n,
                        k,
                        (const hipComplex*)alpha,
                        (const hipComplex*)A,
                        lda,
                        (const hipComplex*)B,
                        ldb,
                        (const hipComplex*)beta,
                        (hipComplex*)C,
                        ldc);
}

hipblasStatus_t hipblasZhemmCast(hipblasHandle_t             handle,
                                 hipblasSideMode_t           side,
                                 hipblasFillMode_t           uplo,
                                 int                         n,
                                 int                         k,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* A,
                                 int                         lda,
                                 const hipblasDoubleComplex* B,
                                 int                         ldb,
                                 const hipblasDoubleComplex* beta,
                                 hipblasDoubleComplex*       C,
                                 int                         ldc)
{
    return hipblasZhemm(handle,
                        side,
                        uplo,
                        n,
                        k,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)A,
                        lda,
                        (const hipDoubleComplex*)B,
                        ldb,
                        (const hipDoubleComplex*)beta,
                        (hipDoubleComplex*)C,
                        ldc);
}

// hemm_batched
hipblasStatus_t hipblasChemmBatchedCast(hipblasHandle_t             handle,
                                        hipblasSideMode_t           side,
                                        hipblasFillMode_t           uplo,
                                        int                         n,
                                        int                         k,
                                        const hipblasComplex*       alpha,
                                        const hipblasComplex* const A[],
                                        int                         lda,
                                        const hipblasComplex* const B[],
                                        int                         ldb,
                                        const hipblasComplex*       beta,
                                        hipblasComplex* const       C[],
                                        int                         ldc,
                                        int                         batchCount)
{
    return hipblasChemmBatched(handle,
                               side,
                               uplo,
                               n,
                               k,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)A,
                               lda,
                               (const hipComplex* const*)B,
                               ldb,
                               (const hipComplex*)beta,
                               (hipComplex* const*)C,
                               ldc,
                               batchCount);
}

hipblasStatus_t hipblasZhemmBatchedCast(hipblasHandle_t                   handle,
                                        hipblasSideMode_t                 side,
                                        hipblasFillMode_t                 uplo,
                                        int                               n,
                                        int                               k,
                                        const hipblasDoubleComplex*       alpha,
                                        const hipblasDoubleComplex* const A[],
                                        int                               lda,
                                        const hipblasDoubleComplex* const B[],
                                        int                               ldb,
                                        const hipblasDoubleComplex*       beta,
                                        hipblasDoubleComplex* const       C[],
                                        int                               ldc,
                                        int                               batchCount)
{
    return hipblasZhemmBatched(handle,
                               side,
                               uplo,
                               n,
                               k,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (const hipDoubleComplex* const*)B,
                               ldb,
                               (const hipDoubleComplex*)beta,
                               (hipDoubleComplex* const*)C,
                               ldc,
                               batchCount);
}

// hemm_strided_batched
hipblasStatus_t hipblasChemmStridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasSideMode_t     side,
                                               hipblasFillMode_t     uplo,
                                               int                   n,
                                               int                   k,
                                               const hipblasComplex* alpha,
                                               const hipblasComplex* A,
                                               int                   lda,
                                               hipblasStride         strideA,
                                               const hipblasComplex* B,
                                               int                   ldb,
                                               hipblasStride         strideB,
                                               const hipblasComplex* beta,
                                               hipblasComplex*       C,
                                               int                   ldc,
                                               hipblasStride         strideC,
                                               int                   batchCount)
{
    return hipblasChemmStridedBatched(handle,
                                      side,
                                      uplo,
                                      n,
                                      k,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipComplex*)B,
                                      ldb,
                                      strideB,
                                      (const hipComplex*)beta,
                                      (hipComplex*)C,
                                      ldc,
                                      strideC,
                                      batchCount);
}

hipblasStatus_t hipblasZhemmStridedBatchedCast(hipblasHandle_t             handle,
                                               hipblasSideMode_t           side,
                                               hipblasFillMode_t           uplo,
                                               int                         n,
                                               int                         k,
                                               const hipblasDoubleComplex* alpha,
                                               const hipblasDoubleComplex* A,
                                               int                         lda,
                                               hipblasStride               strideA,
                                               const hipblasDoubleComplex* B,
                                               int                         ldb,
                                               hipblasStride               strideB,
                                               const hipblasDoubleComplex* beta,
                                               hipblasDoubleComplex*       C,
                                               int                         ldc,
                                               hipblasStride               strideC,
                                               int                         batchCount)
{
    return hipblasZhemmStridedBatched(handle,
                                      side,
                                      uplo,
                                      n,
                                      k,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipDoubleComplex*)B,
                                      ldb,
                                      strideB,
                                      (const hipDoubleComplex*)beta,
                                      (hipDoubleComplex*)C,
                                      ldc,
                                      strideC,
                                      batchCount);
}

// hemm_64
hipblasStatus_t hipblasChemmCast_64(hipblasHandle_t       handle,
                                    hipblasSideMode_t     side,
                                    hipblasFillMode_t     uplo,
                                    int64_t               n,
                                    int64_t               k,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* A,
                                    int64_t               lda,
                                    const hipblasComplex* B,
                                    int64_t               ldb,
                                    const hipblasComplex* beta,
                                    hipblasComplex*       C,
                                    int64_t               ldc)
{
    return hipblasChemm_64(handle,
                           side,
                           uplo,
                           n,
                           k,
                           (const hipComplex*)alpha,
                           (const hipComplex*)A,
                           lda,
                           (const hipComplex*)B,
                           ldb,
                           (const hipComplex*)beta,
                           (hipComplex*)C,
                           ldc);
}

hipblasStatus_t hipblasZhemmCast_64(hipblasHandle_t             handle,
                                    hipblasSideMode_t           side,
                                    hipblasFillMode_t           uplo,
                                    int64_t                     n,
                                    int64_t                     k,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* A,
                                    int64_t                     lda,
                                    const hipblasDoubleComplex* B,
                                    int64_t                     ldb,
                                    const hipblasDoubleComplex* beta,
                                    hipblasDoubleComplex*       C,
                                    int64_t                     ldc)
{
    return hipblasZhemm_64(handle,
                           side,
                           uplo,
                           n,
                           k,
                           (const hipDoubleComplex*)alpha,
                           (const hipDoubleComplex*)A,
                           lda,
                           (const hipDoubleComplex*)B,
                           ldb,
                           (const hipDoubleComplex*)beta,
                           (hipDoubleComplex*)C,
                           ldc);
}

// hemm_batched_64
hipblasStatus_t hipblasChemmBatchedCast_64(hipblasHandle_t             handle,
                                           hipblasSideMode_t           side,
                                           hipblasFillMode_t           uplo,
                                           int64_t                     n,
                                           int64_t                     k,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const A[],
                                           int64_t                     lda,
                                           const hipblasComplex* const B[],
                                           int64_t                     ldb,
                                           const hipblasComplex*       beta,
                                           hipblasComplex* const       C[],
                                           int64_t                     ldc,
                                           int64_t                     batchCount)
{
    return hipblasChemmBatched_64(handle,
                                  side,
                                  uplo,
                                  n,
                                  k,
                                  (const hipComplex*)alpha,
                                  (const hipComplex* const*)A,
                                  lda,
                                  (const hipComplex* const*)B,
                                  ldb,
                                  (const hipComplex*)beta,
                                  (hipComplex* const*)C,
                                  ldc,
                                  batchCount);
}

hipblasStatus_t hipblasZhemmBatchedCast_64(hipblasHandle_t                   handle,
                                           hipblasSideMode_t                 side,
                                           hipblasFillMode_t                 uplo,
                                           int64_t                           n,
                                           int64_t                           k,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const A[],
                                           int64_t                           lda,
                                           const hipblasDoubleComplex* const B[],
                                           int64_t                           ldb,
                                           const hipblasDoubleComplex*       beta,
                                           hipblasDoubleComplex* const       C[],
                                           int64_t                           ldc,
                                           int64_t                           batchCount)
{
    return hipblasZhemmBatched_64(handle,
                                  side,
                                  uplo,
                                  n,
                                  k,
                                  (const hipDoubleComplex*)alpha,
                                  (const hipDoubleComplex* const*)A,
                                  lda,
                                  (const hipDoubleComplex* const*)B,
                                  ldb,
                                  (const hipDoubleComplex*)beta,
                                  (hipDoubleComplex* const*)C,
                                  ldc,
                                  batchCount);
}

// hemm_strided_batched_64
hipblasStatus_t hipblasChemmStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  hipblasSideMode_t     side,
                                                  hipblasFillMode_t     uplo,
                                                  int64_t               n,
                                                  int64_t               k,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* A,
                                                  int64_t               lda,
                                                  hipblasStride         strideA,
                                                  const hipblasComplex* B,
                                                  int64_t               ldb,
                                                  hipblasStride         strideB,
                                                  const hipblasComplex* beta,
                                                  hipblasComplex*       C,
                                                  int64_t               ldc,
                                                  hipblasStride         strideC,
                                                  int64_t               batchCount)
{
    return hipblasChemmStridedBatched_64(handle,
                                         side,
                                         uplo,
                                         n,
                                         k,
                                         (const hipComplex*)alpha,
                                         (const hipComplex*)A,
                                         lda,
                                         strideA,
                                         (const hipComplex*)B,
                                         ldb,
                                         strideB,
                                         (const hipComplex*)beta,
                                         (hipComplex*)C,
                                         ldc,
                                         strideC,
                                         batchCount);
}

hipblasStatus_t hipblasZhemmStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  hipblasSideMode_t           side,
                                                  hipblasFillMode_t           uplo,
                                                  int64_t                     n,
                                                  int64_t                     k,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* A,
                                                  int64_t                     lda,
                                                  hipblasStride               strideA,
                                                  const hipblasDoubleComplex* B,
                                                  int64_t                     ldb,
                                                  hipblasStride               strideB,
                                                  const hipblasDoubleComplex* beta,
                                                  hipblasDoubleComplex*       C,
                                                  int64_t                     ldc,
                                                  hipblasStride               strideC,
                                                  int64_t                     batchCount)
{
    return hipblasZhemmStridedBatched_64(handle,
                                         side,
                                         uplo,
                                         n,
                                         k,
                                         (const hipDoubleComplex*)alpha,
                                         (const hipDoubleComplex*)A,
                                         lda,
                                         strideA,
                                         (const hipDoubleComplex*)B,
                                         ldb,
                                         strideB,
                                         (const hipDoubleComplex*)beta,
                                         (hipDoubleComplex*)C,
                                         ldc,
                                         strideC,
                                         batchCount);
}

// trmm
hipblasStatus_t hipblasCtrmmCast(hipblasHandle_t       handle,
                                 hipblasSideMode_t     side,
                                 hipblasFillMode_t     uplo,
                                 hipblasOperation_t    transA,
                                 hipblasDiagType_t     diag,
                                 int                   m,
                                 int                   n,
                                 const hipblasComplex* alpha,
                                 const hipblasComplex* A,
                                 int                   lda,
                                 const hipblasComplex* B,
                                 int                   ldb,
                                 hipblasComplex*       C,
                                 int                   ldc)
{
    return hipblasCtrmm(handle,
                        side,
                        uplo,
                        transA,
                        diag,
                        m,
                        n,
                        (const hipComplex*)alpha,
                        (const hipComplex*)A,
                        lda,
                        (const hipComplex*)B,
                        ldb,
                        (hipComplex*)C,
                        ldc);
}

hipblasStatus_t hipblasZtrmmCast(hipblasHandle_t             handle,
                                 hipblasSideMode_t           side,
                                 hipblasFillMode_t           uplo,
                                 hipblasOperation_t          transA,
                                 hipblasDiagType_t           diag,
                                 int                         m,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* A,
                                 int                         lda,
                                 const hipblasDoubleComplex* B,
                                 int                         ldb,
                                 hipblasDoubleComplex*       C,
                                 int                         ldc)
{
    return hipblasZtrmm(handle,
                        side,
                        uplo,
                        transA,
                        diag,
                        m,
                        n,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)A,
                        lda,
                        (const hipDoubleComplex*)B,
                        ldb,
                        (hipDoubleComplex*)C,
                        ldc);
}

// trmm_batched
hipblasStatus_t hipblasCtrmmBatchedCast(hipblasHandle_t             handle,
                                        hipblasSideMode_t           side,
                                        hipblasFillMode_t           uplo,
                                        hipblasOperation_t          transA,
                                        hipblasDiagType_t           diag,
                                        int                         m,
                                        int                         n,
                                        const hipblasComplex*       alpha,
                                        const hipblasComplex* const A[],
                                        int                         lda,
                                        const hipblasComplex* const B[],
                                        int                         ldb,
                                        hipblasComplex* const       C[],
                                        int                         ldc,
                                        int                         batchCount)
{
    return hipblasCtrmmBatched(handle,
                               side,
                               uplo,
                               transA,
                               diag,
                               m,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)A,
                               lda,
                               (const hipComplex* const*)B,
                               ldb,
                               (hipComplex* const*)C,
                               ldc,
                               batchCount);
}

hipblasStatus_t hipblasZtrmmBatchedCast(hipblasHandle_t                   handle,
                                        hipblasSideMode_t                 side,
                                        hipblasFillMode_t                 uplo,
                                        hipblasOperation_t                transA,
                                        hipblasDiagType_t                 diag,
                                        int                               m,
                                        int                               n,
                                        const hipblasDoubleComplex*       alpha,
                                        const hipblasDoubleComplex* const A[],
                                        int                               lda,
                                        const hipblasDoubleComplex* const B[],
                                        int                               ldb,
                                        hipblasDoubleComplex* const       C[],
                                        int                               ldc,
                                        int                               batchCount)
{
    return hipblasZtrmmBatched(handle,
                               side,
                               uplo,
                               transA,
                               diag,
                               m,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (const hipDoubleComplex* const*)B,
                               ldb,
                               (hipDoubleComplex* const*)C,
                               ldc,
                               batchCount);
}

// trmm_strided_batched
hipblasStatus_t hipblasCtrmmStridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasSideMode_t     side,
                                               hipblasFillMode_t     uplo,
                                               hipblasOperation_t    transA,
                                               hipblasDiagType_t     diag,
                                               int                   m,
                                               int                   n,
                                               const hipblasComplex* alpha,
                                               const hipblasComplex* A,
                                               int                   lda,
                                               hipblasStride         strideA,
                                               const hipblasComplex* B,
                                               int                   ldb,
                                               hipblasStride         strideB,
                                               hipblasComplex*       C,
                                               int                   ldc,
                                               hipblasStride         strideC,
                                               int                   batchCount)
{
    return hipblasCtrmmStridedBatched(handle,
                                      side,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipComplex*)B,
                                      ldb,
                                      strideB,
                                      (hipComplex*)C,
                                      ldc,
                                      strideC,
                                      batchCount);
}

hipblasStatus_t hipblasZtrmmStridedBatchedCast(hipblasHandle_t             handle,
                                               hipblasSideMode_t           side,
                                               hipblasFillMode_t           uplo,
                                               hipblasOperation_t          transA,
                                               hipblasDiagType_t           diag,
                                               int                         m,
                                               int                         n,
                                               const hipblasDoubleComplex* alpha,
                                               const hipblasDoubleComplex* A,
                                               int                         lda,
                                               hipblasStride               strideA,
                                               const hipblasDoubleComplex* B,
                                               int                         ldb,
                                               hipblasStride               strideB,
                                               hipblasDoubleComplex*       C,
                                               int                         ldc,
                                               hipblasStride               strideC,
                                               int                         batchCount)
{
    return hipblasZtrmmStridedBatched(handle,
                                      side,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipDoubleComplex*)B,
                                      ldb,
                                      strideB,
                                      (hipDoubleComplex*)C,
                                      ldc,
                                      strideC,
                                      batchCount);
}

// trmm_64
hipblasStatus_t hipblasCtrmmCast_64(hipblasHandle_t       handle,
                                    hipblasSideMode_t     side,
                                    hipblasFillMode_t     uplo,
                                    hipblasOperation_t    transA,
                                    hipblasDiagType_t     diag,
                                    int64_t               m,
                                    int64_t               n,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* A,
                                    int64_t               lda,
                                    const hipblasComplex* B,
                                    int64_t               ldb,
                                    hipblasComplex*       C,
                                    int64_t               ldc)
{
    return hipblasCtrmm_64(handle,
                           side,
                           uplo,
                           transA,
                           diag,
                           m,
                           n,
                           (const hipComplex*)alpha,
                           (const hipComplex*)A,
                           lda,
                           (const hipComplex*)B,
                           ldb,
                           (hipComplex*)C,
                           ldc);
}

hipblasStatus_t hipblasZtrmmCast_64(hipblasHandle_t             handle,
                                    hipblasSideMode_t           side,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int64_t                     m,
                                    int64_t                     n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* A,
                                    int64_t                     lda,
                                    const hipblasDoubleComplex* B,
                                    int64_t                     ldb,
                                    hipblasDoubleComplex*       C,
                                    int64_t                     ldc)
{
    return hipblasZtrmm_64(handle,
                           side,
                           uplo,
                           transA,
                           diag,
                           m,
                           n,
                           (const hipDoubleComplex*)alpha,
                           (const hipDoubleComplex*)A,
                           lda,
                           (const hipDoubleComplex*)B,
                           ldb,
                           (hipDoubleComplex*)C,
                           ldc);
}

// trmm_batched_64
hipblasStatus_t hipblasCtrmmBatchedCast_64(hipblasHandle_t             handle,
                                           hipblasSideMode_t           side,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int64_t                     m,
                                           int64_t                     n,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const A[],
                                           int64_t                     lda,
                                           const hipblasComplex* const B[],
                                           int64_t                     ldb,
                                           hipblasComplex* const       C[],
                                           int64_t                     ldc,
                                           int64_t                     batchCount)
{
    return hipblasCtrmmBatched_64(handle,
                                  side,
                                  uplo,
                                  transA,
                                  diag,
                                  m,
                                  n,
                                  (const hipComplex*)alpha,
                                  (const hipComplex* const*)A,
                                  lda,
                                  (const hipComplex* const*)B,
                                  ldb,
                                  (hipComplex* const*)C,
                                  ldc,
                                  batchCount);
}

hipblasStatus_t hipblasZtrmmBatchedCast_64(hipblasHandle_t                   handle,
                                           hipblasSideMode_t                 side,
                                           hipblasFillMode_t                 uplo,
                                           hipblasOperation_t                transA,
                                           hipblasDiagType_t                 diag,
                                           int64_t                           m,
                                           int64_t                           n,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const A[],
                                           int64_t                           lda,
                                           const hipblasDoubleComplex* const B[],
                                           int64_t                           ldb,
                                           hipblasDoubleComplex* const       C[],
                                           int64_t                           ldc,
                                           int64_t                           batchCount)
{
    return hipblasZtrmmBatched_64(handle,
                                  side,
                                  uplo,
                                  transA,
                                  diag,
                                  m,
                                  n,
                                  (const hipDoubleComplex*)alpha,
                                  (const hipDoubleComplex* const*)A,
                                  lda,
                                  (const hipDoubleComplex* const*)B,
                                  ldb,
                                  (hipDoubleComplex* const*)C,
                                  ldc,
                                  batchCount);
}

// trmm_strided_batched_64
hipblasStatus_t hipblasCtrmmStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  hipblasSideMode_t     side,
                                                  hipblasFillMode_t     uplo,
                                                  hipblasOperation_t    transA,
                                                  hipblasDiagType_t     diag,
                                                  int64_t               m,
                                                  int64_t               n,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* A,
                                                  int64_t               lda,
                                                  hipblasStride         strideA,
                                                  const hipblasComplex* B,
                                                  int64_t               ldb,
                                                  hipblasStride         strideB,
                                                  hipblasComplex*       C,
                                                  int64_t               ldc,
                                                  hipblasStride         strideC,
                                                  int64_t               batchCount)
{
    return hipblasCtrmmStridedBatched_64(handle,
                                         side,
                                         uplo,
                                         transA,
                                         diag,
                                         m,
                                         n,
                                         (const hipComplex*)alpha,
                                         (const hipComplex*)A,
                                         lda,
                                         strideA,
                                         (const hipComplex*)B,
                                         ldb,
                                         strideB,
                                         (hipComplex*)C,
                                         ldc,
                                         strideC,
                                         batchCount);
}

hipblasStatus_t hipblasZtrmmStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  hipblasSideMode_t           side,
                                                  hipblasFillMode_t           uplo,
                                                  hipblasOperation_t          transA,
                                                  hipblasDiagType_t           diag,
                                                  int64_t                     m,
                                                  int64_t                     n,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* A,
                                                  int64_t                     lda,
                                                  hipblasStride               strideA,
                                                  const hipblasDoubleComplex* B,
                                                  int64_t                     ldb,
                                                  hipblasStride               strideB,
                                                  hipblasDoubleComplex*       C,
                                                  int64_t                     ldc,
                                                  hipblasStride               strideC,
                                                  int64_t                     batchCount)
{
    return hipblasZtrmmStridedBatched_64(handle,
                                         side,
                                         uplo,
                                         transA,
                                         diag,
                                         m,
                                         n,
                                         (const hipDoubleComplex*)alpha,
                                         (const hipDoubleComplex*)A,
                                         lda,
                                         strideA,
                                         (const hipDoubleComplex*)B,
                                         ldb,
                                         strideB,
                                         (hipDoubleComplex*)C,
                                         ldc,
                                         strideC,
                                         batchCount);
}

// trsm
hipblasStatus_t hipblasCtrsmCast(hipblasHandle_t       handle,
                                 hipblasSideMode_t     side,
                                 hipblasFillMode_t     uplo,
                                 hipblasOperation_t    transA,
                                 hipblasDiagType_t     diag,
                                 int                   m,
                                 int                   n,
                                 const hipblasComplex* alpha,
                                 const hipblasComplex* A,
                                 int                   lda,
                                 hipblasComplex*       B,
                                 int                   ldb)
{
    return hipblasCtrsm(handle,
                        side,
                        uplo,
                        transA,
                        diag,
                        m,
                        n,
                        (const hipComplex*)alpha,
                        (const hipComplex*)A,
                        lda,
                        (hipComplex*)B,
                        ldb);
}

hipblasStatus_t hipblasZtrsmCast(hipblasHandle_t             handle,
                                 hipblasSideMode_t           side,
                                 hipblasFillMode_t           uplo,
                                 hipblasOperation_t          transA,
                                 hipblasDiagType_t           diag,
                                 int                         m,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* A,
                                 int                         lda,
                                 hipblasDoubleComplex*       B,
                                 int                         ldb)
{
    return hipblasZtrsm(handle,
                        side,
                        uplo,
                        transA,
                        diag,
                        m,
                        n,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)A,
                        lda,
                        (hipDoubleComplex*)B,
                        ldb);
}

// trsm_batched
hipblasStatus_t hipblasCtrsmBatchedCast(hipblasHandle_t             handle,
                                        hipblasSideMode_t           side,
                                        hipblasFillMode_t           uplo,
                                        hipblasOperation_t          transA,
                                        hipblasDiagType_t           diag,
                                        int                         m,
                                        int                         n,
                                        const hipblasComplex*       alpha,
                                        const hipblasComplex* const A[],
                                        int                         lda,
                                        hipblasComplex* const       B[],
                                        int                         ldb,
                                        int                         batch_count)
{
    return hipblasCtrsmBatched(handle,
                               side,
                               uplo,
                               transA,
                               diag,
                               m,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)A,
                               lda,
                               (hipComplex* const*)B,
                               ldb,
                               batch_count);
}

hipblasStatus_t hipblasZtrsmBatchedCast(hipblasHandle_t                   handle,
                                        hipblasSideMode_t                 side,
                                        hipblasFillMode_t                 uplo,
                                        hipblasOperation_t                transA,
                                        hipblasDiagType_t                 diag,
                                        int                               m,
                                        int                               n,
                                        const hipblasDoubleComplex*       alpha,
                                        const hipblasDoubleComplex* const A[],
                                        int                               lda,
                                        hipblasDoubleComplex* const       B[],
                                        int                               ldb,
                                        int                               batch_count)
{
    return hipblasZtrsmBatched(handle,
                               side,
                               uplo,
                               transA,
                               diag,
                               m,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (hipDoubleComplex* const*)B,
                               ldb,
                               batch_count);
}

// trsm_strided_batched
hipblasStatus_t hipblasCtrsmStridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasSideMode_t     side,
                                               hipblasFillMode_t     uplo,
                                               hipblasOperation_t    transA,
                                               hipblasDiagType_t     diag,
                                               int                   m,
                                               int                   n,
                                               const hipblasComplex* alpha,
                                               const hipblasComplex* A,
                                               int                   lda,
                                               hipblasStride         strideA,
                                               hipblasComplex*       B,
                                               int                   ldb,
                                               hipblasStride         strideB,
                                               int                   batch_count)
{
    return hipblasCtrsmStridedBatched(handle,
                                      side,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)A,
                                      lda,
                                      strideA,
                                      (hipComplex*)B,
                                      ldb,
                                      strideB,
                                      batch_count);
}

hipblasStatus_t hipblasZtrsmStridedBatchedCast(hipblasHandle_t             handle,
                                               hipblasSideMode_t           side,
                                               hipblasFillMode_t           uplo,
                                               hipblasOperation_t          transA,
                                               hipblasDiagType_t           diag,
                                               int                         m,
                                               int                         n,
                                               const hipblasDoubleComplex* alpha,
                                               const hipblasDoubleComplex* A,
                                               int                         lda,
                                               hipblasStride               strideA,
                                               hipblasDoubleComplex*       B,
                                               int                         ldb,
                                               hipblasStride               strideB,
                                               int                         batch_count)
{
    return hipblasZtrsmStridedBatched(handle,
                                      side,
                                      uplo,
                                      transA,
                                      diag,
                                      m,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      (hipDoubleComplex*)B,
                                      ldb,
                                      strideB,
                                      batch_count);
}

// trsm_64
hipblasStatus_t hipblasCtrsmCast_64(hipblasHandle_t       handle,
                                    hipblasSideMode_t     side,
                                    hipblasFillMode_t     uplo,
                                    hipblasOperation_t    transA,
                                    hipblasDiagType_t     diag,
                                    int64_t               m,
                                    int64_t               n,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* A,
                                    int64_t               lda,
                                    hipblasComplex*       B,
                                    int64_t               ldb)
{
    return hipblasCtrsm_64(handle,
                           side,
                           uplo,
                           transA,
                           diag,
                           m,
                           n,
                           (const hipComplex*)alpha,
                           (const hipComplex*)A,
                           lda,
                           (hipComplex*)B,
                           ldb);
}

hipblasStatus_t hipblasZtrsmCast_64(hipblasHandle_t             handle,
                                    hipblasSideMode_t           side,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int64_t                     m,
                                    int64_t                     n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* A,
                                    int64_t                     lda,
                                    hipblasDoubleComplex*       B,
                                    int64_t                     ldb)
{
    return hipblasZtrsm_64(handle,
                           side,
                           uplo,
                           transA,
                           diag,
                           m,
                           n,
                           (const hipDoubleComplex*)alpha,
                           (const hipDoubleComplex*)A,
                           lda,
                           (hipDoubleComplex*)B,
                           ldb);
}

// trsm_batched_64
hipblasStatus_t hipblasCtrsmBatchedCast_64(hipblasHandle_t             handle,
                                           hipblasSideMode_t           side,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int64_t                     m,
                                           int64_t                     n,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const A[],
                                           int64_t                     lda,
                                           hipblasComplex* const       B[],
                                           int64_t                     ldb,
                                           int64_t                     batch_count)
{
    return hipblasCtrsmBatched_64(handle,
                                  side,
                                  uplo,
                                  transA,
                                  diag,
                                  m,
                                  n,
                                  (const hipComplex*)alpha,
                                  (const hipComplex* const*)A,
                                  lda,
                                  (hipComplex* const*)B,
                                  ldb,
                                  batch_count);
}

hipblasStatus_t hipblasZtrsmBatchedCast_64(hipblasHandle_t                   handle,
                                           hipblasSideMode_t                 side,
                                           hipblasFillMode_t                 uplo,
                                           hipblasOperation_t                transA,
                                           hipblasDiagType_t                 diag,
                                           int64_t                           m,
                                           int64_t                           n,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const A[],
                                           int64_t                           lda,
                                           hipblasDoubleComplex* const       B[],
                                           int64_t                           ldb,
                                           int64_t                           batch_count)
{
    return hipblasZtrsmBatched_64(handle,
                                  side,
                                  uplo,
                                  transA,
                                  diag,
                                  m,
                                  n,
                                  (const hipDoubleComplex*)alpha,
                                  (const hipDoubleComplex* const*)A,
                                  lda,
                                  (hipDoubleComplex* const*)B,
                                  ldb,
                                  batch_count);
}

// trsm_strided_batched_64
hipblasStatus_t hipblasCtrsmStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  hipblasSideMode_t     side,
                                                  hipblasFillMode_t     uplo,
                                                  hipblasOperation_t    transA,
                                                  hipblasDiagType_t     diag,
                                                  int64_t               m,
                                                  int64_t               n,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* A,
                                                  int64_t               lda,
                                                  hipblasStride         strideA,
                                                  hipblasComplex*       B,
                                                  int64_t               ldb,
                                                  hipblasStride         strideB,
                                                  int64_t               batch_count)
{
    return hipblasCtrsmStridedBatched_64(handle,
                                         side,
                                         uplo,
                                         transA,
                                         diag,
                                         m,
                                         n,
                                         (const hipComplex*)alpha,
                                         (const hipComplex*)A,
                                         lda,
                                         strideA,
                                         (hipComplex*)B,
                                         ldb,
                                         strideB,
                                         batch_count);
}

hipblasStatus_t hipblasZtrsmStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  hipblasSideMode_t           side,
                                                  hipblasFillMode_t           uplo,
                                                  hipblasOperation_t          transA,
                                                  hipblasDiagType_t           diag,
                                                  int64_t                     m,
                                                  int64_t                     n,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* A,
                                                  int64_t                     lda,
                                                  hipblasStride               strideA,
                                                  hipblasDoubleComplex*       B,
                                                  int64_t                     ldb,
                                                  hipblasStride               strideB,
                                                  int64_t                     batch_count)
{
    return hipblasZtrsmStridedBatched_64(handle,
                                         side,
                                         uplo,
                                         transA,
                                         diag,
                                         m,
                                         n,
                                         (const hipDoubleComplex*)alpha,
                                         (const hipDoubleComplex*)A,
                                         lda,
                                         strideA,
                                         (hipDoubleComplex*)B,
                                         ldb,
                                         strideB,
                                         batch_count);
}

// geam
hipblasStatus_t hipblasCgeamCast(hipblasHandle_t       handle,
                                 hipblasOperation_t    transA,
                                 hipblasOperation_t    transB,
                                 int                   m,
                                 int                   n,
                                 const hipblasComplex* alpha,
                                 const hipblasComplex* A,
                                 int                   lda,
                                 const hipblasComplex* beta,
                                 const hipblasComplex* B,
                                 int                   ldb,
                                 hipblasComplex*       C,
                                 int                   ldc)
{
    return hipblasCgeam(handle,
                        transA,
                        transB,
                        m,
                        n,
                        (const hipComplex*)alpha,
                        (const hipComplex*)A,
                        lda,
                        (const hipComplex*)beta,
                        (const hipComplex*)B,
                        ldb,
                        (hipComplex*)C,
                        ldc);
}

hipblasStatus_t hipblasZgeamCast(hipblasHandle_t             handle,
                                 hipblasOperation_t          transA,
                                 hipblasOperation_t          transB,
                                 int                         m,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* A,
                                 int                         lda,
                                 const hipblasDoubleComplex* beta,
                                 const hipblasDoubleComplex* B,
                                 int                         ldb,
                                 hipblasDoubleComplex*       C,
                                 int                         ldc)
{
    return hipblasZgeam(handle,
                        transA,
                        transB,
                        m,
                        n,
                        (const hipDoubleComplex*)alpha,
                        (const hipDoubleComplex*)A,
                        lda,
                        (const hipDoubleComplex*)beta,
                        (const hipDoubleComplex*)B,
                        ldb,
                        (hipDoubleComplex*)C,
                        ldc);
}

// geam_batched
hipblasStatus_t hipblasCgeamBatchedCast(hipblasHandle_t             handle,
                                        hipblasOperation_t          transA,
                                        hipblasOperation_t          transB,
                                        int                         m,
                                        int                         n,
                                        const hipblasComplex*       alpha,
                                        const hipblasComplex* const A[],
                                        int                         lda,
                                        const hipblasComplex*       beta,
                                        const hipblasComplex* const B[],
                                        int                         ldb,
                                        hipblasComplex* const       C[],
                                        int                         ldc,
                                        int                         batchCount)
{
    return hipblasCgeamBatched(handle,
                               transA,
                               transB,
                               m,
                               n,
                               (const hipComplex*)alpha,
                               (const hipComplex* const*)A,
                               lda,
                               (const hipComplex*)beta,
                               (const hipComplex* const*)B,
                               ldb,
                               (hipComplex* const*)C,
                               ldc,
                               batchCount);
}

hipblasStatus_t hipblasZgeamBatchedCast(hipblasHandle_t                   handle,
                                        hipblasOperation_t                transA,
                                        hipblasOperation_t                transB,
                                        int                               m,
                                        int                               n,
                                        const hipblasDoubleComplex*       alpha,
                                        const hipblasDoubleComplex* const A[],
                                        int                               lda,
                                        const hipblasDoubleComplex*       beta,
                                        const hipblasDoubleComplex* const B[],
                                        int                               ldb,
                                        hipblasDoubleComplex* const       C[],
                                        int                               ldc,
                                        int                               batchCount)
{
    return hipblasZgeamBatched(handle,
                               transA,
                               transB,
                               m,
                               n,
                               (const hipDoubleComplex*)alpha,
                               (const hipDoubleComplex* const*)A,
                               lda,
                               (const hipDoubleComplex*)beta,
                               (const hipDoubleComplex* const*)B,
                               ldb,
                               (hipDoubleComplex* const*)C,
                               ldc,
                               batchCount);
}

// geam_strided_batched
hipblasStatus_t hipblasCgeamStridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasOperation_t    transA,
                                               hipblasOperation_t    transB,
                                               int                   m,
                                               int                   n,
                                               const hipblasComplex* alpha,
                                               const hipblasComplex* A,
                                               int                   lda,
                                               hipblasStride         strideA,
                                               const hipblasComplex* beta,
                                               const hipblasComplex* B,
                                               int                   ldb,
                                               hipblasStride         strideB,
                                               hipblasComplex*       C,
                                               int                   ldc,
                                               hipblasStride         strideC,
                                               int                   batchCount)
{
    return hipblasCgeamStridedBatched(handle,
                                      transA,
                                      transB,
                                      m,
                                      n,
                                      (const hipComplex*)alpha,
                                      (const hipComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipComplex*)beta,
                                      (const hipComplex*)B,
                                      ldb,
                                      strideB,
                                      (hipComplex*)C,
                                      ldc,
                                      strideC,
                                      batchCount);
}

hipblasStatus_t hipblasZgeamStridedBatchedCast(hipblasHandle_t             handle,
                                               hipblasOperation_t          transA,
                                               hipblasOperation_t          transB,
                                               int                         m,
                                               int                         n,
                                               const hipblasDoubleComplex* alpha,
                                               const hipblasDoubleComplex* A,
                                               int                         lda,
                                               hipblasStride               strideA,
                                               const hipblasDoubleComplex* beta,
                                               const hipblasDoubleComplex* B,
                                               int                         ldb,
                                               hipblasStride               strideB,
                                               hipblasDoubleComplex*       C,
                                               int                         ldc,
                                               hipblasStride               strideC,
                                               int                         batchCount)
{
    return hipblasZgeamStridedBatched(handle,
                                      transA,
                                      transB,
                                      m,
                                      n,
                                      (const hipDoubleComplex*)alpha,
                                      (const hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      (const hipDoubleComplex*)beta,
                                      (const hipDoubleComplex*)B,
                                      ldb,
                                      strideB,
                                      (hipDoubleComplex*)C,
                                      ldc,
                                      strideC,
                                      batchCount);
}

// geam_64
hipblasStatus_t hipblasCgeamCast_64(hipblasHandle_t       handle,
                                    hipblasOperation_t    transA,
                                    hipblasOperation_t    transB,
                                    int64_t               m,
                                    int64_t               n,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* A,
                                    int64_t               lda,
                                    const hipblasComplex* beta,
                                    const hipblasComplex* B,
                                    int64_t               ldb,
                                    hipblasComplex*       C,
                                    int64_t               ldc)
{
    return hipblasCgeam_64(handle,
                           transA,
                           transB,
                           m,
                           n,
                           (const hipComplex*)alpha,
                           (const hipComplex*)A,
                           lda,
                           (const hipComplex*)beta,
                           (const hipComplex*)B,
                           ldb,
                           (hipComplex*)C,
                           ldc);
}

hipblasStatus_t hipblasZgeamCast_64(hipblasHandle_t             handle,
                                    hipblasOperation_t          transA,
                                    hipblasOperation_t          transB,
                                    int64_t                     m,
                                    int64_t                     n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* A,
                                    int64_t                     lda,
                                    const hipblasDoubleComplex* beta,
                                    const hipblasDoubleComplex* B,
                                    int64_t                     ldb,
                                    hipblasDoubleComplex*       C,
                                    int64_t                     ldc)
{
    return hipblasZgeam_64(handle,
                           transA,
                           transB,
                           m,
                           n,
                           (const hipDoubleComplex*)alpha,
                           (const hipDoubleComplex*)A,
                           lda,
                           (const hipDoubleComplex*)beta,
                           (const hipDoubleComplex*)B,
                           ldb,
                           (hipDoubleComplex*)C,
                           ldc);
}

// geam_batched_64
hipblasStatus_t hipblasCgeamBatchedCast_64(hipblasHandle_t             handle,
                                           hipblasOperation_t          transA,
                                           hipblasOperation_t          transB,
                                           int64_t                     m,
                                           int64_t                     n,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const A[],
                                           int64_t                     lda,
                                           const hipblasComplex*       beta,
                                           const hipblasComplex* const B[],
                                           int64_t                     ldb,
                                           hipblasComplex* const       C[],
                                           int64_t                     ldc,
                                           int64_t                     batchCount)
{
    return hipblasCgeamBatched_64(handle,
                                  transA,
                                  transB,
                                  m,
                                  n,
                                  (const hipComplex*)alpha,
                                  (const hipComplex* const*)A,
                                  lda,
                                  (const hipComplex*)beta,
                                  (const hipComplex* const*)B,
                                  ldb,
                                  (hipComplex* const*)C,
                                  ldc,
                                  batchCount);
}

hipblasStatus_t hipblasZgeamBatchedCast_64(hipblasHandle_t                   handle,
                                           hipblasOperation_t                transA,
                                           hipblasOperation_t                transB,
                                           int64_t                           m,
                                           int64_t                           n,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const A[],
                                           int64_t                           lda,
                                           const hipblasDoubleComplex*       beta,
                                           const hipblasDoubleComplex* const B[],
                                           int64_t                           ldb,
                                           hipblasDoubleComplex* const       C[],
                                           int64_t                           ldc,
                                           int64_t                           batchCount)
{
    return hipblasZgeamBatched_64(handle,
                                  transA,
                                  transB,
                                  m,
                                  n,
                                  (const hipDoubleComplex*)alpha,
                                  (const hipDoubleComplex* const*)A,
                                  lda,
                                  (const hipDoubleComplex*)beta,
                                  (const hipDoubleComplex* const*)B,
                                  ldb,
                                  (hipDoubleComplex* const*)C,
                                  ldc,
                                  batchCount);
}

// geam_strided_batched_64
hipblasStatus_t hipblasCgeamStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  hipblasOperation_t    transA,
                                                  hipblasOperation_t    transB,
                                                  int64_t               m,
                                                  int64_t               n,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* A,
                                                  int64_t               lda,
                                                  hipblasStride         strideA,
                                                  const hipblasComplex* beta,
                                                  const hipblasComplex* B,
                                                  int64_t               ldb,
                                                  hipblasStride         strideB,
                                                  hipblasComplex*       C,
                                                  int64_t               ldc,
                                                  hipblasStride         strideC,
                                                  int64_t               batchCount)
{
    return hipblasCgeamStridedBatched_64(handle,
                                         transA,
                                         transB,
                                         m,
                                         n,
                                         (const hipComplex*)alpha,
                                         (const hipComplex*)A,
                                         lda,
                                         strideA,
                                         (const hipComplex*)beta,
                                         (const hipComplex*)B,
                                         ldb,
                                         strideB,
                                         (hipComplex*)C,
                                         ldc,
                                         strideC,
                                         batchCount);
}

hipblasStatus_t hipblasZgeamStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  hipblasOperation_t          transA,
                                                  hipblasOperation_t          transB,
                                                  int64_t                     m,
                                                  int64_t                     n,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* A,
                                                  int64_t                     lda,
                                                  hipblasStride               strideA,
                                                  const hipblasDoubleComplex* beta,
                                                  const hipblasDoubleComplex* B,
                                                  int64_t                     ldb,
                                                  hipblasStride               strideB,
                                                  hipblasDoubleComplex*       C,
                                                  int64_t                     ldc,
                                                  hipblasStride               strideC,
                                                  int64_t                     batchCount)
{
    return hipblasZgeamStridedBatched_64(handle,
                                         transA,
                                         transB,
                                         m,
                                         n,
                                         (const hipDoubleComplex*)alpha,
                                         (const hipDoubleComplex*)A,
                                         lda,
                                         strideA,
                                         (const hipDoubleComplex*)beta,
                                         (const hipDoubleComplex*)B,
                                         ldb,
                                         strideB,
                                         (hipDoubleComplex*)C,
                                         ldc,
                                         strideC,
                                         batchCount);
}

#ifdef __HIP_PLATFORM_SOLVER__

// getrf
hipblasStatus_t hipblasCgetrfCast(
    hipblasHandle_t handle, const int n, hipblasComplex* A, const int lda, int* ipiv, int* info)
{
    return hipblasCgetrf(handle, n, (hipComplex*)A, lda, ipiv, info);
}

hipblasStatus_t hipblasZgetrfCast(hipblasHandle_t       handle,
                                  const int             n,
                                  hipblasDoubleComplex* A,
                                  const int             lda,
                                  int*                  ipiv,
                                  int*                  info)
{
    return hipblasZgetrf(handle, n, (hipDoubleComplex*)A, lda, ipiv, info);
}

// getrf_batched
hipblasStatus_t hipblasCgetrfBatchedCast(hipblasHandle_t       handle,
                                         const int             n,
                                         hipblasComplex* const A[],
                                         const int             lda,
                                         int*                  ipiv,
                                         int*                  info,
                                         const int             batchCount)
{
    return hipblasCgetrfBatched(handle, n, (hipComplex* const*)A, lda, ipiv, info, batchCount);
}

hipblasStatus_t hipblasZgetrfBatchedCast(hipblasHandle_t             handle,
                                         const int                   n,
                                         hipblasDoubleComplex* const A[],
                                         const int                   lda,
                                         int*                        ipiv,
                                         int*                        info,
                                         const int                   batchCount)
{
    return hipblasZgetrfBatched(
        handle, n, (hipDoubleComplex* const*)A, lda, ipiv, info, batchCount);
}

// getrf_strided_batched
hipblasStatus_t hipblasCgetrfStridedBatchedCast(hipblasHandle_t     handle,
                                                const int           n,
                                                hipblasComplex*     A,
                                                const int           lda,
                                                const hipblasStride strideA,
                                                int*                ipiv,
                                                const hipblasStride strideP,
                                                int*                info,
                                                const int           batchCount)
{
    return hipblasCgetrfStridedBatched(
        handle, n, (hipComplex*)A, lda, strideA, ipiv, strideP, info, batchCount);
}

hipblasStatus_t hipblasZgetrfStridedBatchedCast(hipblasHandle_t       handle,
                                                const int             n,
                                                hipblasDoubleComplex* A,
                                                const int             lda,
                                                const hipblasStride   strideA,
                                                int*                  ipiv,
                                                const hipblasStride   strideP,
                                                int*                  info,
                                                const int             batchCount)
{
    return hipblasZgetrfStridedBatched(
        handle, n, (hipDoubleComplex*)A, lda, strideA, ipiv, strideP, info, batchCount);
}

// getrs
hipblasStatus_t hipblasCgetrsCast(hipblasHandle_t          handle,
                                  const hipblasOperation_t trans,
                                  const int                n,
                                  const int                nrhs,
                                  hipblasComplex*          A,
                                  const int                lda,
                                  const int*               ipiv,
                                  hipblasComplex*          B,
                                  const int                ldb,
                                  int*                     info)
{
    return hipblasCgetrs(
        handle, trans, n, nrhs, (hipComplex*)A, lda, ipiv, (hipComplex*)B, ldb, info);
}

hipblasStatus_t hipblasZgetrsCast(hipblasHandle_t          handle,
                                  const hipblasOperation_t trans,
                                  const int                n,
                                  const int                nrhs,
                                  hipblasDoubleComplex*    A,
                                  const int                lda,
                                  const int*               ipiv,
                                  hipblasDoubleComplex*    B,
                                  const int                ldb,
                                  int*                     info)
{
    return hipblasZgetrs(
        handle, trans, n, nrhs, (hipDoubleComplex*)A, lda, ipiv, (hipDoubleComplex*)B, ldb, info);
}

// getrs_batched
hipblasStatus_t hipblasCgetrsBatchedCast(hipblasHandle_t          handle,
                                         const hipblasOperation_t trans,
                                         const int                n,
                                         const int                nrhs,
                                         hipblasComplex* const    A[],
                                         const int                lda,
                                         const int*               ipiv,
                                         hipblasComplex* const    B[],
                                         const int                ldb,
                                         int*                     info,
                                         const int                batchCount)
{
    return hipblasCgetrsBatched(handle,
                                trans,
                                n,
                                nrhs,
                                (hipComplex* const*)A,
                                lda,
                                ipiv,
                                (hipComplex* const*)B,
                                ldb,
                                info,
                                batchCount);
}

hipblasStatus_t hipblasZgetrsBatchedCast(hipblasHandle_t             handle,
                                         const hipblasOperation_t    trans,
                                         const int                   n,
                                         const int                   nrhs,
                                         hipblasDoubleComplex* const A[],
                                         const int                   lda,
                                         const int*                  ipiv,
                                         hipblasDoubleComplex* const B[],
                                         const int                   ldb,
                                         int*                        info,
                                         const int                   batchCount)
{
    return hipblasZgetrsBatched(handle,
                                trans,
                                n,
                                nrhs,
                                (hipDoubleComplex* const*)A,
                                lda,
                                ipiv,
                                (hipDoubleComplex* const*)B,
                                ldb,
                                info,
                                batchCount);
}

// getrs_strided_batched
hipblasStatus_t hipblasCgetrsStridedBatchedCast(hipblasHandle_t          handle,
                                                const hipblasOperation_t trans,
                                                const int                n,
                                                const int                nrhs,
                                                hipblasComplex*          A,
                                                const int                lda,
                                                const hipblasStride      strideA,
                                                const int*               ipiv,
                                                const hipblasStride      strideP,
                                                hipblasComplex*          B,
                                                const int                ldb,
                                                const hipblasStride      strideB,
                                                int*                     info,
                                                const int                batchCount)
{
    return hipblasCgetrsStridedBatched(handle,
                                       trans,
                                       n,
                                       nrhs,
                                       (hipComplex*)A,
                                       lda,
                                       strideA,
                                       ipiv,
                                       strideP,
                                       (hipComplex*)B,
                                       ldb,
                                       strideB,
                                       info,
                                       batchCount);
}

hipblasStatus_t hipblasZgetrsStridedBatchedCast(hipblasHandle_t          handle,
                                                const hipblasOperation_t trans,
                                                const int                n,
                                                const int                nrhs,
                                                hipblasDoubleComplex*    A,
                                                const int                lda,
                                                const hipblasStride      strideA,
                                                const int*               ipiv,
                                                const hipblasStride      strideP,
                                                hipblasDoubleComplex*    B,
                                                const int                ldb,
                                                const hipblasStride      strideB,
                                                int*                     info,
                                                const int                batchCount)
{
    return hipblasZgetrsStridedBatched(handle,
                                       trans,
                                       n,
                                       nrhs,
                                       (hipDoubleComplex*)A,
                                       lda,
                                       strideA,
                                       ipiv,
                                       strideP,
                                       (hipDoubleComplex*)B,
                                       ldb,
                                       strideB,
                                       info,
                                       batchCount);
}

// getri_batched
hipblasStatus_t hipblasCgetriBatchedCast(hipblasHandle_t       handle,
                                         const int             n,
                                         hipblasComplex* const A[],
                                         const int             lda,
                                         int*                  ipiv,
                                         hipblasComplex* const C[],
                                         const int             ldc,
                                         int*                  info,
                                         const int             batchCount)
{
    return hipblasCgetriBatched(
        handle, n, (hipComplex* const*)A, lda, ipiv, (hipComplex* const*)C, ldc, info, batchCount);
}

hipblasStatus_t hipblasZgetriBatchedCast(hipblasHandle_t             handle,
                                         const int                   n,
                                         hipblasDoubleComplex* const A[],
                                         const int                   lda,
                                         int*                        ipiv,
                                         hipblasDoubleComplex* const C[],
                                         const int                   ldc,
                                         int*                        info,
                                         const int                   batchCount)
{
    return hipblasZgetriBatched(handle,
                                n,
                                (hipDoubleComplex* const*)A,
                                lda,
                                ipiv,
                                (hipDoubleComplex* const*)C,
                                ldc,
                                info,
                                batchCount);
}

// geqrf
hipblasStatus_t hipblasCgeqrfCast(hipblasHandle_t handle,
                                  const int       m,
                                  const int       n,
                                  hipblasComplex* A,
                                  const int       lda,
                                  hipblasComplex* ipiv,
                                  int*            info)
{
    return hipblasCgeqrf(handle, m, n, (hipComplex*)A, lda, (hipComplex*)ipiv, info);
}

hipblasStatus_t hipblasZgeqrfCast(hipblasHandle_t       handle,
                                  const int             m,
                                  const int             n,
                                  hipblasDoubleComplex* A,
                                  const int             lda,
                                  hipblasDoubleComplex* ipiv,
                                  int*                  info)
{
    return hipblasZgeqrf(handle, m, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)ipiv, info);
}

// geqrf_batched
hipblasStatus_t hipblasCgeqrfBatchedCast(hipblasHandle_t       handle,
                                         const int             m,
                                         const int             n,
                                         hipblasComplex* const A[],
                                         const int             lda,
                                         hipblasComplex* const ipiv[],
                                         int*                  info,
                                         const int             batchCount)
{
    return hipblasCgeqrfBatched(
        handle, m, n, (hipComplex* const*)A, lda, (hipComplex* const*)ipiv, info, batchCount);
}

hipblasStatus_t hipblasZgeqrfBatchedCast(hipblasHandle_t             handle,
                                         const int                   m,
                                         const int                   n,
                                         hipblasDoubleComplex* const A[],
                                         const int                   lda,
                                         hipblasDoubleComplex* const ipiv[],
                                         int*                        info,
                                         const int                   batchCount)
{
    return hipblasZgeqrfBatched(handle,
                                m,
                                n,
                                (hipDoubleComplex* const*)A,
                                lda,
                                (hipDoubleComplex* const*)ipiv,
                                info,
                                batchCount);
}

// geqrf_strided_batched
hipblasStatus_t hipblasCgeqrfStridedBatchedCast(hipblasHandle_t     handle,
                                                const int           m,
                                                const int           n,
                                                hipblasComplex*     A,
                                                const int           lda,
                                                const hipblasStride strideA,
                                                hipblasComplex*     ipiv,
                                                const hipblasStride strideP,
                                                int*                info,
                                                const int           batchCount)
{
    return hipblasCgeqrfStridedBatched(
        handle, m, n, (hipComplex*)A, lda, strideA, (hipComplex*)ipiv, strideP, info, batchCount);
}

hipblasStatus_t hipblasZgeqrfStridedBatchedCast(hipblasHandle_t       handle,
                                                const int             m,
                                                const int             n,
                                                hipblasDoubleComplex* A,
                                                const int             lda,
                                                const hipblasStride   strideA,
                                                hipblasDoubleComplex* ipiv,
                                                const hipblasStride   strideP,
                                                int*                  info,
                                                const int             batchCount)
{
    return hipblasZgeqrfStridedBatched(handle,
                                       m,
                                       n,
                                       (hipDoubleComplex*)A,
                                       lda,
                                       strideA,
                                       (hipDoubleComplex*)ipiv,
                                       strideP,
                                       info,
                                       batchCount);
}

// gels
hipblasStatus_t hipblasCgelsCast(hipblasHandle_t    handle,
                                 hipblasOperation_t trans,
                                 const int          m,
                                 const int          n,
                                 const int          nrhs,
                                 hipblasComplex*    A,
                                 const int          lda,
                                 hipblasComplex*    B,
                                 const int          ldb,
                                 int*               info,
                                 int*               deviceInfo)
{
    return hipblasCgels(
        handle, trans, m, n, nrhs, (hipComplex*)A, lda, (hipComplex*)B, ldb, info, deviceInfo);
}

hipblasStatus_t hipblasZgelsCast(hipblasHandle_t       handle,
                                 hipblasOperation_t    trans,
                                 const int             m,
                                 const int             n,
                                 const int             nrhs,
                                 hipblasDoubleComplex* A,
                                 const int             lda,
                                 hipblasDoubleComplex* B,
                                 const int             ldb,
                                 int*                  info,
                                 int*                  deviceInfo)
{
    return hipblasZgels(handle,
                        trans,
                        m,
                        n,
                        nrhs,
                        (hipDoubleComplex*)A,
                        lda,
                        (hipDoubleComplex*)B,
                        ldb,
                        info,
                        deviceInfo);
}

// gelsBatched
hipblasStatus_t hipblasCgelsBatchedCast(hipblasHandle_t       handle,
                                        hipblasOperation_t    trans,
                                        const int             m,
                                        const int             n,
                                        const int             nrhs,
                                        hipblasComplex* const A[],
                                        const int             lda,
                                        hipblasComplex* const B[],
                                        const int             ldb,
                                        int*                  info,
                                        int*                  deviceInfo,
                                        const int             batchCount)
{
    return hipblasCgelsBatched(handle,
                               trans,
                               m,
                               n,
                               nrhs,
                               (hipComplex* const*)A,
                               lda,
                               (hipComplex* const*)B,
                               ldb,
                               info,
                               deviceInfo,
                               batchCount);
}

hipblasStatus_t hipblasZgelsBatchedCast(hipblasHandle_t             handle,
                                        hipblasOperation_t          trans,
                                        const int                   m,
                                        const int                   n,
                                        const int                   nrhs,
                                        hipblasDoubleComplex* const A[],
                                        const int                   lda,
                                        hipblasDoubleComplex* const B[],
                                        const int                   ldb,
                                        int*                        info,
                                        int*                        deviceInfo,
                                        const int                   batchCount)
{
    return hipblasZgelsBatched(handle,
                               trans,
                               m,
                               n,
                               nrhs,
                               (hipDoubleComplex* const*)A,
                               lda,
                               (hipDoubleComplex* const*)B,
                               ldb,
                               info,
                               deviceInfo,
                               batchCount);
}

// gelsStridedBatched
hipblasStatus_t hipblasCgelsStridedBatchedCast(hipblasHandle_t     handle,
                                               hipblasOperation_t  trans,
                                               const int           m,
                                               const int           n,
                                               const int           nrhs,
                                               hipblasComplex*     A,
                                               const int           lda,
                                               const hipblasStride strideA,
                                               hipblasComplex*     B,
                                               const int           ldb,
                                               const hipblasStride strideB,
                                               int*                info,
                                               int*                deviceInfo,
                                               const int           batchCount)
{
    return hipblasCgelsStridedBatched(handle,
                                      trans,
                                      m,
                                      n,
                                      nrhs,
                                      (hipComplex*)A,
                                      lda,
                                      strideA,
                                      (hipComplex*)B,
                                      ldb,
                                      strideB,
                                      info,
                                      deviceInfo,
                                      batchCount);
}

hipblasStatus_t hipblasZgelsStridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasOperation_t    trans,
                                               const int             m,
                                               const int             n,
                                               const int             nrhs,
                                               hipblasDoubleComplex* A,
                                               const int             lda,
                                               const hipblasStride   strideA,
                                               hipblasDoubleComplex* B,
                                               const int             ldb,
                                               const hipblasStride   strideB,
                                               int*                  info,
                                               int*                  deviceInfo,
                                               const int             batchCount)
{
    return hipblasZgelsStridedBatched(handle,
                                      trans,
                                      m,
                                      n,
                                      nrhs,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      strideA,
                                      (hipDoubleComplex*)B,
                                      ldb,
                                      strideB,
                                      info,
                                      deviceInfo,
                                      batchCount);
}

#endif // solver
#endif // HIPBLAS_V2
