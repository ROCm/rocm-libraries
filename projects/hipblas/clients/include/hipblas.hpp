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
 * ************************************************************************ */

#pragma once
#ifndef _HIPBLAS_HPP_
#define _HIPBLAS_HPP_

/* library headers */
#include "hipblas.h"

#ifndef WIN32
#include "hipblas_fortran.hpp"
#else
#include "hipblas_no_fortran.hpp"
#endif

#if not defined(__clang_major__)
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

#define GET_MACRO(_1, _2, _3, _4, _5, NAME, ...) NAME

#define MAP2CF(...) GET_MACRO(__VA_ARGS__, MAP2CF5, MAP2CF4, MAP2CF3, dum2, dum1)(__VA_ARGS__)
// dual API C and FORTRAN
#define MAP2CF_D64(...) \
    GET_MACRO(__VA_ARGS__, MAP2DCF5, MAP2DCF4, MAP2DCF3, dum2, dum1)(__VA_ARGS__)

#if !defined(HIPBLAS_V2) \
    && !defined(         \
        WIN32) // HIPBLAS_V2 doesn't have fortran tests during transition period, WIN doesn't have fortran tests
#define MAP2CF3(FN, A, PFN)  \
    template <>              \
    auto FN<A, false> = PFN; \
    template <>              \
    auto FN<A, true> = PFN##Fortran
#define MAP2CF4(FN, A, B, PFN)  \
    template <>                 \
    auto FN<A, B, false> = PFN; \
    template <>                 \
    auto FN<A, B, true> = PFN##Fortran
#define MAP2CF5(FN, A, B, C, PFN)  \
    template <>                    \
    auto FN<A, B, C, false> = PFN; \
    template <>                    \
    auto FN<A, B, C, true> = PFN##Fortran
// dual API C and FORTRAN
#define MAP2DCF3(FN, A, PFN)           \
    template <>                        \
    auto FN<A, false> = PFN;           \
    template <>                        \
    auto FN<A, true> = PFN##Fortran;   \
    template <>                        \
    auto FN##_64<A, false> = PFN##_64; \
    template <>                        \
    auto FN##_64<A, true> = PFN##_64Fortran
#define MAP2DCF4(FN, A, B, PFN)           \
    template <>                           \
    auto FN<A, B, false> = PFN;           \
    template <>                           \
    auto FN<A, B, true> = PFN##Fortran;   \
    template <>                           \
    auto FN##_64<A, B, false> = PFN##_64; \
    template <>                           \
    auto FN##_64<A, B, true> = PFN##_64Fortran
#define MAP2DCF5(FN, A, B, C, PFN)           \
    template <>                              \
    auto FN<A, B, C, false> = PFN;           \
    template <>                              \
    auto FN<A, B, C, true> = PFN##Fortran;   \
    template <>                              \
    auto FN##_64<A, B, C, false> = PFN##_64; \
    template <>                              \
    auto FN##_64<A, B, C, true> = PFN##_64Fortran
#else
// mapping fortran and C to C API
#define MAP2CF3(FN, A, PFN)  \
    template <>              \
    auto FN<A, false> = PFN; \
    template <>              \
    auto FN<A, true> = PFN
#define MAP2CF4(FN, A, B, PFN)  \
    template <>                 \
    auto FN<A, B, false> = PFN; \
    template <>                 \
    auto FN<A, B, true> = PFN
#define MAP2CF5(FN, A, B, C, PFN)  \
    template <>                    \
    auto FN<A, B, C, false> = PFN; \
    template <>                    \
    auto FN<A, B, C, true> = PFN
// dual API C and FORTRAN
#define MAP2DCF3(FN, A, PFN)           \
    template <>                        \
    auto FN<A, false> = PFN;           \
    template <>                        \
    auto FN<A, true> = PFN;            \
    template <>                        \
    auto FN##_64<A, false> = PFN##_64; \
    template <>                        \
    auto FN##_64<A, true> = PFN##_64
#define MAP2DCF4(FN, A, B, PFN)           \
    template <>                           \
    auto FN<A, B, false> = PFN;           \
    template <>                           \
    auto FN<A, B, true> = PFN;            \
    template <>                           \
    auto FN##_64<A, B, false> = PFN##_64; \
    template <>                           \
    auto FN##_64<A, B, true> = PFN##_64
#define MAP2DCF5(FN, A, B, C, PFN)           \
    template <>                              \
    auto FN<A, B, C, false> = PFN;           \
    template <>                              \
    auto FN<A, B, C, true> = PFN;            \
    template <>                              \
    auto FN##_64<A, B, C, false> = PFN##_64; \
    template <>                              \
    auto FN##_64<A, B, C, true> = PFN##_64
#endif

#ifndef HIPBLAS_V2
#define MAP2CF_D64_V2(...) MAP2CF_D64(__VA_ARGS__)
#define MAP2CF_V2(...) MAP2CF(__VA_ARGS__)
#else
#define MAP2CF_D64_V2(...) MAP2CF_D64(__VA_ARGS__##Cast)
#define MAP2CF_V2(...) MAP2CF(__VA_ARGS__##Cast)
#endif

// Need these temporarily during transition period between hipblasComplex -> hipComplex
#ifdef HIPBLAS_V2

// scal
hipblasStatus_t hipblasCscalCast(
    hipblasHandle_t handle, int n, const hipblasComplex* alpha, hipblasComplex* x, int incx);
hipblasStatus_t hipblasCsscalCast(
    hipblasHandle_t handle, int n, const float* alpha, hipblasComplex* x, int incx);
hipblasStatus_t hipblasZscalCast(hipblasHandle_t             handle,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 hipblasDoubleComplex*       x,
                                 int                         incx);
hipblasStatus_t hipblasZdscalCast(
    hipblasHandle_t handle, int n, const double* alpha, hipblasDoubleComplex* x, int incx);
hipblasStatus_t hipblasCscalBatchedCast(hipblasHandle_t       handle,
                                        int                   n,
                                        const hipblasComplex* alpha,
                                        hipblasComplex* const x[],
                                        int                   incx,
                                        int                   batch_count);
hipblasStatus_t hipblasCsscalBatchedCast(hipblasHandle_t       handle,
                                         int                   n,
                                         const float*          alpha,
                                         hipblasComplex* const x[],
                                         int                   incx,
                                         int                   batch_count);
hipblasStatus_t hipblasZscalBatchedCast(hipblasHandle_t             handle,
                                        int                         n,
                                        const hipblasDoubleComplex* alpha,
                                        hipblasDoubleComplex* const x[],
                                        int                         incx,
                                        int                         batch_count);
hipblasStatus_t hipblasZdscalBatchedCast(hipblasHandle_t             handle,
                                         int                         n,
                                         const double*               alpha,
                                         hipblasDoubleComplex* const x[],
                                         int                         incx,
                                         int                         batch_count);
hipblasStatus_t hipblasCscalStridedBatchedCast(hipblasHandle_t       handle,
                                               int                   n,
                                               const hipblasComplex* alpha,
                                               hipblasComplex*       x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               int                   batch_count);
hipblasStatus_t hipblasCsscalStridedBatchedCast(hipblasHandle_t handle,
                                                int             n,
                                                const float*    alpha,
                                                hipblasComplex* x,
                                                int             incx,
                                                hipblasStride   stridex,
                                                int             batch_count);
hipblasStatus_t hipblasZscalStridedBatchedCast(hipblasHandle_t             handle,
                                               int                         n,
                                               const hipblasDoubleComplex* alpha,
                                               hipblasDoubleComplex*       x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               int                         batch_count);
hipblasStatus_t hipblasZdscalStridedBatchedCast(hipblasHandle_t       handle,
                                                int                   n,
                                                const double*         alpha,
                                                hipblasDoubleComplex* x,
                                                int                   incx,
                                                hipblasStride         stridex,
                                                int                   batch_count);

// scal_64
hipblasStatus_t hipblasCscalCast_64(hipblasHandle_t       handle,
                                    int64_t               n,
                                    const hipblasComplex* alpha,
                                    hipblasComplex*       x,
                                    int64_t               incx);
hipblasStatus_t hipblasCsscalCast_64(
    hipblasHandle_t handle, int64_t n, const float* alpha, hipblasComplex* x, int64_t incx);
hipblasStatus_t hipblasZscalCast_64(hipblasHandle_t             handle,
                                    int64_t                     n,
                                    const hipblasDoubleComplex* alpha,
                                    hipblasDoubleComplex*       x,
                                    int64_t                     incx);
hipblasStatus_t hipblasZdscalCast_64(
    hipblasHandle_t handle, int64_t n, const double* alpha, hipblasDoubleComplex* x, int64_t incx);
hipblasStatus_t hipblasCscalBatchedCast_64(hipblasHandle_t       handle,
                                           int64_t               n,
                                           const hipblasComplex* alpha,
                                           hipblasComplex* const x[],
                                           int64_t               incx,
                                           int64_t               batch_count);
hipblasStatus_t hipblasCsscalBatchedCast_64(hipblasHandle_t       handle,
                                            int64_t               n,
                                            const float*          alpha,
                                            hipblasComplex* const x[],
                                            int64_t               incx,
                                            int64_t               batch_count);
hipblasStatus_t hipblasZscalBatchedCast_64(hipblasHandle_t             handle,
                                           int64_t                     n,
                                           const hipblasDoubleComplex* alpha,
                                           hipblasDoubleComplex* const x[],
                                           int64_t                     incx,
                                           int64_t                     batch_count);
hipblasStatus_t hipblasZdscalBatchedCast_64(hipblasHandle_t             handle,
                                            int64_t                     n,
                                            const double*               alpha,
                                            hipblasDoubleComplex* const x[],
                                            int64_t                     incx,
                                            int64_t                     batch_count);
hipblasStatus_t hipblasCscalStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  int64_t               n,
                                                  const hipblasComplex* alpha,
                                                  hipblasComplex*       x,
                                                  int64_t               incx,
                                                  hipblasStride         stridex,
                                                  int64_t               batch_count);
hipblasStatus_t hipblasCsscalStridedBatchedCast_64(hipblasHandle_t handle,
                                                   int64_t         n,
                                                   const float*    alpha,
                                                   hipblasComplex* x,
                                                   int64_t         incx,
                                                   hipblasStride   stridex,
                                                   int64_t         batch_count);
hipblasStatus_t hipblasZscalStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  int64_t                     n,
                                                  const hipblasDoubleComplex* alpha,
                                                  hipblasDoubleComplex*       x,
                                                  int64_t                     incx,
                                                  hipblasStride               stridex,
                                                  int64_t                     batch_count);
hipblasStatus_t hipblasZdscalStridedBatchedCast_64(hipblasHandle_t       handle,
                                                   int64_t               n,
                                                   const double*         alpha,
                                                   hipblasDoubleComplex* x,
                                                   int64_t               incx,
                                                   hipblasStride         stridex,
                                                   int64_t               batch_count);

// copy
hipblasStatus_t hipblasCcopyCast(
    hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, hipblasComplex* y, int incy);
hipblasStatus_t hipblasZcopyCast(hipblasHandle_t             handle,
                                 int                         n,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 hipblasDoubleComplex*       y,
                                 int                         incy);
hipblasStatus_t hipblasCcopyBatchedCast(hipblasHandle_t             handle,
                                        int                         n,
                                        const hipblasComplex* const x[],
                                        int                         incx,
                                        hipblasComplex* const       y[],
                                        int                         incy,
                                        int                         batch_count);
hipblasStatus_t hipblasZcopyBatchedCast(hipblasHandle_t                   handle,
                                        int                               n,
                                        const hipblasDoubleComplex* const x[],
                                        int                               incx,
                                        hipblasDoubleComplex* const       y[],
                                        int                               incy,
                                        int                               batch_count);
hipblasStatus_t hipblasCcopyStridedBatchedCast(hipblasHandle_t       handle,
                                               int                   n,
                                               const hipblasComplex* x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               hipblasComplex*       y,
                                               int                   incy,
                                               hipblasStride         stridey,
                                               int                   batch_count);
hipblasStatus_t hipblasZcopyStridedBatchedCast(hipblasHandle_t             handle,
                                               int                         n,
                                               const hipblasDoubleComplex* x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               hipblasDoubleComplex*       y,
                                               int                         incy,
                                               hipblasStride               stridey,
                                               int                         batch_count);

// 64-bit interface
hipblasStatus_t hipblasCcopyCast_64(hipblasHandle_t       handle,
                                    int64_t               n,
                                    const hipblasComplex* x,
                                    int64_t               incx,
                                    hipblasComplex*       y,
                                    int64_t               incy);
hipblasStatus_t hipblasZcopyCast_64(hipblasHandle_t             handle,
                                    int64_t                     n,
                                    const hipblasDoubleComplex* x,
                                    int64_t                     incx,
                                    hipblasDoubleComplex*       y,
                                    int64_t                     incy);
hipblasStatus_t hipblasCcopyBatchedCast_64(hipblasHandle_t             handle,
                                           int64_t                     n,
                                           const hipblasComplex* const x[],
                                           int64_t                     incx,
                                           hipblasComplex* const       y[],
                                           int64_t                     incy,
                                           int64_t                     batch_count);
hipblasStatus_t hipblasZcopyBatchedCast_64(hipblasHandle_t                   handle,
                                           int64_t                           n,
                                           const hipblasDoubleComplex* const x[],
                                           int64_t                           incx,
                                           hipblasDoubleComplex* const       y[],
                                           int64_t                           incy,
                                           int64_t                           batch_count);
hipblasStatus_t hipblasCcopyStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  int64_t               n,
                                                  const hipblasComplex* x,
                                                  int64_t               incx,
                                                  hipblasStride         stridex,
                                                  hipblasComplex*       y,
                                                  int64_t               incy,
                                                  hipblasStride         stridey,
                                                  int64_t               batch_count);
hipblasStatus_t hipblasZcopyStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  int64_t                     n,
                                                  const hipblasDoubleComplex* x,
                                                  int64_t                     incx,
                                                  hipblasStride               stridex,
                                                  hipblasDoubleComplex*       y,
                                                  int64_t                     incy,
                                                  hipblasStride               stridey,
                                                  int64_t                     batch_count);

// axpy
hipblasStatus_t hipblasCaxpyCast(hipblasHandle_t       handle,
                                 int                   n,
                                 const hipblasComplex* alpha,
                                 const hipblasComplex* x,
                                 int                   incx,
                                 hipblasComplex*       y,
                                 int                   incy);

hipblasStatus_t hipblasZaxpyCast(hipblasHandle_t             handle,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 hipblasDoubleComplex*       y,
                                 int                         incy);

hipblasStatus_t hipblasCaxpyBatchedCast(hipblasHandle_t             handle,
                                        int                         n,
                                        const hipblasComplex*       alpha,
                                        const hipblasComplex* const x[],
                                        int                         incx,
                                        hipblasComplex* const       y[],
                                        int                         incy,
                                        int                         batch_count);

hipblasStatus_t hipblasZaxpyBatchedCast(hipblasHandle_t                   handle,
                                        int                               n,
                                        const hipblasDoubleComplex*       alpha,
                                        const hipblasDoubleComplex* const x[],
                                        int                               incx,
                                        hipblasDoubleComplex* const       y[],
                                        int                               incy,
                                        int                               batch_count);

hipblasStatus_t hipblasCaxpyStridedBatchedCast(hipblasHandle_t       handle,
                                               int                   n,
                                               const hipblasComplex* alpha,
                                               const hipblasComplex* x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               hipblasComplex*       y,
                                               int                   incy,
                                               hipblasStride         stridey,
                                               int                   batch_count);

hipblasStatus_t hipblasZaxpyStridedBatchedCast(hipblasHandle_t             handle,
                                               int                         n,
                                               const hipblasDoubleComplex* alpha,
                                               const hipblasDoubleComplex* x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               hipblasDoubleComplex*       y,
                                               int                         incy,
                                               hipblasStride               stridey,
                                               int                         batch_count);
// axpy_64
hipblasStatus_t hipblasCaxpyCast_64(hipblasHandle_t       handle,
                                    int64_t               n,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* x,
                                    int64_t               incx,
                                    hipblasComplex*       y,
                                    int64_t               incy);

hipblasStatus_t hipblasZaxpyCast_64(hipblasHandle_t             handle,
                                    int64_t                     n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* x,
                                    int64_t                     incx,
                                    hipblasDoubleComplex*       y,
                                    int64_t                     incy);

hipblasStatus_t hipblasCaxpyBatchedCast_64(hipblasHandle_t             handle,
                                           int64_t                     n,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const x[],
                                           int64_t                     incx,
                                           hipblasComplex* const       y[],
                                           int64_t                     incy,
                                           int64_t                     batch_count);

hipblasStatus_t hipblasZaxpyBatchedCast_64(hipblasHandle_t                   handle,
                                           int64_t                           n,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const x[],
                                           int64_t                           incx,
                                           hipblasDoubleComplex* const       y[],
                                           int64_t                           incy,
                                           int64_t                           batch_count);

hipblasStatus_t hipblasCaxpyStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  int64_t               n,
                                                  const hipblasComplex* alpha,
                                                  const hipblasComplex* x,
                                                  int64_t               incx,
                                                  hipblasStride         stridex,
                                                  hipblasComplex*       y,
                                                  int64_t               incy,
                                                  hipblasStride         stridey,
                                                  int64_t               batch_count);

hipblasStatus_t hipblasZaxpyStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  int64_t                     n,
                                                  const hipblasDoubleComplex* alpha,
                                                  const hipblasDoubleComplex* x,
                                                  int64_t                     incx,
                                                  hipblasStride               stridex,
                                                  hipblasDoubleComplex*       y,
                                                  int64_t                     incy,
                                                  hipblasStride               stridey,
                                                  int64_t                     batch_count);

// swap
hipblasStatus_t hipblasCswapCast(
    hipblasHandle_t handle, int n, hipblasComplex* x, int incx, hipblasComplex* y, int incy);

hipblasStatus_t hipblasZswapCast(hipblasHandle_t       handle,
                                 int                   n,
                                 hipblasDoubleComplex* x,
                                 int                   incx,
                                 hipblasDoubleComplex* y,
                                 int                   incy);

hipblasStatus_t hipblasCswapBatchedCast(hipblasHandle_t       handle,
                                        int                   n,
                                        hipblasComplex* const x[],
                                        int                   incx,
                                        hipblasComplex* const y[],
                                        int                   incy,
                                        int                   batch_count);

hipblasStatus_t hipblasZswapBatchedCast(hipblasHandle_t             handle,
                                        int                         n,
                                        hipblasDoubleComplex* const x[],
                                        int                         incx,
                                        hipblasDoubleComplex* const y[],
                                        int                         incy,
                                        int                         batch_count);

hipblasStatus_t hipblasCswapStridedBatchedCast(hipblasHandle_t handle,
                                               int             n,
                                               hipblasComplex* x,
                                               int             incx,
                                               hipblasStride   stridex,
                                               hipblasComplex* y,
                                               int             incy,
                                               hipblasStride   stridey,
                                               int             batch_count);

hipblasStatus_t hipblasZswapStridedBatchedCast(hipblasHandle_t       handle,
                                               int                   n,
                                               hipblasDoubleComplex* x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               hipblasDoubleComplex* y,
                                               int                   incy,
                                               hipblasStride         stridey,
                                               int                   batch_count);

// swap_64
hipblasStatus_t hipblasCswapCast_64(hipblasHandle_t handle,
                                    int64_t         n,
                                    hipblasComplex* x,
                                    int64_t         incx,
                                    hipblasComplex* y,
                                    int64_t         incy);

hipblasStatus_t hipblasZswapCast_64(hipblasHandle_t       handle,
                                    int64_t               n,
                                    hipblasDoubleComplex* x,
                                    int64_t               incx,
                                    hipblasDoubleComplex* y,
                                    int64_t               incy);

hipblasStatus_t hipblasCswapBatchedCast_64(hipblasHandle_t       handle,
                                           int64_t               n,
                                           hipblasComplex* const x[],
                                           int64_t               incx,
                                           hipblasComplex* const y[],
                                           int64_t               incy,
                                           int64_t               batch_count);

hipblasStatus_t hipblasZswapBatchedCast_64(hipblasHandle_t             handle,
                                           int64_t                     n,
                                           hipblasDoubleComplex* const x[],
                                           int64_t                     incx,
                                           hipblasDoubleComplex* const y[],
                                           int64_t                     incy,
                                           int64_t                     batch_count);

hipblasStatus_t hipblasCswapStridedBatchedCast_64(hipblasHandle_t handle,
                                                  int64_t         n,
                                                  hipblasComplex* x,
                                                  int64_t         incx,
                                                  hipblasStride   stridex,
                                                  hipblasComplex* y,
                                                  int64_t         incy,
                                                  hipblasStride   stridey,
                                                  int64_t         batch_count);

hipblasStatus_t hipblasZswapStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  int64_t               n,
                                                  hipblasDoubleComplex* x,
                                                  int64_t               incx,
                                                  hipblasStride         stridex,
                                                  hipblasDoubleComplex* y,
                                                  int64_t               incy,
                                                  hipblasStride         stridey,
                                                  int64_t               batch_count);

// dot
hipblasStatus_t hipblasCdotuCast(hipblasHandle_t       handle,
                                 int                   n,
                                 const hipblasComplex* x,
                                 int                   incx,
                                 const hipblasComplex* y,
                                 int                   incy,
                                 hipblasComplex*       result);

hipblasStatus_t hipblasCdotcCast(hipblasHandle_t       handle,
                                 int                   n,
                                 const hipblasComplex* x,
                                 int                   incx,
                                 const hipblasComplex* y,
                                 int                   incy,
                                 hipblasComplex*       result);

hipblasStatus_t hipblasZdotuCast(hipblasHandle_t             handle,
                                 int                         n,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* y,
                                 int                         incy,
                                 hipblasDoubleComplex*       result);

hipblasStatus_t hipblasZdotcCast(hipblasHandle_t             handle,
                                 int                         n,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* y,
                                 int                         incy,
                                 hipblasDoubleComplex*       result);

hipblasStatus_t hipblasCdotuBatchedCast(hipblasHandle_t             handle,
                                        int                         n,
                                        const hipblasComplex* const x[],
                                        int                         incx,
                                        const hipblasComplex* const y[],
                                        int                         incy,
                                        int                         batch_count,
                                        hipblasComplex*             result);

hipblasStatus_t hipblasCdotcBatchedCast(hipblasHandle_t             handle,
                                        int                         n,
                                        const hipblasComplex* const x[],
                                        int                         incx,
                                        const hipblasComplex* const y[],
                                        int                         incy,
                                        int                         batch_count,
                                        hipblasComplex*             result);

hipblasStatus_t hipblasZdotuBatchedCast(hipblasHandle_t                   handle,
                                        int                               n,
                                        const hipblasDoubleComplex* const x[],
                                        int                               incx,
                                        const hipblasDoubleComplex* const y[],
                                        int                               incy,
                                        int                               batch_count,
                                        hipblasDoubleComplex*             result);

hipblasStatus_t hipblasZdotcBatchedCast(hipblasHandle_t                   handle,
                                        int                               n,
                                        const hipblasDoubleComplex* const x[],
                                        int                               incx,
                                        const hipblasDoubleComplex* const y[],
                                        int                               incy,
                                        int                               batch_count,
                                        hipblasDoubleComplex*             result);

hipblasStatus_t hipblasCdotuStridedBatchedCast(hipblasHandle_t       handle,
                                               int                   n,
                                               const hipblasComplex* x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               const hipblasComplex* y,
                                               int                   incy,
                                               hipblasStride         stridey,
                                               int                   batch_count,
                                               hipblasComplex*       result);

hipblasStatus_t hipblasCdotcStridedBatchedCast(hipblasHandle_t       handle,
                                               int                   n,
                                               const hipblasComplex* x,
                                               int                   incx,
                                               hipblasStride         stridex,
                                               const hipblasComplex* y,
                                               int                   incy,
                                               hipblasStride         stridey,
                                               int                   batch_count,
                                               hipblasComplex*       result);

hipblasStatus_t hipblasZdotuStridedBatchedCast(hipblasHandle_t             handle,
                                               int                         n,
                                               const hipblasDoubleComplex* x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               const hipblasDoubleComplex* y,
                                               int                         incy,
                                               hipblasStride               stridey,
                                               int                         batch_count,
                                               hipblasDoubleComplex*       result);

hipblasStatus_t hipblasZdotcStridedBatchedCast(hipblasHandle_t             handle,
                                               int                         n,
                                               const hipblasDoubleComplex* x,
                                               int                         incx,
                                               hipblasStride               stridex,
                                               const hipblasDoubleComplex* y,
                                               int                         incy,
                                               hipblasStride               stridey,
                                               int                         batch_count,
                                               hipblasDoubleComplex*       result);

// dot_64
hipblasStatus_t hipblasCdotuCast_64(hipblasHandle_t       handle,
                                    int64_t               n,
                                    const hipblasComplex* x,
                                    int64_t               incx,
                                    const hipblasComplex* y,
                                    int64_t               incy,
                                    hipblasComplex*       result);

hipblasStatus_t hipblasCdotcCast_64(hipblasHandle_t       handle,
                                    int64_t               n,
                                    const hipblasComplex* x,
                                    int64_t               incx,
                                    const hipblasComplex* y,
                                    int64_t               incy,
                                    hipblasComplex*       result);

hipblasStatus_t hipblasZdotuCast_64(hipblasHandle_t             handle,
                                    int64_t                     n,
                                    const hipblasDoubleComplex* x,
                                    int64_t                     incx,
                                    const hipblasDoubleComplex* y,
                                    int64_t                     incy,
                                    hipblasDoubleComplex*       result);

hipblasStatus_t hipblasZdotcCast_64(hipblasHandle_t             handle,
                                    int64_t                     n,
                                    const hipblasDoubleComplex* x,
                                    int64_t                     incx,
                                    const hipblasDoubleComplex* y,
                                    int64_t                     incy,
                                    hipblasDoubleComplex*       result);

hipblasStatus_t hipblasCdotuBatchedCast_64(hipblasHandle_t             handle,
                                           int64_t                     n,
                                           const hipblasComplex* const x[],
                                           int64_t                     incx,
                                           const hipblasComplex* const y[],
                                           int64_t                     incy,
                                           int64_t                     batch_count,
                                           hipblasComplex*             result);

hipblasStatus_t hipblasCdotcBatchedCast_64(hipblasHandle_t             handle,
                                           int64_t                     n,
                                           const hipblasComplex* const x[],
                                           int64_t                     incx,
                                           const hipblasComplex* const y[],
                                           int64_t                     incy,
                                           int64_t                     batch_count,
                                           hipblasComplex*             result);

hipblasStatus_t hipblasZdotuBatchedCast_64(hipblasHandle_t                   handle,
                                           int64_t                           n,
                                           const hipblasDoubleComplex* const x[],
                                           int64_t                           incx,
                                           const hipblasDoubleComplex* const y[],
                                           int64_t                           incy,
                                           int64_t                           batch_count,
                                           hipblasDoubleComplex*             result);

hipblasStatus_t hipblasZdotcBatchedCast_64(hipblasHandle_t                   handle,
                                           int64_t                           n,
                                           const hipblasDoubleComplex* const x[],
                                           int64_t                           incx,
                                           const hipblasDoubleComplex* const y[],
                                           int64_t                           incy,
                                           int64_t                           batch_count,
                                           hipblasDoubleComplex*             result);

hipblasStatus_t hipblasCdotuStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  int64_t               n,
                                                  const hipblasComplex* x,
                                                  int64_t               incx,
                                                  hipblasStride         stridex,
                                                  const hipblasComplex* y,
                                                  int64_t               incy,
                                                  hipblasStride         stridey,
                                                  int64_t               batch_count,
                                                  hipblasComplex*       result);

hipblasStatus_t hipblasCdotcStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  int64_t               n,
                                                  const hipblasComplex* x,
                                                  int64_t               incx,
                                                  hipblasStride         stridex,
                                                  const hipblasComplex* y,
                                                  int64_t               incy,
                                                  hipblasStride         stridey,
                                                  int64_t               batch_count,
                                                  hipblasComplex*       result);

hipblasStatus_t hipblasZdotuStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  int64_t                     n,
                                                  const hipblasDoubleComplex* x,
                                                  int64_t                     incx,
                                                  hipblasStride               stridex,
                                                  const hipblasDoubleComplex* y,
                                                  int64_t                     incy,
                                                  hipblasStride               stridey,
                                                  int64_t                     batch_count,
                                                  hipblasDoubleComplex*       result);

hipblasStatus_t hipblasZdotcStridedBatchedCast_64(hipblasHandle_t             handle,
                                                  int64_t                     n,
                                                  const hipblasDoubleComplex* x,
                                                  int64_t                     incx,
                                                  hipblasStride               stridex,
                                                  const hipblasDoubleComplex* y,
                                                  int64_t                     incy,
                                                  hipblasStride               stridey,
                                                  int64_t                     batch_count,
                                                  hipblasDoubleComplex*       result);

// asum
hipblasStatus_t hipblasScasumCast(
    hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, float* result);

hipblasStatus_t hipblasDzasumCast(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, double* result);

hipblasStatus_t hipblasScasumBatchedCast(hipblasHandle_t             handle,
                                         int                         n,
                                         const hipblasComplex* const x[],
                                         int                         incx,
                                         int                         batch_count,
                                         float*                      result);

hipblasStatus_t hipblasDzasumBatchedCast(hipblasHandle_t                   handle,
                                         int                               n,
                                         const hipblasDoubleComplex* const x[],
                                         int                               incx,
                                         int                               batch_count,
                                         double*                           result);

hipblasStatus_t hipblasScasumStridedBatchedCast(hipblasHandle_t       handle,
                                                int                   n,
                                                const hipblasComplex* x,
                                                int                   incx,
                                                hipblasStride         stridex,
                                                int                   batch_count,
                                                float*                result);

hipblasStatus_t hipblasDzasumStridedBatchedCast(hipblasHandle_t             handle,
                                                int                         n,
                                                const hipblasDoubleComplex* x,
                                                int                         incx,
                                                hipblasStride               stridex,
                                                int                         batch_count,
                                                double*                     result);

// asum_64
hipblasStatus_t hipblasScasumCast_64(
    hipblasHandle_t handle, int64_t n, const hipblasComplex* x, int64_t incx, float* result);

hipblasStatus_t hipblasDzasumCast_64(
    hipblasHandle_t handle, int64_t n, const hipblasDoubleComplex* x, int64_t incx, double* result);

hipblasStatus_t hipblasScasumBatchedCast_64(hipblasHandle_t             handle,
                                            int64_t                     n,
                                            const hipblasComplex* const x[],
                                            int64_t                     incx,
                                            int64_t                     batch_count,
                                            float*                      result);

hipblasStatus_t hipblasDzasumBatchedCast_64(hipblasHandle_t                   handle,
                                            int64_t                           n,
                                            const hipblasDoubleComplex* const x[],
                                            int64_t                           incx,
                                            int64_t                           batch_count,
                                            double*                           result);

hipblasStatus_t hipblasScasumStridedBatchedCast_64(hipblasHandle_t       handle,
                                                   int64_t               n,
                                                   const hipblasComplex* x,
                                                   int64_t               incx,
                                                   hipblasStride         stridex,
                                                   int64_t               batch_count,
                                                   float*                result);

hipblasStatus_t hipblasDzasumStridedBatchedCast_64(hipblasHandle_t             handle,
                                                   int64_t                     n,
                                                   const hipblasDoubleComplex* x,
                                                   int64_t                     incx,
                                                   hipblasStride               stridex,
                                                   int64_t                     batch_count,
                                                   double*                     result);

// nrm2
hipblasStatus_t hipblasScnrm2Cast(
    hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, float* result);

hipblasStatus_t hipblasDznrm2Cast(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, double* result);

hipblasStatus_t hipblasScnrm2BatchedCast(hipblasHandle_t             handle,
                                         int                         n,
                                         const hipblasComplex* const x[],
                                         int                         incx,
                                         int                         batch_count,
                                         float*                      result);

hipblasStatus_t hipblasDznrm2BatchedCast(hipblasHandle_t                   handle,
                                         int                               n,
                                         const hipblasDoubleComplex* const x[],
                                         int                               incx,
                                         int                               batch_count,
                                         double*                           result);

hipblasStatus_t hipblasScnrm2StridedBatchedCast(hipblasHandle_t       handle,
                                                int                   n,
                                                const hipblasComplex* x,
                                                int                   incx,
                                                hipblasStride         stridex,
                                                int                   batch_count,
                                                float*                result);

hipblasStatus_t hipblasDznrm2StridedBatchedCast(hipblasHandle_t             handle,
                                                int                         n,
                                                const hipblasDoubleComplex* x,
                                                int                         incx,
                                                hipblasStride               stridex,
                                                int                         batch_count,
                                                double*                     result);

// nrm2_64
hipblasStatus_t hipblasScnrm2Cast_64(
    hipblasHandle_t handle, int64_t n, const hipblasComplex* x, int64_t incx, float* result);

hipblasStatus_t hipblasDznrm2Cast_64(
    hipblasHandle_t handle, int64_t n, const hipblasDoubleComplex* x, int64_t incx, double* result);

hipblasStatus_t hipblasScnrm2BatchedCast_64(hipblasHandle_t             handle,
                                            int64_t                     n,
                                            const hipblasComplex* const x[],
                                            int64_t                     incx,
                                            int64_t                     batch_count,
                                            float*                      result);

hipblasStatus_t hipblasDznrm2BatchedCast_64(hipblasHandle_t                   handle,
                                            int64_t                           n,
                                            const hipblasDoubleComplex* const x[],
                                            int64_t                           incx,
                                            int64_t                           batch_count,
                                            double*                           result);

hipblasStatus_t hipblasScnrm2StridedBatchedCast_64(hipblasHandle_t       handle,
                                                   int64_t               n,
                                                   const hipblasComplex* x,
                                                   int64_t               incx,
                                                   hipblasStride         stridex,
                                                   int64_t               batch_count,
                                                   float*                result);

hipblasStatus_t hipblasDznrm2StridedBatchedCast_64(hipblasHandle_t             handle,
                                                   int64_t                     n,
                                                   const hipblasDoubleComplex* x,
                                                   int64_t                     incx,
                                                   hipblasStride               stridex,
                                                   int64_t                     batch_count,
                                                   double*                     result);

// rot
hipblasStatus_t hipblasCrotCast(hipblasHandle_t       handle,
                                int                   n,
                                hipblasComplex*       x,
                                int                   incx,
                                hipblasComplex*       y,
                                int                   incy,
                                const float*          c,
                                const hipblasComplex* s);

hipblasStatus_t hipblasCsrotCast(hipblasHandle_t handle,
                                 int             n,
                                 hipblasComplex* x,
                                 int             incx,
                                 hipblasComplex* y,
                                 int             incy,
                                 const float*    c,
                                 const float*    s);

hipblasStatus_t hipblasZrotCast(hipblasHandle_t             handle,
                                int                         n,
                                hipblasDoubleComplex*       x,
                                int                         incx,
                                hipblasDoubleComplex*       y,
                                int                         incy,
                                const double*               c,
                                const hipblasDoubleComplex* s);

hipblasStatus_t hipblasZdrotCast(hipblasHandle_t       handle,
                                 int                   n,
                                 hipblasDoubleComplex* x,
                                 int                   incx,
                                 hipblasDoubleComplex* y,
                                 int                   incy,
                                 const double*         c,
                                 const double*         s);

hipblasStatus_t hipblasCrotBatchedCast(hipblasHandle_t       handle,
                                       int                   n,
                                       hipblasComplex* const x[],
                                       int                   incx,
                                       hipblasComplex* const y[],
                                       int                   incy,
                                       const float*          c,
                                       const hipblasComplex* s,
                                       int                   batch_count);

hipblasStatus_t hipblasCsrotBatchedCast(hipblasHandle_t       handle,
                                        int                   n,
                                        hipblasComplex* const x[],
                                        int                   incx,
                                        hipblasComplex* const y[],
                                        int                   incy,
                                        const float*          c,
                                        const float*          s,
                                        int                   batch_count);

hipblasStatus_t hipblasZrotBatchedCast(hipblasHandle_t             handle,
                                       int                         n,
                                       hipblasDoubleComplex* const x[],
                                       int                         incx,
                                       hipblasDoubleComplex* const y[],
                                       int                         incy,
                                       const double*               c,
                                       const hipblasDoubleComplex* s,
                                       int                         batch_count);

hipblasStatus_t hipblasZdrotBatchedCast(hipblasHandle_t             handle,
                                        int                         n,
                                        hipblasDoubleComplex* const x[],
                                        int                         incx,
                                        hipblasDoubleComplex* const y[],
                                        int                         incy,
                                        const double*               c,
                                        const double*               s,
                                        int                         batch_count);

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
                                              int                   batch_count);

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
                                               int             batch_count);

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
                                              int                         batch_count);

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
                                               int                   batch_count);

// rot_64
hipblasStatus_t hipblasCrotCast_64(hipblasHandle_t       handle,
                                   int64_t               n,
                                   hipblasComplex*       x,
                                   int64_t               incx,
                                   hipblasComplex*       y,
                                   int64_t               incy,
                                   const float*          c,
                                   const hipblasComplex* s);

hipblasStatus_t hipblasCsrotCast_64(hipblasHandle_t handle,
                                    int64_t         n,
                                    hipblasComplex* x,
                                    int64_t         incx,
                                    hipblasComplex* y,
                                    int64_t         incy,
                                    const float*    c,
                                    const float*    s);

hipblasStatus_t hipblasZrotCast_64(hipblasHandle_t             handle,
                                   int64_t                     n,
                                   hipblasDoubleComplex*       x,
                                   int64_t                     incx,
                                   hipblasDoubleComplex*       y,
                                   int64_t                     incy,
                                   const double*               c,
                                   const hipblasDoubleComplex* s);

hipblasStatus_t hipblasZdrotCast_64(hipblasHandle_t       handle,
                                    int64_t               n,
                                    hipblasDoubleComplex* x,
                                    int64_t               incx,
                                    hipblasDoubleComplex* y,
                                    int64_t               incy,
                                    const double*         c,
                                    const double*         s);

hipblasStatus_t hipblasCrotBatchedCast_64(hipblasHandle_t       handle,
                                          int64_t               n,
                                          hipblasComplex* const x[],
                                          int64_t               incx,
                                          hipblasComplex* const y[],
                                          int64_t               incy,
                                          const float*          c,
                                          const hipblasComplex* s,
                                          int64_t               batch_count);

hipblasStatus_t hipblasCsrotBatchedCast_64(hipblasHandle_t       handle,
                                           int64_t               n,
                                           hipblasComplex* const x[],
                                           int64_t               incx,
                                           hipblasComplex* const y[],
                                           int64_t               incy,
                                           const float*          c,
                                           const float*          s,
                                           int64_t               batch_count);

hipblasStatus_t hipblasZrotBatchedCast_64(hipblasHandle_t             handle,
                                          int64_t                     n,
                                          hipblasDoubleComplex* const x[],
                                          int64_t                     incx,
                                          hipblasDoubleComplex* const y[],
                                          int64_t                     incy,
                                          const double*               c,
                                          const hipblasDoubleComplex* s,
                                          int64_t                     batch_count);

hipblasStatus_t hipblasZdrotBatchedCast_64(hipblasHandle_t             handle,
                                           int64_t                     n,
                                           hipblasDoubleComplex* const x[],
                                           int64_t                     incx,
                                           hipblasDoubleComplex* const y[],
                                           int64_t                     incy,
                                           const double*               c,
                                           const double*               s,
                                           int64_t                     batch_count);

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
                                                 int64_t               batch_count);

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
                                                  int64_t         batch_count);

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
                                                 int64_t                     batch_count);

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
                                                  int64_t               batch_count);

// rotg
hipblasStatus_t hipblasCrotgCast(
    hipblasHandle_t handle, hipblasComplex* a, hipblasComplex* b, float* c, hipblasComplex* s);

hipblasStatus_t hipblasZrotgCast(hipblasHandle_t       handle,
                                 hipblasDoubleComplex* a,
                                 hipblasDoubleComplex* b,
                                 double*               c,
                                 hipblasDoubleComplex* s);

hipblasStatus_t hipblasCrotgBatchedCast(hipblasHandle_t       handle,
                                        hipblasComplex* const a[],
                                        hipblasComplex* const b[],
                                        float* const          c[],
                                        hipblasComplex* const s[],
                                        int                   batch_count);

hipblasStatus_t hipblasZrotgBatchedCast(hipblasHandle_t             handle,
                                        hipblasDoubleComplex* const a[],
                                        hipblasDoubleComplex* const b[],
                                        double* const               c[],
                                        hipblasDoubleComplex* const s[],
                                        int                         batch_count);

hipblasStatus_t hipblasCrotgStridedBatchedCast(hipblasHandle_t handle,
                                               hipblasComplex* a,
                                               hipblasStride   stridea,
                                               hipblasComplex* b,
                                               hipblasStride   strideb,
                                               float*          c,
                                               hipblasStride   stridec,
                                               hipblasComplex* s,
                                               hipblasStride   strides,
                                               int             batch_count);

hipblasStatus_t hipblasZrotgStridedBatchedCast(hipblasHandle_t       handle,
                                               hipblasDoubleComplex* a,
                                               hipblasStride         stridea,
                                               hipblasDoubleComplex* b,
                                               hipblasStride         strideb,
                                               double*               c,
                                               hipblasStride         stridec,
                                               hipblasDoubleComplex* s,
                                               hipblasStride         strides,
                                               int                   batch_count);

// rotg_64
hipblasStatus_t hipblasCrotgCast_64(
    hipblasHandle_t handle, hipblasComplex* a, hipblasComplex* b, float* c, hipblasComplex* s);

hipblasStatus_t hipblasZrotgCast_64(hipblasHandle_t       handle,
                                    hipblasDoubleComplex* a,
                                    hipblasDoubleComplex* b,
                                    double*               c,
                                    hipblasDoubleComplex* s);

hipblasStatus_t hipblasCrotgBatchedCast_64(hipblasHandle_t       handle,
                                           hipblasComplex* const a[],
                                           hipblasComplex* const b[],
                                           float* const          c[],
                                           hipblasComplex* const s[],
                                           int64_t               batch_count);

hipblasStatus_t hipblasZrotgBatchedCast_64(hipblasHandle_t             handle,
                                           hipblasDoubleComplex* const a[],
                                           hipblasDoubleComplex* const b[],
                                           double* const               c[],
                                           hipblasDoubleComplex* const s[],
                                           int64_t                     batch_count);

hipblasStatus_t hipblasCrotgStridedBatchedCast_64(hipblasHandle_t handle,
                                                  hipblasComplex* a,
                                                  hipblasStride   stridea,
                                                  hipblasComplex* b,
                                                  hipblasStride   strideb,
                                                  float*          c,
                                                  hipblasStride   stridec,
                                                  hipblasComplex* s,
                                                  hipblasStride   strides,
                                                  int64_t         batch_count);

hipblasStatus_t hipblasZrotgStridedBatchedCast_64(hipblasHandle_t       handle,
                                                  hipblasDoubleComplex* a,
                                                  hipblasStride         stridea,
                                                  hipblasDoubleComplex* b,
                                                  hipblasStride         strideb,
                                                  double*               c,
                                                  hipblasStride         stridec,
                                                  hipblasDoubleComplex* s,
                                                  hipblasStride         strides,
                                                  int64_t               batch_count);

// rotm, rotmg - no complex versions

// amax
hipblasStatus_t hipblasIcamaxCast(
    hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result);

hipblasStatus_t hipblasIzamaxCast(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result);

hipblasStatus_t hipblasIcamaxBatchedCast(hipblasHandle_t             handle,
                                         int                         n,
                                         const hipblasComplex* const x[],
                                         int                         incx,
                                         int                         batch_count,
                                         int*                        result);

hipblasStatus_t hipblasIzamaxBatchedCast(hipblasHandle_t                   handle,
                                         int                               n,
                                         const hipblasDoubleComplex* const x[],
                                         int                               incx,
                                         int                               batch_count,
                                         int*                              result);

hipblasStatus_t hipblasIcamaxStridedBatchedCast(hipblasHandle_t       handle,
                                                int                   n,
                                                const hipblasComplex* x,
                                                int                   incx,
                                                hipblasStride         stridex,
                                                int                   batch_count,
                                                int*                  result);

hipblasStatus_t hipblasIzamaxStridedBatchedCast(hipblasHandle_t             handle,
                                                int                         n,
                                                const hipblasDoubleComplex* x,
                                                int                         incx,
                                                hipblasStride               stridex,
                                                int                         batch_count,
                                                int*                        result);

// amax_64
hipblasStatus_t hipblasIcamaxCast_64(
    hipblasHandle_t handle, int64_t n, const hipblasComplex* x, int64_t incx, int64_t* result);

hipblasStatus_t hipblasIzamaxCast_64(hipblasHandle_t             handle,
                                     int64_t                     n,
                                     const hipblasDoubleComplex* x,
                                     int64_t                     incx,
                                     int64_t*                    result);

hipblasStatus_t hipblasIcamaxBatchedCast_64(hipblasHandle_t             handle,
                                            int64_t                     n,
                                            const hipblasComplex* const x[],
                                            int64_t                     incx,
                                            int64_t                     batch_count,
                                            int64_t*                    result);

hipblasStatus_t hipblasIzamaxBatchedCast_64(hipblasHandle_t                   handle,
                                            int64_t                           n,
                                            const hipblasDoubleComplex* const x[],
                                            int64_t                           incx,
                                            int64_t                           batch_count,
                                            int64_t*                          result);

hipblasStatus_t hipblasIcamaxStridedBatchedCast_64(hipblasHandle_t       handle,
                                                   int64_t               n,
                                                   const hipblasComplex* x,
                                                   int64_t               incx,
                                                   hipblasStride         stridex,
                                                   int64_t               batch_count,
                                                   int64_t*              result);

hipblasStatus_t hipblasIzamaxStridedBatchedCast_64(hipblasHandle_t             handle,
                                                   int64_t                     n,
                                                   const hipblasDoubleComplex* x,
                                                   int64_t                     incx,
                                                   hipblasStride               stridex,
                                                   int64_t                     batch_count,
                                                   int64_t*                    result);

// amin
hipblasStatus_t hipblasIcaminCast(
    hipblasHandle_t handle, int n, const hipblasComplex* x, int incx, int* result);

hipblasStatus_t hipblasIzaminCast(
    hipblasHandle_t handle, int n, const hipblasDoubleComplex* x, int incx, int* result);

hipblasStatus_t hipblasIcaminBatchedCast(hipblasHandle_t             handle,
                                         int                         n,
                                         const hipblasComplex* const x[],
                                         int                         incx,
                                         int                         batch_count,
                                         int*                        result);

hipblasStatus_t hipblasIzaminBatchedCast(hipblasHandle_t                   handle,
                                         int                               n,
                                         const hipblasDoubleComplex* const x[],
                                         int                               incx,
                                         int                               batch_count,
                                         int*                              result);

hipblasStatus_t hipblasIcaminStridedBatchedCast(hipblasHandle_t       handle,
                                                int                   n,
                                                const hipblasComplex* x,
                                                int                   incx,
                                                hipblasStride         stridex,
                                                int                   batch_count,
                                                int*                  result);

hipblasStatus_t hipblasIzaminStridedBatchedCast(hipblasHandle_t             handle,
                                                int                         n,
                                                const hipblasDoubleComplex* x,
                                                int                         incx,
                                                hipblasStride               stridex,
                                                int                         batch_count,
                                                int*                        result);

// amin_64
hipblasStatus_t hipblasIcaminCast_64(
    hipblasHandle_t handle, int64_t n, const hipblasComplex* x, int64_t incx, int64_t* result);

hipblasStatus_t hipblasIzaminCast_64(hipblasHandle_t             handle,
                                     int64_t                     n,
                                     const hipblasDoubleComplex* x,
                                     int64_t                     incx,
                                     int64_t*                    result);

hipblasStatus_t hipblasIcaminBatchedCast_64(hipblasHandle_t             handle,
                                            int64_t                     n,
                                            const hipblasComplex* const x[],
                                            int64_t                     incx,
                                            int64_t                     batch_count,
                                            int64_t*                    result);

hipblasStatus_t hipblasIzaminBatchedCast_64(hipblasHandle_t                   handle,
                                            int64_t                           n,
                                            const hipblasDoubleComplex* const x[],
                                            int64_t                           incx,
                                            int64_t                           batch_count,
                                            int64_t*                          result);

hipblasStatus_t hipblasIcaminStridedBatchedCast_64(hipblasHandle_t       handle,
                                                   int64_t               n,
                                                   const hipblasComplex* x,
                                                   int64_t               incx,
                                                   hipblasStride         stridex,
                                                   int64_t               batch_count,
                                                   int64_t*              result);

hipblasStatus_t hipblasIzaminStridedBatchedCast_64(hipblasHandle_t             handle,
                                                   int64_t                     n,
                                                   const hipblasDoubleComplex* x,
                                                   int64_t                     incx,
                                                   hipblasStride               stridex,
                                                   int64_t                     batch_count,
                                                   int64_t*                    result);

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
                                 int                   incy);

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
                                 int                         incy);

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
                                        int                         batch_count);

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
                                        int                               batch_count);

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
                                               int                   batch_count);

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
                                               int                         batch_count);

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
                                    int64_t               incy);

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
                                    int64_t                     incy);

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
                                           int64_t                     batch_count);

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
                                           int64_t                           batch_count);

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
                                                  int64_t               batch_count);

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
                                                  int64_t                     batch_count);

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
                                 int                   incy);

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
                                 int                         incy);

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
                                        int                         batch_count);

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
                                        int                               batch_count);

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
                                               int                   batch_count);

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
                                               int                         batch_count);

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
                                    int64_t               incy);

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
                                    int64_t                     incy);

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
                                           int64_t                     batch_count);

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
                                           int64_t                           batch_count);

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
                                                  int64_t               batch_count);

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
                                                  int64_t                     batch_count);

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
                                 int                   lda);

hipblasStatus_t hipblasCgercCast(hipblasHandle_t       handle,
                                 int                   m,
                                 int                   n,
                                 const hipblasComplex* alpha,
                                 const hipblasComplex* x,
                                 int                   incx,
                                 const hipblasComplex* y,
                                 int                   incy,
                                 hipblasComplex*       A,
                                 int                   lda);

hipblasStatus_t hipblasZgeruCast(hipblasHandle_t             handle,
                                 int                         m,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* y,
                                 int                         incy,
                                 hipblasDoubleComplex*       A,
                                 int                         lda);

hipblasStatus_t hipblasZgercCast(hipblasHandle_t             handle,
                                 int                         m,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* y,
                                 int                         incy,
                                 hipblasDoubleComplex*       A,
                                 int                         lda);

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
                                        int                         batch_count);

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
                                        int                         batch_count);

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
                                        int                               batch_count);

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
                                        int                               batch_count);

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
                                               int                   batch_count);

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
                                               int                   batch_count);

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
                                               int                         batch_count);

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
                                               int                         batch_count);

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
                                    int64_t               lda);

hipblasStatus_t hipblasCgercCast_64(hipblasHandle_t       handle,
                                    int64_t               m,
                                    int64_t               n,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* x,
                                    int64_t               incx,
                                    const hipblasComplex* y,
                                    int64_t               incy,
                                    hipblasComplex*       A,
                                    int64_t               lda);

hipblasStatus_t hipblasZgeruCast_64(hipblasHandle_t             handle,
                                    int64_t                     m,
                                    int64_t                     n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* x,
                                    int64_t                     incx,
                                    const hipblasDoubleComplex* y,
                                    int64_t                     incy,
                                    hipblasDoubleComplex*       A,
                                    int64_t                     lda);

hipblasStatus_t hipblasZgercCast_64(hipblasHandle_t             handle,
                                    int64_t                     m,
                                    int64_t                     n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* x,
                                    int64_t                     incx,
                                    const hipblasDoubleComplex* y,
                                    int64_t                     incy,
                                    hipblasDoubleComplex*       A,
                                    int64_t                     lda);

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
                                           int64_t                     batch_count);

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
                                           int64_t                     batch_count);

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
                                           int64_t                           batch_count);

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
                                           int64_t                           batch_count);

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
                                                  int64_t               batch_count);

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
                                                  int64_t               batch_count);

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
                                                  int64_t                     batch_count);

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
                                                  int64_t                     batch_count);
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
                                 int                   incy);

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
                                 int                         incy);

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
                                        int                         batchCount);

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
                                        int                               batchCount);

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
                                               int                   batchCount);

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
                                               int                         batchCount);

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
                                    int64_t               incy);

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
                                    int64_t                     incy);

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
                                           int64_t                     batchCount);

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
                                           int64_t                           batchCount);

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
                                                  int64_t               batchCount);

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
                                                  int64_t                     batchCount);

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
                                 int                   incy);

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
                                 int                         incy);

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
                                        int                         batch_count);

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
                                        int                               batch_count);

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
                                               int                   batch_count);

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
                                               int                         batch_count);

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
                                    int64_t               incy);

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
                                    int64_t                     incy);

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
                                           int64_t                     batch_count);

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
                                           int64_t                           batch_count);

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
                                                  int64_t               batch_count);

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
                                                  int64_t                     batch_count);

// her
hipblasStatus_t hipblasCherCast(hipblasHandle_t       handle,
                                hipblasFillMode_t     uplo,
                                int                   n,
                                const float*          alpha,
                                const hipblasComplex* x,
                                int                   incx,
                                hipblasComplex*       A,
                                int                   lda);

hipblasStatus_t hipblasZherCast(hipblasHandle_t             handle,
                                hipblasFillMode_t           uplo,
                                int                         n,
                                const double*               alpha,
                                const hipblasDoubleComplex* x,
                                int                         incx,
                                hipblasDoubleComplex*       A,
                                int                         lda);

hipblasStatus_t hipblasCherBatchedCast(hipblasHandle_t             handle,
                                       hipblasFillMode_t           uplo,
                                       int                         n,
                                       const float*                alpha,
                                       const hipblasComplex* const x[],
                                       int                         incx,
                                       hipblasComplex* const       A[],
                                       int                         lda,
                                       int                         batchCount);

hipblasStatus_t hipblasZherBatchedCast(hipblasHandle_t                   handle,
                                       hipblasFillMode_t                 uplo,
                                       int                               n,
                                       const double*                     alpha,
                                       const hipblasDoubleComplex* const x[],
                                       int                               incx,
                                       hipblasDoubleComplex* const       A[],
                                       int                               lda,
                                       int                               batchCount);

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
                                              int                   batchCount);

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
                                              int                         batchCount);

// her_64
hipblasStatus_t hipblasCherCast_64(hipblasHandle_t       handle,
                                   hipblasFillMode_t     uplo,
                                   int64_t               n,
                                   const float*          alpha,
                                   const hipblasComplex* x,
                                   int64_t               incx,
                                   hipblasComplex*       A,
                                   int64_t               lda);

hipblasStatus_t hipblasZherCast_64(hipblasHandle_t             handle,
                                   hipblasFillMode_t           uplo,
                                   int64_t                     n,
                                   const double*               alpha,
                                   const hipblasDoubleComplex* x,
                                   int64_t                     incx,
                                   hipblasDoubleComplex*       A,
                                   int64_t                     lda);

hipblasStatus_t hipblasCherBatchedCast_64(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          int64_t                     n,
                                          const float*                alpha,
                                          const hipblasComplex* const x[],
                                          int64_t                     incx,
                                          hipblasComplex* const       A[],
                                          int64_t                     lda,
                                          int64_t                     batchCount);

hipblasStatus_t hipblasZherBatchedCast_64(hipblasHandle_t                   handle,
                                          hipblasFillMode_t                 uplo,
                                          int64_t                           n,
                                          const double*                     alpha,
                                          const hipblasDoubleComplex* const x[],
                                          int64_t                           incx,
                                          hipblasDoubleComplex* const       A[],
                                          int64_t                           lda,
                                          int64_t                           batchCount);

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
                                                 int64_t               batchCount);

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
                                                 int64_t                     batchCount);

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
                                 int                   lda);

hipblasStatus_t hipblasZher2Cast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* y,
                                 int                         incy,
                                 hipblasDoubleComplex*       A,
                                 int                         lda);

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
                                        int                         batchCount);

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
                                        int                               batchCount);

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
                                               int                   batchCount);

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
                                               int                         batchCount);

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
                                    int64_t               lda);

hipblasStatus_t hipblasZher2Cast_64(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int64_t                     n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* x,
                                    int64_t                     incx,
                                    const hipblasDoubleComplex* y,
                                    int64_t                     incy,
                                    hipblasDoubleComplex*       A,
                                    int64_t                     lda);

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
                                           int64_t                     batchCount);

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
                                           int64_t                           batchCount);

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
                                                  int64_t               batchCount);

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
                                                  int64_t                     batchCount);

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
                                 int                   incy);

hipblasStatus_t hipblasZhpmvCast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* AP,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* beta,
                                 hipblasDoubleComplex*       y,
                                 int                         incy);

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
                                        int                         batchCount);

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
                                        int                               batchCount);

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
                                               int                   batchCount);

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
                                               int                         batchCount);

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
                                    int64_t               incy);

hipblasStatus_t hipblasZhpmvCast_64(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int64_t                     n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* AP,
                                    const hipblasDoubleComplex* x,
                                    int64_t                     incx,
                                    const hipblasDoubleComplex* beta,
                                    hipblasDoubleComplex*       y,
                                    int64_t                     incy);

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
                                           int64_t                     batchCount);

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
                                           int64_t                           batchCount);

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
                                                  int64_t               batchCount);

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
                                                  int64_t                     batchCount);

// hpr
hipblasStatus_t hipblasChprCast(hipblasHandle_t       handle,
                                hipblasFillMode_t     uplo,
                                int                   n,
                                const float*          alpha,
                                const hipblasComplex* x,
                                int                   incx,
                                hipblasComplex*       AP);

hipblasStatus_t hipblasZhprCast(hipblasHandle_t             handle,
                                hipblasFillMode_t           uplo,
                                int                         n,
                                const double*               alpha,
                                const hipblasDoubleComplex* x,
                                int                         incx,
                                hipblasDoubleComplex*       AP);

hipblasStatus_t hipblasChprBatchedCast(hipblasHandle_t             handle,
                                       hipblasFillMode_t           uplo,
                                       int                         n,
                                       const float*                alpha,
                                       const hipblasComplex* const x[],
                                       int                         incx,
                                       hipblasComplex* const       AP[],
                                       int                         batchCount);

hipblasStatus_t hipblasZhprBatchedCast(hipblasHandle_t                   handle,
                                       hipblasFillMode_t                 uplo,
                                       int                               n,
                                       const double*                     alpha,
                                       const hipblasDoubleComplex* const x[],
                                       int                               incx,
                                       hipblasDoubleComplex* const       AP[],
                                       int                               batchCount);

hipblasStatus_t hipblasChprStridedBatchedCast(hipblasHandle_t       handle,
                                              hipblasFillMode_t     uplo,
                                              int                   n,
                                              const float*          alpha,
                                              const hipblasComplex* x,
                                              int                   incx,
                                              hipblasStride         stridex,
                                              hipblasComplex*       AP,
                                              hipblasStride         strideAP,
                                              int                   batchCount);

hipblasStatus_t hipblasZhprStridedBatchedCast(hipblasHandle_t             handle,
                                              hipblasFillMode_t           uplo,
                                              int                         n,
                                              const double*               alpha,
                                              const hipblasDoubleComplex* x,
                                              int                         incx,
                                              hipblasStride               stridex,
                                              hipblasDoubleComplex*       AP,
                                              hipblasStride               strideAP,
                                              int                         batchCount);

// hpr_64
hipblasStatus_t hipblasChprCast_64(hipblasHandle_t       handle,
                                   hipblasFillMode_t     uplo,
                                   int64_t               n,
                                   const float*          alpha,
                                   const hipblasComplex* x,
                                   int64_t               incx,
                                   hipblasComplex*       AP);

hipblasStatus_t hipblasZhprCast_64(hipblasHandle_t             handle,
                                   hipblasFillMode_t           uplo,
                                   int64_t                     n,
                                   const double*               alpha,
                                   const hipblasDoubleComplex* x,
                                   int64_t                     incx,
                                   hipblasDoubleComplex*       AP);

hipblasStatus_t hipblasChprBatchedCast_64(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          int64_t                     n,
                                          const float*                alpha,
                                          const hipblasComplex* const x[],
                                          int64_t                     incx,
                                          hipblasComplex* const       AP[],
                                          int64_t                     batchCount);

hipblasStatus_t hipblasZhprBatchedCast_64(hipblasHandle_t                   handle,
                                          hipblasFillMode_t                 uplo,
                                          int64_t                           n,
                                          const double*                     alpha,
                                          const hipblasDoubleComplex* const x[],
                                          int64_t                           incx,
                                          hipblasDoubleComplex* const       AP[],
                                          int64_t                           batchCount);

hipblasStatus_t hipblasChprStridedBatchedCast_64(hipblasHandle_t       handle,
                                                 hipblasFillMode_t     uplo,
                                                 int64_t               n,
                                                 const float*          alpha,
                                                 const hipblasComplex* x,
                                                 int64_t               incx,
                                                 hipblasStride         stridex,
                                                 hipblasComplex*       AP,
                                                 hipblasStride         strideAP,
                                                 int64_t               batchCount);

hipblasStatus_t hipblasZhprStridedBatchedCast_64(hipblasHandle_t             handle,
                                                 hipblasFillMode_t           uplo,
                                                 int64_t                     n,
                                                 const double*               alpha,
                                                 const hipblasDoubleComplex* x,
                                                 int64_t                     incx,
                                                 hipblasStride               stridex,
                                                 hipblasDoubleComplex*       AP,
                                                 hipblasStride               strideAP,
                                                 int64_t                     batchCount);

// hpr2
hipblasStatus_t hipblasChpr2Cast(hipblasHandle_t       handle,
                                 hipblasFillMode_t     uplo,
                                 int                   n,
                                 const hipblasComplex* alpha,
                                 const hipblasComplex* x,
                                 int                   incx,
                                 const hipblasComplex* y,
                                 int                   incy,
                                 hipblasComplex*       AP);

hipblasStatus_t hipblasZhpr2Cast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* y,
                                 int                         incy,
                                 hipblasDoubleComplex*       AP);

hipblasStatus_t hipblasChpr2BatchedCast(hipblasHandle_t             handle,
                                        hipblasFillMode_t           uplo,
                                        int                         n,
                                        const hipblasComplex*       alpha,
                                        const hipblasComplex* const x[],
                                        int                         incx,
                                        const hipblasComplex* const y[],
                                        int                         incy,
                                        hipblasComplex* const       AP[],
                                        int                         batchCount);

hipblasStatus_t hipblasZhpr2BatchedCast(hipblasHandle_t                   handle,
                                        hipblasFillMode_t                 uplo,
                                        int                               n,
                                        const hipblasDoubleComplex*       alpha,
                                        const hipblasDoubleComplex* const x[],
                                        int                               incx,
                                        const hipblasDoubleComplex* const y[],
                                        int                               incy,
                                        hipblasDoubleComplex* const       AP[],
                                        int                               batchCount);

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
                                               int                   batchCount);

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
                                               int                         batchCount);

// hpr2_64
hipblasStatus_t hipblasChpr2Cast_64(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    int64_t               n,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* x,
                                    int64_t               incx,
                                    const hipblasComplex* y,
                                    int64_t               incy,
                                    hipblasComplex*       AP);

hipblasStatus_t hipblasZhpr2Cast_64(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int64_t                     n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* x,
                                    int64_t                     incx,
                                    const hipblasDoubleComplex* y,
                                    int64_t                     incy,
                                    hipblasDoubleComplex*       AP);

hipblasStatus_t hipblasChpr2BatchedCast_64(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           int64_t                     n,
                                           const hipblasComplex*       alpha,
                                           const hipblasComplex* const x[],
                                           int64_t                     incx,
                                           const hipblasComplex* const y[],
                                           int64_t                     incy,
                                           hipblasComplex* const       AP[],
                                           int64_t                     batchCount);

hipblasStatus_t hipblasZhpr2BatchedCast_64(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           int64_t                           n,
                                           const hipblasDoubleComplex*       alpha,
                                           const hipblasDoubleComplex* const x[],
                                           int64_t                           incx,
                                           const hipblasDoubleComplex* const y[],
                                           int64_t                           incy,
                                           hipblasDoubleComplex* const       AP[],
                                           int64_t                           batchCount);

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
                                                  int64_t               batchCount);

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
                                                  int64_t                     batchCount);

// sbmv, spmv, spr2 no complex versions

// spr
hipblasStatus_t hipblasCsprCast(hipblasHandle_t       handle,
                                hipblasFillMode_t     uplo,
                                int                   n,
                                const hipblasComplex* alpha,
                                const hipblasComplex* x,
                                int                   incx,
                                hipblasComplex*       AP);

hipblasStatus_t hipblasZsprCast(hipblasHandle_t             handle,
                                hipblasFillMode_t           uplo,
                                int                         n,
                                const hipblasDoubleComplex* alpha,
                                const hipblasDoubleComplex* x,
                                int                         incx,
                                hipblasDoubleComplex*       AP);

hipblasStatus_t hipblasCsprBatchedCast(hipblasHandle_t             handle,
                                       hipblasFillMode_t           uplo,
                                       int                         n,
                                       const hipblasComplex*       alpha,
                                       const hipblasComplex* const x[],
                                       int                         incx,
                                       hipblasComplex* const       AP[],
                                       int                         batchCount);

hipblasStatus_t hipblasZsprBatchedCast(hipblasHandle_t                   handle,
                                       hipblasFillMode_t                 uplo,
                                       int                               n,
                                       const hipblasDoubleComplex*       alpha,
                                       const hipblasDoubleComplex* const x[],
                                       int                               incx,
                                       hipblasDoubleComplex* const       AP[],
                                       int                               batchCount);

hipblasStatus_t hipblasCsprStridedBatchedCast(hipblasHandle_t       handle,
                                              hipblasFillMode_t     uplo,
                                              int                   n,
                                              const hipblasComplex* alpha,
                                              const hipblasComplex* x,
                                              int                   incx,
                                              hipblasStride         stridex,
                                              hipblasComplex*       AP,
                                              hipblasStride         strideAP,
                                              int                   batchCount);

hipblasStatus_t hipblasZsprStridedBatchedCast(hipblasHandle_t             handle,
                                              hipblasFillMode_t           uplo,
                                              int                         n,
                                              const hipblasDoubleComplex* alpha,
                                              const hipblasDoubleComplex* x,
                                              int                         incx,
                                              hipblasStride               stridex,
                                              hipblasDoubleComplex*       AP,
                                              hipblasStride               strideAP,
                                              int                         batchCount);

// spr_64
hipblasStatus_t hipblasCsprCast_64(hipblasHandle_t       handle,
                                   hipblasFillMode_t     uplo,
                                   int64_t               n,
                                   const hipblasComplex* alpha,
                                   const hipblasComplex* x,
                                   int64_t               incx,
                                   hipblasComplex*       AP);

hipblasStatus_t hipblasZsprCast_64(hipblasHandle_t             handle,
                                   hipblasFillMode_t           uplo,
                                   int64_t                     n,
                                   const hipblasDoubleComplex* alpha,
                                   const hipblasDoubleComplex* x,
                                   int64_t                     incx,
                                   hipblasDoubleComplex*       AP);

hipblasStatus_t hipblasCsprBatchedCast_64(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          int64_t                     n,
                                          const hipblasComplex*       alpha,
                                          const hipblasComplex* const x[],
                                          int64_t                     incx,
                                          hipblasComplex* const       AP[],
                                          int64_t                     batchCount);

hipblasStatus_t hipblasZsprBatchedCast_64(hipblasHandle_t                   handle,
                                          hipblasFillMode_t                 uplo,
                                          int64_t                           n,
                                          const hipblasDoubleComplex*       alpha,
                                          const hipblasDoubleComplex* const x[],
                                          int64_t                           incx,
                                          hipblasDoubleComplex* const       AP[],
                                          int64_t                           batchCount);

hipblasStatus_t hipblasCsprStridedBatchedCast_64(hipblasHandle_t       handle,
                                                 hipblasFillMode_t     uplo,
                                                 int64_t               n,
                                                 const hipblasComplex* alpha,
                                                 const hipblasComplex* x,
                                                 int64_t               incx,
                                                 hipblasStride         stridex,
                                                 hipblasComplex*       AP,
                                                 hipblasStride         strideAP,
                                                 int64_t               batchCount);

hipblasStatus_t hipblasZsprStridedBatchedCast_64(hipblasHandle_t             handle,
                                                 hipblasFillMode_t           uplo,
                                                 int64_t                     n,
                                                 const hipblasDoubleComplex* alpha,
                                                 const hipblasDoubleComplex* x,
                                                 int64_t                     incx,
                                                 hipblasStride               stridex,
                                                 hipblasDoubleComplex*       AP,
                                                 hipblasStride               strideAP,
                                                 int64_t                     batchCount);

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
                                 int                   incy);

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
                                 int                         incy);

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
                                        int                         batchCount);

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
                                        int                               batchCount);

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
                                               int                   batchCount);

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
                                               int                         batchCount);

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
                                    int64_t               incy);

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
                                    int64_t                     incy);

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
                                           int64_t                     batchCount);

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
                                           int64_t                           batchCount);

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
                                                  int64_t               batchCount);

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
                                                  int64_t                     batchCount);

// syr
hipblasStatus_t hipblasCsyrCast(hipblasHandle_t       handle,
                                hipblasFillMode_t     uplo,
                                int                   n,
                                const hipblasComplex* alpha,
                                const hipblasComplex* x,
                                int                   incx,
                                hipblasComplex*       A,
                                int                   lda);

hipblasStatus_t hipblasZsyrCast(hipblasHandle_t             handle,
                                hipblasFillMode_t           uplo,
                                int                         n,
                                const hipblasDoubleComplex* alpha,
                                const hipblasDoubleComplex* x,
                                int                         incx,
                                hipblasDoubleComplex*       A,
                                int                         lda);

hipblasStatus_t hipblasCsyrBatchedCast(hipblasHandle_t             handle,
                                       hipblasFillMode_t           uplo,
                                       int                         n,
                                       const hipblasComplex*       alpha,
                                       const hipblasComplex* const x[],
                                       int                         incx,
                                       hipblasComplex* const       A[],
                                       int                         lda,
                                       int                         batch_count);

hipblasStatus_t hipblasZsyrBatchedCast(hipblasHandle_t                   handle,
                                       hipblasFillMode_t                 uplo,
                                       int                               n,
                                       const hipblasDoubleComplex*       alpha,
                                       const hipblasDoubleComplex* const x[],
                                       int                               incx,
                                       hipblasDoubleComplex* const       A[],
                                       int                               lda,
                                       int                               batch_count);

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
                                              int                   batch_count);

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
                                              int                         batch_count);

// syr_64
hipblasStatus_t hipblasCsyrCast_64(hipblasHandle_t       handle,
                                   hipblasFillMode_t     uplo,
                                   int64_t               n,
                                   const hipblasComplex* alpha,
                                   const hipblasComplex* x,
                                   int64_t               incx,
                                   hipblasComplex*       A,
                                   int64_t               lda);

hipblasStatus_t hipblasZsyrCast_64(hipblasHandle_t             handle,
                                   hipblasFillMode_t           uplo,
                                   int64_t                     n,
                                   const hipblasDoubleComplex* alpha,
                                   const hipblasDoubleComplex* x,
                                   int64_t                     incx,
                                   hipblasDoubleComplex*       A,
                                   int64_t                     lda);

hipblasStatus_t hipblasCsyrBatchedCast_64(hipblasHandle_t             handle,
                                          hipblasFillMode_t           uplo,
                                          int64_t                     n,
                                          const hipblasComplex*       alpha,
                                          const hipblasComplex* const x[],
                                          int64_t                     incx,
                                          hipblasComplex* const       A[],
                                          int64_t                     lda,
                                          int64_t                     batch_count);

hipblasStatus_t hipblasZsyrBatchedCast_64(hipblasHandle_t                   handle,
                                          hipblasFillMode_t                 uplo,
                                          int64_t                           n,
                                          const hipblasDoubleComplex*       alpha,
                                          const hipblasDoubleComplex* const x[],
                                          int64_t                           incx,
                                          hipblasDoubleComplex* const       A[],
                                          int64_t                           lda,
                                          int64_t                           batch_count);

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
                                                 int64_t               batch_count);

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
                                                 int64_t                     batch_count);

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
                                 int                   lda);

hipblasStatus_t hipblasZsyr2Cast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 int                         n,
                                 const hipblasDoubleComplex* alpha,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 const hipblasDoubleComplex* y,
                                 int                         incy,
                                 hipblasDoubleComplex*       A,
                                 int                         lda);

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
                                        int                         batchCount);

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
                                        int                               batchCount);

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
                                               int                   batchCount);

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
                                               int                         batchCount);

// syr2_64
hipblasStatus_t hipblasCsyr2Cast_64(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    int64_t               n,
                                    const hipblasComplex* alpha,
                                    const hipblasComplex* x,
                                    int64_t               incx,
                                    const hipblasComplex* y,
                                    int64_t               incy,
                                    hipblasComplex*       A,
                                    int64_t               lda);

hipblasStatus_t hipblasZsyr2Cast_64(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    int64_t                     n,
                                    const hipblasDoubleComplex* alpha,
                                    const hipblasDoubleComplex* x,
                                    int64_t                     incx,
                                    const hipblasDoubleComplex* y,
                                    int64_t                     incy,
                                    hipblasDoubleComplex*       A,
                                    int64_t                     lda);

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
                                           int64_t                     batchCount);

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
                                           int64_t                           batchCount);

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
                                                  int64_t               batchCount);

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
                                                  int64_t                     batchCount);

// trsv
hipblasStatus_t hipblasCtrsvCast(hipblasHandle_t       handle,
                                 hipblasFillMode_t     uplo,
                                 hipblasOperation_t    transA,
                                 hipblasDiagType_t     diag,
                                 int                   m,
                                 const hipblasComplex* A,
                                 int                   lda,
                                 hipblasComplex*       x,
                                 int                   incx);

hipblasStatus_t hipblasZtrsvCast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 hipblasOperation_t          transA,
                                 hipblasDiagType_t           diag,
                                 int                         m,
                                 const hipblasDoubleComplex* A,
                                 int                         lda,
                                 hipblasDoubleComplex*       x,
                                 int                         incx);

hipblasStatus_t hipblasCtrsvBatchedCast(hipblasHandle_t             handle,
                                        hipblasFillMode_t           uplo,
                                        hipblasOperation_t          transA,
                                        hipblasDiagType_t           diag,
                                        int                         m,
                                        const hipblasComplex* const A[],
                                        int                         lda,
                                        hipblasComplex* const       x[],
                                        int                         incx,
                                        int                         batch_count);

hipblasStatus_t hipblasZtrsvBatchedCast(hipblasHandle_t                   handle,
                                        hipblasFillMode_t                 uplo,
                                        hipblasOperation_t                transA,
                                        hipblasDiagType_t                 diag,
                                        int                               m,
                                        const hipblasDoubleComplex* const A[],
                                        int                               lda,
                                        hipblasDoubleComplex* const       x[],
                                        int                               incx,
                                        int                               batch_count);

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
                                               int                   batch_count);

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
                                               int                         batch_count);

// trsv_64
hipblasStatus_t hipblasCtrsvCast_64(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    hipblasOperation_t    transA,
                                    hipblasDiagType_t     diag,
                                    int64_t               m,
                                    const hipblasComplex* A,
                                    int64_t               lda,
                                    hipblasComplex*       x,
                                    int64_t               incx);

hipblasStatus_t hipblasZtrsvCast_64(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int64_t                     m,
                                    const hipblasDoubleComplex* A,
                                    int64_t                     lda,
                                    hipblasDoubleComplex*       x,
                                    int64_t                     incx);

hipblasStatus_t hipblasCtrsvBatchedCast_64(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int64_t                     m,
                                           const hipblasComplex* const A[],
                                           int64_t                     lda,
                                           hipblasComplex* const       x[],
                                           int64_t                     incx,
                                           int64_t                     batch_count);

hipblasStatus_t hipblasZtrsvBatchedCast_64(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           hipblasOperation_t                transA,
                                           hipblasDiagType_t                 diag,
                                           int64_t                           m,
                                           const hipblasDoubleComplex* const A[],
                                           int64_t                           lda,
                                           hipblasDoubleComplex* const       x[],
                                           int64_t                           incx,
                                           int64_t                           batch_count);

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
                                                  int64_t               batch_count);

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
                                                  int64_t                     batch_count);

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
                                 int                   incx);

hipblasStatus_t hipblasZtbmvCast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 hipblasOperation_t          transA,
                                 hipblasDiagType_t           diag,
                                 int                         m,
                                 int                         k,
                                 const hipblasDoubleComplex* A,
                                 int                         lda,
                                 hipblasDoubleComplex*       x,
                                 int                         incx);

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
                                        int                         batch_count);

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
                                        int                               batch_count);

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
                                               int                   batch_count);

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
                                               int                         batch_count);

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
                                    int64_t               incx);

hipblasStatus_t hipblasZtbmvCast_64(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int64_t                     m,
                                    int64_t                     k,
                                    const hipblasDoubleComplex* A,
                                    int64_t                     lda,
                                    hipblasDoubleComplex*       x,
                                    int64_t                     incx);

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
                                           int64_t                     batch_count);

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
                                           int64_t                           batch_count);

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
                                                  int64_t               batch_count);

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
                                                  int64_t                     batch_count);

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
                                 int                   incx);

hipblasStatus_t hipblasZtbsvCast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 hipblasOperation_t          transA,
                                 hipblasDiagType_t           diag,
                                 int                         n,
                                 int                         k,
                                 const hipblasDoubleComplex* A,
                                 int                         lda,
                                 hipblasDoubleComplex*       x,
                                 int                         incx);

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
                                        int                         batchCount);

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
                                        int                               batchCount);

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
                                               int                   batchCount);

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
                                               int                         batchCount);

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
                                    int64_t               incx);

hipblasStatus_t hipblasZtbsvCast_64(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int64_t                     n,
                                    int64_t                     k,
                                    const hipblasDoubleComplex* A,
                                    int64_t                     lda,
                                    hipblasDoubleComplex*       x,
                                    int64_t                     incx);

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
                                           int64_t                     batchCount);

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
                                           int64_t                           batchCount);

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
                                                  int64_t               batchCount);

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
                                                  int64_t                     batchCount);

// tpmv
hipblasStatus_t hipblasCtpmvCast(hipblasHandle_t       handle,
                                 hipblasFillMode_t     uplo,
                                 hipblasOperation_t    transA,
                                 hipblasDiagType_t     diag,
                                 int                   m,
                                 const hipblasComplex* AP,
                                 hipblasComplex*       x,
                                 int                   incx);

hipblasStatus_t hipblasZtpmvCast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 hipblasOperation_t          transA,
                                 hipblasDiagType_t           diag,
                                 int                         m,
                                 const hipblasDoubleComplex* AP,
                                 hipblasDoubleComplex*       x,
                                 int                         incx);

hipblasStatus_t hipblasCtpmvBatchedCast(hipblasHandle_t             handle,
                                        hipblasFillMode_t           uplo,
                                        hipblasOperation_t          transA,
                                        hipblasDiagType_t           diag,
                                        int                         m,
                                        const hipblasComplex* const AP[],
                                        hipblasComplex* const       x[],
                                        int                         incx,
                                        int                         batchCount);

hipblasStatus_t hipblasZtpmvBatchedCast(hipblasHandle_t                   handle,
                                        hipblasFillMode_t                 uplo,
                                        hipblasOperation_t                transA,
                                        hipblasDiagType_t                 diag,
                                        int                               m,
                                        const hipblasDoubleComplex* const AP[],
                                        hipblasDoubleComplex* const       x[],
                                        int                               incx,
                                        int                               batchCount);

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
                                               int                   batchCount);

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
                                               int                         batchCount);

// tpmv_64
hipblasStatus_t hipblasCtpmvCast_64(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    hipblasOperation_t    transA,
                                    hipblasDiagType_t     diag,
                                    int64_t               m,
                                    const hipblasComplex* AP,
                                    hipblasComplex*       x,
                                    int64_t               incx);

hipblasStatus_t hipblasZtpmvCast_64(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int64_t                     m,
                                    const hipblasDoubleComplex* AP,
                                    hipblasDoubleComplex*       x,
                                    int64_t                     incx);

hipblasStatus_t hipblasCtpmvBatchedCast_64(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int64_t                     m,
                                           const hipblasComplex* const AP[],
                                           hipblasComplex* const       x[],
                                           int64_t                     incx,
                                           int64_t                     batchCount);

hipblasStatus_t hipblasZtpmvBatchedCast_64(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           hipblasOperation_t                transA,
                                           hipblasDiagType_t                 diag,
                                           int64_t                           m,
                                           const hipblasDoubleComplex* const AP[],
                                           hipblasDoubleComplex* const       x[],
                                           int64_t                           incx,
                                           int64_t                           batchCount);

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
                                                  int64_t               batchCount);

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
                                                  int64_t                     batchCount);

// tpsv
hipblasStatus_t hipblasCtpsvCast(hipblasHandle_t       handle,
                                 hipblasFillMode_t     uplo,
                                 hipblasOperation_t    transA,
                                 hipblasDiagType_t     diag,
                                 int                   m,
                                 const hipblasComplex* AP,
                                 hipblasComplex*       x,
                                 int                   incx);

hipblasStatus_t hipblasZtpsvCast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 hipblasOperation_t          transA,
                                 hipblasDiagType_t           diag,
                                 int                         m,
                                 const hipblasDoubleComplex* AP,
                                 hipblasDoubleComplex*       x,
                                 int                         incx);

hipblasStatus_t hipblasCtpsvBatchedCast(hipblasHandle_t             handle,
                                        hipblasFillMode_t           uplo,
                                        hipblasOperation_t          transA,
                                        hipblasDiagType_t           diag,
                                        int                         m,
                                        const hipblasComplex* const AP[],
                                        hipblasComplex* const       x[],
                                        int                         incx,
                                        int                         batchCount);

hipblasStatus_t hipblasZtpsvBatchedCast(hipblasHandle_t                   handle,
                                        hipblasFillMode_t                 uplo,
                                        hipblasOperation_t                transA,
                                        hipblasDiagType_t                 diag,
                                        int                               m,
                                        const hipblasDoubleComplex* const AP[],
                                        hipblasDoubleComplex* const       x[],
                                        int                               incx,
                                        int                               batchCount);

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
                                               int                   batchCount);

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
                                               int                         batchCount);

// tpsv_64
hipblasStatus_t hipblasCtpsvCast_64(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    hipblasOperation_t    transA,
                                    hipblasDiagType_t     diag,
                                    int64_t               m,
                                    const hipblasComplex* AP,
                                    hipblasComplex*       x,
                                    int64_t               incx);

hipblasStatus_t hipblasZtpsvCast_64(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int64_t                     m,
                                    const hipblasDoubleComplex* AP,
                                    hipblasDoubleComplex*       x,
                                    int64_t                     incx);

hipblasStatus_t hipblasCtpsvBatchedCast_64(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int64_t                     m,
                                           const hipblasComplex* const AP[],
                                           hipblasComplex* const       x[],
                                           int64_t                     incx,
                                           int64_t                     batchCount);

hipblasStatus_t hipblasZtpsvBatchedCast_64(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           hipblasOperation_t                transA,
                                           hipblasDiagType_t                 diag,
                                           int64_t                           m,
                                           const hipblasDoubleComplex* const AP[],
                                           hipblasDoubleComplex* const       x[],
                                           int64_t                           incx,
                                           int64_t                           batchCount);

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
                                                  int64_t               batchCount);

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
                                                  int64_t                     batchCount);

// trmv
hipblasStatus_t hipblasCtrmvCast(hipblasHandle_t       handle,
                                 hipblasFillMode_t     uplo,
                                 hipblasOperation_t    transA,
                                 hipblasDiagType_t     diag,
                                 int                   m,
                                 const hipblasComplex* A,
                                 int                   lda,
                                 hipblasComplex*       x,
                                 int                   incx);

hipblasStatus_t hipblasZtrmvCast(hipblasHandle_t             handle,
                                 hipblasFillMode_t           uplo,
                                 hipblasOperation_t          transA,
                                 hipblasDiagType_t           diag,
                                 int                         m,
                                 const hipblasDoubleComplex* A,
                                 int                         lda,
                                 hipblasDoubleComplex*       x,
                                 int                         incx);

hipblasStatus_t hipblasCtrmvBatchedCast(hipblasHandle_t             handle,
                                        hipblasFillMode_t           uplo,
                                        hipblasOperation_t          transA,
                                        hipblasDiagType_t           diag,
                                        int                         m,
                                        const hipblasComplex* const A[],
                                        int                         lda,
                                        hipblasComplex* const       x[],
                                        int                         incx,
                                        int                         batch_count);

hipblasStatus_t hipblasZtrmvBatchedCast(hipblasHandle_t                   handle,
                                        hipblasFillMode_t                 uplo,
                                        hipblasOperation_t                transA,
                                        hipblasDiagType_t                 diag,
                                        int                               m,
                                        const hipblasDoubleComplex* const A[],
                                        int                               lda,
                                        hipblasDoubleComplex* const       x[],
                                        int                               incx,
                                        int                               batch_count);

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
                                               int                   batch_count);

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
                                               int                         batch_count);

// trmv_64
hipblasStatus_t hipblasCtrmvCast_64(hipblasHandle_t       handle,
                                    hipblasFillMode_t     uplo,
                                    hipblasOperation_t    transA,
                                    hipblasDiagType_t     diag,
                                    int64_t               m,
                                    const hipblasComplex* A,
                                    int64_t               lda,
                                    hipblasComplex*       x,
                                    int64_t               incx);

hipblasStatus_t hipblasZtrmvCast_64(hipblasHandle_t             handle,
                                    hipblasFillMode_t           uplo,
                                    hipblasOperation_t          transA,
                                    hipblasDiagType_t           diag,
                                    int64_t                     m,
                                    const hipblasDoubleComplex* A,
                                    int64_t                     lda,
                                    hipblasDoubleComplex*       x,
                                    int64_t                     incx);

hipblasStatus_t hipblasCtrmvBatchedCast_64(hipblasHandle_t             handle,
                                           hipblasFillMode_t           uplo,
                                           hipblasOperation_t          transA,
                                           hipblasDiagType_t           diag,
                                           int64_t                     m,
                                           const hipblasComplex* const A[],
                                           int64_t                     lda,
                                           hipblasComplex* const       x[],
                                           int64_t                     incx,
                                           int64_t                     batch_count);

hipblasStatus_t hipblasZtrmvBatchedCast_64(hipblasHandle_t                   handle,
                                           hipblasFillMode_t                 uplo,
                                           hipblasOperation_t                transA,
                                           hipblasDiagType_t                 diag,
                                           int64_t                           m,
                                           const hipblasDoubleComplex* const A[],
                                           int64_t                           lda,
                                           hipblasDoubleComplex* const       x[],
                                           int64_t                           incx,
                                           int64_t                           batch_count);

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
                                                  int64_t               batch_count);

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
                                                  int64_t                     batch_count);

// trtri
hipblasStatus_t hipblasCtrtriCast(hipblasHandle_t       handle,
                                  hipblasFillMode_t     uplo,
                                  hipblasDiagType_t     diag,
                                  int                   n,
                                  const hipblasComplex* A,
                                  int                   lda,
                                  hipblasComplex*       invA,
                                  int                   ldinvA);

hipblasStatus_t hipblasZtrtriCast(hipblasHandle_t             handle,
                                  hipblasFillMode_t           uplo,
                                  hipblasDiagType_t           diag,
                                  int                         n,
                                  const hipblasDoubleComplex* A,
                                  int                         lda,
                                  hipblasDoubleComplex*       invA,
                                  int                         ldinvA);
hipblasStatus_t hipblasCtrtriBatchedCast(hipblasHandle_t             handle,
                                         hipblasFillMode_t           uplo,
                                         hipblasDiagType_t           diag,
                                         int                         n,
                                         const hipblasComplex* const A[],
                                         int                         lda,
                                         hipblasComplex*             invA[],
                                         int                         ldinvA,
                                         int                         batch_count);

hipblasStatus_t hipblasZtrtriBatchedCast(hipblasHandle_t                   handle,
                                         hipblasFillMode_t                 uplo,
                                         hipblasDiagType_t                 diag,
                                         int                               n,
                                         const hipblasDoubleComplex* const A[],
                                         int                               lda,
                                         hipblasDoubleComplex*             invA[],
                                         int                               ldinvA,
                                         int                               batch_count);

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
                                                int                   batch_count);

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
                                                int                         batch_count);

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
                                 int                   ldc);

hipblasStatus_t hipblasZdgmmCast(hipblasHandle_t             handle,
                                 hipblasSideMode_t           side,
                                 int                         m,
                                 int                         n,
                                 const hipblasDoubleComplex* A,
                                 int                         lda,
                                 const hipblasDoubleComplex* x,
                                 int                         incx,
                                 hipblasDoubleComplex*       C,
                                 int                         ldc);

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
                                        int                         batch_count);

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
                                        int                               batch_count);

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
                                               int                   batch_count);

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
                                               int                         batch_count);

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
                                    int64_t               ldc);

hipblasStatus_t hipblasZdgmmCast_64(hipblasHandle_t             handle,
                                    hipblasSideMode_t           side,
                                    int64_t                     m,
                                    int64_t                     n,
                                    const hipblasDoubleComplex* A,
                                    int64_t                     lda,
                                    const hipblasDoubleComplex* x,
                                    int64_t                     incx,
                                    hipblasDoubleComplex*       C,
                                    int64_t                     ldc);

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
                                           int64_t                     batch_count);

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
                                           int64_t                           batch_count);

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
                                                  int64_t               batch_count);

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
                                                  int64_t                     batch_count);

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
                                 int                   ldc);

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
                                 int                         ldc);

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
                                        int                         batch_count);

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
                                        int                               batch_count);

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
                                               int                   batch_count);

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
                                               int                         batch_count);

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
                                    int64_t               ldc);

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
                                    int64_t                     ldc);

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
                                           int64_t                     batch_count);

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
                                           int64_t                           batch_count);

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
                                                  int64_t               batch_count);

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
                                                  int64_t                     batch_count);

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
                                 int                   ldc);

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
                                 int                         ldc);

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
                                        int                         batchCount);

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
                                        int                               batchCount);

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
                                               int                   batchCount);

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
                                               int                         batchCount);

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
                                    int64_t               ldc);

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
                                    int64_t                     ldc);

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
                                           int64_t                     batchCount);

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
                                           int64_t                           batchCount);

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
                                                  int64_t               batchCount);

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
                                                  int64_t                     batchCount);

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
                                  int                   ldc);

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
                                  int                         ldc);

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
                                         int                         batchCount);

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
                                         int                               batchCount);

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
                                                int                   batchCount);

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
                                                int                         batchCount);

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
                                     int64_t               ldc);

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
                                     int64_t                     ldc);

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
                                            int64_t                     batchCount);

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
                                            int64_t                           batchCount);

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
                                                   int64_t               batchCount);

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
                                                   int64_t                     batchCount);

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
                                  int                   ldc);

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
                                  int                         ldc);

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
                                         int                         batchCount);

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
                                         int                               batchCount);

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
                                                int                   batchCount);

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
                                                int                         batchCount);

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
                                     int64_t               ldc);

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
                                     int64_t                     ldc);

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
                                            int64_t                     batchCount);

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
                                            int64_t                           batchCount);

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
                                                   int64_t               batchCount);

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
                                                   int64_t                     batchCount);

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
                                 int                   ldc);

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
                                 int                         ldc);

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
                                        int                         batchCount);

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
                                        int                               batchCount);

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
                                               int                   batchCount);

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
                                               int                         batchCount);

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
                                    int64_t               ldc);

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
                                    int64_t                     ldc);

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
                                           int64_t                     batchCount);

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
                                           int64_t                           batchCount);

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
                                                  int64_t               batchCount);

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
                                                  int64_t                     batchCount);

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
                                 int                   ldc);

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
                                 int                         ldc);

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
                                        int                         batchCount);

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
                                        int                               batchCount);

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
                                               int                   batchCount);

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
                                               int                         batchCount);

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
                                    int64_t               ldc);

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
                                    int64_t                     ldc);

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
                                           int64_t                     batchCount);

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
                                           int64_t                           batchCount);

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
                                                  int64_t               batchCount);

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
                                                  int64_t                     batchCount);

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
                                  int                   ldc);

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
                                  int                         ldc);

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
                                         int                         batchCount);

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
                                         int                               batchCount);

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
                                                int                   batchCount);

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
                                                int                         batchCount);

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
                                     int64_t               ldc);

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
                                     int64_t                     ldc);

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
                                            int64_t                     batchCount);

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
                                            int64_t                           batchCount);

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
                                                   int64_t               batchCount);

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
                                                   int64_t                     batchCount);

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
                                  int                   ldc);

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
                                  int                         ldc);

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
                                         int                         batchCount);

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
                                         int                               batchCount);

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
                                                int                   batchCount);

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
                                                int                         batchCount);

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
                                     int64_t               ldc);

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
                                     int64_t                     ldc);

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
                                            int64_t                     batchCount);

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
                                            int64_t                           batchCount);

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
                                                   int64_t               batchCount);

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
                                                   int64_t                     batchCount);

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
                                 int                   ldc);

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
                                 int                         ldc);

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
                                        int                         batchCount);

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
                                        int                               batchCount);

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
                                               int                   batchCount);

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
                                               int                         batchCount);

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
                                    int64_t               ldc);

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
                                    int64_t                     ldc);

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
                                           int64_t                     batchCount);

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
                                           int64_t                           batchCount);

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
                                                  int64_t               batchCount);

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
                                                  int64_t                     batchCount);

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
                                 int                   ldc);

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
                                 int                         ldc);

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
                                        int                         batchCount);

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
                                        int                               batchCount);

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
                                               int                   batchCount);

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
                                               int                         batchCount);

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
                                    int64_t               ldc);

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
                                    int64_t                     ldc);

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
                                           int64_t                     batchCount);

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
                                           int64_t                           batchCount);

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
                                                  int64_t               batchCount);

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
                                                  int64_t                     batchCount);

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
                                 int                   ldb);

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
                                 int                         ldb);

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
                                        int                         batch_count);

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
                                        int                               batch_count);

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
                                               int                   batch_count);

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
                                               int                         batch_count);

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
                                    int64_t               ldb);

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
                                    int64_t                     ldb);

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
                                           int64_t                     batch_count);

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
                                           int64_t                           batch_count);

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
                                                  int64_t               batch_count);

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
                                                  int64_t                     batch_count);

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
                                 int                   ldc);

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
                                 int                         ldc);

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
                                        int                         batchCount);

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
                                        int                               batchCount);

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
                                               int                   batchCount);

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
                                               int                         batchCount);

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
                                    int64_t               ldc);

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
                                    int64_t                     ldc);

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
                                           int64_t                     batchCount);

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
                                           int64_t                           batchCount);

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
                                                  int64_t               batchCount);

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
                                                  int64_t                     batchCount);

// getrf
hipblasStatus_t hipblasCgetrfCast(
    hipblasHandle_t handle, const int n, hipblasComplex* A, const int lda, int* ipiv, int* info);

hipblasStatus_t hipblasZgetrfCast(hipblasHandle_t       handle,
                                  const int             n,
                                  hipblasDoubleComplex* A,
                                  const int             lda,
                                  int*                  ipiv,
                                  int*                  info);

hipblasStatus_t hipblasCgetrfBatchedCast(hipblasHandle_t       handle,
                                         const int             n,
                                         hipblasComplex* const A[],
                                         const int             lda,
                                         int*                  ipiv,
                                         int*                  info,
                                         const int             batchCount);

hipblasStatus_t hipblasZgetrfBatchedCast(hipblasHandle_t             handle,
                                         const int                   n,
                                         hipblasDoubleComplex* const A[],
                                         const int                   lda,
                                         int*                        ipiv,
                                         int*                        info,
                                         const int                   batchCount);

hipblasStatus_t hipblasCgetrfStridedBatchedCast(hipblasHandle_t     handle,
                                                const int           n,
                                                hipblasComplex*     A,
                                                const int           lda,
                                                const hipblasStride strideA,
                                                int*                ipiv,
                                                const hipblasStride strideP,
                                                int*                info,
                                                const int           batchCount);

hipblasStatus_t hipblasZgetrfStridedBatchedCast(hipblasHandle_t       handle,
                                                const int             n,
                                                hipblasDoubleComplex* A,
                                                const int             lda,
                                                const hipblasStride   strideA,
                                                int*                  ipiv,
                                                const hipblasStride   strideP,
                                                int*                  info,
                                                const int             batchCount);

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
                                  int*                     info);

hipblasStatus_t hipblasZgetrsCast(hipblasHandle_t          handle,
                                  const hipblasOperation_t trans,
                                  const int                n,
                                  const int                nrhs,
                                  hipblasDoubleComplex*    A,
                                  const int                lda,
                                  const int*               ipiv,
                                  hipblasDoubleComplex*    B,
                                  const int                ldb,
                                  int*                     info);

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
                                         const int                batchCount);

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
                                         const int                   batchCount);

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
                                                const int                batchCount);

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
                                                const int                batchCount);

hipblasStatus_t hipblasCgetriBatchedCast(hipblasHandle_t       handle,
                                         const int             n,
                                         hipblasComplex* const A[],
                                         const int             lda,
                                         int*                  ipiv,
                                         hipblasComplex* const C[],
                                         const int             ldc,
                                         int*                  info,
                                         const int             batchCount);

hipblasStatus_t hipblasZgetriBatchedCast(hipblasHandle_t             handle,
                                         const int                   n,
                                         hipblasDoubleComplex* const A[],
                                         const int                   lda,
                                         int*                        ipiv,
                                         hipblasDoubleComplex* const C[],
                                         const int                   ldc,
                                         int*                        info,
                                         const int                   batchCount);

// geqrf
hipblasStatus_t hipblasCgeqrfCast(hipblasHandle_t handle,
                                  const int       m,
                                  const int       n,
                                  hipblasComplex* A,
                                  const int       lda,
                                  hipblasComplex* ipiv,
                                  int*            info);

hipblasStatus_t hipblasZgeqrfCast(hipblasHandle_t       handle,
                                  const int             m,
                                  const int             n,
                                  hipblasDoubleComplex* A,
                                  const int             lda,
                                  hipblasDoubleComplex* ipiv,
                                  int*                  info);

hipblasStatus_t hipblasCgeqrfBatchedCast(hipblasHandle_t       handle,
                                         const int             m,
                                         const int             n,
                                         hipblasComplex* const A[],
                                         const int             lda,
                                         hipblasComplex* const ipiv[],
                                         int*                  info,
                                         const int             batchCount);

hipblasStatus_t hipblasZgeqrfBatchedCast(hipblasHandle_t             handle,
                                         const int                   m,
                                         const int                   n,
                                         hipblasDoubleComplex* const A[],
                                         const int                   lda,
                                         hipblasDoubleComplex* const ipiv[],
                                         int*                        info,
                                         const int                   batchCount);

hipblasStatus_t hipblasCgeqrfStridedBatchedCast(hipblasHandle_t     handle,
                                                const int           m,
                                                const int           n,
                                                hipblasComplex*     A,
                                                const int           lda,
                                                const hipblasStride strideA,
                                                hipblasComplex*     ipiv,
                                                const hipblasStride strideP,
                                                int*                info,
                                                const int           batchCount);

hipblasStatus_t hipblasZgeqrfStridedBatchedCast(hipblasHandle_t       handle,
                                                const int             m,
                                                const int             n,
                                                hipblasDoubleComplex* A,
                                                const int             lda,
                                                const hipblasStride   strideA,
                                                hipblasDoubleComplex* ipiv,
                                                const hipblasStride   strideP,
                                                int*                  info,
                                                const int             batchCount);

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
                                 int*               deviceInfo);

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
                                 int*                  deviceInfo);

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
                                        const int             batchCount);

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
                                        const int                   batchCount);

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
                                               const int           batchCount);

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
                                               const int             batchCount);

#endif

namespace
{
    // Scal
    template <typename T, typename U = T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasScal)(hipblasHandle_t handle, int n, const U* alpha, T* x, int incx);

    template <typename T, typename U = T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasScalBatched)(
        hipblasHandle_t handle, int n, const U* alpha, T* const x[], int incx, int batch_count);

    template <typename T, typename U = T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasScalStridedBatched)(hipblasHandle_t handle,
                                                 int             n,
                                                 const U*        alpha,
                                                 T*              x,
                                                 int             incx,
                                                 hipblasStride   stridex,
                                                 int             batch_count);

    template <typename T, typename U = T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasScal_64)(
        hipblasHandle_t handle, int64_t n, const U* alpha, T* x, int64_t incx);

    template <typename T, typename U = T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasScalBatched_64)(hipblasHandle_t handle,
                                             int64_t         n,
                                             const U*        alpha,
                                             T* const        x[],
                                             int64_t         incx,
                                             int64_t         batch_count);

    template <typename T, typename U = T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasScalStridedBatched_64)(hipblasHandle_t handle,
                                                    int64_t         n,
                                                    const U*        alpha,
                                                    T*              x,
                                                    int64_t         incx,
                                                    hipblasStride   stridex,
                                                    int64_t         batch_count);

    MAP2CF_D64(hipblasScal, float, float, hipblasSscal);
    MAP2CF_D64(hipblasScal, double, double, hipblasDscal);
    MAP2CF_D64_V2(hipblasScal, hipblasComplex, hipblasComplex, hipblasCscal);
    MAP2CF_D64_V2(hipblasScal, hipblasDoubleComplex, hipblasDoubleComplex, hipblasZscal);
    MAP2CF_D64_V2(hipblasScal, hipblasComplex, float, hipblasCsscal);
    MAP2CF_D64_V2(hipblasScal, hipblasDoubleComplex, double, hipblasZdscal);

    MAP2CF_D64(hipblasScalBatched, float, float, hipblasSscalBatched);
    MAP2CF_D64(hipblasScalBatched, double, double, hipblasDscalBatched);
    MAP2CF_D64_V2(hipblasScalBatched, hipblasComplex, hipblasComplex, hipblasCscalBatched);
    MAP2CF_D64_V2(hipblasScalBatched,
                  hipblasDoubleComplex,
                  hipblasDoubleComplex,
                  hipblasZscalBatched);
    MAP2CF_D64_V2(hipblasScalBatched, hipblasComplex, float, hipblasCsscalBatched);
    MAP2CF_D64_V2(hipblasScalBatched, hipblasDoubleComplex, double, hipblasZdscalBatched);

    MAP2CF_D64(hipblasScalStridedBatched, float, float, hipblasSscalStridedBatched);
    MAP2CF_D64(hipblasScalStridedBatched, double, double, hipblasDscalStridedBatched);
    MAP2CF_D64_V2(hipblasScalStridedBatched,
                  hipblasComplex,
                  hipblasComplex,
                  hipblasCscalStridedBatched);
    MAP2CF_D64_V2(hipblasScalStridedBatched,
                  hipblasDoubleComplex,
                  hipblasDoubleComplex,
                  hipblasZscalStridedBatched);
    MAP2CF_D64_V2(hipblasScalStridedBatched, hipblasComplex, float, hipblasCsscalStridedBatched);
    MAP2CF_D64_V2(hipblasScalStridedBatched,
                  hipblasDoubleComplex,
                  double,
                  hipblasZdscalStridedBatched);

    // Copy
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasCopy)(
        hipblasHandle_t handle, int n, const T* x, int incx, T* y, int incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasCopyBatched)(hipblasHandle_t handle,
                                          int             n,
                                          const T* const  x[],
                                          int             incx,
                                          T* const        y[],
                                          int             incy,
                                          int             batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasCopyStridedBatched)(hipblasHandle_t handle,
                                                 int             n,
                                                 const T*        x,
                                                 int             incx,
                                                 hipblasStride   stridex,
                                                 T*              y,
                                                 int             incy,
                                                 hipblasStride   stridey,
                                                 int             batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasCopy_64)(
        hipblasHandle_t handle, int64_t n, const T* x, int64_t incx, T* y, int64_t incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasCopyBatched_64)(hipblasHandle_t handle,
                                             int64_t         n,
                                             const T* const  x[],
                                             int64_t         incx,
                                             T* const        y[],
                                             int64_t         incy,
                                             int64_t         batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasCopyStridedBatched_64)(hipblasHandle_t handle,
                                                    int64_t         n,
                                                    const T*        x,
                                                    int64_t         incx,
                                                    hipblasStride   stridex,
                                                    T*              y,
                                                    int64_t         incy,
                                                    hipblasStride   stridey,
                                                    int64_t         batch_count);

    MAP2CF_D64(hipblasCopy, float, hipblasScopy);
    MAP2CF_D64(hipblasCopy, double, hipblasDcopy);
    MAP2CF_D64_V2(hipblasCopy, hipblasComplex, hipblasCcopy);
    MAP2CF_D64_V2(hipblasCopy, hipblasDoubleComplex, hipblasZcopy);

    MAP2CF_D64(hipblasCopyBatched, float, hipblasScopyBatched);
    MAP2CF_D64(hipblasCopyBatched, double, hipblasDcopyBatched);
    MAP2CF_D64_V2(hipblasCopyBatched, hipblasComplex, hipblasCcopyBatched);
    MAP2CF_D64_V2(hipblasCopyBatched, hipblasDoubleComplex, hipblasZcopyBatched);

    MAP2CF_D64(hipblasCopyStridedBatched, float, hipblasScopyStridedBatched);
    MAP2CF_D64(hipblasCopyStridedBatched, double, hipblasDcopyStridedBatched);
    MAP2CF_D64_V2(hipblasCopyStridedBatched, hipblasComplex, hipblasCcopyStridedBatched);
    MAP2CF_D64_V2(hipblasCopyStridedBatched, hipblasDoubleComplex, hipblasZcopyStridedBatched);

    // Swap
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSwap)(hipblasHandle_t handle, int n, T* x, int incx, T* y, int incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSwapBatched)(hipblasHandle_t handle,
                                          int             n,
                                          T* const        x[],
                                          int             incx,
                                          T* const        y[],
                                          int             incy,
                                          int             batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSwapStridedBatched)(hipblasHandle_t handle,
                                                 int             n,
                                                 T*              x,
                                                 int             incx,
                                                 hipblasStride   stridex,
                                                 T*              y,
                                                 int             incy,
                                                 hipblasStride   stridey,
                                                 int             batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSwap_64)(
        hipblasHandle_t handle, int64_t n, T* x, int64_t incx, T* y, int64_t incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSwapBatched_64)(hipblasHandle_t handle,
                                             int64_t         n,
                                             T* const        x[],
                                             int64_t         incx,
                                             T* const        y[],
                                             int64_t         incy,
                                             int64_t         batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSwapStridedBatched_64)(hipblasHandle_t handle,
                                                    int64_t         n,
                                                    T*              x,
                                                    int64_t         incx,
                                                    hipblasStride   stridex,
                                                    T*              y,
                                                    int64_t         incy,
                                                    hipblasStride   stridey,
                                                    int64_t         batch_count);

    MAP2CF_D64(hipblasSwap, float, hipblasSswap);
    MAP2CF_D64(hipblasSwap, double, hipblasDswap);
    MAP2CF_D64_V2(hipblasSwap, hipblasComplex, hipblasCswap);
    MAP2CF_D64_V2(hipblasSwap, hipblasDoubleComplex, hipblasZswap);

    MAP2CF_D64(hipblasSwapBatched, float, hipblasSswapBatched);
    MAP2CF_D64(hipblasSwapBatched, double, hipblasDswapBatched);
    MAP2CF_D64_V2(hipblasSwapBatched, hipblasComplex, hipblasCswapBatched);
    MAP2CF_D64_V2(hipblasSwapBatched, hipblasDoubleComplex, hipblasZswapBatched);

    MAP2CF_D64(hipblasSwapStridedBatched, float, hipblasSswapStridedBatched);
    MAP2CF_D64(hipblasSwapStridedBatched, double, hipblasDswapStridedBatched);
    MAP2CF_D64_V2(hipblasSwapStridedBatched, hipblasComplex, hipblasCswapStridedBatched);
    MAP2CF_D64_V2(hipblasSwapStridedBatched, hipblasDoubleComplex, hipblasZswapStridedBatched);

    // Dot
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDot)(
        hipblasHandle_t handle, int n, const T* x, int incx, const T* y, int incy, T* result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDotc)(
        hipblasHandle_t handle, int n, const T* x, int incx, const T* y, int incy, T* result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDotBatched)(hipblasHandle_t handle,
                                         int             n,
                                         const T* const  x[],
                                         int             incx,
                                         const T* const  y[],
                                         int             incy,
                                         int             batch_count,
                                         T*              result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDotcBatched)(hipblasHandle_t handle,
                                          int             n,
                                          const T* const  x[],
                                          int             incx,
                                          const T* const  y[],
                                          int             incy,
                                          int             batch_count,
                                          T*              result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDotStridedBatched)(hipblasHandle_t handle,
                                                int             n,
                                                const T*        x,
                                                int             incx,
                                                hipblasStride   stridex,
                                                const T*        y,
                                                int             incy,
                                                hipblasStride   stridey,
                                                int             batch_count,
                                                T*              result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDotcStridedBatched)(hipblasHandle_t handle,
                                                 int             n,
                                                 const T*        x,
                                                 int             incx,
                                                 hipblasStride   stridex,
                                                 const T*        y,
                                                 int             incy,
                                                 hipblasStride   stridey,
                                                 int             batch_count,
                                                 T*              result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDot_64)(hipblasHandle_t handle,
                                     int64_t         n,
                                     const T*        x,
                                     int64_t         incx,
                                     const T*        y,
                                     int64_t         incy,
                                     T*              result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDotc_64)(hipblasHandle_t handle,
                                      int64_t         n,
                                      const T*        x,
                                      int64_t         incx,
                                      const T*        y,
                                      int64_t         incy,
                                      T*              result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDotBatched_64)(hipblasHandle_t handle,
                                            int64_t         n,
                                            const T* const  x[],
                                            int64_t         incx,
                                            const T* const  y[],
                                            int64_t         incy,
                                            int64_t         batch_count,
                                            T*              result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDotcBatched_64)(hipblasHandle_t handle,
                                             int64_t         n,
                                             const T* const  x[],
                                             int64_t         incx,
                                             const T* const  y[],
                                             int64_t         incy,
                                             int64_t         batch_count,
                                             T*              result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDotStridedBatched_64)(hipblasHandle_t handle,
                                                   int64_t         n,
                                                   const T*        x,
                                                   int64_t         incx,
                                                   hipblasStride   stridex,
                                                   const T*        y,
                                                   int64_t         incy,
                                                   hipblasStride   stridey,
                                                   int64_t         batch_count,
                                                   T*              result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDotcStridedBatched_64)(hipblasHandle_t handle,
                                                    int64_t         n,
                                                    const T*        x,
                                                    int64_t         incx,
                                                    hipblasStride   stridex,
                                                    const T*        y,
                                                    int64_t         incy,
                                                    hipblasStride   stridey,
                                                    int64_t         batch_count,
                                                    T*              result);

    MAP2CF_D64(hipblasDot, hipblasHalf, hipblasHdot);
    MAP2CF_D64(hipblasDot, hipblasBfloat16, hipblasBfdot);
    MAP2CF_D64(hipblasDot, float, hipblasSdot);
    MAP2CF_D64(hipblasDot, double, hipblasDdot);
    MAP2CF_D64_V2(hipblasDot, hipblasComplex, hipblasCdotu);
    MAP2CF_D64_V2(hipblasDot, hipblasDoubleComplex, hipblasZdotu);
    MAP2CF_D64_V2(hipblasDotc, hipblasComplex, hipblasCdotc);
    MAP2CF_D64_V2(hipblasDotc, hipblasDoubleComplex, hipblasZdotc);

    MAP2CF_D64(hipblasDotBatched, hipblasHalf, hipblasHdotBatched);
    MAP2CF_D64(hipblasDotBatched, hipblasBfloat16, hipblasBfdotBatched);
    MAP2CF_D64(hipblasDotBatched, float, hipblasSdotBatched);
    MAP2CF_D64(hipblasDotBatched, double, hipblasDdotBatched);
    MAP2CF_D64_V2(hipblasDotBatched, hipblasComplex, hipblasCdotuBatched);
    MAP2CF_D64_V2(hipblasDotBatched, hipblasDoubleComplex, hipblasZdotuBatched);
    MAP2CF_D64_V2(hipblasDotcBatched, hipblasComplex, hipblasCdotcBatched);
    MAP2CF_D64_V2(hipblasDotcBatched, hipblasDoubleComplex, hipblasZdotcBatched);

    MAP2CF_D64(hipblasDotStridedBatched, hipblasHalf, hipblasHdotStridedBatched);
    MAP2CF_D64(hipblasDotStridedBatched, hipblasBfloat16, hipblasBfdotStridedBatched);
    MAP2CF_D64(hipblasDotStridedBatched, float, hipblasSdotStridedBatched);
    MAP2CF_D64(hipblasDotStridedBatched, double, hipblasDdotStridedBatched);
    MAP2CF_D64_V2(hipblasDotStridedBatched, hipblasComplex, hipblasCdotuStridedBatched);
    MAP2CF_D64_V2(hipblasDotStridedBatched, hipblasDoubleComplex, hipblasZdotuStridedBatched);
    MAP2CF_D64_V2(hipblasDotcStridedBatched, hipblasComplex, hipblasCdotcStridedBatched);
    MAP2CF_D64_V2(hipblasDotcStridedBatched, hipblasDoubleComplex, hipblasZdotcStridedBatched);

    // Asum
    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAsum)(
        hipblasHandle_t handle, int n, const T1* x, int incx, T2* result);

    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAsumBatched)(
        hipblasHandle_t handle, int n, const T1* const x[], int incx, int batch_count, T2* result);

    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAsumStridedBatched)(hipblasHandle_t handle,
                                                 int             n,
                                                 const T1*       x,
                                                 int             incx,
                                                 hipblasStride   stridex,
                                                 int             batch_count,
                                                 T2*             result);

    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAsum_64)(
        hipblasHandle_t handle, int64_t n, const T1* x, int64_t incx, T2* result);

    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAsumBatched_64)(hipblasHandle_t handle,
                                             int64_t         n,
                                             const T1* const x[],
                                             int64_t         incx,
                                             int64_t         batch_count,
                                             T2*             result);

    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAsumStridedBatched_64)(hipblasHandle_t handle,
                                                    int64_t         n,
                                                    const T1*       x,
                                                    int64_t         incx,
                                                    hipblasStride   stridex,
                                                    int64_t         batch_count,
                                                    T2*             result);

    MAP2CF_D64(hipblasAsum, float, float, hipblasSasum);
    MAP2CF_D64(hipblasAsum, double, double, hipblasDasum);
    MAP2CF_D64_V2(hipblasAsum, hipblasComplex, float, hipblasScasum);
    MAP2CF_D64_V2(hipblasAsum, hipblasDoubleComplex, double, hipblasDzasum);

    MAP2CF_D64(hipblasAsumBatched, float, float, hipblasSasumBatched);
    MAP2CF_D64(hipblasAsumBatched, double, double, hipblasDasumBatched);
    MAP2CF_D64_V2(hipblasAsumBatched, hipblasComplex, float, hipblasScasumBatched);
    MAP2CF_D64_V2(hipblasAsumBatched, hipblasDoubleComplex, double, hipblasDzasumBatched);

    MAP2CF_D64(hipblasAsumStridedBatched, float, float, hipblasSasumStridedBatched);
    MAP2CF_D64(hipblasAsumStridedBatched, double, double, hipblasDasumStridedBatched);
    MAP2CF_D64_V2(hipblasAsumStridedBatched, hipblasComplex, float, hipblasScasumStridedBatched);
    MAP2CF_D64_V2(hipblasAsumStridedBatched,
                  hipblasDoubleComplex,
                  double,
                  hipblasDzasumStridedBatched);

    // nrm2
    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasNrm2)(
        hipblasHandle_t handle, int n, const T1* x, int incx, T2* result);

    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasNrm2Batched)(
        hipblasHandle_t handle, int n, const T1* const x[], int incx, int batch_count, T2* result);

    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasNrm2StridedBatched)(hipblasHandle_t handle,
                                                 int             n,
                                                 const T1*       x,
                                                 int             incx,
                                                 hipblasStride   stridex,
                                                 int             batch_count,
                                                 T2*             result);

    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasNrm2_64)(
        hipblasHandle_t handle, int64_t n, const T1* x, int64_t incx, T2* result);

    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasNrm2Batched_64)(hipblasHandle_t handle,
                                             int64_t         n,
                                             const T1* const x[],
                                             int64_t         incx,
                                             int64_t         batch_count,
                                             T2*             result);

    template <typename T1, typename T2, bool FORTRAN = false>
    hipblasStatus_t (*hipblasNrm2StridedBatched_64)(hipblasHandle_t handle,
                                                    int64_t         n,
                                                    const T1*       x,
                                                    int64_t         incx,
                                                    hipblasStride   stridex,
                                                    int64_t         batch_count,
                                                    T2*             result);

    MAP2CF_D64(hipblasNrm2, float, float, hipblasSnrm2);
    MAP2CF_D64(hipblasNrm2, double, double, hipblasDnrm2);
    MAP2CF_D64_V2(hipblasNrm2, hipblasComplex, float, hipblasScnrm2);
    MAP2CF_D64_V2(hipblasNrm2, hipblasDoubleComplex, double, hipblasDznrm2);

    MAP2CF_D64(hipblasNrm2Batched, float, float, hipblasSnrm2Batched);
    MAP2CF_D64(hipblasNrm2Batched, double, double, hipblasDnrm2Batched);
    MAP2CF_D64_V2(hipblasNrm2Batched, hipblasComplex, float, hipblasScnrm2Batched);
    MAP2CF_D64_V2(hipblasNrm2Batched, hipblasDoubleComplex, double, hipblasDznrm2Batched);

    MAP2CF_D64(hipblasNrm2StridedBatched, float, float, hipblasSnrm2StridedBatched);
    MAP2CF_D64(hipblasNrm2StridedBatched, double, double, hipblasDnrm2StridedBatched);
    MAP2CF_D64_V2(hipblasNrm2StridedBatched, hipblasComplex, float, hipblasScnrm2StridedBatched);
    MAP2CF_D64_V2(hipblasNrm2StridedBatched,
                  hipblasDoubleComplex,
                  double,
                  hipblasDznrm2StridedBatched);

    // Rot
    template <typename T1, typename T2, typename T3 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRot)(
        hipblasHandle_t handle, int n, T1* x, int incx, T1* y, int incy, const T2* c, const T3* s);

    template <typename T1, typename T2 = T1, typename T3 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotBatched)(hipblasHandle_t handle,
                                         int             n,
                                         T1* const       x[],
                                         int             incx,
                                         T1* const       y[],
                                         int             incy,
                                         const T2*       c,
                                         const T3*       s,
                                         int             batch_count);

    template <typename T1, typename T2 = T1, typename T3 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotStridedBatched)(hipblasHandle_t handle,
                                                int             n,
                                                T1*             x,
                                                int             incx,
                                                hipblasStride   stridex,
                                                T1*             y,
                                                int             incy,
                                                hipblasStride   stridey,
                                                const T2*       c,
                                                const T3*       s,
                                                int             batch_count);

    template <typename T1, typename T2, typename T3 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRot_64)(hipblasHandle_t handle,
                                     int64_t         n,
                                     T1*             x,
                                     int64_t         incx,
                                     T1*             y,
                                     int64_t         incy,
                                     const T2*       c,
                                     const T3*       s);

    template <typename T1, typename T2 = T1, typename T3 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotBatched_64)(hipblasHandle_t handle,
                                            int64_t         n,
                                            T1* const       x[],
                                            int64_t         incx,
                                            T1* const       y[],
                                            int64_t         incy,
                                            const T2*       c,
                                            const T3*       s,
                                            int64_t         batch_count);

    template <typename T1, typename T2 = T1, typename T3 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotStridedBatched_64)(hipblasHandle_t handle,
                                                   int64_t         n,
                                                   T1*             x,
                                                   int64_t         incx,
                                                   hipblasStride   stridex,
                                                   T1*             y,
                                                   int64_t         incy,
                                                   hipblasStride   stridey,
                                                   const T2*       c,
                                                   const T3*       s,
                                                   int64_t         batch_count);

    MAP2CF_D64(hipblasRot, float, float, float, hipblasSrot);
    MAP2CF_D64(hipblasRot, double, double, double, hipblasDrot);
    MAP2CF_D64_V2(hipblasRot, hipblasComplex, float, hipblasComplex, hipblasCrot);
    MAP2CF_D64_V2(hipblasRot, hipblasDoubleComplex, double, hipblasDoubleComplex, hipblasZrot);
    MAP2CF_D64_V2(hipblasRot, hipblasComplex, float, float, hipblasCsrot);
    MAP2CF_D64_V2(hipblasRot, hipblasDoubleComplex, double, double, hipblasZdrot);

    MAP2CF_D64(hipblasRotBatched, float, float, float, hipblasSrotBatched);
    MAP2CF_D64(hipblasRotBatched, double, double, double, hipblasDrotBatched);
    MAP2CF_D64_V2(hipblasRotBatched, hipblasComplex, float, hipblasComplex, hipblasCrotBatched);
    MAP2CF_D64_V2(
        hipblasRotBatched, hipblasDoubleComplex, double, hipblasDoubleComplex, hipblasZrotBatched);
    MAP2CF_D64_V2(hipblasRotBatched, hipblasComplex, float, float, hipblasCsrotBatched);
    MAP2CF_D64_V2(hipblasRotBatched, hipblasDoubleComplex, double, double, hipblasZdrotBatched);

    MAP2CF_D64(hipblasRotStridedBatched, float, float, float, hipblasSrotStridedBatched);
    MAP2CF_D64(hipblasRotStridedBatched, double, double, double, hipblasDrotStridedBatched);
    MAP2CF_D64_V2(
        hipblasRotStridedBatched, hipblasComplex, float, hipblasComplex, hipblasCrotStridedBatched);
    MAP2CF_D64_V2(hipblasRotStridedBatched,
                  hipblasDoubleComplex,
                  double,
                  hipblasDoubleComplex,
                  hipblasZrotStridedBatched);
    MAP2CF_D64_V2(
        hipblasRotStridedBatched, hipblasComplex, float, float, hipblasCsrotStridedBatched);
    MAP2CF_D64_V2(
        hipblasRotStridedBatched, hipblasDoubleComplex, double, double, hipblasZdrotStridedBatched);

    // Rotg
    template <typename T1, typename T2 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotg)(hipblasHandle_t handle, T1* a, T1* b, T2* c, T1* s);

    template <typename T1, typename T2 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotgBatched)(hipblasHandle_t handle,
                                          T1* const       a[],
                                          T1* const       b[],
                                          T2* const       c[],
                                          T1* const       s[],
                                          int             batch_count);

    template <typename T1, typename T2 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotgStridedBatched)(hipblasHandle_t handle,
                                                 T1*             a,
                                                 hipblasStride   stridea,
                                                 T1*             b,
                                                 hipblasStride   strideb,
                                                 T2*             c,
                                                 hipblasStride   stridec,
                                                 T1*             s,
                                                 hipblasStride   strides,
                                                 int             batch_count);

    template <typename T1, typename T2 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotg_64)(hipblasHandle_t handle, T1* a, T1* b, T2* c, T1* s);

    template <typename T1, typename T2 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotgBatched_64)(hipblasHandle_t handle,
                                             T1* const       a[],
                                             T1* const       b[],
                                             T2* const       c[],
                                             T1* const       s[],
                                             int64_t         batch_count);

    template <typename T1, typename T2 = T1, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotgStridedBatched_64)(hipblasHandle_t handle,
                                                    T1*             a,
                                                    hipblasStride   stridea,
                                                    T1*             b,
                                                    hipblasStride   strideb,
                                                    T2*             c,
                                                    hipblasStride   stridec,
                                                    T1*             s,
                                                    hipblasStride   strides,
                                                    int64_t         batch_count);

    MAP2CF_D64(hipblasRotg, float, float, hipblasSrotg);
    MAP2CF_D64(hipblasRotg, double, double, hipblasDrotg);
    MAP2CF_D64_V2(hipblasRotg, hipblasComplex, float, hipblasCrotg);
    MAP2CF_D64_V2(hipblasRotg, hipblasDoubleComplex, double, hipblasZrotg);

    MAP2CF_D64(hipblasRotgBatched, float, float, hipblasSrotgBatched);
    MAP2CF_D64(hipblasRotgBatched, double, double, hipblasDrotgBatched);
    MAP2CF_D64_V2(hipblasRotgBatched, hipblasComplex, float, hipblasCrotgBatched);
    MAP2CF_D64_V2(hipblasRotgBatched, hipblasDoubleComplex, double, hipblasZrotgBatched);

    MAP2CF_D64(hipblasRotgStridedBatched, float, float, hipblasSrotgStridedBatched);
    MAP2CF_D64(hipblasRotgStridedBatched, double, double, hipblasDrotgStridedBatched);
    MAP2CF_D64_V2(hipblasRotgStridedBatched, hipblasComplex, float, hipblasCrotgStridedBatched);
    MAP2CF_D64_V2(hipblasRotgStridedBatched,
                  hipblasDoubleComplex,
                  double,
                  hipblasZrotgStridedBatched);

    // rotm
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotm)(
        hipblasHandle_t handle, int n, T* x, int incx, T* y, int incy, const T* param);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotmBatched)(hipblasHandle_t handle,
                                          int             n,
                                          T* const        x[],
                                          int             incx,
                                          T* const        y[],
                                          int             incy,
                                          const T* const  param[],
                                          int             batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotmStridedBatched)(hipblasHandle_t handle,
                                                 int             n,
                                                 T*              x,
                                                 int             incx,
                                                 hipblasStride   stridex,
                                                 T*              y,
                                                 int             incy,
                                                 hipblasStride   stridey,
                                                 const T*        param,
                                                 hipblasStride   strideparam,
                                                 int             batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotm_64)(
        hipblasHandle_t handle, int64_t n, T* x, int64_t incx, T* y, int64_t incy, const T* param);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotmBatched_64)(hipblasHandle_t handle,
                                             int64_t         n,
                                             T* const        x[],
                                             int64_t         incx,
                                             T* const        y[],
                                             int64_t         incy,
                                             const T* const  param[],
                                             int64_t         batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotmStridedBatched_64)(hipblasHandle_t handle,
                                                    int64_t         n,
                                                    T*              x,
                                                    int64_t         incx,
                                                    hipblasStride   stridex,
                                                    T*              y,
                                                    int64_t         incy,
                                                    hipblasStride   stridey,
                                                    const T*        param,
                                                    hipblasStride   strideparam,
                                                    int64_t         batch_count);

    MAP2CF_D64(hipblasRotm, float, hipblasSrotm);
    MAP2CF_D64(hipblasRotm, double, hipblasDrotm);

    MAP2CF_D64(hipblasRotmBatched, float, hipblasSrotmBatched);
    MAP2CF_D64(hipblasRotmBatched, double, hipblasDrotmBatched);

    MAP2CF_D64(hipblasRotmStridedBatched, float, hipblasSrotmStridedBatched);
    MAP2CF_D64(hipblasRotmStridedBatched, double, hipblasDrotmStridedBatched);

    // rotmg
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotmg)(
        hipblasHandle_t handle, T* d1, T* d2, T* x1, const T* y1, T* param);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotmgBatched)(hipblasHandle_t handle,
                                           T* const        d1[],
                                           T* const        d2[],
                                           T* const        x1[],
                                           const T* const  y1[],
                                           T* const        param[],
                                           int             batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotmgStridedBatched)(hipblasHandle_t handle,
                                                  T*              d1,
                                                  hipblasStride   stride_d1,
                                                  T*              d2,
                                                  hipblasStride   stride_d2,
                                                  T*              x1,
                                                  hipblasStride   stride_x1,
                                                  const T*        y1,
                                                  hipblasStride   stride_y1,
                                                  T*              param,
                                                  hipblasStride   strideparam,
                                                  int             batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotmg_64)(
        hipblasHandle_t handle, T* d1, T* d2, T* x1, const T* y1, T* param);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotmgBatched_64)(hipblasHandle_t handle,
                                              T* const        d1[],
                                              T* const        d2[],
                                              T* const        x1[],
                                              const T* const  y1[],
                                              T* const        param[],
                                              int64_t         batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasRotmgStridedBatched_64)(hipblasHandle_t handle,
                                                     T*              d1,
                                                     hipblasStride   stride_d1,
                                                     T*              d2,
                                                     hipblasStride   stride_d2,
                                                     T*              x1,
                                                     hipblasStride   stride_x1,
                                                     const T*        y1,
                                                     hipblasStride   stride_y1,
                                                     T*              param,
                                                     hipblasStride   strideparam,
                                                     int64_t         batch_count);

    MAP2CF_D64(hipblasRotmg, float, hipblasSrotmg);
    MAP2CF_D64(hipblasRotmg, double, hipblasDrotmg);

    MAP2CF_D64(hipblasRotmgBatched, float, hipblasSrotmgBatched);
    MAP2CF_D64(hipblasRotmgBatched, double, hipblasDrotmgBatched);

    MAP2CF_D64(hipblasRotmgStridedBatched, float, hipblasSrotmgStridedBatched);
    MAP2CF_D64(hipblasRotmgStridedBatched, double, hipblasDrotmgStridedBatched);

    // amax
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIamax)(
        hipblasHandle_t handle, int n, const T* x, int incx, int* result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIamaxBatched)(
        hipblasHandle_t handle, int n, const T* const x[], int incx, int batch_count, int* result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIamaxStridedBatched)(hipblasHandle_t handle,
                                                  int             n,
                                                  const T*        x,
                                                  int             incx,
                                                  hipblasStride   stridex,
                                                  int             batch_count,
                                                  int*            result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIamax_64)(
        hipblasHandle_t handle, int64_t n, const T* x, int64_t incx, int64_t* result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIamaxBatched_64)(hipblasHandle_t handle,
                                              int64_t         n,
                                              const T* const  x[],
                                              int64_t         incx,
                                              int64_t         batch_count,
                                              int64_t*        result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIamaxStridedBatched_64)(hipblasHandle_t handle,
                                                     int64_t         n,
                                                     const T*        x,
                                                     int64_t         incx,
                                                     hipblasStride   stridex,
                                                     int64_t         batch_count,
                                                     int64_t*        result);

    MAP2CF_D64(hipblasIamax, float, hipblasIsamax);
    MAP2CF_D64(hipblasIamax, double, hipblasIdamax);
    MAP2CF_D64_V2(hipblasIamax, hipblasComplex, hipblasIcamax);
    MAP2CF_D64_V2(hipblasIamax, hipblasDoubleComplex, hipblasIzamax);

    MAP2CF_D64(hipblasIamaxBatched, float, hipblasIsamaxBatched);
    MAP2CF_D64(hipblasIamaxBatched, double, hipblasIdamaxBatched);
    MAP2CF_D64_V2(hipblasIamaxBatched, hipblasComplex, hipblasIcamaxBatched);
    MAP2CF_D64_V2(hipblasIamaxBatched, hipblasDoubleComplex, hipblasIzamaxBatched);

    MAP2CF_D64(hipblasIamaxStridedBatched, float, hipblasIsamaxStridedBatched);
    MAP2CF_D64(hipblasIamaxStridedBatched, double, hipblasIdamaxStridedBatched);
    MAP2CF_D64_V2(hipblasIamaxStridedBatched, hipblasComplex, hipblasIcamaxStridedBatched);
    MAP2CF_D64_V2(hipblasIamaxStridedBatched, hipblasDoubleComplex, hipblasIzamaxStridedBatched);

    // amin
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIamin)(
        hipblasHandle_t handle, int n, const T* x, int incx, int* result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIaminBatched)(
        hipblasHandle_t handle, int n, const T* const x[], int incx, int batch_count, int* result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIaminStridedBatched)(hipblasHandle_t handle,
                                                  int             n,
                                                  const T*        x,
                                                  int             incx,
                                                  hipblasStride   stridex,
                                                  int             batch_count,
                                                  int*            result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIamin_64)(
        hipblasHandle_t handle, int64_t n, const T* x, int64_t incx, int64_t* result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIaminBatched_64)(hipblasHandle_t handle,
                                              int64_t         n,
                                              const T* const  x[],
                                              int64_t         incx,
                                              int64_t         batch_count,
                                              int64_t*        result);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasIaminStridedBatched_64)(hipblasHandle_t handle,
                                                     int64_t         n,
                                                     const T*        x,
                                                     int64_t         incx,
                                                     hipblasStride   stridex,
                                                     int64_t         batch_count,
                                                     int64_t*        result);

    MAP2CF_D64(hipblasIamin, float, hipblasIsamin);
    MAP2CF_D64(hipblasIamin, double, hipblasIdamin);
    MAP2CF_D64_V2(hipblasIamin, hipblasComplex, hipblasIcamin);
    MAP2CF_D64_V2(hipblasIamin, hipblasDoubleComplex, hipblasIzamin);

    MAP2CF_D64(hipblasIaminBatched, float, hipblasIsaminBatched);
    MAP2CF_D64(hipblasIaminBatched, double, hipblasIdaminBatched);
    MAP2CF_D64_V2(hipblasIaminBatched, hipblasComplex, hipblasIcaminBatched);
    MAP2CF_D64_V2(hipblasIaminBatched, hipblasDoubleComplex, hipblasIzaminBatched);

    MAP2CF_D64(hipblasIaminStridedBatched, float, hipblasIsaminStridedBatched);
    MAP2CF_D64(hipblasIaminStridedBatched, double, hipblasIdaminStridedBatched);
    MAP2CF_D64_V2(hipblasIaminStridedBatched, hipblasComplex, hipblasIcaminStridedBatched);
    MAP2CF_D64_V2(hipblasIaminStridedBatched, hipblasDoubleComplex, hipblasIzaminStridedBatched);

    // axpy
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAxpy)(
        hipblasHandle_t handle, int n, const T* alpha, const T* x, int incx, T* y, int incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAxpyBatched)(hipblasHandle_t handle,
                                          int             n,
                                          const T*        alpha,
                                          const T* const  x[],
                                          int             incx,
                                          T* const        y[],
                                          int             incy,
                                          int             batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAxpyStridedBatched)(hipblasHandle_t handle,
                                                 int             n,
                                                 const T*        alpha,
                                                 const T*        x,
                                                 int             incx,
                                                 hipblasStride   stridex,
                                                 T*              y,
                                                 int             incy,
                                                 hipblasStride   stridey,
                                                 int             batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAxpy_64)(hipblasHandle_t handle,
                                      int64_t         n,
                                      const T*        alpha,
                                      const T*        x,
                                      int64_t         incx,
                                      T*              y,
                                      int64_t         incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAxpyBatched_64)(hipblasHandle_t handle,
                                             int64_t         n,
                                             const T*        alpha,
                                             const T* const  x[],
                                             int64_t         incx,
                                             T* const        y[],
                                             int64_t         incy,
                                             int64_t         batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasAxpyStridedBatched_64)(hipblasHandle_t handle,
                                                    int64_t         n,
                                                    const T*        alpha,
                                                    const T*        x,
                                                    int64_t         incx,
                                                    hipblasStride   stridex,
                                                    T*              y,
                                                    int64_t         incy,
                                                    hipblasStride   stridey,
                                                    int64_t         batch_count);

    MAP2CF_D64(hipblasAxpy, hipblasHalf, hipblasHaxpy);
    MAP2CF_D64(hipblasAxpy, float, hipblasSaxpy);
    MAP2CF_D64(hipblasAxpy, double, hipblasDaxpy);
    MAP2CF_D64_V2(hipblasAxpy, hipblasComplex, hipblasCaxpy);
    MAP2CF_D64_V2(hipblasAxpy, hipblasDoubleComplex, hipblasZaxpy);

    MAP2CF_D64(hipblasAxpyBatched, hipblasHalf, hipblasHaxpyBatched);
    MAP2CF_D64(hipblasAxpyBatched, float, hipblasSaxpyBatched);
    MAP2CF_D64(hipblasAxpyBatched, double, hipblasDaxpyBatched);
    MAP2CF_D64_V2(hipblasAxpyBatched, hipblasComplex, hipblasCaxpyBatched);
    MAP2CF_D64_V2(hipblasAxpyBatched, hipblasDoubleComplex, hipblasZaxpyBatched);

    MAP2CF_D64(hipblasAxpyStridedBatched, hipblasHalf, hipblasHaxpyStridedBatched);
    MAP2CF_D64(hipblasAxpyStridedBatched, float, hipblasSaxpyStridedBatched);
    MAP2CF_D64(hipblasAxpyStridedBatched, double, hipblasDaxpyStridedBatched);
    MAP2CF_D64_V2(hipblasAxpyStridedBatched, hipblasComplex, hipblasCaxpyStridedBatched);
    MAP2CF_D64_V2(hipblasAxpyStridedBatched, hipblasDoubleComplex, hipblasZaxpyStridedBatched);

    // ger
    template <typename T, bool CONJ, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGer)(hipblasHandle_t handle,
                                  int             m,
                                  int             n,
                                  const T*        alpha,
                                  const T*        x,
                                  int             incx,
                                  const T*        y,
                                  int             incy,
                                  T*              A,
                                  int             lda);

    template <typename T, bool CONJ, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGerBatched)(hipblasHandle_t handle,
                                         int             m,
                                         int             n,
                                         const T*        alpha,
                                         const T* const  x[],
                                         int             incx,
                                         const T* const  y[],
                                         int             incy,
                                         T* const        A[],
                                         int             lda,
                                         int             batch_count);

    template <typename T, bool CONJ, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGerStridedBatched)(hipblasHandle_t handle,
                                                int             m,
                                                int             n,
                                                const T*        alpha,
                                                const T*        x,
                                                int             incx,
                                                hipblasStride   stridex,
                                                const T*        y,
                                                int             incy,
                                                hipblasStride   stridey,
                                                T*              A,
                                                int             lda,
                                                hipblasStride   strideA,
                                                int             batch_count);

    // ger_64
    template <typename T, bool CONJ, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGer_64)(hipblasHandle_t handle,
                                     int64_t         m,
                                     int64_t         n,
                                     const T*        alpha,
                                     const T*        x,
                                     int64_t         incx,
                                     const T*        y,
                                     int64_t         incy,
                                     T*              A,
                                     int64_t         lda);

    template <typename T, bool CONJ, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGerBatched_64)(hipblasHandle_t handle,
                                            int64_t         m,
                                            int64_t         n,
                                            const T*        alpha,
                                            const T* const  x[],
                                            int64_t         incx,
                                            const T* const  y[],
                                            int64_t         incy,
                                            T* const        A[],
                                            int64_t         lda,
                                            int64_t         batch_count);

    template <typename T, bool CONJ, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGerStridedBatched_64)(hipblasHandle_t handle,
                                                   int64_t         m,
                                                   int64_t         n,
                                                   const T*        alpha,
                                                   const T*        x,
                                                   int64_t         incx,
                                                   hipblasStride   stridex,
                                                   const T*        y,
                                                   int64_t         incy,
                                                   hipblasStride   stridey,
                                                   T*              A,
                                                   int64_t         lda,
                                                   hipblasStride   strideA,
                                                   int64_t         batch_count);

    MAP2CF_D64(hipblasGer, float, false, hipblasSger);
    MAP2CF_D64(hipblasGer, double, false, hipblasDger);
    MAP2CF_D64_V2(hipblasGer, hipblasComplex, false, hipblasCgeru);
    MAP2CF_D64_V2(hipblasGer, hipblasDoubleComplex, false, hipblasZgeru);
    MAP2CF_D64_V2(hipblasGer, hipblasComplex, true, hipblasCgerc);
    MAP2CF_D64_V2(hipblasGer, hipblasDoubleComplex, true, hipblasZgerc);

    MAP2CF_D64(hipblasGerBatched, float, false, hipblasSgerBatched);
    MAP2CF_D64(hipblasGerBatched, double, false, hipblasDgerBatched);
    MAP2CF_D64_V2(hipblasGerBatched, hipblasComplex, false, hipblasCgeruBatched);
    MAP2CF_D64_V2(hipblasGerBatched, hipblasDoubleComplex, false, hipblasZgeruBatched);
    MAP2CF_D64_V2(hipblasGerBatched, hipblasComplex, true, hipblasCgercBatched);
    MAP2CF_D64_V2(hipblasGerBatched, hipblasDoubleComplex, true, hipblasZgercBatched);

    MAP2CF_D64(hipblasGerStridedBatched, float, false, hipblasSgerStridedBatched);
    MAP2CF_D64(hipblasGerStridedBatched, double, false, hipblasDgerStridedBatched);
    MAP2CF_D64_V2(hipblasGerStridedBatched, hipblasComplex, false, hipblasCgeruStridedBatched);
    MAP2CF_D64_V2(hipblasGerStridedBatched,
                  hipblasDoubleComplex,
                  false,
                  hipblasZgeruStridedBatched);
    MAP2CF_D64_V2(hipblasGerStridedBatched, hipblasComplex, true, hipblasCgercStridedBatched);
    MAP2CF_D64_V2(hipblasGerStridedBatched, hipblasDoubleComplex, true, hipblasZgercStridedBatched);

    // hbmv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHbmv)(hipblasHandle_t   handle,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   int               k,
                                   const T*          alpha,
                                   const T*          A,
                                   int               lda,
                                   const T*          x,
                                   int               incx,
                                   const T*          beta,
                                   T*                y,
                                   int               incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHbmv_64)(hipblasHandle_t   handle,
                                      hipblasFillMode_t uplo,
                                      int64_t           n,
                                      int64_t           k,
                                      const T*          alpha,
                                      const T*          A,
                                      int64_t           lda,
                                      const T*          x,
                                      int64_t           incx,
                                      const T*          beta,
                                      T*                y,
                                      int64_t           incy);

    MAP2CF_D64_V2(hipblasHbmv, hipblasComplex, hipblasChbmv);
    MAP2CF_D64_V2(hipblasHbmv, hipblasDoubleComplex, hipblasZhbmv);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHbmvBatched)(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          int               k,
                                          const T*          alpha,
                                          const T* const    A[],
                                          int               lda,
                                          const T* const    x[],
                                          int               incx,
                                          const T*          beta,
                                          T* const          y[],
                                          int               incy,
                                          int               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHbmvBatched_64)(hipblasHandle_t   handle,
                                             hipblasFillMode_t uplo,
                                             int64_t           n,
                                             int64_t           k,
                                             const T*          alpha,
                                             const T* const    A[],
                                             int64_t           lda,
                                             const T* const    x[],
                                             int64_t           incx,
                                             const T*          beta,
                                             T* const          y[],
                                             int64_t           incy,
                                             int64_t           batchCount);

    MAP2CF_D64_V2(hipblasHbmvBatched, hipblasComplex, hipblasChbmvBatched);
    MAP2CF_D64_V2(hipblasHbmvBatched, hipblasDoubleComplex, hipblasZhbmvBatched);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHbmvStridedBatched)(hipblasHandle_t   handle,
                                                 hipblasFillMode_t uplo,
                                                 int               n,
                                                 int               k,
                                                 const T*          alpha,
                                                 const T*          A,
                                                 int               lda,
                                                 hipblasStride     strideA,
                                                 const T*          x,
                                                 int               incx,
                                                 hipblasStride     stridex,
                                                 const T*          beta,
                                                 T*                y,
                                                 int               incy,
                                                 hipblasStride     stridey,
                                                 int               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHbmvStridedBatched_64)(hipblasHandle_t   handle,
                                                    hipblasFillMode_t uplo,
                                                    int64_t           n,
                                                    int64_t           k,
                                                    const T*          alpha,
                                                    const T*          A,
                                                    int64_t           lda,
                                                    hipblasStride     strideA,
                                                    const T*          x,
                                                    int64_t           incx,
                                                    hipblasStride     stridex,
                                                    const T*          beta,
                                                    T*                y,
                                                    int64_t           incy,
                                                    hipblasStride     stridey,
                                                    int64_t           batchCount);

    MAP2CF_D64_V2(hipblasHbmvStridedBatched, hipblasComplex, hipblasChbmvStridedBatched);
    MAP2CF_D64_V2(hipblasHbmvStridedBatched, hipblasDoubleComplex, hipblasZhbmvStridedBatched);

    // hemv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHemv)(hipblasHandle_t   handle,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   const T*          alpha,
                                   const T*          A,
                                   int               lda,
                                   const T*          x,
                                   int               incx,
                                   const T*          beta,
                                   T*                y,
                                   int               incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHemv_64)(hipblasHandle_t   handle,
                                      hipblasFillMode_t uplo,
                                      int64_t           n,
                                      const T*          alpha,
                                      const T*          A,
                                      int64_t           lda,
                                      const T*          x,
                                      int64_t           incx,
                                      const T*          beta,
                                      T*                y,
                                      int64_t           incy);

    MAP2CF_D64_V2(hipblasHemv, hipblasComplex, hipblasChemv);
    MAP2CF_D64_V2(hipblasHemv, hipblasDoubleComplex, hipblasZhemv);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHemvBatched)(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const T*          alpha,
                                          const T* const    A[],
                                          int               lda,
                                          const T* const    x[],
                                          int               incx,
                                          const T*          beta,
                                          T* const          y[],
                                          int               incy,
                                          int               batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHemvBatched_64)(hipblasHandle_t   handle,
                                             hipblasFillMode_t uplo,
                                             int64_t           n,
                                             const T*          alpha,
                                             const T* const    A[],
                                             int64_t           lda,
                                             const T* const    x[],
                                             int64_t           incx,
                                             const T*          beta,
                                             T* const          y[],
                                             int64_t           incy,
                                             int64_t           batch_count);

    MAP2CF_D64_V2(hipblasHemvBatched, hipblasComplex, hipblasChemvBatched);
    MAP2CF_D64_V2(hipblasHemvBatched, hipblasDoubleComplex, hipblasZhemvBatched);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHemvStridedBatched)(hipblasHandle_t   handle,
                                                 hipblasFillMode_t uplo,
                                                 int               n,
                                                 const T*          alpha,
                                                 const T*          A,
                                                 int               lda,
                                                 hipblasStride     stride_a,
                                                 const T*          x,
                                                 int               incx,
                                                 hipblasStride     stride_x,
                                                 const T*          beta,
                                                 T*                y,
                                                 int               incy,
                                                 hipblasStride     stride_y,
                                                 int               batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHemvStridedBatched_64)(hipblasHandle_t   handle,
                                                    hipblasFillMode_t uplo,
                                                    int64_t           n,
                                                    const T*          alpha,
                                                    const T*          A,
                                                    int64_t           lda,
                                                    hipblasStride     stride_a,
                                                    const T*          x,
                                                    int64_t           incx,
                                                    hipblasStride     stride_x,
                                                    const T*          beta,
                                                    T*                y,
                                                    int64_t           incy,
                                                    hipblasStride     stride_y,
                                                    int64_t           batch_count);

    MAP2CF_D64_V2(hipblasHemvStridedBatched, hipblasComplex, hipblasChemvStridedBatched);
    MAP2CF_D64_V2(hipblasHemvStridedBatched, hipblasDoubleComplex, hipblasZhemvStridedBatched);

    // her
    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer)(hipblasHandle_t   handle,
                                  hipblasFillMode_t uplo,
                                  int               n,
                                  const U*          alpha,
                                  const T*          x,
                                  int               incx,
                                  T*                A,
                                  int               lda);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer_64)(hipblasHandle_t   handle,
                                     hipblasFillMode_t uplo,
                                     int64_t           n,
                                     const U*          alpha,
                                     const T*          x,
                                     int64_t           incx,
                                     T*                A,
                                     int64_t           lda);

    MAP2CF_D64_V2(hipblasHer, hipblasComplex, float, hipblasCher);
    MAP2CF_D64_V2(hipblasHer, hipblasDoubleComplex, double, hipblasZher);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerBatched)(hipblasHandle_t   handle,
                                         hipblasFillMode_t uplo,
                                         int               n,
                                         const U*          alpha,
                                         const T* const    x[],
                                         int               incx,
                                         T* const          A[],
                                         int               lda,
                                         int               batchCount);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerBatched_64)(hipblasHandle_t   handle,
                                            hipblasFillMode_t uplo,
                                            int64_t           n,
                                            const U*          alpha,
                                            const T* const    x[],
                                            int64_t           incx,
                                            T* const          A[],
                                            int64_t           lda,
                                            int64_t           batchCount);

    MAP2CF_D64_V2(hipblasHerBatched, hipblasComplex, float, hipblasCherBatched);
    MAP2CF_D64_V2(hipblasHerBatched, hipblasDoubleComplex, double, hipblasZherBatched);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerStridedBatched)(hipblasHandle_t   handle,
                                                hipblasFillMode_t uplo,
                                                int               n,
                                                const U*          alpha,
                                                const T*          x,
                                                int               incx,
                                                hipblasStride     stridex,
                                                T*                A,
                                                int               lda,
                                                hipblasStride     strideA,
                                                int               batchCount);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerStridedBatched_64)(hipblasHandle_t   handle,
                                                   hipblasFillMode_t uplo,
                                                   int64_t           n,
                                                   const U*          alpha,
                                                   const T*          x,
                                                   int64_t           incx,
                                                   hipblasStride     stridex,
                                                   T*                A,
                                                   int64_t           lda,
                                                   hipblasStride     strideA,
                                                   int64_t           batchCount);

    MAP2CF_D64_V2(hipblasHerStridedBatched, hipblasComplex, float, hipblasCherStridedBatched);
    MAP2CF_D64_V2(hipblasHerStridedBatched,
                  hipblasDoubleComplex,
                  double,
                  hipblasZherStridedBatched);

    // her2
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer2)(hipblasHandle_t   handle,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   const T*          alpha,
                                   const T*          x,
                                   int               incx,
                                   const T*          y,
                                   int               incy,
                                   T*                A,
                                   int               lda);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer2_64)(hipblasHandle_t   handle,
                                      hipblasFillMode_t uplo,
                                      int64_t           n,
                                      const T*          alpha,
                                      const T*          x,
                                      int64_t           incx,
                                      const T*          y,
                                      int64_t           incy,
                                      T*                A,
                                      int64_t           lda);

    MAP2CF_D64_V2(hipblasHer2, hipblasComplex, hipblasCher2);
    MAP2CF_D64_V2(hipblasHer2, hipblasDoubleComplex, hipblasZher2);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer2Batched)(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const T*          alpha,
                                          const T* const    x[],
                                          int               incx,
                                          const T* const    y[],
                                          int               incy,
                                          T* const          A[],
                                          int               lda,
                                          int               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer2Batched_64)(hipblasHandle_t   handle,
                                             hipblasFillMode_t uplo,
                                             int64_t           n,
                                             const T*          alpha,
                                             const T* const    x[],
                                             int64_t           incx,
                                             const T* const    y[],
                                             int64_t           incy,
                                             T* const          A[],
                                             int64_t           lda,
                                             int64_t           batchCount);

    MAP2CF_D64_V2(hipblasHer2Batched, hipblasComplex, hipblasCher2Batched);
    MAP2CF_D64_V2(hipblasHer2Batched, hipblasDoubleComplex, hipblasZher2Batched);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer2StridedBatched)(hipblasHandle_t   handle,
                                                 hipblasFillMode_t uplo,
                                                 int               n,
                                                 const T*          alpha,
                                                 const T*          x,
                                                 int               incx,
                                                 hipblasStride     stridex,
                                                 const T*          y,
                                                 int               incy,
                                                 hipblasStride     stridey,
                                                 T*                A,
                                                 int               lda,
                                                 hipblasStride     strideA,
                                                 int               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer2StridedBatched_64)(hipblasHandle_t   handle,
                                                    hipblasFillMode_t uplo,
                                                    int64_t           n,
                                                    const T*          alpha,
                                                    const T*          x,
                                                    int64_t           incx,
                                                    hipblasStride     stridex,
                                                    const T*          y,
                                                    int64_t           incy,
                                                    hipblasStride     stridey,
                                                    T*                A,
                                                    int64_t           lda,
                                                    hipblasStride     strideA,
                                                    int64_t           batchCount);

    MAP2CF_D64_V2(hipblasHer2StridedBatched, hipblasComplex, hipblasCher2StridedBatched);
    MAP2CF_D64_V2(hipblasHer2StridedBatched, hipblasDoubleComplex, hipblasZher2StridedBatched);

    // hpmv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpmv)(hipblasHandle_t   handle,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   const T*          alpha,
                                   const T*          AP,
                                   const T*          x,
                                   int               incx,
                                   const T*          beta,
                                   T*                y,
                                   int               incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpmv_64)(hipblasHandle_t   handle,
                                      hipblasFillMode_t uplo,
                                      int64_t           n,
                                      const T*          alpha,
                                      const T*          AP,
                                      const T*          x,
                                      int64_t           incx,
                                      const T*          beta,
                                      T*                y,
                                      int64_t           incy);

    MAP2CF_D64_V2(hipblasHpmv, hipblasComplex, hipblasChpmv);
    MAP2CF_D64_V2(hipblasHpmv, hipblasDoubleComplex, hipblasZhpmv);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpmvBatched)(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const T*          alpha,
                                          const T* const    AP[],
                                          const T* const    x[],
                                          int               incx,
                                          const T*          beta,
                                          T* const          y[],
                                          int               incy,
                                          int               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpmvBatched_64)(hipblasHandle_t   handle,
                                             hipblasFillMode_t uplo,
                                             int64_t           n,
                                             const T*          alpha,
                                             const T* const    AP[],
                                             const T* const    x[],
                                             int64_t           incx,
                                             const T*          beta,
                                             T* const          y[],
                                             int64_t           incy,
                                             int64_t           batchCount);

    MAP2CF_D64_V2(hipblasHpmvBatched, hipblasComplex, hipblasChpmvBatched);
    MAP2CF_D64_V2(hipblasHpmvBatched, hipblasDoubleComplex, hipblasZhpmvBatched);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpmvStridedBatched)(hipblasHandle_t   handle,
                                                 hipblasFillMode_t uplo,
                                                 int               n,
                                                 const T*          alpha,
                                                 const T*          AP,
                                                 hipblasStride     strideAP,
                                                 const T*          x,
                                                 int               incx,
                                                 hipblasStride     stridex,
                                                 const T*          beta,
                                                 T*                y,
                                                 int               incy,
                                                 hipblasStride     stridey,
                                                 int               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpmvStridedBatched_64)(hipblasHandle_t   handle,
                                                    hipblasFillMode_t uplo,
                                                    int64_t           n,
                                                    const T*          alpha,
                                                    const T*          AP,
                                                    hipblasStride     strideAP,
                                                    const T*          x,
                                                    int64_t           incx,
                                                    hipblasStride     stridex,
                                                    const T*          beta,
                                                    T*                y,
                                                    int64_t           incy,
                                                    hipblasStride     stridey,
                                                    int64_t           batchCount);

    MAP2CF_D64_V2(hipblasHpmvStridedBatched, hipblasComplex, hipblasChpmvStridedBatched);
    MAP2CF_D64_V2(hipblasHpmvStridedBatched, hipblasDoubleComplex, hipblasZhpmvStridedBatched);

    // hpr
    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpr)(hipblasHandle_t   handle,
                                  hipblasFillMode_t uplo,
                                  int               n,
                                  const U*          alpha,
                                  const T*          x,
                                  int               incx,
                                  T*                AP);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpr_64)(hipblasHandle_t   handle,
                                     hipblasFillMode_t uplo,
                                     int64_t           n,
                                     const U*          alpha,
                                     const T*          x,
                                     int64_t           incx,
                                     T*                AP);

    MAP2CF_D64_V2(hipblasHpr, hipblasComplex, float, hipblasChpr);
    MAP2CF_D64_V2(hipblasHpr, hipblasDoubleComplex, double, hipblasZhpr);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHprBatched)(hipblasHandle_t   handle,
                                         hipblasFillMode_t uplo,
                                         int               n,
                                         const U*          alpha,
                                         const T* const    x[],
                                         int               incx,
                                         T* const          AP[],
                                         int               batchCount);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHprBatched_64)(hipblasHandle_t   handle,
                                            hipblasFillMode_t uplo,
                                            int64_t           n,
                                            const U*          alpha,
                                            const T* const    x[],
                                            int64_t           incx,
                                            T* const          AP[],
                                            int64_t           batchCount);

    MAP2CF_D64_V2(hipblasHprBatched, hipblasComplex, float, hipblasChprBatched);
    MAP2CF_D64_V2(hipblasHprBatched, hipblasDoubleComplex, double, hipblasZhprBatched);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHprStridedBatched)(hipblasHandle_t   handle,
                                                hipblasFillMode_t uplo,
                                                int               n,
                                                const U*          alpha,
                                                const T*          x,
                                                int               incx,
                                                hipblasStride     stridex,
                                                T*                AP,
                                                hipblasStride     strideAP,
                                                int               batchCount);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHprStridedBatched_64)(hipblasHandle_t   handle,
                                                   hipblasFillMode_t uplo,
                                                   int64_t           n,
                                                   const U*          alpha,
                                                   const T*          x,
                                                   int64_t           incx,
                                                   hipblasStride     stridex,
                                                   T*                AP,
                                                   hipblasStride     strideAP,
                                                   int64_t           batchCount);

    MAP2CF_D64_V2(hipblasHprStridedBatched, hipblasComplex, float, hipblasChprStridedBatched);
    MAP2CF_D64_V2(hipblasHprStridedBatched,
                  hipblasDoubleComplex,
                  double,
                  hipblasZhprStridedBatched);

    // hpr2
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpr2)(hipblasHandle_t   handle,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   const T*          alpha,
                                   const T*          x,
                                   int               incx,
                                   const T*          y,
                                   int               incy,
                                   T*                AP);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpr2_64)(hipblasHandle_t   handle,
                                      hipblasFillMode_t uplo,
                                      int64_t           n,
                                      const T*          alpha,
                                      const T*          x,
                                      int64_t           incx,
                                      const T*          y,
                                      int64_t           incy,
                                      T*                AP);

    MAP2CF_D64_V2(hipblasHpr2, hipblasComplex, hipblasChpr2);
    MAP2CF_D64_V2(hipblasHpr2, hipblasDoubleComplex, hipblasZhpr2);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpr2Batched)(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const T*          alpha,
                                          const T* const    x[],
                                          int               incx,
                                          const T* const    y[],
                                          int               incy,
                                          T* const          AP[],
                                          int               batchCount);
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpr2Batched_64)(hipblasHandle_t   handle,
                                             hipblasFillMode_t uplo,
                                             int64_t           n,
                                             const T*          alpha,
                                             const T* const    x[],
                                             int64_t           incx,
                                             const T* const    y[],
                                             int64_t           incy,
                                             T* const          AP[],
                                             int64_t           batchCount);

    MAP2CF_D64_V2(hipblasHpr2Batched, hipblasComplex, hipblasChpr2Batched);
    MAP2CF_D64_V2(hipblasHpr2Batched, hipblasDoubleComplex, hipblasZhpr2Batched);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpr2StridedBatched)(hipblasHandle_t   handle,
                                                 hipblasFillMode_t uplo,
                                                 int               n,
                                                 const T*          alpha,
                                                 const T*          x,
                                                 int               incx,
                                                 hipblasStride     stridex,
                                                 const T*          y,
                                                 int               incy,
                                                 hipblasStride     stridey,
                                                 T*                AP,
                                                 hipblasStride     strideAP,
                                                 int               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHpr2StridedBatched_64)(hipblasHandle_t   handle,
                                                    hipblasFillMode_t uplo,
                                                    int64_t           n,
                                                    const T*          alpha,
                                                    const T*          x,
                                                    int64_t           incx,
                                                    hipblasStride     stridex,
                                                    const T*          y,
                                                    int64_t           incy,
                                                    hipblasStride     stridey,
                                                    T*                AP,
                                                    hipblasStride     strideAP,
                                                    int64_t           batchCount);

    MAP2CF_D64_V2(hipblasHpr2StridedBatched, hipblasComplex, hipblasChpr2StridedBatched);
    MAP2CF_D64_V2(hipblasHpr2StridedBatched, hipblasDoubleComplex, hipblasZhpr2StridedBatched);

    // sbmv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSbmv)(hipblasHandle_t   handle,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   int               k,
                                   const T*          alpha,
                                   const T*          A,
                                   int               lda,
                                   const T*          x,
                                   int               incx,
                                   const T*          beta,
                                   T*                y,
                                   int               incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSbmvBatched)(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          int               k,
                                          const T*          alpha,
                                          const T* const    A[],
                                          int               lda,
                                          const T* const    x[],
                                          int               incx,
                                          const T*          beta,
                                          T* const          y[],
                                          int               incy,
                                          int               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSbmvStridedBatched)(hipblasHandle_t   handle,
                                                 hipblasFillMode_t uplo,
                                                 int               n,
                                                 int               k,
                                                 const T*          alpha,
                                                 const T*          A,
                                                 int               lda,
                                                 hipblasStride     strideA,
                                                 const T*          x,
                                                 int               incx,
                                                 hipblasStride     stridex,
                                                 const T*          beta,
                                                 T*                y,
                                                 int               incy,
                                                 hipblasStride     stridey,
                                                 int               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSbmv_64)(hipblasHandle_t   handle,
                                      hipblasFillMode_t uplo,
                                      int64_t           n,
                                      int64_t           k,
                                      const T*          alpha,
                                      const T*          A,
                                      int64_t           lda,
                                      const T*          x,
                                      int64_t           incx,
                                      const T*          beta,
                                      T*                y,
                                      int64_t           incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSbmvBatched_64)(hipblasHandle_t   handle,
                                             hipblasFillMode_t uplo,
                                             int64_t           n,
                                             int64_t           k,
                                             const T*          alpha,
                                             const T* const    A[],
                                             int64_t           lda,
                                             const T* const    x[],
                                             int64_t           incx,
                                             const T*          beta,
                                             T* const          y[],
                                             int64_t           incy,
                                             int64_t           batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSbmvStridedBatched_64)(hipblasHandle_t   handle,
                                                    hipblasFillMode_t uplo,
                                                    int64_t           n,
                                                    int64_t           k,
                                                    const T*          alpha,
                                                    const T*          A,
                                                    int64_t           lda,
                                                    hipblasStride     strideA,
                                                    const T*          x,
                                                    int64_t           incx,
                                                    hipblasStride     stridex,
                                                    const T*          beta,
                                                    T*                y,
                                                    int64_t           incy,
                                                    hipblasStride     stridey,
                                                    int64_t           batchCount);

    MAP2CF_D64(hipblasSbmv, float, hipblasSsbmv);
    MAP2CF_D64(hipblasSbmv, double, hipblasDsbmv);

    MAP2CF_D64(hipblasSbmvBatched, float, hipblasSsbmvBatched);
    MAP2CF_D64(hipblasSbmvBatched, double, hipblasDsbmvBatched);

    MAP2CF_D64(hipblasSbmvStridedBatched, float, hipblasSsbmvStridedBatched);
    MAP2CF_D64(hipblasSbmvStridedBatched, double, hipblasDsbmvStridedBatched);

    // spmv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpmv)(hipblasHandle_t   handle,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   const T*          alpha,
                                   const T*          AP,
                                   const T*          x,
                                   int               incx,
                                   const T*          beta,
                                   T*                y,
                                   int               incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpmvBatched)(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const T*          alpha,
                                          const T* const    AP[],
                                          const T* const    x[],
                                          int               incx,
                                          const T*          beta,
                                          T* const          y[],
                                          int               incy,
                                          int               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpmvStridedBatched)(hipblasHandle_t   handle,
                                                 hipblasFillMode_t uplo,
                                                 int               n,
                                                 const T*          alpha,
                                                 const T*          AP,
                                                 hipblasStride     strideAP,
                                                 const T*          x,
                                                 int               incx,
                                                 hipblasStride     stridex,
                                                 const T*          beta,
                                                 T*                y,
                                                 int               incy,
                                                 hipblasStride     stridey,
                                                 int               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpmv_64)(hipblasHandle_t   handle,
                                      hipblasFillMode_t uplo,
                                      int64_t           n,
                                      const T*          alpha,
                                      const T*          AP,
                                      const T*          x,
                                      int64_t           incx,
                                      const T*          beta,
                                      T*                y,
                                      int64_t           incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpmvBatched_64)(hipblasHandle_t   handle,
                                             hipblasFillMode_t uplo,
                                             int64_t           n,
                                             const T*          alpha,
                                             const T* const    AP[],
                                             const T* const    x[],
                                             int64_t           incx,
                                             const T*          beta,
                                             T* const          y[],
                                             int64_t           incy,
                                             int64_t           batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpmvStridedBatched_64)(hipblasHandle_t   handle,
                                                    hipblasFillMode_t uplo,
                                                    int64_t           n,
                                                    const T*          alpha,
                                                    const T*          AP,
                                                    hipblasStride     strideAP,
                                                    const T*          x,
                                                    int64_t           incx,
                                                    hipblasStride     stridex,
                                                    const T*          beta,
                                                    T*                y,
                                                    int64_t           incy,
                                                    hipblasStride     stridey,
                                                    int64_t           batchCount);

    MAP2CF_D64(hipblasSpmv, float, hipblasSspmv);
    MAP2CF_D64(hipblasSpmv, double, hipblasDspmv);

    MAP2CF_D64(hipblasSpmvBatched, float, hipblasSspmvBatched);
    MAP2CF_D64(hipblasSpmvBatched, double, hipblasDspmvBatched);

    MAP2CF_D64(hipblasSpmvStridedBatched, float, hipblasSspmvStridedBatched);
    MAP2CF_D64(hipblasSpmvStridedBatched, double, hipblasDspmvStridedBatched);

    // spr
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpr)(hipblasHandle_t   handle,
                                  hipblasFillMode_t uplo,
                                  int               n,
                                  const T*          alpha,
                                  const T*          x,
                                  int               incx,
                                  T*                AP);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSprBatched)(hipblasHandle_t   handle,
                                         hipblasFillMode_t uplo,
                                         int               n,
                                         const T*          alpha,
                                         const T* const    x[],
                                         int               incx,
                                         T* const          AP[],
                                         int               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSprStridedBatched)(hipblasHandle_t   handle,
                                                hipblasFillMode_t uplo,
                                                int               n,
                                                const T*          alpha,
                                                const T*          x,
                                                int               incx,
                                                hipblasStride     stridex,
                                                T*                AP,
                                                hipblasStride     strideAP,
                                                int               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpr_64)(hipblasHandle_t   handle,
                                     hipblasFillMode_t uplo,
                                     int64_t           n,
                                     const T*          alpha,
                                     const T*          x,
                                     int64_t           incx,
                                     T*                AP);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSprBatched_64)(hipblasHandle_t   handle,
                                            hipblasFillMode_t uplo,
                                            int64_t           n,
                                            const T*          alpha,
                                            const T* const    x[],
                                            int64_t           incx,
                                            T* const          AP[],
                                            int64_t           batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSprStridedBatched_64)(hipblasHandle_t   handle,
                                                   hipblasFillMode_t uplo,
                                                   int64_t           n,
                                                   const T*          alpha,
                                                   const T*          x,
                                                   int64_t           incx,
                                                   hipblasStride     stridex,
                                                   T*                AP,
                                                   hipblasStride     strideAP,
                                                   int64_t           batchCount);

    MAP2CF_D64(hipblasSpr, float, hipblasSspr);
    MAP2CF_D64(hipblasSpr, double, hipblasDspr);
    MAP2CF_D64_V2(hipblasSpr, hipblasComplex, hipblasCspr);
    MAP2CF_D64_V2(hipblasSpr, hipblasDoubleComplex, hipblasZspr);

    MAP2CF_D64(hipblasSprBatched, float, hipblasSsprBatched);
    MAP2CF_D64(hipblasSprBatched, double, hipblasDsprBatched);
    MAP2CF_D64_V2(hipblasSprBatched, hipblasComplex, hipblasCsprBatched);
    MAP2CF_D64_V2(hipblasSprBatched, hipblasDoubleComplex, hipblasZsprBatched);

    MAP2CF_D64(hipblasSprStridedBatched, float, hipblasSsprStridedBatched);
    MAP2CF_D64(hipblasSprStridedBatched, double, hipblasDsprStridedBatched);
    MAP2CF_D64_V2(hipblasSprStridedBatched, hipblasComplex, hipblasCsprStridedBatched);
    MAP2CF_D64_V2(hipblasSprStridedBatched, hipblasDoubleComplex, hipblasZsprStridedBatched);

    // spr2
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpr2)(hipblasHandle_t   handle,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   const T*          alpha,
                                   const T*          x,
                                   int               incx,
                                   const T*          y,
                                   int               incy,
                                   T*                AP);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpr2Batched)(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const T*          alpha,
                                          const T* const    x[],
                                          int               incx,
                                          const T* const    y[],
                                          int               incy,
                                          T* const          AP[],
                                          int               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpr2StridedBatched)(hipblasHandle_t   handle,
                                                 hipblasFillMode_t uplo,
                                                 int               n,
                                                 const T*          alpha,
                                                 const T*          x,
                                                 int               incx,
                                                 hipblasStride     stridex,
                                                 const T*          y,
                                                 int               incy,
                                                 hipblasStride     stridey,
                                                 T*                AP,
                                                 hipblasStride     strideAP,
                                                 int               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpr2_64)(hipblasHandle_t   handle,
                                      hipblasFillMode_t uplo,
                                      int64_t           n,
                                      const T*          alpha,
                                      const T*          x,
                                      int64_t           incx,
                                      const T*          y,
                                      int64_t           incy,
                                      T*                AP);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpr2Batched_64)(hipblasHandle_t   handle,
                                             hipblasFillMode_t uplo,
                                             int64_t           n,
                                             const T*          alpha,
                                             const T* const    x[],
                                             int64_t           incx,
                                             const T* const    y[],
                                             int64_t           incy,
                                             T* const          AP[],
                                             int64_t           batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSpr2StridedBatched_64)(hipblasHandle_t   handle,
                                                    hipblasFillMode_t uplo,
                                                    int64_t           n,
                                                    const T*          alpha,
                                                    const T*          x,
                                                    int64_t           incx,
                                                    hipblasStride     stridex,
                                                    const T*          y,
                                                    int64_t           incy,
                                                    hipblasStride     stridey,
                                                    T*                AP,
                                                    hipblasStride     strideAP,
                                                    int64_t           batchCount);

    MAP2CF_D64(hipblasSpr2, float, hipblasSspr2);
    MAP2CF_D64(hipblasSpr2, double, hipblasDspr2);

    MAP2CF_D64(hipblasSpr2Batched, float, hipblasSspr2Batched);
    MAP2CF_D64(hipblasSpr2Batched, double, hipblasDspr2Batched);

    MAP2CF_D64(hipblasSpr2StridedBatched, float, hipblasSspr2StridedBatched);
    MAP2CF_D64(hipblasSpr2StridedBatched, double, hipblasDspr2StridedBatched);

    // symv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymv)(hipblasHandle_t   handle,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   const T*          alpha,
                                   const T*          A,
                                   int               lda,
                                   const T*          x,
                                   int               incx,
                                   const T*          beta,
                                   T*                y,
                                   int               incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymvBatched)(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const T*          alpha,
                                          const T* const    A[],
                                          int               lda,
                                          const T* const    x[],
                                          int               incx,
                                          const T*          beta,
                                          T* const          y[],
                                          int               incy,
                                          int               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymvStridedBatched)(hipblasHandle_t   handle,
                                                 hipblasFillMode_t uplo,
                                                 int               n,
                                                 const T*          alpha,
                                                 const T*          A,
                                                 int               lda,
                                                 hipblasStride     strideA,
                                                 const T*          x,
                                                 int               incx,
                                                 hipblasStride     stridex,
                                                 const T*          beta,
                                                 T*                y,
                                                 int               incy,
                                                 hipblasStride     stridey,
                                                 int               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymv_64)(hipblasHandle_t   handle,
                                      hipblasFillMode_t uplo,
                                      int64_t           n,
                                      const T*          alpha,
                                      const T*          A,
                                      int64_t           lda,
                                      const T*          x,
                                      int64_t           incx,
                                      const T*          beta,
                                      T*                y,
                                      int64_t           incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymvBatched_64)(hipblasHandle_t   handle,
                                             hipblasFillMode_t uplo,
                                             int64_t           n,
                                             const T*          alpha,
                                             const T* const    A[],
                                             int64_t           lda,
                                             const T* const    x[],
                                             int64_t           incx,
                                             const T*          beta,
                                             T* const          y[],
                                             int64_t           incy,
                                             int64_t           batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymvStridedBatched_64)(hipblasHandle_t   handle,
                                                    hipblasFillMode_t uplo,
                                                    int64_t           n,
                                                    const T*          alpha,
                                                    const T*          A,
                                                    int64_t           lda,
                                                    hipblasStride     strideA,
                                                    const T*          x,
                                                    int64_t           incx,
                                                    hipblasStride     stridex,
                                                    const T*          beta,
                                                    T*                y,
                                                    int64_t           incy,
                                                    hipblasStride     stridey,
                                                    int64_t           batchCount);

    MAP2CF_D64(hipblasSymv, float, hipblasSsymv);
    MAP2CF_D64(hipblasSymv, double, hipblasDsymv);
    MAP2CF_D64_V2(hipblasSymv, hipblasComplex, hipblasCsymv);
    MAP2CF_D64_V2(hipblasSymv, hipblasDoubleComplex, hipblasZsymv);

    MAP2CF_D64(hipblasSymvBatched, float, hipblasSsymvBatched);
    MAP2CF_D64(hipblasSymvBatched, double, hipblasDsymvBatched);
    MAP2CF_D64_V2(hipblasSymvBatched, hipblasComplex, hipblasCsymvBatched);
    MAP2CF_D64_V2(hipblasSymvBatched, hipblasDoubleComplex, hipblasZsymvBatched);

    MAP2CF_D64(hipblasSymvStridedBatched, float, hipblasSsymvStridedBatched);
    MAP2CF_D64(hipblasSymvStridedBatched, double, hipblasDsymvStridedBatched);
    MAP2CF_D64_V2(hipblasSymvStridedBatched, hipblasComplex, hipblasCsymvStridedBatched);
    MAP2CF_D64_V2(hipblasSymvStridedBatched, hipblasDoubleComplex, hipblasZsymvStridedBatched);

    // syr
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr)(hipblasHandle_t   handle,
                                  hipblasFillMode_t uplo,
                                  int               n,
                                  const T*          alpha,
                                  const T*          x,
                                  int               incx,
                                  T*                A,
                                  int               lda);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrBatched)(hipblasHandle_t   handle,
                                         hipblasFillMode_t uplo,
                                         int               n,
                                         const T*          alpha,
                                         const T* const    x[],
                                         int               incx,
                                         T* const          A[],
                                         int               lda,
                                         int               batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrStridedBatched)(hipblasHandle_t   handle,
                                                hipblasFillMode_t uplo,
                                                int               n,
                                                const T*          alpha,
                                                const T*          x,
                                                int               incx,
                                                hipblasStride     stridex,
                                                T*                A,
                                                int               lda,
                                                hipblasStride     strideA,
                                                int               batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr_64)(hipblasHandle_t   handle,
                                     hipblasFillMode_t uplo,
                                     int64_t           n,
                                     const T*          alpha,
                                     const T*          x,
                                     int64_t           incx,
                                     T*                A,
                                     int64_t           lda);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrBatched_64)(hipblasHandle_t   handle,
                                            hipblasFillMode_t uplo,
                                            int64_t           n,
                                            const T*          alpha,
                                            const T* const    x[],
                                            int64_t           incx,
                                            T* const          A[],
                                            int64_t           lda,
                                            int64_t           batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrStridedBatched_64)(hipblasHandle_t   handle,
                                                   hipblasFillMode_t uplo,
                                                   int64_t           n,
                                                   const T*          alpha,
                                                   const T*          x,
                                                   int64_t           incx,
                                                   hipblasStride     stridex,
                                                   T*                A,
                                                   int64_t           lda,
                                                   hipblasStride     strideA,
                                                   int64_t           batch_count);

    MAP2CF_D64(hipblasSyr, float, hipblasSsyr);
    MAP2CF_D64(hipblasSyr, double, hipblasDsyr);
    MAP2CF_D64_V2(hipblasSyr, hipblasComplex, hipblasCsyr);
    MAP2CF_D64_V2(hipblasSyr, hipblasDoubleComplex, hipblasZsyr);

    MAP2CF_D64(hipblasSyrBatched, float, hipblasSsyrBatched);
    MAP2CF_D64(hipblasSyrBatched, double, hipblasDsyrBatched);
    MAP2CF_D64_V2(hipblasSyrBatched, hipblasComplex, hipblasCsyrBatched);
    MAP2CF_D64_V2(hipblasSyrBatched, hipblasDoubleComplex, hipblasZsyrBatched);

    MAP2CF_D64(hipblasSyrStridedBatched, float, hipblasSsyrStridedBatched);
    MAP2CF_D64(hipblasSyrStridedBatched, double, hipblasDsyrStridedBatched);
    MAP2CF_D64_V2(hipblasSyrStridedBatched, hipblasComplex, hipblasCsyrStridedBatched);
    MAP2CF_D64_V2(hipblasSyrStridedBatched, hipblasDoubleComplex, hipblasZsyrStridedBatched);

    // syr2
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2)(hipblasHandle_t   handle,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   const T*          alpha,
                                   const T*          x,
                                   int               incx,
                                   const T*          y,
                                   int               incy,
                                   T*                A,
                                   int               lda);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2Batched)(hipblasHandle_t   handle,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          const T*          alpha,
                                          const T* const    x[],
                                          int               incx,
                                          const T* const    y[],
                                          int               incy,
                                          T* const          A[],
                                          int               lda,
                                          int               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2StridedBatched)(hipblasHandle_t   handle,
                                                 hipblasFillMode_t uplo,
                                                 int               n,
                                                 const T*          alpha,
                                                 const T*          x,
                                                 int               incx,
                                                 hipblasStride     stridex,
                                                 const T*          y,
                                                 int               incy,
                                                 hipblasStride     stridey,
                                                 T*                A,
                                                 int               lda,
                                                 hipblasStride     strideA,
                                                 int               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2_64)(hipblasHandle_t   handle,
                                      hipblasFillMode_t uplo,
                                      int64_t           n,
                                      const T*          alpha,
                                      const T*          x,
                                      int64_t           incx,
                                      const T*          y,
                                      int64_t           incy,
                                      T*                A,
                                      int64_t           lda);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2Batched_64)(hipblasHandle_t   handle,
                                             hipblasFillMode_t uplo,
                                             int64_t           n,
                                             const T*          alpha,
                                             const T* const    x[],
                                             int64_t           incx,
                                             const T* const    y[],
                                             int64_t           incy,
                                             T* const          A[],
                                             int64_t           lda,
                                             int64_t           batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2StridedBatched_64)(hipblasHandle_t   handle,
                                                    hipblasFillMode_t uplo,
                                                    int64_t           n,
                                                    const T*          alpha,
                                                    const T*          x,
                                                    int64_t           incx,
                                                    hipblasStride     stridex,
                                                    const T*          y,
                                                    int64_t           incy,
                                                    hipblasStride     stridey,
                                                    T*                A,
                                                    int64_t           lda,
                                                    hipblasStride     strideA,
                                                    int64_t           batchCount);

    MAP2CF_D64(hipblasSyr2, float, hipblasSsyr2);
    MAP2CF_D64(hipblasSyr2, double, hipblasDsyr2);
    MAP2CF_D64_V2(hipblasSyr2, hipblasComplex, hipblasCsyr2);
    MAP2CF_D64_V2(hipblasSyr2, hipblasDoubleComplex, hipblasZsyr2);

    MAP2CF_D64(hipblasSyr2Batched, float, hipblasSsyr2Batched);
    MAP2CF_D64(hipblasSyr2Batched, double, hipblasDsyr2Batched);
    MAP2CF_D64_V2(hipblasSyr2Batched, hipblasComplex, hipblasCsyr2Batched);
    MAP2CF_D64_V2(hipblasSyr2Batched, hipblasDoubleComplex, hipblasZsyr2Batched);

    MAP2CF_D64(hipblasSyr2StridedBatched, float, hipblasSsyr2StridedBatched);
    MAP2CF_D64(hipblasSyr2StridedBatched, double, hipblasDsyr2StridedBatched);
    MAP2CF_D64_V2(hipblasSyr2StridedBatched, hipblasComplex, hipblasCsyr2StridedBatched);
    MAP2CF_D64_V2(hipblasSyr2StridedBatched, hipblasDoubleComplex, hipblasZsyr2StridedBatched);

    // tbmv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbmv)(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   hipblasDiagType_t  diag,
                                   int                m,
                                   int                k,
                                   const T*           A,
                                   int                lda,
                                   T*                 x,
                                   int                incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbmvBatched)(hipblasHandle_t    handle,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          hipblasDiagType_t  diag,
                                          int                m,
                                          int                k,
                                          const T* const     A[],
                                          int                lda,
                                          T* const           x[],
                                          int                incx,
                                          int                batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbmvStridedBatched)(hipblasHandle_t    handle,
                                                 hipblasFillMode_t  uplo,
                                                 hipblasOperation_t transA,
                                                 hipblasDiagType_t  diag,
                                                 int                m,
                                                 int                k,
                                                 const T*           A,
                                                 int                lda,
                                                 hipblasStride      stride_a,
                                                 T*                 x,
                                                 int                incx,
                                                 hipblasStride      stride_x,
                                                 int                batch_count);
    //tbmv_64
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbmv_64)(hipblasHandle_t    handle,
                                      hipblasFillMode_t  uplo,
                                      hipblasOperation_t transA,
                                      hipblasDiagType_t  diag,
                                      int64_t            m,
                                      int64_t            k,
                                      const T*           A,
                                      int64_t            lda,
                                      T*                 x,
                                      int64_t            incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbmvBatched_64)(hipblasHandle_t    handle,
                                             hipblasFillMode_t  uplo,
                                             hipblasOperation_t transA,
                                             hipblasDiagType_t  diag,
                                             int64_t            m,
                                             int64_t            k,
                                             const T* const     A[],
                                             int64_t            lda,
                                             T* const           x[],
                                             int64_t            incx,
                                             int64_t            batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbmvStridedBatched_64)(hipblasHandle_t    handle,
                                                    hipblasFillMode_t  uplo,
                                                    hipblasOperation_t transA,
                                                    hipblasDiagType_t  diag,
                                                    int64_t            m,
                                                    int64_t            k,
                                                    const T*           A,
                                                    int64_t            lda,
                                                    hipblasStride      stride_a,
                                                    T*                 x,
                                                    int64_t            incx,
                                                    hipblasStride      stride_x,
                                                    int64_t            batch_count);

    MAP2CF_D64(hipblasTbmv, float, hipblasStbmv);
    MAP2CF_D64(hipblasTbmv, double, hipblasDtbmv);
    MAP2CF_D64_V2(hipblasTbmv, hipblasComplex, hipblasCtbmv);
    MAP2CF_D64_V2(hipblasTbmv, hipblasDoubleComplex, hipblasZtbmv);

    MAP2CF_D64(hipblasTbmvBatched, float, hipblasStbmvBatched);
    MAP2CF_D64(hipblasTbmvBatched, double, hipblasDtbmvBatched);
    MAP2CF_D64_V2(hipblasTbmvBatched, hipblasComplex, hipblasCtbmvBatched);
    MAP2CF_D64_V2(hipblasTbmvBatched, hipblasDoubleComplex, hipblasZtbmvBatched);

    MAP2CF_D64(hipblasTbmvStridedBatched, float, hipblasStbmvStridedBatched);
    MAP2CF_D64(hipblasTbmvStridedBatched, double, hipblasDtbmvStridedBatched);
    MAP2CF_D64_V2(hipblasTbmvStridedBatched, hipblasComplex, hipblasCtbmvStridedBatched);
    MAP2CF_D64_V2(hipblasTbmvStridedBatched, hipblasDoubleComplex, hipblasZtbmvStridedBatched);

    // tbsv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbsv)(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   hipblasDiagType_t  diag,
                                   int                m,
                                   int                k,
                                   const T*           A,
                                   int                lda,
                                   T*                 x,
                                   int                incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbsvBatched)(hipblasHandle_t    handle,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          hipblasDiagType_t  diag,
                                          int                m,
                                          int                k,
                                          const T* const     A[],
                                          int                lda,
                                          T* const           x[],
                                          int                incx,
                                          int                batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbsvStridedBatched)(hipblasHandle_t    handle,
                                                 hipblasFillMode_t  uplo,
                                                 hipblasOperation_t transA,
                                                 hipblasDiagType_t  diag,
                                                 int                m,
                                                 int                k,
                                                 const T*           A,
                                                 int                lda,
                                                 hipblasStride      strideA,
                                                 T*                 x,
                                                 int                incx,
                                                 hipblasStride      stridex,
                                                 int                batchCount);

    // tbsv_64
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbsv_64)(hipblasHandle_t    handle,
                                      hipblasFillMode_t  uplo,
                                      hipblasOperation_t transA,
                                      hipblasDiagType_t  diag,
                                      int64_t            m,
                                      int64_t            k,
                                      const T*           A,
                                      int64_t            lda,
                                      T*                 x,
                                      int64_t            incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbsvBatched_64)(hipblasHandle_t    handle,
                                             hipblasFillMode_t  uplo,
                                             hipblasOperation_t transA,
                                             hipblasDiagType_t  diag,
                                             int64_t            m,
                                             int64_t            k,
                                             const T* const     A[],
                                             int64_t            lda,
                                             T* const           x[],
                                             int64_t            incx,
                                             int64_t            batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTbsvStridedBatched_64)(hipblasHandle_t    handle,
                                                    hipblasFillMode_t  uplo,
                                                    hipblasOperation_t transA,
                                                    hipblasDiagType_t  diag,
                                                    int64_t            m,
                                                    int64_t            k,
                                                    const T*           A,
                                                    int64_t            lda,
                                                    hipblasStride      strideA,
                                                    T*                 x,
                                                    int64_t            incx,
                                                    hipblasStride      stridex,
                                                    int64_t            batchCount);

    MAP2CF_D64(hipblasTbsv, float, hipblasStbsv);
    MAP2CF_D64(hipblasTbsv, double, hipblasDtbsv);
    MAP2CF_D64_V2(hipblasTbsv, hipblasComplex, hipblasCtbsv);
    MAP2CF_D64_V2(hipblasTbsv, hipblasDoubleComplex, hipblasZtbsv);

    MAP2CF_D64(hipblasTbsvBatched, float, hipblasStbsvBatched);
    MAP2CF_D64(hipblasTbsvBatched, double, hipblasDtbsvBatched);
    MAP2CF_D64_V2(hipblasTbsvBatched, hipblasComplex, hipblasCtbsvBatched);
    MAP2CF_D64_V2(hipblasTbsvBatched, hipblasDoubleComplex, hipblasZtbsvBatched);

    MAP2CF_D64(hipblasTbsvStridedBatched, float, hipblasStbsvStridedBatched);
    MAP2CF_D64(hipblasTbsvStridedBatched, double, hipblasDtbsvStridedBatched);
    MAP2CF_D64_V2(hipblasTbsvStridedBatched, hipblasComplex, hipblasCtbsvStridedBatched);
    MAP2CF_D64_V2(hipblasTbsvStridedBatched, hipblasDoubleComplex, hipblasZtbsvStridedBatched);

    // tpmv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpmv)(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   hipblasDiagType_t  diag,
                                   int                m,
                                   const T*           AP,
                                   T*                 x,
                                   int                incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpmvBatched)(hipblasHandle_t    handle,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          hipblasDiagType_t  diag,
                                          int                m,
                                          const T* const     AP[],
                                          T* const           x[],
                                          int                incx,
                                          int                batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpmvStridedBatched)(hipblasHandle_t    handle,
                                                 hipblasFillMode_t  uplo,
                                                 hipblasOperation_t transA,
                                                 hipblasDiagType_t  diag,
                                                 int                m,
                                                 const T*           AP,
                                                 hipblasStride      strideAP,
                                                 T*                 x,
                                                 int                incx,
                                                 hipblasStride      stridex,
                                                 int                batchCount);

    // tpmv_64
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpmv_64)(hipblasHandle_t    handle,
                                      hipblasFillMode_t  uplo,
                                      hipblasOperation_t transA,
                                      hipblasDiagType_t  diag,
                                      int64_t            m,
                                      const T*           AP,
                                      T*                 x,
                                      int64_t            incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpmvBatched_64)(hipblasHandle_t    handle,
                                             hipblasFillMode_t  uplo,
                                             hipblasOperation_t transA,
                                             hipblasDiagType_t  diag,
                                             int64_t            m,
                                             const T* const     AP[],
                                             T* const           x[],
                                             int64_t            incx,
                                             int64_t            batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpmvStridedBatched_64)(hipblasHandle_t    handle,
                                                    hipblasFillMode_t  uplo,
                                                    hipblasOperation_t transA,
                                                    hipblasDiagType_t  diag,
                                                    int64_t            m,
                                                    const T*           AP,
                                                    hipblasStride      strideAP,
                                                    T*                 x,
                                                    int64_t            incx,
                                                    hipblasStride      stridex,
                                                    int64_t            batchCount);

    MAP2CF_D64(hipblasTpmv, float, hipblasStpmv);
    MAP2CF_D64(hipblasTpmv, double, hipblasDtpmv);
    MAP2CF_D64_V2(hipblasTpmv, hipblasComplex, hipblasCtpmv);
    MAP2CF_D64_V2(hipblasTpmv, hipblasDoubleComplex, hipblasZtpmv);

    MAP2CF_D64(hipblasTpmvBatched, float, hipblasStpmvBatched);
    MAP2CF_D64(hipblasTpmvBatched, double, hipblasDtpmvBatched);
    MAP2CF_D64_V2(hipblasTpmvBatched, hipblasComplex, hipblasCtpmvBatched);
    MAP2CF_D64_V2(hipblasTpmvBatched, hipblasDoubleComplex, hipblasZtpmvBatched);

    MAP2CF_D64(hipblasTpmvStridedBatched, float, hipblasStpmvStridedBatched);
    MAP2CF_D64(hipblasTpmvStridedBatched, double, hipblasDtpmvStridedBatched);
    MAP2CF_D64_V2(hipblasTpmvStridedBatched, hipblasComplex, hipblasCtpmvStridedBatched);
    MAP2CF_D64_V2(hipblasTpmvStridedBatched, hipblasDoubleComplex, hipblasZtpmvStridedBatched);

    // tpsv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpsv)(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   hipblasDiagType_t  diag,
                                   int                m,
                                   const T*           AP,
                                   T*                 x,
                                   int                incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpsvBatched)(hipblasHandle_t    handle,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          hipblasDiagType_t  diag,
                                          int                m,
                                          const T* const     AP[],
                                          T* const           x[],
                                          int                incx,
                                          int                batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpsvStridedBatched)(hipblasHandle_t    handle,
                                                 hipblasFillMode_t  uplo,
                                                 hipblasOperation_t transA,
                                                 hipblasDiagType_t  diag,
                                                 int                m,
                                                 const T*           AP,
                                                 hipblasStride      strideAP,
                                                 T*                 x,
                                                 int                incx,
                                                 hipblasStride      stridex,
                                                 int                batchCount);

    // tpsv_64
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpsv_64)(hipblasHandle_t    handle,
                                      hipblasFillMode_t  uplo,
                                      hipblasOperation_t transA,
                                      hipblasDiagType_t  diag,
                                      int64_t            m,
                                      const T*           AP,
                                      T*                 x,
                                      int64_t            incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpsvBatched_64)(hipblasHandle_t    handle,
                                             hipblasFillMode_t  uplo,
                                             hipblasOperation_t transA,
                                             hipblasDiagType_t  diag,
                                             int64_t            m,
                                             const T* const     AP[],
                                             T* const           x[],
                                             int64_t            incx,
                                             int64_t            batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTpsvStridedBatched_64)(hipblasHandle_t    handle,
                                                    hipblasFillMode_t  uplo,
                                                    hipblasOperation_t transA,
                                                    hipblasDiagType_t  diag,
                                                    int64_t            m,
                                                    const T*           AP,
                                                    hipblasStride      strideAP,
                                                    T*                 x,
                                                    int64_t            incx,
                                                    hipblasStride      stridex,
                                                    int64_t            batchCount);

    MAP2CF_D64(hipblasTpsv, float, hipblasStpsv);
    MAP2CF_D64(hipblasTpsv, double, hipblasDtpsv);
    MAP2CF_D64_V2(hipblasTpsv, hipblasComplex, hipblasCtpsv);
    MAP2CF_D64_V2(hipblasTpsv, hipblasDoubleComplex, hipblasZtpsv);

    MAP2CF_D64(hipblasTpsvBatched, float, hipblasStpsvBatched);
    MAP2CF_D64(hipblasTpsvBatched, double, hipblasDtpsvBatched);
    MAP2CF_D64_V2(hipblasTpsvBatched, hipblasComplex, hipblasCtpsvBatched);
    MAP2CF_D64_V2(hipblasTpsvBatched, hipblasDoubleComplex, hipblasZtpsvBatched);

    MAP2CF_D64(hipblasTpsvStridedBatched, float, hipblasStpsvStridedBatched);
    MAP2CF_D64(hipblasTpsvStridedBatched, double, hipblasDtpsvStridedBatched);
    MAP2CF_D64_V2(hipblasTpsvStridedBatched, hipblasComplex, hipblasCtpsvStridedBatched);
    MAP2CF_D64_V2(hipblasTpsvStridedBatched, hipblasDoubleComplex, hipblasZtpsvStridedBatched);

    // trmv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmv)(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   hipblasDiagType_t  diag,
                                   int                m,
                                   const T*           A,
                                   int                lda,
                                   T*                 x,
                                   int                incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmvBatched)(hipblasHandle_t    handle,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          hipblasDiagType_t  diag,
                                          int                m,
                                          const T* const     A[],
                                          int                lda,
                                          T* const           x[],
                                          int                incx,
                                          int                batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmvStridedBatched)(hipblasHandle_t    handle,
                                                 hipblasFillMode_t  uplo,
                                                 hipblasOperation_t transA,
                                                 hipblasDiagType_t  diag,
                                                 int                m,
                                                 const T*           A,
                                                 int                lda,
                                                 hipblasStride      stride_a,
                                                 T*                 x,
                                                 int                incx,
                                                 hipblasStride      stride_x,
                                                 int                batch_count);

    // trmv_64
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmv_64)(hipblasHandle_t    handle,
                                      hipblasFillMode_t  uplo,
                                      hipblasOperation_t transA,
                                      hipblasDiagType_t  diag,
                                      int64_t            m,
                                      const T*           A,
                                      int64_t            lda,
                                      T*                 x,
                                      int64_t            incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmvBatched_64)(hipblasHandle_t    handle,
                                             hipblasFillMode_t  uplo,
                                             hipblasOperation_t transA,
                                             hipblasDiagType_t  diag,
                                             int64_t            m,
                                             const T* const     A[],
                                             int64_t            lda,
                                             T* const           x[],
                                             int64_t            incx,
                                             int64_t            batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmvStridedBatched_64)(hipblasHandle_t    handle,
                                                    hipblasFillMode_t  uplo,
                                                    hipblasOperation_t transA,
                                                    hipblasDiagType_t  diag,
                                                    int64_t            m,
                                                    const T*           A,
                                                    int64_t            lda,
                                                    hipblasStride      stride_a,
                                                    T*                 x,
                                                    int64_t            incx,
                                                    hipblasStride      stride_x,
                                                    int64_t            batch_count);

    MAP2CF_D64(hipblasTrmv, float, hipblasStrmv);
    MAP2CF_D64(hipblasTrmv, double, hipblasDtrmv);
    MAP2CF_D64_V2(hipblasTrmv, hipblasComplex, hipblasCtrmv);
    MAP2CF_D64_V2(hipblasTrmv, hipblasDoubleComplex, hipblasZtrmv);

    MAP2CF_D64(hipblasTrmvBatched, float, hipblasStrmvBatched);
    MAP2CF_D64(hipblasTrmvBatched, double, hipblasDtrmvBatched);
    MAP2CF_D64_V2(hipblasTrmvBatched, hipblasComplex, hipblasCtrmvBatched);
    MAP2CF_D64_V2(hipblasTrmvBatched, hipblasDoubleComplex, hipblasZtrmvBatched);

    MAP2CF_D64(hipblasTrmvStridedBatched, float, hipblasStrmvStridedBatched);
    MAP2CF_D64(hipblasTrmvStridedBatched, double, hipblasDtrmvStridedBatched);
    MAP2CF_D64_V2(hipblasTrmvStridedBatched, hipblasComplex, hipblasCtrmvStridedBatched);
    MAP2CF_D64_V2(hipblasTrmvStridedBatched, hipblasDoubleComplex, hipblasZtrmvStridedBatched);

    // trsv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsv)(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   hipblasDiagType_t  diag,
                                   int                m,
                                   const T*           A,
                                   int                lda,
                                   T*                 x,
                                   int                incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsvBatched)(hipblasHandle_t    handle,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          hipblasDiagType_t  diag,
                                          int                m,
                                          const T* const     A[],
                                          int                lda,
                                          T* const           x[],
                                          int                incx,
                                          int                batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsvStridedBatched)(hipblasHandle_t    handle,
                                                 hipblasFillMode_t  uplo,
                                                 hipblasOperation_t transA,
                                                 hipblasDiagType_t  diag,
                                                 int                m,
                                                 const T*           A,
                                                 int                lda,
                                                 hipblasStride      strideA,
                                                 T*                 x,
                                                 int                incx,
                                                 hipblasStride      stridex,
                                                 int                batch_count);

    // trsv_64
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsv_64)(hipblasHandle_t    handle,
                                      hipblasFillMode_t  uplo,
                                      hipblasOperation_t transA,
                                      hipblasDiagType_t  diag,
                                      int64_t            m,
                                      const T*           A,
                                      int64_t            lda,
                                      T*                 x,
                                      int64_t            incx);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsvBatched_64)(hipblasHandle_t    handle,
                                             hipblasFillMode_t  uplo,
                                             hipblasOperation_t transA,
                                             hipblasDiagType_t  diag,
                                             int64_t            m,
                                             const T* const     A[],
                                             int64_t            lda,
                                             T* const           x[],
                                             int64_t            incx,
                                             int64_t            batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsvStridedBatched_64)(hipblasHandle_t    handle,
                                                    hipblasFillMode_t  uplo,
                                                    hipblasOperation_t transA,
                                                    hipblasDiagType_t  diag,
                                                    int64_t            m,
                                                    const T*           A,
                                                    int64_t            lda,
                                                    hipblasStride      strideA,
                                                    T*                 x,
                                                    int64_t            incx,
                                                    hipblasStride      stridex,
                                                    int64_t            batch_count);

    MAP2CF_D64(hipblasTrsv, float, hipblasStrsv);
    MAP2CF_D64(hipblasTrsv, double, hipblasDtrsv);
    MAP2CF_D64_V2(hipblasTrsv, hipblasComplex, hipblasCtrsv);
    MAP2CF_D64_V2(hipblasTrsv, hipblasDoubleComplex, hipblasZtrsv);

    MAP2CF_D64(hipblasTrsvBatched, float, hipblasStrsvBatched);
    MAP2CF_D64(hipblasTrsvBatched, double, hipblasDtrsvBatched);
    MAP2CF_D64_V2(hipblasTrsvBatched, hipblasComplex, hipblasCtrsvBatched);
    MAP2CF_D64_V2(hipblasTrsvBatched, hipblasDoubleComplex, hipblasZtrsvBatched);

    MAP2CF_D64(hipblasTrsvStridedBatched, float, hipblasStrsvStridedBatched);
    MAP2CF_D64(hipblasTrsvStridedBatched, double, hipblasDtrsvStridedBatched);
    MAP2CF_D64_V2(hipblasTrsvStridedBatched, hipblasComplex, hipblasCtrsvStridedBatched);
    MAP2CF_D64_V2(hipblasTrsvStridedBatched, hipblasDoubleComplex, hipblasZtrsvStridedBatched);

    // gbmv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGbmv)(hipblasHandle_t    handle,
                                   hipblasOperation_t transA,
                                   int                m,
                                   int                n,
                                   int                kl,
                                   int                ku,
                                   const T*           alpha,
                                   const T*           A,
                                   int                lda,
                                   const T*           x,
                                   int                incx,
                                   const T*           beta,
                                   T*                 y,
                                   int                incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGbmvBatched)(hipblasHandle_t    handle,
                                          hipblasOperation_t transA,
                                          int                m,
                                          int                n,
                                          int                kl,
                                          int                ku,
                                          const T*           alpha,
                                          const T* const     A[],
                                          int                lda,
                                          const T* const     x[],
                                          int                incx,
                                          const T*           beta,
                                          T* const           y[],
                                          int                incy,
                                          int                batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGbmvStridedBatched)(hipblasHandle_t    handle,
                                                 hipblasOperation_t transA,
                                                 int                m,
                                                 int                n,
                                                 int                kl,
                                                 int                ku,
                                                 const T*           alpha,
                                                 const T*           A,
                                                 int                lda,
                                                 hipblasStride      stride_a,
                                                 const T*           x,
                                                 int                incx,
                                                 hipblasStride      stride_x,
                                                 const T*           beta,
                                                 T*                 y,
                                                 int                incy,
                                                 hipblasStride      stride_y,
                                                 int                batch_count);

    // gbmv_64
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGbmv_64)(hipblasHandle_t    handle,
                                      hipblasOperation_t transA,
                                      int64_t            m,
                                      int64_t            n,
                                      int64_t            kl,
                                      int64_t            ku,
                                      const T*           alpha,
                                      const T*           A,
                                      int64_t            lda,
                                      const T*           x,
                                      int64_t            incx,
                                      const T*           beta,
                                      T*                 y,
                                      int64_t            incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGbmvBatched_64)(hipblasHandle_t    handle,
                                             hipblasOperation_t transA,
                                             int64_t            m,
                                             int64_t            n,
                                             int64_t            kl,
                                             int64_t            ku,
                                             const T*           alpha,
                                             const T* const     A[],
                                             int64_t            lda,
                                             const T* const     x[],
                                             int64_t            incx,
                                             const T*           beta,
                                             T* const           y[],
                                             int64_t            incy,
                                             int64_t            batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGbmvStridedBatched_64)(hipblasHandle_t    handle,
                                                    hipblasOperation_t transA,
                                                    int64_t            m,
                                                    int64_t            n,
                                                    int64_t            kl,
                                                    int64_t            ku,
                                                    const T*           alpha,
                                                    const T*           A,
                                                    int64_t            lda,
                                                    hipblasStride      stride_a,
                                                    const T*           x,
                                                    int64_t            incx,
                                                    hipblasStride      stride_x,
                                                    const T*           beta,
                                                    T*                 y,
                                                    int64_t            incy,
                                                    hipblasStride      stride_y,
                                                    int64_t            batch_count);

    MAP2CF_D64(hipblasGbmv, float, hipblasSgbmv);
    MAP2CF_D64(hipblasGbmv, double, hipblasDgbmv);
    MAP2CF_D64_V2(hipblasGbmv, hipblasComplex, hipblasCgbmv);
    MAP2CF_D64_V2(hipblasGbmv, hipblasDoubleComplex, hipblasZgbmv);

    MAP2CF_D64(hipblasGbmvBatched, float, hipblasSgbmvBatched);
    MAP2CF_D64(hipblasGbmvBatched, double, hipblasDgbmvBatched);
    MAP2CF_D64_V2(hipblasGbmvBatched, hipblasComplex, hipblasCgbmvBatched);
    MAP2CF_D64_V2(hipblasGbmvBatched, hipblasDoubleComplex, hipblasZgbmvBatched);

    MAP2CF_D64(hipblasGbmvStridedBatched, float, hipblasSgbmvStridedBatched);
    MAP2CF_D64(hipblasGbmvStridedBatched, double, hipblasDgbmvStridedBatched);
    MAP2CF_D64_V2(hipblasGbmvStridedBatched, hipblasComplex, hipblasCgbmvStridedBatched);
    MAP2CF_D64_V2(hipblasGbmvStridedBatched, hipblasDoubleComplex, hipblasZgbmvStridedBatched);

    // gemv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemv)(hipblasHandle_t    handle,
                                   hipblasOperation_t transA,
                                   int                m,
                                   int                n,
                                   const T*           alpha,
                                   const T*           A,
                                   int                lda,
                                   const T*           x,
                                   int                incx,
                                   const T*           beta,
                                   T*                 y,
                                   int                incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemvBatched)(hipblasHandle_t    handle,
                                          hipblasOperation_t transA,
                                          int                m,
                                          int                n,
                                          const T*           alpha,
                                          const T* const     A[],
                                          int                lda,
                                          const T* const     x[],
                                          int                incx,
                                          const T*           beta,
                                          T* const           y[],
                                          int                incy,
                                          int                batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemvStridedBatched)(hipblasHandle_t    handle,
                                                 hipblasOperation_t transA,
                                                 int                m,
                                                 int                n,
                                                 const T*           alpha,
                                                 const T*           A,
                                                 int                lda,
                                                 hipblasStride      stride_a,
                                                 const T*           x,
                                                 int                incx,
                                                 hipblasStride      stride_x,
                                                 const T*           beta,
                                                 T*                 y,
                                                 int                incy,
                                                 hipblasStride      stride_y,
                                                 int                batch_count);

    // gemv
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemv_64)(hipblasHandle_t    handle,
                                      hipblasOperation_t transA,
                                      int64_t            m,
                                      int64_t            n,
                                      const T*           alpha,
                                      const T*           A,
                                      int64_t            lda,
                                      const T*           x,
                                      int64_t            incx,
                                      const T*           beta,
                                      T*                 y,
                                      int64_t            incy);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemvBatched_64)(hipblasHandle_t    handle,
                                             hipblasOperation_t transA,
                                             int64_t            m,
                                             int64_t            n,
                                             const T*           alpha,
                                             const T* const     A[],
                                             int64_t            lda,
                                             const T* const     x[],
                                             int64_t            incx,
                                             const T*           beta,
                                             T* const           y[],
                                             int64_t            incy,
                                             int64_t            batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemvStridedBatched_64)(hipblasHandle_t    handle,
                                                    hipblasOperation_t transA,
                                                    int64_t            m,
                                                    int64_t            n,
                                                    const T*           alpha,
                                                    const T*           A,
                                                    int64_t            lda,
                                                    hipblasStride      stride_a,
                                                    const T*           x,
                                                    int64_t            incx,
                                                    hipblasStride      stride_x,
                                                    const T*           beta,
                                                    T*                 y,
                                                    int64_t            incy,
                                                    hipblasStride      stride_y,
                                                    int64_t            batch_count);

    MAP2CF_D64(hipblasGemv, float, hipblasSgemv);
    MAP2CF_D64(hipblasGemv, double, hipblasDgemv);
    MAP2CF_D64_V2(hipblasGemv, hipblasComplex, hipblasCgemv);
    MAP2CF_D64_V2(hipblasGemv, hipblasDoubleComplex, hipblasZgemv);

    MAP2CF_D64(hipblasGemvBatched, float, hipblasSgemvBatched);
    MAP2CF_D64(hipblasGemvBatched, double, hipblasDgemvBatched);
    MAP2CF_D64_V2(hipblasGemvBatched, hipblasComplex, hipblasCgemvBatched);
    MAP2CF_D64_V2(hipblasGemvBatched, hipblasDoubleComplex, hipblasZgemvBatched);

    MAP2CF_D64(hipblasGemvStridedBatched, float, hipblasSgemvStridedBatched);
    MAP2CF_D64(hipblasGemvStridedBatched, double, hipblasDgemvStridedBatched);
    MAP2CF_D64_V2(hipblasGemvStridedBatched, hipblasComplex, hipblasCgemvStridedBatched);
    MAP2CF_D64_V2(hipblasGemvStridedBatched, hipblasDoubleComplex, hipblasZgemvStridedBatched);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemm)(hipblasHandle_t    handle,
                                   hipblasOperation_t transA,
                                   hipblasOperation_t transB,
                                   int                m,
                                   int                n,
                                   int                k,
                                   const T*           alpha,
                                   const T*           A,
                                   int                lda,
                                   const T*           B,
                                   int                ldb,
                                   const T*           beta,
                                   T*                 C,
                                   int                ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemmStridedBatched)(hipblasHandle_t    handle,
                                                 hipblasOperation_t transA,
                                                 hipblasOperation_t transB,
                                                 int                m,
                                                 int                n,
                                                 int                k,
                                                 const T*           alpha,
                                                 const T*           A,
                                                 int                lda,
                                                 int                bsa,
                                                 const T*           B,
                                                 int                ldb,
                                                 int                bsb,
                                                 const T*           beta,
                                                 T*                 C,
                                                 int                ldc,
                                                 int                bsc,
                                                 int                batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemmBatched)(hipblasHandle_t    handle,
                                          hipblasOperation_t transA,
                                          hipblasOperation_t transB,
                                          int                m,
                                          int                n,
                                          int                k,
                                          const T*           alpha,
                                          const T* const     A[],
                                          int                lda,
                                          const T* const     B[],
                                          int                ldb,
                                          const T*           beta,
                                          T* const           C[],
                                          int                ldc,
                                          int                batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemm_64)(hipblasHandle_t    handle,
                                      hipblasOperation_t transA,
                                      hipblasOperation_t transB,
                                      int64_t            m,
                                      int64_t            n,
                                      int64_t            k,
                                      const T*           alpha,
                                      const T*           A,
                                      int64_t            lda,
                                      const T*           B,
                                      int64_t            ldb,
                                      const T*           beta,
                                      T*                 C,
                                      int64_t            ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemmStridedBatched_64)(hipblasHandle_t    handle,
                                                    hipblasOperation_t transA,
                                                    hipblasOperation_t transB,
                                                    int64_t            m,
                                                    int64_t            n,
                                                    int64_t            k,
                                                    const T*           alpha,
                                                    const T*           A,
                                                    int64_t            lda,
                                                    int64_t            bsa,
                                                    const T*           B,
                                                    int64_t            ldb,
                                                    int64_t            bsb,
                                                    const T*           beta,
                                                    T*                 C,
                                                    int64_t            ldc,
                                                    int64_t            bsc,
                                                    int64_t            batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGemmBatched_64)(hipblasHandle_t    handle,
                                             hipblasOperation_t transA,
                                             hipblasOperation_t transB,
                                             int64_t            m,
                                             int64_t            n,
                                             int64_t            k,
                                             const T*           alpha,
                                             const T* const     A[],
                                             int64_t            lda,
                                             const T* const     B[],
                                             int64_t            ldb,
                                             const T*           beta,
                                             T* const           C[],
                                             int64_t            ldc,
                                             int64_t            batch_count);

    MAP2CF_D64(hipblasGemm, hipblasHalf, hipblasHgemm);
    MAP2CF_D64(hipblasGemm, float, hipblasSgemm);
    MAP2CF_D64(hipblasGemm, double, hipblasDgemm);
    MAP2CF_D64_V2(hipblasGemm, hipblasComplex, hipblasCgemm);
    MAP2CF_D64_V2(hipblasGemm, hipblasDoubleComplex, hipblasZgemm);

    MAP2CF_D64(hipblasGemmBatched, hipblasHalf, hipblasHgemmBatched);
    MAP2CF_D64(hipblasGemmBatched, float, hipblasSgemmBatched);
    MAP2CF_D64(hipblasGemmBatched, double, hipblasDgemmBatched);
    MAP2CF_D64_V2(hipblasGemmBatched, hipblasComplex, hipblasCgemmBatched);
    MAP2CF_D64_V2(hipblasGemmBatched, hipblasDoubleComplex, hipblasZgemmBatched);

    MAP2CF_D64(hipblasGemmStridedBatched, hipblasHalf, hipblasHgemmStridedBatched);
    MAP2CF_D64(hipblasGemmStridedBatched, float, hipblasSgemmStridedBatched);
    MAP2CF_D64(hipblasGemmStridedBatched, double, hipblasDgemmStridedBatched);
    MAP2CF_D64_V2(hipblasGemmStridedBatched, hipblasComplex, hipblasCgemmStridedBatched);
    MAP2CF_D64_V2(hipblasGemmStridedBatched, hipblasDoubleComplex, hipblasZgemmStridedBatched);

    // herk
    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerk)(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   int                n,
                                   int                k,
                                   const U*           alpha,
                                   const T*           A,
                                   int                lda,
                                   const U*           beta,
                                   T*                 C,
                                   int                ldc);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerkBatched)(hipblasHandle_t    handle,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          int                n,
                                          int                k,
                                          const U*           alpha,
                                          const T* const     A[],
                                          int                lda,
                                          const U*           beta,
                                          T* const           C[],
                                          int                ldc,
                                          int                batchCount);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerkStridedBatched)(hipblasHandle_t    handle,
                                                 hipblasFillMode_t  uplo,
                                                 hipblasOperation_t transA,
                                                 int                n,
                                                 int                k,
                                                 const U*           alpha,
                                                 const T*           A,
                                                 int                lda,
                                                 hipblasStride      strideA,
                                                 const U*           beta,
                                                 T*                 C,
                                                 int                ldc,
                                                 hipblasStride      strideC,
                                                 int                batchCount);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerk_64)(hipblasHandle_t    handle,
                                      hipblasFillMode_t  uplo,
                                      hipblasOperation_t transA,
                                      int64_t            n,
                                      int64_t            k,
                                      const U*           alpha,
                                      const T*           A,
                                      int64_t            lda,
                                      const U*           beta,
                                      T*                 C,
                                      int64_t            ldc);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerkBatched_64)(hipblasHandle_t    handle,
                                             hipblasFillMode_t  uplo,
                                             hipblasOperation_t transA,
                                             int64_t            n,
                                             int64_t            k,
                                             const U*           alpha,
                                             const T* const     A[],
                                             int64_t            lda,
                                             const U*           beta,
                                             T* const           C[],
                                             int64_t            ldc,
                                             int64_t            batchCount);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerkStridedBatched_64)(hipblasHandle_t    handle,
                                                    hipblasFillMode_t  uplo,
                                                    hipblasOperation_t transA,
                                                    int64_t            n,
                                                    int64_t            k,
                                                    const U*           alpha,
                                                    const T*           A,
                                                    int64_t            lda,
                                                    hipblasStride      strideA,
                                                    const U*           beta,
                                                    T*                 C,
                                                    int64_t            ldc,
                                                    hipblasStride      strideC,
                                                    int64_t            batchCount);

    MAP2CF_D64_V2(hipblasHerk, hipblasComplex, float, hipblasCherk);
    MAP2CF_D64_V2(hipblasHerk, hipblasDoubleComplex, double, hipblasZherk);

    MAP2CF_D64_V2(hipblasHerkBatched, hipblasComplex, float, hipblasCherkBatched);
    MAP2CF_D64_V2(hipblasHerkBatched, hipblasDoubleComplex, double, hipblasZherkBatched);

    MAP2CF_D64_V2(hipblasHerkStridedBatched, hipblasComplex, float, hipblasCherkStridedBatched);
    MAP2CF_D64_V2(hipblasHerkStridedBatched,
                  hipblasDoubleComplex,
                  double,
                  hipblasZherkStridedBatched);

    // her2k
    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer2k)(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    int                n,
                                    int                k,
                                    const T*           alpha,
                                    const T*           A,
                                    int                lda,
                                    const T*           B,
                                    int                ldb,
                                    const U*           beta,
                                    T*                 C,
                                    int                ldc);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer2kBatched)(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           int                n,
                                           int                k,
                                           const T*           alpha,
                                           const T* const     A[],
                                           int                lda,
                                           const T* const     B[],
                                           int                ldb,
                                           const U*           beta,
                                           T* const           C[],
                                           int                ldc,
                                           int                batchCount);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer2kStridedBatched)(hipblasHandle_t    handle,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  int                n,
                                                  int                k,
                                                  const T*           alpha,
                                                  const T*           A,
                                                  int                lda,
                                                  hipblasStride      strideA,
                                                  const T*           B,
                                                  int                ldb,
                                                  hipblasStride      strideB,
                                                  const U*           beta,
                                                  T*                 C,
                                                  int                ldc,
                                                  hipblasStride      strideC,
                                                  int                batchCount);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer2k_64)(hipblasHandle_t    handle,
                                       hipblasFillMode_t  uplo,
                                       hipblasOperation_t transA,
                                       int64_t            n,
                                       int64_t            k,
                                       const T*           alpha,
                                       const T*           A,
                                       int64_t            lda,
                                       const T*           B,
                                       int64_t            ldb,
                                       const U*           beta,
                                       T*                 C,
                                       int64_t            ldc);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer2kBatched_64)(hipblasHandle_t    handle,
                                              hipblasFillMode_t  uplo,
                                              hipblasOperation_t transA,
                                              int64_t            n,
                                              int64_t            k,
                                              const T*           alpha,
                                              const T* const     A[],
                                              int64_t            lda,
                                              const T* const     B[],
                                              int64_t            ldb,
                                              const U*           beta,
                                              T* const           C[],
                                              int64_t            ldc,
                                              int64_t            batchCount);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHer2kStridedBatched_64)(hipblasHandle_t    handle,
                                                     hipblasFillMode_t  uplo,
                                                     hipblasOperation_t transA,
                                                     int64_t            n,
                                                     int64_t            k,
                                                     const T*           alpha,
                                                     const T*           A,
                                                     int64_t            lda,
                                                     hipblasStride      strideA,
                                                     const T*           B,
                                                     int64_t            ldb,
                                                     hipblasStride      strideB,
                                                     const U*           beta,
                                                     T*                 C,
                                                     int64_t            ldc,
                                                     hipblasStride      strideC,
                                                     int64_t            batchCount);

    MAP2CF_D64_V2(hipblasHer2k, hipblasComplex, float, hipblasCher2k);
    MAP2CF_D64_V2(hipblasHer2k, hipblasDoubleComplex, double, hipblasZher2k);

    MAP2CF_D64_V2(hipblasHer2kBatched, hipblasComplex, float, hipblasCher2kBatched);
    MAP2CF_D64_V2(hipblasHer2kBatched, hipblasDoubleComplex, double, hipblasZher2kBatched);

    MAP2CF_D64_V2(hipblasHer2kStridedBatched, hipblasComplex, float, hipblasCher2kStridedBatched);
    MAP2CF_D64_V2(hipblasHer2kStridedBatched,
                  hipblasDoubleComplex,
                  double,
                  hipblasZher2kStridedBatched);

    // herkx
    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerkx)(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    int                n,
                                    int                k,
                                    const T*           alpha,
                                    const T*           A,
                                    int                lda,
                                    const T*           B,
                                    int                ldb,
                                    const U*           beta,
                                    T*                 C,
                                    int                ldc);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerkxBatched)(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           int                n,
                                           int                k,
                                           const T*           alpha,
                                           const T* const     A[],
                                           int                lda,
                                           const T* const     B[],
                                           int                ldb,
                                           const U*           beta,
                                           T* const           C[],
                                           int                ldc,
                                           int                batchCount);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerkxStridedBatched)(hipblasHandle_t    handle,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  int                n,
                                                  int                k,
                                                  const T*           alpha,
                                                  const T*           A,
                                                  int                lda,
                                                  hipblasStride      strideA,
                                                  const T*           B,
                                                  int                ldb,
                                                  hipblasStride      strideB,
                                                  const U*           beta,
                                                  T*                 C,
                                                  int                ldc,
                                                  hipblasStride      strideC,
                                                  int                batchCount);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerkx_64)(hipblasHandle_t    handle,
                                       hipblasFillMode_t  uplo,
                                       hipblasOperation_t transA,
                                       int64_t            n,
                                       int64_t            k,
                                       const T*           alpha,
                                       const T*           A,
                                       int64_t            lda,
                                       const T*           B,
                                       int64_t            ldb,
                                       const U*           beta,
                                       T*                 C,
                                       int64_t            ldc);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerkxBatched_64)(hipblasHandle_t    handle,
                                              hipblasFillMode_t  uplo,
                                              hipblasOperation_t transA,
                                              int64_t            n,
                                              int64_t            k,
                                              const T*           alpha,
                                              const T* const     A[],
                                              int64_t            lda,
                                              const T* const     B[],
                                              int64_t            ldb,
                                              const U*           beta,
                                              T* const           C[],
                                              int64_t            ldc,
                                              int64_t            batchCount);

    template <typename T, typename U, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHerkxStridedBatched_64)(hipblasHandle_t    handle,
                                                     hipblasFillMode_t  uplo,
                                                     hipblasOperation_t transA,
                                                     int64_t            n,
                                                     int64_t            k,
                                                     const T*           alpha,
                                                     const T*           A,
                                                     int64_t            lda,
                                                     hipblasStride      strideA,
                                                     const T*           B,
                                                     int64_t            ldb,
                                                     hipblasStride      strideB,
                                                     const U*           beta,
                                                     T*                 C,
                                                     int64_t            ldc,
                                                     hipblasStride      strideC,
                                                     int64_t            batchCount);

    MAP2CF_D64_V2(hipblasHerkx, hipblasComplex, float, hipblasCherkx);
    MAP2CF_D64_V2(hipblasHerkx, hipblasDoubleComplex, double, hipblasZherkx);

    MAP2CF_D64_V2(hipblasHerkxBatched, hipblasComplex, float, hipblasCherkxBatched);
    MAP2CF_D64_V2(hipblasHerkxBatched, hipblasDoubleComplex, double, hipblasZherkxBatched);

    MAP2CF_D64_V2(hipblasHerkxStridedBatched, hipblasComplex, float, hipblasCherkxStridedBatched);
    MAP2CF_D64_V2(hipblasHerkxStridedBatched,
                  hipblasDoubleComplex,
                  double,
                  hipblasZherkxStridedBatched);

    // symm
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymm)(hipblasHandle_t   handle,
                                   hipblasSideMode_t side,
                                   hipblasFillMode_t uplo,
                                   int               m,
                                   int               n,
                                   const T*          alpha,
                                   const T*          A,
                                   int               lda,
                                   const T*          B,
                                   int               ldb,
                                   const T*          beta,
                                   T*                C,
                                   int               ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymmBatched)(hipblasHandle_t   handle,
                                          hipblasSideMode_t side,
                                          hipblasFillMode_t uplo,
                                          int               m,
                                          int               n,
                                          const T*          alpha,
                                          const T* const    A[],
                                          int               lda,
                                          const T* const    B[],
                                          int               ldb,
                                          const T*          beta,
                                          T* const          C[],
                                          int               ldc,
                                          int               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymmStridedBatched)(hipblasHandle_t   handle,
                                                 hipblasSideMode_t side,
                                                 hipblasFillMode_t uplo,
                                                 int               m,
                                                 int               n,
                                                 const T*          alpha,
                                                 const T*          A,
                                                 int               lda,
                                                 hipblasStride     strideA,
                                                 const T*          B,
                                                 int               ldb,
                                                 hipblasStride     strideB,
                                                 const T*          beta,
                                                 T*                C,
                                                 int               ldc,
                                                 hipblasStride     strideC,
                                                 int               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymm_64)(hipblasHandle_t   handle,
                                      hipblasSideMode_t side,
                                      hipblasFillMode_t uplo,
                                      int64_t           m,
                                      int64_t           n,
                                      const T*          alpha,
                                      const T*          A,
                                      int64_t           lda,
                                      const T*          B,
                                      int64_t           ldb,
                                      const T*          beta,
                                      T*                C,
                                      int64_t           ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymmBatched_64)(hipblasHandle_t   handle,
                                             hipblasSideMode_t side,
                                             hipblasFillMode_t uplo,
                                             int64_t           m,
                                             int64_t           n,
                                             const T*          alpha,
                                             const T* const    A[],
                                             int64_t           lda,
                                             const T* const    B[],
                                             int64_t           ldb,
                                             const T*          beta,
                                             T* const          C[],
                                             int64_t           ldc,
                                             int64_t           batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSymmStridedBatched_64)(hipblasHandle_t   handle,
                                                    hipblasSideMode_t side,
                                                    hipblasFillMode_t uplo,
                                                    int64_t           m,
                                                    int64_t           n,
                                                    const T*          alpha,
                                                    const T*          A,
                                                    int64_t           lda,
                                                    hipblasStride     strideA,
                                                    const T*          B,
                                                    int64_t           ldb,
                                                    hipblasStride     strideB,
                                                    const T*          beta,
                                                    T*                C,
                                                    int64_t           ldc,
                                                    hipblasStride     strideC,
                                                    int64_t           batchCount);

    MAP2CF_D64(hipblasSymm, float, hipblasSsymm);
    MAP2CF_D64(hipblasSymm, double, hipblasDsymm);
    MAP2CF_D64_V2(hipblasSymm, hipblasComplex, hipblasCsymm);
    MAP2CF_D64_V2(hipblasSymm, hipblasDoubleComplex, hipblasZsymm);

    MAP2CF_D64(hipblasSymmBatched, float, hipblasSsymmBatched);
    MAP2CF_D64(hipblasSymmBatched, double, hipblasDsymmBatched);
    MAP2CF_D64_V2(hipblasSymmBatched, hipblasComplex, hipblasCsymmBatched);
    MAP2CF_D64_V2(hipblasSymmBatched, hipblasDoubleComplex, hipblasZsymmBatched);

    MAP2CF_D64(hipblasSymmStridedBatched, float, hipblasSsymmStridedBatched);
    MAP2CF_D64(hipblasSymmStridedBatched, double, hipblasDsymmStridedBatched);
    MAP2CF_D64_V2(hipblasSymmStridedBatched, hipblasComplex, hipblasCsymmStridedBatched);
    MAP2CF_D64_V2(hipblasSymmStridedBatched, hipblasDoubleComplex, hipblasZsymmStridedBatched);

    // syrk
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrk)(hipblasHandle_t    handle,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   int                n,
                                   int                k,
                                   const T*           alpha,
                                   const T*           A,
                                   int                lda,
                                   const T*           beta,
                                   T*                 C,
                                   int                ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrkBatched)(hipblasHandle_t    handle,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          int                n,
                                          int                k,
                                          const T*           alpha,
                                          const T* const     A[],
                                          int                lda,
                                          const T*           beta,
                                          T* const           C[],
                                          int                ldc,
                                          int                batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrkStridedBatched)(hipblasHandle_t    handle,
                                                 hipblasFillMode_t  uplo,
                                                 hipblasOperation_t transA,
                                                 int                n,
                                                 int                k,
                                                 const T*           alpha,
                                                 const T*           A,
                                                 int                lda,
                                                 hipblasStride      strideA,
                                                 const T*           beta,
                                                 T*                 C,
                                                 int                ldc,
                                                 hipblasStride      strideC,
                                                 int                batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrk_64)(hipblasHandle_t    handle,
                                      hipblasFillMode_t  uplo,
                                      hipblasOperation_t transA,
                                      int64_t            n,
                                      int64_t            k,
                                      const T*           alpha,
                                      const T*           A,
                                      int64_t            lda,
                                      const T*           beta,
                                      T*                 C,
                                      int64_t            ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrkBatched_64)(hipblasHandle_t    handle,
                                             hipblasFillMode_t  uplo,
                                             hipblasOperation_t transA,
                                             int64_t            n,
                                             int64_t            k,
                                             const T*           alpha,
                                             const T* const     A[],
                                             int64_t            lda,
                                             const T*           beta,
                                             T* const           C[],
                                             int64_t            ldc,
                                             int64_t            batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrkStridedBatched_64)(hipblasHandle_t    handle,
                                                    hipblasFillMode_t  uplo,
                                                    hipblasOperation_t transA,
                                                    int64_t            n,
                                                    int64_t            k,
                                                    const T*           alpha,
                                                    const T*           A,
                                                    int64_t            lda,
                                                    hipblasStride      strideA,
                                                    const T*           beta,
                                                    T*                 C,
                                                    int64_t            ldc,
                                                    hipblasStride      strideC,
                                                    int64_t            batchCount);

    MAP2CF_D64(hipblasSyrk, float, hipblasSsyrk);
    MAP2CF_D64(hipblasSyrk, double, hipblasDsyrk);
    MAP2CF_D64_V2(hipblasSyrk, hipblasComplex, hipblasCsyrk);
    MAP2CF_D64_V2(hipblasSyrk, hipblasDoubleComplex, hipblasZsyrk);

    MAP2CF_D64(hipblasSyrkBatched, float, hipblasSsyrkBatched);
    MAP2CF_D64(hipblasSyrkBatched, double, hipblasDsyrkBatched);
    MAP2CF_D64_V2(hipblasSyrkBatched, hipblasComplex, hipblasCsyrkBatched);
    MAP2CF_D64_V2(hipblasSyrkBatched, hipblasDoubleComplex, hipblasZsyrkBatched);

    MAP2CF_D64(hipblasSyrkStridedBatched, float, hipblasSsyrkStridedBatched);
    MAP2CF_D64(hipblasSyrkStridedBatched, double, hipblasDsyrkStridedBatched);
    MAP2CF_D64_V2(hipblasSyrkStridedBatched, hipblasComplex, hipblasCsyrkStridedBatched);
    MAP2CF_D64_V2(hipblasSyrkStridedBatched, hipblasDoubleComplex, hipblasZsyrkStridedBatched);

    // syr2k
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2k)(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    int                n,
                                    int                k,
                                    const T*           alpha,
                                    const T*           A,
                                    int                lda,
                                    const T*           B,
                                    int                ldb,
                                    const T*           beta,
                                    T*                 C,
                                    int                ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2kBatched)(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           int                n,
                                           int                k,
                                           const T*           alpha,
                                           const T* const     A[],
                                           int                lda,
                                           const T* const     B[],
                                           int                ldb,
                                           const T*           beta,
                                           T* const           C[],
                                           int                ldc,
                                           int                batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2kStridedBatched)(hipblasHandle_t    handle,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  int                n,
                                                  int                k,
                                                  const T*           alpha,
                                                  const T*           A,
                                                  int                lda,
                                                  hipblasStride      strideA,
                                                  const T*           B,
                                                  int                ldb,
                                                  hipblasStride      strideB,
                                                  const T*           beta,
                                                  T*                 C,
                                                  int                ldc,
                                                  hipblasStride      strideC,
                                                  int                batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2k_64)(hipblasHandle_t    handle,
                                       hipblasFillMode_t  uplo,
                                       hipblasOperation_t transA,
                                       int64_t            n,
                                       int64_t            k,
                                       const T*           alpha,
                                       const T*           A,
                                       int64_t            lda,
                                       const T*           B,
                                       int64_t            ldb,
                                       const T*           beta,
                                       T*                 C,
                                       int64_t            ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2kBatched_64)(hipblasHandle_t    handle,
                                              hipblasFillMode_t  uplo,
                                              hipblasOperation_t transA,
                                              int64_t            n,
                                              int64_t            k,
                                              const T*           alpha,
                                              const T* const     A[],
                                              int64_t            lda,
                                              const T* const     B[],
                                              int64_t            ldb,
                                              const T*           beta,
                                              T* const           C[],
                                              int64_t            ldc,
                                              int64_t            batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyr2kStridedBatched_64)(hipblasHandle_t    handle,
                                                     hipblasFillMode_t  uplo,
                                                     hipblasOperation_t transA,
                                                     int64_t            n,
                                                     int64_t            k,
                                                     const T*           alpha,
                                                     const T*           A,
                                                     int64_t            lda,
                                                     hipblasStride      strideA,
                                                     const T*           B,
                                                     int64_t            ldb,
                                                     hipblasStride      strideB,
                                                     const T*           beta,
                                                     T*                 C,
                                                     int64_t            ldc,
                                                     hipblasStride      strideC,
                                                     int64_t            batchCount);

    MAP2CF_D64(hipblasSyr2k, float, hipblasSsyr2k);
    MAP2CF_D64(hipblasSyr2k, double, hipblasDsyr2k);
    MAP2CF_D64_V2(hipblasSyr2k, hipblasComplex, hipblasCsyr2k);
    MAP2CF_D64_V2(hipblasSyr2k, hipblasDoubleComplex, hipblasZsyr2k);

    MAP2CF_D64(hipblasSyr2kBatched, float, hipblasSsyr2kBatched);
    MAP2CF_D64(hipblasSyr2kBatched, double, hipblasDsyr2kBatched);
    MAP2CF_D64_V2(hipblasSyr2kBatched, hipblasComplex, hipblasCsyr2kBatched);
    MAP2CF_D64_V2(hipblasSyr2kBatched, hipblasDoubleComplex, hipblasZsyr2kBatched);

    MAP2CF_D64(hipblasSyr2kStridedBatched, float, hipblasSsyr2kStridedBatched);
    MAP2CF_D64(hipblasSyr2kStridedBatched, double, hipblasDsyr2kStridedBatched);
    MAP2CF_D64_V2(hipblasSyr2kStridedBatched, hipblasComplex, hipblasCsyr2kStridedBatched);
    MAP2CF_D64_V2(hipblasSyr2kStridedBatched, hipblasDoubleComplex, hipblasZsyr2kStridedBatched);

    // syrkx
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrkx)(hipblasHandle_t    handle,
                                    hipblasFillMode_t  uplo,
                                    hipblasOperation_t transA,
                                    int                n,
                                    int                k,
                                    const T*           alpha,
                                    const T*           A,
                                    int                lda,
                                    const T*           B,
                                    int                ldb,
                                    const T*           beta,
                                    T*                 C,
                                    int                ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrkxBatched)(hipblasHandle_t    handle,
                                           hipblasFillMode_t  uplo,
                                           hipblasOperation_t transA,
                                           int                n,
                                           int                k,
                                           const T*           alpha,
                                           const T* const     A[],
                                           int                lda,
                                           const T* const     B[],
                                           int                ldb,
                                           const T*           beta,
                                           T* const           C[],
                                           int                ldc,
                                           int                batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrkxStridedBatched)(hipblasHandle_t    handle,
                                                  hipblasFillMode_t  uplo,
                                                  hipblasOperation_t transA,
                                                  int                n,
                                                  int                k,
                                                  const T*           alpha,
                                                  const T*           A,
                                                  int                lda,
                                                  hipblasStride      strideA,
                                                  const T*           B,
                                                  int                ldb,
                                                  hipblasStride      strideB,
                                                  const T*           beta,
                                                  T*                 C,
                                                  int                ldc,
                                                  hipblasStride      strideC,
                                                  int                batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrkx_64)(hipblasHandle_t    handle,
                                       hipblasFillMode_t  uplo,
                                       hipblasOperation_t transA,
                                       int64_t            n,
                                       int64_t            k,
                                       const T*           alpha,
                                       const T*           A,
                                       int64_t            lda,
                                       const T*           B,
                                       int64_t            ldb,
                                       const T*           beta,
                                       T*                 C,
                                       int64_t            ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrkxBatched_64)(hipblasHandle_t    handle,
                                              hipblasFillMode_t  uplo,
                                              hipblasOperation_t transA,
                                              int64_t            n,
                                              int64_t            k,
                                              const T*           alpha,
                                              const T* const     A[],
                                              int64_t            lda,
                                              const T* const     B[],
                                              int64_t            ldb,
                                              const T*           beta,
                                              T* const           C[],
                                              int64_t            ldc,
                                              int64_t            batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasSyrkxStridedBatched_64)(hipblasHandle_t    handle,
                                                     hipblasFillMode_t  uplo,
                                                     hipblasOperation_t transA,
                                                     int64_t            n,
                                                     int64_t            k,
                                                     const T*           alpha,
                                                     const T*           A,
                                                     int64_t            lda,
                                                     hipblasStride      strideA,
                                                     const T*           B,
                                                     int64_t            ldb,
                                                     hipblasStride      strideB,
                                                     const T*           beta,
                                                     T*                 C,
                                                     int64_t            ldc,
                                                     hipblasStride      strideC,
                                                     int64_t            batchCount);

    MAP2CF_D64(hipblasSyrkx, float, hipblasSsyrkx);
    MAP2CF_D64(hipblasSyrkx, double, hipblasDsyrkx);
    MAP2CF_D64_V2(hipblasSyrkx, hipblasComplex, hipblasCsyrkx);
    MAP2CF_D64_V2(hipblasSyrkx, hipblasDoubleComplex, hipblasZsyrkx);

    MAP2CF_D64(hipblasSyrkxBatched, float, hipblasSsyrkxBatched);
    MAP2CF_D64(hipblasSyrkxBatched, double, hipblasDsyrkxBatched);
    MAP2CF_D64_V2(hipblasSyrkxBatched, hipblasComplex, hipblasCsyrkxBatched);
    MAP2CF_D64_V2(hipblasSyrkxBatched, hipblasDoubleComplex, hipblasZsyrkxBatched);

    MAP2CF_D64(hipblasSyrkxStridedBatched, float, hipblasSsyrkxStridedBatched);
    MAP2CF_D64(hipblasSyrkxStridedBatched, double, hipblasDsyrkxStridedBatched);
    MAP2CF_D64_V2(hipblasSyrkxStridedBatched, hipblasComplex, hipblasCsyrkxStridedBatched);
    MAP2CF_D64_V2(hipblasSyrkxStridedBatched, hipblasDoubleComplex, hipblasZsyrkxStridedBatched);

    // geam
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGeam)(hipblasHandle_t    handle,
                                   hipblasOperation_t transA,
                                   hipblasOperation_t transB,
                                   int                m,
                                   int                n,
                                   const T*           alpha,
                                   const T*           A,
                                   int                lda,
                                   const T*           beta,
                                   const T*           B,
                                   int                ldb,
                                   T*                 C,
                                   int                ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGeamBatched)(hipblasHandle_t    handle,
                                          hipblasOperation_t transA,
                                          hipblasOperation_t transB,
                                          int                m,
                                          int                n,
                                          const T*           alpha,
                                          const T* const     A[],
                                          int                lda,
                                          const T*           beta,
                                          const T* const     B[],
                                          int                ldb,
                                          T* const           C[],
                                          int                ldc,
                                          int                batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGeamStridedBatched)(hipblasHandle_t    handle,
                                                 hipblasOperation_t transA,
                                                 hipblasOperation_t transB,
                                                 int                m,
                                                 int                n,
                                                 const T*           alpha,
                                                 const T*           A,
                                                 int                lda,
                                                 hipblasStride      strideA,
                                                 const T*           beta,
                                                 const T*           B,
                                                 int                ldb,
                                                 hipblasStride      strideB,
                                                 T*                 C,
                                                 int                ldc,
                                                 hipblasStride      strideC,
                                                 int                batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGeam_64)(hipblasHandle_t    handle,
                                      hipblasOperation_t transA,
                                      hipblasOperation_t transB,
                                      int64_t            m,
                                      int64_t            n,
                                      const T*           alpha,
                                      const T*           A,
                                      int64_t            lda,
                                      const T*           beta,
                                      const T*           B,
                                      int64_t            ldb,
                                      T*                 C,
                                      int64_t            ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGeamBatched_64)(hipblasHandle_t    handle,
                                             hipblasOperation_t transA,
                                             hipblasOperation_t transB,
                                             int64_t            m,
                                             int64_t            n,
                                             const T*           alpha,
                                             const T* const     A[],
                                             int64_t            lda,
                                             const T*           beta,
                                             const T* const     B[],
                                             int64_t            ldb,
                                             T* const           C[],
                                             int64_t            ldc,
                                             int64_t            batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGeamStridedBatched_64)(hipblasHandle_t    handle,
                                                    hipblasOperation_t transA,
                                                    hipblasOperation_t transB,
                                                    int64_t            m,
                                                    int64_t            n,
                                                    const T*           alpha,
                                                    const T*           A,
                                                    int64_t            lda,
                                                    hipblasStride      strideA,
                                                    const T*           beta,
                                                    const T*           B,
                                                    int64_t            ldb,
                                                    hipblasStride      strideB,
                                                    T*                 C,
                                                    int64_t            ldc,
                                                    hipblasStride      strideC,
                                                    int64_t            batchCount);

    MAP2CF_D64(hipblasGeam, float, hipblasSgeam);
    MAP2CF_D64(hipblasGeam, double, hipblasDgeam);
    MAP2CF_D64_V2(hipblasGeam, hipblasComplex, hipblasCgeam);
    MAP2CF_D64_V2(hipblasGeam, hipblasDoubleComplex, hipblasZgeam);

    MAP2CF_D64(hipblasGeamBatched, float, hipblasSgeamBatched);
    MAP2CF_D64(hipblasGeamBatched, double, hipblasDgeamBatched);
    MAP2CF_D64_V2(hipblasGeamBatched, hipblasComplex, hipblasCgeamBatched);
    MAP2CF_D64_V2(hipblasGeamBatched, hipblasDoubleComplex, hipblasZgeamBatched);

    MAP2CF_D64(hipblasGeamStridedBatched, float, hipblasSgeamStridedBatched);
    MAP2CF_D64(hipblasGeamStridedBatched, double, hipblasDgeamStridedBatched);
    MAP2CF_D64_V2(hipblasGeamStridedBatched, hipblasComplex, hipblasCgeamStridedBatched);
    MAP2CF_D64_V2(hipblasGeamStridedBatched, hipblasDoubleComplex, hipblasZgeamStridedBatched);

    // hemm
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHemm)(hipblasHandle_t   handle,
                                   hipblasSideMode_t side,
                                   hipblasFillMode_t uplo,
                                   int               n,
                                   int               k,
                                   const T*          alpha,
                                   const T*          A,
                                   int               lda,
                                   const T*          B,
                                   int               ldb,
                                   const T*          beta,
                                   T*                C,
                                   int               ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHemmBatched)(hipblasHandle_t   handle,
                                          hipblasSideMode_t side,
                                          hipblasFillMode_t uplo,
                                          int               n,
                                          int               k,
                                          const T*          alpha,
                                          const T* const    A[],
                                          int               lda,
                                          const T* const    B[],
                                          int               ldb,
                                          const T*          beta,
                                          T* const          C[],
                                          int               ldc,
                                          int               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHemmStridedBatched)(hipblasHandle_t   handle,
                                                 hipblasSideMode_t side,
                                                 hipblasFillMode_t uplo,
                                                 int               n,
                                                 int               k,
                                                 const T*          alpha,
                                                 const T*          A,
                                                 int               lda,
                                                 hipblasStride     strideA,
                                                 const T*          B,
                                                 int               ldb,
                                                 hipblasStride     strideB,
                                                 const T*          beta,
                                                 T*                C,
                                                 int               ldc,
                                                 hipblasStride     strideC,
                                                 int               batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHemm_64)(hipblasHandle_t   handle,
                                      hipblasSideMode_t side,
                                      hipblasFillMode_t uplo,
                                      int64_t           n,
                                      int64_t           k,
                                      const T*          alpha,
                                      const T*          A,
                                      int64_t           lda,
                                      const T*          B,
                                      int64_t           ldb,
                                      const T*          beta,
                                      T*                C,
                                      int64_t           ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHemmBatched_64)(hipblasHandle_t   handle,
                                             hipblasSideMode_t side,
                                             hipblasFillMode_t uplo,
                                             int64_t           n,
                                             int64_t           k,
                                             const T*          alpha,
                                             const T* const    A[],
                                             int64_t           lda,
                                             const T* const    B[],
                                             int64_t           ldb,
                                             const T*          beta,
                                             T* const          C[],
                                             int64_t           ldc,
                                             int64_t           batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasHemmStridedBatched_64)(hipblasHandle_t   handle,
                                                    hipblasSideMode_t side,
                                                    hipblasFillMode_t uplo,
                                                    int64_t           n,
                                                    int64_t           k,
                                                    const T*          alpha,
                                                    const T*          A,
                                                    int64_t           lda,
                                                    hipblasStride     strideA,
                                                    const T*          B,
                                                    int64_t           ldb,
                                                    hipblasStride     strideB,
                                                    const T*          beta,
                                                    T*                C,
                                                    int64_t           ldc,
                                                    hipblasStride     strideC,
                                                    int64_t           batchCount);

    MAP2CF_D64_V2(hipblasHemm, hipblasComplex, hipblasChemm);
    MAP2CF_D64_V2(hipblasHemm, hipblasDoubleComplex, hipblasZhemm);

    MAP2CF_D64_V2(hipblasHemmBatched, hipblasComplex, hipblasChemmBatched);
    MAP2CF_D64_V2(hipblasHemmBatched, hipblasDoubleComplex, hipblasZhemmBatched);

    MAP2CF_D64_V2(hipblasHemmStridedBatched, hipblasComplex, hipblasChemmStridedBatched);
    MAP2CF_D64_V2(hipblasHemmStridedBatched, hipblasDoubleComplex, hipblasZhemmStridedBatched);

    // trmm
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmm)(hipblasHandle_t    handle,
                                   hipblasSideMode_t  side,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   hipblasDiagType_t  diag,
                                   int                m,
                                   int                n,
                                   const T*           alpha,
                                   const T*           A,
                                   int                lda,
                                   const T*           B,
                                   int                ldb,
                                   T*                 C,
                                   int                ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmmBatched)(hipblasHandle_t    handle,
                                          hipblasSideMode_t  side,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          hipblasDiagType_t  diag,
                                          int                m,
                                          int                n,
                                          const T*           alpha,
                                          const T* const     A[],
                                          int                lda,
                                          const T* const     B[],
                                          int                ldb,
                                          T* const           C[],
                                          int                ldc,
                                          int                batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmmStridedBatched)(hipblasHandle_t    handle,
                                                 hipblasSideMode_t  side,
                                                 hipblasFillMode_t  uplo,
                                                 hipblasOperation_t transA,
                                                 hipblasDiagType_t  diag,
                                                 int                m,
                                                 int                n,
                                                 const T*           alpha,
                                                 const T*           A,
                                                 int                lda,
                                                 hipblasStride      strideA,
                                                 const T*           B,
                                                 int                ldb,
                                                 hipblasStride      strideB,
                                                 T*                 C,
                                                 int                ldc,
                                                 hipblasStride      strideC,
                                                 int                batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmm_64)(hipblasHandle_t    handle,
                                      hipblasSideMode_t  side,
                                      hipblasFillMode_t  uplo,
                                      hipblasOperation_t transA,
                                      hipblasDiagType_t  diag,
                                      int64_t            m,
                                      int64_t            n,
                                      const T*           alpha,
                                      const T*           A,
                                      int64_t            lda,
                                      const T*           B,
                                      int64_t            ldb,
                                      T*                 C,
                                      int64_t            ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmmBatched_64)(hipblasHandle_t    handle,
                                             hipblasSideMode_t  side,
                                             hipblasFillMode_t  uplo,
                                             hipblasOperation_t transA,
                                             hipblasDiagType_t  diag,
                                             int64_t            m,
                                             int64_t            n,
                                             const T*           alpha,
                                             const T* const     A[],
                                             int64_t            lda,
                                             const T* const     B[],
                                             int64_t            ldb,
                                             T* const           C[],
                                             int64_t            ldc,
                                             int64_t            batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrmmStridedBatched_64)(hipblasHandle_t    handle,
                                                    hipblasSideMode_t  side,
                                                    hipblasFillMode_t  uplo,
                                                    hipblasOperation_t transA,
                                                    hipblasDiagType_t  diag,
                                                    int64_t            m,
                                                    int64_t            n,
                                                    const T*           alpha,
                                                    const T*           A,
                                                    int64_t            lda,
                                                    hipblasStride      strideA,
                                                    const T*           B,
                                                    int64_t            ldb,
                                                    hipblasStride      strideB,
                                                    T*                 C,
                                                    int64_t            ldc,
                                                    hipblasStride      strideC,
                                                    int64_t            batchCount);

    MAP2CF_D64(hipblasTrmm, float, hipblasStrmm);
    MAP2CF_D64(hipblasTrmm, double, hipblasDtrmm);
    MAP2CF_D64_V2(hipblasTrmm, hipblasComplex, hipblasCtrmm);
    MAP2CF_D64_V2(hipblasTrmm, hipblasDoubleComplex, hipblasZtrmm);

    MAP2CF_D64(hipblasTrmmBatched, float, hipblasStrmmBatched);
    MAP2CF_D64(hipblasTrmmBatched, double, hipblasDtrmmBatched);
    MAP2CF_D64_V2(hipblasTrmmBatched, hipblasComplex, hipblasCtrmmBatched);
    MAP2CF_D64_V2(hipblasTrmmBatched, hipblasDoubleComplex, hipblasZtrmmBatched);

    MAP2CF_D64(hipblasTrmmStridedBatched, float, hipblasStrmmStridedBatched);
    MAP2CF_D64(hipblasTrmmStridedBatched, double, hipblasDtrmmStridedBatched);
    MAP2CF_D64_V2(hipblasTrmmStridedBatched, hipblasComplex, hipblasCtrmmStridedBatched);
    MAP2CF_D64_V2(hipblasTrmmStridedBatched, hipblasDoubleComplex, hipblasZtrmmStridedBatched);

    // trsm
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsm)(hipblasHandle_t    handle,
                                   hipblasSideMode_t  side,
                                   hipblasFillMode_t  uplo,
                                   hipblasOperation_t transA,
                                   hipblasDiagType_t  diag,
                                   int                m,
                                   int                n,
                                   const T*           alpha,
                                   const T*           A,
                                   int                lda,
                                   T*                 B,
                                   int                ldb);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsmBatched)(hipblasHandle_t    handle,
                                          hipblasSideMode_t  side,
                                          hipblasFillMode_t  uplo,
                                          hipblasOperation_t transA,
                                          hipblasDiagType_t  diag,
                                          int                m,
                                          int                n,
                                          const T*           alpha,
                                          const T* const     A[],
                                          int                lda,
                                          T* const           B[],
                                          int                ldb,
                                          int                batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsmStridedBatched)(hipblasHandle_t    handle,
                                                 hipblasSideMode_t  side,
                                                 hipblasFillMode_t  uplo,
                                                 hipblasOperation_t transA,
                                                 hipblasDiagType_t  diag,
                                                 int                m,
                                                 int                n,
                                                 const T*           alpha,
                                                 const T*           A,
                                                 int                lda,
                                                 hipblasStride      strideA,
                                                 T*                 B,
                                                 int                ldb,
                                                 hipblasStride      strideB,
                                                 int                batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsm_64)(hipblasHandle_t    handle,
                                      hipblasSideMode_t  side,
                                      hipblasFillMode_t  uplo,
                                      hipblasOperation_t transA,
                                      hipblasDiagType_t  diag,
                                      int64_t            m,
                                      int64_t            n,
                                      const T*           alpha,
                                      const T*           A,
                                      int64_t            lda,
                                      T*                 B,
                                      int64_t            ldb);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsmBatched_64)(hipblasHandle_t    handle,
                                             hipblasSideMode_t  side,
                                             hipblasFillMode_t  uplo,
                                             hipblasOperation_t transA,
                                             hipblasDiagType_t  diag,
                                             int64_t            m,
                                             int64_t            n,
                                             const T*           alpha,
                                             const T* const     A[],
                                             int64_t            lda,
                                             T* const           B[],
                                             int64_t            ldb,
                                             int64_t            batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrsmStridedBatched_64)(hipblasHandle_t    handle,
                                                    hipblasSideMode_t  side,
                                                    hipblasFillMode_t  uplo,
                                                    hipblasOperation_t transA,
                                                    hipblasDiagType_t  diag,
                                                    int64_t            m,
                                                    int64_t            n,
                                                    const T*           alpha,
                                                    const T*           A,
                                                    int64_t            lda,
                                                    hipblasStride      strideA,
                                                    T*                 B,
                                                    int64_t            ldb,
                                                    hipblasStride      strideB,
                                                    int64_t            batch_count);

    MAP2CF_D64(hipblasTrsm, float, hipblasStrsm);
    MAP2CF_D64(hipblasTrsm, double, hipblasDtrsm);
    MAP2CF_D64_V2(hipblasTrsm, hipblasComplex, hipblasCtrsm);
    MAP2CF_D64_V2(hipblasTrsm, hipblasDoubleComplex, hipblasZtrsm);

    MAP2CF_D64(hipblasTrsmBatched, float, hipblasStrsmBatched);
    MAP2CF_D64(hipblasTrsmBatched, double, hipblasDtrsmBatched);
    MAP2CF_D64_V2(hipblasTrsmBatched, hipblasComplex, hipblasCtrsmBatched);
    MAP2CF_D64_V2(hipblasTrsmBatched, hipblasDoubleComplex, hipblasZtrsmBatched);

    MAP2CF_D64(hipblasTrsmStridedBatched, float, hipblasStrsmStridedBatched);
    MAP2CF_D64(hipblasTrsmStridedBatched, double, hipblasDtrsmStridedBatched);
    MAP2CF_D64_V2(hipblasTrsmStridedBatched, hipblasComplex, hipblasCtrsmStridedBatched);
    MAP2CF_D64_V2(hipblasTrsmStridedBatched, hipblasDoubleComplex, hipblasZtrsmStridedBatched);

    // dgmm
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDgmm)(hipblasHandle_t   handle,
                                   hipblasSideMode_t side,
                                   int               m,
                                   int               n,
                                   const T*          A,
                                   int               lda,
                                   const T*          x,
                                   int               incx,
                                   T*                C,
                                   int               ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDgmmBatched)(hipblasHandle_t   handle,
                                          hipblasSideMode_t side,
                                          int               m,
                                          int               n,
                                          const T* const    A[],
                                          int               lda,
                                          const T* const    x[],
                                          int               incx,
                                          T* const          C[],
                                          int               ldc,
                                          int               batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDgmmStridedBatched)(hipblasHandle_t   handle,
                                                 hipblasSideMode_t side,
                                                 int               m,
                                                 int               n,
                                                 const T*          A,
                                                 int               lda,
                                                 hipblasStride     stride_A,
                                                 const T*          x,
                                                 int               incx,
                                                 hipblasStride     stride_x,
                                                 T*                C,
                                                 int               ldc,
                                                 hipblasStride     stride_C,
                                                 int               batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDgmm_64)(hipblasHandle_t   handle,
                                      hipblasSideMode_t side,
                                      int64_t           m,
                                      int64_t           n,
                                      const T*          A,
                                      int64_t           lda,
                                      const T*          x,
                                      int64_t           incx,
                                      T*                C,
                                      int64_t           ldc);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDgmmBatched_64)(hipblasHandle_t   handle,
                                             hipblasSideMode_t side,
                                             int64_t           m,
                                             int64_t           n,
                                             const T* const    A[],
                                             int64_t           lda,
                                             const T* const    x[],
                                             int64_t           incx,
                                             T* const          C[],
                                             int64_t           ldc,
                                             int64_t           batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasDgmmStridedBatched_64)(hipblasHandle_t   handle,
                                                    hipblasSideMode_t side,
                                                    int64_t           m,
                                                    int64_t           n,
                                                    const T*          A,
                                                    int64_t           lda,
                                                    hipblasStride     stride_A,
                                                    const T*          x,
                                                    int64_t           incx,
                                                    hipblasStride     stride_x,
                                                    T*                C,
                                                    int64_t           ldc,
                                                    hipblasStride     stride_C,
                                                    int64_t           batch_count);

    MAP2CF_D64(hipblasDgmm, float, hipblasSdgmm);
    MAP2CF_D64(hipblasDgmm, double, hipblasDdgmm);
    MAP2CF_D64_V2(hipblasDgmm, hipblasComplex, hipblasCdgmm);
    MAP2CF_D64_V2(hipblasDgmm, hipblasDoubleComplex, hipblasZdgmm);

    MAP2CF_D64(hipblasDgmmBatched, float, hipblasSdgmmBatched);
    MAP2CF_D64(hipblasDgmmBatched, double, hipblasDdgmmBatched);
    MAP2CF_D64_V2(hipblasDgmmBatched, hipblasComplex, hipblasCdgmmBatched);
    MAP2CF_D64_V2(hipblasDgmmBatched, hipblasDoubleComplex, hipblasZdgmmBatched);

    MAP2CF_D64(hipblasDgmmStridedBatched, float, hipblasSdgmmStridedBatched);
    MAP2CF_D64(hipblasDgmmStridedBatched, double, hipblasDdgmmStridedBatched);
    MAP2CF_D64_V2(hipblasDgmmStridedBatched, hipblasComplex, hipblasCdgmmStridedBatched);
    MAP2CF_D64_V2(hipblasDgmmStridedBatched, hipblasDoubleComplex, hipblasZdgmmStridedBatched);

    // trtri
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrtri)(hipblasHandle_t   handle,
                                    hipblasFillMode_t uplo,
                                    hipblasDiagType_t diag,
                                    int               n,
                                    const T*          A,
                                    int               lda,
                                    T*                invA,
                                    int               ldinvA);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrtriBatched)(hipblasHandle_t   handle,
                                           hipblasFillMode_t uplo,
                                           hipblasDiagType_t diag,
                                           int               n,
                                           const T* const    A[],
                                           int               lda,
                                           T*                invA[],
                                           int               ldinvA,
                                           int               batch_count);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasTrtriStridedBatched)(hipblasHandle_t   handle,
                                                  hipblasFillMode_t uplo,
                                                  hipblasDiagType_t diag,
                                                  int               n,
                                                  const T*          A,
                                                  int               lda,
                                                  hipblasStride     stride_A,
                                                  T*                invA,
                                                  int               ldinvA,
                                                  hipblasStride     stride_invA,
                                                  int               batch_count);

    MAP2CF(hipblasTrtri, float, hipblasStrtri);
    MAP2CF(hipblasTrtri, double, hipblasDtrtri);
    MAP2CF_V2(hipblasTrtri, hipblasComplex, hipblasCtrtri);
    MAP2CF_V2(hipblasTrtri, hipblasDoubleComplex, hipblasZtrtri);

    MAP2CF(hipblasTrtriBatched, float, hipblasStrtriBatched);
    MAP2CF(hipblasTrtriBatched, double, hipblasDtrtriBatched);
    MAP2CF_V2(hipblasTrtriBatched, hipblasComplex, hipblasCtrtriBatched);
    MAP2CF_V2(hipblasTrtriBatched, hipblasDoubleComplex, hipblasZtrtriBatched);

    MAP2CF(hipblasTrtriStridedBatched, float, hipblasStrtriStridedBatched);
    MAP2CF(hipblasTrtriStridedBatched, double, hipblasDtrtriStridedBatched);
    MAP2CF_V2(hipblasTrtriStridedBatched, hipblasComplex, hipblasCtrtriStridedBatched);
    MAP2CF_V2(hipblasTrtriStridedBatched, hipblasDoubleComplex, hipblasZtrtriStridedBatched);

#ifdef __HIP_PLATFORM_SOLVER__

    // getrf
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGetrf)(
        hipblasHandle_t handle, const int n, T* A, const int lda, int* ipiv, int* info);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGetrfBatched)(hipblasHandle_t handle,
                                           const int       n,
                                           T* const        A[],
                                           const int       lda,
                                           int*            ipiv,
                                           int*            info,
                                           const int       batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGetrfStridedBatched)(hipblasHandle_t     handle,
                                                  const int           n,
                                                  T*                  A,
                                                  const int           lda,
                                                  const hipblasStride strideA,
                                                  int*                ipiv,
                                                  const hipblasStride strideP,
                                                  int*                info,
                                                  const int           batchCount);

    MAP2CF(hipblasGetrf, float, hipblasSgetrf);
    MAP2CF(hipblasGetrf, double, hipblasDgetrf);
    MAP2CF_V2(hipblasGetrf, hipblasComplex, hipblasCgetrf);
    MAP2CF_V2(hipblasGetrf, hipblasDoubleComplex, hipblasZgetrf);

    MAP2CF(hipblasGetrfBatched, float, hipblasSgetrfBatched);
    MAP2CF(hipblasGetrfBatched, double, hipblasDgetrfBatched);
    MAP2CF_V2(hipblasGetrfBatched, hipblasComplex, hipblasCgetrfBatched);
    MAP2CF_V2(hipblasGetrfBatched, hipblasDoubleComplex, hipblasZgetrfBatched);

    MAP2CF(hipblasGetrfStridedBatched, float, hipblasSgetrfStridedBatched);
    MAP2CF(hipblasGetrfStridedBatched, double, hipblasDgetrfStridedBatched);
    MAP2CF_V2(hipblasGetrfStridedBatched, hipblasComplex, hipblasCgetrfStridedBatched);
    MAP2CF_V2(hipblasGetrfStridedBatched, hipblasDoubleComplex, hipblasZgetrfStridedBatched);

    // getrs
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGetrs)(hipblasHandle_t          handle,
                                    const hipblasOperation_t trans,
                                    const int                n,
                                    const int                nrhs,
                                    T*                       A,
                                    const int                lda,
                                    const int*               ipiv,
                                    T*                       B,
                                    const int                ldb,
                                    int*                     info);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGetrsBatched)(hipblasHandle_t          handle,
                                           const hipblasOperation_t trans,
                                           const int                n,
                                           const int                nrhs,
                                           T* const                 A[],
                                           const int                lda,
                                           const int*               ipiv,
                                           T* const                 B[],
                                           const int                ldb,
                                           int*                     info,
                                           const int                batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGetrsStridedBatched)(hipblasHandle_t          handle,
                                                  const hipblasOperation_t trans,
                                                  const int                n,
                                                  const int                nrhs,
                                                  T*                       A,
                                                  const int                lda,
                                                  const hipblasStride      strideA,
                                                  const int*               ipiv,
                                                  const hipblasStride      strideP,
                                                  T*                       B,
                                                  const int                ldb,
                                                  const hipblasStride      strideB,
                                                  int*                     info,
                                                  const int                batchCount);

    MAP2CF(hipblasGetrs, float, hipblasSgetrs);
    MAP2CF(hipblasGetrs, double, hipblasDgetrs);
    MAP2CF_V2(hipblasGetrs, hipblasComplex, hipblasCgetrs);
    MAP2CF_V2(hipblasGetrs, hipblasDoubleComplex, hipblasZgetrs);

    MAP2CF(hipblasGetrsBatched, float, hipblasSgetrsBatched);
    MAP2CF(hipblasGetrsBatched, double, hipblasDgetrsBatched);
    MAP2CF_V2(hipblasGetrsBatched, hipblasComplex, hipblasCgetrsBatched);
    MAP2CF_V2(hipblasGetrsBatched, hipblasDoubleComplex, hipblasZgetrsBatched);

    MAP2CF(hipblasGetrsStridedBatched, float, hipblasSgetrsStridedBatched);
    MAP2CF(hipblasGetrsStridedBatched, double, hipblasDgetrsStridedBatched);
    MAP2CF_V2(hipblasGetrsStridedBatched, hipblasComplex, hipblasCgetrsStridedBatched);
    MAP2CF_V2(hipblasGetrsStridedBatched, hipblasDoubleComplex, hipblasZgetrsStridedBatched);

    // getri
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGetriBatched)(hipblasHandle_t handle,
                                           const int       n,
                                           T* const        A[],
                                           const int       lda,
                                           int*            ipiv,
                                           T* const        C[],
                                           const int       ldc,
                                           int*            info,
                                           const int       batchCount);

    MAP2CF(hipblasGetriBatched, float, hipblasSgetriBatched);
    MAP2CF(hipblasGetriBatched, double, hipblasDgetriBatched);
    MAP2CF_V2(hipblasGetriBatched, hipblasComplex, hipblasCgetriBatched);
    MAP2CF_V2(hipblasGetriBatched, hipblasDoubleComplex, hipblasZgetriBatched);

    // geqrf
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGeqrf)(
        hipblasHandle_t handle, const int m, const int n, T* A, const int lda, T* ipiv, int* info);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGeqrfBatched)(hipblasHandle_t handle,
                                           const int       m,
                                           const int       n,
                                           T* const        A[],
                                           const int       lda,
                                           T* const        ipiv[],
                                           int*            info,
                                           const int       batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGeqrfStridedBatched)(hipblasHandle_t     handle,
                                                  const int           m,
                                                  const int           n,
                                                  T*                  A,
                                                  const int           lda,
                                                  const hipblasStride strideA,
                                                  T*                  ipiv,
                                                  const hipblasStride strideP,
                                                  int*                info,
                                                  const int           batchCount);

    MAP2CF(hipblasGeqrf, float, hipblasSgeqrf);
    MAP2CF(hipblasGeqrf, double, hipblasDgeqrf);
    MAP2CF_V2(hipblasGeqrf, hipblasComplex, hipblasCgeqrf);
    MAP2CF_V2(hipblasGeqrf, hipblasDoubleComplex, hipblasZgeqrf);

    MAP2CF(hipblasGeqrfBatched, float, hipblasSgeqrfBatched);
    MAP2CF(hipblasGeqrfBatched, double, hipblasDgeqrfBatched);
    MAP2CF_V2(hipblasGeqrfBatched, hipblasComplex, hipblasCgeqrfBatched);
    MAP2CF_V2(hipblasGeqrfBatched, hipblasDoubleComplex, hipblasZgeqrfBatched);

    MAP2CF(hipblasGeqrfStridedBatched, float, hipblasSgeqrfStridedBatched);
    MAP2CF(hipblasGeqrfStridedBatched, double, hipblasDgeqrfStridedBatched);
    MAP2CF_V2(hipblasGeqrfStridedBatched, hipblasComplex, hipblasCgeqrfStridedBatched);
    MAP2CF_V2(hipblasGeqrfStridedBatched, hipblasDoubleComplex, hipblasZgeqrfStridedBatched);

    // gels
    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGels)(hipblasHandle_t    handle,
                                   hipblasOperation_t trans,
                                   const int          m,
                                   const int          n,
                                   const int          nrhs,
                                   T*                 A,
                                   const int          lda,
                                   T*                 B,
                                   const int          ldb,
                                   int*               info,
                                   int*               deviceInfo);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGelsBatched)(hipblasHandle_t    handle,
                                          hipblasOperation_t trans,
                                          const int          m,
                                          const int          n,
                                          const int          nrhs,
                                          T* const           A[],
                                          const int          lda,
                                          T* const           B[],
                                          const int          ldb,
                                          int*               info,
                                          int*               deviceInfo,
                                          const int          batchCount);

    template <typename T, bool FORTRAN = false>
    hipblasStatus_t (*hipblasGelsStridedBatched)(hipblasHandle_t     handle,
                                                 hipblasOperation_t  trans,
                                                 const int           m,
                                                 const int           n,
                                                 const int           nrhs,
                                                 T*                  A,
                                                 const int           lda,
                                                 const hipblasStride strideA,
                                                 T*                  B,
                                                 const int           ldb,
                                                 const hipblasStride strideB,
                                                 int*                info,
                                                 int*                deviceInfo,
                                                 const int           batchCount);

    MAP2CF(hipblasGels, float, hipblasSgels);
    MAP2CF(hipblasGels, double, hipblasDgels);
    MAP2CF_V2(hipblasGels, hipblasComplex, hipblasCgels);
    MAP2CF_V2(hipblasGels, hipblasDoubleComplex, hipblasZgels);

    MAP2CF(hipblasGelsBatched, float, hipblasSgelsBatched);
    MAP2CF(hipblasGelsBatched, double, hipblasDgelsBatched);
    MAP2CF_V2(hipblasGelsBatched, hipblasComplex, hipblasCgelsBatched);
    MAP2CF_V2(hipblasGelsBatched, hipblasDoubleComplex, hipblasZgelsBatched);

    MAP2CF(hipblasGelsStridedBatched, float, hipblasSgelsStridedBatched);
    MAP2CF(hipblasGelsStridedBatched, double, hipblasDgelsStridedBatched);
    MAP2CF_V2(hipblasGelsStridedBatched, hipblasComplex, hipblasCgelsStridedBatched);
    MAP2CF_V2(hipblasGelsStridedBatched, hipblasDoubleComplex, hipblasZgelsStridedBatched);

#endif
}

#endif // _ROCBLAS_HPP_
