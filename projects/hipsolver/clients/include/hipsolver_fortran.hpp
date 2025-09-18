/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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
 *
 * ************************************************************************ */

#pragma once

#include "hipsolver.h"

/*!\file
 *  This file interfaces with our Fortran LAPACK interface.
 */

/*
 * ============================================================================
 *     Fortran functions
 * ============================================================================
 */

extern "C" {

/* ==========
 *   LAPACK
 * ========== */

// orgbr/ungbr
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSorgbr_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverSideMode_t side,
                                                                     int                 m,
                                                                     int                 n,
                                                                     int                 k,
                                                                     float*              A,
                                                                     int                 lda,
                                                                     float*              tau,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDorgbr_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverSideMode_t side,
                                                                     int                 m,
                                                                     int                 n,
                                                                     int                 k,
                                                                     double*             A,
                                                                     int                 lda,
                                                                     double*             tau,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCungbr_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverSideMode_t side,
                                                                     int                 m,
                                                                     int                 n,
                                                                     int                 k,
                                                                     hipFloatComplex*    A,
                                                                     int                 lda,
                                                                     hipFloatComplex*    tau,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZungbr_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverSideMode_t side,
                                                                     int                 m,
                                                                     int                 n,
                                                                     int                 k,
                                                                     hipDoubleComplex*   A,
                                                                     int                 lda,
                                                                     hipDoubleComplex*   tau,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSorgbrFortran(hipsolverHandle_t   handle,
                                                          hipsolverSideMode_t side,
                                                          int                 m,
                                                          int                 n,
                                                          int                 k,
                                                          float*              A,
                                                          int                 lda,
                                                          float*              tau,
                                                          float*              work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDorgbrFortran(hipsolverHandle_t   handle,
                                                          hipsolverSideMode_t side,
                                                          int                 m,
                                                          int                 n,
                                                          int                 k,
                                                          double*             A,
                                                          int                 lda,
                                                          double*             tau,
                                                          double*             work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCungbrFortran(hipsolverHandle_t   handle,
                                                          hipsolverSideMode_t side,
                                                          int                 m,
                                                          int                 n,
                                                          int                 k,
                                                          hipFloatComplex*    A,
                                                          int                 lda,
                                                          hipFloatComplex*    tau,
                                                          hipFloatComplex*    work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZungbrFortran(hipsolverHandle_t   handle,
                                                          hipsolverSideMode_t side,
                                                          int                 m,
                                                          int                 n,
                                                          int                 k,
                                                          hipDoubleComplex*   A,
                                                          int                 lda,
                                                          hipDoubleComplex*   tau,
                                                          hipDoubleComplex*   work,
                                                          int                 lwork,
                                                          int*                devInfo);

// orgqr/ungqr
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSorgqr_bufferSizeFortran(
    hipsolverHandle_t handle, int m, int n, int k, float* A, int lda, float* tau, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDorgqr_bufferSizeFortran(
    hipsolverHandle_t handle, int m, int n, int k, double* A, int lda, double* tau, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCungqr_bufferSizeFortran(hipsolverHandle_t handle,
                                                                     int               m,
                                                                     int               n,
                                                                     int               k,
                                                                     hipFloatComplex*  A,
                                                                     int               lda,
                                                                     hipFloatComplex*  tau,
                                                                     int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZungqr_bufferSizeFortran(hipsolverHandle_t handle,
                                                                     int               m,
                                                                     int               n,
                                                                     int               k,
                                                                     hipDoubleComplex* A,
                                                                     int               lda,
                                                                     hipDoubleComplex* tau,
                                                                     int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSorgqrFortran(hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          int               k,
                                                          float*            A,
                                                          int               lda,
                                                          float*            tau,
                                                          float*            work,
                                                          int               lwork,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDorgqrFortran(hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          int               k,
                                                          double*           A,
                                                          int               lda,
                                                          double*           tau,
                                                          double*           work,
                                                          int               lwork,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCungqrFortran(hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          int               k,
                                                          hipFloatComplex*  A,
                                                          int               lda,
                                                          hipFloatComplex*  tau,
                                                          hipFloatComplex*  work,
                                                          int               lwork,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZungqrFortran(hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          int               k,
                                                          hipDoubleComplex* A,
                                                          int               lda,
                                                          hipDoubleComplex* tau,
                                                          hipDoubleComplex* work,
                                                          int               lwork,
                                                          int*              devInfo);

// orgtr/ungtr
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSorgtr_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     float*              A,
                                                                     int                 lda,
                                                                     float*              tau,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDorgtr_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     double*             A,
                                                                     int                 lda,
                                                                     double*             tau,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCungtr_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     hipFloatComplex*    A,
                                                                     int                 lda,
                                                                     hipFloatComplex*    tau,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZungtr_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     hipDoubleComplex*   A,
                                                                     int                 lda,
                                                                     hipDoubleComplex*   tau,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSorgtrFortran(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          float*              A,
                                                          int                 lda,
                                                          float*              tau,
                                                          float*              work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDorgtrFortran(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          double*             A,
                                                          int                 lda,
                                                          double*             tau,
                                                          double*             work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCungtrFortran(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipFloatComplex*    A,
                                                          int                 lda,
                                                          hipFloatComplex*    tau,
                                                          hipFloatComplex*    work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZungtrFortran(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipDoubleComplex*   A,
                                                          int                 lda,
                                                          hipDoubleComplex*   tau,
                                                          hipDoubleComplex*   work,
                                                          int                 lwork,
                                                          int*                devInfo);

// ormqr/unmqr
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSormqr_bufferSizeFortran(hipsolverHandle_t    handle,
                                                                     hipsolverSideMode_t  side,
                                                                     hipsolverOperation_t trans,
                                                                     int                  m,
                                                                     int                  n,
                                                                     int                  k,
                                                                     float*               A,
                                                                     int                  lda,
                                                                     float*               tau,
                                                                     float*               C,
                                                                     int                  ldc,
                                                                     int*                 lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDormqr_bufferSizeFortran(hipsolverHandle_t    handle,
                                                                     hipsolverSideMode_t  side,
                                                                     hipsolverOperation_t trans,
                                                                     int                  m,
                                                                     int                  n,
                                                                     int                  k,
                                                                     double*              A,
                                                                     int                  lda,
                                                                     double*              tau,
                                                                     double*              C,
                                                                     int                  ldc,
                                                                     int*                 lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCunmqr_bufferSizeFortran(hipsolverHandle_t    handle,
                                                                     hipsolverSideMode_t  side,
                                                                     hipsolverOperation_t trans,
                                                                     int                  m,
                                                                     int                  n,
                                                                     int                  k,
                                                                     hipFloatComplex*     A,
                                                                     int                  lda,
                                                                     hipFloatComplex*     tau,
                                                                     hipFloatComplex*     C,
                                                                     int                  ldc,
                                                                     int*                 lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZunmqr_bufferSizeFortran(hipsolverHandle_t    handle,
                                                                     hipsolverSideMode_t  side,
                                                                     hipsolverOperation_t trans,
                                                                     int                  m,
                                                                     int                  n,
                                                                     int                  k,
                                                                     hipDoubleComplex*    A,
                                                                     int                  lda,
                                                                     hipDoubleComplex*    tau,
                                                                     hipDoubleComplex*    C,
                                                                     int                  ldc,
                                                                     int*                 lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSormqrFortran(hipsolverHandle_t    handle,
                                                          hipsolverSideMode_t  side,
                                                          hipsolverOperation_t trans,
                                                          int                  m,
                                                          int                  n,
                                                          int                  k,
                                                          float*               A,
                                                          int                  lda,
                                                          float*               tau,
                                                          float*               C,
                                                          int                  ldc,
                                                          float*               work,
                                                          int                  lwork,
                                                          int*                 devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDormqrFortran(hipsolverHandle_t    handle,
                                                          hipsolverSideMode_t  side,
                                                          hipsolverOperation_t trans,
                                                          int                  m,
                                                          int                  n,
                                                          int                  k,
                                                          double*              A,
                                                          int                  lda,
                                                          double*              tau,
                                                          double*              C,
                                                          int                  ldc,
                                                          double*              work,
                                                          int                  lwork,
                                                          int*                 devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCunmqrFortran(hipsolverHandle_t    handle,
                                                          hipsolverSideMode_t  side,
                                                          hipsolverOperation_t trans,
                                                          int                  m,
                                                          int                  n,
                                                          int                  k,
                                                          hipFloatComplex*     A,
                                                          int                  lda,
                                                          hipFloatComplex*     tau,
                                                          hipFloatComplex*     C,
                                                          int                  ldc,
                                                          hipFloatComplex*     work,
                                                          int                  lwork,
                                                          int*                 devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZunmqrFortran(hipsolverHandle_t    handle,
                                                          hipsolverSideMode_t  side,
                                                          hipsolverOperation_t trans,
                                                          int                  m,
                                                          int                  n,
                                                          int                  k,
                                                          hipDoubleComplex*    A,
                                                          int                  lda,
                                                          hipDoubleComplex*    tau,
                                                          hipDoubleComplex*    C,
                                                          int                  ldc,
                                                          hipDoubleComplex*    work,
                                                          int                  lwork,
                                                          int*                 devInfo);

// ormtr/unmtr
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSormtr_bufferSizeFortran(hipsolverHandle_t    handle,
                                                                     hipsolverSideMode_t  side,
                                                                     hipsolverFillMode_t  uplo,
                                                                     hipsolverOperation_t trans,
                                                                     int                  m,
                                                                     int                  n,
                                                                     float*               A,
                                                                     int                  lda,
                                                                     float*               tau,
                                                                     float*               C,
                                                                     int                  ldc,
                                                                     int*                 lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDormtr_bufferSizeFortran(hipsolverHandle_t    handle,
                                                                     hipsolverSideMode_t  side,
                                                                     hipsolverFillMode_t  uplo,
                                                                     hipsolverOperation_t trans,
                                                                     int                  m,
                                                                     int                  n,
                                                                     double*              A,
                                                                     int                  lda,
                                                                     double*              tau,
                                                                     double*              C,
                                                                     int                  ldc,
                                                                     int*                 lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCunmtr_bufferSizeFortran(hipsolverHandle_t    handle,
                                                                     hipsolverSideMode_t  side,
                                                                     hipsolverFillMode_t  uplo,
                                                                     hipsolverOperation_t trans,
                                                                     int                  m,
                                                                     int                  n,
                                                                     hipFloatComplex*     A,
                                                                     int                  lda,
                                                                     hipFloatComplex*     tau,
                                                                     hipFloatComplex*     C,
                                                                     int                  ldc,
                                                                     int*                 lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZunmtr_bufferSizeFortran(hipsolverHandle_t    handle,
                                                                     hipsolverSideMode_t  side,
                                                                     hipsolverFillMode_t  uplo,
                                                                     hipsolverOperation_t trans,
                                                                     int                  m,
                                                                     int                  n,
                                                                     hipDoubleComplex*    A,
                                                                     int                  lda,
                                                                     hipDoubleComplex*    tau,
                                                                     hipDoubleComplex*    C,
                                                                     int                  ldc,
                                                                     int*                 lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSormtrFortran(hipsolverHandle_t    handle,
                                                          hipsolverSideMode_t  side,
                                                          hipsolverFillMode_t  uplo,
                                                          hipsolverOperation_t trans,
                                                          int                  m,
                                                          int                  n,
                                                          float*               A,
                                                          int                  lda,
                                                          float*               tau,
                                                          float*               C,
                                                          int                  ldc,
                                                          float*               work,
                                                          int                  lwork,
                                                          int*                 devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDormtrFortran(hipsolverHandle_t    handle,
                                                          hipsolverSideMode_t  side,
                                                          hipsolverFillMode_t  uplo,
                                                          hipsolverOperation_t trans,
                                                          int                  m,
                                                          int                  n,
                                                          double*              A,
                                                          int                  lda,
                                                          double*              tau,
                                                          double*              C,
                                                          int                  ldc,
                                                          double*              work,
                                                          int                  lwork,
                                                          int*                 devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCunmtrFortran(hipsolverHandle_t    handle,
                                                          hipsolverSideMode_t  side,
                                                          hipsolverFillMode_t  uplo,
                                                          hipsolverOperation_t trans,
                                                          int                  m,
                                                          int                  n,
                                                          hipFloatComplex*     A,
                                                          int                  lda,
                                                          hipFloatComplex*     tau,
                                                          hipFloatComplex*     C,
                                                          int                  ldc,
                                                          hipFloatComplex*     work,
                                                          int                  lwork,
                                                          int*                 devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZunmtrFortran(hipsolverHandle_t    handle,
                                                          hipsolverSideMode_t  side,
                                                          hipsolverFillMode_t  uplo,
                                                          hipsolverOperation_t trans,
                                                          int                  m,
                                                          int                  n,
                                                          hipDoubleComplex*    A,
                                                          int                  lda,
                                                          hipDoubleComplex*    tau,
                                                          hipDoubleComplex*    C,
                                                          int                  ldc,
                                                          hipDoubleComplex*    work,
                                                          int                  lwork,
                                                          int*                 devInfo);

// gebrd
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgebrd_bufferSizeFortran(hipsolverHandle_t handle,
                                                                     int               m,
                                                                     int               n,
                                                                     int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgebrd_bufferSizeFortran(hipsolverHandle_t handle,
                                                                     int               m,
                                                                     int               n,
                                                                     int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgebrd_bufferSizeFortran(hipsolverHandle_t handle,
                                                                     int               m,
                                                                     int               n,
                                                                     int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgebrd_bufferSizeFortran(hipsolverHandle_t handle,
                                                                     int               m,
                                                                     int               n,
                                                                     int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgebrdFortran(hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          float*            A,
                                                          int               lda,
                                                          float*            D,
                                                          float*            E,
                                                          float*            tauq,
                                                          float*            taup,
                                                          float*            work,
                                                          int               lwork,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgebrdFortran(hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          double*           A,
                                                          int               lda,
                                                          double*           D,
                                                          double*           E,
                                                          double*           tauq,
                                                          double*           taup,
                                                          double*           work,
                                                          int               lwork,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgebrdFortran(hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          hipFloatComplex*  A,
                                                          int               lda,
                                                          float*            D,
                                                          float*            E,
                                                          hipFloatComplex*  tauq,
                                                          hipFloatComplex*  taup,
                                                          hipFloatComplex*  work,
                                                          int               lwork,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgebrdFortran(hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          hipDoubleComplex* A,
                                                          int               lda,
                                                          double*           D,
                                                          double*           E,
                                                          hipDoubleComplex* tauq,
                                                          hipDoubleComplex* taup,
                                                          hipDoubleComplex* work,
                                                          int               lwork,
                                                          int*              devInfo);

// gels
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSSgels_bufferSizeFortran(hipsolverHandle_t handle,
                                                                     int               m,
                                                                     int               n,
                                                                     int               nrhs,
                                                                     float*            A,
                                                                     int               lda,
                                                                     float*            B,
                                                                     int               ldb,
                                                                     float*            X,
                                                                     int               ldx,
                                                                     size_t*           lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDDgels_bufferSizeFortran(hipsolverHandle_t handle,
                                                                     int               m,
                                                                     int               n,
                                                                     int               nrhs,
                                                                     double*           A,
                                                                     int               lda,
                                                                     double*           B,
                                                                     int               ldb,
                                                                     double*           X,
                                                                     int               ldx,
                                                                     size_t*           lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCCgels_bufferSizeFortran(hipsolverHandle_t handle,
                                                                     int               m,
                                                                     int               n,
                                                                     int               nrhs,
                                                                     hipFloatComplex*  A,
                                                                     int               lda,
                                                                     hipFloatComplex*  B,
                                                                     int               ldb,
                                                                     hipFloatComplex*  X,
                                                                     int               ldx,
                                                                     size_t*           lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZZgels_bufferSizeFortran(hipsolverHandle_t handle,
                                                                     int               m,
                                                                     int               n,
                                                                     int               nrhs,
                                                                     hipDoubleComplex* A,
                                                                     int               lda,
                                                                     hipDoubleComplex* B,
                                                                     int               ldb,
                                                                     hipDoubleComplex* X,
                                                                     int               ldx,
                                                                     size_t*           lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSSgelsFortran(hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          int               nrhs,
                                                          float*            A,
                                                          int               lda,
                                                          float*            B,
                                                          int               ldb,
                                                          float*            X,
                                                          int               ldx,
                                                          void*             work,
                                                          size_t            lwork,
                                                          int*              niters,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDDgelsFortran(hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          int               nrhs,
                                                          double*           A,
                                                          int               lda,
                                                          double*           B,
                                                          int               ldb,
                                                          double*           X,
                                                          int               ldx,
                                                          void*             work,
                                                          size_t            lwork,
                                                          int*              niters,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCCgelsFortran(hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          int               nrhs,
                                                          hipFloatComplex*  A,
                                                          int               lda,
                                                          hipFloatComplex*  B,
                                                          int               ldb,
                                                          hipFloatComplex*  X,
                                                          int               ldx,
                                                          void*             work,
                                                          size_t            lwork,
                                                          int*              niters,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZZgelsFortran(hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          int               nrhs,
                                                          hipDoubleComplex* A,
                                                          int               lda,
                                                          hipDoubleComplex* B,
                                                          int               ldb,
                                                          hipDoubleComplex* X,
                                                          int               ldx,
                                                          void*             work,
                                                          size_t            lwork,
                                                          int*              niters,
                                                          int*              devInfo);

// geqrf
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgeqrf_bufferSizeFortran(
    hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgeqrf_bufferSizeFortran(
    hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgeqrf_bufferSizeFortran(
    hipsolverHandle_t handle, int m, int n, hipFloatComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgeqrf_bufferSizeFortran(
    hipsolverHandle_t handle, int m, int n, hipDoubleComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgeqrfFortran(hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          float*            A,
                                                          int               lda,
                                                          float*            tau,
                                                          float*            work,
                                                          int               lwork,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgeqrfFortran(hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          double*           A,
                                                          int               lda,
                                                          double*           tau,
                                                          double*           work,
                                                          int               lwork,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgeqrfFortran(hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          hipFloatComplex*  A,
                                                          int               lda,
                                                          hipFloatComplex*  tau,
                                                          hipFloatComplex*  work,
                                                          int               lwork,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgeqrfFortran(hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          hipDoubleComplex* A,
                                                          int               lda,
                                                          hipDoubleComplex* tau,
                                                          hipDoubleComplex* work,
                                                          int               lwork,
                                                          int*              devInfo);

// gesv
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSSgesv_bufferSizeFortran(hipsolverHandle_t handle,
                                                                     int               n,
                                                                     int               nrhs,
                                                                     float*            A,
                                                                     int               lda,
                                                                     int*              devIpiv,
                                                                     float*            B,
                                                                     int               ldb,
                                                                     float*            X,
                                                                     int               ldx,
                                                                     size_t*           lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDDgesv_bufferSizeFortran(hipsolverHandle_t handle,
                                                                     int               n,
                                                                     int               nrhs,
                                                                     double*           A,
                                                                     int               lda,
                                                                     int*              devIpiv,
                                                                     double*           B,
                                                                     int               ldb,
                                                                     double*           X,
                                                                     int               ldx,
                                                                     size_t*           lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCCgesv_bufferSizeFortran(hipsolverHandle_t handle,
                                                                     int               n,
                                                                     int               nrhs,
                                                                     hipFloatComplex*  A,
                                                                     int               lda,
                                                                     int*              devIpiv,
                                                                     hipFloatComplex*  B,
                                                                     int               ldb,
                                                                     hipFloatComplex*  X,
                                                                     int               ldx,
                                                                     size_t*           lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZZgesv_bufferSizeFortran(hipsolverHandle_t handle,
                                                                     int               n,
                                                                     int               nrhs,
                                                                     hipDoubleComplex* A,
                                                                     int               lda,
                                                                     int*              devIpiv,
                                                                     hipDoubleComplex* B,
                                                                     int               ldb,
                                                                     hipDoubleComplex* X,
                                                                     int               ldx,
                                                                     size_t*           lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSSgesvFortran(hipsolverHandle_t handle,
                                                          int               n,
                                                          int               nrhs,
                                                          float*            A,
                                                          int               lda,
                                                          int*              devIpiv,
                                                          float*            B,
                                                          int               ldb,
                                                          float*            X,
                                                          int               ldx,
                                                          void*             work,
                                                          size_t            lwork,
                                                          int*              niters,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDDgesvFortran(hipsolverHandle_t handle,
                                                          int               n,
                                                          int               nrhs,
                                                          double*           A,
                                                          int               lda,
                                                          int*              devIpiv,
                                                          double*           B,
                                                          int               ldb,
                                                          double*           X,
                                                          int               ldx,
                                                          void*             work,
                                                          size_t            lwork,
                                                          int*              niters,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCCgesvFortran(hipsolverHandle_t handle,
                                                          int               n,
                                                          int               nrhs,
                                                          hipFloatComplex*  A,
                                                          int               lda,
                                                          int*              devIpiv,
                                                          hipFloatComplex*  B,
                                                          int               ldb,
                                                          hipFloatComplex*  X,
                                                          int               ldx,
                                                          void*             work,
                                                          size_t            lwork,
                                                          int*              niters,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZZgesvFortran(hipsolverHandle_t handle,
                                                          int               n,
                                                          int               nrhs,
                                                          hipDoubleComplex* A,
                                                          int               lda,
                                                          int*              devIpiv,
                                                          hipDoubleComplex* B,
                                                          int               ldb,
                                                          hipDoubleComplex* X,
                                                          int               ldx,
                                                          void*             work,
                                                          size_t            lwork,
                                                          int*              niters,
                                                          int*              devInfo);

// gesvd
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgesvd_bufferSizeFortran(
    hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgesvd_bufferSizeFortran(
    hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgesvd_bufferSizeFortran(
    hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgesvd_bufferSizeFortran(
    hipsolverHandle_t handle, signed char jobu, signed char jobv, int m, int n, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgesvdFortran(hipsolverHandle_t handle,
                                                          signed char       jobu,
                                                          signed char       jobv,
                                                          int               m,
                                                          int               n,
                                                          float*            A,
                                                          int               lda,
                                                          float*            S,
                                                          float*            U,
                                                          int               ldu,
                                                          float*            V,
                                                          int               ldv,
                                                          float*            work,
                                                          int               lwork,
                                                          float*            rwork,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgesvdFortran(hipsolverHandle_t handle,
                                                          signed char       jobu,
                                                          signed char       jobv,
                                                          int               m,
                                                          int               n,
                                                          double*           A,
                                                          int               lda,
                                                          double*           S,
                                                          double*           U,
                                                          int               ldu,
                                                          double*           V,
                                                          int               ldv,
                                                          double*           work,
                                                          int               lwork,
                                                          double*           rwork,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgesvdFortran(hipsolverHandle_t handle,
                                                          signed char       jobu,
                                                          signed char       jobv,
                                                          int               m,
                                                          int               n,
                                                          hipFloatComplex*  A,
                                                          int               lda,
                                                          float*            S,
                                                          hipFloatComplex*  U,
                                                          int               ldu,
                                                          hipFloatComplex*  V,
                                                          int               ldv,
                                                          hipFloatComplex*  work,
                                                          int               lwork,
                                                          float*            rwork,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgesvdFortran(hipsolverHandle_t handle,
                                                          signed char       jobu,
                                                          signed char       jobv,
                                                          int               m,
                                                          int               n,
                                                          hipDoubleComplex* A,
                                                          int               lda,
                                                          double*           S,
                                                          hipDoubleComplex* U,
                                                          int               ldu,
                                                          hipDoubleComplex* V,
                                                          int               ldv,
                                                          hipDoubleComplex* work,
                                                          int               lwork,
                                                          double*           rwork,
                                                          int*              devInfo);

// gesvdj
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgesvdj_bufferSizeFortran(hipsolverHandle_t     handle,
                                                                      hipsolverEigMode_t    jobz,
                                                                      int                   econ,
                                                                      int                   m,
                                                                      int                   n,
                                                                      const float*          A,
                                                                      int                   lda,
                                                                      const float*          S,
                                                                      const float*          U,
                                                                      int                   ldu,
                                                                      const float*          V,
                                                                      int                   ldv,
                                                                      int*                  lwork,
                                                                      hipsolverGesvdjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgesvdj_bufferSizeFortran(hipsolverHandle_t     handle,
                                                                      hipsolverEigMode_t    jobz,
                                                                      int                   econ,
                                                                      int                   m,
                                                                      int                   n,
                                                                      const double*         A,
                                                                      int                   lda,
                                                                      const double*         S,
                                                                      const double*         U,
                                                                      int                   ldu,
                                                                      const double*         V,
                                                                      int                   ldv,
                                                                      int*                  lwork,
                                                                      hipsolverGesvdjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgesvdj_bufferSizeFortran(hipsolverHandle_t      handle,
                                                                      hipsolverEigMode_t     jobz,
                                                                      int                    econ,
                                                                      int                    m,
                                                                      int                    n,
                                                                      const hipFloatComplex* A,
                                                                      int                    lda,
                                                                      const float*           S,
                                                                      const hipFloatComplex* U,
                                                                      int                    ldu,
                                                                      const hipFloatComplex* V,
                                                                      int                    ldv,
                                                                      int*                   lwork,
                                                                      hipsolverGesvdjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgesvdj_bufferSizeFortran(hipsolverHandle_t  handle,
                                                                      hipsolverEigMode_t jobz,
                                                                      int                econ,
                                                                      int                m,
                                                                      int                n,
                                                                      const hipDoubleComplex* A,
                                                                      int                     lda,
                                                                      const double*           S,
                                                                      const hipDoubleComplex* U,
                                                                      int                     ldu,
                                                                      const hipDoubleComplex* V,
                                                                      int                     ldv,
                                                                      int*                    lwork,
                                                                      hipsolverGesvdjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgesvdjFortran(hipsolverHandle_t     handle,
                                                           hipsolverEigMode_t    jobz,
                                                           int                   econ,
                                                           int                   m,
                                                           int                   n,
                                                           float*                A,
                                                           int                   lda,
                                                           float*                S,
                                                           float*                U,
                                                           int                   ldu,
                                                           float*                V,
                                                           int                   ldv,
                                                           float*                work,
                                                           int                   lwork,
                                                           int*                  devInfo,
                                                           hipsolverGesvdjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgesvdjFortran(hipsolverHandle_t     handle,
                                                           hipsolverEigMode_t    jobz,
                                                           int                   econ,
                                                           int                   m,
                                                           int                   n,
                                                           double*               A,
                                                           int                   lda,
                                                           double*               S,
                                                           double*               U,
                                                           int                   ldu,
                                                           double*               V,
                                                           int                   ldv,
                                                           double*               work,
                                                           int                   lwork,
                                                           int*                  devInfo,
                                                           hipsolverGesvdjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgesvdjFortran(hipsolverHandle_t     handle,
                                                           hipsolverEigMode_t    jobz,
                                                           int                   econ,
                                                           int                   m,
                                                           int                   n,
                                                           hipFloatComplex*      A,
                                                           int                   lda,
                                                           float*                S,
                                                           hipFloatComplex*      U,
                                                           int                   ldu,
                                                           hipFloatComplex*      V,
                                                           int                   ldv,
                                                           hipFloatComplex*      work,
                                                           int                   lwork,
                                                           int*                  devInfo,
                                                           hipsolverGesvdjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgesvdjFortran(hipsolverHandle_t     handle,
                                                           hipsolverEigMode_t    jobz,
                                                           int                   econ,
                                                           int                   m,
                                                           int                   n,
                                                           hipDoubleComplex*     A,
                                                           int                   lda,
                                                           double*               S,
                                                           hipDoubleComplex*     U,
                                                           int                   ldu,
                                                           hipDoubleComplex*     V,
                                                           int                   ldv,
                                                           hipDoubleComplex*     work,
                                                           int                   lwork,
                                                           int*                  devInfo,
                                                           hipsolverGesvdjInfo_t params);

// gesvdj_batched
HIPSOLVER_EXPORT hipsolverStatus_t
    hipsolverSgesvdjBatched_bufferSizeFortran(hipsolverHandle_t     handle,
                                              hipsolverEigMode_t    jobz,
                                              int                   m,
                                              int                   n,
                                              const float*          A,
                                              int                   lda,
                                              const float*          S,
                                              const float*          U,
                                              int                   ldu,
                                              const float*          V,
                                              int                   ldv,
                                              int*                  lwork,
                                              hipsolverGesvdjInfo_t params,
                                              int                   batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t
    hipsolverDgesvdjBatched_bufferSizeFortran(hipsolverHandle_t     handle,
                                              hipsolverEigMode_t    jobz,
                                              int                   m,
                                              int                   n,
                                              const double*         A,
                                              int                   lda,
                                              const double*         S,
                                              const double*         U,
                                              int                   ldu,
                                              const double*         V,
                                              int                   ldv,
                                              int*                  lwork,
                                              hipsolverGesvdjInfo_t params,
                                              int                   batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t
    hipsolverCgesvdjBatched_bufferSizeFortran(hipsolverHandle_t      handle,
                                              hipsolverEigMode_t     jobz,
                                              int                    m,
                                              int                    n,
                                              const hipFloatComplex* A,
                                              int                    lda,
                                              const float*           S,
                                              const hipFloatComplex* U,
                                              int                    ldu,
                                              const hipFloatComplex* V,
                                              int                    ldv,
                                              int*                   lwork,
                                              hipsolverGesvdjInfo_t  params,
                                              int                    batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t
    hipsolverZgesvdjBatched_bufferSizeFortran(hipsolverHandle_t       handle,
                                              hipsolverEigMode_t      jobz,
                                              int                     m,
                                              int                     n,
                                              const hipDoubleComplex* A,
                                              int                     lda,
                                              const double*           S,
                                              const hipDoubleComplex* U,
                                              int                     ldu,
                                              const hipDoubleComplex* V,
                                              int                     ldv,
                                              int*                    lwork,
                                              hipsolverGesvdjInfo_t   params,
                                              int                     batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgesvdjBatchedFortran(hipsolverHandle_t     handle,
                                                                  hipsolverEigMode_t    jobz,
                                                                  int                   m,
                                                                  int                   n,
                                                                  float*                A,
                                                                  int                   lda,
                                                                  float*                S,
                                                                  float*                U,
                                                                  int                   ldu,
                                                                  float*                V,
                                                                  int                   ldv,
                                                                  float*                work,
                                                                  int                   lwork,
                                                                  int*                  devInfo,
                                                                  hipsolverGesvdjInfo_t params,
                                                                  int batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgesvdjBatchedFortran(hipsolverHandle_t     handle,
                                                                  hipsolverEigMode_t    jobz,
                                                                  int                   m,
                                                                  int                   n,
                                                                  double*               A,
                                                                  int                   lda,
                                                                  double*               S,
                                                                  double*               U,
                                                                  int                   ldu,
                                                                  double*               V,
                                                                  int                   ldv,
                                                                  double*               work,
                                                                  int                   lwork,
                                                                  int*                  devInfo,
                                                                  hipsolverGesvdjInfo_t params,
                                                                  int batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgesvdjBatchedFortran(hipsolverHandle_t     handle,
                                                                  hipsolverEigMode_t    jobz,
                                                                  int                   m,
                                                                  int                   n,
                                                                  hipFloatComplex*      A,
                                                                  int                   lda,
                                                                  float*                S,
                                                                  hipFloatComplex*      U,
                                                                  int                   ldu,
                                                                  hipFloatComplex*      V,
                                                                  int                   ldv,
                                                                  hipFloatComplex*      work,
                                                                  int                   lwork,
                                                                  int*                  devInfo,
                                                                  hipsolverGesvdjInfo_t params,
                                                                  int batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgesvdjBatchedFortran(hipsolverHandle_t     handle,
                                                                  hipsolverEigMode_t    jobz,
                                                                  int                   m,
                                                                  int                   n,
                                                                  hipDoubleComplex*     A,
                                                                  int                   lda,
                                                                  double*               S,
                                                                  hipDoubleComplex*     U,
                                                                  int                   ldu,
                                                                  hipDoubleComplex*     V,
                                                                  int                   ldv,
                                                                  hipDoubleComplex*     work,
                                                                  int                   lwork,
                                                                  int*                  devInfo,
                                                                  hipsolverGesvdjInfo_t params,
                                                                  int batch_count);

// getrf
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgetrf_bufferSizeFortran(
    hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgetrf_bufferSizeFortran(
    hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgetrf_bufferSizeFortran(
    hipsolverHandle_t handle, int m, int n, hipFloatComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgetrf_bufferSizeFortran(
    hipsolverHandle_t handle, int m, int n, hipDoubleComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgetrfFortran(hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          float*            A,
                                                          int               lda,
                                                          float*            work,
                                                          int               lwork,
                                                          int*              devIpiv,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgetrfFortran(hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          double*           A,
                                                          int               lda,
                                                          double*           work,
                                                          int               lwork,
                                                          int*              devIpiv,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgetrfFortran(hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          hipFloatComplex*  A,
                                                          int               lda,
                                                          hipFloatComplex*  work,
                                                          int               lwork,
                                                          int*              devIpiv,
                                                          int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgetrfFortran(hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          hipDoubleComplex* A,
                                                          int               lda,
                                                          hipDoubleComplex* work,
                                                          int               lwork,
                                                          int*              devIpiv,
                                                          int*              devInfo);

// getrs
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgetrs_bufferSizeFortran(hipsolverHandle_t    handle,
                                                                     hipsolverOperation_t trans,
                                                                     int                  n,
                                                                     int                  nrhs,
                                                                     float*               A,
                                                                     int                  lda,
                                                                     int*                 devIpiv,
                                                                     float*               B,
                                                                     int                  ldb,
                                                                     int*                 lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgetrs_bufferSizeFortran(hipsolverHandle_t    handle,
                                                                     hipsolverOperation_t trans,
                                                                     int                  n,
                                                                     int                  nrhs,
                                                                     double*              A,
                                                                     int                  lda,
                                                                     int*                 devIpiv,
                                                                     double*              B,
                                                                     int                  ldb,
                                                                     int*                 lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgetrs_bufferSizeFortran(hipsolverHandle_t    handle,
                                                                     hipsolverOperation_t trans,
                                                                     int                  n,
                                                                     int                  nrhs,
                                                                     hipFloatComplex*     A,
                                                                     int                  lda,
                                                                     int*                 devIpiv,
                                                                     hipFloatComplex*     B,
                                                                     int                  ldb,
                                                                     int*                 lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgetrs_bufferSizeFortran(hipsolverHandle_t    handle,
                                                                     hipsolverOperation_t trans,
                                                                     int                  n,
                                                                     int                  nrhs,
                                                                     hipDoubleComplex*    A,
                                                                     int                  lda,
                                                                     int*                 devIpiv,
                                                                     hipDoubleComplex*    B,
                                                                     int                  ldb,
                                                                     int*                 lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSgetrsFortran(hipsolverHandle_t    handle,
                                                          hipsolverOperation_t trans,
                                                          int                  n,
                                                          int                  nrhs,
                                                          float*               A,
                                                          int                  lda,
                                                          int*                 devIpiv,
                                                          float*               B,
                                                          int                  ldb,
                                                          float*               work,
                                                          int                  lwork,
                                                          int*                 devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDgetrsFortran(hipsolverHandle_t    handle,
                                                          hipsolverOperation_t trans,
                                                          int                  n,
                                                          int                  nrhs,
                                                          double*              A,
                                                          int                  lda,
                                                          int*                 devIpiv,
                                                          double*              B,
                                                          int                  ldb,
                                                          double*              work,
                                                          int                  lwork,
                                                          int*                 devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCgetrsFortran(hipsolverHandle_t    handle,
                                                          hipsolverOperation_t trans,
                                                          int                  n,
                                                          int                  nrhs,
                                                          hipFloatComplex*     A,
                                                          int                  lda,
                                                          int*                 devIpiv,
                                                          hipFloatComplex*     B,
                                                          int                  ldb,
                                                          hipFloatComplex*     work,
                                                          int                  lwork,
                                                          int*                 devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZgetrsFortran(hipsolverHandle_t    handle,
                                                          hipsolverOperation_t trans,
                                                          int                  n,
                                                          int                  nrhs,
                                                          hipDoubleComplex*    A,
                                                          int                  lda,
                                                          int*                 devIpiv,
                                                          hipDoubleComplex*    B,
                                                          int                  ldb,
                                                          hipDoubleComplex*    work,
                                                          int                  lwork,
                                                          int*                 devInfo);

// potrf
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpotrf_bufferSizeFortran(
    hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, float* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDpotrf_bufferSizeFortran(
    hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, double* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCpotrf_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     hipFloatComplex*    A,
                                                                     int                 lda,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZpotrf_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     hipDoubleComplex*   A,
                                                                     int                 lda,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpotrfFortran(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          float*              A,
                                                          int                 lda,
                                                          float*              work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDpotrfFortran(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          double*             A,
                                                          int                 lda,
                                                          double*             work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCpotrfFortran(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipFloatComplex*    A,
                                                          int                 lda,
                                                          hipFloatComplex*    work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZpotrfFortran(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipDoubleComplex*   A,
                                                          int                 lda,
                                                          hipDoubleComplex*   work,
                                                          int                 lwork,
                                                          int*                devInfo);

// potrf_batched
HIPSOLVER_EXPORT hipsolverStatus_t
    hipsolverSpotrfBatched_bufferSizeFortran(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             float*              A[],
                                             int                 lda,
                                             int*                lwork,
                                             int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t
    hipsolverDpotrfBatched_bufferSizeFortran(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             double*             A[],
                                             int                 lda,
                                             int*                lwork,
                                             int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t
    hipsolverCpotrfBatched_bufferSizeFortran(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             hipFloatComplex*    A[],
                                             int                 lda,
                                             int*                lwork,
                                             int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t
    hipsolverZpotrfBatched_bufferSizeFortran(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             hipDoubleComplex*   A[],
                                             int                 lda,
                                             int*                lwork,
                                             int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpotrfBatchedFortran(hipsolverHandle_t   handle,
                                                                 hipsolverFillMode_t uplo,
                                                                 int                 n,
                                                                 float*              A[],
                                                                 int                 lda,
                                                                 float*              work,
                                                                 int                 lwork,
                                                                 int*                devInfo,
                                                                 int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDpotrfBatchedFortran(hipsolverHandle_t   handle,
                                                                 hipsolverFillMode_t uplo,
                                                                 int                 n,
                                                                 double*             A[],
                                                                 int                 lda,
                                                                 double*             work,
                                                                 int                 lwork,
                                                                 int*                devInfo,
                                                                 int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCpotrfBatchedFortran(hipsolverHandle_t   handle,
                                                                 hipsolverFillMode_t uplo,
                                                                 int                 n,
                                                                 hipFloatComplex*    A[],
                                                                 int                 lda,
                                                                 hipFloatComplex*    work,
                                                                 int                 lwork,
                                                                 int*                devInfo,
                                                                 int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZpotrfBatchedFortran(hipsolverHandle_t   handle,
                                                                 hipsolverFillMode_t uplo,
                                                                 int                 n,
                                                                 hipDoubleComplex*   A[],
                                                                 int                 lda,
                                                                 hipDoubleComplex*   work,
                                                                 int                 lwork,
                                                                 int*                devInfo,
                                                                 int                 batch_count);

// potri
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpotri_bufferSizeFortran(
    hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, float* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDpotri_bufferSizeFortran(
    hipsolverHandle_t handle, hipsolverFillMode_t uplo, int n, double* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCpotri_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     hipFloatComplex*    A,
                                                                     int                 lda,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZpotri_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     hipDoubleComplex*   A,
                                                                     int                 lda,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpotriFortran(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          float*              A,
                                                          int                 lda,
                                                          float*              work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDpotriFortran(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          double*             A,
                                                          int                 lda,
                                                          double*             work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCpotriFortran(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipFloatComplex*    A,
                                                          int                 lda,
                                                          hipFloatComplex*    work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZpotriFortran(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipDoubleComplex*   A,
                                                          int                 lda,
                                                          hipDoubleComplex*   work,
                                                          int                 lwork,
                                                          int*                devInfo);

// potrs
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpotrs_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     int                 nrhs,
                                                                     float*              A,
                                                                     int                 lda,
                                                                     float*              B,
                                                                     int                 ldb,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDpotrs_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     int                 nrhs,
                                                                     double*             A,
                                                                     int                 lda,
                                                                     double*             B,
                                                                     int                 ldb,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCpotrs_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     int                 nrhs,
                                                                     hipFloatComplex*    A,
                                                                     int                 lda,
                                                                     hipFloatComplex*    B,
                                                                     int                 ldb,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZpotrs_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     int                 nrhs,
                                                                     hipDoubleComplex*   A,
                                                                     int                 lda,
                                                                     hipDoubleComplex*   B,
                                                                     int                 ldb,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpotrsFortran(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          int                 nrhs,
                                                          float*              A,
                                                          int                 lda,
                                                          float*              B,
                                                          int                 ldb,
                                                          float*              work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDpotrsFortran(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          int                 nrhs,
                                                          double*             A,
                                                          int                 lda,
                                                          double*             B,
                                                          int                 ldb,
                                                          double*             work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCpotrsFortran(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          int                 nrhs,
                                                          hipFloatComplex*    A,
                                                          int                 lda,
                                                          hipFloatComplex*    B,
                                                          int                 ldb,
                                                          hipFloatComplex*    work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZpotrsFortran(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          int                 nrhs,
                                                          hipDoubleComplex*   A,
                                                          int                 lda,
                                                          hipDoubleComplex*   B,
                                                          int                 ldb,
                                                          hipDoubleComplex*   work,
                                                          int                 lwork,
                                                          int*                devInfo);

// potrs_batched
HIPSOLVER_EXPORT hipsolverStatus_t
    hipsolverSpotrsBatched_bufferSizeFortran(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             int                 nrhs,
                                             float*              A[],
                                             int                 lda,
                                             float*              B[],
                                             int                 ldb,
                                             int*                lwork,
                                             int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t
    hipsolverDpotrsBatched_bufferSizeFortran(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             int                 nrhs,
                                             double*             A[],
                                             int                 lda,
                                             double*             B[],
                                             int                 ldb,
                                             int*                lwork,
                                             int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t
    hipsolverCpotrsBatched_bufferSizeFortran(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             int                 nrhs,
                                             hipFloatComplex*    A[],
                                             int                 lda,
                                             hipFloatComplex*    B[],
                                             int                 ldb,
                                             int*                lwork,
                                             int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t
    hipsolverZpotrsBatched_bufferSizeFortran(hipsolverHandle_t   handle,
                                             hipsolverFillMode_t uplo,
                                             int                 n,
                                             int                 nrhs,
                                             hipDoubleComplex*   A[],
                                             int                 lda,
                                             hipDoubleComplex*   B[],
                                             int                 ldb,
                                             int*                lwork,
                                             int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSpotrsBatchedFortran(hipsolverHandle_t   handle,
                                                                 hipsolverFillMode_t uplo,
                                                                 int                 n,
                                                                 int                 nrhs,
                                                                 float*              A[],
                                                                 int                 lda,
                                                                 float*              B[],
                                                                 int                 ldb,
                                                                 float*              work,
                                                                 int                 lwork,
                                                                 int*                devInfo,
                                                                 int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDpotrsBatchedFortran(hipsolverHandle_t   handle,
                                                                 hipsolverFillMode_t uplo,
                                                                 int                 n,
                                                                 int                 nrhs,
                                                                 double*             A[],
                                                                 int                 lda,
                                                                 double*             B[],
                                                                 int                 ldb,
                                                                 double*             work,
                                                                 int                 lwork,
                                                                 int*                devInfo,
                                                                 int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCpotrsBatchedFortran(hipsolverHandle_t   handle,
                                                                 hipsolverFillMode_t uplo,
                                                                 int                 n,
                                                                 int                 nrhs,
                                                                 hipFloatComplex*    A[],
                                                                 int                 lda,
                                                                 hipFloatComplex*    B[],
                                                                 int                 ldb,
                                                                 hipFloatComplex*    work,
                                                                 int                 lwork,
                                                                 int*                devInfo,
                                                                 int                 batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZpotrsBatchedFortran(hipsolverHandle_t   handle,
                                                                 hipsolverFillMode_t uplo,
                                                                 int                 n,
                                                                 int                 nrhs,
                                                                 hipDoubleComplex*   A[],
                                                                 int                 lda,
                                                                 hipDoubleComplex*   B[],
                                                                 int                 ldb,
                                                                 hipDoubleComplex*   work,
                                                                 int                 lwork,
                                                                 int*                devInfo,
                                                                 int                 batch_count);

// syevd/heevd
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSsyevd_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverEigMode_t  jobz,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     float*              A,
                                                                     int                 lda,
                                                                     float*              W,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDsyevd_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverEigMode_t  jobz,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     double*             A,
                                                                     int                 lda,
                                                                     double*             W,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCheevd_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverEigMode_t  jobz,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     hipFloatComplex*    A,
                                                                     int                 lda,
                                                                     float*              W,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZheevd_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverEigMode_t  jobz,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     hipDoubleComplex*   A,
                                                                     int                 lda,
                                                                     double*             W,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSsyevdFortran(hipsolverHandle_t   handle,
                                                          hipsolverEigMode_t  jobz,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          float*              A,
                                                          int                 lda,
                                                          float*              W,
                                                          float*              work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDsyevdFortran(hipsolverHandle_t   handle,
                                                          hipsolverEigMode_t  jobz,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          double*             A,
                                                          int                 lda,
                                                          double*             W,
                                                          double*             work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCheevdFortran(hipsolverHandle_t   handle,
                                                          hipsolverEigMode_t  jobz,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipFloatComplex*    A,
                                                          int                 lda,
                                                          float*              W,
                                                          hipFloatComplex*    work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZheevdFortran(hipsolverHandle_t   handle,
                                                          hipsolverEigMode_t  jobz,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipDoubleComplex*   A,
                                                          int                 lda,
                                                          double*             W,
                                                          hipDoubleComplex*   work,
                                                          int                 lwork,
                                                          int*                devInfo);

// syevj/heevj
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSsyevj_bufferSizeFortran(hipsolverHandle_t    handle,
                                                                     hipsolverEigMode_t   jobz,
                                                                     hipsolverFillMode_t  uplo,
                                                                     int                  n,
                                                                     float*               A,
                                                                     int                  lda,
                                                                     float*               W,
                                                                     int*                 lwork,
                                                                     hipsolverSyevjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDsyevj_bufferSizeFortran(hipsolverHandle_t    handle,
                                                                     hipsolverEigMode_t   jobz,
                                                                     hipsolverFillMode_t  uplo,
                                                                     int                  n,
                                                                     double*              A,
                                                                     int                  lda,
                                                                     double*              W,
                                                                     int*                 lwork,
                                                                     hipsolverSyevjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCheevj_bufferSizeFortran(hipsolverHandle_t    handle,
                                                                     hipsolverEigMode_t   jobz,
                                                                     hipsolverFillMode_t  uplo,
                                                                     int                  n,
                                                                     hipFloatComplex*     A,
                                                                     int                  lda,
                                                                     float*               W,
                                                                     int*                 lwork,
                                                                     hipsolverSyevjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZheevj_bufferSizeFortran(hipsolverHandle_t    handle,
                                                                     hipsolverEigMode_t   jobz,
                                                                     hipsolverFillMode_t  uplo,
                                                                     int                  n,
                                                                     hipDoubleComplex*    A,
                                                                     int                  lda,
                                                                     double*              W,
                                                                     int*                 lwork,
                                                                     hipsolverSyevjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSsyevjFortran(hipsolverHandle_t    handle,
                                                          hipsolverEigMode_t   jobz,
                                                          hipsolverFillMode_t  uplo,
                                                          int                  n,
                                                          float*               A,
                                                          int                  lda,
                                                          float*               W,
                                                          float*               work,
                                                          int                  lwork,
                                                          int*                 devInfo,
                                                          hipsolverSyevjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDsyevjFortran(hipsolverHandle_t    handle,
                                                          hipsolverEigMode_t   jobz,
                                                          hipsolverFillMode_t  uplo,
                                                          int                  n,
                                                          double*              A,
                                                          int                  lda,
                                                          double*              W,
                                                          double*              work,
                                                          int                  lwork,
                                                          int*                 devInfo,
                                                          hipsolverSyevjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCheevjFortran(hipsolverHandle_t    handle,
                                                          hipsolverEigMode_t   jobz,
                                                          hipsolverFillMode_t  uplo,
                                                          int                  n,
                                                          hipFloatComplex*     A,
                                                          int                  lda,
                                                          float*               W,
                                                          hipFloatComplex*     work,
                                                          int                  lwork,
                                                          int*                 devInfo,
                                                          hipsolverSyevjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZheevjFortran(hipsolverHandle_t    handle,
                                                          hipsolverEigMode_t   jobz,
                                                          hipsolverFillMode_t  uplo,
                                                          int                  n,
                                                          hipDoubleComplex*    A,
                                                          int                  lda,
                                                          double*              W,
                                                          hipDoubleComplex*    work,
                                                          int                  lwork,
                                                          int*                 devInfo,
                                                          hipsolverSyevjInfo_t params);

// syevj_batched/heevj_batched
HIPSOLVER_EXPORT hipsolverStatus_t
    hipsolverSsyevjBatched_bufferSizeFortran(hipsolverHandle_t    handle,
                                             hipsolverEigMode_t   jobz,
                                             hipsolverFillMode_t  uplo,
                                             int                  n,
                                             float*               A,
                                             int                  lda,
                                             float*               W,
                                             int*                 lwork,
                                             hipsolverSyevjInfo_t params,
                                             int                  batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t
    hipsolverDsyevjBatched_bufferSizeFortran(hipsolverHandle_t    handle,
                                             hipsolverEigMode_t   jobz,
                                             hipsolverFillMode_t  uplo,
                                             int                  n,
                                             double*              A,
                                             int                  lda,
                                             double*              W,
                                             int*                 lwork,
                                             hipsolverSyevjInfo_t params,
                                             int                  batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t
    hipsolverCheevjBatched_bufferSizeFortran(hipsolverHandle_t    handle,
                                             hipsolverEigMode_t   jobz,
                                             hipsolverFillMode_t  uplo,
                                             int                  n,
                                             hipFloatComplex*     A,
                                             int                  lda,
                                             float*               W,
                                             int*                 lwork,
                                             hipsolverSyevjInfo_t params,
                                             int                  batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t
    hipsolverZheevjBatched_bufferSizeFortran(hipsolverHandle_t    handle,
                                             hipsolverEigMode_t   jobz,
                                             hipsolverFillMode_t  uplo,
                                             int                  n,
                                             hipDoubleComplex*    A,
                                             int                  lda,
                                             double*              W,
                                             int*                 lwork,
                                             hipsolverSyevjInfo_t params,
                                             int                  batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSsyevjBatchedFortran(hipsolverHandle_t    handle,
                                                                 hipsolverEigMode_t   jobz,
                                                                 hipsolverFillMode_t  uplo,
                                                                 int                  n,
                                                                 float*               A,
                                                                 int                  lda,
                                                                 float*               W,
                                                                 float*               work,
                                                                 int                  lwork,
                                                                 int*                 devInfo,
                                                                 hipsolverSyevjInfo_t params,
                                                                 int                  batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDsyevjBatchedFortran(hipsolverHandle_t    handle,
                                                                 hipsolverEigMode_t   jobz,
                                                                 hipsolverFillMode_t  uplo,
                                                                 int                  n,
                                                                 double*              A,
                                                                 int                  lda,
                                                                 double*              W,
                                                                 double*              work,
                                                                 int                  lwork,
                                                                 int*                 devInfo,
                                                                 hipsolverSyevjInfo_t params,
                                                                 int                  batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCheevjBatchedFortran(hipsolverHandle_t    handle,
                                                                 hipsolverEigMode_t   jobz,
                                                                 hipsolverFillMode_t  uplo,
                                                                 int                  n,
                                                                 hipFloatComplex*     A,
                                                                 int                  lda,
                                                                 float*               W,
                                                                 hipFloatComplex*     work,
                                                                 int                  lwork,
                                                                 int*                 devInfo,
                                                                 hipsolverSyevjInfo_t params,
                                                                 int                  batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZheevjBatchedFortran(hipsolverHandle_t    handle,
                                                                 hipsolverEigMode_t   jobz,
                                                                 hipsolverFillMode_t  uplo,
                                                                 int                  n,
                                                                 hipDoubleComplex*    A,
                                                                 int                  lda,
                                                                 double*              W,
                                                                 hipDoubleComplex*    work,
                                                                 int                  lwork,
                                                                 int*                 devInfo,
                                                                 hipsolverSyevjInfo_t params,
                                                                 int                  batch_count);

// sygvd/hegvd
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSsygvd_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverEigType_t  itype,
                                                                     hipsolverEigMode_t  jobz,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     float*              A,
                                                                     int                 lda,
                                                                     float*              B,
                                                                     int                 ldb,
                                                                     float*              W,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDsygvd_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverEigType_t  itype,
                                                                     hipsolverEigMode_t  jobz,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     double*             A,
                                                                     int                 lda,
                                                                     double*             B,
                                                                     int                 ldb,
                                                                     double*             W,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverChegvd_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverEigType_t  itype,
                                                                     hipsolverEigMode_t  jobz,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     hipFloatComplex*    A,
                                                                     int                 lda,
                                                                     hipFloatComplex*    B,
                                                                     int                 ldb,
                                                                     float*              W,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZhegvd_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverEigType_t  itype,
                                                                     hipsolverEigMode_t  jobz,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     hipDoubleComplex*   A,
                                                                     int                 lda,
                                                                     hipDoubleComplex*   B,
                                                                     int                 ldb,
                                                                     double*             W,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSsygvdFortran(hipsolverHandle_t   handle,
                                                          hipsolverEigType_t  itype,
                                                          hipsolverEigMode_t  jobz,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          float*              A,
                                                          int                 lda,
                                                          float*              B,
                                                          int                 ldb,
                                                          float*              W,
                                                          float*              work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDsygvdFortran(hipsolverHandle_t   handle,
                                                          hipsolverEigType_t  itype,
                                                          hipsolverEigMode_t  jobz,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          double*             A,
                                                          int                 lda,
                                                          double*             B,
                                                          int                 ldb,
                                                          double*             W,
                                                          double*             work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverChegvdFortran(hipsolverHandle_t   handle,
                                                          hipsolverEigType_t  itype,
                                                          hipsolverEigMode_t  jobz,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipFloatComplex*    A,
                                                          int                 lda,
                                                          hipFloatComplex*    B,
                                                          int                 ldb,
                                                          float*              W,
                                                          hipFloatComplex*    work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZhegvdFortran(hipsolverHandle_t   handle,
                                                          hipsolverEigType_t  itype,
                                                          hipsolverEigMode_t  jobz,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipDoubleComplex*   A,
                                                          int                 lda,
                                                          hipDoubleComplex*   B,
                                                          int                 ldb,
                                                          double*             W,
                                                          hipDoubleComplex*   work,
                                                          int                 lwork,
                                                          int*                devInfo);

// sygvj/hegvj
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSsygvj_bufferSizeFortran(hipsolverHandle_t    handle,
                                                                     hipsolverEigType_t   itype,
                                                                     hipsolverEigMode_t   jobz,
                                                                     hipsolverFillMode_t  uplo,
                                                                     int                  n,
                                                                     float*               A,
                                                                     int                  lda,
                                                                     float*               B,
                                                                     int                  ldb,
                                                                     float*               W,
                                                                     int*                 lwork,
                                                                     hipsolverSyevjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDsygvj_bufferSizeFortran(hipsolverHandle_t    handle,
                                                                     hipsolverEigType_t   itype,
                                                                     hipsolverEigMode_t   jobz,
                                                                     hipsolverFillMode_t  uplo,
                                                                     int                  n,
                                                                     double*              A,
                                                                     int                  lda,
                                                                     double*              B,
                                                                     int                  ldb,
                                                                     double*              W,
                                                                     int*                 lwork,
                                                                     hipsolverSyevjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverChegvj_bufferSizeFortran(hipsolverHandle_t    handle,
                                                                     hipsolverEigType_t   itype,
                                                                     hipsolverEigMode_t   jobz,
                                                                     hipsolverFillMode_t  uplo,
                                                                     int                  n,
                                                                     hipFloatComplex*     A,
                                                                     int                  lda,
                                                                     hipFloatComplex*     B,
                                                                     int                  ldb,
                                                                     float*               W,
                                                                     int*                 lwork,
                                                                     hipsolverSyevjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZhegvj_bufferSizeFortran(hipsolverHandle_t    handle,
                                                                     hipsolverEigType_t   itype,
                                                                     hipsolverEigMode_t   jobz,
                                                                     hipsolverFillMode_t  uplo,
                                                                     int                  n,
                                                                     hipDoubleComplex*    A,
                                                                     int                  lda,
                                                                     hipDoubleComplex*    B,
                                                                     int                  ldb,
                                                                     double*              W,
                                                                     int*                 lwork,
                                                                     hipsolverSyevjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSsygvjFortran(hipsolverHandle_t    handle,
                                                          hipsolverEigType_t   itype,
                                                          hipsolverEigMode_t   jobz,
                                                          hipsolverFillMode_t  uplo,
                                                          int                  n,
                                                          float*               A,
                                                          int                  lda,
                                                          float*               B,
                                                          int                  ldb,
                                                          float*               W,
                                                          float*               work,
                                                          int                  lwork,
                                                          int*                 devInfo,
                                                          hipsolverSyevjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDsygvjFortran(hipsolverHandle_t    handle,
                                                          hipsolverEigType_t   itype,
                                                          hipsolverEigMode_t   jobz,
                                                          hipsolverFillMode_t  uplo,
                                                          int                  n,
                                                          double*              A,
                                                          int                  lda,
                                                          double*              B,
                                                          int                  ldb,
                                                          double*              W,
                                                          double*              work,
                                                          int                  lwork,
                                                          int*                 devInfo,
                                                          hipsolverSyevjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverChegvjFortran(hipsolverHandle_t    handle,
                                                          hipsolverEigType_t   itype,
                                                          hipsolverEigMode_t   jobz,
                                                          hipsolverFillMode_t  uplo,
                                                          int                  n,
                                                          hipFloatComplex*     A,
                                                          int                  lda,
                                                          hipFloatComplex*     B,
                                                          int                  ldb,
                                                          float*               W,
                                                          hipFloatComplex*     work,
                                                          int                  lwork,
                                                          int*                 devInfo,
                                                          hipsolverSyevjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZhegvjFortran(hipsolverHandle_t    handle,
                                                          hipsolverEigType_t   itype,
                                                          hipsolverEigMode_t   jobz,
                                                          hipsolverFillMode_t  uplo,
                                                          int                  n,
                                                          hipDoubleComplex*    A,
                                                          int                  lda,
                                                          hipDoubleComplex*    B,
                                                          int                  ldb,
                                                          double*              W,
                                                          hipDoubleComplex*    work,
                                                          int                  lwork,
                                                          int*                 devInfo,
                                                          hipsolverSyevjInfo_t params);

// sytrd/hetrd
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSsytrd_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     float*              A,
                                                                     int                 lda,
                                                                     float*              D,
                                                                     float*              E,
                                                                     float*              tau,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDsytrd_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     double*             A,
                                                                     int                 lda,
                                                                     double*             D,
                                                                     double*             E,
                                                                     double*             tau,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverChetrd_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     hipFloatComplex*    A,
                                                                     int                 lda,
                                                                     float*              D,
                                                                     float*              E,
                                                                     hipFloatComplex*    tau,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZhetrd_bufferSizeFortran(hipsolverHandle_t   handle,
                                                                     hipsolverFillMode_t uplo,
                                                                     int                 n,
                                                                     hipDoubleComplex*   A,
                                                                     int                 lda,
                                                                     double*             D,
                                                                     double*             E,
                                                                     hipDoubleComplex*   tau,
                                                                     int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSsytrdFortran(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          float*              A,
                                                          int                 lda,
                                                          float*              D,
                                                          float*              E,
                                                          float*              tau,
                                                          float*              work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDsytrdFortran(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          double*             A,
                                                          int                 lda,
                                                          double*             D,
                                                          double*             E,
                                                          double*             tau,
                                                          double*             work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverChetrdFortran(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipFloatComplex*    A,
                                                          int                 lda,
                                                          float*              D,
                                                          float*              E,
                                                          hipFloatComplex*    tau,
                                                          hipFloatComplex*    work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZhetrdFortran(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipDoubleComplex*   A,
                                                          int                 lda,
                                                          double*             D,
                                                          double*             E,
                                                          hipDoubleComplex*   tau,
                                                          hipDoubleComplex*   work,
                                                          int                 lwork,
                                                          int*                devInfo);

// sytrf
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSsytrf_bufferSizeFortran(
    hipsolverHandle_t handle, int n, float* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDsytrf_bufferSizeFortran(
    hipsolverHandle_t handle, int n, double* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCsytrf_bufferSizeFortran(
    hipsolverHandle_t handle, int n, hipFloatComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZsytrf_bufferSizeFortran(
    hipsolverHandle_t handle, int n, hipDoubleComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverSsytrfFortran(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          float*              A,
                                                          int                 lda,
                                                          int*                ipiv,
                                                          float*              work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDsytrfFortran(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          double*             A,
                                                          int                 lda,
                                                          int*                ipiv,
                                                          double*             work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverCsytrfFortran(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipFloatComplex*    A,
                                                          int                 lda,
                                                          int*                ipiv,
                                                          hipFloatComplex*    work,
                                                          int                 lwork,
                                                          int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverZsytrfFortran(hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipDoubleComplex*   A,
                                                          int                 lda,
                                                          int*                ipiv,
                                                          hipDoubleComplex*   work,
                                                          int                 lwork,
                                                          int*                devInfo);
}
