/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifdef HAVE_HIPSOLVER_FORTRAN_CLIENT
#include "hipsolver_fortran.hpp"
#else
#include "hipsolver_no_fortran.hpp"
#endif

// Most functions within this file exist to provide a consistent interface for our templated tests.
// Function overloading is used to select between the float, double, rocblas_float_complex
// and rocblas_double_complex variants, and to distinguish the batched case (T**) from the normal
// and strided_batched cases (T*).
//
// The normal and strided_batched cases are distinguished from each other by passing a boolean
// parameter, STRIDED. Variants such as the blocked and unblocked versions of algorithms, may be
// provided in similar ways.

typedef enum
{
    API_NORMAL,
    API_FORTRAN,
    API_COMPAT
} testAPI_t;

typedef enum
{
    C_NORMAL,
    C_NORMAL_ALT,
    FORTRAN_NORMAL,
    FORTRAN_NORMAL_ALT,
    COMPAT_NORMAL,
    COMPAT_NORMAL_ALT,
    INVALID_API_SPEC
} testMarshal_t;

inline testMarshal_t api2marshal(testAPI_t API, bool ALT)
{
    switch(API)
    {
    case API_NORMAL:
        if(!ALT)
            return C_NORMAL;
        else
            return C_NORMAL_ALT;
    case API_FORTRAN:
        if(!ALT)
            return FORTRAN_NORMAL;
        else
            return FORTRAN_NORMAL_ALT;
    case API_COMPAT:
        if(!ALT)
            return COMPAT_NORMAL;
        else
            return COMPAT_NORMAL_ALT;
    default:
        return INVALID_API_SPEC;
    }
}

/******************** ORGBR/UNGBR ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_orgbr_ungbr_bufferSize(testAPI_t           API,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverSideMode_t side,
                                                          int                 m,
                                                          int                 n,
                                                          int                 k,
                                                          float*              A,
                                                          int                 lda,
                                                          float*              tau,
                                                          int*                lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork);
    case API_FORTRAN:
        return hipsolverSorgbr_bufferSizeFortran(handle, side, m, n, k, A, lda, tau, lwork);
    case API_COMPAT:
        return hipsolverDnSorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_orgbr_ungbr_bufferSize(testAPI_t           API,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverSideMode_t side,
                                                          int                 m,
                                                          int                 n,
                                                          int                 k,
                                                          double*             A,
                                                          int                 lda,
                                                          double*             tau,
                                                          int*                lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork);
    case API_FORTRAN:
        return hipsolverDorgbr_bufferSizeFortran(handle, side, m, n, k, A, lda, tau, lwork);
    case API_COMPAT:
        return hipsolverDnDorgbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_orgbr_ungbr_bufferSize(testAPI_t           API,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverSideMode_t side,
                                                          int                 m,
                                                          int                 n,
                                                          int                 k,
                                                          hipsolverComplex*   A,
                                                          int                 lda,
                                                          hipsolverComplex*   tau,
                                                          int*                lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCungbr_bufferSize(
            handle, side, m, n, k, (hipFloatComplex*)A, lda, (hipFloatComplex*)tau, lwork);
    case API_FORTRAN:
        return hipsolverCungbr_bufferSizeFortran(
            handle, side, m, n, k, (hipFloatComplex*)A, lda, (hipFloatComplex*)tau, lwork);
    case API_COMPAT:
        return hipsolverDnCungbr_bufferSize(
            handle, side, m, n, k, (hipFloatComplex*)A, lda, (hipFloatComplex*)tau, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_orgbr_ungbr_bufferSize(testAPI_t               API,
                                                          hipsolverHandle_t       handle,
                                                          hipsolverSideMode_t     side,
                                                          int                     m,
                                                          int                     n,
                                                          int                     k,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          hipsolverDoubleComplex* tau,
                                                          int*                    lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZungbr_bufferSize(
            handle, side, m, n, k, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)tau, lwork);
    case API_FORTRAN:
        return hipsolverZungbr_bufferSizeFortran(
            handle, side, m, n, k, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)tau, lwork);
    case API_COMPAT:
        return hipsolverDnZungbr_bufferSize(
            handle, side, m, n, k, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)tau, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_orgbr_ungbr(testAPI_t           API,
                                               hipsolverHandle_t   handle,
                                               hipsolverSideMode_t side,
                                               int                 m,
                                               int                 n,
                                               int                 k,
                                               float*              A,
                                               int                 lda,
                                               float*              tau,
                                               float*              work,
                                               int                 lwork,
                                               int*                info)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, info);
    case API_FORTRAN:
        return hipsolverSorgbrFortran(handle, side, m, n, k, A, lda, tau, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnSorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_orgbr_ungbr(testAPI_t           API,
                                               hipsolverHandle_t   handle,
                                               hipsolverSideMode_t side,
                                               int                 m,
                                               int                 n,
                                               int                 k,
                                               double*             A,
                                               int                 lda,
                                               double*             tau,
                                               double*             work,
                                               int                 lwork,
                                               int*                info)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, info);
    case API_FORTRAN:
        return hipsolverDorgbrFortran(handle, side, m, n, k, A, lda, tau, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnDorgbr(handle, side, m, n, k, A, lda, tau, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_orgbr_ungbr(testAPI_t           API,
                                               hipsolverHandle_t   handle,
                                               hipsolverSideMode_t side,
                                               int                 m,
                                               int                 n,
                                               int                 k,
                                               hipsolverComplex*   A,
                                               int                 lda,
                                               hipsolverComplex*   tau,
                                               hipsolverComplex*   work,
                                               int                 lwork,
                                               int*                info)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCungbr(handle,
                               side,
                               m,
                               n,
                               k,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)tau,
                               (hipFloatComplex*)work,
                               lwork,
                               info);
    case API_FORTRAN:
        return hipsolverCungbrFortran(handle,
                                      side,
                                      m,
                                      n,
                                      k,
                                      (hipFloatComplex*)A,
                                      lda,
                                      (hipFloatComplex*)tau,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info);
    case API_COMPAT:
        return hipsolverDnCungbr(handle,
                                 side,
                                 m,
                                 n,
                                 k,
                                 (hipFloatComplex*)A,
                                 lda,
                                 (hipFloatComplex*)tau,
                                 (hipFloatComplex*)work,
                                 lwork,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_orgbr_ungbr(testAPI_t               API,
                                               hipsolverHandle_t       handle,
                                               hipsolverSideMode_t     side,
                                               int                     m,
                                               int                     n,
                                               int                     k,
                                               hipsolverDoubleComplex* A,
                                               int                     lda,
                                               hipsolverDoubleComplex* tau,
                                               hipsolverDoubleComplex* work,
                                               int                     lwork,
                                               int*                    info)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZungbr(handle,
                               side,
                               m,
                               n,
                               k,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)tau,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);
    case API_FORTRAN:
        return hipsolverZungbrFortran(handle,
                                      side,
                                      m,
                                      n,
                                      k,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      (hipDoubleComplex*)tau,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      info);
    case API_COMPAT:
        return hipsolverDnZungbr(handle,
                                 side,
                                 m,
                                 n,
                                 k,
                                 (hipDoubleComplex*)A,
                                 lda,
                                 (hipDoubleComplex*)tau,
                                 (hipDoubleComplex*)work,
                                 lwork,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** ORGQR/UNGQR ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_orgqr_ungqr_bufferSize(testAPI_t         API,
                                                          hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          int               k,
                                                          float*            A,
                                                          int               lda,
                                                          float*            tau,
                                                          int*              lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
    case API_FORTRAN:
        return hipsolverSorgqr_bufferSizeFortran(handle, m, n, k, A, lda, tau, lwork);
    case API_COMPAT:
        return hipsolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_orgqr_ungqr_bufferSize(testAPI_t         API,
                                                          hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          int               k,
                                                          double*           A,
                                                          int               lda,
                                                          double*           tau,
                                                          int*              lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
    case API_FORTRAN:
        return hipsolverDorgqr_bufferSizeFortran(handle, m, n, k, A, lda, tau, lwork);
    case API_COMPAT:
        return hipsolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_orgqr_ungqr_bufferSize(testAPI_t         API,
                                                          hipsolverHandle_t handle,
                                                          int               m,
                                                          int               n,
                                                          int               k,
                                                          hipsolverComplex* A,
                                                          int               lda,
                                                          hipsolverComplex* tau,
                                                          int*              lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCungqr_bufferSize(
            handle, m, n, k, (hipFloatComplex*)A, lda, (hipFloatComplex*)tau, lwork);
    case API_FORTRAN:
        return hipsolverCungqr_bufferSizeFortran(
            handle, m, n, k, (hipFloatComplex*)A, lda, (hipFloatComplex*)tau, lwork);
    case API_COMPAT:
        return hipsolverDnCungqr_bufferSize(
            handle, m, n, k, (hipFloatComplex*)A, lda, (hipFloatComplex*)tau, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_orgqr_ungqr_bufferSize(testAPI_t               API,
                                                          hipsolverHandle_t       handle,
                                                          int                     m,
                                                          int                     n,
                                                          int                     k,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          hipsolverDoubleComplex* tau,
                                                          int*                    lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZungqr_bufferSize(
            handle, m, n, k, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)tau, lwork);
    case API_FORTRAN:
        return hipsolverZungqr_bufferSizeFortran(
            handle, m, n, k, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)tau, lwork);
    case API_COMPAT:
        return hipsolverDnZungqr_bufferSize(
            handle, m, n, k, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)tau, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_orgqr_ungqr(testAPI_t         API,
                                               hipsolverHandle_t handle,
                                               int               m,
                                               int               n,
                                               int               k,
                                               float*            A,
                                               int               lda,
                                               float*            tau,
                                               float*            work,
                                               int               lwork,
                                               int*              info)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSorgqr(handle, m, n, k, A, lda, tau, work, lwork, info);
    case API_FORTRAN:
        return hipsolverSorgqrFortran(handle, m, n, k, A, lda, tau, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnSorgqr(handle, m, n, k, A, lda, tau, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_orgqr_ungqr(testAPI_t         API,
                                               hipsolverHandle_t handle,
                                               int               m,
                                               int               n,
                                               int               k,
                                               double*           A,
                                               int               lda,
                                               double*           tau,
                                               double*           work,
                                               int               lwork,
                                               int*              info)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDorgqr(handle, m, n, k, A, lda, tau, work, lwork, info);
    case API_FORTRAN:
        return hipsolverDorgqrFortran(handle, m, n, k, A, lda, tau, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_orgqr_ungqr(testAPI_t         API,
                                               hipsolverHandle_t handle,
                                               int               m,
                                               int               n,
                                               int               k,
                                               hipsolverComplex* A,
                                               int               lda,
                                               hipsolverComplex* tau,
                                               hipsolverComplex* work,
                                               int               lwork,
                                               int*              info)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCungqr(handle,
                               m,
                               n,
                               k,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)tau,
                               (hipFloatComplex*)work,
                               lwork,
                               info);
    case API_FORTRAN:
        return hipsolverCungqrFortran(handle,
                                      m,
                                      n,
                                      k,
                                      (hipFloatComplex*)A,
                                      lda,
                                      (hipFloatComplex*)tau,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info);
    case API_COMPAT:
        return hipsolverDnCungqr(handle,
                                 m,
                                 n,
                                 k,
                                 (hipFloatComplex*)A,
                                 lda,
                                 (hipFloatComplex*)tau,
                                 (hipFloatComplex*)work,
                                 lwork,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_orgqr_ungqr(testAPI_t               API,
                                               hipsolverHandle_t       handle,
                                               int                     m,
                                               int                     n,
                                               int                     k,
                                               hipsolverDoubleComplex* A,
                                               int                     lda,
                                               hipsolverDoubleComplex* tau,
                                               hipsolverDoubleComplex* work,
                                               int                     lwork,
                                               int*                    info)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZungqr(handle,
                               m,
                               n,
                               k,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)tau,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);
    case API_FORTRAN:
        return hipsolverZungqrFortran(handle,
                                      m,
                                      n,
                                      k,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      (hipDoubleComplex*)tau,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      info);
    case API_COMPAT:
        return hipsolverDnZungqr(handle,
                                 m,
                                 n,
                                 k,
                                 (hipDoubleComplex*)A,
                                 lda,
                                 (hipDoubleComplex*)tau,
                                 (hipDoubleComplex*)work,
                                 lwork,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** ORGTR/UNGTR ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_orgtr_ungtr_bufferSize(testAPI_t           API,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          float*              A,
                                                          int                 lda,
                                                          float*              tau,
                                                          int*                lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork);
    case API_FORTRAN:
        return hipsolverSorgtr_bufferSizeFortran(handle, uplo, n, A, lda, tau, lwork);
    case API_COMPAT:
        return hipsolverDnSorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_orgtr_ungtr_bufferSize(testAPI_t           API,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          double*             A,
                                                          int                 lda,
                                                          double*             tau,
                                                          int*                lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork);
    case API_FORTRAN:
        return hipsolverDorgtr_bufferSizeFortran(handle, uplo, n, A, lda, tau, lwork);
    case API_COMPAT:
        return hipsolverDnDorgtr_bufferSize(handle, uplo, n, A, lda, tau, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_orgtr_ungtr_bufferSize(testAPI_t           API,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipsolverComplex*   A,
                                                          int                 lda,
                                                          hipsolverComplex*   tau,
                                                          int*                lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCungtr_bufferSize(
            handle, uplo, n, (hipFloatComplex*)A, lda, (hipFloatComplex*)tau, lwork);
    case API_FORTRAN:
        return hipsolverCungtr_bufferSizeFortran(
            handle, uplo, n, (hipFloatComplex*)A, lda, (hipFloatComplex*)tau, lwork);
    case API_COMPAT:
        return hipsolverDnCungtr_bufferSize(
            handle, uplo, n, (hipFloatComplex*)A, lda, (hipFloatComplex*)tau, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_orgtr_ungtr_bufferSize(testAPI_t               API,
                                                          hipsolverHandle_t       handle,
                                                          hipsolverFillMode_t     uplo,
                                                          int                     n,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          hipsolverDoubleComplex* tau,
                                                          int*                    lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZungtr_bufferSize(
            handle, uplo, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)tau, lwork);
    case API_FORTRAN:
        return hipsolverZungtr_bufferSizeFortran(
            handle, uplo, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)tau, lwork);
    case API_COMPAT:
        return hipsolverDnZungtr_bufferSize(
            handle, uplo, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)tau, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_orgtr_ungtr(testAPI_t           API,
                                               hipsolverHandle_t   handle,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               float*              A,
                                               int                 lda,
                                               float*              tau,
                                               float*              work,
                                               int                 lwork,
                                               int*                info)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSorgtr(handle, uplo, n, A, lda, tau, work, lwork, info);
    case API_FORTRAN:
        return hipsolverSorgtrFortran(handle, uplo, n, A, lda, tau, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnSorgtr(handle, uplo, n, A, lda, tau, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_orgtr_ungtr(testAPI_t           API,
                                               hipsolverHandle_t   handle,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               double*             A,
                                               int                 lda,
                                               double*             tau,
                                               double*             work,
                                               int                 lwork,
                                               int*                info)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDorgtr(handle, uplo, n, A, lda, tau, work, lwork, info);
    case API_FORTRAN:
        return hipsolverDorgtrFortran(handle, uplo, n, A, lda, tau, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnDorgtr(handle, uplo, n, A, lda, tau, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_orgtr_ungtr(testAPI_t           API,
                                               hipsolverHandle_t   handle,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               hipsolverComplex*   A,
                                               int                 lda,
                                               hipsolverComplex*   tau,
                                               hipsolverComplex*   work,
                                               int                 lwork,
                                               int*                info)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCungtr(handle,
                               uplo,
                               n,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)tau,
                               (hipFloatComplex*)work,
                               lwork,
                               info);
    case API_FORTRAN:
        return hipsolverCungtrFortran(handle,
                                      uplo,
                                      n,
                                      (hipFloatComplex*)A,
                                      lda,
                                      (hipFloatComplex*)tau,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info);
    case API_COMPAT:
        return hipsolverDnCungtr(handle,
                                 uplo,
                                 n,
                                 (hipFloatComplex*)A,
                                 lda,
                                 (hipFloatComplex*)tau,
                                 (hipFloatComplex*)work,
                                 lwork,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_orgtr_ungtr(testAPI_t               API,
                                               hipsolverHandle_t       handle,
                                               hipsolverFillMode_t     uplo,
                                               int                     n,
                                               hipsolverDoubleComplex* A,
                                               int                     lda,
                                               hipsolverDoubleComplex* tau,
                                               hipsolverDoubleComplex* work,
                                               int                     lwork,
                                               int*                    info)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZungtr(handle,
                               uplo,
                               n,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)tau,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);
    case API_FORTRAN:
        return hipsolverZungtrFortran(handle,
                                      uplo,
                                      n,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      (hipDoubleComplex*)tau,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      info);
    case API_COMPAT:
        return hipsolverDnZungtr(handle,
                                 uplo,
                                 n,
                                 (hipDoubleComplex*)A,
                                 lda,
                                 (hipDoubleComplex*)tau,
                                 (hipDoubleComplex*)work,
                                 lwork,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** ORMQR/UNMQR ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_ormqr_unmqr_bufferSize(testAPI_t            API,
                                                          hipsolverHandle_t    handle,
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
                                                          int*                 lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
    case API_FORTRAN:
        return hipsolverSormqr_bufferSizeFortran(
            handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
    case API_COMPAT:
        return hipsolverDnSormqr_bufferSize(
            handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_ormqr_unmqr_bufferSize(testAPI_t            API,
                                                          hipsolverHandle_t    handle,
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
                                                          int*                 lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
    case API_FORTRAN:
        return hipsolverDormqr_bufferSizeFortran(
            handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
    case API_COMPAT:
        return hipsolverDnDormqr_bufferSize(
            handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_ormqr_unmqr_bufferSize(testAPI_t            API,
                                                          hipsolverHandle_t    handle,
                                                          hipsolverSideMode_t  side,
                                                          hipsolverOperation_t trans,
                                                          int                  m,
                                                          int                  n,
                                                          int                  k,
                                                          hipsolverComplex*    A,
                                                          int                  lda,
                                                          hipsolverComplex*    tau,
                                                          hipsolverComplex*    C,
                                                          int                  ldc,
                                                          int*                 lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCunmqr_bufferSize(handle,
                                          side,
                                          trans,
                                          m,
                                          n,
                                          k,
                                          (hipFloatComplex*)A,
                                          lda,
                                          (hipFloatComplex*)tau,
                                          (hipFloatComplex*)C,
                                          ldc,
                                          lwork);
    case API_FORTRAN:
        return hipsolverCunmqr_bufferSizeFortran(handle,
                                                 side,
                                                 trans,
                                                 m,
                                                 n,
                                                 k,
                                                 (hipFloatComplex*)A,
                                                 lda,
                                                 (hipFloatComplex*)tau,
                                                 (hipFloatComplex*)C,
                                                 ldc,
                                                 lwork);
    case API_COMPAT:
        return hipsolverDnCunmqr_bufferSize(handle,
                                            side,
                                            trans,
                                            m,
                                            n,
                                            k,
                                            (hipFloatComplex*)A,
                                            lda,
                                            (hipFloatComplex*)tau,
                                            (hipFloatComplex*)C,
                                            ldc,
                                            lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_ormqr_unmqr_bufferSize(testAPI_t               API,
                                                          hipsolverHandle_t       handle,
                                                          hipsolverSideMode_t     side,
                                                          hipsolverOperation_t    trans,
                                                          int                     m,
                                                          int                     n,
                                                          int                     k,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          hipsolverDoubleComplex* tau,
                                                          hipsolverDoubleComplex* C,
                                                          int                     ldc,
                                                          int*                    lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZunmqr_bufferSize(handle,
                                          side,
                                          trans,
                                          m,
                                          n,
                                          k,
                                          (hipDoubleComplex*)A,
                                          lda,
                                          (hipDoubleComplex*)tau,
                                          (hipDoubleComplex*)C,
                                          ldc,
                                          lwork);
    case API_FORTRAN:
        return hipsolverZunmqr_bufferSizeFortran(handle,
                                                 side,
                                                 trans,
                                                 m,
                                                 n,
                                                 k,
                                                 (hipDoubleComplex*)A,
                                                 lda,
                                                 (hipDoubleComplex*)tau,
                                                 (hipDoubleComplex*)C,
                                                 ldc,
                                                 lwork);
    case API_COMPAT:
        return hipsolverDnZunmqr_bufferSize(handle,
                                            side,
                                            trans,
                                            m,
                                            n,
                                            k,
                                            (hipDoubleComplex*)A,
                                            lda,
                                            (hipDoubleComplex*)tau,
                                            (hipDoubleComplex*)C,
                                            ldc,
                                            lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_ormqr_unmqr(testAPI_t            API,
                                               hipsolverHandle_t    handle,
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
                                               int*                 info)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSormqr(
            handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info);
    case API_FORTRAN:
        return hipsolverSormqrFortran(
            handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnSormqr(
            handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_ormqr_unmqr(testAPI_t            API,
                                               hipsolverHandle_t    handle,
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
                                               int*                 info)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDormqr(
            handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info);
    case API_FORTRAN:
        return hipsolverDormqrFortran(
            handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnDormqr(
            handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_ormqr_unmqr(testAPI_t            API,
                                               hipsolverHandle_t    handle,
                                               hipsolverSideMode_t  side,
                                               hipsolverOperation_t trans,
                                               int                  m,
                                               int                  n,
                                               int                  k,
                                               hipsolverComplex*    A,
                                               int                  lda,
                                               hipsolverComplex*    tau,
                                               hipsolverComplex*    C,
                                               int                  ldc,
                                               hipsolverComplex*    work,
                                               int                  lwork,
                                               int*                 info)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCunmqr(handle,
                               side,
                               trans,
                               m,
                               n,
                               k,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)tau,
                               (hipFloatComplex*)C,
                               ldc,
                               (hipFloatComplex*)work,
                               lwork,
                               info);
    case API_FORTRAN:
        return hipsolverCunmqrFortran(handle,
                                      side,
                                      trans,
                                      m,
                                      n,
                                      k,
                                      (hipFloatComplex*)A,
                                      lda,
                                      (hipFloatComplex*)tau,
                                      (hipFloatComplex*)C,
                                      ldc,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info);
    case API_COMPAT:
        return hipsolverDnCunmqr(handle,
                                 side,
                                 trans,
                                 m,
                                 n,
                                 k,
                                 (hipFloatComplex*)A,
                                 lda,
                                 (hipFloatComplex*)tau,
                                 (hipFloatComplex*)C,
                                 ldc,
                                 (hipFloatComplex*)work,
                                 lwork,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_ormqr_unmqr(testAPI_t               API,
                                               hipsolverHandle_t       handle,
                                               hipsolverSideMode_t     side,
                                               hipsolverOperation_t    trans,
                                               int                     m,
                                               int                     n,
                                               int                     k,
                                               hipsolverDoubleComplex* A,
                                               int                     lda,
                                               hipsolverDoubleComplex* tau,
                                               hipsolverDoubleComplex* C,
                                               int                     ldc,
                                               hipsolverDoubleComplex* work,
                                               int                     lwork,
                                               int*                    info)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZunmqr(handle,
                               side,
                               trans,
                               m,
                               n,
                               k,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)tau,
                               (hipDoubleComplex*)C,
                               ldc,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);
    case API_FORTRAN:
        return hipsolverZunmqrFortran(handle,
                                      side,
                                      trans,
                                      m,
                                      n,
                                      k,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      (hipDoubleComplex*)tau,
                                      (hipDoubleComplex*)C,
                                      ldc,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      info);
    case API_COMPAT:
        return hipsolverDnZunmqr(handle,
                                 side,
                                 trans,
                                 m,
                                 n,
                                 k,
                                 (hipDoubleComplex*)A,
                                 lda,
                                 (hipDoubleComplex*)tau,
                                 (hipDoubleComplex*)C,
                                 ldc,
                                 (hipDoubleComplex*)work,
                                 lwork,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** ORMTR/UNMTR ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_ormtr_unmtr_bufferSize(testAPI_t            API,
                                                          hipsolverHandle_t    handle,
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
                                                          int*                 lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSormtr_bufferSize(
            handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork);
    case API_FORTRAN:
        return hipsolverSormtr_bufferSizeFortran(
            handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork);
    case API_COMPAT:
        return hipsolverDnSormtr_bufferSize(
            handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_ormtr_unmtr_bufferSize(testAPI_t            API,
                                                          hipsolverHandle_t    handle,
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
                                                          int*                 lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDormtr_bufferSize(
            handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork);
    case API_FORTRAN:
        return hipsolverDormtr_bufferSizeFortran(
            handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork);
    case API_COMPAT:
        return hipsolverDnDormtr_bufferSize(
            handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_ormtr_unmtr_bufferSize(testAPI_t            API,
                                                          hipsolverHandle_t    handle,
                                                          hipsolverSideMode_t  side,
                                                          hipsolverFillMode_t  uplo,
                                                          hipsolverOperation_t trans,
                                                          int                  m,
                                                          int                  n,
                                                          hipsolverComplex*    A,
                                                          int                  lda,
                                                          hipsolverComplex*    tau,
                                                          hipsolverComplex*    C,
                                                          int                  ldc,
                                                          int*                 lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCunmtr_bufferSize(handle,
                                          side,
                                          uplo,
                                          trans,
                                          m,
                                          n,
                                          (hipFloatComplex*)A,
                                          lda,
                                          (hipFloatComplex*)tau,
                                          (hipFloatComplex*)C,
                                          ldc,
                                          lwork);
    case API_FORTRAN:
        return hipsolverCunmtr_bufferSizeFortran(handle,
                                                 side,
                                                 uplo,
                                                 trans,
                                                 m,
                                                 n,
                                                 (hipFloatComplex*)A,
                                                 lda,
                                                 (hipFloatComplex*)tau,
                                                 (hipFloatComplex*)C,
                                                 ldc,
                                                 lwork);
    case API_COMPAT:
        return hipsolverDnCunmtr_bufferSize(handle,
                                            side,
                                            uplo,
                                            trans,
                                            m,
                                            n,
                                            (hipFloatComplex*)A,
                                            lda,
                                            (hipFloatComplex*)tau,
                                            (hipFloatComplex*)C,
                                            ldc,
                                            lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_ormtr_unmtr_bufferSize(testAPI_t               API,
                                                          hipsolverHandle_t       handle,
                                                          hipsolverSideMode_t     side,
                                                          hipsolverFillMode_t     uplo,
                                                          hipsolverOperation_t    trans,
                                                          int                     m,
                                                          int                     n,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          hipsolverDoubleComplex* tau,
                                                          hipsolverDoubleComplex* C,
                                                          int                     ldc,
                                                          int*                    lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZunmtr_bufferSize(handle,
                                          side,
                                          uplo,
                                          trans,
                                          m,
                                          n,
                                          (hipDoubleComplex*)A,
                                          lda,
                                          (hipDoubleComplex*)tau,
                                          (hipDoubleComplex*)C,
                                          ldc,
                                          lwork);
    case API_FORTRAN:
        return hipsolverZunmtr_bufferSizeFortran(handle,
                                                 side,
                                                 uplo,
                                                 trans,
                                                 m,
                                                 n,
                                                 (hipDoubleComplex*)A,
                                                 lda,
                                                 (hipDoubleComplex*)tau,
                                                 (hipDoubleComplex*)C,
                                                 ldc,
                                                 lwork);
    case API_COMPAT:
        return hipsolverDnZunmtr_bufferSize(handle,
                                            side,
                                            uplo,
                                            trans,
                                            m,
                                            n,
                                            (hipDoubleComplex*)A,
                                            lda,
                                            (hipDoubleComplex*)tau,
                                            (hipDoubleComplex*)C,
                                            ldc,
                                            lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_ormtr_unmtr(testAPI_t            API,
                                               hipsolverHandle_t    handle,
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
                                               int*                 info)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSormtr(
            handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info);
    case API_FORTRAN:
        return hipsolverSormtrFortran(
            handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnSormtr(
            handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_ormtr_unmtr(testAPI_t            API,
                                               hipsolverHandle_t    handle,
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
                                               int*                 info)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDormtr(
            handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info);
    case API_FORTRAN:
        return hipsolverDormtrFortran(
            handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnDormtr(
            handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_ormtr_unmtr(testAPI_t            API,
                                               hipsolverHandle_t    handle,
                                               hipsolverSideMode_t  side,
                                               hipsolverFillMode_t  uplo,
                                               hipsolverOperation_t trans,
                                               int                  m,
                                               int                  n,
                                               hipsolverComplex*    A,
                                               int                  lda,
                                               hipsolverComplex*    tau,
                                               hipsolverComplex*    C,
                                               int                  ldc,
                                               hipsolverComplex*    work,
                                               int                  lwork,
                                               int*                 info)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCunmtr(handle,
                               side,
                               uplo,
                               trans,
                               m,
                               n,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)tau,
                               (hipFloatComplex*)C,
                               ldc,
                               (hipFloatComplex*)work,
                               lwork,
                               info);
    case API_FORTRAN:
        return hipsolverCunmtrFortran(handle,
                                      side,
                                      uplo,
                                      trans,
                                      m,
                                      n,
                                      (hipFloatComplex*)A,
                                      lda,
                                      (hipFloatComplex*)tau,
                                      (hipFloatComplex*)C,
                                      ldc,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info);
    case API_COMPAT:
        return hipsolverDnCunmtr(handle,
                                 side,
                                 uplo,
                                 trans,
                                 m,
                                 n,
                                 (hipFloatComplex*)A,
                                 lda,
                                 (hipFloatComplex*)tau,
                                 (hipFloatComplex*)C,
                                 ldc,
                                 (hipFloatComplex*)work,
                                 lwork,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_ormtr_unmtr(testAPI_t               API,
                                               hipsolverHandle_t       handle,
                                               hipsolverSideMode_t     side,
                                               hipsolverFillMode_t     uplo,
                                               hipsolverOperation_t    trans,
                                               int                     m,
                                               int                     n,
                                               hipsolverDoubleComplex* A,
                                               int                     lda,
                                               hipsolverDoubleComplex* tau,
                                               hipsolverDoubleComplex* C,
                                               int                     ldc,
                                               hipsolverDoubleComplex* work,
                                               int                     lwork,
                                               int*                    info)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZunmtr(handle,
                               side,
                               uplo,
                               trans,
                               m,
                               n,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)tau,
                               (hipDoubleComplex*)C,
                               ldc,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);
    case API_FORTRAN:
        return hipsolverZunmtrFortran(handle,
                                      side,
                                      uplo,
                                      trans,
                                      m,
                                      n,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      (hipDoubleComplex*)tau,
                                      (hipDoubleComplex*)C,
                                      ldc,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      info);
    case API_COMPAT:
        return hipsolverDnZunmtr(handle,
                                 side,
                                 uplo,
                                 trans,
                                 m,
                                 n,
                                 (hipDoubleComplex*)A,
                                 lda,
                                 (hipDoubleComplex*)tau,
                                 (hipDoubleComplex*)C,
                                 ldc,
                                 (hipDoubleComplex*)work,
                                 lwork,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** GEBRD ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_gebrd_bufferSize(
    testAPI_t API, hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSgebrd_bufferSize(handle, m, n, lwork);
    case API_FORTRAN:
        return hipsolverSgebrd_bufferSizeFortran(handle, m, n, lwork);
    case API_COMPAT:
        return hipsolverDnSgebrd_bufferSize(handle, m, n, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gebrd_bufferSize(
    testAPI_t API, hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDgebrd_bufferSize(handle, m, n, lwork);
    case API_FORTRAN:
        return hipsolverDgebrd_bufferSizeFortran(handle, m, n, lwork);
    case API_COMPAT:
        return hipsolverDnDgebrd_bufferSize(handle, m, n, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gebrd_bufferSize(
    testAPI_t API, hipsolverHandle_t handle, int m, int n, hipsolverComplex* A, int lda, int* lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCgebrd_bufferSize(handle, m, n, lwork);
    case API_FORTRAN:
        return hipsolverCgebrd_bufferSizeFortran(handle, m, n, lwork);
    case API_COMPAT:
        return hipsolverDnCgebrd_bufferSize(handle, m, n, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gebrd_bufferSize(testAPI_t               API,
                                                    hipsolverHandle_t       handle,
                                                    int                     m,
                                                    int                     n,
                                                    hipsolverDoubleComplex* A,
                                                    int                     lda,
                                                    int*                    lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZgebrd_bufferSize(handle, m, n, lwork);
    case API_FORTRAN:
        return hipsolverZgebrd_bufferSizeFortran(handle, m, n, lwork);
    case API_COMPAT:
        return hipsolverDnZgebrd_bufferSize(handle, m, n, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gebrd(testAPI_t         API,
                                         hipsolverHandle_t handle,
                                         int               m,
                                         int               n,
                                         float*            A,
                                         int               lda,
                                         int               stA,
                                         float*            D,
                                         int               stD,
                                         float*            E,
                                         int               stE,
                                         float*            tauq,
                                         int               stQ,
                                         float*            taup,
                                         int               stP,
                                         float*            work,
                                         int               lwork,
                                         int*              info,
                                         int               bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, info);
    case API_FORTRAN:
        return hipsolverSgebrdFortran(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnSgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gebrd(testAPI_t         API,
                                         hipsolverHandle_t handle,
                                         int               m,
                                         int               n,
                                         double*           A,
                                         int               lda,
                                         int               stA,
                                         double*           D,
                                         int               stD,
                                         double*           E,
                                         int               stE,
                                         double*           tauq,
                                         int               stQ,
                                         double*           taup,
                                         int               stP,
                                         double*           work,
                                         int               lwork,
                                         int*              info,
                                         int               bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, info);
    case API_FORTRAN:
        return hipsolverDgebrdFortran(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnDgebrd(handle, m, n, A, lda, D, E, tauq, taup, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gebrd(testAPI_t         API,
                                         hipsolverHandle_t handle,
                                         int               m,
                                         int               n,
                                         hipsolverComplex* A,
                                         int               lda,
                                         int               stA,
                                         float*            D,
                                         int               stD,
                                         float*            E,
                                         int               stE,
                                         hipsolverComplex* tauq,
                                         int               stQ,
                                         hipsolverComplex* taup,
                                         int               stP,
                                         hipsolverComplex* work,
                                         int               lwork,
                                         int*              info,
                                         int               bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCgebrd(handle,
                               m,
                               n,
                               (hipFloatComplex*)A,
                               lda,
                               D,
                               E,
                               (hipFloatComplex*)tauq,
                               (hipFloatComplex*)taup,
                               (hipFloatComplex*)work,
                               lwork,
                               info);
    case API_FORTRAN:
        return hipsolverCgebrdFortran(handle,
                                      m,
                                      n,
                                      (hipFloatComplex*)A,
                                      lda,
                                      D,
                                      E,
                                      (hipFloatComplex*)tauq,
                                      (hipFloatComplex*)taup,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info);
    case API_COMPAT:
        return hipsolverDnCgebrd(handle,
                                 m,
                                 n,
                                 (hipFloatComplex*)A,
                                 lda,
                                 D,
                                 E,
                                 (hipFloatComplex*)tauq,
                                 (hipFloatComplex*)taup,
                                 (hipFloatComplex*)work,
                                 lwork,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gebrd(testAPI_t               API,
                                         hipsolverHandle_t       handle,
                                         int                     m,
                                         int                     n,
                                         hipsolverDoubleComplex* A,
                                         int                     lda,
                                         int                     stA,
                                         double*                 D,
                                         int                     stD,
                                         double*                 E,
                                         int                     stE,
                                         hipsolverDoubleComplex* tauq,
                                         int                     stQ,
                                         hipsolverDoubleComplex* taup,
                                         int                     stP,
                                         hipsolverDoubleComplex* work,
                                         int                     lwork,
                                         int*                    info,
                                         int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZgebrd(handle,
                               m,
                               n,
                               (hipDoubleComplex*)A,
                               lda,
                               D,
                               E,
                               (hipDoubleComplex*)tauq,
                               (hipDoubleComplex*)taup,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);
    case API_FORTRAN:
        return hipsolverZgebrdFortran(handle,
                                      m,
                                      n,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      D,
                                      E,
                                      (hipDoubleComplex*)tauq,
                                      (hipDoubleComplex*)taup,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      info);
    case API_COMPAT:
        return hipsolverDnZgebrd(handle,
                                 m,
                                 n,
                                 (hipDoubleComplex*)A,
                                 lda,
                                 D,
                                 E,
                                 (hipDoubleComplex*)tauq,
                                 (hipDoubleComplex*)taup,
                                 (hipDoubleComplex*)work,
                                 lwork,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** GEQRF ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_gels_bufferSize(testAPI_t         API,
                                                   hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nrhs,
                                                   float*            A,
                                                   int               lda,
                                                   float*            B,
                                                   int               ldb,
                                                   float*            X,
                                                   int               ldx,
                                                   size_t*           lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSSgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, lwork);
    case API_FORTRAN:
        return hipsolverSSgels_bufferSizeFortran(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, lwork);
    case API_COMPAT:
        return hipsolverDnSSgels_bufferSize(
            handle, m, n, nrhs, A, lda, B, ldb, X, ldx, nullptr, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gels_bufferSize(testAPI_t         API,
                                                   hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nrhs,
                                                   double*           A,
                                                   int               lda,
                                                   double*           B,
                                                   int               ldb,
                                                   double*           X,
                                                   int               ldx,
                                                   size_t*           lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDDgels_bufferSize(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, lwork);
    case API_FORTRAN:
        return hipsolverDDgels_bufferSizeFortran(handle, m, n, nrhs, A, lda, B, ldb, X, ldx, lwork);
    case API_COMPAT:
        return hipsolverDnDDgels_bufferSize(
            handle, m, n, nrhs, A, lda, B, ldb, X, ldx, nullptr, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gels_bufferSize(testAPI_t         API,
                                                   hipsolverHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nrhs,
                                                   hipsolverComplex* A,
                                                   int               lda,
                                                   hipsolverComplex* B,
                                                   int               ldb,
                                                   hipsolverComplex* X,
                                                   int               ldx,
                                                   size_t*           lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCCgels_bufferSize(handle,
                                          m,
                                          n,
                                          nrhs,
                                          (hipFloatComplex*)A,
                                          lda,
                                          (hipFloatComplex*)B,
                                          ldb,
                                          (hipFloatComplex*)X,
                                          ldx,
                                          lwork);
    case API_FORTRAN:
        return hipsolverCCgels_bufferSizeFortran(handle,
                                                 m,
                                                 n,
                                                 nrhs,
                                                 (hipFloatComplex*)A,
                                                 lda,
                                                 (hipFloatComplex*)B,
                                                 ldb,
                                                 (hipFloatComplex*)X,
                                                 ldx,
                                                 lwork);
    case API_COMPAT:
        return hipsolverDnCCgels_bufferSize(handle,
                                            m,
                                            n,
                                            nrhs,
                                            (hipFloatComplex*)A,
                                            lda,
                                            (hipFloatComplex*)B,
                                            ldb,
                                            (hipFloatComplex*)X,
                                            ldx,
                                            nullptr,
                                            lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gels_bufferSize(testAPI_t               API,
                                                   hipsolverHandle_t       handle,
                                                   int                     m,
                                                   int                     n,
                                                   int                     nrhs,
                                                   hipsolverDoubleComplex* A,
                                                   int                     lda,
                                                   hipsolverDoubleComplex* B,
                                                   int                     ldb,
                                                   hipsolverDoubleComplex* X,
                                                   int                     ldx,
                                                   size_t*                 lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZZgels_bufferSize(handle,
                                          m,
                                          n,
                                          nrhs,
                                          (hipDoubleComplex*)A,
                                          lda,
                                          (hipDoubleComplex*)B,
                                          ldb,
                                          (hipDoubleComplex*)X,
                                          ldx,
                                          lwork);
    case API_FORTRAN:
        return hipsolverZZgels_bufferSizeFortran(handle,
                                                 m,
                                                 n,
                                                 nrhs,
                                                 (hipDoubleComplex*)A,
                                                 lda,
                                                 (hipDoubleComplex*)B,
                                                 ldb,
                                                 (hipDoubleComplex*)X,
                                                 ldx,
                                                 lwork);
    case API_COMPAT:
        return hipsolverDnZZgels_bufferSize(handle,
                                            m,
                                            n,
                                            nrhs,
                                            (hipDoubleComplex*)A,
                                            lda,
                                            (hipDoubleComplex*)B,
                                            ldb,
                                            (hipDoubleComplex*)X,
                                            ldx,
                                            nullptr,
                                            lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gels(testAPI_t         API,
                                        bool              INPLACE,
                                        hipsolverHandle_t handle,
                                        int               m,
                                        int               n,
                                        int               nrhs,
                                        float*            A,
                                        int               lda,
                                        int               stA,
                                        float*            B,
                                        int               ldb,
                                        int               stB,
                                        float*            X,
                                        int               ldx,
                                        int               stX,
                                        float*            work,
                                        size_t            lwork,
                                        int*              niters,
                                        int*              info,
                                        int               bc)
{
    switch(api2marshal(API, INPLACE))
    {
    case C_NORMAL:
        return hipsolverSSgels(
            handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, info);
    case C_NORMAL_ALT:
        return hipsolverSSgels(
            handle, m, n, nrhs, A, lda, B, ldb, B, ldb, work, lwork, niters, info);
    case FORTRAN_NORMAL:
        return hipsolverSSgelsFortran(
            handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, info);
    case COMPAT_NORMAL:
        return hipsolverDnSSgels(
            handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gels(testAPI_t         API,
                                        bool              INPLACE,
                                        hipsolverHandle_t handle,
                                        int               m,
                                        int               n,
                                        int               nrhs,
                                        double*           A,
                                        int               lda,
                                        int               stA,
                                        double*           B,
                                        int               ldb,
                                        int               stB,
                                        double*           X,
                                        int               ldx,
                                        int               stX,
                                        double*           work,
                                        size_t            lwork,
                                        int*              niters,
                                        int*              info,
                                        int               bc)
{
    switch(api2marshal(API, INPLACE))
    {
    case C_NORMAL:
        return hipsolverDDgels(
            handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, info);
    case C_NORMAL_ALT:
        return hipsolverDDgels(
            handle, m, n, nrhs, A, lda, B, ldb, B, ldb, work, lwork, niters, info);
    case FORTRAN_NORMAL:
        return hipsolverDDgelsFortran(
            handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, info);
    case COMPAT_NORMAL:
        return hipsolverDnDDgels(
            handle, m, n, nrhs, A, lda, B, ldb, X, ldx, work, lwork, niters, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gels(testAPI_t         API,
                                        bool              INPLACE,
                                        hipsolverHandle_t handle,
                                        int               m,
                                        int               n,
                                        int               nrhs,
                                        hipsolverComplex* A,
                                        int               lda,
                                        int               stA,
                                        hipsolverComplex* B,
                                        int               ldb,
                                        int               stB,
                                        hipsolverComplex* X,
                                        int               ldx,
                                        int               stX,
                                        hipsolverComplex* work,
                                        size_t            lwork,
                                        int*              niters,
                                        int*              info,
                                        int               bc)
{
    switch(api2marshal(API, INPLACE))
    {
    case C_NORMAL:
        return hipsolverCCgels(handle,
                               m,
                               n,
                               nrhs,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)B,
                               ldb,
                               (hipFloatComplex*)X,
                               ldx,
                               work,
                               lwork,
                               niters,
                               info);
    case C_NORMAL_ALT:
        return hipsolverCCgels(handle,
                               m,
                               n,
                               nrhs,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)B,
                               ldb,
                               (hipFloatComplex*)B,
                               ldb,
                               work,
                               lwork,
                               niters,
                               info);
    case FORTRAN_NORMAL:
        return hipsolverCCgelsFortran(handle,
                                      m,
                                      n,
                                      nrhs,
                                      (hipFloatComplex*)A,
                                      lda,
                                      (hipFloatComplex*)B,
                                      ldb,
                                      (hipFloatComplex*)X,
                                      ldx,
                                      work,
                                      lwork,
                                      niters,
                                      info);
    case COMPAT_NORMAL:
        return hipsolverDnCCgels(handle,
                                 m,
                                 n,
                                 nrhs,
                                 (hipFloatComplex*)A,
                                 lda,
                                 (hipFloatComplex*)B,
                                 ldb,
                                 (hipFloatComplex*)X,
                                 ldx,
                                 work,
                                 lwork,
                                 niters,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gels(testAPI_t               API,
                                        bool                    INPLACE,
                                        hipsolverHandle_t       handle,
                                        int                     m,
                                        int                     n,
                                        int                     nrhs,
                                        hipsolverDoubleComplex* A,
                                        int                     lda,
                                        int                     stA,
                                        hipsolverDoubleComplex* B,
                                        int                     ldb,
                                        int                     stB,
                                        hipsolverDoubleComplex* X,
                                        int                     ldx,
                                        int                     stX,
                                        hipsolverDoubleComplex* work,
                                        size_t                  lwork,
                                        int*                    niters,
                                        int*                    info,
                                        int                     bc)
{
    switch(api2marshal(API, INPLACE))
    {
    case C_NORMAL:
        return hipsolverZZgels(handle,
                               m,
                               n,
                               nrhs,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)B,
                               ldb,
                               (hipDoubleComplex*)X,
                               ldx,
                               work,
                               lwork,
                               niters,
                               info);
    case C_NORMAL_ALT:
        return hipsolverZZgels(handle,
                               m,
                               n,
                               nrhs,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)B,
                               ldb,
                               (hipDoubleComplex*)B,
                               ldb,
                               work,
                               lwork,
                               niters,
                               info);
    case FORTRAN_NORMAL:
        return hipsolverZZgelsFortran(handle,
                                      m,
                                      n,
                                      nrhs,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      (hipDoubleComplex*)B,
                                      ldb,
                                      (hipDoubleComplex*)X,
                                      ldx,
                                      work,
                                      lwork,
                                      niters,
                                      info);
    case COMPAT_NORMAL:
        return hipsolverDnZZgels(handle,
                                 m,
                                 n,
                                 nrhs,
                                 (hipDoubleComplex*)A,
                                 lda,
                                 (hipDoubleComplex*)B,
                                 ldb,
                                 (hipDoubleComplex*)X,
                                 ldx,
                                 work,
                                 lwork,
                                 niters,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** GEQRF ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_geqrf_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverDnParams_t params,
                                                    int                 m,
                                                    int                 n,
                                                    float*              A,
                                                    int                 lda,
                                                    float*              tau,
                                                    int*                lworkOnDevice,
                                                    int*                lworkOnHost)
{
    *lworkOnHost = 0;
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSgeqrf_bufferSize(handle, m, n, A, lda, lworkOnDevice);
    case API_FORTRAN:
        return hipsolverSgeqrf_bufferSizeFortran(handle, m, n, A, lda, lworkOnDevice);
    case API_COMPAT:
        return hipsolverDnSgeqrf_bufferSize(handle, m, n, A, lda, lworkOnDevice);
    default:
        *lworkOnDevice = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_geqrf_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverDnParams_t params,
                                                    int                 m,
                                                    int                 n,
                                                    double*             A,
                                                    int                 lda,
                                                    double*             tau,
                                                    int*                lworkOnDevice,
                                                    int*                lworkOnHost)
{
    *lworkOnHost = 0;
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDgeqrf_bufferSize(handle, m, n, A, lda, lworkOnDevice);
    case API_FORTRAN:
        return hipsolverDgeqrf_bufferSizeFortran(handle, m, n, A, lda, lworkOnDevice);
    case API_COMPAT:
        return hipsolverDnDgeqrf_bufferSize(handle, m, n, A, lda, lworkOnDevice);
    default:
        *lworkOnDevice = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_geqrf_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverDnParams_t params,
                                                    int                 m,
                                                    int                 n,
                                                    hipsolverComplex*   A,
                                                    int                 lda,
                                                    hipsolverComplex*   tau,
                                                    int*                lworkOnDevice,
                                                    int*                lworkOnHost)
{
    *lworkOnHost = 0;
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCgeqrf_bufferSize(handle, m, n, (hipFloatComplex*)A, lda, lworkOnDevice);
    case API_FORTRAN:
        return hipsolverCgeqrf_bufferSizeFortran(
            handle, m, n, (hipFloatComplex*)A, lda, lworkOnDevice);
    case API_COMPAT:
        return hipsolverDnCgeqrf_bufferSize(handle, m, n, (hipFloatComplex*)A, lda, lworkOnDevice);
    default:
        *lworkOnDevice = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_geqrf_bufferSize(testAPI_t               API,
                                                    hipsolverHandle_t       handle,
                                                    hipsolverDnParams_t     params,
                                                    int                     m,
                                                    int                     n,
                                                    hipsolverDoubleComplex* A,
                                                    int                     lda,
                                                    hipsolverDoubleComplex* tau,
                                                    int*                    lworkOnDevice,
                                                    int*                    lworkOnHost)
{
    *lworkOnHost = 0;
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZgeqrf_bufferSize(handle, m, n, (hipDoubleComplex*)A, lda, lworkOnDevice);
    case API_FORTRAN:
        return hipsolverZgeqrf_bufferSizeFortran(
            handle, m, n, (hipDoubleComplex*)A, lda, lworkOnDevice);
    case API_COMPAT:
        return hipsolverDnZgeqrf_bufferSize(handle, m, n, (hipDoubleComplex*)A, lda, lworkOnDevice);
    default:
        *lworkOnDevice = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_geqrf_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverDnParams_t params,
                                                    int64_t             m,
                                                    int64_t             n,
                                                    float*              A,
                                                    int64_t             lda,
                                                    float*              tau,
                                                    size_t*             lworkOnDevice,
                                                    size_t*             lworkOnHost)
{
    switch(API)
    {
    case API_COMPAT:
        return hipsolverDnXgeqrf_bufferSize(handle,
                                            params,
                                            m,
                                            n,
                                            HIP_R_32F,
                                            A,
                                            lda,
                                            HIP_R_32F,
                                            tau,
                                            HIP_R_32F,
                                            lworkOnDevice,
                                            lworkOnHost);
    default:
        *lworkOnDevice = 0;
        *lworkOnHost   = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_geqrf_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverDnParams_t params,
                                                    int64_t             m,
                                                    int64_t             n,
                                                    double*             A,
                                                    int64_t             lda,
                                                    double*             tau,
                                                    size_t*             lworkOnDevice,
                                                    size_t*             lworkOnHost)
{
    switch(API)
    {
    case API_COMPAT:
        return hipsolverDnXgeqrf_bufferSize(handle,
                                            params,
                                            m,
                                            n,
                                            HIP_R_64F,
                                            A,
                                            lda,
                                            HIP_R_64F,
                                            tau,
                                            HIP_R_64F,
                                            lworkOnDevice,
                                            lworkOnHost);
    default:
        *lworkOnDevice = 0;
        *lworkOnHost   = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_geqrf_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverDnParams_t params,
                                                    int64_t             m,
                                                    int64_t             n,
                                                    hipsolverComplex*   A,
                                                    int64_t             lda,
                                                    hipsolverComplex*   tau,
                                                    size_t*             lworkOnDevice,
                                                    size_t*             lworkOnHost)
{
    switch(API)
    {
    case API_COMPAT:
        return hipsolverDnXgeqrf_bufferSize(handle,
                                            params,
                                            m,
                                            n,
                                            HIP_C_32F,
                                            A,
                                            lda,
                                            HIP_C_32F,
                                            tau,
                                            HIP_C_32F,
                                            lworkOnDevice,
                                            lworkOnHost);
    default:
        *lworkOnDevice = 0;
        *lworkOnHost   = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_geqrf_bufferSize(testAPI_t               API,
                                                    hipsolverHandle_t       handle,
                                                    hipsolverDnParams_t     params,
                                                    int64_t                 m,
                                                    int64_t                 n,
                                                    hipsolverDoubleComplex* A,
                                                    int64_t                 lda,
                                                    hipsolverDoubleComplex* tau,
                                                    size_t*                 lworkOnDevice,
                                                    size_t*                 lworkOnHost)
{
    switch(API)
    {
    case API_COMPAT:
        return hipsolverDnXgeqrf_bufferSize(handle,
                                            params,
                                            m,
                                            n,
                                            HIP_C_64F,
                                            A,
                                            lda,
                                            HIP_C_64F,
                                            tau,
                                            HIP_C_64F,
                                            lworkOnDevice,
                                            lworkOnHost);
    default:
        *lworkOnDevice = 0;
        *lworkOnHost   = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_geqrf(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverDnParams_t params,
                                         int                 m,
                                         int                 n,
                                         float*              A,
                                         int                 lda,
                                         int                 stA,
                                         float*              tau,
                                         int                 stT,
                                         float*              workOnDevice,
                                         int                 lworkOnDevice,
                                         float*              workOnHost,
                                         int                 lworkOnHost,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSgeqrf(handle, m, n, A, lda, tau, workOnDevice, lworkOnDevice, info);
    case API_FORTRAN:
        return hipsolverSgeqrfFortran(handle, m, n, A, lda, tau, workOnDevice, lworkOnDevice, info);
    case API_COMPAT:
        return hipsolverDnSgeqrf(handle, m, n, A, lda, tau, workOnDevice, lworkOnDevice, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_geqrf(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverDnParams_t params,
                                         int                 m,
                                         int                 n,
                                         double*             A,
                                         int                 lda,
                                         int                 stA,
                                         double*             tau,
                                         int                 stT,
                                         double*             workOnDevice,
                                         int                 lworkOnDevice,
                                         double*             workOnHost,
                                         int                 lworkOnHost,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDgeqrf(handle, m, n, A, lda, tau, workOnDevice, lworkOnDevice, info);
    case API_FORTRAN:
        return hipsolverDgeqrfFortran(handle, m, n, A, lda, tau, workOnDevice, lworkOnDevice, info);
    case API_COMPAT:
        return hipsolverDnDgeqrf(handle, m, n, A, lda, tau, workOnDevice, lworkOnDevice, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_geqrf(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverDnParams_t params,
                                         int                 m,
                                         int                 n,
                                         hipsolverComplex*   A,
                                         int                 lda,
                                         int                 stA,
                                         hipsolverComplex*   tau,
                                         int                 stT,
                                         hipsolverComplex*   workOnDevice,
                                         int                 lworkOnDevice,
                                         hipsolverComplex*   workOnHost,
                                         int                 lworkOnHost,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCgeqrf(handle,
                               m,
                               n,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)tau,
                               (hipFloatComplex*)workOnDevice,
                               lworkOnDevice,
                               info);
    case API_FORTRAN:
        return hipsolverCgeqrfFortran(handle,
                                      m,
                                      n,
                                      (hipFloatComplex*)A,
                                      lda,
                                      (hipFloatComplex*)tau,
                                      (hipFloatComplex*)workOnDevice,
                                      lworkOnDevice,
                                      info);
    case API_COMPAT:
        return hipsolverDnCgeqrf(handle,
                                 m,
                                 n,
                                 (hipFloatComplex*)A,
                                 lda,
                                 (hipFloatComplex*)tau,
                                 (hipFloatComplex*)workOnDevice,
                                 lworkOnDevice,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_geqrf(testAPI_t               API,
                                         hipsolverHandle_t       handle,
                                         hipsolverDnParams_t     params,
                                         int                     m,
                                         int                     n,
                                         hipsolverDoubleComplex* A,
                                         int                     lda,
                                         int                     stA,
                                         hipsolverDoubleComplex* tau,
                                         int                     stT,
                                         hipsolverDoubleComplex* workOnDevice,
                                         int                     lworkOnDevice,
                                         hipsolverDoubleComplex* workOnHost,
                                         int                     lworkOnHost,
                                         int*                    info,
                                         int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZgeqrf(handle,
                               m,
                               n,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)tau,
                               (hipDoubleComplex*)workOnDevice,
                               lworkOnDevice,
                               info);
    case API_FORTRAN:
        return hipsolverZgeqrfFortran(handle,
                                      m,
                                      n,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      (hipDoubleComplex*)tau,
                                      (hipDoubleComplex*)workOnDevice,
                                      lworkOnDevice,
                                      info);
    case API_COMPAT:
        return hipsolverDnZgeqrf(handle,
                                 m,
                                 n,
                                 (hipDoubleComplex*)A,
                                 lda,
                                 (hipDoubleComplex*)tau,
                                 (hipDoubleComplex*)workOnDevice,
                                 lworkOnDevice,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_geqrf(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverDnParams_t params,
                                         int64_t             m,
                                         int64_t             n,
                                         float*              A,
                                         int64_t             lda,
                                         int64_t             stA,
                                         float*              tau,
                                         int64_t             stT,
                                         float*              workOnDevice,
                                         size_t              lworkOnDevice,
                                         float*              workOnHost,
                                         size_t              lworkOnHost,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_COMPAT:
        return hipsolverDnXgeqrf(handle,
                                 params,
                                 m,
                                 n,
                                 HIP_R_32F,
                                 A,
                                 lda,
                                 HIP_R_32F,
                                 tau,
                                 HIP_R_32F,
                                 workOnDevice,
                                 lworkOnDevice,
                                 workOnHost,
                                 lworkOnHost,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_geqrf(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverDnParams_t params,
                                         int64_t             m,
                                         int64_t             n,
                                         double*             A,
                                         int64_t             lda,
                                         int64_t             stA,
                                         double*             tau,
                                         int64_t             stT,
                                         double*             workOnDevice,
                                         size_t              lworkOnDevice,
                                         double*             workOnHost,
                                         size_t              lworkOnHost,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_COMPAT:
        return hipsolverDnXgeqrf(handle,
                                 params,
                                 m,
                                 n,
                                 HIP_R_64F,
                                 A,
                                 lda,
                                 HIP_R_64F,
                                 tau,
                                 HIP_R_64F,
                                 workOnDevice,
                                 lworkOnDevice,
                                 workOnHost,
                                 lworkOnHost,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_geqrf(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverDnParams_t params,
                                         int64_t             m,
                                         int64_t             n,
                                         hipsolverComplex*   A,
                                         int64_t             lda,
                                         int64_t             stA,
                                         hipsolverComplex*   tau,
                                         int64_t             stT,
                                         hipsolverComplex*   workOnDevice,
                                         size_t              lworkOnDevice,
                                         hipsolverComplex*   workOnHost,
                                         size_t              lworkOnHost,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_COMPAT:
        return hipsolverDnXgeqrf(handle,
                                 params,
                                 m,
                                 n,
                                 HIP_C_32F,
                                 A,
                                 lda,
                                 HIP_C_32F,
                                 tau,
                                 HIP_C_32F,
                                 workOnDevice,
                                 lworkOnDevice,
                                 workOnHost,
                                 lworkOnHost,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_geqrf(testAPI_t               API,
                                         hipsolverHandle_t       handle,
                                         hipsolverDnParams_t     params,
                                         int64_t                 m,
                                         int64_t                 n,
                                         hipsolverDoubleComplex* A,
                                         int64_t                 lda,
                                         int64_t                 stA,
                                         hipsolverDoubleComplex* tau,
                                         int64_t                 stT,
                                         hipsolverDoubleComplex* workOnDevice,
                                         size_t                  lworkOnDevice,
                                         hipsolverDoubleComplex* workOnHost,
                                         size_t                  lworkOnHost,
                                         int*                    info,
                                         int                     bc)
{
    switch(API)
    {
    case API_COMPAT:
        return hipsolverDnXgeqrf(handle,
                                 params,
                                 m,
                                 n,
                                 HIP_C_64F,
                                 A,
                                 lda,
                                 HIP_C_64F,
                                 tau,
                                 HIP_C_64F,
                                 workOnDevice,
                                 lworkOnDevice,
                                 workOnHost,
                                 lworkOnHost,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** GESV ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_gesv_bufferSize(testAPI_t         API,
                                                   hipsolverHandle_t handle,
                                                   int               n,
                                                   int               nrhs,
                                                   float*            A,
                                                   int               lda,
                                                   int*              ipiv,
                                                   float*            B,
                                                   int               ldb,
                                                   float*            X,
                                                   int               ldx,
                                                   size_t*           lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSSgesv_bufferSize(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, lwork);
    case API_FORTRAN:
        return hipsolverSSgesv_bufferSizeFortran(
            handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, lwork);
    case API_COMPAT:
        return hipsolverDnSSgesv_bufferSize(
            handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, nullptr, lwork);
    default:
        *lwork;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesv_bufferSize(testAPI_t         API,
                                                   hipsolverHandle_t handle,
                                                   int               n,
                                                   int               nrhs,
                                                   double*           A,
                                                   int               lda,
                                                   int*              ipiv,
                                                   double*           B,
                                                   int               ldb,
                                                   double*           X,
                                                   int               ldx,
                                                   size_t*           lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDDgesv_bufferSize(handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, lwork);
    case API_FORTRAN:
        return hipsolverDDgesv_bufferSizeFortran(
            handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, lwork);
    case API_COMPAT:
        return hipsolverDnDDgesv_bufferSize(
            handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, nullptr, lwork);
    default:
        *lwork;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesv_bufferSize(testAPI_t         API,
                                                   hipsolverHandle_t handle,
                                                   int               n,
                                                   int               nrhs,
                                                   hipsolverComplex* A,
                                                   int               lda,
                                                   int*              ipiv,
                                                   hipsolverComplex* B,
                                                   int               ldb,
                                                   hipsolverComplex* X,
                                                   int               ldx,
                                                   size_t*           lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCCgesv_bufferSize(handle,
                                          n,
                                          nrhs,
                                          (hipFloatComplex*)A,
                                          lda,
                                          ipiv,
                                          (hipFloatComplex*)B,
                                          ldb,
                                          (hipFloatComplex*)X,
                                          ldx,
                                          lwork);
    case API_FORTRAN:
        return hipsolverCCgesv_bufferSizeFortran(handle,
                                                 n,
                                                 nrhs,
                                                 (hipFloatComplex*)A,
                                                 lda,
                                                 ipiv,
                                                 (hipFloatComplex*)B,
                                                 ldb,
                                                 (hipFloatComplex*)X,
                                                 ldx,
                                                 lwork);
    case API_COMPAT:
        return hipsolverDnCCgesv_bufferSize(handle,
                                            n,
                                            nrhs,
                                            (hipFloatComplex*)A,
                                            lda,
                                            ipiv,
                                            (hipFloatComplex*)B,
                                            ldb,
                                            (hipFloatComplex*)X,
                                            ldx,
                                            nullptr,
                                            lwork);
    default:
        *lwork;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesv_bufferSize(testAPI_t               API,
                                                   hipsolverHandle_t       handle,
                                                   int                     n,
                                                   int                     nrhs,
                                                   hipsolverDoubleComplex* A,
                                                   int                     lda,
                                                   int*                    ipiv,
                                                   hipsolverDoubleComplex* B,
                                                   int                     ldb,
                                                   hipsolverDoubleComplex* X,
                                                   int                     ldx,
                                                   size_t*                 lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZZgesv_bufferSize(handle,
                                          n,
                                          nrhs,
                                          (hipDoubleComplex*)A,
                                          lda,
                                          ipiv,
                                          (hipDoubleComplex*)B,
                                          ldb,
                                          (hipDoubleComplex*)X,
                                          ldx,
                                          lwork);
    case API_FORTRAN:
        return hipsolverZZgesv_bufferSizeFortran(handle,
                                                 n,
                                                 nrhs,
                                                 (hipDoubleComplex*)A,
                                                 lda,
                                                 ipiv,
                                                 (hipDoubleComplex*)B,
                                                 ldb,
                                                 (hipDoubleComplex*)X,
                                                 ldx,
                                                 lwork);
    case API_COMPAT:
        return hipsolverDnZZgesv_bufferSize(handle,
                                            n,
                                            nrhs,
                                            (hipDoubleComplex*)A,
                                            lda,
                                            ipiv,
                                            (hipDoubleComplex*)B,
                                            ldb,
                                            (hipDoubleComplex*)X,
                                            ldx,
                                            nullptr,
                                            lwork);
    default:
        *lwork;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesv(testAPI_t         API,
                                        bool              INPLACE,
                                        hipsolverHandle_t handle,
                                        int               n,
                                        int               nrhs,
                                        float*            A,
                                        int               lda,
                                        int               stA,
                                        int*              ipiv,
                                        int               stP,
                                        float*            B,
                                        int               ldb,
                                        int               stB,
                                        float*            X,
                                        int               ldx,
                                        int               stX,
                                        float*            work,
                                        size_t            lwork,
                                        int*              niters,
                                        int*              info,
                                        int               bc)
{
    switch(api2marshal(API, INPLACE))
    {
    case C_NORMAL:
        return hipsolverSSgesv(
            handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, work, lwork, niters, info);
    case C_NORMAL_ALT:
        return hipsolverSSgesv(
            handle, n, nrhs, A, lda, ipiv, B, ldb, B, ldb, work, lwork, niters, info);
    case FORTRAN_NORMAL:
        return hipsolverSSgesvFortran(
            handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, work, lwork, niters, info);
    case COMPAT_NORMAL:
        return hipsolverDnSSgesv(
            handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, work, lwork, niters, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesv(testAPI_t         API,
                                        bool              INPLACE,
                                        hipsolverHandle_t handle,
                                        int               n,
                                        int               nrhs,
                                        double*           A,
                                        int               lda,
                                        int               stA,
                                        int*              ipiv,
                                        int               stP,
                                        double*           B,
                                        int               ldb,
                                        int               stB,
                                        double*           X,
                                        int               ldx,
                                        int               stX,
                                        double*           work,
                                        size_t            lwork,
                                        int*              niters,
                                        int*              info,
                                        int               bc)
{
    switch(api2marshal(API, INPLACE))
    {
    case C_NORMAL:
        return hipsolverDDgesv(
            handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, work, lwork, niters, info);
    case C_NORMAL_ALT:
        return hipsolverDDgesv(
            handle, n, nrhs, A, lda, ipiv, B, ldb, B, ldb, work, lwork, niters, info);
    case FORTRAN_NORMAL:
        return hipsolverDDgesvFortran(
            handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, work, lwork, niters, info);
    case COMPAT_NORMAL:
        return hipsolverDnDDgesv(
            handle, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, work, lwork, niters, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesv(testAPI_t         API,
                                        bool              INPLACE,
                                        hipsolverHandle_t handle,
                                        int               n,
                                        int               nrhs,
                                        hipsolverComplex* A,
                                        int               lda,
                                        int               stA,
                                        int*              ipiv,
                                        int               stP,
                                        hipsolverComplex* B,
                                        int               ldb,
                                        int               stB,
                                        hipsolverComplex* X,
                                        int               ldx,
                                        int               stX,
                                        hipsolverComplex* work,
                                        size_t            lwork,
                                        int*              niters,
                                        int*              info,
                                        int               bc)
{
    switch(api2marshal(API, INPLACE))
    {
    case C_NORMAL:
        return hipsolverCCgesv(handle,
                               n,
                               nrhs,
                               (hipFloatComplex*)A,
                               lda,
                               ipiv,
                               (hipFloatComplex*)B,
                               ldb,
                               (hipFloatComplex*)X,
                               ldx,
                               work,
                               lwork,
                               niters,
                               info);
    case C_NORMAL_ALT:
        return hipsolverCCgesv(handle,
                               n,
                               nrhs,
                               (hipFloatComplex*)A,
                               lda,
                               ipiv,
                               (hipFloatComplex*)B,
                               ldb,
                               (hipFloatComplex*)B,
                               ldb,
                               work,
                               lwork,
                               niters,
                               info);
    case FORTRAN_NORMAL:
        return hipsolverCCgesvFortran(handle,
                                      n,
                                      nrhs,
                                      (hipFloatComplex*)A,
                                      lda,
                                      ipiv,
                                      (hipFloatComplex*)B,
                                      ldb,
                                      (hipFloatComplex*)X,
                                      ldx,
                                      work,
                                      lwork,
                                      niters,
                                      info);
    case COMPAT_NORMAL:
        return hipsolverDnCCgesv(handle,
                                 n,
                                 nrhs,
                                 (hipFloatComplex*)A,
                                 lda,
                                 ipiv,
                                 (hipFloatComplex*)B,
                                 ldb,
                                 (hipFloatComplex*)X,
                                 ldx,
                                 work,
                                 lwork,
                                 niters,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesv(testAPI_t               API,
                                        bool                    INPLACE,
                                        hipsolverHandle_t       handle,
                                        int                     n,
                                        int                     nrhs,
                                        hipsolverDoubleComplex* A,
                                        int                     lda,
                                        int                     stA,
                                        int*                    ipiv,
                                        int                     stP,
                                        hipsolverDoubleComplex* B,
                                        int                     ldb,
                                        int                     stB,
                                        hipsolverDoubleComplex* X,
                                        int                     ldx,
                                        int                     stX,
                                        hipsolverDoubleComplex* work,
                                        size_t                  lwork,
                                        int*                    niters,
                                        int*                    info,
                                        int                     bc)
{
    switch(api2marshal(API, INPLACE))
    {
    case C_NORMAL:
        return hipsolverZZgesv(handle,
                               n,
                               nrhs,
                               (hipDoubleComplex*)A,
                               lda,
                               ipiv,
                               (hipDoubleComplex*)B,
                               ldb,
                               (hipDoubleComplex*)X,
                               ldx,
                               work,
                               lwork,
                               niters,
                               info);
    case C_NORMAL_ALT:
        return hipsolverZZgesv(handle,
                               n,
                               nrhs,
                               (hipDoubleComplex*)A,
                               lda,
                               ipiv,
                               (hipDoubleComplex*)B,
                               ldb,
                               (hipDoubleComplex*)B,
                               ldb,
                               work,
                               lwork,
                               niters,
                               info);
    case FORTRAN_NORMAL:
        return hipsolverZZgesvFortran(handle,
                                      n,
                                      nrhs,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      ipiv,
                                      (hipDoubleComplex*)B,
                                      ldb,
                                      (hipDoubleComplex*)X,
                                      ldx,
                                      work,
                                      lwork,
                                      niters,
                                      info);
    case COMPAT_NORMAL:
        return hipsolverDnZZgesv(handle,
                                 n,
                                 nrhs,
                                 (hipDoubleComplex*)A,
                                 lda,
                                 ipiv,
                                 (hipDoubleComplex*)B,
                                 ldb,
                                 (hipDoubleComplex*)X,
                                 ldx,
                                 work,
                                 lwork,
                                 niters,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** GESVD ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_gesvd_bufferSize(testAPI_t         API,
                                                    hipsolverHandle_t handle,
                                                    signed char       jobu,
                                                    signed char       jobv,
                                                    int               m,
                                                    int               n,
                                                    float*            A,
                                                    int               lda,
                                                    int*              lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSgesvd_bufferSize(handle, jobu, jobv, m, n, lwork);
    case API_FORTRAN:
        return hipsolverSgesvd_bufferSizeFortran(handle, jobu, jobv, m, n, lwork);
    case API_COMPAT:
        return hipsolverDnSgesvd_bufferSize(handle, m, n, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvd_bufferSize(testAPI_t         API,
                                                    hipsolverHandle_t handle,
                                                    signed char       jobu,
                                                    signed char       jobv,
                                                    int               m,
                                                    int               n,
                                                    double*           A,
                                                    int               lda,
                                                    int*              lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDgesvd_bufferSize(handle, jobu, jobv, m, n, lwork);
    case API_FORTRAN:
        return hipsolverDgesvd_bufferSizeFortran(handle, jobu, jobv, m, n, lwork);
    case API_COMPAT:
        return hipsolverDnDgesvd_bufferSize(handle, m, n, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvd_bufferSize(testAPI_t         API,
                                                    hipsolverHandle_t handle,
                                                    signed char       jobu,
                                                    signed char       jobv,
                                                    int               m,
                                                    int               n,
                                                    hipsolverComplex* A,
                                                    int               lda,
                                                    int*              lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCgesvd_bufferSize(handle, jobu, jobv, m, n, lwork);
    case API_FORTRAN:
        return hipsolverCgesvd_bufferSizeFortran(handle, jobu, jobv, m, n, lwork);
    case API_COMPAT:
        return hipsolverDnCgesvd_bufferSize(handle, m, n, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvd_bufferSize(testAPI_t               API,
                                                    hipsolverHandle_t       handle,
                                                    signed char             jobu,
                                                    signed char             jobv,
                                                    int                     m,
                                                    int                     n,
                                                    hipsolverDoubleComplex* A,
                                                    int                     lda,
                                                    int*                    lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZgesvd_bufferSize(handle, jobu, jobv, m, n, lwork);
    case API_FORTRAN:
        return hipsolverZgesvd_bufferSizeFortran(handle, jobu, jobv, m, n, lwork);
    case API_COMPAT:
        return hipsolverDnZgesvd_bufferSize(handle, m, n, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvd(testAPI_t         API,
                                         bool              NRWK,
                                         hipsolverHandle_t handle,
                                         signed char       jobu,
                                         signed char       jobv,
                                         int               m,
                                         int               n,
                                         float*            A,
                                         int               lda,
                                         int               stA,
                                         float*            S,
                                         int               stS,
                                         float*            U,
                                         int               ldu,
                                         int               stU,
                                         float*            V,
                                         int               ldv,
                                         int               stV,
                                         float*            work,
                                         int               lwork,
                                         float*            rwork,
                                         int               stRW,
                                         int*              info,
                                         int               bc)
{
    switch(api2marshal(API, NRWK))
    {
    case C_NORMAL:
        return hipsolverSgesvd(
            handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info);
    case C_NORMAL_ALT:
        return hipsolverSgesvd(
            handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, nullptr, info);
    case FORTRAN_NORMAL:
        return hipsolverSgesvdFortran(
            handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info);
    case COMPAT_NORMAL:
        return hipsolverDnSgesvd(
            handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvd(testAPI_t         API,
                                         bool              NRWK,
                                         hipsolverHandle_t handle,
                                         signed char       jobu,
                                         signed char       jobv,
                                         int               m,
                                         int               n,
                                         double*           A,
                                         int               lda,
                                         int               stA,
                                         double*           S,
                                         int               stS,
                                         double*           U,
                                         int               ldu,
                                         int               stU,
                                         double*           V,
                                         int               ldv,
                                         int               stV,
                                         double*           work,
                                         int               lwork,
                                         double*           rwork,
                                         int               stRW,
                                         int*              info,
                                         int               bc)
{
    switch(api2marshal(API, NRWK))
    {
    case C_NORMAL:
        return hipsolverDgesvd(
            handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info);
    case C_NORMAL_ALT:
        return hipsolverDgesvd(
            handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, nullptr, info);
    case FORTRAN_NORMAL:
        return hipsolverDgesvdFortran(
            handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info);
    case COMPAT_NORMAL:
        return hipsolverDnDgesvd(
            handle, jobu, jobv, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, rwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvd(testAPI_t         API,
                                         bool              NRWK,
                                         hipsolverHandle_t handle,
                                         signed char       jobu,
                                         signed char       jobv,
                                         int               m,
                                         int               n,
                                         hipsolverComplex* A,
                                         int               lda,
                                         int               stA,
                                         float*            S,
                                         int               stS,
                                         hipsolverComplex* U,
                                         int               ldu,
                                         int               stU,
                                         hipsolverComplex* V,
                                         int               ldv,
                                         int               stV,
                                         hipsolverComplex* work,
                                         int               lwork,
                                         float*            rwork,
                                         int               stRW,
                                         int*              info,
                                         int               bc)
{
    switch(api2marshal(API, NRWK))
    {
    case C_NORMAL:
        return hipsolverCgesvd(handle,
                               jobu,
                               jobv,
                               m,
                               n,
                               (hipFloatComplex*)A,
                               lda,
                               S,
                               (hipFloatComplex*)U,
                               ldu,
                               (hipFloatComplex*)V,
                               ldv,
                               (hipFloatComplex*)work,
                               lwork,
                               rwork,
                               info);
    case C_NORMAL_ALT:
        return hipsolverCgesvd(handle,
                               jobu,
                               jobv,
                               m,
                               n,
                               (hipFloatComplex*)A,
                               lda,
                               S,
                               (hipFloatComplex*)U,
                               ldu,
                               (hipFloatComplex*)V,
                               ldv,
                               (hipFloatComplex*)work,
                               lwork,
                               nullptr,
                               info);
    case FORTRAN_NORMAL:
        return hipsolverCgesvdFortran(handle,
                                      jobu,
                                      jobv,
                                      m,
                                      n,
                                      (hipFloatComplex*)A,
                                      lda,
                                      S,
                                      (hipFloatComplex*)U,
                                      ldu,
                                      (hipFloatComplex*)V,
                                      ldv,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      rwork,
                                      info);
    case COMPAT_NORMAL:
        return hipsolverDnCgesvd(handle,
                                 jobu,
                                 jobv,
                                 m,
                                 n,
                                 (hipFloatComplex*)A,
                                 lda,
                                 S,
                                 (hipFloatComplex*)U,
                                 ldu,
                                 (hipFloatComplex*)V,
                                 ldv,
                                 (hipFloatComplex*)work,
                                 lwork,
                                 rwork,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvd(testAPI_t               API,
                                         bool                    NRWK,
                                         hipsolverHandle_t       handle,
                                         signed char             jobu,
                                         signed char             jobv,
                                         int                     m,
                                         int                     n,
                                         hipsolverDoubleComplex* A,
                                         int                     lda,
                                         int                     stA,
                                         double*                 S,
                                         int                     stS,
                                         hipsolverDoubleComplex* U,
                                         int                     ldu,
                                         int                     stU,
                                         hipsolverDoubleComplex* V,
                                         int                     ldv,
                                         int                     stV,
                                         hipsolverDoubleComplex* work,
                                         int                     lwork,
                                         double*                 rwork,
                                         int                     stRW,
                                         int*                    info,
                                         int                     bc)
{
    switch(api2marshal(API, NRWK))
    {
    case C_NORMAL:
        return hipsolverZgesvd(handle,
                               jobu,
                               jobv,
                               m,
                               n,
                               (hipDoubleComplex*)A,
                               lda,
                               S,
                               (hipDoubleComplex*)U,
                               ldu,
                               (hipDoubleComplex*)V,
                               ldv,
                               (hipDoubleComplex*)work,
                               lwork,
                               rwork,
                               info);
    case C_NORMAL_ALT:
        return hipsolverZgesvd(handle,
                               jobu,
                               jobv,
                               m,
                               n,
                               (hipDoubleComplex*)A,
                               lda,
                               S,
                               (hipDoubleComplex*)U,
                               ldu,
                               (hipDoubleComplex*)V,
                               ldv,
                               (hipDoubleComplex*)work,
                               lwork,
                               nullptr,
                               info);
    case FORTRAN_NORMAL:
        return hipsolverZgesvdFortran(handle,
                                      jobu,
                                      jobv,
                                      m,
                                      n,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      S,
                                      (hipDoubleComplex*)U,
                                      ldu,
                                      (hipDoubleComplex*)V,
                                      ldv,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      rwork,
                                      info);
    case COMPAT_NORMAL:
        return hipsolverDnZgesvd(handle,
                                 jobu,
                                 jobv,
                                 m,
                                 n,
                                 (hipDoubleComplex*)A,
                                 lda,
                                 S,
                                 (hipDoubleComplex*)U,
                                 ldu,
                                 (hipDoubleComplex*)V,
                                 ldv,
                                 (hipDoubleComplex*)work,
                                 lwork,
                                 rwork,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** GESVDJ ********************/
inline hipsolverStatus_t hipsolver_gesvdj_bufferSize(testAPI_t             API,
                                                     bool                  STRIDED,
                                                     hipsolverHandle_t     handle,
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
                                                     int*                  lwork,
                                                     hipsolverGesvdjInfo_t params,
                                                     int                   bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverSgesvdj_bufferSize(
            handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params);
    case C_NORMAL_ALT:
        return hipsolverSgesvdjBatched_bufferSize(
            handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, bc);
    case FORTRAN_NORMAL:
        return hipsolverSgesvdj_bufferSizeFortran(
            handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params);
    case FORTRAN_NORMAL_ALT:
        return hipsolverSgesvdjBatched_bufferSizeFortran(
            handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, bc);
    case COMPAT_NORMAL:
        return hipsolverDnSgesvdj_bufferSize(
            handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params);
    case COMPAT_NORMAL_ALT:
        return hipsolverDnSgesvdjBatched_bufferSize(
            handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvdj_bufferSize(testAPI_t             API,
                                                     bool                  STRIDED,
                                                     hipsolverHandle_t     handle,
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
                                                     int*                  lwork,
                                                     hipsolverGesvdjInfo_t params,
                                                     int                   bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverDgesvdj_bufferSize(
            handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params);
    case C_NORMAL_ALT:
        return hipsolverDgesvdjBatched_bufferSize(
            handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, bc);
    case FORTRAN_NORMAL:
        return hipsolverDgesvdj_bufferSizeFortran(
            handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params);
    case FORTRAN_NORMAL_ALT:
        return hipsolverDgesvdjBatched_bufferSizeFortran(
            handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, bc);
    case COMPAT_NORMAL:
        return hipsolverDnDgesvdj_bufferSize(
            handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params);
    case COMPAT_NORMAL_ALT:
        return hipsolverDnDgesvdjBatched_bufferSize(
            handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork, params, bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvdj_bufferSize(testAPI_t             API,
                                                     bool                  STRIDED,
                                                     hipsolverHandle_t     handle,
                                                     hipsolverEigMode_t    jobz,
                                                     int                   econ,
                                                     int                   m,
                                                     int                   n,
                                                     hipsolverComplex*     A,
                                                     int                   lda,
                                                     float*                S,
                                                     hipsolverComplex*     U,
                                                     int                   ldu,
                                                     hipsolverComplex*     V,
                                                     int                   ldv,
                                                     int*                  lwork,
                                                     hipsolverGesvdjInfo_t params,
                                                     int                   bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverCgesvdj_bufferSize(handle,
                                           jobz,
                                           econ,
                                           m,
                                           n,
                                           (hipFloatComplex*)A,
                                           lda,
                                           S,
                                           (hipFloatComplex*)U,
                                           ldu,
                                           (hipFloatComplex*)V,
                                           ldv,
                                           lwork,
                                           params);
    case C_NORMAL_ALT:
        return hipsolverCgesvdjBatched_bufferSize(handle,
                                                  jobz,
                                                  m,
                                                  n,
                                                  (hipFloatComplex*)A,
                                                  lda,
                                                  S,
                                                  (hipFloatComplex*)U,
                                                  ldu,
                                                  (hipFloatComplex*)V,
                                                  ldv,
                                                  lwork,
                                                  params,
                                                  bc);
    case FORTRAN_NORMAL:
        return hipsolverCgesvdj_bufferSizeFortran(handle,
                                                  jobz,
                                                  econ,
                                                  m,
                                                  n,
                                                  (hipFloatComplex*)A,
                                                  lda,
                                                  S,
                                                  (hipFloatComplex*)U,
                                                  ldu,
                                                  (hipFloatComplex*)V,
                                                  ldv,
                                                  lwork,
                                                  params);
    case FORTRAN_NORMAL_ALT:
        return hipsolverCgesvdjBatched_bufferSizeFortran(handle,
                                                         jobz,
                                                         m,
                                                         n,
                                                         (hipFloatComplex*)A,
                                                         lda,
                                                         S,
                                                         (hipFloatComplex*)U,
                                                         ldu,
                                                         (hipFloatComplex*)V,
                                                         ldv,
                                                         lwork,
                                                         params,
                                                         bc);
    case COMPAT_NORMAL:
        return hipsolverDnCgesvdj_bufferSize(handle,
                                             jobz,
                                             econ,
                                             m,
                                             n,
                                             (hipFloatComplex*)A,
                                             lda,
                                             S,
                                             (hipFloatComplex*)U,
                                             ldu,
                                             (hipFloatComplex*)V,
                                             ldv,
                                             lwork,
                                             params);
    case COMPAT_NORMAL_ALT:
        return hipsolverDnCgesvdjBatched_bufferSize(handle,
                                                    jobz,
                                                    m,
                                                    n,
                                                    (hipFloatComplex*)A,
                                                    lda,
                                                    S,
                                                    (hipFloatComplex*)U,
                                                    ldu,
                                                    (hipFloatComplex*)V,
                                                    ldv,
                                                    lwork,
                                                    params,
                                                    bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvdj_bufferSize(testAPI_t               API,
                                                     bool                    STRIDED,
                                                     hipsolverHandle_t       handle,
                                                     hipsolverEigMode_t      jobz,
                                                     int                     econ,
                                                     int                     m,
                                                     int                     n,
                                                     hipsolverDoubleComplex* A,
                                                     int                     lda,
                                                     double*                 S,
                                                     hipsolverDoubleComplex* U,
                                                     int                     ldu,
                                                     hipsolverDoubleComplex* V,
                                                     int                     ldv,
                                                     int*                    lwork,
                                                     hipsolverGesvdjInfo_t   params,
                                                     int                     bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverZgesvdj_bufferSize(handle,
                                           jobz,
                                           econ,
                                           m,
                                           n,
                                           (hipDoubleComplex*)A,
                                           lda,
                                           S,
                                           (hipDoubleComplex*)U,
                                           ldu,
                                           (hipDoubleComplex*)V,
                                           ldv,
                                           lwork,
                                           params);
    case C_NORMAL_ALT:
        return hipsolverZgesvdjBatched_bufferSize(handle,
                                                  jobz,
                                                  m,
                                                  n,
                                                  (hipDoubleComplex*)A,
                                                  lda,
                                                  S,
                                                  (hipDoubleComplex*)U,
                                                  ldu,
                                                  (hipDoubleComplex*)V,
                                                  ldv,
                                                  lwork,
                                                  params,
                                                  bc);
    case FORTRAN_NORMAL:
        return hipsolverZgesvdj_bufferSizeFortran(handle,
                                                  jobz,
                                                  econ,
                                                  m,
                                                  n,
                                                  (hipDoubleComplex*)A,
                                                  lda,
                                                  S,
                                                  (hipDoubleComplex*)U,
                                                  ldu,
                                                  (hipDoubleComplex*)V,
                                                  ldv,
                                                  lwork,
                                                  params);
    case FORTRAN_NORMAL_ALT:
        return hipsolverZgesvdjBatched_bufferSizeFortran(handle,
                                                         jobz,
                                                         m,
                                                         n,
                                                         (hipDoubleComplex*)A,
                                                         lda,
                                                         S,
                                                         (hipDoubleComplex*)U,
                                                         ldu,
                                                         (hipDoubleComplex*)V,
                                                         ldv,
                                                         lwork,
                                                         params,
                                                         bc);
    case COMPAT_NORMAL:
        return hipsolverDnZgesvdj_bufferSize(handle,
                                             jobz,
                                             econ,
                                             m,
                                             n,
                                             (hipDoubleComplex*)A,
                                             lda,
                                             S,
                                             (hipDoubleComplex*)U,
                                             ldu,
                                             (hipDoubleComplex*)V,
                                             ldv,
                                             lwork,
                                             params);
    case COMPAT_NORMAL_ALT:
        return hipsolverDnZgesvdjBatched_bufferSize(handle,
                                                    jobz,
                                                    m,
                                                    n,
                                                    (hipDoubleComplex*)A,
                                                    lda,
                                                    S,
                                                    (hipDoubleComplex*)U,
                                                    ldu,
                                                    (hipDoubleComplex*)V,
                                                    ldv,
                                                    lwork,
                                                    params,
                                                    bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvdj(testAPI_t             API,
                                          bool                  STRIDED,
                                          hipsolverHandle_t     handle,
                                          hipsolverEigMode_t    jobz,
                                          int                   econ,
                                          int                   m,
                                          int                   n,
                                          float*                A,
                                          int                   lda,
                                          int                   stA,
                                          float*                S,
                                          int                   stS,
                                          float*                U,
                                          int                   ldu,
                                          int                   stU,
                                          float*                V,
                                          int                   ldv,
                                          int                   stV,
                                          float*                work,
                                          int                   lwork,
                                          int*                  info,
                                          hipsolverGesvdjInfo_t params,
                                          int                   bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverSgesvdj(
            handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params);
    case C_NORMAL_ALT:
        return hipsolverSgesvdjBatched(
            handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, bc);
    case FORTRAN_NORMAL:
        return hipsolverSgesvdjFortran(
            handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params);
    case FORTRAN_NORMAL_ALT:
        return hipsolverSgesvdjBatchedFortran(
            handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, bc);
    case COMPAT_NORMAL:
        return hipsolverDnSgesvdj(
            handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params);
    case COMPAT_NORMAL_ALT:
        return hipsolverDnSgesvdjBatched(
            handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvdj(testAPI_t             API,
                                          bool                  STRIDED,
                                          hipsolverHandle_t     handle,
                                          hipsolverEigMode_t    jobz,
                                          int                   econ,
                                          int                   m,
                                          int                   n,
                                          double*               A,
                                          int                   lda,
                                          int                   stA,
                                          double*               S,
                                          int                   stS,
                                          double*               U,
                                          int                   ldu,
                                          int                   stU,
                                          double*               V,
                                          int                   ldv,
                                          int                   stV,
                                          double*               work,
                                          int                   lwork,
                                          int*                  info,
                                          hipsolverGesvdjInfo_t params,
                                          int                   bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverDgesvdj(
            handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params);
    case C_NORMAL_ALT:
        return hipsolverDgesvdjBatched(
            handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, bc);
    case FORTRAN_NORMAL:
        return hipsolverDgesvdjFortran(
            handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params);
    case FORTRAN_NORMAL_ALT:
        return hipsolverDgesvdjBatchedFortran(
            handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, bc);
    case COMPAT_NORMAL:
        return hipsolverDnDgesvdj(
            handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params);
    case COMPAT_NORMAL_ALT:
        return hipsolverDnDgesvdjBatched(
            handle, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params, bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvdj(testAPI_t             API,
                                          bool                  STRIDED,
                                          hipsolverHandle_t     handle,
                                          hipsolverEigMode_t    jobz,
                                          int                   econ,
                                          int                   m,
                                          int                   n,
                                          hipsolverComplex*     A,
                                          int                   lda,
                                          int                   stA,
                                          float*                S,
                                          int                   stS,
                                          hipsolverComplex*     U,
                                          int                   ldu,
                                          int                   stU,
                                          hipsolverComplex*     V,
                                          int                   ldv,
                                          int                   stV,
                                          hipsolverComplex*     work,
                                          int                   lwork,
                                          int*                  info,
                                          hipsolverGesvdjInfo_t params,
                                          int                   bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverCgesvdj(handle,
                                jobz,
                                econ,
                                m,
                                n,
                                (hipFloatComplex*)A,
                                lda,
                                S,
                                (hipFloatComplex*)U,
                                ldu,
                                (hipFloatComplex*)V,
                                ldv,
                                (hipFloatComplex*)work,
                                lwork,
                                info,
                                params);
    case C_NORMAL_ALT:
        return hipsolverCgesvdjBatched(handle,
                                       jobz,
                                       m,
                                       n,
                                       (hipFloatComplex*)A,
                                       lda,
                                       S,
                                       (hipFloatComplex*)U,
                                       ldu,
                                       (hipFloatComplex*)V,
                                       ldv,
                                       (hipFloatComplex*)work,
                                       lwork,
                                       info,
                                       params,
                                       bc);
    case FORTRAN_NORMAL:
        return hipsolverCgesvdjFortran(handle,
                                       jobz,
                                       econ,
                                       m,
                                       n,
                                       (hipFloatComplex*)A,
                                       lda,
                                       S,
                                       (hipFloatComplex*)U,
                                       ldu,
                                       (hipFloatComplex*)V,
                                       ldv,
                                       (hipFloatComplex*)work,
                                       lwork,
                                       info,
                                       params);
    case FORTRAN_NORMAL_ALT:
        return hipsolverCgesvdjBatchedFortran(handle,
                                              jobz,
                                              m,
                                              n,
                                              (hipFloatComplex*)A,
                                              lda,
                                              S,
                                              (hipFloatComplex*)U,
                                              ldu,
                                              (hipFloatComplex*)V,
                                              ldv,
                                              (hipFloatComplex*)work,
                                              lwork,
                                              info,
                                              params,
                                              bc);
    case COMPAT_NORMAL:
        return hipsolverDnCgesvdj(handle,
                                  jobz,
                                  econ,
                                  m,
                                  n,
                                  (hipFloatComplex*)A,
                                  lda,
                                  S,
                                  (hipFloatComplex*)U,
                                  ldu,
                                  (hipFloatComplex*)V,
                                  ldv,
                                  (hipFloatComplex*)work,
                                  lwork,
                                  info,
                                  params);
    case COMPAT_NORMAL_ALT:
        return hipsolverDnCgesvdjBatched(handle,
                                         jobz,
                                         m,
                                         n,
                                         (hipFloatComplex*)A,
                                         lda,
                                         S,
                                         (hipFloatComplex*)U,
                                         ldu,
                                         (hipFloatComplex*)V,
                                         ldv,
                                         (hipFloatComplex*)work,
                                         lwork,
                                         info,
                                         params,
                                         bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvdj(testAPI_t               API,
                                          bool                    STRIDED,
                                          hipsolverHandle_t       handle,
                                          hipsolverEigMode_t      jobz,
                                          int                     econ,
                                          int                     m,
                                          int                     n,
                                          hipsolverDoubleComplex* A,
                                          int                     lda,
                                          int                     stA,
                                          double*                 S,
                                          int                     stS,
                                          hipsolverDoubleComplex* U,
                                          int                     ldu,
                                          int                     stU,
                                          hipsolverDoubleComplex* V,
                                          int                     ldv,
                                          int                     stV,
                                          hipsolverDoubleComplex* work,
                                          int                     lwork,
                                          int*                    info,
                                          hipsolverGesvdjInfo_t   params,
                                          int                     bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverZgesvdj(handle,
                                jobz,
                                econ,
                                m,
                                n,
                                (hipDoubleComplex*)A,
                                lda,
                                S,
                                (hipDoubleComplex*)U,
                                ldu,
                                (hipDoubleComplex*)V,
                                ldv,
                                (hipDoubleComplex*)work,
                                lwork,
                                info,
                                params);
    case C_NORMAL_ALT:
        return hipsolverZgesvdjBatched(handle,
                                       jobz,
                                       m,
                                       n,
                                       (hipDoubleComplex*)A,
                                       lda,
                                       S,
                                       (hipDoubleComplex*)U,
                                       ldu,
                                       (hipDoubleComplex*)V,
                                       ldv,
                                       (hipDoubleComplex*)work,
                                       lwork,
                                       info,
                                       params,
                                       bc);
    case FORTRAN_NORMAL:
        return hipsolverZgesvdjFortran(handle,
                                       jobz,
                                       econ,
                                       m,
                                       n,
                                       (hipDoubleComplex*)A,
                                       lda,
                                       S,
                                       (hipDoubleComplex*)U,
                                       ldu,
                                       (hipDoubleComplex*)V,
                                       ldv,
                                       (hipDoubleComplex*)work,
                                       lwork,
                                       info,
                                       params);
    case FORTRAN_NORMAL_ALT:
        return hipsolverZgesvdjBatchedFortran(handle,
                                              jobz,
                                              m,
                                              n,
                                              (hipDoubleComplex*)A,
                                              lda,
                                              S,
                                              (hipDoubleComplex*)U,
                                              ldu,
                                              (hipDoubleComplex*)V,
                                              ldv,
                                              (hipDoubleComplex*)work,
                                              lwork,
                                              info,
                                              params,
                                              bc);
    case COMPAT_NORMAL:
        return hipsolverDnZgesvdj(handle,
                                  jobz,
                                  econ,
                                  m,
                                  n,
                                  (hipDoubleComplex*)A,
                                  lda,
                                  S,
                                  (hipDoubleComplex*)U,
                                  ldu,
                                  (hipDoubleComplex*)V,
                                  ldv,
                                  (hipDoubleComplex*)work,
                                  lwork,
                                  info,
                                  params);
    case COMPAT_NORMAL_ALT:
        return hipsolverDnZgesvdjBatched(handle,
                                         jobz,
                                         m,
                                         n,
                                         (hipDoubleComplex*)A,
                                         lda,
                                         S,
                                         (hipDoubleComplex*)U,
                                         ldu,
                                         (hipDoubleComplex*)V,
                                         ldv,
                                         (hipDoubleComplex*)work,
                                         lwork,
                                         info,
                                         params,
                                         bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** GESVDA ********************/
inline hipsolverStatus_t hipsolver_gesvda_bufferSize(testAPI_t          API,
                                                     bool               STRIDED,
                                                     hipsolverHandle_t  handle,
                                                     hipsolverEigMode_t jobz,
                                                     int                rank,
                                                     int                m,
                                                     int                n,
                                                     float*             A,
                                                     int                lda,
                                                     long long int      stA,
                                                     float*             S,
                                                     long long int      stS,
                                                     float*             U,
                                                     int                ldu,
                                                     long long int      stU,
                                                     float*             V,
                                                     int                ldv,
                                                     long long int      stV,
                                                     int*               lwork,
                                                     int                bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case COMPAT_NORMAL_ALT:
        return hipsolverDnSgesvdaStridedBatched_bufferSize(
            handle, jobz, rank, m, n, A, lda, stA, S, stS, U, ldu, stU, V, ldv, stV, lwork, bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvda_bufferSize(testAPI_t          API,
                                                     bool               STRIDED,
                                                     hipsolverHandle_t  handle,
                                                     hipsolverEigMode_t jobz,
                                                     int                rank,
                                                     int                m,
                                                     int                n,
                                                     double*            A,
                                                     int                lda,
                                                     long long int      stA,
                                                     double*            S,
                                                     long long int      stS,
                                                     double*            U,
                                                     int                ldu,
                                                     long long int      stU,
                                                     double*            V,
                                                     int                ldv,
                                                     long long int      stV,
                                                     int*               lwork,
                                                     int                bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case COMPAT_NORMAL_ALT:
        return hipsolverDnDgesvdaStridedBatched_bufferSize(
            handle, jobz, rank, m, n, A, lda, stA, S, stS, U, ldu, stU, V, ldv, stV, lwork, bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvda_bufferSize(testAPI_t          API,
                                                     bool               STRIDED,
                                                     hipsolverHandle_t  handle,
                                                     hipsolverEigMode_t jobz,
                                                     int                rank,
                                                     int                m,
                                                     int                n,
                                                     hipsolverComplex*  A,
                                                     int                lda,
                                                     long long int      stA,
                                                     float*             S,
                                                     long long int      stS,
                                                     hipsolverComplex*  U,
                                                     int                ldu,
                                                     long long int      stU,
                                                     hipsolverComplex*  V,
                                                     int                ldv,
                                                     long long int      stV,
                                                     int*               lwork,
                                                     int                bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case COMPAT_NORMAL_ALT:
        return hipsolverDnCgesvdaStridedBatched_bufferSize(handle,
                                                           jobz,
                                                           rank,
                                                           m,
                                                           n,
                                                           (hipFloatComplex*)A,
                                                           lda,
                                                           stA,
                                                           S,
                                                           stS,
                                                           (hipFloatComplex*)U,
                                                           ldu,
                                                           stU,
                                                           (hipFloatComplex*)V,
                                                           ldv,
                                                           stV,
                                                           lwork,
                                                           bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvda_bufferSize(testAPI_t               API,
                                                     bool                    STRIDED,
                                                     hipsolverHandle_t       handle,
                                                     hipsolverEigMode_t      jobz,
                                                     int                     rank,
                                                     int                     m,
                                                     int                     n,
                                                     hipsolverDoubleComplex* A,
                                                     int                     lda,
                                                     long long int           stA,
                                                     double*                 S,
                                                     long long int           stS,
                                                     hipsolverDoubleComplex* U,
                                                     int                     ldu,
                                                     long long int           stU,
                                                     hipsolverDoubleComplex* V,
                                                     int                     ldv,
                                                     long long int           stV,
                                                     int*                    lwork,
                                                     int                     bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case COMPAT_NORMAL_ALT:
        return hipsolverDnZgesvdaStridedBatched_bufferSize(handle,
                                                           jobz,
                                                           rank,
                                                           m,
                                                           n,
                                                           (hipDoubleComplex*)A,
                                                           lda,
                                                           stA,
                                                           S,
                                                           stS,
                                                           (hipDoubleComplex*)U,
                                                           ldu,
                                                           stU,
                                                           (hipDoubleComplex*)V,
                                                           ldv,
                                                           stV,
                                                           lwork,
                                                           bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvda(testAPI_t          API,
                                          bool               STRIDED,
                                          hipsolverHandle_t  handle,
                                          hipsolverEigMode_t jobz,
                                          int                rank,
                                          int                m,
                                          int                n,
                                          float*             A,
                                          int                lda,
                                          int                stA,
                                          float*             S,
                                          int                stS,
                                          float*             U,
                                          int                ldu,
                                          int                stU,
                                          float*             V,
                                          int                ldv,
                                          int                stV,
                                          float*             work,
                                          int                lwork,
                                          int*               info,
                                          double*            hRnrmF,
                                          int                bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case COMPAT_NORMAL_ALT:
        return hipsolverDnSgesvdaStridedBatched(handle,
                                                jobz,
                                                rank,
                                                m,
                                                n,
                                                A,
                                                lda,
                                                stA,
                                                S,
                                                stS,
                                                U,
                                                ldu,
                                                stU,
                                                V,
                                                ldv,
                                                stV,
                                                work,
                                                lwork,
                                                info,
                                                hRnrmF,
                                                bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvda(testAPI_t          API,
                                          bool               STRIDED,
                                          hipsolverHandle_t  handle,
                                          hipsolverEigMode_t jobz,
                                          int                rank,
                                          int                m,
                                          int                n,
                                          double*            A,
                                          int                lda,
                                          int                stA,
                                          double*            S,
                                          int                stS,
                                          double*            U,
                                          int                ldu,
                                          int                stU,
                                          double*            V,
                                          int                ldv,
                                          int                stV,
                                          double*            work,
                                          int                lwork,
                                          int*               info,
                                          double*            hRnrmF,
                                          int                bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case COMPAT_NORMAL_ALT:
        return hipsolverDnDgesvdaStridedBatched(handle,
                                                jobz,
                                                rank,
                                                m,
                                                n,
                                                A,
                                                lda,
                                                stA,
                                                S,
                                                stS,
                                                U,
                                                ldu,
                                                stU,
                                                V,
                                                ldv,
                                                stV,
                                                work,
                                                lwork,
                                                info,
                                                hRnrmF,
                                                bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvda(testAPI_t          API,
                                          bool               STRIDED,
                                          hipsolverHandle_t  handle,
                                          hipsolverEigMode_t jobz,
                                          int                rank,
                                          int                m,
                                          int                n,
                                          hipsolverComplex*  A,
                                          int                lda,
                                          int                stA,
                                          float*             S,
                                          int                stS,
                                          hipsolverComplex*  U,
                                          int                ldu,
                                          int                stU,
                                          hipsolverComplex*  V,
                                          int                ldv,
                                          int                stV,
                                          hipsolverComplex*  work,
                                          int                lwork,
                                          int*               info,
                                          double*            hRnrmF,
                                          int                bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case COMPAT_NORMAL_ALT:
        return hipsolverDnCgesvdaStridedBatched(handle,
                                                jobz,
                                                rank,
                                                m,
                                                n,
                                                (hipFloatComplex*)A,
                                                lda,
                                                stA,
                                                S,
                                                stS,
                                                (hipFloatComplex*)U,
                                                ldu,
                                                stU,
                                                (hipFloatComplex*)V,
                                                ldv,
                                                stV,
                                                (hipFloatComplex*)work,
                                                lwork,
                                                info,
                                                hRnrmF,
                                                bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_gesvda(testAPI_t               API,
                                          bool                    STRIDED,
                                          hipsolverHandle_t       handle,
                                          hipsolverEigMode_t      jobz,
                                          int                     rank,
                                          int                     m,
                                          int                     n,
                                          hipsolverDoubleComplex* A,
                                          int                     lda,
                                          int                     stA,
                                          double*                 S,
                                          int                     stS,
                                          hipsolverDoubleComplex* U,
                                          int                     ldu,
                                          int                     stU,
                                          hipsolverDoubleComplex* V,
                                          int                     ldv,
                                          int                     stV,
                                          hipsolverDoubleComplex* work,
                                          int                     lwork,
                                          int*                    info,
                                          double*                 hRnrmF,
                                          int                     bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case COMPAT_NORMAL_ALT:
        return hipsolverDnZgesvdaStridedBatched(handle,
                                                jobz,
                                                rank,
                                                m,
                                                n,
                                                (hipDoubleComplex*)A,
                                                lda,
                                                stA,
                                                S,
                                                stS,
                                                (hipDoubleComplex*)U,
                                                ldu,
                                                stU,
                                                (hipDoubleComplex*)V,
                                                ldv,
                                                stV,
                                                (hipDoubleComplex*)work,
                                                lwork,
                                                info,
                                                hRnrmF,
                                                bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** GETRF ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_getrf_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverDnParams_t params,
                                                    int                 m,
                                                    int                 n,
                                                    float*              A,
                                                    int                 lda,
                                                    int*                lworkOnDevice,
                                                    int*                lworkOnHost)
{
    *lworkOnHost = 0;
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSgetrf_bufferSize(handle, m, n, A, lda, lworkOnDevice);
    case API_FORTRAN:
        return hipsolverSgetrf_bufferSizeFortran(handle, m, n, A, lda, lworkOnDevice);
    case API_COMPAT:
        return hipsolverDnSgetrf_bufferSize(handle, m, n, A, lda, lworkOnDevice);
    default:
        *lworkOnDevice = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverDnParams_t params,
                                                    int                 m,
                                                    int                 n,
                                                    double*             A,
                                                    int                 lda,
                                                    int*                lworkOnDevice,
                                                    int*                lworkOnHost)
{
    *lworkOnHost = 0;
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDgetrf_bufferSize(handle, m, n, A, lda, lworkOnDevice);
    case API_FORTRAN:
        return hipsolverDgetrf_bufferSizeFortran(handle, m, n, A, lda, lworkOnDevice);
    case API_COMPAT:
        return hipsolverDnDgetrf_bufferSize(handle, m, n, A, lda, lworkOnDevice);
    default:
        *lworkOnDevice = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverDnParams_t params,
                                                    int                 m,
                                                    int                 n,
                                                    hipsolverComplex*   A,
                                                    int                 lda,
                                                    int*                lworkOnDevice,
                                                    int*                lworkOnHost)
{
    *lworkOnHost = 0;
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCgetrf_bufferSize(handle, m, n, (hipFloatComplex*)A, lda, lworkOnDevice);
    case API_FORTRAN:
        return hipsolverCgetrf_bufferSizeFortran(
            handle, m, n, (hipFloatComplex*)A, lda, lworkOnDevice);
    case API_COMPAT:
        return hipsolverDnCgetrf_bufferSize(handle, m, n, (hipFloatComplex*)A, lda, lworkOnDevice);
    default:
        *lworkOnDevice = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf_bufferSize(testAPI_t               API,
                                                    hipsolverHandle_t       handle,
                                                    hipsolverDnParams_t     params,
                                                    int                     m,
                                                    int                     n,
                                                    hipsolverDoubleComplex* A,
                                                    int                     lda,
                                                    int*                    lworkOnDevice,
                                                    int*                    lworkOnHost)
{
    *lworkOnHost = 0;
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZgetrf_bufferSize(handle, m, n, (hipDoubleComplex*)A, lda, lworkOnDevice);
    case API_FORTRAN:
        return hipsolverZgetrf_bufferSizeFortran(
            handle, m, n, (hipDoubleComplex*)A, lda, lworkOnDevice);
    case API_COMPAT:
        return hipsolverDnZgetrf_bufferSize(handle, m, n, (hipDoubleComplex*)A, lda, lworkOnDevice);
    default:
        *lworkOnDevice = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverDnParams_t params,
                                                    int64_t             m,
                                                    int64_t             n,
                                                    float*              A,
                                                    int64_t             lda,
                                                    size_t*             lworkOnDevice,
                                                    size_t*             lworkOnHost)
{
    switch(API)
    {
    case API_COMPAT:
        return hipsolverDnXgetrf_bufferSize(
            handle, params, m, n, HIP_R_32F, A, lda, HIP_R_32F, lworkOnDevice, lworkOnHost);
    default:
        *lworkOnDevice = 0;
        *lworkOnHost   = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverDnParams_t params,
                                                    int64_t             m,
                                                    int64_t             n,
                                                    double*             A,
                                                    int64_t             lda,
                                                    size_t*             lworkOnDevice,
                                                    size_t*             lworkOnHost)
{
    switch(API)
    {
    case API_COMPAT:
        return hipsolverDnXgetrf_bufferSize(
            handle, params, m, n, HIP_R_64F, A, lda, HIP_R_64F, lworkOnDevice, lworkOnHost);
    default:
        *lworkOnDevice = 0;
        *lworkOnHost   = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverDnParams_t params,
                                                    int64_t             m,
                                                    int64_t             n,
                                                    hipsolverComplex*   A,
                                                    int64_t             lda,
                                                    size_t*             lworkOnDevice,
                                                    size_t*             lworkOnHost)
{
    switch(API)
    {
    case API_COMPAT:
        return hipsolverDnXgetrf_bufferSize(
            handle, params, m, n, HIP_C_32F, A, lda, HIP_C_32F, lworkOnDevice, lworkOnHost);
    default:
        *lworkOnDevice = 0;
        *lworkOnHost   = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf_bufferSize(testAPI_t               API,
                                                    hipsolverHandle_t       handle,
                                                    hipsolverDnParams_t     params,
                                                    int64_t                 m,
                                                    int64_t                 n,
                                                    hipsolverDoubleComplex* A,
                                                    int64_t                 lda,
                                                    size_t*                 lworkOnDevice,
                                                    size_t*                 lworkOnHost)
{
    switch(API)
    {
    case API_COMPAT:
        return hipsolverDnXgetrf_bufferSize(
            handle, params, m, n, HIP_C_64F, A, lda, HIP_C_64F, lworkOnDevice, lworkOnHost);
    default:
        *lworkOnDevice = 0;
        *lworkOnHost   = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf(testAPI_t           API,
                                         bool                NPVT,
                                         hipsolverHandle_t   handle,
                                         hipsolverDnParams_t params,
                                         int                 m,
                                         int                 n,
                                         float*              A,
                                         int                 lda,
                                         int                 stA,
                                         int*                ipiv,
                                         int                 stP,
                                         float*              workOnDevice,
                                         int                 lworkOnDevice,
                                         float*              workOnHost,
                                         int                 lworkOnHost,
                                         int*                info,
                                         int                 bc)
{
    switch(api2marshal(API, NPVT))
    {
    case C_NORMAL:
        return hipsolverSgetrf(handle, m, n, A, lda, workOnDevice, lworkOnDevice, ipiv, info);
    case C_NORMAL_ALT:
        return hipsolverSgetrf(handle, m, n, A, lda, workOnDevice, lworkOnDevice, nullptr, info);
    case FORTRAN_NORMAL:
        return hipsolverSgetrfFortran(
            handle, m, n, A, lda, workOnDevice, lworkOnDevice, ipiv, info);
    case COMPAT_NORMAL:
        return hipsolverDnSgetrf(handle, m, n, A, lda, workOnDevice, ipiv, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf(testAPI_t           API,
                                         bool                NPVT,
                                         hipsolverHandle_t   handle,
                                         hipsolverDnParams_t params,
                                         int                 m,
                                         int                 n,
                                         double*             A,
                                         int                 lda,
                                         int                 stA,
                                         int*                ipiv,
                                         int                 stP,
                                         double*             workOnDevice,
                                         int                 lworkOnDevice,
                                         double*             workOnHost,
                                         int                 lworkOnHost,
                                         int*                info,
                                         int                 bc)
{
    switch(api2marshal(API, NPVT))
    {
    case C_NORMAL:
        return hipsolverDgetrf(handle, m, n, A, lda, workOnDevice, lworkOnDevice, ipiv, info);
    case C_NORMAL_ALT:
        return hipsolverDgetrf(handle, m, n, A, lda, workOnDevice, lworkOnDevice, nullptr, info);
    case FORTRAN_NORMAL:
        return hipsolverDgetrfFortran(
            handle, m, n, A, lda, workOnDevice, lworkOnDevice, ipiv, info);
    case COMPAT_NORMAL:
        return hipsolverDnDgetrf(handle, m, n, A, lda, workOnDevice, ipiv, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf(testAPI_t           API,
                                         bool                NPVT,
                                         hipsolverHandle_t   handle,
                                         hipsolverDnParams_t params,
                                         int                 m,
                                         int                 n,
                                         hipsolverComplex*   A,
                                         int                 lda,
                                         int                 stA,
                                         int*                ipiv,
                                         int                 stP,
                                         hipsolverComplex*   workOnDevice,
                                         int                 lworkOnDevice,
                                         hipsolverComplex*   workOnHost,
                                         int                 lworkOnHost,
                                         int*                info,
                                         int                 bc)
{
    switch(api2marshal(API, NPVT))
    {
    case C_NORMAL:
        return hipsolverCgetrf(handle,
                               m,
                               n,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)workOnDevice,
                               lworkOnDevice,
                               ipiv,
                               info);
    case C_NORMAL_ALT:
        return hipsolverCgetrf(handle,
                               m,
                               n,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)workOnDevice,
                               lworkOnDevice,
                               nullptr,
                               info);
    case FORTRAN_NORMAL:
        return hipsolverCgetrfFortran(handle,
                                      m,
                                      n,
                                      (hipFloatComplex*)A,
                                      lda,
                                      (hipFloatComplex*)workOnDevice,
                                      lworkOnDevice,
                                      ipiv,
                                      info);
    case COMPAT_NORMAL:
        return hipsolverDnCgetrf(
            handle, m, n, (hipFloatComplex*)A, lda, (hipFloatComplex*)workOnDevice, ipiv, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf(testAPI_t               API,
                                         bool                    NPVT,
                                         hipsolverHandle_t       handle,
                                         hipsolverDnParams_t     params,
                                         int                     m,
                                         int                     n,
                                         hipsolverDoubleComplex* A,
                                         int                     lda,
                                         int                     stA,
                                         int*                    ipiv,
                                         int                     stP,
                                         hipsolverDoubleComplex* workOnDevice,
                                         int                     lworkOnDevice,
                                         hipsolverDoubleComplex* workOnHost,
                                         int                     lworkOnHost,
                                         int*                    info,
                                         int                     bc)
{
    switch(api2marshal(API, NPVT))
    {
    case C_NORMAL:
        return hipsolverZgetrf(handle,
                               m,
                               n,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)workOnDevice,
                               lworkOnDevice,
                               ipiv,
                               info);
    case C_NORMAL_ALT:
        return hipsolverZgetrf(handle,
                               m,
                               n,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)workOnDevice,
                               lworkOnDevice,
                               nullptr,
                               info);
    case FORTRAN_NORMAL:
        return hipsolverZgetrfFortran(handle,
                                      m,
                                      n,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      (hipDoubleComplex*)workOnDevice,
                                      lworkOnDevice,
                                      ipiv,
                                      info);
    case COMPAT_NORMAL:
        return hipsolverDnZgetrf(
            handle, m, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)workOnDevice, ipiv, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf(testAPI_t           API,
                                         bool                NPVT,
                                         hipsolverHandle_t   handle,
                                         hipsolverDnParams_t params,
                                         int64_t             m,
                                         int64_t             n,
                                         float*              A,
                                         int64_t             lda,
                                         int64_t             stA,
                                         int64_t*            ipiv,
                                         int64_t             stP,
                                         float*              workOnDevice,
                                         size_t              lworkOnDevice,
                                         float*              workOnHost,
                                         size_t              lworkOnHost,
                                         int*                info,
                                         int                 bc)
{
    switch(api2marshal(API, NPVT))
    {
    case COMPAT_NORMAL:
        return hipsolverDnXgetrf(handle,
                                 params,
                                 m,
                                 n,
                                 HIP_R_32F,
                                 A,
                                 lda,
                                 ipiv,
                                 HIP_R_32F,
                                 workOnDevice,
                                 lworkOnDevice,
                                 workOnHost,
                                 lworkOnHost,
                                 info);
    case COMPAT_NORMAL_ALT:
        return hipsolverDnXgetrf(handle,
                                 params,
                                 m,
                                 n,
                                 HIP_R_32F,
                                 A,
                                 lda,
                                 nullptr,
                                 HIP_R_32F,
                                 workOnDevice,
                                 lworkOnDevice,
                                 workOnHost,
                                 lworkOnHost,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf(testAPI_t           API,
                                         bool                NPVT,
                                         hipsolverHandle_t   handle,
                                         hipsolverDnParams_t params,
                                         int64_t             m,
                                         int64_t             n,
                                         double*             A,
                                         int64_t             lda,
                                         int64_t             stA,
                                         int64_t*            ipiv,
                                         int64_t             stP,
                                         double*             workOnDevice,
                                         size_t              lworkOnDevice,
                                         double*             workOnHost,
                                         size_t              lworkOnHost,
                                         int*                info,
                                         int                 bc)
{
    switch(api2marshal(API, NPVT))
    {
    case COMPAT_NORMAL:
        return hipsolverDnXgetrf(handle,
                                 params,
                                 m,
                                 n,
                                 HIP_R_64F,
                                 A,
                                 lda,
                                 ipiv,
                                 HIP_R_64F,
                                 workOnDevice,
                                 lworkOnDevice,
                                 workOnHost,
                                 lworkOnHost,
                                 info);
    case COMPAT_NORMAL_ALT:
        return hipsolverDnXgetrf(handle,
                                 params,
                                 m,
                                 n,
                                 HIP_R_64F,
                                 A,
                                 lda,
                                 nullptr,
                                 HIP_R_64F,
                                 workOnDevice,
                                 lworkOnDevice,
                                 workOnHost,
                                 lworkOnHost,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf(testAPI_t           API,
                                         bool                NPVT,
                                         hipsolverHandle_t   handle,
                                         hipsolverDnParams_t params,
                                         int64_t             m,
                                         int64_t             n,
                                         hipsolverComplex*   A,
                                         int64_t             lda,
                                         int64_t             stA,
                                         int64_t*            ipiv,
                                         int64_t             stP,
                                         hipsolverComplex*   workOnDevice,
                                         size_t              lworkOnDevice,
                                         hipsolverComplex*   workOnHost,
                                         size_t              lworkOnHost,
                                         int*                info,
                                         int                 bc)
{
    switch(api2marshal(API, NPVT))
    {
    case COMPAT_NORMAL:
        return hipsolverDnXgetrf(handle,
                                 params,
                                 m,
                                 n,
                                 HIP_C_32F,
                                 A,
                                 lda,
                                 ipiv,
                                 HIP_C_32F,
                                 workOnDevice,
                                 lworkOnDevice,
                                 workOnHost,
                                 lworkOnHost,
                                 info);
    case COMPAT_NORMAL_ALT:
        return hipsolverDnXgetrf(handle,
                                 params,
                                 m,
                                 n,
                                 HIP_C_32F,
                                 A,
                                 lda,
                                 nullptr,
                                 HIP_C_32F,
                                 workOnDevice,
                                 lworkOnDevice,
                                 workOnHost,
                                 lworkOnHost,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrf(testAPI_t               API,
                                         bool                    NPVT,
                                         hipsolverHandle_t       handle,
                                         hipsolverDnParams_t     params,
                                         int64_t                 m,
                                         int64_t                 n,
                                         hipsolverDoubleComplex* A,
                                         int64_t                 lda,
                                         int64_t                 stA,
                                         int64_t*                ipiv,
                                         int64_t                 stP,
                                         hipsolverDoubleComplex* workOnDevice,
                                         size_t                  lworkOnDevice,
                                         hipsolverDoubleComplex* workOnHost,
                                         size_t                  lworkOnHost,
                                         int*                    info,
                                         int                     bc)
{
    switch(api2marshal(API, NPVT))
    {
    case COMPAT_NORMAL:
        return hipsolverDnXgetrf(handle,
                                 params,
                                 m,
                                 n,
                                 HIP_C_64F,
                                 A,
                                 lda,
                                 ipiv,
                                 HIP_C_64F,
                                 workOnDevice,
                                 lworkOnDevice,
                                 workOnHost,
                                 lworkOnHost,
                                 info);
    case COMPAT_NORMAL_ALT:
        return hipsolverDnXgetrf(handle,
                                 params,
                                 m,
                                 n,
                                 HIP_C_64F,
                                 A,
                                 lda,
                                 nullptr,
                                 HIP_C_64F,
                                 workOnDevice,
                                 lworkOnDevice,
                                 workOnHost,
                                 lworkOnHost,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** GETRS ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_getrs_bufferSize(testAPI_t            API,
                                                    hipsolverHandle_t    handle,
                                                    hipsolverDnParams_t  params,
                                                    hipsolverOperation_t trans,
                                                    int                  n,
                                                    int                  nrhs,
                                                    float*               A,
                                                    int                  lda,
                                                    int*                 ipiv,
                                                    float*               B,
                                                    int                  ldb,
                                                    int*                 lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSgetrs_bufferSize(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, lwork);
    case API_FORTRAN:
        return hipsolverSgetrs_bufferSizeFortran(
            handle, trans, n, nrhs, A, lda, ipiv, B, ldb, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrs_bufferSize(testAPI_t            API,
                                                    hipsolverHandle_t    handle,
                                                    hipsolverDnParams_t  params,
                                                    hipsolverOperation_t trans,
                                                    int                  n,
                                                    int                  nrhs,
                                                    double*              A,
                                                    int                  lda,
                                                    int*                 ipiv,
                                                    double*              B,
                                                    int                  ldb,
                                                    int*                 lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDgetrs_bufferSize(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, lwork);
    case API_FORTRAN:
        return hipsolverDgetrs_bufferSizeFortran(
            handle, trans, n, nrhs, A, lda, ipiv, B, ldb, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrs_bufferSize(testAPI_t            API,
                                                    hipsolverHandle_t    handle,
                                                    hipsolverDnParams_t  params,
                                                    hipsolverOperation_t trans,
                                                    int                  n,
                                                    int                  nrhs,
                                                    hipsolverComplex*    A,
                                                    int                  lda,
                                                    int*                 ipiv,
                                                    hipsolverComplex*    B,
                                                    int                  ldb,
                                                    int*                 lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCgetrs_bufferSize(handle,
                                          trans,
                                          n,
                                          nrhs,
                                          (hipFloatComplex*)A,
                                          lda,
                                          ipiv,
                                          (hipFloatComplex*)B,
                                          ldb,
                                          lwork);
    case API_FORTRAN:
        return hipsolverCgetrs_bufferSizeFortran(handle,
                                                 trans,
                                                 n,
                                                 nrhs,
                                                 (hipFloatComplex*)A,
                                                 lda,
                                                 ipiv,
                                                 (hipFloatComplex*)B,
                                                 ldb,
                                                 lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrs_bufferSize(testAPI_t               API,
                                                    hipsolverHandle_t       handle,
                                                    hipsolverDnParams_t     params,
                                                    hipsolverOperation_t    trans,
                                                    int                     n,
                                                    int                     nrhs,
                                                    hipsolverDoubleComplex* A,
                                                    int                     lda,
                                                    int*                    ipiv,
                                                    hipsolverDoubleComplex* B,
                                                    int                     ldb,
                                                    int*                    lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZgetrs_bufferSize(handle,
                                          trans,
                                          n,
                                          nrhs,
                                          (hipDoubleComplex*)A,
                                          lda,
                                          ipiv,
                                          (hipDoubleComplex*)B,
                                          ldb,
                                          lwork);
    case API_FORTRAN:
        return hipsolverZgetrs_bufferSizeFortran(handle,
                                                 trans,
                                                 n,
                                                 nrhs,
                                                 (hipDoubleComplex*)A,
                                                 lda,
                                                 ipiv,
                                                 (hipDoubleComplex*)B,
                                                 ldb,
                                                 lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrs_bufferSize(testAPI_t            API,
                                                    hipsolverHandle_t    handle,
                                                    hipsolverDnParams_t  params,
                                                    hipsolverOperation_t trans,
                                                    int64_t              n,
                                                    int64_t              nrhs,
                                                    float*               A,
                                                    int64_t              lda,
                                                    int64_t*             ipiv,
                                                    float*               B,
                                                    int64_t              ldb,
                                                    size_t*              lwork)
{
    switch(API)
    {
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrs_bufferSize(testAPI_t            API,
                                                    hipsolverHandle_t    handle,
                                                    hipsolverDnParams_t  params,
                                                    hipsolverOperation_t trans,
                                                    int64_t              n,
                                                    int64_t              nrhs,
                                                    double*              A,
                                                    int64_t              lda,
                                                    int64_t*             ipiv,
                                                    double*              B,
                                                    int64_t              ldb,
                                                    size_t*              lwork)
{
    switch(API)
    {
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrs_bufferSize(testAPI_t            API,
                                                    hipsolverHandle_t    handle,
                                                    hipsolverDnParams_t  params,
                                                    hipsolverOperation_t trans,
                                                    int64_t              n,
                                                    int64_t              nrhs,
                                                    hipsolverComplex*    A,
                                                    int64_t              lda,
                                                    int64_t*             ipiv,
                                                    hipsolverComplex*    B,
                                                    int64_t              ldb,
                                                    size_t*              lwork)
{
    switch(API)
    {
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrs_bufferSize(testAPI_t               API,
                                                    hipsolverHandle_t       handle,
                                                    hipsolverDnParams_t     params,
                                                    hipsolverOperation_t    trans,
                                                    int64_t                 n,
                                                    int64_t                 nrhs,
                                                    hipsolverDoubleComplex* A,
                                                    int64_t                 lda,
                                                    int64_t*                ipiv,
                                                    hipsolverDoubleComplex* B,
                                                    int64_t                 ldb,
                                                    size_t*                 lwork)
{
    switch(API)
    {
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrs(testAPI_t            API,
                                         hipsolverHandle_t    handle,
                                         hipsolverDnParams_t  params,
                                         hipsolverOperation_t trans,
                                         int                  n,
                                         int                  nrhs,
                                         float*               A,
                                         int                  lda,
                                         int                  stA,
                                         int*                 ipiv,
                                         int                  stP,
                                         float*               B,
                                         int                  ldb,
                                         int                  stB,
                                         float*               work,
                                         int                  lwork,
                                         int*                 info,
                                         int                  bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info);
    case API_FORTRAN:
        return hipsolverSgetrsFortran(
            handle, trans, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnSgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrs(testAPI_t            API,
                                         hipsolverHandle_t    handle,
                                         hipsolverDnParams_t  params,
                                         hipsolverOperation_t trans,
                                         int                  n,
                                         int                  nrhs,
                                         double*              A,
                                         int                  lda,
                                         int                  stA,
                                         int*                 ipiv,
                                         int                  stP,
                                         double*              B,
                                         int                  ldb,
                                         int                  stB,
                                         double*              work,
                                         int                  lwork,
                                         int*                 info,
                                         int                  bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info);
    case API_FORTRAN:
        return hipsolverDgetrsFortran(
            handle, trans, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnDgetrs(handle, trans, n, nrhs, A, lda, ipiv, B, ldb, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrs(testAPI_t            API,
                                         hipsolverHandle_t    handle,
                                         hipsolverDnParams_t  params,
                                         hipsolverOperation_t trans,
                                         int                  n,
                                         int                  nrhs,
                                         hipsolverComplex*    A,
                                         int                  lda,
                                         int                  stA,
                                         int*                 ipiv,
                                         int                  stP,
                                         hipsolverComplex*    B,
                                         int                  ldb,
                                         int                  stB,
                                         hipsolverComplex*    work,
                                         int                  lwork,
                                         int*                 info,
                                         int                  bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCgetrs(handle,
                               trans,
                               n,
                               nrhs,
                               (hipFloatComplex*)A,
                               lda,
                               ipiv,
                               (hipFloatComplex*)B,
                               ldb,
                               (hipFloatComplex*)work,
                               lwork,
                               info);
    case API_FORTRAN:
        return hipsolverCgetrsFortran(handle,
                                      trans,
                                      n,
                                      nrhs,
                                      (hipFloatComplex*)A,
                                      lda,
                                      ipiv,
                                      (hipFloatComplex*)B,
                                      ldb,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info);
    case API_COMPAT:
        return hipsolverDnCgetrs(
            handle, trans, n, nrhs, (hipFloatComplex*)A, lda, ipiv, (hipFloatComplex*)B, ldb, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrs(testAPI_t               API,
                                         hipsolverHandle_t       handle,
                                         hipsolverDnParams_t     params,
                                         hipsolverOperation_t    trans,
                                         int                     n,
                                         int                     nrhs,
                                         hipsolverDoubleComplex* A,
                                         int                     lda,
                                         int                     stA,
                                         int*                    ipiv,
                                         int                     stP,
                                         hipsolverDoubleComplex* B,
                                         int                     ldb,
                                         int                     stB,
                                         hipsolverDoubleComplex* work,
                                         int                     lwork,
                                         int*                    info,
                                         int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZgetrs(handle,
                               trans,
                               n,
                               nrhs,
                               (hipDoubleComplex*)A,
                               lda,
                               ipiv,
                               (hipDoubleComplex*)B,
                               ldb,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);
    case API_FORTRAN:
        return hipsolverZgetrsFortran(handle,
                                      trans,
                                      n,
                                      nrhs,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      ipiv,
                                      (hipDoubleComplex*)B,
                                      ldb,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      info);
    case API_COMPAT:
        return hipsolverDnZgetrs(handle,
                                 trans,
                                 n,
                                 nrhs,
                                 (hipDoubleComplex*)A,
                                 lda,
                                 ipiv,
                                 (hipDoubleComplex*)B,
                                 ldb,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrs(testAPI_t            API,
                                         hipsolverHandle_t    handle,
                                         hipsolverDnParams_t  params,
                                         hipsolverOperation_t trans,
                                         int64_t              n,
                                         int64_t              nrhs,
                                         float*               A,
                                         int64_t              lda,
                                         int64_t              stA,
                                         int64_t*             ipiv,
                                         int64_t              stP,
                                         float*               B,
                                         int64_t              ldb,
                                         int64_t              stB,
                                         float*               work,
                                         size_t               lwork,
                                         int*                 info,
                                         int                  bc)
{
    switch(API)
    {
    case API_COMPAT:
        return hipsolverDnXgetrs(
            handle, params, trans, n, nrhs, HIP_R_32F, A, lda, ipiv, HIP_R_32F, B, ldb, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrs(testAPI_t            API,
                                         hipsolverHandle_t    handle,
                                         hipsolverDnParams_t  params,
                                         hipsolverOperation_t trans,
                                         int64_t              n,
                                         int64_t              nrhs,
                                         double*              A,
                                         int64_t              lda,
                                         int64_t              stA,
                                         int64_t*             ipiv,
                                         int64_t              stP,
                                         double*              B,
                                         int64_t              ldb,
                                         int64_t              stB,
                                         double*              work,
                                         size_t               lwork,
                                         int*                 info,
                                         int                  bc)
{
    switch(API)
    {
    case API_COMPAT:
        return hipsolverDnXgetrs(
            handle, params, trans, n, nrhs, HIP_R_64F, A, lda, ipiv, HIP_R_64F, B, ldb, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrs(testAPI_t            API,
                                         hipsolverHandle_t    handle,
                                         hipsolverDnParams_t  params,
                                         hipsolverOperation_t trans,
                                         int64_t              n,
                                         int64_t              nrhs,
                                         hipsolverComplex*    A,
                                         int64_t              lda,
                                         int64_t              stA,
                                         int64_t*             ipiv,
                                         int64_t              stP,
                                         hipsolverComplex*    B,
                                         int64_t              ldb,
                                         int64_t              stB,
                                         hipsolverComplex*    work,
                                         size_t               lwork,
                                         int*                 info,
                                         int                  bc)
{
    switch(API)
    {
    case API_COMPAT:
        return hipsolverDnXgetrs(
            handle, params, trans, n, nrhs, HIP_C_32F, A, lda, ipiv, HIP_C_32F, B, ldb, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_getrs(testAPI_t               API,
                                         hipsolverHandle_t       handle,
                                         hipsolverDnParams_t     params,
                                         hipsolverOperation_t    trans,
                                         int64_t                 n,
                                         int64_t                 nrhs,
                                         hipsolverDoubleComplex* A,
                                         int64_t                 lda,
                                         int64_t                 stA,
                                         int64_t*                ipiv,
                                         int64_t                 stP,
                                         hipsolverDoubleComplex* B,
                                         int64_t                 ldb,
                                         int64_t                 stB,
                                         hipsolverDoubleComplex* work,
                                         size_t                  lwork,
                                         int*                    info,
                                         int                     bc)
{
    switch(API)
    {
    case API_COMPAT:
        return hipsolverDnXgetrs(
            handle, params, trans, n, nrhs, HIP_C_64F, A, lda, ipiv, HIP_C_64F, B, ldb, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** POTRF ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_potrf_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    float*              A,
                                                    int                 lda,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSpotrf_bufferSize(handle, uplo, n, A, lda, lwork);
    case API_FORTRAN:
        return hipsolverSpotrf_bufferSizeFortran(handle, uplo, n, A, lda, lwork);
    case API_COMPAT:
        return hipsolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    double*             A,
                                                    int                 lda,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDpotrf_bufferSize(handle, uplo, n, A, lda, lwork);
    case API_FORTRAN:
        return hipsolverDpotrf_bufferSizeFortran(handle, uplo, n, A, lda, lwork);
    case API_COMPAT:
        return hipsolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    hipsolverComplex*   A,
                                                    int                 lda,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCpotrf_bufferSize(handle, uplo, n, (hipFloatComplex*)A, lda, lwork);
    case API_FORTRAN:
        return hipsolverCpotrf_bufferSizeFortran(handle, uplo, n, (hipFloatComplex*)A, lda, lwork);
    case API_COMPAT:
        return hipsolverDnCpotrf_bufferSize(handle, uplo, n, (hipFloatComplex*)A, lda, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf_bufferSize(testAPI_t               API,
                                                    hipsolverHandle_t       handle,
                                                    hipsolverFillMode_t     uplo,
                                                    int                     n,
                                                    hipsolverDoubleComplex* A,
                                                    int                     lda,
                                                    int*                    lwork,
                                                    int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZpotrf_bufferSize(handle, uplo, n, (hipDoubleComplex*)A, lda, lwork);
    case API_FORTRAN:
        return hipsolverZpotrf_bufferSizeFortran(handle, uplo, n, (hipDoubleComplex*)A, lda, lwork);
    case API_COMPAT:
        return hipsolverDnZpotrf_bufferSize(handle, uplo, n, (hipDoubleComplex*)A, lda, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         float*              A,
                                         int                 lda,
                                         int                 stA,
                                         float*              work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSpotrf(handle, uplo, n, A, lda, work, lwork, info);
    case API_FORTRAN:
        return hipsolverSpotrfFortran(handle, uplo, n, A, lda, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnSpotrf(handle, uplo, n, A, lda, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         double*             A,
                                         int                 lda,
                                         int                 stA,
                                         double*             work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDpotrf(handle, uplo, n, A, lda, work, lwork, info);
    case API_FORTRAN:
        return hipsolverDpotrfFortran(handle, uplo, n, A, lda, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnDpotrf(handle, uplo, n, A, lda, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         hipsolverComplex*   A,
                                         int                 lda,
                                         int                 stA,
                                         hipsolverComplex*   work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCpotrf(
            handle, uplo, n, (hipFloatComplex*)A, lda, (hipFloatComplex*)work, lwork, info);
    case API_FORTRAN:
        return hipsolverCpotrfFortran(
            handle, uplo, n, (hipFloatComplex*)A, lda, (hipFloatComplex*)work, lwork, info);
    case API_COMPAT:
        return hipsolverDnCpotrf(
            handle, uplo, n, (hipFloatComplex*)A, lda, (hipFloatComplex*)work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf(testAPI_t               API,
                                         hipsolverHandle_t       handle,
                                         hipsolverFillMode_t     uplo,
                                         int                     n,
                                         hipsolverDoubleComplex* A,
                                         int                     lda,
                                         int                     stA,
                                         hipsolverDoubleComplex* work,
                                         int                     lwork,
                                         int*                    info,
                                         int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZpotrf(
            handle, uplo, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)work, lwork, info);
    case API_FORTRAN:
        return hipsolverZpotrfFortran(
            handle, uplo, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)work, lwork, info);
    case API_COMPAT:
        return hipsolverDnZpotrf(
            handle, uplo, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

// batched
inline hipsolverStatus_t hipsolver_potrf_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    float*              A[],
                                                    int                 lda,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSpotrfBatched_bufferSize(handle, uplo, n, A, lda, lwork, bc);
    case API_FORTRAN:
        return hipsolverSpotrfBatched_bufferSizeFortran(handle, uplo, n, A, lda, lwork, bc);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    double*             A[],
                                                    int                 lda,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDpotrfBatched_bufferSize(handle, uplo, n, A, lda, lwork, bc);
    case API_FORTRAN:
        return hipsolverDpotrfBatched_bufferSizeFortran(handle, uplo, n, A, lda, lwork, bc);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    hipsolverComplex*   A[],
                                                    int                 lda,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCpotrfBatched_bufferSize(
            handle, uplo, n, (hipFloatComplex**)A, lda, lwork, bc);
    case API_FORTRAN:
        return hipsolverCpotrfBatched_bufferSizeFortran(
            handle, uplo, n, (hipFloatComplex**)A, lda, lwork, bc);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf_bufferSize(testAPI_t               API,
                                                    hipsolverHandle_t       handle,
                                                    hipsolverFillMode_t     uplo,
                                                    int                     n,
                                                    hipsolverDoubleComplex* A[],
                                                    int                     lda,
                                                    int*                    lwork,
                                                    int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZpotrfBatched_bufferSize(
            handle, uplo, n, (hipDoubleComplex**)A, lda, lwork, bc);
    case API_FORTRAN:
        return hipsolverZpotrfBatched_bufferSizeFortran(
            handle, uplo, n, (hipDoubleComplex**)A, lda, lwork, bc);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         float*              A[],
                                         int                 lda,
                                         int                 stA,
                                         float*              work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSpotrfBatched(handle, uplo, n, A, lda, work, lwork, info, bc);
    case API_FORTRAN:
        return hipsolverSpotrfBatchedFortran(handle, uplo, n, A, lda, work, lwork, info, bc);
    case API_COMPAT:
        return hipsolverDnSpotrfBatched(handle, uplo, n, A, lda, info, bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         double*             A[],
                                         int                 lda,
                                         int                 stA,
                                         double*             work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDpotrfBatched(handle, uplo, n, A, lda, work, lwork, info, bc);
    case API_FORTRAN:
        return hipsolverDpotrfBatchedFortran(handle, uplo, n, A, lda, work, lwork, info, bc);
    case API_COMPAT:
        return hipsolverDnDpotrfBatched(handle, uplo, n, A, lda, info, bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         hipsolverComplex*   A[],
                                         int                 lda,
                                         int                 stA,
                                         hipsolverComplex*   work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCpotrfBatched(
            handle, uplo, n, (hipFloatComplex**)A, lda, (hipFloatComplex*)work, lwork, info, bc);
    case API_FORTRAN:
        return hipsolverCpotrfBatchedFortran(
            handle, uplo, n, (hipFloatComplex**)A, lda, (hipFloatComplex*)work, lwork, info, bc);
    case API_COMPAT:
        return hipsolverDnCpotrfBatched(handle, uplo, n, (hipFloatComplex**)A, lda, info, bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrf(testAPI_t               API,
                                         hipsolverHandle_t       handle,
                                         hipsolverFillMode_t     uplo,
                                         int                     n,
                                         hipsolverDoubleComplex* A[],
                                         int                     lda,
                                         int                     stA,
                                         hipsolverDoubleComplex* work,
                                         int                     lwork,
                                         int*                    info,
                                         int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZpotrfBatched(
            handle, uplo, n, (hipDoubleComplex**)A, lda, (hipDoubleComplex*)work, lwork, info, bc);
    case API_FORTRAN:
        return hipsolverZpotrfBatchedFortran(
            handle, uplo, n, (hipDoubleComplex**)A, lda, (hipDoubleComplex*)work, lwork, info, bc);
    case API_COMPAT:
        return hipsolverDnZpotrfBatched(handle, uplo, n, (hipDoubleComplex**)A, lda, info, bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** POTRI ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_potri_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    float*              A,
                                                    int                 lda,
                                                    int*                lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSpotri_bufferSize(handle, uplo, n, A, lda, lwork);
    case API_FORTRAN:
        return hipsolverSpotri_bufferSizeFortran(handle, uplo, n, A, lda, lwork);
    case API_COMPAT:
        return hipsolverDnSpotri_bufferSize(handle, uplo, n, A, lda, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potri_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    double*             A,
                                                    int                 lda,
                                                    int*                lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDpotri_bufferSize(handle, uplo, n, A, lda, lwork);
    case API_FORTRAN:
        return hipsolverDpotri_bufferSizeFortran(handle, uplo, n, A, lda, lwork);
    case API_COMPAT:
        return hipsolverDnDpotri_bufferSize(handle, uplo, n, A, lda, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potri_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    hipsolverComplex*   A,
                                                    int                 lda,
                                                    int*                lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCpotri_bufferSize(handle, uplo, n, (hipFloatComplex*)A, lda, lwork);
    case API_FORTRAN:
        return hipsolverCpotri_bufferSizeFortran(handle, uplo, n, (hipFloatComplex*)A, lda, lwork);
    case API_COMPAT:
        return hipsolverDnCpotri_bufferSize(handle, uplo, n, (hipFloatComplex*)A, lda, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potri_bufferSize(testAPI_t               API,
                                                    hipsolverHandle_t       handle,
                                                    hipsolverFillMode_t     uplo,
                                                    int                     n,
                                                    hipsolverDoubleComplex* A,
                                                    int                     lda,
                                                    int*                    lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZpotri_bufferSize(handle, uplo, n, (hipDoubleComplex*)A, lda, lwork);
    case API_FORTRAN:
        return hipsolverZpotri_bufferSizeFortran(handle, uplo, n, (hipDoubleComplex*)A, lda, lwork);
    case API_COMPAT:
        return hipsolverDnZpotri_bufferSize(handle, uplo, n, (hipDoubleComplex*)A, lda, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potri(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         float*              A,
                                         int                 lda,
                                         int                 stA,
                                         float*              work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSpotri(handle, uplo, n, A, lda, work, lwork, info);
    case API_FORTRAN:
        return hipsolverSpotriFortran(handle, uplo, n, A, lda, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnSpotri(handle, uplo, n, A, lda, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potri(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         double*             A,
                                         int                 lda,
                                         int                 stA,
                                         double*             work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDpotri(handle, uplo, n, A, lda, work, lwork, info);
    case API_FORTRAN:
        return hipsolverDpotriFortran(handle, uplo, n, A, lda, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnDpotri(handle, uplo, n, A, lda, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potri(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         hipsolverComplex*   A,
                                         int                 lda,
                                         int                 stA,
                                         hipsolverComplex*   work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCpotri(
            handle, uplo, n, (hipFloatComplex*)A, lda, (hipFloatComplex*)work, lwork, info);
    case API_FORTRAN:
        return hipsolverCpotriFortran(
            handle, uplo, n, (hipFloatComplex*)A, lda, (hipFloatComplex*)work, lwork, info);
    case API_COMPAT:
        return hipsolverDnCpotri(
            handle, uplo, n, (hipFloatComplex*)A, lda, (hipFloatComplex*)work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potri(testAPI_t               API,
                                         hipsolverHandle_t       handle,
                                         hipsolverFillMode_t     uplo,
                                         int                     n,
                                         hipsolverDoubleComplex* A,
                                         int                     lda,
                                         int                     stA,
                                         hipsolverDoubleComplex* work,
                                         int                     lwork,
                                         int*                    info,
                                         int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZpotri(
            handle, uplo, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)work, lwork, info);
    case API_FORTRAN:
        return hipsolverZpotriFortran(
            handle, uplo, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)work, lwork, info);
    case API_COMPAT:
        return hipsolverDnZpotri(
            handle, uplo, n, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** POTRS ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_potrs_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    int                 nrhs,
                                                    float*              A,
                                                    int                 lda,
                                                    float*              B,
                                                    int                 ldb,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSpotrs_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork);
    case API_FORTRAN:
        return hipsolverSpotrs_bufferSizeFortran(handle, uplo, n, nrhs, A, lda, B, ldb, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    int                 nrhs,
                                                    double*             A,
                                                    int                 lda,
                                                    double*             B,
                                                    int                 ldb,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDpotrs_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork);
    case API_FORTRAN:
        return hipsolverDpotrs_bufferSizeFortran(handle, uplo, n, nrhs, A, lda, B, ldb, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    int                 nrhs,
                                                    hipsolverComplex*   A,
                                                    int                 lda,
                                                    hipsolverComplex*   B,
                                                    int                 ldb,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCpotrs_bufferSize(
            handle, uplo, n, nrhs, (hipFloatComplex*)A, lda, (hipFloatComplex*)B, ldb, lwork);
    case API_FORTRAN:
        return hipsolverCpotrs_bufferSizeFortran(
            handle, uplo, n, nrhs, (hipFloatComplex*)A, lda, (hipFloatComplex*)B, ldb, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs_bufferSize(testAPI_t               API,
                                                    hipsolverHandle_t       handle,
                                                    hipsolverFillMode_t     uplo,
                                                    int                     n,
                                                    int                     nrhs,
                                                    hipsolverDoubleComplex* A,
                                                    int                     lda,
                                                    hipsolverDoubleComplex* B,
                                                    int                     ldb,
                                                    int*                    lwork,
                                                    int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZpotrs_bufferSize(
            handle, uplo, n, nrhs, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)B, ldb, lwork);
    case API_FORTRAN:
        return hipsolverZpotrs_bufferSizeFortran(
            handle, uplo, n, nrhs, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)B, ldb, lwork);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         int                 nrhs,
                                         float*              A,
                                         int                 lda,
                                         int                 stA,
                                         float*              B,
                                         int                 ldb,
                                         int                 stB,
                                         float*              work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info);
    case API_FORTRAN:
        return hipsolverSpotrsFortran(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         int                 nrhs,
                                         double*             A,
                                         int                 lda,
                                         int                 stA,
                                         double*             B,
                                         int                 ldb,
                                         int                 stB,
                                         double*             work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info);
    case API_FORTRAN:
        return hipsolverDpotrsFortran(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         int                 nrhs,
                                         hipsolverComplex*   A,
                                         int                 lda,
                                         int                 stA,
                                         hipsolverComplex*   B,
                                         int                 ldb,
                                         int                 stB,
                                         hipsolverComplex*   work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCpotrs(handle,
                               uplo,
                               n,
                               nrhs,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)B,
                               ldb,
                               (hipFloatComplex*)work,
                               lwork,
                               info);
    case API_FORTRAN:
        return hipsolverCpotrsFortran(handle,
                                      uplo,
                                      n,
                                      nrhs,
                                      (hipFloatComplex*)A,
                                      lda,
                                      (hipFloatComplex*)B,
                                      ldb,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info);
    case API_COMPAT:
        return hipsolverDnCpotrs(
            handle, uplo, n, nrhs, (hipFloatComplex*)A, lda, (hipFloatComplex*)B, ldb, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs(testAPI_t               API,
                                         hipsolverHandle_t       handle,
                                         hipsolverFillMode_t     uplo,
                                         int                     n,
                                         int                     nrhs,
                                         hipsolverDoubleComplex* A,
                                         int                     lda,
                                         int                     stA,
                                         hipsolverDoubleComplex* B,
                                         int                     ldb,
                                         int                     stB,
                                         hipsolverDoubleComplex* work,
                                         int                     lwork,
                                         int*                    info,
                                         int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZpotrs(handle,
                               uplo,
                               n,
                               nrhs,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)B,
                               ldb,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);
    case API_FORTRAN:
        return hipsolverZpotrsFortran(handle,
                                      uplo,
                                      n,
                                      nrhs,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      (hipDoubleComplex*)B,
                                      ldb,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      info);
    case API_COMPAT:
        return hipsolverDnZpotrs(
            handle, uplo, n, nrhs, (hipDoubleComplex*)A, lda, (hipDoubleComplex*)B, ldb, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

// batched
inline hipsolverStatus_t hipsolver_potrs_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    int                 nrhs,
                                                    float*              A[],
                                                    int                 lda,
                                                    float*              B[],
                                                    int                 ldb,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSpotrsBatched_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork, bc);
    case API_FORTRAN:
        return hipsolverSpotrsBatched_bufferSizeFortran(
            handle, uplo, n, nrhs, A, lda, B, ldb, lwork, bc);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    int                 nrhs,
                                                    double*             A[],
                                                    int                 lda,
                                                    double*             B[],
                                                    int                 ldb,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDpotrsBatched_bufferSize(handle, uplo, n, nrhs, A, lda, B, ldb, lwork, bc);
    case API_FORTRAN:
        return hipsolverDpotrsBatched_bufferSizeFortran(
            handle, uplo, n, nrhs, A, lda, B, ldb, lwork, bc);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs_bufferSize(testAPI_t           API,
                                                    hipsolverHandle_t   handle,
                                                    hipsolverFillMode_t uplo,
                                                    int                 n,
                                                    int                 nrhs,
                                                    hipsolverComplex*   A[],
                                                    int                 lda,
                                                    hipsolverComplex*   B[],
                                                    int                 ldb,
                                                    int*                lwork,
                                                    int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCpotrsBatched_bufferSize(
            handle, uplo, n, nrhs, (hipFloatComplex**)A, lda, (hipFloatComplex**)B, ldb, lwork, bc);
    case API_FORTRAN:
        return hipsolverCpotrsBatched_bufferSizeFortran(
            handle, uplo, n, nrhs, (hipFloatComplex**)A, lda, (hipFloatComplex**)B, ldb, lwork, bc);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs_bufferSize(testAPI_t               API,
                                                    hipsolverHandle_t       handle,
                                                    hipsolverFillMode_t     uplo,
                                                    int                     n,
                                                    int                     nrhs,
                                                    hipsolverDoubleComplex* A[],
                                                    int                     lda,
                                                    hipsolverDoubleComplex* B[],
                                                    int                     ldb,
                                                    int*                    lwork,
                                                    int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZpotrsBatched_bufferSize(handle,
                                                 uplo,
                                                 n,
                                                 nrhs,
                                                 (hipDoubleComplex**)A,
                                                 lda,
                                                 (hipDoubleComplex**)B,
                                                 ldb,
                                                 lwork,
                                                 bc);
    case API_FORTRAN:
        return hipsolverZpotrsBatched_bufferSizeFortran(handle,
                                                        uplo,
                                                        n,
                                                        nrhs,
                                                        (hipDoubleComplex**)A,
                                                        lda,
                                                        (hipDoubleComplex**)B,
                                                        ldb,
                                                        lwork,
                                                        bc);
    default:
        *lwork = 0;
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         int                 nrhs,
                                         float*              A[],
                                         int                 lda,
                                         int                 stA,
                                         float*              B[],
                                         int                 ldb,
                                         int                 stB,
                                         float*              work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info, bc);
    case API_FORTRAN:
        return hipsolverSpotrsBatchedFortran(
            handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info, bc);
    case API_COMPAT:
        return hipsolverDnSpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, info, bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         int                 nrhs,
                                         double*             A[],
                                         int                 lda,
                                         int                 stA,
                                         double*             B[],
                                         int                 ldb,
                                         int                 stB,
                                         double*             work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info, bc);
    case API_FORTRAN:
        return hipsolverDpotrsBatchedFortran(
            handle, uplo, n, nrhs, A, lda, B, ldb, work, lwork, info, bc);
    case API_COMPAT:
        return hipsolverDnDpotrsBatched(handle, uplo, n, nrhs, A, lda, B, ldb, info, bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         int                 nrhs,
                                         hipsolverComplex*   A[],
                                         int                 lda,
                                         int                 stA,
                                         hipsolverComplex*   B[],
                                         int                 ldb,
                                         int                 stB,
                                         hipsolverComplex*   work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCpotrsBatched(handle,
                                      uplo,
                                      n,
                                      nrhs,
                                      (hipFloatComplex**)A,
                                      lda,
                                      (hipFloatComplex**)B,
                                      ldb,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info,
                                      bc);
    case API_FORTRAN:
        return hipsolverCpotrsBatchedFortran(handle,
                                             uplo,
                                             n,
                                             nrhs,
                                             (hipFloatComplex**)A,
                                             lda,
                                             (hipFloatComplex**)B,
                                             ldb,
                                             (hipFloatComplex*)work,
                                             lwork,
                                             info,
                                             bc);
    case API_COMPAT:
        return hipsolverDnCpotrsBatched(
            handle, uplo, n, nrhs, (hipFloatComplex**)A, lda, (hipFloatComplex**)B, ldb, info, bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_potrs(testAPI_t               API,
                                         hipsolverHandle_t       handle,
                                         hipsolverFillMode_t     uplo,
                                         int                     n,
                                         int                     nrhs,
                                         hipsolverDoubleComplex* A[],
                                         int                     lda,
                                         int                     stA,
                                         hipsolverDoubleComplex* B[],
                                         int                     ldb,
                                         int                     stB,
                                         hipsolverDoubleComplex* work,
                                         int                     lwork,
                                         int*                    info,
                                         int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZpotrsBatched(handle,
                                      uplo,
                                      n,
                                      nrhs,
                                      (hipDoubleComplex**)A,
                                      lda,
                                      (hipDoubleComplex**)B,
                                      ldb,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      info,
                                      bc);
    case API_FORTRAN:
        return hipsolverZpotrsBatchedFortran(handle,
                                             uplo,
                                             n,
                                             nrhs,
                                             (hipDoubleComplex**)A,
                                             lda,
                                             (hipDoubleComplex**)B,
                                             ldb,
                                             (hipDoubleComplex*)work,
                                             lwork,
                                             info,
                                             bc);
    case API_COMPAT:
        return hipsolverDnZpotrsBatched(handle,
                                        uplo,
                                        n,
                                        nrhs,
                                        (hipDoubleComplex**)A,
                                        lda,
                                        (hipDoubleComplex**)B,
                                        ldb,
                                        info,
                                        bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** SYEVD/HEEVD ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_syevd_heevd_bufferSize(testAPI_t           API,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverEigMode_t  jobz,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          float*              A,
                                                          int                 lda,
                                                          float*              W,
                                                          int*                lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork);
    case API_FORTRAN:
        return hipsolverSsyevd_bufferSizeFortran(handle, jobz, uplo, n, A, lda, W, lwork);
    case API_COMPAT:
        return hipsolverDnSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevd_heevd_bufferSize(testAPI_t           API,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverEigMode_t  jobz,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          double*             A,
                                                          int                 lda,
                                                          double*             W,
                                                          int*                lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork);
    case API_FORTRAN:
        return hipsolverDsyevd_bufferSizeFortran(handle, jobz, uplo, n, A, lda, W, lwork);
    case API_COMPAT:
        return hipsolverDnDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevd_heevd_bufferSize(testAPI_t           API,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverEigMode_t  jobz,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipsolverComplex*   A,
                                                          int                 lda,
                                                          float*              W,
                                                          int*                lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCheevd_bufferSize(
            handle, jobz, uplo, n, (hipFloatComplex*)A, lda, W, lwork);
    case API_FORTRAN:
        return hipsolverCheevd_bufferSizeFortran(
            handle, jobz, uplo, n, (hipFloatComplex*)A, lda, W, lwork);
    case API_COMPAT:
        return hipsolverDnCheevd_bufferSize(
            handle, jobz, uplo, n, (hipFloatComplex*)A, lda, W, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevd_heevd_bufferSize(testAPI_t               API,
                                                          hipsolverHandle_t       handle,
                                                          hipsolverEigMode_t      jobz,
                                                          hipsolverFillMode_t     uplo,
                                                          int                     n,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          double*                 W,
                                                          int*                    lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZheevd_bufferSize(
            handle, jobz, uplo, n, (hipDoubleComplex*)A, lda, W, lwork);
    case API_FORTRAN:
        return hipsolverZheevd_bufferSizeFortran(
            handle, jobz, uplo, n, (hipDoubleComplex*)A, lda, W, lwork);
    case API_COMPAT:
        return hipsolverDnZheevd_bufferSize(
            handle, jobz, uplo, n, (hipDoubleComplex*)A, lda, W, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevd_heevd(testAPI_t           API,
                                               hipsolverHandle_t   handle,
                                               hipsolverEigMode_t  jobz,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               float*              A,
                                               int                 lda,
                                               int                 stA,
                                               float*              W,
                                               int                 stW,
                                               float*              work,
                                               int                 lwork,
                                               int*                info,
                                               int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info);
    case API_FORTRAN:
        return hipsolverSsyevdFortran(handle, jobz, uplo, n, A, lda, W, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevd_heevd(testAPI_t           API,
                                               hipsolverHandle_t   handle,
                                               hipsolverEigMode_t  jobz,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               double*             A,
                                               int                 lda,
                                               int                 stA,
                                               double*             W,
                                               int                 stW,
                                               double*             work,
                                               int                 lwork,
                                               int*                info,
                                               int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info);
    case API_FORTRAN:
        return hipsolverDsyevdFortran(handle, jobz, uplo, n, A, lda, W, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevd_heevd(testAPI_t           API,
                                               hipsolverHandle_t   handle,
                                               hipsolverEigMode_t  jobz,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               hipsolverComplex*   A,
                                               int                 lda,
                                               int                 stA,
                                               float*              W,
                                               int                 stW,
                                               hipsolverComplex*   work,
                                               int                 lwork,
                                               int*                info,
                                               int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCheevd(handle,
                               jobz,
                               uplo,
                               n,
                               (hipFloatComplex*)A,
                               lda,
                               W,
                               (hipFloatComplex*)work,
                               lwork,
                               info);
    case API_FORTRAN:
        return hipsolverCheevdFortran(handle,
                                      jobz,
                                      uplo,
                                      n,
                                      (hipFloatComplex*)A,
                                      lda,
                                      W,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info);
    case API_COMPAT:
        return hipsolverDnCheevd(handle,
                                 jobz,
                                 uplo,
                                 n,
                                 (hipFloatComplex*)A,
                                 lda,
                                 W,
                                 (hipFloatComplex*)work,
                                 lwork,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevd_heevd(testAPI_t               API,
                                               hipsolverHandle_t       handle,
                                               hipsolverEigMode_t      jobz,
                                               hipsolverFillMode_t     uplo,
                                               int                     n,
                                               hipsolverDoubleComplex* A,
                                               int                     lda,
                                               int                     stA,
                                               double*                 W,
                                               int                     stW,
                                               hipsolverDoubleComplex* work,
                                               int                     lwork,
                                               int*                    info,
                                               int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZheevd(handle,
                               jobz,
                               uplo,
                               n,
                               (hipDoubleComplex*)A,
                               lda,
                               W,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);
    case API_FORTRAN:
        return hipsolverZheevdFortran(handle,
                                      jobz,
                                      uplo,
                                      n,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      W,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      info);
    case API_COMPAT:
        return hipsolverDnZheevd(handle,
                                 jobz,
                                 uplo,
                                 n,
                                 (hipDoubleComplex*)A,
                                 lda,
                                 W,
                                 (hipDoubleComplex*)work,
                                 lwork,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** SYEVDX/HEEVDX ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_syevdx_heevdx_bufferSize(testAPI_t           API,
                                                            hipsolverHandle_t   handle,
                                                            hipsolverEigMode_t  jobz,
                                                            hipsolverEigRange_t range,
                                                            hipsolverFillMode_t uplo,
                                                            int                 n,
                                                            float*              A,
                                                            int                 lda,
                                                            float               vl,
                                                            float               vu,
                                                            int                 il,
                                                            int                 iu,
                                                            int*                nev,
                                                            float*              W,
                                                            int*                lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSsyevdx_bufferSize(
            handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, lwork);
    case API_COMPAT:
        return hipsolverDnSsyevdx_bufferSize(
            handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevdx_heevdx_bufferSize(testAPI_t           API,
                                                            hipsolverHandle_t   handle,
                                                            hipsolverEigMode_t  jobz,
                                                            hipsolverEigRange_t range,
                                                            hipsolverFillMode_t uplo,
                                                            int                 n,
                                                            double*             A,
                                                            int                 lda,
                                                            double              vl,
                                                            double              vu,
                                                            int                 il,
                                                            int                 iu,
                                                            int*                nev,
                                                            double*             W,
                                                            int*                lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDsyevdx_bufferSize(
            handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, lwork);
    case API_COMPAT:
        return hipsolverDnDsyevdx_bufferSize(
            handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevdx_heevdx_bufferSize(testAPI_t           API,
                                                            hipsolverHandle_t   handle,
                                                            hipsolverEigMode_t  jobz,
                                                            hipsolverEigRange_t range,
                                                            hipsolverFillMode_t uplo,
                                                            int                 n,
                                                            hipsolverComplex*   A,
                                                            int                 lda,
                                                            float               vl,
                                                            float               vu,
                                                            int                 il,
                                                            int                 iu,
                                                            int*                nev,
                                                            float*              W,
                                                            int*                lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCheevdx_bufferSize(
            handle, jobz, range, uplo, n, (hipFloatComplex*)A, lda, vl, vu, il, iu, nev, W, lwork);
    case API_COMPAT:
        return hipsolverDnCheevdx_bufferSize(
            handle, jobz, range, uplo, n, (hipFloatComplex*)A, lda, vl, vu, il, iu, nev, W, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevdx_heevdx_bufferSize(testAPI_t               API,
                                                            hipsolverHandle_t       handle,
                                                            hipsolverEigMode_t      jobz,
                                                            hipsolverEigRange_t     range,
                                                            hipsolverFillMode_t     uplo,
                                                            int                     n,
                                                            hipsolverDoubleComplex* A,
                                                            int                     lda,
                                                            double                  vl,
                                                            double                  vu,
                                                            int                     il,
                                                            int                     iu,
                                                            int*                    nev,
                                                            double*                 W,
                                                            int*                    lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZheevdx_bufferSize(
            handle, jobz, range, uplo, n, (hipDoubleComplex*)A, lda, vl, vu, il, iu, nev, W, lwork);
    case API_COMPAT:
        return hipsolverDnZheevdx_bufferSize(
            handle, jobz, range, uplo, n, (hipDoubleComplex*)A, lda, vl, vu, il, iu, nev, W, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevdx_heevdx(testAPI_t           API,
                                                 hipsolverHandle_t   handle,
                                                 hipsolverEigMode_t  jobz,
                                                 hipsolverEigRange_t range,
                                                 hipsolverFillMode_t uplo,
                                                 int                 n,
                                                 float*              A,
                                                 int                 lda,
                                                 int                 stA,
                                                 float               vl,
                                                 float               vu,
                                                 int                 il,
                                                 int                 iu,
                                                 int*                nev,
                                                 float*              W,
                                                 int                 stW,
                                                 float*              work,
                                                 int                 lwork,
                                                 int*                info,
                                                 int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSsyevdx(
            handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnSsyevdx(
            handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevdx_heevdx(testAPI_t           API,
                                                 hipsolverHandle_t   handle,
                                                 hipsolverEigMode_t  jobz,
                                                 hipsolverEigRange_t range,
                                                 hipsolverFillMode_t uplo,
                                                 int                 n,
                                                 double*             A,
                                                 int                 lda,
                                                 int                 stA,
                                                 double              vl,
                                                 double              vu,
                                                 int                 il,
                                                 int                 iu,
                                                 int*                nev,
                                                 double*             W,
                                                 int                 stW,
                                                 double*             work,
                                                 int                 lwork,
                                                 int*                info,
                                                 int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDsyevdx(
            handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnDsyevdx(
            handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, nev, W, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevdx_heevdx(testAPI_t           API,
                                                 hipsolverHandle_t   handle,
                                                 hipsolverEigMode_t  jobz,
                                                 hipsolverEigRange_t range,
                                                 hipsolverFillMode_t uplo,
                                                 int                 n,
                                                 hipsolverComplex*   A,
                                                 int                 lda,
                                                 int                 stA,
                                                 float               vl,
                                                 float               vu,
                                                 int                 il,
                                                 int                 iu,
                                                 int*                nev,
                                                 float*              W,
                                                 int                 stW,
                                                 hipsolverComplex*   work,
                                                 int                 lwork,
                                                 int*                info,
                                                 int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCheevdx(handle,
                                jobz,
                                range,
                                uplo,
                                n,
                                (hipFloatComplex*)A,
                                lda,
                                vl,
                                vu,
                                il,
                                iu,
                                nev,
                                W,
                                (hipFloatComplex*)work,
                                lwork,
                                info);
    case API_COMPAT:
        return hipsolverDnCheevdx(handle,
                                  jobz,
                                  range,
                                  uplo,
                                  n,
                                  (hipFloatComplex*)A,
                                  lda,
                                  vl,
                                  vu,
                                  il,
                                  iu,
                                  nev,
                                  W,
                                  (hipFloatComplex*)work,
                                  lwork,
                                  info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevdx_heevdx(testAPI_t               API,
                                                 hipsolverHandle_t       handle,
                                                 hipsolverEigMode_t      jobz,
                                                 hipsolverEigRange_t     range,
                                                 hipsolverFillMode_t     uplo,
                                                 int                     n,
                                                 hipsolverDoubleComplex* A,
                                                 int                     lda,
                                                 int                     stA,
                                                 double                  vl,
                                                 double                  vu,
                                                 int                     il,
                                                 int                     iu,
                                                 int*                    nev,
                                                 double*                 W,
                                                 int                     stW,
                                                 hipsolverDoubleComplex* work,
                                                 int                     lwork,
                                                 int*                    info,
                                                 int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZheevdx(handle,
                                jobz,
                                range,
                                uplo,
                                n,
                                (hipDoubleComplex*)A,
                                lda,
                                vl,
                                vu,
                                il,
                                iu,
                                nev,
                                W,
                                (hipDoubleComplex*)work,
                                lwork,
                                info);
    case API_COMPAT:
        return hipsolverDnZheevdx(handle,
                                  jobz,
                                  range,
                                  uplo,
                                  n,
                                  (hipDoubleComplex*)A,
                                  lda,
                                  vl,
                                  vu,
                                  il,
                                  iu,
                                  nev,
                                  W,
                                  (hipDoubleComplex*)work,
                                  lwork,
                                  info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** SYEVJ/HEEVJ ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_syevj_heevj_bufferSize(testAPI_t            API,
                                                          bool                 STRIDED,
                                                          hipsolverHandle_t    handle,
                                                          hipsolverEigMode_t   jobz,
                                                          hipsolverFillMode_t  uplo,
                                                          int                  n,
                                                          float*               A,
                                                          int                  lda,
                                                          float*               W,
                                                          int*                 lwork,
                                                          hipsolverSyevjInfo_t params,
                                                          int                  bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverSsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params);
    case C_NORMAL_ALT:
        return hipsolverSsyevjBatched_bufferSize(
            handle, jobz, uplo, n, A, lda, W, lwork, params, bc);
    case FORTRAN_NORMAL:
        return hipsolverSsyevj_bufferSizeFortran(handle, jobz, uplo, n, A, lda, W, lwork, params);
    case FORTRAN_NORMAL_ALT:
        return hipsolverSsyevjBatched_bufferSizeFortran(
            handle, jobz, uplo, n, A, lda, W, lwork, params, bc);
    case COMPAT_NORMAL:
        return hipsolverDnSsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params);
    case COMPAT_NORMAL_ALT:
        return hipsolverDnSsyevjBatched_bufferSize(
            handle, jobz, uplo, n, A, lda, W, lwork, params, bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevj_heevj_bufferSize(testAPI_t            API,
                                                          bool                 STRIDED,
                                                          hipsolverHandle_t    handle,
                                                          hipsolverEigMode_t   jobz,
                                                          hipsolverFillMode_t  uplo,
                                                          int                  n,
                                                          double*              A,
                                                          int                  lda,
                                                          double*              W,
                                                          int*                 lwork,
                                                          hipsolverSyevjInfo_t params,
                                                          int                  bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverDsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params);
    case C_NORMAL_ALT:
        return hipsolverDsyevjBatched_bufferSize(
            handle, jobz, uplo, n, A, lda, W, lwork, params, bc);
    case FORTRAN_NORMAL:
        return hipsolverDsyevj_bufferSizeFortran(handle, jobz, uplo, n, A, lda, W, lwork, params);
    case FORTRAN_NORMAL_ALT:
        return hipsolverDsyevjBatched_bufferSizeFortran(
            handle, jobz, uplo, n, A, lda, W, lwork, params, bc);
    case COMPAT_NORMAL:
        return hipsolverDnDsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params);
    case COMPAT_NORMAL_ALT:
        return hipsolverDnDsyevjBatched_bufferSize(
            handle, jobz, uplo, n, A, lda, W, lwork, params, bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevj_heevj_bufferSize(testAPI_t            API,
                                                          bool                 STRIDED,
                                                          hipsolverHandle_t    handle,
                                                          hipsolverEigMode_t   jobz,
                                                          hipsolverFillMode_t  uplo,
                                                          int                  n,
                                                          hipsolverComplex*    A,
                                                          int                  lda,
                                                          float*               W,
                                                          int*                 lwork,
                                                          hipsolverSyevjInfo_t params,
                                                          int                  bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverCheevj_bufferSize(
            handle, jobz, uplo, n, (hipFloatComplex*)A, lda, W, lwork, params);
    case C_NORMAL_ALT:
        return hipsolverCheevjBatched_bufferSize(
            handle, jobz, uplo, n, (hipFloatComplex*)A, lda, W, lwork, params, bc);
    case FORTRAN_NORMAL:
        return hipsolverCheevj_bufferSizeFortran(
            handle, jobz, uplo, n, (hipFloatComplex*)A, lda, W, lwork, params);
    case FORTRAN_NORMAL_ALT:
        return hipsolverCheevjBatched_bufferSizeFortran(
            handle, jobz, uplo, n, (hipFloatComplex*)A, lda, W, lwork, params, bc);
    case COMPAT_NORMAL:
        return hipsolverDnCheevj_bufferSize(
            handle, jobz, uplo, n, (hipFloatComplex*)A, lda, W, lwork, params);
    case COMPAT_NORMAL_ALT:
        return hipsolverDnCheevjBatched_bufferSize(
            handle, jobz, uplo, n, (hipFloatComplex*)A, lda, W, lwork, params, bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevj_heevj_bufferSize(testAPI_t               API,
                                                          bool                    STRIDED,
                                                          hipsolverHandle_t       handle,
                                                          hipsolverEigMode_t      jobz,
                                                          hipsolverFillMode_t     uplo,
                                                          int                     n,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          double*                 W,
                                                          int*                    lwork,
                                                          hipsolverSyevjInfo_t    params,
                                                          int                     bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverZheevj_bufferSize(
            handle, jobz, uplo, n, (hipDoubleComplex*)A, lda, W, lwork, params);
    case C_NORMAL_ALT:
        return hipsolverZheevjBatched_bufferSize(
            handle, jobz, uplo, n, (hipDoubleComplex*)A, lda, W, lwork, params, bc);
    case FORTRAN_NORMAL:
        return hipsolverZheevj_bufferSizeFortran(
            handle, jobz, uplo, n, (hipDoubleComplex*)A, lda, W, lwork, params);
    case FORTRAN_NORMAL_ALT:
        return hipsolverZheevjBatched_bufferSizeFortran(
            handle, jobz, uplo, n, (hipDoubleComplex*)A, lda, W, lwork, params, bc);
    case COMPAT_NORMAL:
        return hipsolverDnZheevj_bufferSize(
            handle, jobz, uplo, n, (hipDoubleComplex*)A, lda, W, lwork, params);
    case COMPAT_NORMAL_ALT:
        return hipsolverDnZheevjBatched_bufferSize(
            handle, jobz, uplo, n, (hipDoubleComplex*)A, lda, W, lwork, params, bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevj_heevj(testAPI_t            API,
                                               bool                 STRIDED,
                                               hipsolverHandle_t    handle,
                                               hipsolverEigMode_t   jobz,
                                               hipsolverFillMode_t  uplo,
                                               int                  n,
                                               float*               A,
                                               int                  lda,
                                               int                  stA,
                                               float*               W,
                                               int                  stW,
                                               float*               work,
                                               int                  lwork,
                                               int*                 info,
                                               hipsolverSyevjInfo_t params,
                                               int                  bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverSsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params);
    case C_NORMAL_ALT:
        return hipsolverSsyevjBatched(
            handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, bc);
    case FORTRAN_NORMAL:
        return hipsolverSsyevjFortran(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params);
    case FORTRAN_NORMAL_ALT:
        return hipsolverSsyevjBatchedFortran(
            handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, bc);
    case COMPAT_NORMAL:
        return hipsolverDnSsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params);
    case COMPAT_NORMAL_ALT:
        return hipsolverDnSsyevjBatched(
            handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevj_heevj(testAPI_t            API,
                                               bool                 STRIDED,
                                               hipsolverHandle_t    handle,
                                               hipsolverEigMode_t   jobz,
                                               hipsolverFillMode_t  uplo,
                                               int                  n,
                                               double*              A,
                                               int                  lda,
                                               int                  stA,
                                               double*              W,
                                               int                  stW,
                                               double*              work,
                                               int                  lwork,
                                               int*                 info,
                                               hipsolverSyevjInfo_t params,
                                               int                  bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverDsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params);
    case C_NORMAL_ALT:
        return hipsolverDsyevjBatched(
            handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, bc);
    case FORTRAN_NORMAL:
        return hipsolverDsyevjFortran(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params);
    case FORTRAN_NORMAL_ALT:
        return hipsolverDsyevjBatchedFortran(
            handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, bc);
    case COMPAT_NORMAL:
        return hipsolverDnDsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params);
    case COMPAT_NORMAL_ALT:
        return hipsolverDnDsyevjBatched(
            handle, jobz, uplo, n, A, lda, W, work, lwork, info, params, bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevj_heevj(testAPI_t            API,
                                               bool                 STRIDED,
                                               hipsolverHandle_t    handle,
                                               hipsolverEigMode_t   jobz,
                                               hipsolverFillMode_t  uplo,
                                               int                  n,
                                               hipsolverComplex*    A,
                                               int                  lda,
                                               int                  stA,
                                               float*               W,
                                               int                  stW,
                                               hipsolverComplex*    work,
                                               int                  lwork,
                                               int*                 info,
                                               hipsolverSyevjInfo_t params,
                                               int                  bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverCheevj(handle,
                               jobz,
                               uplo,
                               n,
                               (hipFloatComplex*)A,
                               lda,
                               W,
                               (hipFloatComplex*)work,
                               lwork,
                               info,
                               params);
    case C_NORMAL_ALT:
        return hipsolverCheevjBatched(handle,
                                      jobz,
                                      uplo,
                                      n,
                                      (hipFloatComplex*)A,
                                      lda,
                                      W,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info,
                                      params,
                                      bc);
    case FORTRAN_NORMAL:
        return hipsolverCheevjFortran(handle,
                                      jobz,
                                      uplo,
                                      n,
                                      (hipFloatComplex*)A,
                                      lda,
                                      W,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info,
                                      params);
    case FORTRAN_NORMAL_ALT:
        return hipsolverCheevjBatchedFortran(handle,
                                             jobz,
                                             uplo,
                                             n,
                                             (hipFloatComplex*)A,
                                             lda,
                                             W,
                                             (hipFloatComplex*)work,
                                             lwork,
                                             info,
                                             params,
                                             bc);
    case COMPAT_NORMAL:
        return hipsolverDnCheevj(handle,
                                 jobz,
                                 uplo,
                                 n,
                                 (hipFloatComplex*)A,
                                 lda,
                                 W,
                                 (hipFloatComplex*)work,
                                 lwork,
                                 info,
                                 params);
    case COMPAT_NORMAL_ALT:
        return hipsolverDnCheevjBatched(handle,
                                        jobz,
                                        uplo,
                                        n,
                                        (hipFloatComplex*)A,
                                        lda,
                                        W,
                                        (hipFloatComplex*)work,
                                        lwork,
                                        info,
                                        params,
                                        bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_syevj_heevj(testAPI_t               API,
                                               bool                    STRIDED,
                                               hipsolverHandle_t       handle,
                                               hipsolverEigMode_t      jobz,
                                               hipsolverFillMode_t     uplo,
                                               int                     n,
                                               hipsolverDoubleComplex* A,
                                               int                     lda,
                                               int                     stA,
                                               double*                 W,
                                               int                     stW,
                                               hipsolverDoubleComplex* work,
                                               int                     lwork,
                                               int*                    info,
                                               hipsolverSyevjInfo_t    params,
                                               int                     bc)
{
    switch(api2marshal(API, STRIDED))
    {
    case C_NORMAL:
        return hipsolverZheevj(handle,
                               jobz,
                               uplo,
                               n,
                               (hipDoubleComplex*)A,
                               lda,
                               W,
                               (hipDoubleComplex*)work,
                               lwork,
                               info,
                               params);
    case C_NORMAL_ALT:
        return hipsolverZheevjBatched(handle,
                                      jobz,
                                      uplo,
                                      n,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      W,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      info,
                                      params,
                                      bc);
    case FORTRAN_NORMAL:
        return hipsolverZheevjFortran(handle,
                                      jobz,
                                      uplo,
                                      n,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      W,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      info,
                                      params);
    case FORTRAN_NORMAL_ALT:
        return hipsolverZheevjBatchedFortran(handle,
                                             jobz,
                                             uplo,
                                             n,
                                             (hipDoubleComplex*)A,
                                             lda,
                                             W,
                                             (hipDoubleComplex*)work,
                                             lwork,
                                             info,
                                             params,
                                             bc);
    case COMPAT_NORMAL:
        return hipsolverDnZheevj(handle,
                                 jobz,
                                 uplo,
                                 n,
                                 (hipDoubleComplex*)A,
                                 lda,
                                 W,
                                 (hipDoubleComplex*)work,
                                 lwork,
                                 info,
                                 params);
    case COMPAT_NORMAL_ALT:
        return hipsolverDnZheevjBatched(handle,
                                        jobz,
                                        uplo,
                                        n,
                                        (hipDoubleComplex*)A,
                                        lda,
                                        W,
                                        (hipDoubleComplex*)work,
                                        lwork,
                                        info,
                                        params,
                                        bc);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** SYGVD/HEGVD ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_sygvd_hegvd_bufferSize(testAPI_t           API,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverEigType_t  itype,
                                                          hipsolverEigMode_t  jobz,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          float*              A,
                                                          int                 lda,
                                                          float*              B,
                                                          int                 ldb,
                                                          float*              W,
                                                          int*                lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork);
    case API_FORTRAN:
        return hipsolverSsygvd_bufferSizeFortran(
            handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork);
    case API_COMPAT:
        return hipsolverDnSsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvd_hegvd_bufferSize(testAPI_t           API,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverEigType_t  itype,
                                                          hipsolverEigMode_t  jobz,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          double*             A,
                                                          int                 lda,
                                                          double*             B,
                                                          int                 ldb,
                                                          double*             W,
                                                          int*                lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork);
    case API_FORTRAN:
        return hipsolverDsygvd_bufferSizeFortran(
            handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork);
    case API_COMPAT:
        return hipsolverDnDsygvd_bufferSize(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvd_hegvd_bufferSize(testAPI_t           API,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverEigType_t  itype,
                                                          hipsolverEigMode_t  jobz,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipsolverComplex*   A,
                                                          int                 lda,
                                                          hipsolverComplex*   B,
                                                          int                 ldb,
                                                          float*              W,
                                                          int*                lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverChegvd_bufferSize(handle,
                                          itype,
                                          jobz,
                                          uplo,
                                          n,
                                          (hipFloatComplex*)A,
                                          lda,
                                          (hipFloatComplex*)B,
                                          ldb,
                                          W,
                                          lwork);
    case API_FORTRAN:
        return hipsolverChegvd_bufferSizeFortran(handle,
                                                 itype,
                                                 jobz,
                                                 uplo,
                                                 n,
                                                 (hipFloatComplex*)A,
                                                 lda,
                                                 (hipFloatComplex*)B,
                                                 ldb,
                                                 W,
                                                 lwork);
    case API_COMPAT:
        return hipsolverDnChegvd_bufferSize(handle,
                                            itype,
                                            jobz,
                                            uplo,
                                            n,
                                            (hipFloatComplex*)A,
                                            lda,
                                            (hipFloatComplex*)B,
                                            ldb,
                                            W,
                                            lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvd_hegvd_bufferSize(testAPI_t               API,
                                                          hipsolverHandle_t       handle,
                                                          hipsolverEigType_t      itype,
                                                          hipsolverEigMode_t      jobz,
                                                          hipsolverFillMode_t     uplo,
                                                          int                     n,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          hipsolverDoubleComplex* B,
                                                          int                     ldb,
                                                          double*                 W,
                                                          int*                    lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZhegvd_bufferSize(handle,
                                          itype,
                                          jobz,
                                          uplo,
                                          n,
                                          (hipDoubleComplex*)A,
                                          lda,
                                          (hipDoubleComplex*)B,
                                          ldb,
                                          W,
                                          lwork);
    case API_FORTRAN:
        return hipsolverZhegvd_bufferSizeFortran(handle,
                                                 itype,
                                                 jobz,
                                                 uplo,
                                                 n,
                                                 (hipDoubleComplex*)A,
                                                 lda,
                                                 (hipDoubleComplex*)B,
                                                 ldb,
                                                 W,
                                                 lwork);
    case API_COMPAT:
        return hipsolverDnZhegvd_bufferSize(handle,
                                            itype,
                                            jobz,
                                            uplo,
                                            n,
                                            (hipDoubleComplex*)A,
                                            lda,
                                            (hipDoubleComplex*)B,
                                            ldb,
                                            W,
                                            lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvd_hegvd(testAPI_t           API,
                                               hipsolverHandle_t   handle,
                                               hipsolverEigType_t  itype,
                                               hipsolverEigMode_t  jobz,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               float*              A,
                                               int                 lda,
                                               int                 stA,
                                               float*              B,
                                               int                 ldb,
                                               int                 stB,
                                               float*              W,
                                               int                 stW,
                                               float*              work,
                                               int                 lwork,
                                               int*                info,
                                               int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info);
    case API_FORTRAN:
        return hipsolverSsygvdFortran(
            handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnSsygvd(
            handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvd_hegvd(testAPI_t           API,
                                               hipsolverHandle_t   handle,
                                               hipsolverEigType_t  itype,
                                               hipsolverEigMode_t  jobz,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               double*             A,
                                               int                 lda,
                                               int                 stA,
                                               double*             B,
                                               int                 ldb,
                                               int                 stB,
                                               double*             W,
                                               int                 stW,
                                               double*             work,
                                               int                 lwork,
                                               int*                info,
                                               int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDsygvd(handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info);
    case API_FORTRAN:
        return hipsolverDsygvdFortran(
            handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnDsygvd(
            handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvd_hegvd(testAPI_t           API,
                                               hipsolverHandle_t   handle,
                                               hipsolverEigType_t  itype,
                                               hipsolverEigMode_t  jobz,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               hipsolverComplex*   A,
                                               int                 lda,
                                               int                 stA,
                                               hipsolverComplex*   B,
                                               int                 ldb,
                                               int                 stB,
                                               float*              W,
                                               int                 stW,
                                               hipsolverComplex*   work,
                                               int                 lwork,
                                               int*                info,
                                               int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverChegvd(handle,
                               itype,
                               jobz,
                               uplo,
                               n,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)B,
                               ldb,
                               W,
                               (hipFloatComplex*)work,
                               lwork,
                               info);
    case API_FORTRAN:
        return hipsolverChegvdFortran(handle,
                                      itype,
                                      jobz,
                                      uplo,
                                      n,
                                      (hipFloatComplex*)A,
                                      lda,
                                      (hipFloatComplex*)B,
                                      ldb,
                                      W,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info);
    case API_COMPAT:
        return hipsolverDnChegvd(handle,
                                 itype,
                                 jobz,
                                 uplo,
                                 n,
                                 (hipFloatComplex*)A,
                                 lda,
                                 (hipFloatComplex*)B,
                                 ldb,
                                 W,
                                 (hipFloatComplex*)work,
                                 lwork,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvd_hegvd(testAPI_t               API,
                                               hipsolverHandle_t       handle,
                                               hipsolverEigType_t      itype,
                                               hipsolverEigMode_t      jobz,
                                               hipsolverFillMode_t     uplo,
                                               int                     n,
                                               hipsolverDoubleComplex* A,
                                               int                     lda,
                                               int                     stA,
                                               hipsolverDoubleComplex* B,
                                               int                     ldb,
                                               int                     stB,
                                               double*                 W,
                                               int                     stW,
                                               hipsolverDoubleComplex* work,
                                               int                     lwork,
                                               int*                    info,
                                               int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZhegvd(handle,
                               itype,
                               jobz,
                               uplo,
                               n,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)B,
                               ldb,
                               W,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);
    case API_FORTRAN:
        return hipsolverZhegvdFortran(handle,
                                      itype,
                                      jobz,
                                      uplo,
                                      n,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      (hipDoubleComplex*)B,
                                      ldb,
                                      W,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      info);
    case API_COMPAT:
        return hipsolverDnZhegvd(handle,
                                 itype,
                                 jobz,
                                 uplo,
                                 n,
                                 (hipDoubleComplex*)A,
                                 lda,
                                 (hipDoubleComplex*)B,
                                 ldb,
                                 W,
                                 (hipDoubleComplex*)work,
                                 lwork,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** SYGVDX/HEGVDX ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_sygvdx_hegvdx_bufferSize(testAPI_t           API,
                                                            hipsolverHandle_t   handle,
                                                            hipsolverEigType_t  itype,
                                                            hipsolverEigMode_t  jobz,
                                                            hipsolverEigRange_t range,
                                                            hipsolverFillMode_t uplo,
                                                            int                 n,
                                                            float*              A,
                                                            int                 lda,
                                                            float*              B,
                                                            int                 ldb,
                                                            float               vl,
                                                            float               vu,
                                                            int                 il,
                                                            int                 iu,
                                                            int*                nev,
                                                            float*              W,
                                                            int*                lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSsygvdx_bufferSize(
            handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, nev, W, lwork);
    case API_COMPAT:
        return hipsolverDnSsygvdx_bufferSize(
            handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, nev, W, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvdx_hegvdx_bufferSize(testAPI_t           API,
                                                            hipsolverHandle_t   handle,
                                                            hipsolverEigType_t  itype,
                                                            hipsolverEigMode_t  jobz,
                                                            hipsolverEigRange_t range,
                                                            hipsolverFillMode_t uplo,
                                                            int                 n,
                                                            double*             A,
                                                            int                 lda,
                                                            double*             B,
                                                            int                 ldb,
                                                            double              vl,
                                                            double              vu,
                                                            int                 il,
                                                            int                 iu,
                                                            int*                nev,
                                                            double*             W,
                                                            int*                lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDsygvdx_bufferSize(
            handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, nev, W, lwork);
    case API_COMPAT:
        return hipsolverDnDsygvdx_bufferSize(
            handle, itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, nev, W, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvdx_hegvdx_bufferSize(testAPI_t           API,
                                                            hipsolverHandle_t   handle,
                                                            hipsolverEigType_t  itype,
                                                            hipsolverEigMode_t  jobz,
                                                            hipsolverEigRange_t range,
                                                            hipsolverFillMode_t uplo,
                                                            int                 n,
                                                            hipsolverComplex*   A,
                                                            int                 lda,
                                                            hipsolverComplex*   B,
                                                            int                 ldb,
                                                            float               vl,
                                                            float               vu,
                                                            int                 il,
                                                            int                 iu,
                                                            int*                nev,
                                                            float*              W,
                                                            int*                lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverChegvdx_bufferSize(handle,
                                           itype,
                                           jobz,
                                           range,
                                           uplo,
                                           n,
                                           (hipFloatComplex*)A,
                                           lda,
                                           (hipFloatComplex*)B,
                                           ldb,
                                           vl,
                                           vu,
                                           il,
                                           iu,
                                           nev,
                                           W,
                                           lwork);
    case API_COMPAT:
        return hipsolverDnChegvdx_bufferSize(handle,
                                             itype,
                                             jobz,
                                             range,
                                             uplo,
                                             n,
                                             (hipFloatComplex*)A,
                                             lda,
                                             (hipFloatComplex*)B,
                                             ldb,
                                             vl,
                                             vu,
                                             il,
                                             iu,
                                             nev,
                                             W,
                                             lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvdx_hegvdx_bufferSize(testAPI_t               API,
                                                            hipsolverHandle_t       handle,
                                                            hipsolverEigType_t      itype,
                                                            hipsolverEigMode_t      jobz,
                                                            hipsolverEigRange_t     range,
                                                            hipsolverFillMode_t     uplo,
                                                            int                     n,
                                                            hipsolverDoubleComplex* A,
                                                            int                     lda,
                                                            hipsolverDoubleComplex* B,
                                                            int                     ldb,
                                                            double                  vl,
                                                            double                  vu,
                                                            int                     il,
                                                            int                     iu,
                                                            int*                    nev,
                                                            double*                 W,
                                                            int*                    lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZhegvdx_bufferSize(handle,
                                           itype,
                                           jobz,
                                           range,
                                           uplo,
                                           n,
                                           (hipDoubleComplex*)A,
                                           lda,
                                           (hipDoubleComplex*)B,
                                           ldb,
                                           vl,
                                           vu,
                                           il,
                                           iu,
                                           nev,
                                           W,
                                           lwork);
    case API_COMPAT:
        return hipsolverDnZhegvdx_bufferSize(handle,
                                             itype,
                                             jobz,
                                             range,
                                             uplo,
                                             n,
                                             (hipDoubleComplex*)A,
                                             lda,
                                             (hipDoubleComplex*)B,
                                             ldb,
                                             vl,
                                             vu,
                                             il,
                                             iu,
                                             nev,
                                             W,
                                             lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvdx_hegvdx(testAPI_t           API,
                                                 hipsolverHandle_t   handle,
                                                 hipsolverEigType_t  itype,
                                                 hipsolverEigMode_t  jobz,
                                                 hipsolverEigRange_t range,
                                                 hipsolverFillMode_t uplo,
                                                 int                 n,
                                                 float*              A,
                                                 int                 lda,
                                                 int                 stA,
                                                 float*              B,
                                                 int                 ldb,
                                                 int                 stB,
                                                 float               vl,
                                                 float               vu,
                                                 int                 il,
                                                 int                 iu,
                                                 int*                nev,
                                                 float*              W,
                                                 int                 stW,
                                                 float*              work,
                                                 int                 lwork,
                                                 int*                info,
                                                 int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSsygvdx(handle,
                                itype,
                                jobz,
                                range,
                                uplo,
                                n,
                                A,
                                lda,
                                B,
                                ldb,
                                vl,
                                vu,
                                il,
                                iu,
                                nev,
                                W,
                                work,
                                lwork,
                                info);
    case API_COMPAT:
        return hipsolverDnSsygvdx(handle,
                                  itype,
                                  jobz,
                                  range,
                                  uplo,
                                  n,
                                  A,
                                  lda,
                                  B,
                                  ldb,
                                  vl,
                                  vu,
                                  il,
                                  iu,
                                  nev,
                                  W,
                                  work,
                                  lwork,
                                  info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvdx_hegvdx(testAPI_t           API,
                                                 hipsolverHandle_t   handle,
                                                 hipsolverEigType_t  itype,
                                                 hipsolverEigMode_t  jobz,
                                                 hipsolverEigRange_t range,
                                                 hipsolverFillMode_t uplo,
                                                 int                 n,
                                                 double*             A,
                                                 int                 lda,
                                                 int                 stA,
                                                 double*             B,
                                                 int                 ldb,
                                                 int                 stB,
                                                 double              vl,
                                                 double              vu,
                                                 int                 il,
                                                 int                 iu,
                                                 int*                nev,
                                                 double*             W,
                                                 int                 stW,
                                                 double*             work,
                                                 int                 lwork,
                                                 int*                info,
                                                 int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDsygvdx(handle,
                                itype,
                                jobz,
                                range,
                                uplo,
                                n,
                                A,
                                lda,
                                B,
                                ldb,
                                vl,
                                vu,
                                il,
                                iu,
                                nev,
                                W,
                                work,
                                lwork,
                                info);
    case API_COMPAT:
        return hipsolverDnDsygvdx(handle,
                                  itype,
                                  jobz,
                                  range,
                                  uplo,
                                  n,
                                  A,
                                  lda,
                                  B,
                                  ldb,
                                  vl,
                                  vu,
                                  il,
                                  iu,
                                  nev,
                                  W,
                                  work,
                                  lwork,
                                  info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvdx_hegvdx(testAPI_t           API,
                                                 hipsolverHandle_t   handle,
                                                 hipsolverEigType_t  itype,
                                                 hipsolverEigMode_t  jobz,
                                                 hipsolverEigRange_t range,
                                                 hipsolverFillMode_t uplo,
                                                 int                 n,
                                                 hipsolverComplex*   A,
                                                 int                 lda,
                                                 int                 stA,
                                                 hipsolverComplex*   B,
                                                 int                 ldb,
                                                 int                 stB,
                                                 float               vl,
                                                 float               vu,
                                                 int                 il,
                                                 int                 iu,
                                                 int*                nev,
                                                 float*              W,
                                                 int                 stW,
                                                 hipsolverComplex*   work,
                                                 int                 lwork,
                                                 int*                info,
                                                 int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverChegvdx(handle,
                                itype,
                                jobz,
                                range,
                                uplo,
                                n,
                                (hipFloatComplex*)A,
                                lda,
                                (hipFloatComplex*)B,
                                ldb,
                                vl,
                                vu,
                                il,
                                iu,
                                nev,
                                W,
                                (hipFloatComplex*)work,
                                lwork,
                                info);
    case API_COMPAT:
        return hipsolverDnChegvdx(handle,
                                  itype,
                                  jobz,
                                  range,
                                  uplo,
                                  n,
                                  (hipFloatComplex*)A,
                                  lda,
                                  (hipFloatComplex*)B,
                                  ldb,
                                  vl,
                                  vu,
                                  il,
                                  iu,
                                  nev,
                                  W,
                                  (hipFloatComplex*)work,
                                  lwork,
                                  info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvdx_hegvdx(testAPI_t               API,
                                                 hipsolverHandle_t       handle,
                                                 hipsolverEigType_t      itype,
                                                 hipsolverEigMode_t      jobz,
                                                 hipsolverEigRange_t     range,
                                                 hipsolverFillMode_t     uplo,
                                                 int                     n,
                                                 hipsolverDoubleComplex* A,
                                                 int                     lda,
                                                 int                     stA,
                                                 hipsolverDoubleComplex* B,
                                                 int                     ldb,
                                                 int                     stB,
                                                 double                  vl,
                                                 double                  vu,
                                                 int                     il,
                                                 int                     iu,
                                                 int*                    nev,
                                                 double*                 W,
                                                 int                     stW,
                                                 hipsolverDoubleComplex* work,
                                                 int                     lwork,
                                                 int*                    info,
                                                 int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZhegvdx(handle,
                                itype,
                                jobz,
                                range,
                                uplo,
                                n,
                                (hipDoubleComplex*)A,
                                lda,
                                (hipDoubleComplex*)B,
                                ldb,
                                vl,
                                vu,
                                il,
                                iu,
                                nev,
                                W,
                                (hipDoubleComplex*)work,
                                lwork,
                                info);
    case API_COMPAT:
        return hipsolverDnZhegvdx(handle,
                                  itype,
                                  jobz,
                                  range,
                                  uplo,
                                  n,
                                  (hipDoubleComplex*)A,
                                  lda,
                                  (hipDoubleComplex*)B,
                                  ldb,
                                  vl,
                                  vu,
                                  il,
                                  iu,
                                  nev,
                                  W,
                                  (hipDoubleComplex*)work,
                                  lwork,
                                  info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** SYGVJ/HEGVJ ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_sygvj_hegvj_bufferSize(testAPI_t            API,
                                                          hipsolverHandle_t    handle,
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
                                                          hipsolverSyevjInfo_t params)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSsygvj_bufferSize(
            handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params);
    case API_FORTRAN:
        return hipsolverSsygvj_bufferSizeFortran(
            handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params);
    case API_COMPAT:
        return hipsolverDnSsygvj_bufferSize(
            handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvj_hegvj_bufferSize(testAPI_t            API,
                                                          hipsolverHandle_t    handle,
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
                                                          hipsolverSyevjInfo_t params)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDsygvj_bufferSize(
            handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params);
    case API_FORTRAN:
        return hipsolverDsygvj_bufferSizeFortran(
            handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params);
    case API_COMPAT:
        return hipsolverDnDsygvj_bufferSize(
            handle, itype, jobz, uplo, n, A, lda, B, ldb, W, lwork, params);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvj_hegvj_bufferSize(testAPI_t            API,
                                                          hipsolverHandle_t    handle,
                                                          hipsolverEigType_t   itype,
                                                          hipsolverEigMode_t   jobz,
                                                          hipsolverFillMode_t  uplo,
                                                          int                  n,
                                                          hipsolverComplex*    A,
                                                          int                  lda,
                                                          hipsolverComplex*    B,
                                                          int                  ldb,
                                                          float*               W,
                                                          int*                 lwork,
                                                          hipsolverSyevjInfo_t params)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverChegvj_bufferSize(handle,
                                          itype,
                                          jobz,
                                          uplo,
                                          n,
                                          (hipFloatComplex*)A,
                                          lda,
                                          (hipFloatComplex*)B,
                                          ldb,
                                          W,
                                          lwork,
                                          params);
    case API_FORTRAN:
        return hipsolverChegvj_bufferSizeFortran(handle,
                                                 itype,
                                                 jobz,
                                                 uplo,
                                                 n,
                                                 (hipFloatComplex*)A,
                                                 lda,
                                                 (hipFloatComplex*)B,
                                                 ldb,
                                                 W,
                                                 lwork,
                                                 params);
    case API_COMPAT:
        return hipsolverDnChegvj_bufferSize(handle,
                                            itype,
                                            jobz,
                                            uplo,
                                            n,
                                            (hipFloatComplex*)A,
                                            lda,
                                            (hipFloatComplex*)B,
                                            ldb,
                                            W,
                                            lwork,
                                            params);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvj_hegvj_bufferSize(testAPI_t               API,
                                                          hipsolverHandle_t       handle,
                                                          hipsolverEigType_t      itype,
                                                          hipsolverEigMode_t      jobz,
                                                          hipsolverFillMode_t     uplo,
                                                          int                     n,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          hipsolverDoubleComplex* B,
                                                          int                     ldb,
                                                          double*                 W,
                                                          int*                    lwork,
                                                          hipsolverSyevjInfo_t    params)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZhegvj_bufferSize(handle,
                                          itype,
                                          jobz,
                                          uplo,
                                          n,
                                          (hipDoubleComplex*)A,
                                          lda,
                                          (hipDoubleComplex*)B,
                                          ldb,
                                          W,
                                          lwork,
                                          params);
    case API_FORTRAN:
        return hipsolverZhegvj_bufferSizeFortran(handle,
                                                 itype,
                                                 jobz,
                                                 uplo,
                                                 n,
                                                 (hipDoubleComplex*)A,
                                                 lda,
                                                 (hipDoubleComplex*)B,
                                                 ldb,
                                                 W,
                                                 lwork,
                                                 params);
    case API_COMPAT:
        return hipsolverDnZhegvj_bufferSize(handle,
                                            itype,
                                            jobz,
                                            uplo,
                                            n,
                                            (hipDoubleComplex*)A,
                                            lda,
                                            (hipDoubleComplex*)B,
                                            ldb,
                                            W,
                                            lwork,
                                            params);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvj_hegvj(testAPI_t            API,
                                               hipsolverHandle_t    handle,
                                               hipsolverEigType_t   itype,
                                               hipsolverEigMode_t   jobz,
                                               hipsolverFillMode_t  uplo,
                                               int                  n,
                                               float*               A,
                                               int                  lda,
                                               int                  stA,
                                               float*               B,
                                               int                  ldb,
                                               int                  stB,
                                               float*               W,
                                               int                  stW,
                                               float*               work,
                                               int                  lwork,
                                               int*                 info,
                                               hipsolverSyevjInfo_t params,
                                               int                  bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSsygvj(
            handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params);
    case API_FORTRAN:
        return hipsolverSsygvjFortran(
            handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params);
    case API_COMPAT:
        return hipsolverDnSsygvj(
            handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvj_hegvj(testAPI_t            API,
                                               hipsolverHandle_t    handle,
                                               hipsolverEigType_t   itype,
                                               hipsolverEigMode_t   jobz,
                                               hipsolverFillMode_t  uplo,
                                               int                  n,
                                               double*              A,
                                               int                  lda,
                                               int                  stA,
                                               double*              B,
                                               int                  ldb,
                                               int                  stB,
                                               double*              W,
                                               int                  stW,
                                               double*              work,
                                               int                  lwork,
                                               int*                 info,
                                               hipsolverSyevjInfo_t params,
                                               int                  bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDsygvj(
            handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params);
    case API_FORTRAN:
        return hipsolverDsygvjFortran(
            handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params);
    case API_COMPAT:
        return hipsolverDnDsygvj(
            handle, itype, jobz, uplo, n, A, lda, B, ldb, W, work, lwork, info, params);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvj_hegvj(testAPI_t            API,
                                               hipsolverHandle_t    handle,
                                               hipsolverEigType_t   itype,
                                               hipsolverEigMode_t   jobz,
                                               hipsolverFillMode_t  uplo,
                                               int                  n,
                                               hipsolverComplex*    A,
                                               int                  lda,
                                               int                  stA,
                                               hipsolverComplex*    B,
                                               int                  ldb,
                                               int                  stB,
                                               float*               W,
                                               int                  stW,
                                               hipsolverComplex*    work,
                                               int                  lwork,
                                               int*                 info,
                                               hipsolverSyevjInfo_t params,
                                               int                  bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverChegvj(handle,
                               itype,
                               jobz,
                               uplo,
                               n,
                               (hipFloatComplex*)A,
                               lda,
                               (hipFloatComplex*)B,
                               ldb,
                               W,
                               (hipFloatComplex*)work,
                               lwork,
                               info,
                               params);
    case API_FORTRAN:
        return hipsolverChegvjFortran(handle,
                                      itype,
                                      jobz,
                                      uplo,
                                      n,
                                      (hipFloatComplex*)A,
                                      lda,
                                      (hipFloatComplex*)B,
                                      ldb,
                                      W,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info,
                                      params);
    case API_COMPAT:
        return hipsolverDnChegvj(handle,
                                 itype,
                                 jobz,
                                 uplo,
                                 n,
                                 (hipFloatComplex*)A,
                                 lda,
                                 (hipFloatComplex*)B,
                                 ldb,
                                 W,
                                 (hipFloatComplex*)work,
                                 lwork,
                                 info,
                                 params);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sygvj_hegvj(testAPI_t               API,
                                               hipsolverHandle_t       handle,
                                               hipsolverEigType_t      itype,
                                               hipsolverEigMode_t      jobz,
                                               hipsolverFillMode_t     uplo,
                                               int                     n,
                                               hipsolverDoubleComplex* A,
                                               int                     lda,
                                               int                     stA,
                                               hipsolverDoubleComplex* B,
                                               int                     ldb,
                                               int                     stB,
                                               double*                 W,
                                               int                     stW,
                                               hipsolverDoubleComplex* work,
                                               int                     lwork,
                                               int*                    info,
                                               hipsolverSyevjInfo_t    params,
                                               int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZhegvj(handle,
                               itype,
                               jobz,
                               uplo,
                               n,
                               (hipDoubleComplex*)A,
                               lda,
                               (hipDoubleComplex*)B,
                               ldb,
                               W,
                               (hipDoubleComplex*)work,
                               lwork,
                               info,
                               params);
    case API_FORTRAN:
        return hipsolverZhegvjFortran(handle,
                                      itype,
                                      jobz,
                                      uplo,
                                      n,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      (hipDoubleComplex*)B,
                                      ldb,
                                      W,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      info,
                                      params);
    case API_COMPAT:
        return hipsolverDnZhegvj(handle,
                                 itype,
                                 jobz,
                                 uplo,
                                 n,
                                 (hipDoubleComplex*)A,
                                 lda,
                                 (hipDoubleComplex*)B,
                                 ldb,
                                 W,
                                 (hipDoubleComplex*)work,
                                 lwork,
                                 info,
                                 params);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** SYTRD/HETRD ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_sytrd_hetrd_bufferSize(testAPI_t           API,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          float*              A,
                                                          int                 lda,
                                                          float*              D,
                                                          float*              E,
                                                          float*              tau,
                                                          int*                lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSsytrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, lwork);
    case API_FORTRAN:
        return hipsolverSsytrd_bufferSizeFortran(handle, uplo, n, A, lda, D, E, tau, lwork);
    case API_COMPAT:
        return hipsolverDnSsytrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sytrd_hetrd_bufferSize(testAPI_t           API,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          double*             A,
                                                          int                 lda,
                                                          double*             D,
                                                          double*             E,
                                                          double*             tau,
                                                          int*                lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDsytrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, lwork);
    case API_FORTRAN:
        return hipsolverDsytrd_bufferSizeFortran(handle, uplo, n, A, lda, D, E, tau, lwork);
    case API_COMPAT:
        return hipsolverDnDsytrd_bufferSize(handle, uplo, n, A, lda, D, E, tau, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sytrd_hetrd_bufferSize(testAPI_t           API,
                                                          hipsolverHandle_t   handle,
                                                          hipsolverFillMode_t uplo,
                                                          int                 n,
                                                          hipsolverComplex*   A,
                                                          int                 lda,
                                                          float*              D,
                                                          float*              E,
                                                          hipsolverComplex*   tau,
                                                          int*                lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverChetrd_bufferSize(
            handle, uplo, n, (hipFloatComplex*)A, lda, D, E, (hipFloatComplex*)tau, lwork);
    case API_FORTRAN:
        return hipsolverChetrd_bufferSizeFortran(
            handle, uplo, n, (hipFloatComplex*)A, lda, D, E, (hipFloatComplex*)tau, lwork);
    case API_COMPAT:
        return hipsolverDnChetrd_bufferSize(
            handle, uplo, n, (hipFloatComplex*)A, lda, D, E, (hipFloatComplex*)tau, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sytrd_hetrd_bufferSize(testAPI_t               API,
                                                          hipsolverHandle_t       handle,
                                                          hipsolverFillMode_t     uplo,
                                                          int                     n,
                                                          hipsolverDoubleComplex* A,
                                                          int                     lda,
                                                          double*                 D,
                                                          double*                 E,
                                                          hipsolverDoubleComplex* tau,
                                                          int*                    lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZhetrd_bufferSize(
            handle, uplo, n, (hipDoubleComplex*)A, lda, D, E, (hipDoubleComplex*)tau, lwork);
    case API_FORTRAN:
        return hipsolverZhetrd_bufferSizeFortran(
            handle, uplo, n, (hipDoubleComplex*)A, lda, D, E, (hipDoubleComplex*)tau, lwork);
    case API_COMPAT:
        return hipsolverDnZhetrd_bufferSize(
            handle, uplo, n, (hipDoubleComplex*)A, lda, D, E, (hipDoubleComplex*)tau, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sytrd_hetrd(testAPI_t           API,
                                               hipsolverHandle_t   handle,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               float*              A,
                                               int                 lda,
                                               int                 stA,
                                               float*              D,
                                               int                 stD,
                                               float*              E,
                                               int                 stE,
                                               float*              tau,
                                               int                 stP,
                                               float*              work,
                                               int                 lwork,
                                               int*                info,
                                               int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSsytrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, info);
    case API_FORTRAN:
        return hipsolverSsytrdFortran(handle, uplo, n, A, lda, D, E, tau, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnSsytrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sytrd_hetrd(testAPI_t           API,
                                               hipsolverHandle_t   handle,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               double*             A,
                                               int                 lda,
                                               int                 stA,
                                               double*             D,
                                               int                 stD,
                                               double*             E,
                                               int                 stE,
                                               double*             tau,
                                               int                 stP,
                                               double*             work,
                                               int                 lwork,
                                               int*                info,
                                               int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDsytrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, info);
    case API_FORTRAN:
        return hipsolverDsytrdFortran(handle, uplo, n, A, lda, D, E, tau, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnDsytrd(handle, uplo, n, A, lda, D, E, tau, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sytrd_hetrd(testAPI_t           API,
                                               hipsolverHandle_t   handle,
                                               hipsolverFillMode_t uplo,
                                               int                 n,
                                               hipsolverComplex*   A,
                                               int                 lda,
                                               int                 stA,
                                               float*              D,
                                               int                 stD,
                                               float*              E,
                                               int                 stE,
                                               hipsolverComplex*   tau,
                                               int                 stP,
                                               hipsolverComplex*   work,
                                               int                 lwork,
                                               int*                info,
                                               int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverChetrd(handle,
                               uplo,
                               n,
                               (hipFloatComplex*)A,
                               lda,
                               D,
                               E,
                               (hipFloatComplex*)tau,
                               (hipFloatComplex*)work,
                               lwork,
                               info);
    case API_FORTRAN:
        return hipsolverChetrdFortran(handle,
                                      uplo,
                                      n,
                                      (hipFloatComplex*)A,
                                      lda,
                                      D,
                                      E,
                                      (hipFloatComplex*)tau,
                                      (hipFloatComplex*)work,
                                      lwork,
                                      info);
    case API_COMPAT:
        return hipsolverDnChetrd(handle,
                                 uplo,
                                 n,
                                 (hipFloatComplex*)A,
                                 lda,
                                 D,
                                 E,
                                 (hipFloatComplex*)tau,
                                 (hipFloatComplex*)work,
                                 lwork,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sytrd_hetrd(testAPI_t               API,
                                               hipsolverHandle_t       handle,
                                               hipsolverFillMode_t     uplo,
                                               int                     n,
                                               hipsolverDoubleComplex* A,
                                               int                     lda,
                                               int                     stA,
                                               double*                 D,
                                               int                     stD,
                                               double*                 E,
                                               int                     stE,
                                               hipsolverDoubleComplex* tau,
                                               int                     stP,
                                               hipsolverDoubleComplex* work,
                                               int                     lwork,
                                               int*                    info,
                                               int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZhetrd(handle,
                               uplo,
                               n,
                               (hipDoubleComplex*)A,
                               lda,
                               D,
                               E,
                               (hipDoubleComplex*)tau,
                               (hipDoubleComplex*)work,
                               lwork,
                               info);
    case API_FORTRAN:
        return hipsolverZhetrdFortran(handle,
                                      uplo,
                                      n,
                                      (hipDoubleComplex*)A,
                                      lda,
                                      D,
                                      E,
                                      (hipDoubleComplex*)tau,
                                      (hipDoubleComplex*)work,
                                      lwork,
                                      info);
    case API_COMPAT:
        return hipsolverDnZhetrd(handle,
                                 uplo,
                                 n,
                                 (hipDoubleComplex*)A,
                                 lda,
                                 D,
                                 E,
                                 (hipDoubleComplex*)tau,
                                 (hipDoubleComplex*)work,
                                 lwork,
                                 info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/

/******************** SYTRF ********************/
// normal and strided_batched
inline hipsolverStatus_t hipsolver_sytrf_bufferSize(
    testAPI_t API, hipsolverHandle_t handle, int n, float* A, int lda, int* lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSsytrf_bufferSize(handle, n, A, lda, lwork);
    case API_FORTRAN:
        return hipsolverSsytrf_bufferSizeFortran(handle, n, A, lda, lwork);
    case API_COMPAT:
        return hipsolverDnSsytrf_bufferSize(handle, n, A, lda, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sytrf_bufferSize(
    testAPI_t API, hipsolverHandle_t handle, int n, double* A, int lda, int* lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDsytrf_bufferSize(handle, n, A, lda, lwork);
    case API_FORTRAN:
        return hipsolverDsytrf_bufferSizeFortran(handle, n, A, lda, lwork);
    case API_COMPAT:
        return hipsolverDnDsytrf_bufferSize(handle, n, A, lda, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sytrf_bufferSize(
    testAPI_t API, hipsolverHandle_t handle, int n, hipsolverComplex* A, int lda, int* lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCsytrf_bufferSize(handle, n, (hipFloatComplex*)A, lda, lwork);
    case API_FORTRAN:
        return hipsolverCsytrf_bufferSizeFortran(handle, n, (hipFloatComplex*)A, lda, lwork);
    case API_COMPAT:
        return hipsolverDnCsytrf_bufferSize(handle, n, (hipFloatComplex*)A, lda, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sytrf_bufferSize(
    testAPI_t API, hipsolverHandle_t handle, int n, hipsolverDoubleComplex* A, int lda, int* lwork)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZsytrf_bufferSize(handle, n, (hipDoubleComplex*)A, lda, lwork);
    case API_FORTRAN:
        return hipsolverZsytrf_bufferSizeFortran(handle, n, (hipDoubleComplex*)A, lda, lwork);
    case API_COMPAT:
        return hipsolverDnZsytrf_bufferSize(handle, n, (hipDoubleComplex*)A, lda, lwork);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sytrf(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         float*              A,
                                         int                 lda,
                                         int                 stA,
                                         int*                ipiv,
                                         int                 stP,
                                         float*              work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverSsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info);
    case API_FORTRAN:
        return hipsolverSsytrfFortran(handle, uplo, n, A, lda, ipiv, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnSsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sytrf(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         double*             A,
                                         int                 lda,
                                         int                 stA,
                                         int*                ipiv,
                                         int                 stP,
                                         double*             work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverDsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info);
    case API_FORTRAN:
        return hipsolverDsytrfFortran(handle, uplo, n, A, lda, ipiv, work, lwork, info);
    case API_COMPAT:
        return hipsolverDnDsytrf(handle, uplo, n, A, lda, ipiv, work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sytrf(testAPI_t           API,
                                         hipsolverHandle_t   handle,
                                         hipsolverFillMode_t uplo,
                                         int                 n,
                                         hipsolverComplex*   A,
                                         int                 lda,
                                         int                 stA,
                                         int*                ipiv,
                                         int                 stP,
                                         hipsolverComplex*   work,
                                         int                 lwork,
                                         int*                info,
                                         int                 bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverCsytrf(
            handle, uplo, n, (hipFloatComplex*)A, lda, ipiv, (hipFloatComplex*)work, lwork, info);
    case API_FORTRAN:
        return hipsolverCsytrfFortran(
            handle, uplo, n, (hipFloatComplex*)A, lda, ipiv, (hipFloatComplex*)work, lwork, info);
    case API_COMPAT:
        return hipsolverDnCsytrf(
            handle, uplo, n, (hipFloatComplex*)A, lda, ipiv, (hipFloatComplex*)work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}

inline hipsolverStatus_t hipsolver_sytrf(testAPI_t               API,
                                         hipsolverHandle_t       handle,
                                         hipsolverFillMode_t     uplo,
                                         int                     n,
                                         hipsolverDoubleComplex* A,
                                         int                     lda,
                                         int                     stA,
                                         int*                    ipiv,
                                         int                     stP,
                                         hipsolverDoubleComplex* work,
                                         int                     lwork,
                                         int*                    info,
                                         int                     bc)
{
    switch(API)
    {
    case API_NORMAL:
        return hipsolverZsytrf(
            handle, uplo, n, (hipDoubleComplex*)A, lda, ipiv, (hipDoubleComplex*)work, lwork, info);
    case API_FORTRAN:
        return hipsolverZsytrfFortran(
            handle, uplo, n, (hipDoubleComplex*)A, lda, ipiv, (hipDoubleComplex*)work, lwork, info);
    case API_COMPAT:
        return hipsolverDnZsytrf(
            handle, uplo, n, (hipDoubleComplex*)A, lda, ipiv, (hipDoubleComplex*)work, lwork, info);
    default:
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
/********************************************************/
