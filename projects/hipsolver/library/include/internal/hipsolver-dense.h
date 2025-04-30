/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc.
 * ************************************************************************ */

/*! \file
 *  \brief hipsolver-dense.h provides a compatibility API for users of
 *  cuSOLVER wishing to transition to hipSOLVER. Functions in this file
 *  have parameter lists matching those of cuSOLVER, but may suffer from
 *  performance issues when using the rocSOLVER backend. Switching to the
 *  use of functions from hipsolver-functions.h is recommended.
 */

#ifndef HIPSOLVER_DENSE_H
#define HIPSOLVER_DENSE_H

#include "hipsolver-types.h"

/*! \brief Provided for convenience when porting code from cuSOLVER.
 ********************************************************************************/
typedef hipsolverHandle_t hipsolverDnHandle_t;

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief An alias for #hipsolverCreate.
 ********************************************************************************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCreate(hipsolverHandle_t* handle);

/*! \brief An alias for #hipsolverDestroy.
 ********************************************************************************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDestroy(hipsolverHandle_t handle);

/*! \brief An alias for #hipsolverSetStream.
 ********************************************************************************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSetStream(hipsolverHandle_t handle,
                                                        hipStream_t       streamId);

/*! \brief An alias for #hipsolverGetStream.
 ********************************************************************************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnGetStream(hipsolverHandle_t handle,
                                                        hipStream_t*      streamId);

/*! \brief An alias for #hipsolverSetDeterministicMode.
 ********************************************************************************/
HIPSOLVER_EXPORT hipsolverStatus_t
    hipsolverDnSetDeterministicMode(hipsolverHandle_t handle, hipsolverDeterministicMode_t mode);

/*! \brief An alias for #hipsolverGetDeterministicMode.
 ********************************************************************************/
HIPSOLVER_EXPORT hipsolverStatus_t
    hipsolverDnGetDeterministicMode(hipsolverHandle_t handle, hipsolverDeterministicMode_t* mode);

// gesvdj params
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCreateGesvdjInfo(hipsolverGesvdjInfo_t* info);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDestroyGesvdjInfo(hipsolverGesvdjInfo_t info);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXgesvdjSetMaxSweeps(hipsolverGesvdjInfo_t info,
                                                                  int                   max_sweeps);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXgesvdjSetSortEig(hipsolverGesvdjInfo_t info,
                                                                int                   sort_eig);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXgesvdjSetTolerance(hipsolverGesvdjInfo_t info,
                                                                  double                tolerance);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXgesvdjGetResidual(hipsolverDnHandle_t   handle,
                                                                 hipsolverGesvdjInfo_t info,
                                                                 double*               residual);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXgesvdjGetSweeps(hipsolverDnHandle_t   handle,
                                                               hipsolverGesvdjInfo_t info,
                                                               int* executed_sweeps);

// syevj params
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCreateSyevjInfo(hipsolverSyevjInfo_t* info);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDestroySyevjInfo(hipsolverSyevjInfo_t info);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXsyevjSetMaxSweeps(hipsolverSyevjInfo_t info,
                                                                 int                  max_sweeps);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXsyevjSetSortEig(hipsolverSyevjInfo_t info,
                                                               int                  sort_eig);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXsyevjSetTolerance(hipsolverSyevjInfo_t info,
                                                                 double               tolerance);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXsyevjGetResidual(hipsolverDnHandle_t  handle,
                                                                hipsolverSyevjInfo_t info,
                                                                double*              residual);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXsyevjGetSweeps(hipsolverDnHandle_t  handle,
                                                              hipsolverSyevjInfo_t info,
                                                              int*                 executed_sweeps);

// orgbr/ungbr
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSorgbr_bufferSize(hipsolverHandle_t handle,
                                                                hipblasSideMode_t side,
                                                                int               m,
                                                                int               n,
                                                                int               k,
                                                                const float*      A,
                                                                int               lda,
                                                                const float*      tau,
                                                                int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDorgbr_bufferSize(hipsolverHandle_t handle,
                                                                hipblasSideMode_t side,
                                                                int               m,
                                                                int               n,
                                                                int               k,
                                                                const double*     A,
                                                                int               lda,
                                                                const double*     tau,
                                                                int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCungbr_bufferSize(hipsolverHandle_t      handle,
                                                                hipblasSideMode_t      side,
                                                                int                    m,
                                                                int                    n,
                                                                int                    k,
                                                                const hipFloatComplex* A,
                                                                int                    lda,
                                                                const hipFloatComplex* tau,
                                                                int*                   lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZungbr_bufferSize(hipsolverHandle_t       handle,
                                                                hipblasSideMode_t       side,
                                                                int                     m,
                                                                int                     n,
                                                                int                     k,
                                                                const hipDoubleComplex* A,
                                                                int                     lda,
                                                                const hipDoubleComplex* tau,
                                                                int*                    lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSorgbr(hipsolverHandle_t handle,
                                                     hipblasSideMode_t side,
                                                     int               m,
                                                     int               n,
                                                     int               k,
                                                     float*            A,
                                                     int               lda,
                                                     const float*      tau,
                                                     float*            work,
                                                     int               lwork,
                                                     int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDorgbr(hipsolverHandle_t handle,
                                                     hipblasSideMode_t side,
                                                     int               m,
                                                     int               n,
                                                     int               k,
                                                     double*           A,
                                                     int               lda,
                                                     const double*     tau,
                                                     double*           work,
                                                     int               lwork,
                                                     int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCungbr(hipsolverHandle_t      handle,
                                                     hipblasSideMode_t      side,
                                                     int                    m,
                                                     int                    n,
                                                     int                    k,
                                                     hipFloatComplex*       A,
                                                     int                    lda,
                                                     const hipFloatComplex* tau,
                                                     hipFloatComplex*       work,
                                                     int                    lwork,
                                                     int*                   devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZungbr(hipsolverHandle_t       handle,
                                                     hipblasSideMode_t       side,
                                                     int                     m,
                                                     int                     n,
                                                     int                     k,
                                                     hipDoubleComplex*       A,
                                                     int                     lda,
                                                     const hipDoubleComplex* tau,
                                                     hipDoubleComplex*       work,
                                                     int                     lwork,
                                                     int*                    devInfo);

// orgqr/ungqr
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSorgqr_bufferSize(hipsolverHandle_t handle,
                                                                int               m,
                                                                int               n,
                                                                int               k,
                                                                const float*      A,
                                                                int               lda,
                                                                const float*      tau,
                                                                int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDorgqr_bufferSize(hipsolverHandle_t handle,
                                                                int               m,
                                                                int               n,
                                                                int               k,
                                                                const double*     A,
                                                                int               lda,
                                                                const double*     tau,
                                                                int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCungqr_bufferSize(hipsolverHandle_t      handle,
                                                                int                    m,
                                                                int                    n,
                                                                int                    k,
                                                                const hipFloatComplex* A,
                                                                int                    lda,
                                                                const hipFloatComplex* tau,
                                                                int*                   lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZungqr_bufferSize(hipsolverHandle_t       handle,
                                                                int                     m,
                                                                int                     n,
                                                                int                     k,
                                                                const hipDoubleComplex* A,
                                                                int                     lda,
                                                                const hipDoubleComplex* tau,
                                                                int*                    lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSorgqr(hipsolverHandle_t handle,
                                                     int               m,
                                                     int               n,
                                                     int               k,
                                                     float*            A,
                                                     int               lda,
                                                     const float*      tau,
                                                     float*            work,
                                                     int               lwork,
                                                     int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDorgqr(hipsolverHandle_t handle,
                                                     int               m,
                                                     int               n,
                                                     int               k,
                                                     double*           A,
                                                     int               lda,
                                                     const double*     tau,
                                                     double*           work,
                                                     int               lwork,
                                                     int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCungqr(hipsolverHandle_t      handle,
                                                     int                    m,
                                                     int                    n,
                                                     int                    k,
                                                     hipFloatComplex*       A,
                                                     int                    lda,
                                                     const hipFloatComplex* tau,
                                                     hipFloatComplex*       work,
                                                     int                    lwork,
                                                     int*                   devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZungqr(hipsolverHandle_t       handle,
                                                     int                     m,
                                                     int                     n,
                                                     int                     k,
                                                     hipDoubleComplex*       A,
                                                     int                     lda,
                                                     const hipDoubleComplex* tau,
                                                     hipDoubleComplex*       work,
                                                     int                     lwork,
                                                     int*                    devInfo);

// orgtr/ungtr
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSorgtr_bufferSize(hipsolverHandle_t handle,
                                                                hipblasFillMode_t uplo,
                                                                int               n,
                                                                const float*      A,
                                                                int               lda,
                                                                const float*      tau,
                                                                int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDorgtr_bufferSize(hipsolverHandle_t handle,
                                                                hipblasFillMode_t uplo,
                                                                int               n,
                                                                const double*     A,
                                                                int               lda,
                                                                const double*     tau,
                                                                int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCungtr_bufferSize(hipsolverHandle_t      handle,
                                                                hipblasFillMode_t      uplo,
                                                                int                    n,
                                                                const hipFloatComplex* A,
                                                                int                    lda,
                                                                const hipFloatComplex* tau,
                                                                int*                   lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZungtr_bufferSize(hipsolverHandle_t       handle,
                                                                hipblasFillMode_t       uplo,
                                                                int                     n,
                                                                const hipDoubleComplex* A,
                                                                int                     lda,
                                                                const hipDoubleComplex* tau,
                                                                int*                    lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSorgtr(hipsolverHandle_t handle,
                                                     hipblasFillMode_t uplo,
                                                     int               n,
                                                     float*            A,
                                                     int               lda,
                                                     const float*      tau,
                                                     float*            work,
                                                     int               lwork,
                                                     int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDorgtr(hipsolverHandle_t handle,
                                                     hipblasFillMode_t uplo,
                                                     int               n,
                                                     double*           A,
                                                     int               lda,
                                                     const double*     tau,
                                                     double*           work,
                                                     int               lwork,
                                                     int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCungtr(hipsolverHandle_t      handle,
                                                     hipblasFillMode_t      uplo,
                                                     int                    n,
                                                     hipFloatComplex*       A,
                                                     int                    lda,
                                                     const hipFloatComplex* tau,
                                                     hipFloatComplex*       work,
                                                     int                    lwork,
                                                     int*                   devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZungtr(hipsolverHandle_t       handle,
                                                     hipblasFillMode_t       uplo,
                                                     int                     n,
                                                     hipDoubleComplex*       A,
                                                     int                     lda,
                                                     const hipDoubleComplex* tau,
                                                     hipDoubleComplex*       work,
                                                     int                     lwork,
                                                     int*                    devInfo);

// ormqr/unmqr
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSormqr_bufferSize(hipsolverHandle_t  handle,
                                                                hipblasSideMode_t  side,
                                                                hipblasOperation_t trans,
                                                                int                m,
                                                                int                n,
                                                                int                k,
                                                                const float*       A,
                                                                int                lda,
                                                                const float*       tau,
                                                                const float*       C,
                                                                int                ldc,
                                                                int*               lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDormqr_bufferSize(hipsolverHandle_t  handle,
                                                                hipblasSideMode_t  side,
                                                                hipblasOperation_t trans,
                                                                int                m,
                                                                int                n,
                                                                int                k,
                                                                const double*      A,
                                                                int                lda,
                                                                const double*      tau,
                                                                const double*      C,
                                                                int                ldc,
                                                                int*               lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCunmqr_bufferSize(hipsolverHandle_t      handle,
                                                                hipblasSideMode_t      side,
                                                                hipblasOperation_t     trans,
                                                                int                    m,
                                                                int                    n,
                                                                int                    k,
                                                                const hipFloatComplex* A,
                                                                int                    lda,
                                                                const hipFloatComplex* tau,
                                                                const hipFloatComplex* C,
                                                                int                    ldc,
                                                                int*                   lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZunmqr_bufferSize(hipsolverHandle_t       handle,
                                                                hipblasSideMode_t       side,
                                                                hipblasOperation_t      trans,
                                                                int                     m,
                                                                int                     n,
                                                                int                     k,
                                                                const hipDoubleComplex* A,
                                                                int                     lda,
                                                                const hipDoubleComplex* tau,
                                                                const hipDoubleComplex* C,
                                                                int                     ldc,
                                                                int*                    lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSormqr(hipsolverHandle_t  handle,
                                                     hipblasSideMode_t  side,
                                                     hipblasOperation_t trans,
                                                     int                m,
                                                     int                n,
                                                     int                k,
                                                     const float*       A,
                                                     int                lda,
                                                     const float*       tau,
                                                     float*             C,
                                                     int                ldc,
                                                     float*             work,
                                                     int                lwork,
                                                     int*               devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDormqr(hipsolverHandle_t  handle,
                                                     hipblasSideMode_t  side,
                                                     hipblasOperation_t trans,
                                                     int                m,
                                                     int                n,
                                                     int                k,
                                                     const double*      A,
                                                     int                lda,
                                                     const double*      tau,
                                                     double*            C,
                                                     int                ldc,
                                                     double*            work,
                                                     int                lwork,
                                                     int*               devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCunmqr(hipsolverHandle_t      handle,
                                                     hipblasSideMode_t      side,
                                                     hipblasOperation_t     trans,
                                                     int                    m,
                                                     int                    n,
                                                     int                    k,
                                                     const hipFloatComplex* A,
                                                     int                    lda,
                                                     const hipFloatComplex* tau,
                                                     hipFloatComplex*       C,
                                                     int                    ldc,
                                                     hipFloatComplex*       work,
                                                     int                    lwork,
                                                     int*                   devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZunmqr(hipsolverHandle_t       handle,
                                                     hipblasSideMode_t       side,
                                                     hipblasOperation_t      trans,
                                                     int                     m,
                                                     int                     n,
                                                     int                     k,
                                                     const hipDoubleComplex* A,
                                                     int                     lda,
                                                     const hipDoubleComplex* tau,
                                                     hipDoubleComplex*       C,
                                                     int                     ldc,
                                                     hipDoubleComplex*       work,
                                                     int                     lwork,
                                                     int*                    devInfo);

// ormtr/unmtr
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSormtr_bufferSize(hipsolverHandle_t  handle,
                                                                hipblasSideMode_t  side,
                                                                hipblasFillMode_t  uplo,
                                                                hipblasOperation_t trans,
                                                                int                m,
                                                                int                n,
                                                                const float*       A,
                                                                int                lda,
                                                                const float*       tau,
                                                                const float*       C,
                                                                int                ldc,
                                                                int*               lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDormtr_bufferSize(hipsolverHandle_t  handle,
                                                                hipblasSideMode_t  side,
                                                                hipblasFillMode_t  uplo,
                                                                hipblasOperation_t trans,
                                                                int                m,
                                                                int                n,
                                                                const double*      A,
                                                                int                lda,
                                                                const double*      tau,
                                                                const double*      C,
                                                                int                ldc,
                                                                int*               lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCunmtr_bufferSize(hipsolverHandle_t      handle,
                                                                hipblasSideMode_t      side,
                                                                hipblasFillMode_t      uplo,
                                                                hipblasOperation_t     trans,
                                                                int                    m,
                                                                int                    n,
                                                                const hipFloatComplex* A,
                                                                int                    lda,
                                                                const hipFloatComplex* tau,
                                                                const hipFloatComplex* C,
                                                                int                    ldc,
                                                                int*                   lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZunmtr_bufferSize(hipsolverHandle_t       handle,
                                                                hipblasSideMode_t       side,
                                                                hipblasFillMode_t       uplo,
                                                                hipblasOperation_t      trans,
                                                                int                     m,
                                                                int                     n,
                                                                const hipDoubleComplex* A,
                                                                int                     lda,
                                                                const hipDoubleComplex* tau,
                                                                const hipDoubleComplex* C,
                                                                int                     ldc,
                                                                int*                    lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSormtr(hipsolverHandle_t  handle,
                                                     hipblasSideMode_t  side,
                                                     hipblasFillMode_t  uplo,
                                                     hipblasOperation_t trans,
                                                     int                m,
                                                     int                n,
                                                     float*             A,
                                                     int                lda,
                                                     float*             tau,
                                                     float*             C,
                                                     int                ldc,
                                                     float*             work,
                                                     int                lwork,
                                                     int*               devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDormtr(hipsolverHandle_t  handle,
                                                     hipblasSideMode_t  side,
                                                     hipblasFillMode_t  uplo,
                                                     hipblasOperation_t trans,
                                                     int                m,
                                                     int                n,
                                                     double*            A,
                                                     int                lda,
                                                     double*            tau,
                                                     double*            C,
                                                     int                ldc,
                                                     double*            work,
                                                     int                lwork,
                                                     int*               devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCunmtr(hipsolverHandle_t  handle,
                                                     hipblasSideMode_t  side,
                                                     hipblasFillMode_t  uplo,
                                                     hipblasOperation_t trans,
                                                     int                m,
                                                     int                n,
                                                     hipFloatComplex*   A,
                                                     int                lda,
                                                     hipFloatComplex*   tau,
                                                     hipFloatComplex*   C,
                                                     int                ldc,
                                                     hipFloatComplex*   work,
                                                     int                lwork,
                                                     int*               devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZunmtr(hipsolverHandle_t  handle,
                                                     hipblasSideMode_t  side,
                                                     hipblasFillMode_t  uplo,
                                                     hipblasOperation_t trans,
                                                     int                m,
                                                     int                n,
                                                     hipDoubleComplex*  A,
                                                     int                lda,
                                                     hipDoubleComplex*  tau,
                                                     hipDoubleComplex*  C,
                                                     int                ldc,
                                                     hipDoubleComplex*  work,
                                                     int                lwork,
                                                     int*               devInfo);

// gebrd
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgebrd_bufferSize(hipsolverHandle_t handle,
                                                                int               m,
                                                                int               n,
                                                                int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgebrd_bufferSize(hipsolverHandle_t handle,
                                                                int               m,
                                                                int               n,
                                                                int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgebrd_bufferSize(hipsolverHandle_t handle,
                                                                int               m,
                                                                int               n,
                                                                int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgebrd_bufferSize(hipsolverHandle_t handle,
                                                                int               m,
                                                                int               n,
                                                                int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgebrd(hipsolverHandle_t handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgebrd(hipsolverHandle_t handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgebrd(hipsolverHandle_t handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgebrd(hipsolverHandle_t handle,
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
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSSgels_bufferSize(hipsolverHandle_t handle,
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
                                                                size_t*           lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDDgels_bufferSize(hipsolverHandle_t handle,
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
                                                                size_t*           lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCCgels_bufferSize(hipsolverHandle_t handle,
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
                                                                size_t*           lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZZgels_bufferSize(hipsolverHandle_t handle,
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
                                                                size_t*           lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSSgels(hipsolverHandle_t handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDDgels(hipsolverHandle_t handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCCgels(hipsolverHandle_t handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZZgels(hipsolverHandle_t handle,
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
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipFloatComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgeqrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipDoubleComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgeqrf(hipsolverHandle_t handle,
                                                     int               m,
                                                     int               n,
                                                     float*            A,
                                                     int               lda,
                                                     float*            tau,
                                                     float*            work,
                                                     int               lwork,
                                                     int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgeqrf(hipsolverHandle_t handle,
                                                     int               m,
                                                     int               n,
                                                     double*           A,
                                                     int               lda,
                                                     double*           tau,
                                                     double*           work,
                                                     int               lwork,
                                                     int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgeqrf(hipsolverHandle_t handle,
                                                     int               m,
                                                     int               n,
                                                     hipFloatComplex*  A,
                                                     int               lda,
                                                     hipFloatComplex*  tau,
                                                     hipFloatComplex*  work,
                                                     int               lwork,
                                                     int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgeqrf(hipsolverHandle_t handle,
                                                     int               m,
                                                     int               n,
                                                     hipDoubleComplex* A,
                                                     int               lda,
                                                     hipDoubleComplex* tau,
                                                     hipDoubleComplex* work,
                                                     int               lwork,
                                                     int*              devInfo);

// gesv
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSSgesv_bufferSize(hipsolverHandle_t handle,
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
                                                                size_t*           lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDDgesv_bufferSize(hipsolverHandle_t handle,
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
                                                                size_t*           lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCCgesv_bufferSize(hipsolverHandle_t handle,
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
                                                                size_t*           lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZZgesv_bufferSize(hipsolverHandle_t handle,
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
                                                                size_t*           lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSSgesv(hipsolverHandle_t handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDDgesv(hipsolverHandle_t handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCCgesv(hipsolverHandle_t handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZZgesv(hipsolverHandle_t handle,
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
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgesvd_bufferSize(hipsolverHandle_t handle,
                                                                int               m,
                                                                int               n,
                                                                int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgesvd_bufferSize(hipsolverHandle_t handle,
                                                                int               m,
                                                                int               n,
                                                                int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgesvd_bufferSize(hipsolverHandle_t handle,
                                                                int               m,
                                                                int               n,
                                                                int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgesvd_bufferSize(hipsolverHandle_t handle,
                                                                int               m,
                                                                int               n,
                                                                int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgesvd(hipsolverHandle_t handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgesvd(hipsolverHandle_t handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgesvd(hipsolverHandle_t handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgesvd(hipsolverHandle_t handle,
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
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgesvdj_bufferSize(hipsolverDnHandle_t   handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgesvdj_bufferSize(hipsolverDnHandle_t   handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgesvdj_bufferSize(hipsolverDnHandle_t    handle,
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
                                                                 hipsolverGesvdjInfo_t  params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgesvdj_bufferSize(hipsolverDnHandle_t     handle,
                                                                 hipsolverEigMode_t      jobz,
                                                                 int                     econ,
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
                                                                 hipsolverGesvdjInfo_t   params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgesvdj(hipsolverDnHandle_t   handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgesvdj(hipsolverDnHandle_t   handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgesvdj(hipsolverDnHandle_t   handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgesvdj(hipsolverDnHandle_t   handle,
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
    hipsolverDnSgesvdjBatched_bufferSize(hipsolverDnHandle_t   handle,
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
    hipsolverDnDgesvdjBatched_bufferSize(hipsolverDnHandle_t   handle,
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
    hipsolverDnCgesvdjBatched_bufferSize(hipsolverDnHandle_t    handle,
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
    hipsolverDnZgesvdjBatched_bufferSize(hipsolverDnHandle_t     handle,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgesvdjBatched(hipsolverDnHandle_t   handle,
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
                                                             int                   batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgesvdjBatched(hipsolverDnHandle_t   handle,
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
                                                             int                   batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgesvdjBatched(hipsolverDnHandle_t   handle,
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
                                                             int                   batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgesvdjBatched(hipsolverDnHandle_t   handle,
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
                                                             int                   batch_count);

// gesvda_strided_batched
HIPSOLVER_EXPORT hipsolverStatus_t
    hipsolverDnSgesvdaStridedBatched_bufferSize(hipsolverHandle_t  handle,
                                                hipsolverEigMode_t jobz,
                                                int                rank,
                                                int                m,
                                                int                n,
                                                const float*       A,
                                                int                lda,
                                                long long int      strideA,
                                                const float*       S,
                                                long long int      strideS,
                                                const float*       U,
                                                int                ldu,
                                                long long int      strideU,
                                                const float*       V,
                                                int                ldv,
                                                long long int      strideV,
                                                int*               lwork,
                                                int                batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t
    hipsolverDnDgesvdaStridedBatched_bufferSize(hipsolverHandle_t  handle,
                                                hipsolverEigMode_t jobz,
                                                int                rank,
                                                int                m,
                                                int                n,
                                                const double*      A,
                                                int                lda,
                                                long long int      strideA,
                                                const double*      S,
                                                long long int      strideS,
                                                const double*      U,
                                                int                ldu,
                                                long long int      strideU,
                                                const double*      V,
                                                int                ldv,
                                                long long int      strideV,
                                                int*               lwork,
                                                int                batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t
    hipsolverDnCgesvdaStridedBatched_bufferSize(hipsolverHandle_t      handle,
                                                hipsolverEigMode_t     jobz,
                                                int                    rank,
                                                int                    m,
                                                int                    n,
                                                const hipFloatComplex* A,
                                                int                    lda,
                                                long long int          strideA,
                                                const float*           S,
                                                long long int          strideS,
                                                const hipFloatComplex* U,
                                                int                    ldu,
                                                long long int          strideU,
                                                const hipFloatComplex* V,
                                                int                    ldv,
                                                long long int          strideV,
                                                int*                   lwork,
                                                int                    batch_count);
HIPSOLVER_EXPORT hipsolverStatus_t
    hipsolverDnZgesvdaStridedBatched_bufferSize(hipsolverHandle_t       handle,
                                                hipsolverEigMode_t      jobz,
                                                int                     rank,
                                                int                     m,
                                                int                     n,
                                                const hipDoubleComplex* A,
                                                int                     lda,
                                                long long int           strideA,
                                                const double*           S,
                                                long long int           strideS,
                                                const hipDoubleComplex* U,
                                                int                     ldu,
                                                long long int           strideU,
                                                const hipDoubleComplex* V,
                                                int                     ldv,
                                                long long int           strideV,
                                                int*                    lwork,
                                                int                     batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgesvdaStridedBatched(hipsolverHandle_t  handle,
                                                                    hipsolverEigMode_t jobz,
                                                                    int                rank,
                                                                    int                m,
                                                                    int                n,
                                                                    const float*       A,
                                                                    int                lda,
                                                                    long long int      strideA,
                                                                    float*             S,
                                                                    long long int      strideS,
                                                                    float*             U,
                                                                    int                ldu,
                                                                    long long int      strideU,
                                                                    float*             V,
                                                                    int                ldv,
                                                                    long long int      strideV,
                                                                    float*             work,
                                                                    int                lwork,
                                                                    int*               devInfo,
                                                                    double*            hRnrmF,
                                                                    int                batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgesvdaStridedBatched(hipsolverHandle_t  handle,
                                                                    hipsolverEigMode_t jobz,
                                                                    int                rank,
                                                                    int                m,
                                                                    int                n,
                                                                    const double*      A,
                                                                    int                lda,
                                                                    long long int      strideA,
                                                                    double*            S,
                                                                    long long int      strideS,
                                                                    double*            U,
                                                                    int                ldu,
                                                                    long long int      strideU,
                                                                    double*            V,
                                                                    int                ldv,
                                                                    long long int      strideV,
                                                                    double*            work,
                                                                    int                lwork,
                                                                    int*               devInfo,
                                                                    double*            hRnrmF,
                                                                    int                batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgesvdaStridedBatched(hipsolverHandle_t      handle,
                                                                    hipsolverEigMode_t     jobz,
                                                                    int                    rank,
                                                                    int                    m,
                                                                    int                    n,
                                                                    const hipFloatComplex* A,
                                                                    int                    lda,
                                                                    long long int          strideA,
                                                                    float*                 S,
                                                                    long long int          strideS,
                                                                    hipFloatComplex*       U,
                                                                    int                    ldu,
                                                                    long long int          strideU,
                                                                    hipFloatComplex*       V,
                                                                    int                    ldv,
                                                                    long long int          strideV,
                                                                    hipFloatComplex*       work,
                                                                    int                    lwork,
                                                                    int*                   devInfo,
                                                                    double*                hRnrmF,
                                                                    int batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgesvdaStridedBatched(hipsolverHandle_t       handle,
                                                                    hipsolverEigMode_t      jobz,
                                                                    int                     rank,
                                                                    int                     m,
                                                                    int                     n,
                                                                    const hipDoubleComplex* A,
                                                                    int                     lda,
                                                                    long long int           strideA,
                                                                    double*                 S,
                                                                    long long int           strideS,
                                                                    hipDoubleComplex*       U,
                                                                    int                     ldu,
                                                                    long long int           strideU,
                                                                    hipDoubleComplex*       V,
                                                                    int                     ldv,
                                                                    long long int           strideV,
                                                                    hipDoubleComplex*       work,
                                                                    int                     lwork,
                                                                    int*                    devInfo,
                                                                    double*                 hRnrmF,
                                                                    int batch_count);

// getrf
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, float* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, double* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipFloatComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgetrf_bufferSize(
    hipsolverHandle_t handle, int m, int n, hipDoubleComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgetrf(hipsolverHandle_t handle,
                                                     int               m,
                                                     int               n,
                                                     float*            A,
                                                     int               lda,
                                                     float*            work,
                                                     int*              devIpiv,
                                                     int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgetrf(hipsolverHandle_t handle,
                                                     int               m,
                                                     int               n,
                                                     double*           A,
                                                     int               lda,
                                                     double*           work,
                                                     int*              devIpiv,
                                                     int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgetrf(hipsolverHandle_t handle,
                                                     int               m,
                                                     int               n,
                                                     hipFloatComplex*  A,
                                                     int               lda,
                                                     hipFloatComplex*  work,
                                                     int*              devIpiv,
                                                     int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgetrf(hipsolverHandle_t handle,
                                                     int               m,
                                                     int               n,
                                                     hipDoubleComplex* A,
                                                     int               lda,
                                                     hipDoubleComplex* work,
                                                     int*              devIpiv,
                                                     int*              devInfo);

// getrs
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSgetrs(hipsolverHandle_t  handle,
                                                     hipblasOperation_t trans,
                                                     int                n,
                                                     int                nrhs,
                                                     const float*       A,
                                                     int                lda,
                                                     const int*         devIpiv,
                                                     float*             B,
                                                     int                ldb,
                                                     int*               devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDgetrs(hipsolverHandle_t  handle,
                                                     hipblasOperation_t trans,
                                                     int                n,
                                                     int                nrhs,
                                                     const double*      A,
                                                     int                lda,
                                                     const int*         devIpiv,
                                                     double*            B,
                                                     int                ldb,
                                                     int*               devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCgetrs(hipsolverHandle_t      handle,
                                                     hipblasOperation_t     trans,
                                                     int                    n,
                                                     int                    nrhs,
                                                     const hipFloatComplex* A,
                                                     int                    lda,
                                                     const int*             devIpiv,
                                                     hipFloatComplex*       B,
                                                     int                    ldb,
                                                     int*                   devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZgetrs(hipsolverHandle_t       handle,
                                                     hipblasOperation_t      trans,
                                                     int                     n,
                                                     int                     nrhs,
                                                     const hipDoubleComplex* A,
                                                     int                     lda,
                                                     const int*              devIpiv,
                                                     hipDoubleComplex*       B,
                                                     int                     ldb,
                                                     int*                    devInfo);

// potrf
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSpotrf_bufferSize(
    hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, float* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDpotrf_bufferSize(
    hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, double* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCpotrf_bufferSize(hipsolverHandle_t handle,
                                                                hipblasFillMode_t uplo,
                                                                int               n,
                                                                hipFloatComplex*  A,
                                                                int               lda,
                                                                int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZpotrf_bufferSize(hipsolverHandle_t handle,
                                                                hipblasFillMode_t uplo,
                                                                int               n,
                                                                hipDoubleComplex* A,
                                                                int               lda,
                                                                int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSpotrf(hipsolverHandle_t handle,
                                                     hipblasFillMode_t uplo,
                                                     int               n,
                                                     float*            A,
                                                     int               lda,
                                                     float*            work,
                                                     int               lwork,
                                                     int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDpotrf(hipsolverHandle_t handle,
                                                     hipblasFillMode_t uplo,
                                                     int               n,
                                                     double*           A,
                                                     int               lda,
                                                     double*           work,
                                                     int               lwork,
                                                     int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCpotrf(hipsolverHandle_t handle,
                                                     hipblasFillMode_t uplo,
                                                     int               n,
                                                     hipFloatComplex*  A,
                                                     int               lda,
                                                     hipFloatComplex*  work,
                                                     int               lwork,
                                                     int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZpotrf(hipsolverHandle_t handle,
                                                     hipblasFillMode_t uplo,
                                                     int               n,
                                                     hipDoubleComplex* A,
                                                     int               lda,
                                                     hipDoubleComplex* work,
                                                     int               lwork,
                                                     int*              devInfo);

// potrf_batched
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSpotrfBatched(hipsolverHandle_t handle,
                                                            hipblasFillMode_t uplo,
                                                            int               n,
                                                            float*            A[],
                                                            int               lda,
                                                            int*              devInfo,
                                                            int               batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDpotrfBatched(hipsolverHandle_t handle,
                                                            hipblasFillMode_t uplo,
                                                            int               n,
                                                            double*           A[],
                                                            int               lda,
                                                            int*              devInfo,
                                                            int               batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCpotrfBatched(hipsolverHandle_t handle,
                                                            hipblasFillMode_t uplo,
                                                            int               n,
                                                            hipFloatComplex*  A[],
                                                            int               lda,
                                                            int*              devInfo,
                                                            int               batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZpotrfBatched(hipsolverHandle_t handle,
                                                            hipblasFillMode_t uplo,
                                                            int               n,
                                                            hipDoubleComplex* A[],
                                                            int               lda,
                                                            int*              devInfo,
                                                            int               batch_count);

// potri
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSpotri_bufferSize(
    hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, float* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDpotri_bufferSize(
    hipsolverHandle_t handle, hipblasFillMode_t uplo, int n, double* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCpotri_bufferSize(hipsolverHandle_t handle,
                                                                hipblasFillMode_t uplo,
                                                                int               n,
                                                                hipFloatComplex*  A,
                                                                int               lda,
                                                                int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZpotri_bufferSize(hipsolverHandle_t handle,
                                                                hipblasFillMode_t uplo,
                                                                int               n,
                                                                hipDoubleComplex* A,
                                                                int               lda,
                                                                int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSpotri(hipsolverHandle_t handle,
                                                     hipblasFillMode_t uplo,
                                                     int               n,
                                                     float*            A,
                                                     int               lda,
                                                     float*            work,
                                                     int               lwork,
                                                     int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDpotri(hipsolverHandle_t handle,
                                                     hipblasFillMode_t uplo,
                                                     int               n,
                                                     double*           A,
                                                     int               lda,
                                                     double*           work,
                                                     int               lwork,
                                                     int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCpotri(hipsolverHandle_t handle,
                                                     hipblasFillMode_t uplo,
                                                     int               n,
                                                     hipFloatComplex*  A,
                                                     int               lda,
                                                     hipFloatComplex*  work,
                                                     int               lwork,
                                                     int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZpotri(hipsolverHandle_t handle,
                                                     hipblasFillMode_t uplo,
                                                     int               n,
                                                     hipDoubleComplex* A,
                                                     int               lda,
                                                     hipDoubleComplex* work,
                                                     int               lwork,
                                                     int*              devInfo);

// potrs
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSpotrs(hipsolverHandle_t handle,
                                                     hipblasFillMode_t uplo,
                                                     int               n,
                                                     int               nrhs,
                                                     const float*      A,
                                                     int               lda,
                                                     float*            B,
                                                     int               ldb,
                                                     int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDpotrs(hipsolverHandle_t handle,
                                                     hipblasFillMode_t uplo,
                                                     int               n,
                                                     int               nrhs,
                                                     const double*     A,
                                                     int               lda,
                                                     double*           B,
                                                     int               ldb,
                                                     int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCpotrs(hipsolverHandle_t      handle,
                                                     hipblasFillMode_t      uplo,
                                                     int                    n,
                                                     int                    nrhs,
                                                     const hipFloatComplex* A,
                                                     int                    lda,
                                                     hipFloatComplex*       B,
                                                     int                    ldb,
                                                     int*                   devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZpotrs(hipsolverHandle_t       handle,
                                                     hipblasFillMode_t       uplo,
                                                     int                     n,
                                                     int                     nrhs,
                                                     const hipDoubleComplex* A,
                                                     int                     lda,
                                                     hipDoubleComplex*       B,
                                                     int                     ldb,
                                                     int*                    devInfo);

// potrs_batched
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSpotrsBatched(hipsolverHandle_t handle,
                                                            hipblasFillMode_t uplo,
                                                            int               n,
                                                            int               nrhs,
                                                            float*            A[],
                                                            int               lda,
                                                            float*            B[],
                                                            int               ldb,
                                                            int*              devInfo,
                                                            int               batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDpotrsBatched(hipsolverHandle_t handle,
                                                            hipblasFillMode_t uplo,
                                                            int               n,
                                                            int               nrhs,
                                                            double*           A[],
                                                            int               lda,
                                                            double*           B[],
                                                            int               ldb,
                                                            int*              devInfo,
                                                            int               batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCpotrsBatched(hipsolverHandle_t handle,
                                                            hipblasFillMode_t uplo,
                                                            int               n,
                                                            int               nrhs,
                                                            hipFloatComplex*  A[],
                                                            int               lda,
                                                            hipFloatComplex*  B[],
                                                            int               ldb,
                                                            int*              devInfo,
                                                            int               batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZpotrsBatched(hipsolverHandle_t handle,
                                                            hipblasFillMode_t uplo,
                                                            int               n,
                                                            int               nrhs,
                                                            hipDoubleComplex* A[],
                                                            int               lda,
                                                            hipDoubleComplex* B[],
                                                            int               ldb,
                                                            int*              devInfo,
                                                            int               batch_count);

// syevd/heevd
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsyevd_bufferSize(hipsolverHandle_t  handle,
                                                                hipsolverEigMode_t jobz,
                                                                hipblasFillMode_t  uplo,
                                                                int                n,
                                                                const float*       A,
                                                                int                lda,
                                                                const float*       W,
                                                                int*               lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsyevd_bufferSize(hipsolverHandle_t  handle,
                                                                hipsolverEigMode_t jobz,
                                                                hipblasFillMode_t  uplo,
                                                                int                n,
                                                                const double*      A,
                                                                int                lda,
                                                                const double*      W,
                                                                int*               lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCheevd_bufferSize(hipsolverHandle_t      handle,
                                                                hipsolverEigMode_t     jobz,
                                                                hipblasFillMode_t      uplo,
                                                                int                    n,
                                                                const hipFloatComplex* A,
                                                                int                    lda,
                                                                const float*           W,
                                                                int*                   lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZheevd_bufferSize(hipsolverHandle_t       handle,
                                                                hipsolverEigMode_t      jobz,
                                                                hipblasFillMode_t       uplo,
                                                                int                     n,
                                                                const hipDoubleComplex* A,
                                                                int                     lda,
                                                                const double*           W,
                                                                int*                    lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsyevd(hipsolverHandle_t  handle,
                                                     hipsolverEigMode_t jobz,
                                                     hipblasFillMode_t  uplo,
                                                     int                n,
                                                     float*             A,
                                                     int                lda,
                                                     float*             W,
                                                     float*             work,
                                                     int                lwork,
                                                     int*               devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsyevd(hipsolverHandle_t  handle,
                                                     hipsolverEigMode_t jobz,
                                                     hipblasFillMode_t  uplo,
                                                     int                n,
                                                     double*            A,
                                                     int                lda,
                                                     double*            W,
                                                     double*            work,
                                                     int                lwork,
                                                     int*               devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCheevd(hipsolverHandle_t  handle,
                                                     hipsolverEigMode_t jobz,
                                                     hipblasFillMode_t  uplo,
                                                     int                n,
                                                     hipFloatComplex*   A,
                                                     int                lda,
                                                     float*             W,
                                                     hipFloatComplex*   work,
                                                     int                lwork,
                                                     int*               devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZheevd(hipsolverHandle_t  handle,
                                                     hipsolverEigMode_t jobz,
                                                     hipblasFillMode_t  uplo,
                                                     int                n,
                                                     hipDoubleComplex*  A,
                                                     int                lda,
                                                     double*            W,
                                                     hipDoubleComplex*  work,
                                                     int                lwork,
                                                     int*               devInfo);

// syevdx/heevdx
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsyevdx_bufferSize(hipsolverHandle_t   handle,
                                                                 hipsolverEigMode_t  jobz,
                                                                 hipsolverEigRange_t range,
                                                                 hipblasFillMode_t   uplo,
                                                                 int                 n,
                                                                 const float*        A,
                                                                 int                 lda,
                                                                 float               vl,
                                                                 float               vu,
                                                                 int                 il,
                                                                 int                 iu,
                                                                 int*                nev,
                                                                 const float*        W,
                                                                 int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsyevdx_bufferSize(hipsolverHandle_t   handle,
                                                                 hipsolverEigMode_t  jobz,
                                                                 hipsolverEigRange_t range,
                                                                 hipblasFillMode_t   uplo,
                                                                 int                 n,
                                                                 const double*       A,
                                                                 int                 lda,
                                                                 double              vl,
                                                                 double              vu,
                                                                 int                 il,
                                                                 int                 iu,
                                                                 int*                nev,
                                                                 const double*       W,
                                                                 int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCheevdx_bufferSize(hipsolverHandle_t      handle,
                                                                 hipsolverEigMode_t     jobz,
                                                                 hipsolverEigRange_t    range,
                                                                 hipblasFillMode_t      uplo,
                                                                 int                    n,
                                                                 const hipFloatComplex* A,
                                                                 int                    lda,
                                                                 float                  vl,
                                                                 float                  vu,
                                                                 int                    il,
                                                                 int                    iu,
                                                                 int*                   nev,
                                                                 const float*           W,
                                                                 int*                   lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZheevdx_bufferSize(hipsolverHandle_t       handle,
                                                                 hipsolverEigMode_t      jobz,
                                                                 hipsolverEigRange_t     range,
                                                                 hipblasFillMode_t       uplo,
                                                                 int                     n,
                                                                 const hipDoubleComplex* A,
                                                                 int                     lda,
                                                                 double                  vl,
                                                                 double                  vu,
                                                                 int                     il,
                                                                 int                     iu,
                                                                 int*                    nev,
                                                                 const double*           W,
                                                                 int*                    lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsyevdx(hipsolverHandle_t   handle,
                                                      hipsolverEigMode_t  jobz,
                                                      hipsolverEigRange_t range,
                                                      hipblasFillMode_t   uplo,
                                                      int                 n,
                                                      float*              A,
                                                      int                 lda,
                                                      float               vl,
                                                      float               vu,
                                                      int                 il,
                                                      int                 iu,
                                                      int*                nev,
                                                      float*              W,
                                                      float*              work,
                                                      int                 lwork,
                                                      int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsyevdx(hipsolverHandle_t   handle,
                                                      hipsolverEigMode_t  jobz,
                                                      hipsolverEigRange_t range,
                                                      hipblasFillMode_t   uplo,
                                                      int                 n,
                                                      double*             A,
                                                      int                 lda,
                                                      double              vl,
                                                      double              vu,
                                                      int                 il,
                                                      int                 iu,
                                                      int*                nev,
                                                      double*             W,
                                                      double*             work,
                                                      int                 lwork,
                                                      int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCheevdx(hipsolverHandle_t   handle,
                                                      hipsolverEigMode_t  jobz,
                                                      hipsolverEigRange_t range,
                                                      hipblasFillMode_t   uplo,
                                                      int                 n,
                                                      hipFloatComplex*    A,
                                                      int                 lda,
                                                      float               vl,
                                                      float               vu,
                                                      int                 il,
                                                      int                 iu,
                                                      int*                nev,
                                                      float*              W,
                                                      hipFloatComplex*    work,
                                                      int                 lwork,
                                                      int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZheevdx(hipsolverHandle_t   handle,
                                                      hipsolverEigMode_t  jobz,
                                                      hipsolverEigRange_t range,
                                                      hipblasFillMode_t   uplo,
                                                      int                 n,
                                                      hipDoubleComplex*   A,
                                                      int                 lda,
                                                      double              vl,
                                                      double              vu,
                                                      int                 il,
                                                      int                 iu,
                                                      int*                nev,
                                                      double*             W,
                                                      hipDoubleComplex*   work,
                                                      int                 lwork,
                                                      int*                devInfo);

// syevj/heevj
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsyevj_bufferSize(hipsolverDnHandle_t  handle,
                                                                hipsolverEigMode_t   jobz,
                                                                hipblasFillMode_t    uplo,
                                                                int                  n,
                                                                const float*         A,
                                                                int                  lda,
                                                                const float*         W,
                                                                int*                 lwork,
                                                                hipsolverSyevjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsyevj_bufferSize(hipsolverDnHandle_t  handle,
                                                                hipsolverEigMode_t   jobz,
                                                                hipblasFillMode_t    uplo,
                                                                int                  n,
                                                                const double*        A,
                                                                int                  lda,
                                                                const double*        W,
                                                                int*                 lwork,
                                                                hipsolverSyevjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCheevj_bufferSize(hipsolverDnHandle_t    handle,
                                                                hipsolverEigMode_t     jobz,
                                                                hipblasFillMode_t      uplo,
                                                                int                    n,
                                                                const hipFloatComplex* A,
                                                                int                    lda,
                                                                const float*           W,
                                                                int*                   lwork,
                                                                hipsolverSyevjInfo_t   params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZheevj_bufferSize(hipsolverDnHandle_t     handle,
                                                                hipsolverEigMode_t      jobz,
                                                                hipblasFillMode_t       uplo,
                                                                int                     n,
                                                                const hipDoubleComplex* A,
                                                                int                     lda,
                                                                const double*           W,
                                                                int*                    lwork,
                                                                hipsolverSyevjInfo_t    params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsyevj(hipsolverDnHandle_t  handle,
                                                     hipsolverEigMode_t   jobz,
                                                     hipblasFillMode_t    uplo,
                                                     int                  n,
                                                     float*               A,
                                                     int                  lda,
                                                     float*               W,
                                                     float*               work,
                                                     int                  lwork,
                                                     int*                 devInfo,
                                                     hipsolverSyevjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsyevj(hipsolverDnHandle_t  handle,
                                                     hipsolverEigMode_t   jobz,
                                                     hipblasFillMode_t    uplo,
                                                     int                  n,
                                                     double*              A,
                                                     int                  lda,
                                                     double*              W,
                                                     double*              work,
                                                     int                  lwork,
                                                     int*                 devInfo,
                                                     hipsolverSyevjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCheevj(hipsolverDnHandle_t  handle,
                                                     hipsolverEigMode_t   jobz,
                                                     hipblasFillMode_t    uplo,
                                                     int                  n,
                                                     hipFloatComplex*     A,
                                                     int                  lda,
                                                     float*               W,
                                                     hipFloatComplex*     work,
                                                     int                  lwork,
                                                     int*                 devInfo,
                                                     hipsolverSyevjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZheevj(hipsolverDnHandle_t  handle,
                                                     hipsolverEigMode_t   jobz,
                                                     hipblasFillMode_t    uplo,
                                                     int                  n,
                                                     hipDoubleComplex*    A,
                                                     int                  lda,
                                                     double*              W,
                                                     hipDoubleComplex*    work,
                                                     int                  lwork,
                                                     int*                 devInfo,
                                                     hipsolverSyevjInfo_t params);

// syevj_batched/heevj_batched
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsyevjBatched_bufferSize(hipsolverDnHandle_t  handle,
                                                                       hipsolverEigMode_t   jobz,
                                                                       hipblasFillMode_t    uplo,
                                                                       int                  n,
                                                                       const float*         A,
                                                                       int                  lda,
                                                                       const float*         W,
                                                                       int*                 lwork,
                                                                       hipsolverSyevjInfo_t params,
                                                                       int batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsyevjBatched_bufferSize(hipsolverDnHandle_t  handle,
                                                                       hipsolverEigMode_t   jobz,
                                                                       hipblasFillMode_t    uplo,
                                                                       int                  n,
                                                                       const double*        A,
                                                                       int                  lda,
                                                                       const double*        W,
                                                                       int*                 lwork,
                                                                       hipsolverSyevjInfo_t params,
                                                                       int batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCheevjBatched_bufferSize(hipsolverDnHandle_t handle,
                                                                       hipsolverEigMode_t  jobz,
                                                                       hipblasFillMode_t   uplo,
                                                                       int                 n,
                                                                       const hipFloatComplex* A,
                                                                       int                    lda,
                                                                       const float*           W,
                                                                       int*                   lwork,
                                                                       hipsolverSyevjInfo_t params,
                                                                       int batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZheevjBatched_bufferSize(hipsolverDnHandle_t handle,
                                                                       hipsolverEigMode_t  jobz,
                                                                       hipblasFillMode_t   uplo,
                                                                       int                 n,
                                                                       const hipDoubleComplex* A,
                                                                       int                     lda,
                                                                       const double*           W,
                                                                       int*                 lwork,
                                                                       hipsolverSyevjInfo_t params,
                                                                       int batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsyevjBatched(hipsolverDnHandle_t  handle,
                                                            hipsolverEigMode_t   jobz,
                                                            hipblasFillMode_t    uplo,
                                                            int                  n,
                                                            float*               A,
                                                            int                  lda,
                                                            float*               W,
                                                            float*               work,
                                                            int                  lwork,
                                                            int*                 devInfo,
                                                            hipsolverSyevjInfo_t params,
                                                            int                  batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsyevjBatched(hipsolverDnHandle_t  handle,
                                                            hipsolverEigMode_t   jobz,
                                                            hipblasFillMode_t    uplo,
                                                            int                  n,
                                                            double*              A,
                                                            int                  lda,
                                                            double*              W,
                                                            double*              work,
                                                            int                  lwork,
                                                            int*                 devInfo,
                                                            hipsolverSyevjInfo_t params,
                                                            int                  batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCheevjBatched(hipsolverDnHandle_t  handle,
                                                            hipsolverEigMode_t   jobz,
                                                            hipblasFillMode_t    uplo,
                                                            int                  n,
                                                            hipFloatComplex*     A,
                                                            int                  lda,
                                                            float*               W,
                                                            hipFloatComplex*     work,
                                                            int                  lwork,
                                                            int*                 devInfo,
                                                            hipsolverSyevjInfo_t params,
                                                            int                  batch_count);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZheevjBatched(hipsolverDnHandle_t  handle,
                                                            hipsolverEigMode_t   jobz,
                                                            hipblasFillMode_t    uplo,
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
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsygvd_bufferSize(hipsolverHandle_t  handle,
                                                                hipsolverEigType_t itype,
                                                                hipsolverEigMode_t jobz,
                                                                hipblasFillMode_t  uplo,
                                                                int                n,
                                                                const float*       A,
                                                                int                lda,
                                                                const float*       B,
                                                                int                ldb,
                                                                const float*       W,
                                                                int*               lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsygvd_bufferSize(hipsolverHandle_t  handle,
                                                                hipsolverEigType_t itype,
                                                                hipsolverEigMode_t jobz,
                                                                hipblasFillMode_t  uplo,
                                                                int                n,
                                                                const double*      A,
                                                                int                lda,
                                                                const double*      B,
                                                                int                ldb,
                                                                const double*      W,
                                                                int*               lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnChegvd_bufferSize(hipsolverHandle_t      handle,
                                                                hipsolverEigType_t     itype,
                                                                hipsolverEigMode_t     jobz,
                                                                hipblasFillMode_t      uplo,
                                                                int                    n,
                                                                const hipFloatComplex* A,
                                                                int                    lda,
                                                                const hipFloatComplex* B,
                                                                int                    ldb,
                                                                const float*           W,
                                                                int*                   lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZhegvd_bufferSize(hipsolverHandle_t       handle,
                                                                hipsolverEigType_t      itype,
                                                                hipsolverEigMode_t      jobz,
                                                                hipblasFillMode_t       uplo,
                                                                int                     n,
                                                                const hipDoubleComplex* A,
                                                                int                     lda,
                                                                const hipDoubleComplex* B,
                                                                int                     ldb,
                                                                const double*           W,
                                                                int*                    lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsygvd(hipsolverHandle_t  handle,
                                                     hipsolverEigType_t itype,
                                                     hipsolverEigMode_t jobz,
                                                     hipblasFillMode_t  uplo,
                                                     int                n,
                                                     float*             A,
                                                     int                lda,
                                                     float*             B,
                                                     int                ldb,
                                                     float*             W,
                                                     float*             work,
                                                     int                lwork,
                                                     int*               devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsygvd(hipsolverHandle_t  handle,
                                                     hipsolverEigType_t itype,
                                                     hipsolverEigMode_t jobz,
                                                     hipblasFillMode_t  uplo,
                                                     int                n,
                                                     double*            A,
                                                     int                lda,
                                                     double*            B,
                                                     int                ldb,
                                                     double*            W,
                                                     double*            work,
                                                     int                lwork,
                                                     int*               devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnChegvd(hipsolverHandle_t  handle,
                                                     hipsolverEigType_t itype,
                                                     hipsolverEigMode_t jobz,
                                                     hipblasFillMode_t  uplo,
                                                     int                n,
                                                     hipFloatComplex*   A,
                                                     int                lda,
                                                     hipFloatComplex*   B,
                                                     int                ldb,
                                                     float*             W,
                                                     hipFloatComplex*   work,
                                                     int                lwork,
                                                     int*               devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZhegvd(hipsolverHandle_t  handle,
                                                     hipsolverEigType_t itype,
                                                     hipsolverEigMode_t jobz,
                                                     hipblasFillMode_t  uplo,
                                                     int                n,
                                                     hipDoubleComplex*  A,
                                                     int                lda,
                                                     hipDoubleComplex*  B,
                                                     int                ldb,
                                                     double*            W,
                                                     hipDoubleComplex*  work,
                                                     int                lwork,
                                                     int*               devInfo);

// sygvdx/hegvdx
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsygvdx_bufferSize(hipsolverHandle_t   handle,
                                                                 hipsolverEigType_t  itype,
                                                                 hipsolverEigMode_t  jobz,
                                                                 hipsolverEigRange_t range,
                                                                 hipblasFillMode_t   uplo,
                                                                 int                 n,
                                                                 const float*        A,
                                                                 int                 lda,
                                                                 const float*        B,
                                                                 int                 ldb,
                                                                 float               vl,
                                                                 float               vu,
                                                                 int                 il,
                                                                 int                 iu,
                                                                 int*                nev,
                                                                 const float*        W,
                                                                 int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsygvdx_bufferSize(hipsolverHandle_t   handle,
                                                                 hipsolverEigType_t  itype,
                                                                 hipsolverEigMode_t  jobz,
                                                                 hipsolverEigRange_t range,
                                                                 hipblasFillMode_t   uplo,
                                                                 int                 n,
                                                                 const double*       A,
                                                                 int                 lda,
                                                                 const double*       B,
                                                                 int                 ldb,
                                                                 double              vl,
                                                                 double              vu,
                                                                 int                 il,
                                                                 int                 iu,
                                                                 int*                nev,
                                                                 const double*       W,
                                                                 int*                lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnChegvdx_bufferSize(hipsolverHandle_t      handle,
                                                                 hipsolverEigType_t     itype,
                                                                 hipsolverEigMode_t     jobz,
                                                                 hipsolverEigRange_t    range,
                                                                 hipblasFillMode_t      uplo,
                                                                 int                    n,
                                                                 const hipFloatComplex* A,
                                                                 int                    lda,
                                                                 const hipFloatComplex* B,
                                                                 int                    ldb,
                                                                 float                  vl,
                                                                 float                  vu,
                                                                 int                    il,
                                                                 int                    iu,
                                                                 int*                   nev,
                                                                 const float*           W,
                                                                 int*                   lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZhegvdx_bufferSize(hipsolverHandle_t       handle,
                                                                 hipsolverEigType_t      itype,
                                                                 hipsolverEigMode_t      jobz,
                                                                 hipsolverEigRange_t     range,
                                                                 hipblasFillMode_t       uplo,
                                                                 int                     n,
                                                                 const hipDoubleComplex* A,
                                                                 int                     lda,
                                                                 const hipDoubleComplex* B,
                                                                 int                     ldb,
                                                                 double                  vl,
                                                                 double                  vu,
                                                                 int                     il,
                                                                 int                     iu,
                                                                 int*                    nev,
                                                                 const double*           W,
                                                                 int*                    lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsygvdx(hipsolverHandle_t   handle,
                                                      hipsolverEigType_t  itype,
                                                      hipsolverEigMode_t  jobz,
                                                      hipsolverEigRange_t range,
                                                      hipblasFillMode_t   uplo,
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
                                                      float*              work,
                                                      int                 lwork,
                                                      int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsygvdx(hipsolverHandle_t   handle,
                                                      hipsolverEigType_t  itype,
                                                      hipsolverEigMode_t  jobz,
                                                      hipsolverEigRange_t range,
                                                      hipblasFillMode_t   uplo,
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
                                                      double*             work,
                                                      int                 lwork,
                                                      int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnChegvdx(hipsolverHandle_t   handle,
                                                      hipsolverEigType_t  itype,
                                                      hipsolverEigMode_t  jobz,
                                                      hipsolverEigRange_t range,
                                                      hipblasFillMode_t   uplo,
                                                      int                 n,
                                                      hipFloatComplex*    A,
                                                      int                 lda,
                                                      hipFloatComplex*    B,
                                                      int                 ldb,
                                                      float               vl,
                                                      float               vu,
                                                      int                 il,
                                                      int                 iu,
                                                      int*                nev,
                                                      float*              W,
                                                      hipFloatComplex*    work,
                                                      int                 lwork,
                                                      int*                devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZhegvdx(hipsolverHandle_t   handle,
                                                      hipsolverEigType_t  itype,
                                                      hipsolverEigMode_t  jobz,
                                                      hipsolverEigRange_t range,
                                                      hipblasFillMode_t   uplo,
                                                      int                 n,
                                                      hipDoubleComplex*   A,
                                                      int                 lda,
                                                      hipDoubleComplex*   B,
                                                      int                 ldb,
                                                      double              vl,
                                                      double              vu,
                                                      int                 il,
                                                      int                 iu,
                                                      int*                nev,
                                                      double*             W,
                                                      hipDoubleComplex*   work,
                                                      int                 lwork,
                                                      int*                devInfo);

// sygvj/hegvj
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsygvj_bufferSize(hipsolverDnHandle_t  handle,
                                                                hipsolverEigType_t   itype,
                                                                hipsolverEigMode_t   jobz,
                                                                hipblasFillMode_t    uplo,
                                                                int                  n,
                                                                const float*         A,
                                                                int                  lda,
                                                                const float*         B,
                                                                int                  ldb,
                                                                const float*         W,
                                                                int*                 lwork,
                                                                hipsolverSyevjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsygvj_bufferSize(hipsolverDnHandle_t  handle,
                                                                hipsolverEigType_t   itype,
                                                                hipsolverEigMode_t   jobz,
                                                                hipblasFillMode_t    uplo,
                                                                int                  n,
                                                                const double*        A,
                                                                int                  lda,
                                                                const double*        B,
                                                                int                  ldb,
                                                                const double*        W,
                                                                int*                 lwork,
                                                                hipsolverSyevjInfo_t params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnChegvj_bufferSize(hipsolverDnHandle_t    handle,
                                                                hipsolverEigType_t     itype,
                                                                hipsolverEigMode_t     jobz,
                                                                hipblasFillMode_t      uplo,
                                                                int                    n,
                                                                const hipFloatComplex* A,
                                                                int                    lda,
                                                                const hipFloatComplex* B,
                                                                int                    ldb,
                                                                const float*           W,
                                                                int*                   lwork,
                                                                hipsolverSyevjInfo_t   params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZhegvj_bufferSize(hipsolverDnHandle_t     handle,
                                                                hipsolverEigType_t      itype,
                                                                hipsolverEigMode_t      jobz,
                                                                hipblasFillMode_t       uplo,
                                                                int                     n,
                                                                const hipDoubleComplex* A,
                                                                int                     lda,
                                                                const hipDoubleComplex* B,
                                                                int                     ldb,
                                                                const double*           W,
                                                                int*                    lwork,
                                                                hipsolverSyevjInfo_t    params);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsygvj(hipsolverDnHandle_t  handle,
                                                     hipsolverEigType_t   itype,
                                                     hipsolverEigMode_t   jobz,
                                                     hipblasFillMode_t    uplo,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsygvj(hipsolverDnHandle_t  handle,
                                                     hipsolverEigType_t   itype,
                                                     hipsolverEigMode_t   jobz,
                                                     hipblasFillMode_t    uplo,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnChegvj(hipsolverDnHandle_t  handle,
                                                     hipsolverEigType_t   itype,
                                                     hipsolverEigMode_t   jobz,
                                                     hipblasFillMode_t    uplo,
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

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZhegvj(hipsolverDnHandle_t  handle,
                                                     hipsolverEigType_t   itype,
                                                     hipsolverEigMode_t   jobz,
                                                     hipblasFillMode_t    uplo,
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
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsytrd_bufferSize(hipsolverHandle_t handle,
                                                                hipblasFillMode_t uplo,
                                                                int               n,
                                                                const float*      A,
                                                                int               lda,
                                                                const float*      D,
                                                                const float*      E,
                                                                const float*      tau,
                                                                int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsytrd_bufferSize(hipsolverHandle_t handle,
                                                                hipblasFillMode_t uplo,
                                                                int               n,
                                                                const double*     A,
                                                                int               lda,
                                                                const double*     D,
                                                                const double*     E,
                                                                const double*     tau,
                                                                int*              lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnChetrd_bufferSize(hipsolverHandle_t      handle,
                                                                hipblasFillMode_t      uplo,
                                                                int                    n,
                                                                const hipFloatComplex* A,
                                                                int                    lda,
                                                                const float*           D,
                                                                const float*           E,
                                                                const hipFloatComplex* tau,
                                                                int*                   lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZhetrd_bufferSize(hipsolverHandle_t       handle,
                                                                hipblasFillMode_t       uplo,
                                                                int                     n,
                                                                const hipDoubleComplex* A,
                                                                int                     lda,
                                                                const double*           D,
                                                                const double*           E,
                                                                const hipDoubleComplex* tau,
                                                                int*                    lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsytrd(hipsolverHandle_t handle,
                                                     hipblasFillMode_t uplo,
                                                     int               n,
                                                     float*            A,
                                                     int               lda,
                                                     float*            D,
                                                     float*            E,
                                                     float*            tau,
                                                     float*            work,
                                                     int               lwork,
                                                     int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsytrd(hipsolverHandle_t handle,
                                                     hipblasFillMode_t uplo,
                                                     int               n,
                                                     double*           A,
                                                     int               lda,
                                                     double*           D,
                                                     double*           E,
                                                     double*           tau,
                                                     double*           work,
                                                     int               lwork,
                                                     int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnChetrd(hipsolverHandle_t handle,
                                                     hipblasFillMode_t uplo,
                                                     int               n,
                                                     hipFloatComplex*  A,
                                                     int               lda,
                                                     float*            D,
                                                     float*            E,
                                                     hipFloatComplex*  tau,
                                                     hipFloatComplex*  work,
                                                     int               lwork,
                                                     int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZhetrd(hipsolverHandle_t handle,
                                                     hipblasFillMode_t uplo,
                                                     int               n,
                                                     hipDoubleComplex* A,
                                                     int               lda,
                                                     double*           D,
                                                     double*           E,
                                                     hipDoubleComplex* tau,
                                                     hipDoubleComplex* work,
                                                     int               lwork,
                                                     int*              devInfo);

// sytrf
HIPSOLVER_EXPORT hipsolverStatus_t
    hipsolverDnSsytrf_bufferSize(hipsolverHandle_t handle, int n, float* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t
    hipsolverDnDsytrf_bufferSize(hipsolverHandle_t handle, int n, double* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCsytrf_bufferSize(
    hipsolverHandle_t handle, int n, hipFloatComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZsytrf_bufferSize(
    hipsolverHandle_t handle, int n, hipDoubleComplex* A, int lda, int* lwork);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnSsytrf(hipsolverHandle_t handle,
                                                     hipblasFillMode_t uplo,
                                                     int               n,
                                                     float*            A,
                                                     int               lda,
                                                     int*              ipiv,
                                                     float*            work,
                                                     int               lwork,
                                                     int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnDsytrf(hipsolverHandle_t handle,
                                                     hipblasFillMode_t uplo,
                                                     int               n,
                                                     double*           A,
                                                     int               lda,
                                                     int*              ipiv,
                                                     double*           work,
                                                     int               lwork,
                                                     int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnCsytrf(hipsolverHandle_t handle,
                                                     hipblasFillMode_t uplo,
                                                     int               n,
                                                     hipFloatComplex*  A,
                                                     int               lda,
                                                     int*              ipiv,
                                                     hipFloatComplex*  work,
                                                     int               lwork,
                                                     int*              devInfo);

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnZsytrf(hipsolverHandle_t handle,
                                                     hipblasFillMode_t uplo,
                                                     int               n,
                                                     hipDoubleComplex* A,
                                                     int               lda,
                                                     int*              ipiv,
                                                     hipDoubleComplex* work,
                                                     int               lwork,
                                                     int*              devInfo);

#ifdef __cplusplus
}
#endif

#endif // HIPSOLVER_DENSE_H
