/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

/*! \file
 *  \brief Implementation of the compatibility sparse APIs that require especial
 *  calls to hipSOLVER or cuSOLVER.
 */

#include "exceptions.hpp"
#include "hipsolver.h"
#include "hipsolver_conversions.hpp"
#include "lib_macros.hpp"

#include <cusolverSp.h>

extern "C" {

/******************** HANDLE ********************/
hipsolverStatus_t hipsolverSpCreate(hipsolverSpHandle_t* handle)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return hipsolver::cuda2hip_status(cusolverSpCreate((cusolverSpHandle_t*)handle));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverSpDestroy(hipsolverSpHandle_t handle)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return hipsolver::cuda2hip_status(cusolverSpDestroy((cusolverSpHandle_t)handle));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverSpSetStream(hipsolverSpHandle_t handle, hipStream_t streamId)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;

    return hipsolver::cuda2hip_status(
        cusolverSpSetStream((cusolverSpHandle_t)handle, (cudaStream_t)streamId));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

/******************** CSRLSVCHOL ********************/
hipsolverStatus_t hipsolverSpScsrlsvchol(hipsolverSpHandle_t       handle,
                                         int                       n,
                                         int                       nnzA,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              csrVal,
                                         const int*                csrRowPtr,
                                         const int*                csrColInd,
                                         const float*              b,
                                         float                     tolerance,
                                         int                       reorder,
                                         float*                    x,
                                         int*                      singularity)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return hipsolver::cuda2hip_status(cusolverSpScsrlsvchol((cusolverSpHandle_t)handle,
                                                            n,
                                                            nnzA,
                                                            (cusparseMatDescr_t)descrA,
                                                            csrVal,
                                                            csrRowPtr,
                                                            csrColInd,
                                                            b,
                                                            tolerance,
                                                            reorder,
                                                            x,
                                                            singularity));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverSpDcsrlsvchol(hipsolverSpHandle_t       handle,
                                         int                       n,
                                         int                       nnzA,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             csrVal,
                                         const int*                csrRowPtr,
                                         const int*                csrColInd,
                                         const double*             b,
                                         double                    tolerance,
                                         int                       reorder,
                                         double*                   x,
                                         int*                      singularity)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return hipsolver::cuda2hip_status(cusolverSpDcsrlsvchol((cusolverSpHandle_t)handle,
                                                            n,
                                                            nnzA,
                                                            (cusparseMatDescr_t)descrA,
                                                            csrVal,
                                                            csrRowPtr,
                                                            csrColInd,
                                                            b,
                                                            tolerance,
                                                            reorder,
                                                            x,
                                                            singularity));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

/*hipsolverStatus_t hipsolverSpCcsrlsvchol(hipsolverSpHandle_t       handle,
                                         int                       n,
                                         int                       nnzA,
                                         const hipsparseMatDescr_t descrA,
                                         const hipFloatComplex*    csrVal,
                                         const int*                csrRowPtr,
                                         const int*                csrColInd,
                                         const hipFloatComplex*    b,
                                         float                     tolerance,
                                         int                       reorder,
                                         hipFloatComplex*          x,
                                         int*                      singularity)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return hipsolver::cuda2hip_status(cusolverSpCcsrlsvchol((cusolverSpHandle_t)handle,
                                                 n,
                                                 nnzA,
                                                 (cusparseMatDescr_t)descrA,
                                                 (cuComplex*)csrVal,
                                                 csrRowPtr,
                                                 csrColInd,
                                                 (cuComplex*)b,
                                                 tolerance,
                                                 reorder,
                                                 (cuComplex*)x,
                                                 singularity));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverSpZcsrlsvchol(hipsolverSpHandle_t       handle,
                                         int                       n,
                                         int                       nnzA,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   csrVal,
                                         const int*                csrRowPtr,
                                         const int*                csrColInd,
                                         const hipDoubleComplex*   b,
                                         double                    tolerance,
                                         int                       reorder,
                                         hipDoubleComplex*         x,
                                         int*                      singularity)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return hipsolver::cuda2hip_status(cusolverSpZcsrlsvchol((cusolverSpHandle_t)handle,
                                                 n,
                                                 nnzA,
                                                 (cusparseMatDescr_t)descrA,
                                                 (cuDoubleComplex*)csrVal,
                                                 csrRowPtr,
                                                 csrColInd,
                                                 (cuDoubleComplex*)b,
                                                 tolerance,
                                                 reorder,
                                                 (cuDoubleComplex*)x,
                                                 singularity));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}*/

hipsolverStatus_t hipsolverSpScsrlsvcholHost(hipsolverSpHandle_t       handle,
                                             int                       n,
                                             int                       nnzA,
                                             const hipsparseMatDescr_t descrA,
                                             const float*              csrVal,
                                             const int*                csrRowPtr,
                                             const int*                csrColInd,
                                             const float*              b,
                                             float                     tolerance,
                                             int                       reorder,
                                             float*                    x,
                                             int*                      singularity)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return hipsolver::cuda2hip_status(cusolverSpScsrlsvcholHost((cusolverSpHandle_t)handle,
                                                                n,
                                                                nnzA,
                                                                (cusparseMatDescr_t)descrA,
                                                                csrVal,
                                                                csrRowPtr,
                                                                csrColInd,
                                                                b,
                                                                tolerance,
                                                                reorder,
                                                                x,
                                                                singularity));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverSpDcsrlsvcholHost(hipsolverSpHandle_t       handle,
                                             int                       n,
                                             int                       nnzA,
                                             const hipsparseMatDescr_t descrA,
                                             const double*             csrVal,
                                             const int*                csrRowPtr,
                                             const int*                csrColInd,
                                             const double*             b,
                                             double                    tolerance,
                                             int                       reorder,
                                             double*                   x,
                                             int*                      singularity)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return hipsolver::cuda2hip_status(cusolverSpDcsrlsvcholHost((cusolverSpHandle_t)handle,
                                                                n,
                                                                nnzA,
                                                                (cusparseMatDescr_t)descrA,
                                                                csrVal,
                                                                csrRowPtr,
                                                                csrColInd,
                                                                b,
                                                                tolerance,
                                                                reorder,
                                                                x,
                                                                singularity));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

/*hipsolverStatus_t hipsolverSpCcsrlsvcholHost(hipsolverSpHandle_t       handle,
                                             int                       n,
                                             int                       nnzA,
                                             const hipsparseMatDescr_t descrA,
                                             const hipFloatComplex*    csrVal,
                                             const int*                csrRowPtr,
                                             const int*                csrColInd,
                                             const hipFloatComplex*    b,
                                             float                     tolerance,
                                             int                       reorder,
                                             hipFloatComplex*          x,
                                             int*                      singularity)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return hipsolver::cuda2hip_status(cusolverSpCcsrlsvcholHost((cusolverSpHandle_t)handle,
                                                     n,
                                                     nnzA,
                                                     (cusparseMatDescr_t)descrA,
                                                     (cuComplex*)csrVal,
                                                     csrRowPtr,
                                                     csrColInd,
                                                     (cuComplex*)b,
                                                     tolerance,
                                                     reorder,
                                                     (cuComplex*)x,
                                                     singularity));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverSpZcsrlsvcholHost(hipsolverSpHandle_t       handle,
                                             int                       n,
                                             int                       nnzA,
                                             const hipsparseMatDescr_t descrA,
                                             const hipDoubleComplex*   csrVal,
                                             const int*                csrRowPtr,
                                             const int*                csrColInd,
                                             const hipDoubleComplex*   b,
                                             double                    tolerance,
                                             int                       reorder,
                                             hipDoubleComplex*         x,
                                             int*                      singularity)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return hipsolver::cuda2hip_status(cusolverSpZcsrlsvcholHost((cusolverSpHandle_t)handle,
                                                     n,
                                                     nnzA,
                                                     (cusparseMatDescr_t)descrA,
                                                     (cuDoubleComplex*)csrVal,
                                                     csrRowPtr,
                                                     csrColInd,
                                                     (cuDoubleComplex*)b,
                                                     tolerance,
                                                     reorder,
                                                     (cuDoubleComplex*)x,
                                                     singularity));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}*/

/******************** CSRLSVQR ********************/
hipsolverStatus_t hipsolverSpScsrlsvqr(hipsolverSpHandle_t       handle,
                                       int                       n,
                                       int                       nnzA,
                                       const hipsparseMatDescr_t descrA,
                                       const float*              csrVal,
                                       const int*                csrRowPtr,
                                       const int*                csrColInd,
                                       const float*              b,
                                       double                    tolerance,
                                       int                       reorder,
                                       float*                    x,
                                       int*                      singularity)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return hipsolver::cuda2hip_status(cusolverSpScsrlsvqr((cusolverSpHandle_t)handle,
                                                          n,
                                                          nnzA,
                                                          (cusparseMatDescr_t)descrA,
                                                          csrVal,
                                                          csrRowPtr,
                                                          csrColInd,
                                                          b,
                                                          tolerance,
                                                          reorder,
                                                          x,
                                                          singularity));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverSpDcsrlsvqr(hipsolverSpHandle_t       handle,
                                       int                       n,
                                       int                       nnzA,
                                       const hipsparseMatDescr_t descrA,
                                       const double*             csrVal,
                                       const int*                csrRowPtr,
                                       const int*                csrColInd,
                                       const double*             b,
                                       double                    tolerance,
                                       int                       reorder,
                                       double*                   x,
                                       int*                      singularity)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return hipsolver::cuda2hip_status(cusolverSpDcsrlsvqr((cusolverSpHandle_t)handle,
                                                          n,
                                                          nnzA,
                                                          (cusparseMatDescr_t)descrA,
                                                          csrVal,
                                                          csrRowPtr,
                                                          csrColInd,
                                                          b,
                                                          tolerance,
                                                          reorder,
                                                          x,
                                                          singularity));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

/*hipsolverStatus_t hipsolverSpCcsrlsvqr(hipsolverSpHandle_t       handle,
                                       int                       n,
                                       int                       nnzA,
                                       const hipsparseMatDescr_t descrA,
                                       const hipFloatComplex*    csrVal,
                                       const int*                csrRowPtr,
                                       const int*                csrColInd,
                                       const hipFloatComplex*    b,
                                       double                    tolerance,
                                       int                       reorder,
                                       hipFloatComplex*          x,
                                       int*                      singularity)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return hipsolver::cuda2hip_status(cusolverSpCcsrlsvqr((cusolverSpHandle_t)handle,
                                                          n,
                                                          nnzA,
                                                          (cusparseMatDescr_t)descrA,
                                                          (cuComplex*)csrVal,
                                                          csrRowPtr,
                                                          csrColInd,
                                                          (cuComplex*)b,
                                                          tolerance,
                                                          reorder,
                                                          (cuComplex*)x,
                                                          singularity));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverSpZcsrlsvqr(hipsolverSpHandle_t       handle,
                                       int                       n,
                                       int                       nnzA,
                                       const hipsparseMatDescr_t descrA,
                                       const hipDoubleComplex*   csrVal,
                                       const int*                csrRowPtr,
                                       const int*                csrColInd,
                                       const hipDoubleComplex*   b,
                                       double                    tolerance,
                                       int                       reorder,
                                       hipDoubleComplex*         x,
                                       int*                      singularity)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!descrA)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return hipsolver::cuda2hip_status(cusolverSpZcsrlsvqr((cusolverSpHandle_t)handle,
                                                          n,
                                                          nnzA,
                                                          (cusparseMatDescr_t)descrA,
                                                          (cuDoubleComplex*)csrVal,
                                                          csrRowPtr,
                                                          csrColInd,
                                                          (cuDoubleComplex*)b,
                                                          tolerance,
                                                          reorder,
                                                          (cuDoubleComplex*)x,
                                                          singularity));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}*/

} //extern C
