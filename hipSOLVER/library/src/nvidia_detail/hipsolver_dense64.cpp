/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
 *  \brief Implementation of the compatibility APIs that require especial calls
 *  to hipSOLVER on the cuSOLVER side.
 */

#include "exceptions.hpp"
#include "hipsolver.h"
#include "hipsolver_conversions.hpp"

#include <cusolverDn.h>

extern "C" {

hipsolverStatus_t hipsolverDnCreateParams(hipsolverDnParams_t* params)
try
{
    if(!params)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return hipsolver::cuda2hip_status(cusolverDnCreateParams((cusolverDnParams_t*)params));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverDnDestroyParams(hipsolverDnParams_t params)
try
{
    if(!params)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return hipsolver::cuda2hip_status(cusolverDnDestroyParams((cusolverDnParams_t)params));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverDnSetAdvOptions(hipsolverDnParams_t   params,
                                           hipsolverDnFunction_t func,
                                           hipsolverAlgMode_t    alg)
try
{
    if(!params)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return hipsolver::cuda2hip_status(cusolverDnSetAdvOptions((cusolverDnParams_t)params,
                                                              hipsolver::hip2cuda_function(func),
                                                              hipsolver::hip2cuda_algmode(alg)));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

/******************** GEQRF ********************/
hipsolverStatus_t hipsolverDnXgeqrf_bufferSize(hipsolverDnHandle_t handle,
                                               hipsolverDnParams_t params,
                                               int64_t             m,
                                               int64_t             n,
                                               hipDataType         dataTypeA,
                                               const void*         A,
                                               int64_t             lda,
                                               hipDataType         dataTypeTau,
                                               const void*         tau,
                                               hipDataType         computeType,
                                               size_t*             lworkOnDevice,
                                               size_t*             lworkOnHost)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!params)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return hipsolver::cuda2hip_status(cusolverDnXgeqrf_bufferSize((cusolverDnHandle_t)handle,
                                                                  (cusolverDnParams_t)params,
                                                                  m,
                                                                  n,
                                                                  dataTypeA,
                                                                  A,
                                                                  lda,
                                                                  dataTypeTau,
                                                                  tau,
                                                                  computeType,
                                                                  lworkOnDevice,
                                                                  lworkOnHost));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverDnXgeqrf(hipsolverDnHandle_t handle,
                                    hipsolverDnParams_t params,
                                    int64_t             m,
                                    int64_t             n,
                                    hipDataType         dataTypeA,
                                    void*               A,
                                    int64_t             lda,
                                    hipDataType         dataTypeTau,
                                    void*               tau,
                                    hipDataType         computeType,
                                    void*               workOnDevice,
                                    size_t              lworkOnDevice,
                                    void*               workOnHost,
                                    size_t              lworkOnHost,
                                    int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!params)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return hipsolver::cuda2hip_status(cusolverDnXgeqrf((cusolverDnHandle_t)handle,
                                                       (cusolverDnParams_t)params,
                                                       m,
                                                       n,
                                                       dataTypeA,
                                                       A,
                                                       lda,
                                                       dataTypeTau,
                                                       tau,
                                                       computeType,
                                                       workOnDevice,
                                                       lworkOnDevice,
                                                       workOnHost,
                                                       lworkOnHost,
                                                       devInfo));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

/******************** GETRF ********************/
hipsolverStatus_t hipsolverDnXgetrf_bufferSize(hipsolverDnHandle_t handle,
                                               hipsolverDnParams_t params,
                                               int64_t             m,
                                               int64_t             n,
                                               hipDataType         dataTypeA,
                                               const void*         A,
                                               int64_t             lda,
                                               hipDataType         computeType,
                                               size_t*             lworkOnDevice,
                                               size_t*             lworkOnHost)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!params)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return hipsolver::cuda2hip_status(cusolverDnXgetrf_bufferSize((cusolverDnHandle_t)handle,
                                                                  (cusolverDnParams_t)params,
                                                                  m,
                                                                  n,
                                                                  dataTypeA,
                                                                  A,
                                                                  lda,
                                                                  computeType,
                                                                  lworkOnDevice,
                                                                  lworkOnHost));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverDnXgetrf(hipsolverDnHandle_t handle,
                                    hipsolverDnParams_t params,
                                    int64_t             m,
                                    int64_t             n,
                                    hipDataType         dataTypeA,
                                    void*               A,
                                    int64_t             lda,
                                    int64_t*            devIpiv,
                                    hipDataType         computeType,
                                    void*               workOnDevice,
                                    size_t              lworkOnDevice,
                                    void*               workOnHost,
                                    size_t              lworkOnHost,
                                    int*                devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!params)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return hipsolver::cuda2hip_status(cusolverDnXgetrf((cusolverDnHandle_t)handle,
                                                       (cusolverDnParams_t)params,
                                                       m,
                                                       n,
                                                       dataTypeA,
                                                       A,
                                                       lda,
                                                       devIpiv,
                                                       computeType,
                                                       workOnDevice,
                                                       lworkOnDevice,
                                                       workOnHost,
                                                       lworkOnHost,
                                                       devInfo));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

/******************** GETRS ********************/
hipsolverStatus_t hipsolverDnXgetrs(hipsolverDnHandle_t  handle,
                                    hipsolverDnParams_t  params,
                                    hipsolverOperation_t trans,
                                    int64_t              n,
                                    int64_t              nrhs,
                                    hipDataType          dataTypeA,
                                    const void*          A,
                                    int64_t              lda,
                                    const int64_t*       devIpiv,
                                    hipDataType          dataTypeB,
                                    void*                B,
                                    int64_t              ldb,
                                    int*                 devInfo)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!params)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return hipsolver::cuda2hip_status(cusolverDnXgetrs((cusolverDnHandle_t)handle,
                                                       (cusolverDnParams_t)params,
                                                       hipsolver::hip2cuda_operation(trans),
                                                       n,
                                                       nrhs,
                                                       dataTypeA,
                                                       A,
                                                       lda,
                                                       devIpiv,
                                                       dataTypeB,
                                                       B,
                                                       ldb,
                                                       devInfo));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

/******************** POTRF ********************/
hipsolverStatus_t hipsolverDnXpotrf_bufferSize(hipsolverDnHandle_t handle,
                                               hipsolverDnParams_t params,
                                               hipsolverFillMode_t uplo,
                                               int64_t             n,
                                               hipDataType         dataTypeA,
                                               const void*         A,
                                               int64_t             lda,
                                               hipDataType         computeType,
                                               size_t*             lworkOnDevice,
                                               size_t*             lworkOnHost)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!params)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return hipsolver::cuda2hip_status(cusolverDnXpotrf_bufferSize((cusolverDnHandle_t)handle,
                                                                  (cusolverDnParams_t)params,
                                                                  hipsolver::hip2cuda_fill(uplo),
                                                                  n,
                                                                  dataTypeA,
                                                                  A,
                                                                  lda,
                                                                  computeType,
                                                                  lworkOnDevice,
                                                                  lworkOnHost));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverDnXpotrf(hipsolverDnHandle_t handle,
                                    hipsolverDnParams_t params,
                                    hipsolverFillMode_t uplo,
                                    int64_t             n,
                                    hipDataType         dataTypeA,
                                    void*               A,
                                    int64_t             lda,
                                    hipDataType         computeType,
                                    void*               workOnDevice,
                                    size_t              lworkOnDevice,
                                    void*               workOnHost,
                                    size_t              lworkOnHost,
                                    int*                info)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!params)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return hipsolver::cuda2hip_status(cusolverDnXpotrf((cusolverDnHandle_t)handle,
                                                       (cusolverDnParams_t)params,
                                                       hipsolver::hip2cuda_fill(uplo),
                                                       n,
                                                       dataTypeA,
                                                       A,
                                                       lda,
                                                       computeType,
                                                       workOnDevice,
                                                       lworkOnDevice,
                                                       workOnHost,
                                                       lworkOnHost,
                                                       info));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

/******************** POTRS ********************/
hipsolverStatus_t hipsolverDnXpotrs(hipsolverDnHandle_t handle,
                                    hipsolverDnParams_t params,
                                    hipsolverFillMode_t uplo,
                                    int64_t             n,
                                    int64_t             nrhs,
                                    hipDataType         dataTypeA,
                                    const void*         A,
                                    int64_t             lda,
                                    hipDataType         dataTypeB,
                                    void*               B,
                                    int64_t             ldb,
                                    int*                info)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!params)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    return hipsolver::cuda2hip_status(cusolverDnXpotrs((cusolverDnHandle_t)handle,
                                                       (cusolverDnParams_t)params,
                                                       hipsolver::hip2cuda_fill(uplo),
                                                       n,
                                                       nrhs,
                                                       dataTypeA,
                                                       A,
                                                       lda,
                                                       dataTypeB,
                                                       B,
                                                       ldb,
                                                       info));
}
catch(...)
{
    return hipsolver::exception2hip_status();
}
} //extern C
