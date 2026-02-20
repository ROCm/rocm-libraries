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
 *  to hipSOLVER on the rocSOLVER side.
 */

#include "exceptions.hpp"
#include "hipsolver.h"
#include "hipsolver_conversions.hpp"
#include "lib_macros.hpp"
#include "utility.hpp"

#include "rocblas/internal/rocblas_device_malloc.hpp"
#include "rocblas/rocblas.h"
#include "rocsolver/rocsolver.h"
#include <algorithm>
#include <climits>
#include <functional>
#include <iostream>
#include <math.h>

extern "C" {

// The following functions are not included in the public API of rocSOLVER and must be declared

rocblas_status rocsolver_sgetrf_info32(rocblas_handle handle,
                                       const int64_t  m,
                                       const int64_t  n,
                                       float*         A,
                                       const int64_t  lda,
                                       int64_t*       ipiv,
                                       rocblas_int*   info);

rocblas_status rocsolver_dgetrf_info32(rocblas_handle handle,
                                       const int64_t  m,
                                       const int64_t  n,
                                       double*        A,
                                       const int64_t  lda,
                                       int64_t*       ipiv,
                                       rocblas_int*   info);

rocblas_status rocsolver_cgetrf_info32(rocblas_handle         handle,
                                       const int64_t          m,
                                       const int64_t          n,
                                       rocblas_float_complex* A,
                                       const int64_t          lda,
                                       int64_t*               ipiv,
                                       rocblas_int*           info);

rocblas_status rocsolver_zgetrf_info32(rocblas_handle          handle,
                                       const int64_t           m,
                                       const int64_t           n,
                                       rocblas_double_complex* A,
                                       const int64_t           lda,
                                       int64_t*                ipiv,
                                       rocblas_int*            info);

rocblas_status rocsolver_sgetrf_npvt_info32(rocblas_handle handle,
                                            const int64_t  m,
                                            const int64_t  n,
                                            float*         A,
                                            const int64_t  lda,
                                            rocblas_int*   info);

rocblas_status rocsolver_dgetrf_npvt_info32(rocblas_handle handle,
                                            const int64_t  m,
                                            const int64_t  n,
                                            double*        A,
                                            const int64_t  lda,
                                            rocblas_int*   info);

rocblas_status rocsolver_cgetrf_npvt_info32(rocblas_handle         handle,
                                            const int64_t          m,
                                            const int64_t          n,
                                            rocblas_float_complex* A,
                                            const int64_t          lda,
                                            rocblas_int*           info);

rocblas_status rocsolver_zgetrf_npvt_info32(rocblas_handle          handle,
                                            const int64_t           m,
                                            const int64_t           n,
                                            rocblas_double_complex* A,
                                            const int64_t           lda,
                                            rocblas_int*            info);

rocblas_status rocsolver_spotrf_info32(rocblas_handle     handle,
                                       const rocblas_fill uplo,
                                       const int64_t      n,
                                       float*             A,
                                       const int64_t      lda,
                                       rocblas_int*       info);

rocblas_status rocsolver_dpotrf_info32(rocblas_handle     handle,
                                       const rocblas_fill uplo,
                                       const int64_t      n,
                                       double*            A,
                                       const int64_t      lda,
                                       rocblas_int*       info);

rocblas_status rocsolver_cpotrf_info32(rocblas_handle         handle,
                                       const rocblas_fill     uplo,
                                       const int64_t          n,
                                       rocblas_float_complex* A,
                                       const int64_t          lda,
                                       rocblas_int*           info);

rocblas_status rocsolver_zpotrf_info32(rocblas_handle          handle,
                                       const rocblas_fill      uplo,
                                       const int64_t           n,
                                       rocblas_double_complex* A,
                                       const int64_t           lda,
                                       rocblas_int*            info);

/******************** PARAMS ********************/
struct hipsolverParams
{
    hipsolverDnFunction_t func;
    hipsolverAlgMode_t    alg;

    // Constructor
    explicit hipsolverParams()
        : func(HIPSOLVERDN_GETRF)
        , alg(HIPSOLVER_ALG_0)
    {
    }
};

hipsolverStatus_t hipsolverDnCreateParams(hipsolverDnParams_t* info)
try
{
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *info = new hipsolverParams;

    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverDnDestroyParams(hipsolverDnParams_t info)
try
{
    if(!info)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    hipsolverParams* params = (hipsolverParams*)info;
    delete params;

    return HIPSOLVER_STATUS_SUCCESS;
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
    return HIPSOLVER_STATUS_NOT_SUPPORTED;
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
    if(!lworkOnDevice || !lworkOnHost)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lworkOnDevice = 0;
    *lworkOnHost   = 0;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status;
    if(dataTypeA == HIP_R_32F && dataTypeTau == HIP_R_32F && computeType == HIP_R_32F)
    {
        status = hipsolver::rocblas2hip_status(
            rocsolver_sgeqrf_64((rocblas_handle)handle, m, n, nullptr, lda, nullptr));
    }
    else if(dataTypeA == HIP_R_64F && dataTypeTau == HIP_R_64F && computeType == HIP_R_64F)
    {
        status = hipsolver::rocblas2hip_status(
            rocsolver_dgeqrf_64((rocblas_handle)handle, m, n, nullptr, lda, nullptr));
    }
    else if(dataTypeA == HIP_C_32F && dataTypeTau == HIP_C_32F && computeType == HIP_C_32F)
    {
        status = hipsolver::rocblas2hip_status(
            rocsolver_cgeqrf_64((rocblas_handle)handle, m, n, nullptr, lda, nullptr));
    }
    else if(dataTypeA == HIP_C_64F && dataTypeTau == HIP_C_64F && computeType == HIP_C_64F)
    {
        status = hipsolver::rocblas2hip_status(
            rocsolver_zgeqrf_64((rocblas_handle)handle, m, n, nullptr, lda, nullptr));
    }
    else
        return HIPSOLVER_STATUS_INVALID_ENUM;
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, lworkOnDevice);

    return status;
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

    if(workOnDevice && lworkOnDevice)
        CHECK_ROCBLAS_ERROR(
            rocblas_set_workspace((rocblas_handle)handle, workOnDevice, lworkOnDevice));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverDnXgeqrf_bufferSize((rocblas_handle)handle,
                                                           params,
                                                           m,
                                                           n,
                                                           dataTypeA,
                                                           A,
                                                           lda,
                                                           dataTypeTau,
                                                           tau,
                                                           computeType,
                                                           &lworkOnDevice,
                                                           &lworkOnHost));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lworkOnDevice));
    }

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    if(dataTypeA == HIP_R_32F && dataTypeTau == HIP_R_32F && computeType == HIP_R_32F)
    {
        return hipsolver::rocblas2hip_status(
            rocsolver_sgeqrf_64((rocblas_handle)handle, m, n, (float*)A, lda, (float*)tau));
    }
    else if(dataTypeA == HIP_R_64F && dataTypeTau == HIP_R_64F && computeType == HIP_R_64F)
    {
        return hipsolver::rocblas2hip_status(
            rocsolver_dgeqrf_64((rocblas_handle)handle, m, n, (double*)A, lda, (double*)tau));
    }
    else if(dataTypeA == HIP_C_32F && dataTypeTau == HIP_C_32F && computeType == HIP_C_32F)
    {
        return hipsolver::rocblas2hip_status(rocsolver_cgeqrf_64((rocblas_handle)handle,
                                                                 m,
                                                                 n,
                                                                 (rocblas_float_complex*)A,
                                                                 lda,
                                                                 (rocblas_float_complex*)tau));
    }
    else if(dataTypeA == HIP_C_64F && dataTypeTau == HIP_C_64F && computeType == HIP_C_64F)
    {
        return hipsolver::rocblas2hip_status(rocsolver_zgeqrf_64((rocblas_handle)handle,
                                                                 m,
                                                                 n,
                                                                 (rocblas_double_complex*)A,
                                                                 lda,
                                                                 (rocblas_double_complex*)tau));
    }
    else
        return HIPSOLVER_STATUS_INVALID_ENUM;
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
    if(!lworkOnDevice || !lworkOnHost)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lworkOnDevice = 0;
    *lworkOnHost   = 0;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status;
    if(dataTypeA == HIP_R_32F && computeType == HIP_R_32F)
    {
        status = hipsolver::rocblas2hip_status(
            rocsolver_sgetrf_info32((rocblas_handle)handle, m, n, nullptr, lda, nullptr, nullptr));
        rocsolver_sgetrf_npvt_info32((rocblas_handle)handle, m, n, nullptr, lda, nullptr);
    }
    else if(dataTypeA == HIP_R_64F && computeType == HIP_R_64F)
    {
        status = hipsolver::rocblas2hip_status(
            rocsolver_dgetrf_info32((rocblas_handle)handle, m, n, nullptr, lda, nullptr, nullptr));
        rocsolver_dgetrf_npvt_info32((rocblas_handle)handle, m, n, nullptr, lda, nullptr);
    }
    else if(dataTypeA == HIP_C_32F && computeType == HIP_C_32F)
    {
        status = hipsolver::rocblas2hip_status(
            rocsolver_cgetrf_info32((rocblas_handle)handle, m, n, nullptr, lda, nullptr, nullptr));
        rocsolver_cgetrf_npvt_info32((rocblas_handle)handle, m, n, nullptr, lda, nullptr);
    }
    else if(dataTypeA == HIP_C_64F && computeType == HIP_C_64F)
    {
        status = hipsolver::rocblas2hip_status(
            rocsolver_zgetrf_info32((rocblas_handle)handle, m, n, nullptr, lda, nullptr, nullptr));
        rocsolver_zgetrf_npvt_info32((rocblas_handle)handle, m, n, nullptr, lda, nullptr);
    }
    else
        return HIPSOLVER_STATUS_INVALID_ENUM;
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, lworkOnDevice);

    return status;
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

    if(workOnDevice && lworkOnDevice)
        CHECK_ROCBLAS_ERROR(
            rocblas_set_workspace((rocblas_handle)handle, workOnDevice, lworkOnDevice));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverDnXgetrf_bufferSize((rocblas_handle)handle,
                                                           params,
                                                           m,
                                                           n,
                                                           dataTypeA,
                                                           A,
                                                           lda,
                                                           computeType,
                                                           &lworkOnDevice,
                                                           &lworkOnHost));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lworkOnDevice));
    }

    if(devIpiv != nullptr)
    {
        if(dataTypeA == HIP_R_32F && computeType == HIP_R_32F)
        {
            return hipsolver::rocblas2hip_status(rocsolver_sgetrf_info32(
                (rocblas_handle)handle, m, n, (float*)A, lda, devIpiv, devInfo));
        }
        else if(dataTypeA == HIP_R_64F && computeType == HIP_R_64F)
        {
            return hipsolver::rocblas2hip_status(rocsolver_dgetrf_info32(
                (rocblas_handle)handle, m, n, (double*)A, lda, devIpiv, devInfo));
        }
        else if(dataTypeA == HIP_C_32F && computeType == HIP_C_32F)
        {
            return hipsolver::rocblas2hip_status(rocsolver_cgetrf_info32(
                (rocblas_handle)handle, m, n, (rocblas_float_complex*)A, lda, devIpiv, devInfo));
        }
        else if(dataTypeA == HIP_C_64F && computeType == HIP_C_64F)
        {
            return hipsolver::rocblas2hip_status(rocsolver_zgetrf_info32(
                (rocblas_handle)handle, m, n, (rocblas_double_complex*)A, lda, devIpiv, devInfo));
        }
        else
            return HIPSOLVER_STATUS_INVALID_ENUM;
    }
    else
    {
        if(dataTypeA == HIP_R_32F && computeType == HIP_R_32F)
        {
            return hipsolver::rocblas2hip_status(rocsolver_sgetrf_npvt_info32(
                (rocblas_handle)handle, m, n, (float*)A, lda, devInfo));
        }
        else if(dataTypeA == HIP_R_64F && computeType == HIP_R_64F)
        {
            return hipsolver::rocblas2hip_status(rocsolver_dgetrf_npvt_info32(
                (rocblas_handle)handle, m, n, (double*)A, lda, devInfo));
        }
        else if(dataTypeA == HIP_C_32F && computeType == HIP_C_32F)
        {
            return hipsolver::rocblas2hip_status(rocsolver_cgetrf_npvt_info32(
                (rocblas_handle)handle, m, n, (rocblas_float_complex*)A, lda, devInfo));
        }
        else if(dataTypeA == HIP_C_64F && computeType == HIP_C_64F)
        {
            return hipsolver::rocblas2hip_status(rocsolver_zgetrf_npvt_info32(
                (rocblas_handle)handle, m, n, (rocblas_double_complex*)A, lda, devInfo));
        }
        else
            return HIPSOLVER_STATUS_INVALID_ENUM;
    }
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

/******************** GETRS ********************/
hipsolverStatus_t hipsolverInternalXgetrs_bufferSize(hipsolverHandle_t    handle,
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
                                                     size_t*              lwork)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!params)
        return HIPSOLVER_STATUS_INVALID_VALUE;
    if(!lwork)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lwork = 0;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status;
    if(dataTypeA == HIP_R_32F && dataTypeB == HIP_R_32F)
    {
        status = hipsolver::rocblas2hip_status(
            rocsolver_sgetrs_64((rocblas_handle)handle,
                                hipsolver::hip2rocblas_operation(trans),
                                n,
                                nrhs,
                                nullptr,
                                lda,
                                nullptr,
                                nullptr,
                                ldb));
    }
    else if(dataTypeA == HIP_R_64F && dataTypeB == HIP_R_64F)
    {
        status = hipsolver::rocblas2hip_status(
            rocsolver_dgetrs_64((rocblas_handle)handle,
                                hipsolver::hip2rocblas_operation(trans),
                                n,
                                nrhs,
                                nullptr,
                                lda,
                                nullptr,
                                nullptr,
                                ldb));
    }
    else if(dataTypeA == HIP_C_32F && dataTypeB == HIP_C_32F)
    {
        status = hipsolver::rocblas2hip_status(
            rocsolver_cgetrs_64((rocblas_handle)handle,
                                hipsolver::hip2rocblas_operation(trans),
                                n,
                                nrhs,
                                nullptr,
                                lda,
                                nullptr,
                                nullptr,
                                ldb));
    }
    else if(dataTypeA == HIP_C_64F && dataTypeB == HIP_C_64F)
    {
        status = hipsolver::rocblas2hip_status(
            rocsolver_zgetrs_64((rocblas_handle)handle,
                                hipsolver::hip2rocblas_operation(trans),
                                n,
                                nrhs,
                                nullptr,
                                lda,
                                nullptr,
                                nullptr,
                                ldb));
    }
    else
        return HIPSOLVER_STATUS_INVALID_ENUM;
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, lwork);

    return status;
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

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
    size_t lwork;
    CHECK_HIPSOLVER_ERROR(hipsolverInternalXgetrs_bufferSize((rocblas_handle)handle,
                                                             params,
                                                             trans,
                                                             n,
                                                             nrhs,
                                                             dataTypeA,
                                                             A,
                                                             lda,
                                                             devIpiv,
                                                             dataTypeB,
                                                             B,
                                                             ldb,
                                                             &lwork));
    CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork));

    CHECK_ROCBLAS_ERROR(hipsolverZeroInfo((rocblas_handle)handle, devInfo, 1));

    if(dataTypeA == HIP_R_32F && dataTypeB == HIP_R_32F)
    {
        return hipsolver::rocblas2hip_status(
            rocsolver_sgetrs_64((rocblas_handle)handle,
                                hipsolver::hip2rocblas_operation(trans),
                                n,
                                nrhs,
                                (float*)const_cast<void*>(A),
                                lda,
                                const_cast<int64_t*>(devIpiv),
                                (float*)B,
                                ldb));
    }
    else if(dataTypeA == HIP_R_64F && dataTypeB == HIP_R_64F)
    {
        return hipsolver::rocblas2hip_status(
            rocsolver_dgetrs_64((rocblas_handle)handle,
                                hipsolver::hip2rocblas_operation(trans),
                                n,
                                nrhs,
                                (double*)const_cast<void*>(A),
                                lda,
                                const_cast<int64_t*>(devIpiv),
                                (double*)B,
                                ldb));
    }
    else if(dataTypeA == HIP_C_32F && dataTypeB == HIP_C_32F)
    {
        return hipsolver::rocblas2hip_status(
            rocsolver_cgetrs_64((rocblas_handle)handle,
                                hipsolver::hip2rocblas_operation(trans),
                                n,
                                nrhs,
                                (rocblas_float_complex*)const_cast<void*>(A),
                                lda,
                                const_cast<int64_t*>(devIpiv),
                                (rocblas_float_complex*)B,
                                ldb));
    }
    else if(dataTypeA == HIP_C_64F && dataTypeB == HIP_C_64F)
    {
        return hipsolver::rocblas2hip_status(
            rocsolver_zgetrs_64((rocblas_handle)handle,
                                hipsolver::hip2rocblas_operation(trans),
                                n,
                                nrhs,
                                (rocblas_double_complex*)const_cast<void*>(A),
                                lda,
                                const_cast<int64_t*>(devIpiv),
                                (rocblas_double_complex*)B,
                                ldb));
    }
    else
        return HIPSOLVER_STATUS_INVALID_ENUM;
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

/******************** POTRF ********************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXpotrf_bufferSize(hipsolverDnHandle_t handle,
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
    if(!lworkOnDevice || !lworkOnHost)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    *lworkOnDevice = 0;
    *lworkOnHost   = 0;

    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status;

    if(dataTypeA == HIP_R_32F && computeType == HIP_R_32F)
    {
        status = hipsolver::rocblas2hip_status(rocsolver_spotrf_64(
            (rocblas_handle)handle, hipsolver::hip2rocblas_fill(uplo), n, nullptr, lda, nullptr));
    }
    else if(dataTypeA == HIP_R_64F && computeType == HIP_R_64F)
    {
        status = hipsolver::rocblas2hip_status(rocsolver_dpotrf_64(
            (rocblas_handle)handle, hipsolver::hip2rocblas_fill(uplo), n, nullptr, lda, nullptr));
    }
    else if(dataTypeA == HIP_C_32F && computeType == HIP_C_32F)
    {
        status = hipsolver::rocblas2hip_status(rocsolver_cpotrf_64(
            (rocblas_handle)handle, hipsolver::hip2rocblas_fill(uplo), n, nullptr, lda, nullptr));
    }
    else if(dataTypeA == HIP_C_64F && computeType == HIP_C_64F)
    {
        status = hipsolver::rocblas2hip_status(rocsolver_zpotrf_64(
            (rocblas_handle)handle, hipsolver::hip2rocblas_fill(uplo), n, nullptr, lda, nullptr));
    }
    else
        return HIPSOLVER_STATUS_INVALID_ENUM;
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, lworkOnDevice);
    return status;
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXpotrf(hipsolverDnHandle_t handle,
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

    if(workOnDevice && lworkOnDevice)
        CHECK_ROCBLAS_ERROR(
            rocblas_set_workspace((rocblas_handle)handle, workOnDevice, lworkOnDevice));
    else
    {
        CHECK_HIPSOLVER_ERROR(hipsolverDnXpotrf_bufferSize((rocblas_handle)handle,
                                                           params,
                                                           uplo,
                                                           n,
                                                           dataTypeA,
                                                           A,
                                                           lda,
                                                           computeType,
                                                           &lworkOnDevice,
                                                           &lworkOnHost));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lworkOnDevice));
    }

    if(dataTypeA == HIP_R_32F && computeType == HIP_R_32F)
    {
        return hipsolver::rocblas2hip_status(rocsolver_spotrf_info32(
            (rocblas_handle)handle, hipsolver::hip2rocblas_fill(uplo), n, (float*)A, lda, info));
    }
    else if(dataTypeA == HIP_R_64F && computeType == HIP_R_64F)
    {
        return hipsolver::rocblas2hip_status(rocsolver_dpotrf_info32(
            (rocblas_handle)handle, hipsolver::hip2rocblas_fill(uplo), n, (double*)A, lda, info));
    }
    else if(dataTypeA == HIP_C_32F && computeType == HIP_C_32F)
    {
        return hipsolver::rocblas2hip_status(
            rocsolver_cpotrf_info32((rocblas_handle)handle,
                                    hipsolver::hip2rocblas_fill(uplo),
                                    n,
                                    (rocblas_float_complex*)A,
                                    lda,
                                    info));
    }
    else if(dataTypeA == HIP_C_64F && computeType == HIP_C_64F)
    {
        return hipsolver::rocblas2hip_status(
            rocsolver_zpotrf_info32((rocblas_handle)handle,
                                    hipsolver::hip2rocblas_fill(uplo),
                                    n,
                                    (rocblas_double_complex*)A,
                                    lda,
                                    info));
    }
    else
        return HIPSOLVER_STATUS_INVALID_ENUM;
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

/******************** POTRS ********************/
HIPSOLVER_EXPORT hipsolverStatus_t hipsolverDnXpotrs(hipsolverDnHandle_t handle,
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

    if(dataTypeA == HIP_R_32F && dataTypeB == HIP_R_32F)
    {
        return hipsolver::rocblas2hip_status(rocsolver_spotrs((rocblas_handle)handle,
                                                              hipsolver::hip2rocblas_fill(uplo),
                                                              n,
                                                              nrhs,
                                                              (float*)A,
                                                              lda,
                                                              (float*)B,
                                                              ldb));
    }
    else if(dataTypeA == HIP_R_64F && dataTypeB == HIP_R_64F)
    {
        return hipsolver::rocblas2hip_status(rocsolver_dpotrs((rocblas_handle)handle,
                                                              hipsolver::hip2rocblas_fill(uplo),
                                                              n,
                                                              nrhs,
                                                              (double*)A,
                                                              lda,
                                                              (double*)B,
                                                              ldb));
    }
    else if(dataTypeA == HIP_C_32F && dataTypeB == HIP_C_32F)
    {
        return hipsolver::rocblas2hip_status(rocsolver_cpotrs((rocblas_handle)handle,
                                                              hipsolver::hip2rocblas_fill(uplo),
                                                              n,
                                                              nrhs,
                                                              (rocblas_float_complex*)A,
                                                              lda,
                                                              (rocblas_float_complex*)B,
                                                              ldb));
    }
    else if(dataTypeA == HIP_C_64F && dataTypeB == HIP_C_64F)
    {
        return hipsolver::rocblas2hip_status(rocsolver_zpotrs((rocblas_handle)handle,
                                                              hipsolver::hip2rocblas_fill(uplo),
                                                              n,
                                                              nrhs,
                                                              (rocblas_double_complex*)A,
                                                              lda,
                                                              (rocblas_double_complex*)B,
                                                              ldb));
    }
    else
        return HIPSOLVER_STATUS_INVALID_ENUM;
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

} //extern C
