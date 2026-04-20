/* ************************************************************************
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <cstddef>
#include <functional>
#include <iostream>
#include <limits>
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

/******************** SYEVD ********************/
hipsolverStatus_t hipsolverDnXsyevd_bufferSize(hipsolverDnHandle_t handle,
                                               hipsolverDnParams_t params,
                                               hipsolverEigMode_t  jobz,
                                               hipsolverFillMode_t uplo,
                                               int64_t             n,
                                               hipDataType         dataTypeA,
                                               const void*         A,
                                               int64_t             lda,
                                               hipDataType         dataTypeW,
                                               const void*         W,
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

    // TODO: Update to call 64-bit rocsolver_*syevd_64 / rocsolver_*heevd_64 once available in rocSOLVER.
    // Currently rocSOLVER only has 32-bit versions, so we cast int64_t to rocblas_int.
    auto const MAX_INT             = std::numeric_limits<int32_t>::max();
    bool const is_integer_overflow = (int64_t(lda) * n) > MAX_INT;
    if(is_integer_overflow)
        return HIPSOLVER_STATUS_NOT_SUPPORTED;

    size_t sz;
    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    hipsolverStatus_t status;

    if(dataTypeA == HIP_R_32F && dataTypeW == HIP_R_32F && computeType == HIP_R_32F)
    {
        status = hipsolver::rocblas2hip_status(rocsolver_ssyevd((rocblas_handle)handle,
                                                                hipsolver::hip2rocblas_evect(jobz),
                                                                hipsolver::hip2rocblas_fill(uplo),
                                                                static_cast<rocblas_int>(n),
                                                                nullptr,
                                                                static_cast<rocblas_int>(lda),
                                                                nullptr,
                                                                nullptr,
                                                                nullptr));
    }
    else if(dataTypeA == HIP_R_64F && dataTypeW == HIP_R_64F && computeType == HIP_R_64F)
    {
        status = hipsolver::rocblas2hip_status(rocsolver_dsyevd((rocblas_handle)handle,
                                                                hipsolver::hip2rocblas_evect(jobz),
                                                                hipsolver::hip2rocblas_fill(uplo),
                                                                static_cast<rocblas_int>(n),
                                                                nullptr,
                                                                static_cast<rocblas_int>(lda),
                                                                nullptr,
                                                                nullptr,
                                                                nullptr));
    }
    else if(dataTypeA == HIP_C_32F && dataTypeW == HIP_R_32F && computeType == HIP_C_32F)
    {
        status = hipsolver::rocblas2hip_status(rocsolver_cheevd((rocblas_handle)handle,
                                                                hipsolver::hip2rocblas_evect(jobz),
                                                                hipsolver::hip2rocblas_fill(uplo),
                                                                static_cast<rocblas_int>(n),
                                                                nullptr,
                                                                static_cast<rocblas_int>(lda),
                                                                nullptr,
                                                                nullptr,
                                                                nullptr));
    }
    else if(dataTypeA == HIP_C_64F && dataTypeW == HIP_R_64F && computeType == HIP_C_64F)
    {
        status = hipsolver::rocblas2hip_status(rocsolver_zheevd((rocblas_handle)handle,
                                                                hipsolver::hip2rocblas_evect(jobz),
                                                                hipsolver::hip2rocblas_fill(uplo),
                                                                static_cast<rocblas_int>(n),
                                                                nullptr,
                                                                static_cast<rocblas_int>(lda),
                                                                nullptr,
                                                                nullptr,
                                                                nullptr));
    }
    else
    {
        rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);
        return HIPSOLVER_STATUS_INVALID_ENUM;
    }

    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    // Additional space for E array (off-diagonal elements, size n, real-valued)
    size_t size_E = 0;
    if(n > 0)
    {
        if(dataTypeA == HIP_R_32F || dataTypeA == HIP_C_32F)
            size_E = sizeof(float) * n;
        else
            size_E = sizeof(double) * n;
    }

    // Combine workspace and E array sizes
    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_set_optimal_device_memory_size((rocblas_handle)handle, sz, size_E);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, lworkOnDevice);

    return status;
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverDnXsyevd(hipsolverDnHandle_t handle,
                                    hipsolverDnParams_t params,
                                    hipsolverEigMode_t  jobz,
                                    hipsolverFillMode_t uplo,
                                    int64_t             n,
                                    hipDataType         dataTypeA,
                                    void*               A,
                                    int64_t             lda,
                                    hipDataType         dataTypeW,
                                    void*               W,
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

    // Validate datatype combination before workspace setup
    if(!((dataTypeA == HIP_R_32F && dataTypeW == HIP_R_32F && computeType == HIP_R_32F)
         || (dataTypeA == HIP_R_64F && dataTypeW == HIP_R_64F && computeType == HIP_R_64F)
         || (dataTypeA == HIP_C_32F && dataTypeW == HIP_R_32F && computeType == HIP_C_32F)
         || (dataTypeA == HIP_C_64F && dataTypeW == HIP_R_64F && computeType == HIP_C_64F)))
        return HIPSOLVER_STATUS_INVALID_ENUM;

    // TODO: Update to call 64-bit rocsolver_*syevd_64 / rocsolver_*heevd_64 once available in rocSOLVER.
    // Currently rocSOLVER only has 32-bit versions, so we cast int64_t to rocblas_int.
    auto const MAX_INT             = std::numeric_limits<int32_t>::max();
    bool const is_integer_overflow = (int64_t(lda) * n) > MAX_INT;
    if(is_integer_overflow)
        return HIPSOLVER_STATUS_NOT_SUPPORTED;

    // Determine the real type size for E array allocation
    size_t realTypeSize
        = (dataTypeA == HIP_R_32F || dataTypeA == HIP_C_32F) ? sizeof(float) : sizeof(double);

    rocblas_device_malloc mem((rocblas_handle)handle);
    void*                 E = nullptr;

    if(workOnDevice && lworkOnDevice)
    {
        // Use provided workspace: carve out E array from the beginning
        size_t e_bytes = realTypeSize * n;
        if(lworkOnDevice < e_bytes)
            return HIPSOLVER_STATUS_INVALID_VALUE;

        E = workOnDevice;
        if(n > 0)
            workOnDevice = static_cast<std::byte*>(workOnDevice) + e_bytes;

        CHECK_ROCBLAS_ERROR(
            rocblas_set_workspace((rocblas_handle)handle, workOnDevice, lworkOnDevice - e_bytes));
    }
    else
    {
        size_t lwork_device, lwork_host;
        CHECK_HIPSOLVER_ERROR(hipsolverDnXsyevd_bufferSize(handle,
                                                           params,
                                                           jobz,
                                                           uplo,
                                                           n,
                                                           dataTypeA,
                                                           A,
                                                           lda,
                                                           dataTypeW,
                                                           W,
                                                           computeType,
                                                           &lwork_device,
                                                           &lwork_host));
        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lwork_device));

        mem = rocblas_device_malloc((rocblas_handle)handle, realTypeSize * n);
        if(!mem)
            return HIPSOLVER_STATUS_ALLOC_FAILED;
        E = (void*)mem[0];
    }
    if(dataTypeA == HIP_R_32F && dataTypeW == HIP_R_32F && computeType == HIP_R_32F)
    {
        return hipsolver::rocblas2hip_status(rocsolver_ssyevd((rocblas_handle)handle,
                                                              hipsolver::hip2rocblas_evect(jobz),
                                                              hipsolver::hip2rocblas_fill(uplo),
                                                              static_cast<rocblas_int>(n),
                                                              (float*)A,
                                                              static_cast<rocblas_int>(lda),
                                                              (float*)W,
                                                              (float*)E,
                                                              devInfo));
    }
    else if(dataTypeA == HIP_R_64F && dataTypeW == HIP_R_64F && computeType == HIP_R_64F)
    {
        return hipsolver::rocblas2hip_status(rocsolver_dsyevd((rocblas_handle)handle,
                                                              hipsolver::hip2rocblas_evect(jobz),
                                                              hipsolver::hip2rocblas_fill(uplo),
                                                              static_cast<rocblas_int>(n),
                                                              (double*)A,
                                                              static_cast<rocblas_int>(lda),
                                                              (double*)W,
                                                              (double*)E,
                                                              devInfo));
    }
    else if(dataTypeA == HIP_C_32F && dataTypeW == HIP_R_32F && computeType == HIP_C_32F)
    {
        return hipsolver::rocblas2hip_status(rocsolver_cheevd((rocblas_handle)handle,
                                                              hipsolver::hip2rocblas_evect(jobz),
                                                              hipsolver::hip2rocblas_fill(uplo),
                                                              static_cast<rocblas_int>(n),
                                                              (rocblas_float_complex*)A,
                                                              static_cast<rocblas_int>(lda),
                                                              (float*)W,
                                                              (float*)E,
                                                              devInfo));
    }
    else if(dataTypeA == HIP_C_64F && dataTypeW == HIP_R_64F && computeType == HIP_C_64F)
    {
        return hipsolver::rocblas2hip_status(rocsolver_zheevd((rocblas_handle)handle,
                                                              hipsolver::hip2rocblas_evect(jobz),
                                                              hipsolver::hip2rocblas_fill(uplo),
                                                              static_cast<rocblas_int>(n),
                                                              (rocblas_double_complex*)A,
                                                              static_cast<rocblas_int>(lda),
                                                              (double*)W,
                                                              (double*)E,
                                                              devInfo));
    }
    else
        return HIPSOLVER_STATUS_INVALID_ENUM;
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

/******************** SYEV_BATCHED ********************/
hipsolverStatus_t hipsolverDnXsyevBatched_bufferSize(hipsolverDnHandle_t handle,
                                                     hipsolverDnParams_t params,
                                                     hipsolverEigMode_t  jobz,
                                                     hipsolverFillMode_t uplo,
                                                     int64_t             n,
                                                     hipDataType         dataTypeA,
                                                     const void*         A,
                                                     int64_t             lda,
                                                     hipDataType         dataTypeW,
                                                     const void*         W,
                                                     hipDataType         computeType,
                                                     size_t*             lworkOnDevice,
                                                     size_t*             lworkOnHost,
                                                     int64_t             batchSize)
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

    // rocSOLVER does not yet have 64-bit syev_strided_batched; validate args fit in 32-bit
    if(n > INT_MAX || lda > INT_MAX || batchSize > INT_MAX || int64_t(lda) * n > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    size_t sz;
    rocblas_start_device_memory_size_query((rocblas_handle)handle);

    hipsolverStatus_t status  = HIPSOLVER_STATUS_SUCCESS;
    int64_t           strideA = int64_t(lda) * n;
    int64_t           strideW = n;
    int64_t           strideE = n;

    // Pass nullptr for E during workspace query
    if(dataTypeA == HIP_R_32F && dataTypeW == HIP_R_32F && computeType == HIP_R_32F)
    {
        status = hipsolver::rocblas2hip_status(
            rocsolver_ssyev_strided_batched((rocblas_handle)handle,
                                            hipsolver::hip2rocblas_evect(jobz),
                                            hipsolver::hip2rocblas_fill(uplo),
                                            static_cast<rocblas_int>(n),
                                            nullptr,
                                            static_cast<rocblas_int>(lda),
                                            strideA,
                                            nullptr,
                                            strideW,
                                            nullptr,
                                            strideE,
                                            nullptr,
                                            static_cast<rocblas_int>(batchSize)));
    }
    else if(dataTypeA == HIP_R_64F && dataTypeW == HIP_R_64F && computeType == HIP_R_64F)
    {
        status = hipsolver::rocblas2hip_status(
            rocsolver_dsyev_strided_batched((rocblas_handle)handle,
                                            hipsolver::hip2rocblas_evect(jobz),
                                            hipsolver::hip2rocblas_fill(uplo),
                                            static_cast<rocblas_int>(n),
                                            nullptr,
                                            static_cast<rocblas_int>(lda),
                                            strideA,
                                            nullptr,
                                            strideW,
                                            nullptr,
                                            strideE,
                                            nullptr,
                                            static_cast<rocblas_int>(batchSize)));
    }
    else if(dataTypeA == HIP_C_32F && dataTypeW == HIP_R_32F && computeType == HIP_C_32F)
    {
        status = hipsolver::rocblas2hip_status(
            rocsolver_cheev_strided_batched((rocblas_handle)handle,
                                            hipsolver::hip2rocblas_evect(jobz),
                                            hipsolver::hip2rocblas_fill(uplo),
                                            static_cast<rocblas_int>(n),
                                            nullptr,
                                            static_cast<rocblas_int>(lda),
                                            strideA,
                                            nullptr,
                                            strideW,
                                            nullptr,
                                            strideE,
                                            nullptr,
                                            static_cast<rocblas_int>(batchSize)));
    }
    else if(dataTypeA == HIP_C_64F && dataTypeW == HIP_R_64F && computeType == HIP_C_64F)
    {
        status = hipsolver::rocblas2hip_status(
            rocsolver_zheev_strided_batched((rocblas_handle)handle,
                                            hipsolver::hip2rocblas_evect(jobz),
                                            hipsolver::hip2rocblas_fill(uplo),
                                            static_cast<rocblas_int>(n),
                                            nullptr,
                                            static_cast<rocblas_int>(lda),
                                            strideA,
                                            nullptr,
                                            strideW,
                                            nullptr,
                                            strideE,
                                            nullptr,
                                            static_cast<rocblas_int>(batchSize)));
    }
    else
        return HIPSOLVER_STATUS_NOT_SUPPORTED;

    rocblas_stop_device_memory_size_query((rocblas_handle)handle, &sz);

    if(status != HIPSOLVER_STATUS_SUCCESS)
        return status;

    // space for E array
    size_t size_E = 0;
    if(n > 0 && batchSize > 0)
    {
        if(dataTypeW == HIP_R_32F)
            size_E = sizeof(float) * n * batchSize;
        else
            size_E = sizeof(double) * n * batchSize;
    }

    // update size
    rocblas_start_device_memory_size_query((rocblas_handle)handle);
    rocblas_set_optimal_device_memory_size((rocblas_handle)handle, sz, size_E);
    rocblas_stop_device_memory_size_query((rocblas_handle)handle, lworkOnDevice);

    return HIPSOLVER_STATUS_SUCCESS;
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

hipsolverStatus_t hipsolverDnXsyevBatched(hipsolverDnHandle_t handle,
                                          hipsolverDnParams_t params,
                                          hipsolverEigMode_t  jobz,
                                          hipsolverFillMode_t uplo,
                                          int64_t             n,
                                          hipDataType         dataTypeA,
                                          void*               A,
                                          int64_t             lda,
                                          hipDataType         dataTypeW,
                                          void*               W,
                                          hipDataType         computeType,
                                          void*               workOnDevice,
                                          size_t              lworkOnDevice,
                                          void*               workOnHost,
                                          size_t              lworkOnHost,
                                          int*                devInfo,
                                          int64_t             batchSize)
try
{
    if(!handle)
        return HIPSOLVER_STATUS_NOT_INITIALIZED;
    if(!params)
        return HIPSOLVER_STATUS_INVALID_VALUE;

    // rocSOLVER does not yet have 64-bit syev_strided_batched; validate args fit in 32-bit
    if(n > INT_MAX || lda > INT_MAX || batchSize > INT_MAX || int64_t(lda) * n > INT_MAX)
        return HIPSOLVER_STATUS_INTERNAL_ERROR;

    // Calculate E workspace size (E has same type as W -- real eigenvalues)
    size_t e_workspace_size = 0;
    if(dataTypeW == HIP_R_32F)
        e_workspace_size = sizeof(float) * n * batchSize;
    else if(dataTypeW == HIP_R_64F)
        e_workspace_size = sizeof(double) * n * batchSize;

    rocblas_device_malloc mem((rocblas_handle)handle);
    void*                 E_workspace;

    if(workOnDevice && lworkOnDevice)
    {
        if(lworkOnDevice < e_workspace_size)
            return HIPSOLVER_STATUS_INVALID_VALUE;

        // User provided workspace: E at the beginning, rocSOLVER workspace after
        E_workspace           = workOnDevice;
        void*  rocsolver_work = reinterpret_cast<std::byte*>(workOnDevice) + e_workspace_size;
        size_t lwork_computed = lworkOnDevice - e_workspace_size;
        CHECK_ROCBLAS_ERROR(
            rocblas_set_workspace((rocblas_handle)handle, rocsolver_work, lwork_computed));
    }
    else
    {
        // No user workspace: query required size, use managed workspace for rocSOLVER, device_malloc for E
        CHECK_HIPSOLVER_ERROR(hipsolverDnXsyevBatched_bufferSize(handle,
                                                                 params,
                                                                 jobz,
                                                                 uplo,
                                                                 n,
                                                                 dataTypeA,
                                                                 A,
                                                                 lda,
                                                                 dataTypeW,
                                                                 W,
                                                                 computeType,
                                                                 &lworkOnDevice,
                                                                 &lworkOnHost,
                                                                 batchSize));

        CHECK_ROCBLAS_ERROR(hipsolverManageWorkspace((rocblas_handle)handle, lworkOnDevice));

        mem = rocblas_device_malloc((rocblas_handle)handle, e_workspace_size);
        if(!mem)
            return HIPSOLVER_STATUS_ALLOC_FAILED;
        E_workspace = (void*)mem[0];
    }

    // TODO: Update to call 64-bit versions once rocSOLVER adds rocsolver_*syev_strided_batched_64
    // Currently rocSOLVER only has 32-bit versions, so we cast int64_t to rocblas_int

    // Calculate strides for strided-batched layout
    // Matrix A: each matrix is lda * n elements
    // Eigenvalues W: each vector is n elements
    // Workspace E: each vector is n elements
    int64_t strideA = int64_t(lda) * n;
    int64_t strideW = n;
    int64_t strideE = n;

    if(dataTypeA == HIP_R_32F && dataTypeW == HIP_R_32F && computeType == HIP_R_32F)
    {
        return hipsolver::rocblas2hip_status(
            rocsolver_ssyev_strided_batched((rocblas_handle)handle,
                                            hipsolver::hip2rocblas_evect(jobz),
                                            hipsolver::hip2rocblas_fill(uplo),
                                            static_cast<rocblas_int>(n),
                                            (float*)A,
                                            static_cast<rocblas_int>(lda),
                                            strideA,
                                            (float*)W,
                                            strideW,
                                            (float*)E_workspace,
                                            strideE,
                                            devInfo,
                                            static_cast<rocblas_int>(batchSize)));
    }
    else if(dataTypeA == HIP_R_64F && dataTypeW == HIP_R_64F && computeType == HIP_R_64F)
    {
        return hipsolver::rocblas2hip_status(
            rocsolver_dsyev_strided_batched((rocblas_handle)handle,
                                            hipsolver::hip2rocblas_evect(jobz),
                                            hipsolver::hip2rocblas_fill(uplo),
                                            static_cast<rocblas_int>(n),
                                            (double*)A,
                                            static_cast<rocblas_int>(lda),
                                            strideA,
                                            (double*)W,
                                            strideW,
                                            (double*)E_workspace,
                                            strideE,
                                            devInfo,
                                            static_cast<rocblas_int>(batchSize)));
    }
    else if(dataTypeA == HIP_C_32F && dataTypeW == HIP_R_32F && computeType == HIP_C_32F)
    {
        return hipsolver::rocblas2hip_status(
            rocsolver_cheev_strided_batched((rocblas_handle)handle,
                                            hipsolver::hip2rocblas_evect(jobz),
                                            hipsolver::hip2rocblas_fill(uplo),
                                            static_cast<rocblas_int>(n),
                                            (rocblas_float_complex*)A,
                                            static_cast<rocblas_int>(lda),
                                            strideA,
                                            (float*)W,
                                            strideW,
                                            (float*)E_workspace,
                                            strideE,
                                            devInfo,
                                            static_cast<rocblas_int>(batchSize)));
    }
    else if(dataTypeA == HIP_C_64F && dataTypeW == HIP_R_64F && computeType == HIP_C_64F)
    {
        return hipsolver::rocblas2hip_status(
            rocsolver_zheev_strided_batched((rocblas_handle)handle,
                                            hipsolver::hip2rocblas_evect(jobz),
                                            hipsolver::hip2rocblas_fill(uplo),
                                            static_cast<rocblas_int>(n),
                                            (rocblas_double_complex*)A,
                                            static_cast<rocblas_int>(lda),
                                            strideA,
                                            (double*)W,
                                            strideW,
                                            (double*)E_workspace,
                                            strideE,
                                            devInfo,
                                            static_cast<rocblas_int>(batchSize)));
    }
    else
    {
        return HIPSOLVER_STATUS_NOT_SUPPORTED;
    }
}
catch(...)
{
    return hipsolver::exception2hip_status();
}

} //extern C
