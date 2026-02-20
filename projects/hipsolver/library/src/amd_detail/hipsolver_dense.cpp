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

/*! \file
 *  \brief Implementation of the compatibility APIs that require especial calls
 *  to hipSOLVER on the rocSOLVER side.
 */

#include "hipsolver.h"
#include "lib_macros.hpp"
#include <algorithm>
#include <iostream>

extern "C" {

// gesvd
hipsolverStatus_t hipsolverDnSgesvd_bufferSize(hipsolverHandle_t handle, int m, int n, int* lwork)
{
    int temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9, temp10, temp11, temp12,
        temp13, temp14, temp15;

    CHECK_HIPSOLVER_ERROR(hipsolverSgesvd_bufferSize(handle, 'N', 'N', m, n, &temp1));
    CHECK_HIPSOLVER_ERROR(hipsolverSgesvd_bufferSize(handle, 'N', 'A', m, n, &temp2));
    CHECK_HIPSOLVER_ERROR(hipsolverSgesvd_bufferSize(handle, 'N', 'S', m, n, &temp3));
    CHECK_HIPSOLVER_ERROR(hipsolverSgesvd_bufferSize(handle, 'N', 'O', m, n, &temp4));
    CHECK_HIPSOLVER_ERROR(hipsolverSgesvd_bufferSize(handle, 'A', 'N', m, n, &temp5));
    CHECK_HIPSOLVER_ERROR(hipsolverSgesvd_bufferSize(handle, 'A', 'A', m, n, &temp6));
    CHECK_HIPSOLVER_ERROR(hipsolverSgesvd_bufferSize(handle, 'A', 'S', m, n, &temp7));
    CHECK_HIPSOLVER_ERROR(hipsolverSgesvd_bufferSize(handle, 'A', 'O', m, n, &temp8));
    CHECK_HIPSOLVER_ERROR(hipsolverSgesvd_bufferSize(handle, 'S', 'N', m, n, &temp9));
    CHECK_HIPSOLVER_ERROR(hipsolverSgesvd_bufferSize(handle, 'S', 'A', m, n, &temp10));
    CHECK_HIPSOLVER_ERROR(hipsolverSgesvd_bufferSize(handle, 'S', 'S', m, n, &temp11));
    CHECK_HIPSOLVER_ERROR(hipsolverSgesvd_bufferSize(handle, 'S', 'O', m, n, &temp12));
    CHECK_HIPSOLVER_ERROR(hipsolverSgesvd_bufferSize(handle, 'O', 'N', m, n, &temp13));
    CHECK_HIPSOLVER_ERROR(hipsolverSgesvd_bufferSize(handle, 'O', 'A', m, n, &temp14));
    CHECK_HIPSOLVER_ERROR(hipsolverSgesvd_bufferSize(handle, 'O', 'S', m, n, &temp15));

    *lwork = std::max({temp1,
                       temp2,
                       temp3,
                       temp4,
                       temp5,
                       temp6,
                       temp7,
                       temp8,
                       temp9,
                       temp10,
                       temp11,
                       temp12,
                       temp13,
                       temp14,
                       temp15});

    return HIPSOLVER_STATUS_SUCCESS;
}

hipsolverStatus_t hipsolverDnDgesvd_bufferSize(hipsolverHandle_t handle, int m, int n, int* lwork)
{
    int temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9, temp10, temp11, temp12,
        temp13, temp14, temp15;

    CHECK_HIPSOLVER_ERROR(hipsolverDgesvd_bufferSize(handle, 'N', 'N', m, n, &temp1));
    CHECK_HIPSOLVER_ERROR(hipsolverDgesvd_bufferSize(handle, 'N', 'A', m, n, &temp2));
    CHECK_HIPSOLVER_ERROR(hipsolverDgesvd_bufferSize(handle, 'N', 'S', m, n, &temp3));
    CHECK_HIPSOLVER_ERROR(hipsolverDgesvd_bufferSize(handle, 'N', 'O', m, n, &temp4));
    CHECK_HIPSOLVER_ERROR(hipsolverDgesvd_bufferSize(handle, 'A', 'N', m, n, &temp5));
    CHECK_HIPSOLVER_ERROR(hipsolverDgesvd_bufferSize(handle, 'A', 'A', m, n, &temp6));
    CHECK_HIPSOLVER_ERROR(hipsolverDgesvd_bufferSize(handle, 'A', 'S', m, n, &temp7));
    CHECK_HIPSOLVER_ERROR(hipsolverDgesvd_bufferSize(handle, 'A', 'O', m, n, &temp8));
    CHECK_HIPSOLVER_ERROR(hipsolverDgesvd_bufferSize(handle, 'S', 'N', m, n, &temp9));
    CHECK_HIPSOLVER_ERROR(hipsolverDgesvd_bufferSize(handle, 'S', 'A', m, n, &temp10));
    CHECK_HIPSOLVER_ERROR(hipsolverDgesvd_bufferSize(handle, 'S', 'S', m, n, &temp11));
    CHECK_HIPSOLVER_ERROR(hipsolverDgesvd_bufferSize(handle, 'S', 'O', m, n, &temp12));
    CHECK_HIPSOLVER_ERROR(hipsolverDgesvd_bufferSize(handle, 'O', 'N', m, n, &temp13));
    CHECK_HIPSOLVER_ERROR(hipsolverDgesvd_bufferSize(handle, 'O', 'A', m, n, &temp14));
    CHECK_HIPSOLVER_ERROR(hipsolverDgesvd_bufferSize(handle, 'O', 'S', m, n, &temp15));

    *lwork = std::max({temp1,
                       temp2,
                       temp3,
                       temp4,
                       temp5,
                       temp6,
                       temp7,
                       temp8,
                       temp9,
                       temp10,
                       temp11,
                       temp12,
                       temp13,
                       temp14,
                       temp15});

    return HIPSOLVER_STATUS_SUCCESS;
}

hipsolverStatus_t hipsolverDnCgesvd_bufferSize(hipsolverHandle_t handle, int m, int n, int* lwork)
{
    int temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9, temp10, temp11, temp12,
        temp13, temp14, temp15;

    CHECK_HIPSOLVER_ERROR(hipsolverCgesvd_bufferSize(handle, 'N', 'N', m, n, &temp1));
    CHECK_HIPSOLVER_ERROR(hipsolverCgesvd_bufferSize(handle, 'N', 'A', m, n, &temp2));
    CHECK_HIPSOLVER_ERROR(hipsolverCgesvd_bufferSize(handle, 'N', 'S', m, n, &temp3));
    CHECK_HIPSOLVER_ERROR(hipsolverCgesvd_bufferSize(handle, 'N', 'O', m, n, &temp4));
    CHECK_HIPSOLVER_ERROR(hipsolverCgesvd_bufferSize(handle, 'A', 'N', m, n, &temp5));
    CHECK_HIPSOLVER_ERROR(hipsolverCgesvd_bufferSize(handle, 'A', 'A', m, n, &temp6));
    CHECK_HIPSOLVER_ERROR(hipsolverCgesvd_bufferSize(handle, 'A', 'S', m, n, &temp7));
    CHECK_HIPSOLVER_ERROR(hipsolverCgesvd_bufferSize(handle, 'A', 'O', m, n, &temp8));
    CHECK_HIPSOLVER_ERROR(hipsolverCgesvd_bufferSize(handle, 'S', 'N', m, n, &temp9));
    CHECK_HIPSOLVER_ERROR(hipsolverCgesvd_bufferSize(handle, 'S', 'A', m, n, &temp10));
    CHECK_HIPSOLVER_ERROR(hipsolverCgesvd_bufferSize(handle, 'S', 'S', m, n, &temp11));
    CHECK_HIPSOLVER_ERROR(hipsolverCgesvd_bufferSize(handle, 'S', 'O', m, n, &temp12));
    CHECK_HIPSOLVER_ERROR(hipsolverCgesvd_bufferSize(handle, 'O', 'N', m, n, &temp13));
    CHECK_HIPSOLVER_ERROR(hipsolverCgesvd_bufferSize(handle, 'O', 'A', m, n, &temp14));
    CHECK_HIPSOLVER_ERROR(hipsolverCgesvd_bufferSize(handle, 'O', 'S', m, n, &temp15));

    *lwork = std::max({temp1,
                       temp2,
                       temp3,
                       temp4,
                       temp5,
                       temp6,
                       temp7,
                       temp8,
                       temp9,
                       temp10,
                       temp11,
                       temp12,
                       temp13,
                       temp14,
                       temp15});

    return HIPSOLVER_STATUS_SUCCESS;
}

hipsolverStatus_t hipsolverDnZgesvd_bufferSize(hipsolverHandle_t handle, int m, int n, int* lwork)
{
    int temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp9, temp10, temp11, temp12,
        temp13, temp14, temp15;

    CHECK_HIPSOLVER_ERROR(hipsolverZgesvd_bufferSize(handle, 'N', 'N', m, n, &temp1));
    CHECK_HIPSOLVER_ERROR(hipsolverZgesvd_bufferSize(handle, 'N', 'A', m, n, &temp2));
    CHECK_HIPSOLVER_ERROR(hipsolverZgesvd_bufferSize(handle, 'N', 'S', m, n, &temp3));
    CHECK_HIPSOLVER_ERROR(hipsolverZgesvd_bufferSize(handle, 'N', 'O', m, n, &temp4));
    CHECK_HIPSOLVER_ERROR(hipsolverZgesvd_bufferSize(handle, 'A', 'N', m, n, &temp5));
    CHECK_HIPSOLVER_ERROR(hipsolverZgesvd_bufferSize(handle, 'A', 'A', m, n, &temp6));
    CHECK_HIPSOLVER_ERROR(hipsolverZgesvd_bufferSize(handle, 'A', 'S', m, n, &temp7));
    CHECK_HIPSOLVER_ERROR(hipsolverZgesvd_bufferSize(handle, 'A', 'O', m, n, &temp8));
    CHECK_HIPSOLVER_ERROR(hipsolverZgesvd_bufferSize(handle, 'S', 'N', m, n, &temp9));
    CHECK_HIPSOLVER_ERROR(hipsolverZgesvd_bufferSize(handle, 'S', 'A', m, n, &temp10));
    CHECK_HIPSOLVER_ERROR(hipsolverZgesvd_bufferSize(handle, 'S', 'S', m, n, &temp11));
    CHECK_HIPSOLVER_ERROR(hipsolverZgesvd_bufferSize(handle, 'S', 'O', m, n, &temp12));
    CHECK_HIPSOLVER_ERROR(hipsolverZgesvd_bufferSize(handle, 'O', 'N', m, n, &temp13));
    CHECK_HIPSOLVER_ERROR(hipsolverZgesvd_bufferSize(handle, 'O', 'A', m, n, &temp14));
    CHECK_HIPSOLVER_ERROR(hipsolverZgesvd_bufferSize(handle, 'O', 'S', m, n, &temp15));

    *lwork = std::max({temp1,
                       temp2,
                       temp3,
                       temp4,
                       temp5,
                       temp6,
                       temp7,
                       temp8,
                       temp9,
                       temp10,
                       temp11,
                       temp12,
                       temp13,
                       temp14,
                       temp15});

    return HIPSOLVER_STATUS_SUCCESS;
}

// getrf
hipsolverStatus_t hipsolverDnSgetrf(hipsolverHandle_t handle,
                                    int               m,
                                    int               n,
                                    float*            A,
                                    int               lda,
                                    float*            work,
                                    int*              devIpiv,
                                    int*              devInfo)
{
    int lwork;
    CHECK_HIPSOLVER_ERROR(hipsolverSgetrf_bufferSize(handle, m, n, A, lda, &lwork));
    return hipsolverSgetrf(handle, m, n, A, lda, work, lwork, devIpiv, devInfo);
}

hipsolverStatus_t hipsolverDnDgetrf(hipsolverHandle_t handle,
                                    int               m,
                                    int               n,
                                    double*           A,
                                    int               lda,
                                    double*           work,
                                    int*              devIpiv,
                                    int*              devInfo)
{
    int lwork;
    CHECK_HIPSOLVER_ERROR(hipsolverDgetrf_bufferSize(handle, m, n, A, lda, &lwork));
    return hipsolverDgetrf(handle, m, n, A, lda, work, lwork, devIpiv, devInfo);
}

hipsolverStatus_t hipsolverDnCgetrf(hipsolverHandle_t handle,
                                    int               m,
                                    int               n,
                                    hipFloatComplex*  A,
                                    int               lda,
                                    hipFloatComplex*  work,
                                    int*              devIpiv,
                                    int*              devInfo)
{
    int lwork;
    CHECK_HIPSOLVER_ERROR(hipsolverCgetrf_bufferSize(handle, m, n, A, lda, &lwork));
    return hipsolverCgetrf(handle, m, n, A, lda, work, lwork, devIpiv, devInfo);
}

hipsolverStatus_t hipsolverDnZgetrf(hipsolverHandle_t handle,
                                    int               m,
                                    int               n,
                                    hipDoubleComplex* A,
                                    int               lda,
                                    hipDoubleComplex* work,
                                    int*              devIpiv,
                                    int*              devInfo)
{
    int lwork;
    CHECK_HIPSOLVER_ERROR(hipsolverZgetrf_bufferSize(handle, m, n, A, lda, &lwork));
    return hipsolverZgetrf(handle, m, n, A, lda, work, lwork, devIpiv, devInfo);
}

} //extern C
