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
 *  to hipSOLVER on the cuSOLVER side.
 */

#include "hipsolver.h"

extern "C" {

// gesvd
hipsolverStatus_t hipsolverDnSgesvd_bufferSize(hipsolverHandle_t handle, int m, int n, int* lwork)
{
    return hipsolverSgesvd_bufferSize(handle, 'N', 'N', m, n, lwork);
}

hipsolverStatus_t hipsolverDnDgesvd_bufferSize(hipsolverHandle_t handle, int m, int n, int* lwork)
{
    return hipsolverDgesvd_bufferSize(handle, 'N', 'N', m, n, lwork);
}

hipsolverStatus_t hipsolverDnCgesvd_bufferSize(hipsolverHandle_t handle, int m, int n, int* lwork)
{
    return hipsolverCgesvd_bufferSize(handle, 'N', 'N', m, n, lwork);
}

hipsolverStatus_t hipsolverDnZgesvd_bufferSize(hipsolverHandle_t handle, int m, int n, int* lwork)
{
    return hipsolverZgesvd_bufferSize(handle, 'N', 'N', m, n, lwork);
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
    return hipsolverSgetrf(handle, m, n, A, lda, work, 0, devIpiv, devInfo);
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
    return hipsolverDgetrf(handle, m, n, A, lda, work, 0, devIpiv, devInfo);
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
    return hipsolverCgetrf(handle, m, n, A, lda, work, 0, devIpiv, devInfo);
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
    return hipsolverZgetrf(handle, m, n, A, lda, work, 0, devIpiv, devInfo);
}

} //extern C
