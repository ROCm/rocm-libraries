/* ************************************************************************
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

// concaternate the two arguments, evaluating them first if they are macros
#define HIPSOLVER_CONCAT2_HELPER(a, b) a##b
#define HIPSOLVER_CONCAT2(a, b) HIPSOLVER_CONCAT2_HELPER(a, b)

#define HIPSOLVER_CONCAT4_HELPER(a, b, c, d) a##b##c##d
#define HIPSOLVER_CONCAT4(a, b, c, d) HIPSOLVER_CONCAT4_HELPER(a, b, c, d)

#if hipsolverVersionMinor < 10
#define hipsolverVersionMinor_PADDED HIPSOLVER_CONCAT2(0, hipsolverVersionMinor)
#else
#define hipsolverVersionMinor_PADDED hipsolverVersionMinor
#endif

#if hipsolverVersionPatch < 10
#define hipsolverVersionPatch_PADDED HIPSOLVER_CONCAT2(0, hipsolverVersionPatch)
#else
#define hipsolverVersionPatch_PADDED hipsolverVersionPatch
#endif

#ifndef HIPSOLVER_BEGIN_NAMESPACE
#define HIPSOLVER_BEGIN_NAMESPACE                                        \
    namespace hipsolver                                                  \
    {                                                                    \
        inline namespace HIPSOLVER_CONCAT4(v,                            \
                                           hipsolverVersionMajor,        \
                                           hipsolverVersionMinor_PADDED, \
                                           hipsolverVersionPatch_PADDED) \
        {
#define HIPSOLVER_END_NAMESPACE \
    }                           \
    }
#endif

#define CHECK_HIPSOLVER_ERROR(STATUS)           \
    do                                          \
    {                                           \
        hipsolverStatus_t _status = (STATUS);   \
        if(_status != HIPSOLVER_STATUS_SUCCESS) \
            return _status;                     \
    } while(0)

#define CHECK_ROCBLAS_ERROR(STATUS)                        \
    do                                                     \
    {                                                      \
        rocblas_status _status = (STATUS);                 \
        if(_status != rocblas_status_success)              \
            return hipsolver::rocblas2hip_status(_status); \
    } while(0)

#define CHECK_CUSOLVER_ERROR(STATUS)                    \
    do                                                  \
    {                                                   \
        cusolverStatus_t _status = (STATUS);            \
        if(_status != CUSOLVER_STATUS_SUCCESS)          \
            return hipsolver::cuda2hip_status(_status); \
    } while(0)

#define CHECK_HIP_ERROR(STATUS)                     \
    do                                              \
    {                                               \
        hipError_t _status = (STATUS);              \
        if(_status != hipSuccess)                   \
            return HIPSOLVER_STATUS_INTERNAL_ERROR; \
    } while(0)
