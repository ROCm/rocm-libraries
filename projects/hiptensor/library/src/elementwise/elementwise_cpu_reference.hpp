/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <hip/library_types.h>
#include <vector>

#include <hiptensor/hiptensor.h>

HIPTENSOR_EXPORT hiptensorStatus_t
    hiptensorElementwisePermuteReference(const void*                       alpha,
                                         const void*                       A,
                                         const hiptensorTensorDescriptor_t descA,
                                         const int32_t                     modeA[],
                                         const hiptensorOperator_t         opA,
                                         void*                             B,
                                         const hiptensorTensorDescriptor_t descB,
                                         const int32_t                     modeB[],
                                         const hiptensorDataType_t         typeScalar,
                                         const hipStream_t                 stream);

HIPTENSOR_EXPORT hiptensorStatus_t
    hiptensorElementwiseBinaryOpReference(const void*                       alpha,
                                          const void*                       A,
                                          const hiptensorTensorDescriptor_t descA,
                                          const int32_t                     modeA[],
                                          const hiptensorOperator_t         opA,
                                          const void*                       gamma,
                                          const void*                       C,
                                          const hiptensorTensorDescriptor_t descC,
                                          const int32_t                     modeC[],
                                          const hiptensorOperator_t         opC,
                                          void*                             D,
                                          const hiptensorTensorDescriptor_t descD,
                                          const int32_t                     modeD[],
                                          hiptensorOperator_t               opAC,
                                          hiptensorDataType_t               typeScalar,
                                          hipStream_t                       stream);
HIPTENSOR_EXPORT hiptensorStatus_t
    hiptensorElementwiseTrinaryOpReference(const void*                       alpha,
                                           const void*                       A,
                                           const hiptensorTensorDescriptor_t descA,
                                           const int32_t                     modeA[],
                                           const hiptensorOperator_t         opA,
                                           const void*                       beta,
                                           const void*                       B,
                                           const hiptensorTensorDescriptor_t descB,
                                           const int32_t                     modeB[],
                                           const hiptensorOperator_t         opB,
                                           const void*                       gamma,
                                           const void*                       C,
                                           const hiptensorTensorDescriptor_t descC,
                                           const int32_t                     modeC[],
                                           const hiptensorOperator_t         opC,
                                           void*                             D,
                                           const hiptensorTensorDescriptor_t descD,
                                           const int32_t                     modeD[],
                                           hiptensorOperator_t               opAB,
                                           hiptensorOperator_t               opABC,
                                           hiptensorDataType_t               typeScalar,
                                           hipStream_t                       stream);
