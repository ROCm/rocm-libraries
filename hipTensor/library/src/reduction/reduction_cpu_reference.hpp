/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPTENSOR_REDUCTION_CPU_REFERENCE_HPP
#define HIPTENSOR_REDUCTION_CPU_REFERENCE_HPP

#include <hip/library_types.h>
#include <vector>

#include <hiptensor/hiptensor.hpp>

hiptensorStatus_t hiptensorReductionReference(const void*                        alpha,
                                              const void*                        A,
                                              const hiptensorTensorDescriptor_t* descA,
                                              const int32_t                      modeA[],
                                              const void*                        beta,
                                              const void*                        C,
                                              const hiptensorTensorDescriptor_t* descC,
                                              const int32_t                      modeC[],
                                              void*                              D,
                                              const hiptensorTensorDescriptor_t* descD,
                                              const int32_t                      modeD[],
                                              hiptensorOperator_t                opReduce,
                                              hiptensorComputeType_t             typeCompute,
                                              hipStream_t                        stream);
#endif // HIPTENSOR_REDUCTION_CPU_REFERENCE_HPP
