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

#include <vector>

#include "../elementwise/elementwise_cpu_reference.hpp"
#include "reduction_cpu_reference.hpp"
#include "reduction_cpu_reference_impl.hpp"
#include "reduction_cpu_reference_instances.hpp"

hiptensorStatus_t hiptensorReductionReference(const void*                       alpha,
                                              const void*                       A,
                                              const hiptensorTensorDescriptor_t descA,
                                              const int32_t                     modeA[],
                                              const hiptensorOperator_t         opA,
                                              const void*                       beta,
                                              const void*                       C,
                                              const hiptensorTensorDescriptor_t descC,
                                              const int32_t                     modeC[],
                                              const hiptensorOperator_t         opC,
                                              void*                             D,
                                              const hiptensorTensorDescriptor_t descD,
                                              const int32_t                     modeD[],
                                              hiptensorOperator_t               opReduce,
                                              hiptensorComputeDescriptor_t      typeCompute,
                                              hipStream_t                       stream)
{
    int  rankA        = descA->mLengths.size();
    int  numReduceDim = descA->mLengths.size() - descD->mLengths.size();
    auto ADataType    = descA->mType;
    auto DDataType    = descD->mType;

    auto internalTypeCompute = typeCompute;
    if(typeCompute == HIPTENSOR_COMPUTE_DESC_16F || typeCompute == HIPTENSOR_COMPUTE_DESC_16BF)
    {
        // CK does not support f16 or bf16 as compute type
        internalTypeCompute = HIPTENSOR_COMPUTE_DESC_32F;
    }

    if(descA->mLengths.size() == descD->mLengths.size())
    {
        // Composable Kernels (CK) does not handle reductions where the input and
        // output tensors maintain the same rank. For those scenarios, employ
        // elementwise binary operations.
        return hiptensorElementwiseBinaryOpReference(
            alpha,
            A,
            descA,
            modeA,
            opA,
            beta,
            C,
            descC,
            modeC,
            opC,
            D,
            descD,
            modeD,
            HIPTENSOR_OP_ADD,
            *hiptensor::convertToHipTensorDataType(typeCompute),
            stream);
    }

    auto& instances = hiptensor::ReductionCpuReferenceInstances::instance();
    auto  solutionQ = instances->querySolutions(ADataType,
                                               internalTypeCompute,
                                               DDataType,
                                               rankA,
                                               numReduceDim,
                                               opReduce,
                                               true, // propagateNan
                                               false); // outputIndex

    double alphaD;
    if(alpha != nullptr)
    {
        alphaD = hiptensor::readVal<double>(alpha, typeCompute);
    }
    double betaD;
    if(beta != nullptr)
    {
        betaD = hiptensor::readVal<double>(beta, typeCompute);
    }

    if(C && C != D)
    {
        // CK API can only process $D = alpha * reduce(A) + beta * D$
        // Need to copy C to D if C != D
        CHECK_HIP_ERROR(hipMemcpy(D,
                                  C,
                                  hiptensor::elementsFromLengths(descC->mLengths)
                                      * hiptensor::hiptensorDataTypeSize(descC->mType),
                                  hipMemcpyHostToHost));
    }

    // CK requires all the modes of the reduced tensors (C/D) to be sorted according to the
    // non-reduced tensor mode (modeA).
    //
    // By permuting the reduced tensor modes and their lengths, if the strides are permuted
    // following the same order, then it's possible to satisfy CK's requirement without
    // physically permuting the data.
    //
    // For example:
    //  - A lengths are [3, 5, 8], C/D modes are [2, 1].
    //  - CK requires sorted modes as [1, 2] for C/D, and the respective lengths [5, 8].
    //  - Strides of output are generated in this way:
    //    - output dims are [2, 1]
    //      => corresponding lengths are [8(2), 5(1)]
    //      => strides are [5(2), 1(1)]
    //      => permuted strides are [1(1), 5(2)]
    //
    // The code below implements this logic.
    hiptensorTensorDescriptor sortedDescC    = *descC;
    std::vector<int>          sortingIndices = {};

    CHECK_HIPTENSOR_ERROR(hiptensor::getSortingIndices(
        modeA, descA->mLengths.size(), modeC, sortedDescC.mLengths.size(), sortingIndices));
    CHECK_HIPTENSOR_ERROR(hiptensor::applySortingIndices(sortingIndices, sortedDescC.mLengths));
    CHECK_HIPTENSOR_ERROR(hiptensor::applySortingIndices(sortingIndices, sortedDescC.mStrides));

    for(auto [_, pSolution] : solutionQ.solutions())
    {
        // Perform reduction with timing if LOG_LEVEL_PERF_TRACE
        auto streamConfig        = StreamConfig{stream, false};
        auto [isSupported, time] = (*pSolution)(descA->mLengths,
                                                descA->mStrides,
                                                {modeA, modeA + descA->mLengths.size()},
                                                sortedDescC.mLengths,
                                                sortedDescC.mStrides,
                                                {modeC, modeC + sortedDescC.mLengths.size()},
                                                opA,
                                                opC,
                                                alphaD,
                                                betaD,
                                                A,
                                                D,
                                                opReduce,
                                                streamConfig);
        if(isSupported)
        {
            if(time < 0)
            {
                return HIPTENSOR_STATUS_CK_ERROR;
            }
            return HIPTENSOR_STATUS_SUCCESS;
        }
    }

    return HIPTENSOR_STATUS_INTERNAL_ERROR;
}
