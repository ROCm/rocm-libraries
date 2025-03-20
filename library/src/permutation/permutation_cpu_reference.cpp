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

#include "permutation_cpu_reference.hpp"
#include "permutation_cpu_reference_impl.hpp"
#include "permutation_cpu_reference_instances.hpp"

hiptensorStatus_t hiptensorPermutationReference(const hiptensorHandle_t*           handle,
                                                const void*                        alpha,
                                                const void*                        A,
                                                const hiptensorTensorDescriptor_t* descA,
                                                const int32_t                      modeA[],
                                                void*                              B,
                                                const hiptensorTensorDescriptor_t* descB,
                                                const int32_t                      modeB[],
                                                const hipDataType                  typeScalar,
                                                const hipStream_t                  stream)
{
    const int32_t dim       = descA->mLengths.size();
    auto&         instances = hiptensor::PermutationCpuReferenceInstances::instance();

    float alphaF;
    if(alpha != nullptr)
    {
        alphaF = hiptensor::readVal<float>(alpha, hiptensor::convertToComputeType(typeScalar));
    }

    auto refCandidates = instances->query({alphaF},
                                          descA->mLengths,
                                          {descA->mType},
                                          {descB->mType},
                                          {{modeA, modeA + descA->mLengths.size()}},
                                          {{modeB, modeB + descB->mLengths.size()}},
                                          {descA->mUnaryOp, descB->mUnaryOp},
                                          hiptensor::ElementwiseExecutionSpaceType_t::HOST);

    for(auto refCandidate : refCandidates)
    {
        if(refCandidate->initArgs({alphaF},
                                  {descA->mLengths},
                                  {descA->mStrides},
                                  {std::vector<int32_t>(modeA, modeA + descA->mLengths.size())},
                                  {descB->mLengths},
                                  {descB->mStrides},
                                  {std::vector<int32_t>(modeB, modeB + descB->mLengths.size())},
                                  {descA->mUnaryOp},
                                  {A},
                                  {B}))
        {
            (*refCandidate)();
            return HIPTENSOR_STATUS_SUCCESS;
        }
    }

    return HIPTENSOR_STATUS_INTERNAL_ERROR;
}

hiptensorStatus_t hiptensorElementwiseBinaryOpReference(const hiptensorHandle_t*           handle,
                                                        const void*                        alpha,
                                                        const void*                        A,
                                                        const hiptensorTensorDescriptor_t* descA,
                                                        const int32_t                      modeA[],
                                                        const void*                        gamma,
                                                        const void*                        C,
                                                        const hiptensorTensorDescriptor_t* descC,
                                                        const int32_t                      modeC[],
                                                        void*                              D,
                                                        const hiptensorTensorDescriptor_t* descD,
                                                        const int32_t                      modeD[],
                                                        hiptensorOperator_t                opAC,
                                                        hipDataType typeScalar,
                                                        hipStream_t stream)
{
    const int32_t dim       = descA->mLengths.size();
    auto&         instances = hiptensor::PermutationCpuReferenceInstances::instance();

    float alphaF;
    if(alpha != nullptr)
    {
        alphaF = hiptensor::readVal<float>(alpha, hiptensor::convertToComputeType(typeScalar));
    }
    float gammaF;
    if(gamma != nullptr)
    {
        gammaF = hiptensor::readVal<float>(gamma, hiptensor::convertToComputeType(typeScalar));
    }

    auto refCandidates = instances->query(
        {alphaF, gammaF},
        descA->mLengths,
        {descA->mType, descC->mType},
        {descD->mType},
        {{modeA, modeA + descA->mLengths.size()}, {modeC, modeC + descC->mLengths.size()}},
        {{modeD, modeD + descD->mLengths.size()}},
        {descA->mUnaryOp, descC->mUnaryOp},
        hiptensor::ElementwiseExecutionSpaceType_t::HOST);

    for(auto refCandidate : refCandidates)
    {
        if(refCandidate->initArgs({alphaF, gammaF},
                                  {descA->mLengths, descC->mLengths},
                                  {descA->mStrides, descC->mStrides},
                                  {std::vector<int32_t>(modeA, modeA + descA->mLengths.size()),
                                   std::vector<int32_t>(modeC, modeC + descC->mLengths.size())},
                                  {descD->mLengths},
                                  {descD->mStrides},
                                  {std::vector<int32_t>(modeD, modeD + descD->mLengths.size())},
                                  {opAC, descA->mUnaryOp, descC->mUnaryOp},
                                  {A, C},
                                  {D}))
        {
            (*refCandidate)();
            return HIPTENSOR_STATUS_SUCCESS;
        }
    }

    return HIPTENSOR_STATUS_INTERNAL_ERROR;
}

hiptensorStatus_t hiptensorElementwiseTrinaryOpReference(const hiptensorHandle_t*           handle,
                                                         const void*                        alpha,
                                                         const void*                        A,
                                                         const hiptensorTensorDescriptor_t* descA,
                                                         const int32_t                      modeA[],
                                                         const void*                        beta,
                                                         const void*                        B,
                                                         const hiptensorTensorDescriptor_t* descB,
                                                         const int32_t                      modeB[],
                                                         const void*                        gamma,
                                                         const void*                        C,
                                                         const hiptensorTensorDescriptor_t* descC,
                                                         const int32_t                      modeC[],
                                                         void*                              D,
                                                         const hiptensorTensorDescriptor_t* descD,
                                                         const int32_t                      modeD[],
                                                         hiptensorOperator_t                opAB,
                                                         hiptensorOperator_t                opABC,
                                                         hipDataType typeScalar,
                                                         hipStream_t stream)
{
    const int32_t dim       = descA->mLengths.size();
    auto&         instances = hiptensor::PermutationCpuReferenceInstances::instance();

    float alphaF;
    if(alpha != nullptr)
    {
        alphaF = hiptensor::readVal<float>(alpha, hiptensor::convertToComputeType(typeScalar));
    }
    float betaF;
    if(beta != nullptr)
    {
        betaF = hiptensor::readVal<float>(beta, hiptensor::convertToComputeType(typeScalar));
    }
    float gammaF;
    if(gamma != nullptr)
    {
        gammaF = hiptensor::readVal<float>(gamma, hiptensor::convertToComputeType(typeScalar));
    }

    auto refCandidates
        = instances->query({alphaF, betaF, gammaF},
                           descA->mLengths,
                           {descA->mType, descB->mType, descC->mType},
                           {descD->mType},
                           {{modeA, modeA + descA->mLengths.size()},
                            {modeB, modeB + descB->mLengths.size()},
                            {modeC, modeC + descC->mLengths.size()}},
                           {{modeD, modeD + descD->mLengths.size()}},
                           {opABC, opAB, descA->mUnaryOp, descB->mUnaryOp, descC->mUnaryOp},
                           hiptensor::ElementwiseExecutionSpaceType_t::HOST);

    for(auto refCandidate : refCandidates)
    {
        if(refCandidate->initArgs({alphaF, betaF, gammaF},
                                  {descA->mLengths, descB->mLengths, descC->mLengths},
                                  {descA->mStrides, descB->mStrides, descC->mStrides},
                                  {std::vector<int32_t>(modeA, modeA + descA->mLengths.size()),
                                   std::vector<int32_t>(modeB, modeB + descB->mLengths.size()),
                                   std::vector<int32_t>(modeC, modeC + descC->mLengths.size())},
                                  {descD->mLengths},
                                  {descD->mStrides},
                                  {std::vector<int32_t>(modeD, modeD + descD->mLengths.size())},
                                  {opABC, opAB, descA->mUnaryOp, descB->mUnaryOp, descC->mUnaryOp},
                                  {A, B, C},
                                  {D}))
        {
            (*refCandidate)();
            return HIPTENSOR_STATUS_SUCCESS;
        }
    }

    return HIPTENSOR_STATUS_INTERNAL_ERROR;
}
