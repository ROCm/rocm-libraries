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

#include "permutation_solution_registry.hpp"
#include "permutation_instance_selection.hpp"
#include "permutation_solution.hpp"

namespace hiptensor
{
    /////////////////////////////////////////
    /// Class PermutationSolutionRegistry ///
    /////////////////////////////////////////
    std::vector<PermutationSolution*>
        PermutationSolutionRegistry::query(const void*                        alpha,
                                           const hiptensorTensorDescriptor_t* descA,
                                           const int32_t                      modeA[],
                                           const hiptensorTensorDescriptor_t* descB,
                                           const int32_t                      modeB[],
                                           const hipDataType                  typeScalar,
                                           PermutationInstanceType_t          instanceType) const
    {
        int  nDims      = descA->mLengths.size();
        auto ADataType  = descA->mType;
        auto BDataType  = descB->mType;
        auto AOp        = descA->mUnaryOp;
        auto BOp        = descB->mUnaryOp;
        auto outputDims = hiptensor::findIndices({modeA, modeA + descA->mLengths.size()},
                                                 {modeB, modeB + descB->mLengths.size()});
        auto instanceParams
            = instanceType == PermutationInstanceType_t::Device
                  ? selectInstanceParams(descA->mLengths, outputDims, ADataType, BDataType, nDims)
                  : InstanceHyperParams{0, 0, 0, 0, 0, {0, 0}, 0, 0};

        float alphaValue = 1.0F;
        if(alpha != nullptr)
        {
            alphaValue
                = hiptensor::readVal<float>(alpha, hiptensor::convertToComputeType(typeScalar));
        }

        /// When AOp and BOp are both pass_through and alpha is 1.0. Permutation only moves data around.
        /// Use PermutationOpId_t::PASS_THROUGH instead of PermutationOpId_t::SCALE in this case so that the performance is much better.
        /// Some special noop instances are created for this case.
        ///
        /// Do not use PermutationOpId_t::PASS_THROUGH when instanceType is Host since no such special
        /// instances have been created.
        bool usePassThroughIfAlphaIsOne
            = (alphaValue == 1.0F && AOp == HIPTENSOR_OP_IDENTITY && BOp == HIPTENSOR_OP_IDENTITY
               && instanceType == PermutationInstanceType_t::Device);
        auto scale     = usePassThroughIfAlphaIsOne ? hiptensor::PermutationOpId_t::PASS_THROUGH
                                                    : hiptensor::PermutationOpId_t::SCALE;
        auto hashCodes = ck::tensor_operation::device::instance::getHashCodeOfBestPerfInstances(
            ADataType, BDataType, scale, nDims, instanceParams);
        std::vector<PermutationSolution*> solutions;
        for(auto hashCode : hashCodes)
        {
            if(auto solution = mAllSolutions.find(hashCode); solution != mAllSolutions.end())
            {
                solutions.push_back(solution->second.get());
            }
        }

        return solutions;
    }

    void PermutationSolutionRegistry::registerSolutions(
        std::unordered_map<Uid, std::unique_ptr<PermutationSolution>>&& solutions)
    {
        for(auto&& solution : solutions)
        {
            // Register with the query then take ownership
            mAllSolutions.insert(std::move(solution));
        }
    }

    uint32_t PermutationSolutionRegistry::solutionCount() const
    {
        return mAllSolutions.size();
    }

} // namespace hiptensor
