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
        PermutationSolutionRegistry::query(std::vector<float> const&                scalarValues,
                                           std::vector<std::size_t> const&          lengths,
                                           std::vector<hipDataType> const&          inDataTypes,
                                           std::vector<hipDataType> const&          outDataTypes,
                                           std::vector<std::vector<int32_t>> const& inModesArray,
                                           std::vector<std::vector<int32_t>> const& outModesArray,
                                           std::vector<hiptensorOperator_t> const&  operators,
                                           ElementwiseExecutionSpaceType_t instanceType) const
    {
        int nDims = lengths.size();
        // TODO Only handle all input tensors have the same modes here. Need to handle cases when they are not.
        auto outputDims = hiptensor::findIndices(inModesArray[0], outModesArray[0]);

        // TODO Only handle A, B have the same types here. Need to handle A, B are different types
        auto instanceParams
            = instanceType == ElementwiseExecutionSpaceType_t::DEVICE
                  ? selectInstanceParams(lengths, outputDims, inDataTypes, outDataTypes, nDims)
                  : InstanceHyperParams{};

        /// When all operators are both pass_through and alpha is 1.0. Permutation only moves data around.
        /// Use PermutationOpId_t::PASS_THROUGH instead of PermutationOpId_t::SCALE in this case so that the performance is much better.
        /// Some special noop instances are created for this case.
        ///
        /// Do not use PermutationOpId_t::PASS_THROUGH when instanceType is Host since no such special
        /// instances have been created.
        bool usePassThroughIfAlphaIsOne
            = (std::all_of(
                   scalarValues.cbegin(), scalarValues.cend(), [](auto v) { return v == 1.0F; })
               && std::all_of(operators.cbegin(),
                              operators.cend(),
                              [](auto v) { return v == HIPTENSOR_OP_IDENTITY; })
               && instanceType == ElementwiseExecutionSpaceType_t::DEVICE);
        auto scale = usePassThroughIfAlphaIsOne ? hiptensor::PermutationOpId_t::PASS_THROUGH
                                                : hiptensor::PermutationOpId_t::SCALE;
        auto hashCodes = ck::tensor_operation::device::instance::getHashCodeOfBestPerfInstances(
            inDataTypes, outDataTypes, scale, nDims, instanceParams);

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

} // namespace hiptensor
