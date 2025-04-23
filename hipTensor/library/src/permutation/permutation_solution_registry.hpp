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

#ifndef HIPTENSOR_PERMUTATION_SOLUTION_REGISTRY_HPP
#define HIPTENSOR_PERMUTATION_SOLUTION_REGISTRY_HPP

#include <memory>
#include <unordered_map>
#include <vector>

#include "data_types.hpp"
#include "permutation_types.hpp"
#include "singleton.hpp"

namespace hiptensor
{
    class PermutationSolution;

    class PermutationSolutionRegistry
    {
    protected:
        // Move only
        PermutationSolutionRegistry()                                              = default;
        PermutationSolutionRegistry(PermutationSolutionRegistry&&)                 = default;
        PermutationSolutionRegistry& operator=(PermutationSolutionRegistry&&)      = default;
        PermutationSolutionRegistry(PermutationSolutionRegistry const&)            = delete;
        PermutationSolutionRegistry& operator=(PermutationSolutionRegistry const&) = delete;

        // Import permutation solutions for the registry to manage
        void registerSolutions(
            std::unordered_map<Uid, std::unique_ptr<PermutationSolution>>&& solutions);

    public:
        virtual ~PermutationSolutionRegistry() = default;

        std::vector<PermutationSolution*>
            query(std::vector<float> const&                scalarValues,
                  std::vector<std::size_t> const&          lengths,
                  std::vector<hipDataType> const&          inDataTypes,
                  std::vector<hipDataType> const&          outDataTypes,
                  std::vector<std::vector<int32_t>> const& inModesArray,
                  std::vector<std::vector<int32_t>> const& outModesArray,
                  std::vector<hiptensorOperator_t> const&  operators,
                  ElementwiseExecutionSpaceType_t          instanceType) const;

    private:
        std::unordered_map<Uid, std::unique_ptr<PermutationSolution>> mAllSolutions;
    };

} // namespace hiptensor

#endif // HIPTENSOR_PERMUTATION_SOLUTION_REGISTRY_HPP
