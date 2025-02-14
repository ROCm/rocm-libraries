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

#ifndef HIPTENSOR_PERMUTATION_SOLUTION_PARAMS_HPP
#define HIPTENSOR_PERMUTATION_SOLUTION_PARAMS_HPP

#include <memory>
#include <unordered_map>
#include <vector>

#include "data_types.hpp"
#include "permutation_types.hpp"
#include "singleton.hpp"

namespace hiptensor
{
    struct PermutationSolutionParams
    {
        PermutationSolutionParams()                                            = default;
        virtual ~PermutationSolutionParams()                                   = default;
        PermutationSolutionParams(PermutationSolutionParams const&)            = default;
        PermutationSolutionParams(PermutationSolutionParams&&)                 = default;
        PermutationSolutionParams& operator=(PermutationSolutionParams const&) = default;
        PermutationSolutionParams& operator=(PermutationSolutionParams&&)      = default;

        // Map tensor dimension
        virtual int32_t dim() const = 0;

        // Map to hipDataType
        virtual hipDataType typeIn() const  = 0;
        virtual hipDataType typeOut() const = 0;

        // Map to operators
        virtual PermutationOpId_t opScale() const = 0;
    };

} // namespace hiptensor

#include "permutation_solution_params_impl.hpp"

#endif // HIPTENSOR_PERMUTATION_SOLUTION_PARAMS_HPP
