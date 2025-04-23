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

#ifndef HIPTENSOR_REDUCTION_SOLUTION_PARAMS_HPP
#define HIPTENSOR_REDUCTION_SOLUTION_PARAMS_HPP

#include <memory>
#include <unordered_map>
#include <vector>

#include "data_types.hpp"
#include "reduction_types.hpp"
#include "singleton.hpp"

namespace hiptensor
{
    struct ReductionSolutionParams
    {
        ReductionSolutionParams()                                          = default;
        virtual ~ReductionSolutionParams()                                 = default;
        ReductionSolutionParams(ReductionSolutionParams const&)            = default;
        ReductionSolutionParams(ReductionSolutionParams&&)                 = default;
        ReductionSolutionParams& operator=(ReductionSolutionParams const&) = default;
        ReductionSolutionParams& operator=(ReductionSolutionParams&&)      = default;

        virtual int32_t rankIn() const        = 0;
        virtual int32_t numReducedDim() const = 0;
        virtual bool    propagateNan() const  = 0;
        virtual bool    outputIndex() const   = 0;

        virtual hipDataType            typeIn() const  = 0;
        virtual hiptensorComputeType_t typeAcc() const = 0;
        virtual hipDataType            typeOut() const = 0;

        virtual hiptensorOperator_t opReduce() const = 0;
    };

} // namespace hiptensor

#include "reduction_solution_params_impl.hpp"

#endif // HIPTENSOR_REDUCTION_SOLUTION_PARAMS_HPP
