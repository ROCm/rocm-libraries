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

#ifndef HIPTENSOR_CONTRACTION_SOLUTION_PARAMS_HPP
#define HIPTENSOR_CONTRACTION_SOLUTION_PARAMS_HPP

#include <hiptensor/hiptensor_types.hpp>

#include "contraction_types.hpp"
#include "data_types.hpp"

namespace hiptensor
{
    struct ContractionSolutionParams
    {
        ContractionSolutionParams()                                            = default;
        virtual ~ContractionSolutionParams()                                   = default;
        ContractionSolutionParams(ContractionSolutionParams const&)            = default;
        ContractionSolutionParams(ContractionSolutionParams&&)                 = default;
        ContractionSolutionParams& operator=(ContractionSolutionParams const&) = default;
        ContractionSolutionParams& operator=(ContractionSolutionParams&&)      = default;

        // Map tensor dimensions
        virtual int32_t dimsM() const = 0;
        virtual int32_t dimsN() const = 0;
        virtual int32_t dimsK() const = 0;

        // Map to hipDataType
        virtual hipDataType            typeA() const       = 0;
        virtual hipDataType            typeB() const       = 0;
        virtual hipDataType            typeC() const       = 0;
        virtual hipDataType            typeD() const       = 0;
        virtual hiptensorComputeType_t typeCompute() const = 0;

        // Map to operators
        virtual hiptensorOperator_t opA() const   = 0;
        virtual hiptensorOperator_t opB() const   = 0;
        virtual ContractionOpId_t   opCDE() const = 0;
    };

} // namespace hiptensor

#include "contraction_solution_params_impl.hpp"

#endif // HIPTENSOR_CONTRACTION_SOLUTION_PARAMS_HPP
