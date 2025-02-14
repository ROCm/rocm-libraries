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

#include "../permutation_solution.hpp"
#include "../permutation_solution_instances.hpp"

// Ensure access to
#include "../device/hiptensor_permutation_scale_instances.hpp"
#include <hiptensor_unary_element_wise_operation.hpp>

namespace hiptensor
{
    void PermutationSolutionInstances::PermutationSolution6DHalfInstances()
    {
        // Register all the solutions exactly once
        // 6d Permutation
        registerSolutions(
            enumeratePermutationSolutions<ck::Tuple<ck::half_t>,
                                          ck::Tuple<ck::half_t>,
                                          ck::tensor_operation::element_wise::HiptensorUnaryOp,
                                          ck::tensor_operation::element_wise::HiptensorUnaryOp,
                                          ck::tensor_operation::element_wise::Scale,
                                          6>());
    }
} // namespace hiptensor
