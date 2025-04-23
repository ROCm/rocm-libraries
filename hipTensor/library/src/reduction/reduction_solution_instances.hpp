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

#ifndef HIPTENSOR_REDUCTION_SOLUTION_INSTANCES_HPP
#define HIPTENSOR_REDUCTION_SOLUTION_INSTANCES_HPP

#include <memory>

#include "reduction_solution_registry.hpp"
#include "singleton.hpp"

#define REG_REDUCTION_SOLUTION(dim_count, reduced_dim_count, type, computeType) \
    registerSolutions(enumerateReductionSolutions<type,                         \
                                                  computeType,                  \
                                                  type,                         \
                                                  dim_count,                    \
                                                  reduced_dim_count,            \
                                                  HIPTENSOR_OP_ADD,             \
                                                  true,                         \
                                                  false>());                    \
    registerSolutions(enumerateReductionSolutions<type,                         \
                                                  computeType,                  \
                                                  type,                         \
                                                  dim_count,                    \
                                                  reduced_dim_count,            \
                                                  HIPTENSOR_OP_MUL,             \
                                                  true,                         \
                                                  false>());                    \
    registerSolutions(enumerateReductionSolutions<type,                         \
                                                  computeType,                  \
                                                  type,                         \
                                                  dim_count,                    \
                                                  reduced_dim_count,            \
                                                  HIPTENSOR_OP_MIN,             \
                                                  true,                         \
                                                  false>());                    \
    registerSolutions(enumerateReductionSolutions<type,                         \
                                                  computeType,                  \
                                                  type,                         \
                                                  dim_count,                    \
                                                  reduced_dim_count,            \
                                                  HIPTENSOR_OP_MAX,             \
                                                  true,                         \
                                                  false>());

namespace hiptensor
{
    class ReductionSolutionInstances : public ReductionSolutionRegistry,
                                       public LazySingleton<ReductionSolutionInstances>
    {
    public:
        // For static initialization
        friend std::unique_ptr<ReductionSolutionInstances>
            std::make_unique<ReductionSolutionInstances>();

        ~ReductionSolutionInstances() = default;

    private:
        // Singleton: only one instance
        ReductionSolutionInstances();
        ReductionSolutionInstances(ReductionSolutionInstances const&)            = delete;
        ReductionSolutionInstances(ReductionSolutionInstances&&)                 = delete;
        ReductionSolutionInstances& operator=(ReductionSolutionInstances const&) = delete;
        ReductionSolutionInstances& operator=(ReductionSolutionInstances&&)      = delete;

        void genReductionSolution1x1BF16F32Instances();
        void genReductionSolution2x1BF16F32Instances();
        void genReductionSolution2x2BF16F32Instances();
        void genReductionSolution3x1BF16F32Instances();
        void genReductionSolution3x2BF16F32Instances();
        void genReductionSolution3x3BF16F32Instances();
        void genReductionSolution4x1BF16F32Instances();
        void genReductionSolution4x2BF16F32Instances();
        void genReductionSolution4x3BF16F32Instances();
        void genReductionSolution4x4BF16F32Instances();
        void genReductionSolution5x1BF16F32Instances();
        void genReductionSolution5x2BF16F32Instances();
        void genReductionSolution5x3BF16F32Instances();
        void genReductionSolution5x4BF16F32Instances();
        void genReductionSolution5x5BF16F32Instances();
        void genReductionSolution6x1BF16F32Instances();
        void genReductionSolution6x2BF16F32Instances();
        void genReductionSolution6x3BF16F32Instances();
        void genReductionSolution6x4BF16F32Instances();
        void genReductionSolution6x5BF16F32Instances();
        void genReductionSolution6x6BF16F32Instances();

        void genReductionSolution1x1F16F32Instances();
        void genReductionSolution2x1F16F32Instances();
        void genReductionSolution2x2F16F32Instances();
        void genReductionSolution3x1F16F32Instances();
        void genReductionSolution3x2F16F32Instances();
        void genReductionSolution3x3F16F32Instances();
        void genReductionSolution4x1F16F32Instances();
        void genReductionSolution4x2F16F32Instances();
        void genReductionSolution4x3F16F32Instances();
        void genReductionSolution4x4F16F32Instances();
        void genReductionSolution5x1F16F32Instances();
        void genReductionSolution5x2F16F32Instances();
        void genReductionSolution5x3F16F32Instances();
        void genReductionSolution5x4F16F32Instances();
        void genReductionSolution5x5F16F32Instances();
        void genReductionSolution6x1F16F32Instances();
        void genReductionSolution6x2F16F32Instances();
        void genReductionSolution6x3F16F32Instances();
        void genReductionSolution6x4F16F32Instances();
        void genReductionSolution6x5F16F32Instances();
        void genReductionSolution6x6F16F32Instances();

        void genReductionSolution1x1F32F32Instances();
        void genReductionSolution2x1F32F32Instances();
        void genReductionSolution2x2F32F32Instances();
        void genReductionSolution3x1F32F32Instances();
        void genReductionSolution3x2F32F32Instances();
        void genReductionSolution3x3F32F32Instances();
        void genReductionSolution4x1F32F32Instances();
        void genReductionSolution4x2F32F32Instances();
        void genReductionSolution4x3F32F32Instances();
        void genReductionSolution4x4F32F32Instances();
        void genReductionSolution5x1F32F32Instances();
        void genReductionSolution5x2F32F32Instances();
        void genReductionSolution5x3F32F32Instances();
        void genReductionSolution5x4F32F32Instances();
        void genReductionSolution5x5F32F32Instances();
        void genReductionSolution6x1F32F32Instances();
        void genReductionSolution6x2F32F32Instances();
        void genReductionSolution6x3F32F32Instances();
        void genReductionSolution6x4F32F32Instances();
        void genReductionSolution6x5F32F32Instances();
        void genReductionSolution6x6F32F32Instances();

        void genReductionSolution1x1F64F64Instances();
        void genReductionSolution2x1F64F64Instances();
        void genReductionSolution2x2F64F64Instances();
        void genReductionSolution3x1F64F64Instances();
        void genReductionSolution3x2F64F64Instances();
        void genReductionSolution3x3F64F64Instances();
        void genReductionSolution4x1F64F64Instances();
        void genReductionSolution4x2F64F64Instances();
        void genReductionSolution4x3F64F64Instances();
        void genReductionSolution4x4F64F64Instances();
        void genReductionSolution5x1F64F64Instances();
        void genReductionSolution5x2F64F64Instances();
        void genReductionSolution5x3F64F64Instances();
        void genReductionSolution5x4F64F64Instances();
        void genReductionSolution5x5F64F64Instances();
        void genReductionSolution6x1F64F64Instances();
        void genReductionSolution6x2F64F64Instances();
        void genReductionSolution6x3F64F64Instances();
        void genReductionSolution6x4F64F64Instances();
        void genReductionSolution6x5F64F64Instances();
        void genReductionSolution6x6F64F64Instances();
    };

} // namespace hiptensor

#endif // HIPTENSOR_REDUCTION_SOLUTION_INSTANCES_HPP
