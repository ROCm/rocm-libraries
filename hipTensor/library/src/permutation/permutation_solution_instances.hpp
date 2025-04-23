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

#ifndef HIPTENSOR_PERMUTATION_SOLUTION_INSTANCES_HPP
#define HIPTENSOR_PERMUTATION_SOLUTION_INSTANCES_HPP

#include <memory>

#include "permutation_solution_registry.hpp"
#include "singleton.hpp"

namespace hiptensor
{
    class PermutationSolutionInstances : public PermutationSolutionRegistry,
                                         public LazySingleton<PermutationSolutionInstances>
    {
    public:
        // For static initialization
        friend std::unique_ptr<PermutationSolutionInstances>
            std::make_unique<PermutationSolutionInstances>();

        ~PermutationSolutionInstances() = default;

    private:
        void ElementwiseBinarySolution2DDoubleInstances();
        void ElementwiseBinarySolution2DFloatInstances();
        void ElementwiseBinarySolution2DHalfInstances();
        void ElementwiseBinarySolution3DDoubleInstances();
        void ElementwiseBinarySolution3DFloatInstances();
        void ElementwiseBinarySolution3DHalfInstances();
        void ElementwiseBinarySolution4DDoubleInstances();
        void ElementwiseBinarySolution4DFloatInstances();
        void ElementwiseBinarySolution4DHalfInstances();
        void ElementwiseBinarySolution5DDoubleInstances();
        void ElementwiseBinarySolution5DFloatInstances();
        void ElementwiseBinarySolution5DHalfInstances();
        void ElementwiseBinarySolution6DDoubleInstances();
        void ElementwiseBinarySolution6DFloatInstances();
        void ElementwiseBinarySolution6DHalfInstances();
        void ElementwiseTrinarySolution2DDoubleInstances();
        void ElementwiseTrinarySolution2DFloatInstances();
        void ElementwiseTrinarySolution2DHalfInstances();
        void ElementwiseTrinarySolution3DDoubleInstances();
        void ElementwiseTrinarySolution3DFloatInstances();
        void ElementwiseTrinarySolution3DHalfInstances();
        void ElementwiseTrinarySolution4DDoubleInstances();
        void ElementwiseTrinarySolution4DFloatInstances();
        void ElementwiseTrinarySolution4DHalfInstances();
        void ElementwiseTrinarySolution5DDoubleInstances();
        void ElementwiseTrinarySolution5DFloatInstances();
        void ElementwiseTrinarySolution5DHalfInstances();
        void ElementwiseTrinarySolution6DDoubleInstances();
        void ElementwiseTrinarySolution6DFloatInstances();
        void ElementwiseTrinarySolution6DHalfInstances();
        void PermutationSolution2DFloatInstances();
        void PermutationSolution2DFloatNoopInstances();
        void PermutationSolution2DHalfInstances();
        void PermutationSolution2DHalfNoopInstances();
        void PermutationSolution3DFloatInstances();
        void PermutationSolution3DFloatNoopInstances();
        void PermutationSolution3DHalfInstances();
        void PermutationSolution3DHalfNoopInstances();
        void PermutationSolution4DFloatInstances();
        void PermutationSolution4DFloatNoopInstances();
        void PermutationSolution4DHalfInstances();
        void PermutationSolution4DHalfNoopInstances();
        void PermutationSolution5DFloatInstances();
        void PermutationSolution5DFloatNoopInstances();
        void PermutationSolution5DHalfInstances();
        void PermutationSolution5DHalfNoopInstances();
        void PermutationSolution6DFloatInstances();
        void PermutationSolution6DFloatNoopInstances();
        void PermutationSolution6DHalfInstances();
        void PermutationSolution6DHalfNoopInstances();
        // Singleton: only one instance
        PermutationSolutionInstances();
        PermutationSolutionInstances(PermutationSolutionInstances const&)            = delete;
        PermutationSolutionInstances(PermutationSolutionInstances&&)                 = delete;
        PermutationSolutionInstances& operator=(PermutationSolutionInstances const&) = delete;
        PermutationSolutionInstances& operator=(PermutationSolutionInstances&&)      = delete;
    };

} // namespace hiptensor

#endif // HIPTENSOR_PERMUTATION_SOLUTION_INSTANCES_HPP
