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

#ifndef HIPTENSOR_PERMUTATION_CPU_REFERENCE_INSTANCES_HPP
#define HIPTENSOR_PERMUTATION_CPU_REFERENCE_INSTANCES_HPP

#include <memory>

#include "permutation_solution_registry.hpp"
#include "singleton.hpp"

namespace hiptensor
{
    class PermutationCpuReferenceInstances : public PermutationSolutionRegistry,
                                             public LazySingleton<PermutationCpuReferenceInstances>
    {
    public:
        // For static initialization
        friend std::unique_ptr<PermutationCpuReferenceInstances>
            std::make_unique<PermutationCpuReferenceInstances>();

        ~PermutationCpuReferenceInstances() = default;

    private:
        void PermutationCpuReference2DInstances();
        void PermutationCpuReference3DInstances();
        void PermutationCpuReference4DInstances();
        void PermutationCpuReference5DInstances();
        void PermutationCpuReference6DInstances();
        // Singleton: only one instance
        PermutationCpuReferenceInstances();
        PermutationCpuReferenceInstances(PermutationCpuReferenceInstances const&) = delete;
        PermutationCpuReferenceInstances(PermutationCpuReferenceInstances&&)      = delete;
        PermutationCpuReferenceInstances& operator=(PermutationCpuReferenceInstances const&)
            = delete;
        PermutationCpuReferenceInstances& operator=(PermutationCpuReferenceInstances&&) = delete;
    };

} // namespace hiptensor

#endif // HIPTENSOR_PERMUTATION_SOLUTION_INSTANCES_HPP
