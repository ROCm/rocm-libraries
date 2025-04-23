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

#ifndef HIPTENSOR_CONTRACTION_CPU_REFERENCE_INSTANCES_HPP
#define HIPTENSOR_CONTRACTION_CPU_REFERENCE_INSTANCES_HPP

#include <memory>

#include "contraction_solution_registry.hpp"
#include "singleton.hpp"

namespace hiptensor
{
    class ContractionCpuReferenceInstances : public ContractionSolutionRegistry,
                                             public LazySingleton<ContractionCpuReferenceInstances>
    {
    public:
        // For static initialization
        friend std::unique_ptr<ContractionCpuReferenceInstances>
            std::make_unique<ContractionCpuReferenceInstances>();

        ~ContractionCpuReferenceInstances() = default;

    private:
        // Singleton: only one instance
        ContractionCpuReferenceInstances();
        ContractionCpuReferenceInstances(ContractionCpuReferenceInstances const&) = delete;
        ContractionCpuReferenceInstances(ContractionCpuReferenceInstances&&)      = delete;
        ContractionCpuReferenceInstances& operator=(ContractionCpuReferenceInstances const&)
            = delete;
        ContractionCpuReferenceInstances& operator=(ContractionCpuReferenceInstances&&) = delete;
    };

} // namespace hiptensor

#endif // HIPTENSOR_CONTRACTION_SOLUTION_INSTANCES_HPP
