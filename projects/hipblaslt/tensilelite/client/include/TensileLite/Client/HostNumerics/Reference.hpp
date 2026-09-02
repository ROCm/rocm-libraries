/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

// TensileLite client-private adapter.

// TensileLite reference API owned by the shared host-numerics component.

#include <Tensile/ContractionProblem.hpp>

#include <Tensile/DataTypes.hpp>
#include <roc/host_numerics/gemm.hpp>
#include <span>

namespace TensileLite
{
    namespace Client
    {
        void SolveCPU(ContractionProblem const* contraction,
                      ProblemInputs const*      inputs,
                      std::span<const roc::host_numerics::OutputSelection> outputSelections);

        // Specialized solver for ungrouped GEMM problems. Automatic execution
        // selects among the component-owned implementations.
        roc::host_numerics::GemmBackend
            SolveGemmCPU(ContractionProblemGemm const&       problem,
                         ContractionInputs const&            inputs,
                         roc::host_numerics::OutputSelection outputSelection);

        roc::host_numerics::GemmBackend SolveGemmCPU(ContractionProblemGemm const& problem,
                                                     ContractionInputs const&      inputs,
                                                     size_t elementsToValidate);

        // Translates and executes one ungrouped GEMM. Unsupported descriptors or
        // backends throw std::invalid_argument before copying the current batch's
        // staged outputs to caller storage.
        roc::host_numerics::GemmBackend
            executeReferenceGemm(ContractionProblemGemm const&       problem,
                                 ContractionInputs const&            inputs,
                                 roc::host_numerics::OutputSelection outputSelection,
                                 roc::host_numerics::GemmBackend     backend
                                 = roc::host_numerics::GemmBackend::Automatic);

        roc::host_numerics::GemmBackend
            executeReferenceGemm(ContractionProblemGemm const&   problem,
                                 ContractionInputs const&        inputs,
                                 size_t                          elementsToValidate,
                                 roc::host_numerics::GemmBackend backend
                                 = roc::host_numerics::GemmBackend::Automatic);

    } // namespace Client
} // namespace TensileLite
