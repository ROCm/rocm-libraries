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

// Product-private TensileLite adapter.

// TensileLite reference API owned by the shared host-validation component.

#include <Tensile/ContractionProblem.hpp>

#include <Tensile/DataTypes.hpp>
#include <optional>
#include <roc/host_validation/gemm.hpp>

namespace TensileLite
{
    namespace Client
    {
        void SolveCPU(ContractionProblem const* contraction,
                      ProblemInputs const*      inputs,
                      size_t                    elementsToValidate);

        // Specialized solver for ungrouped GEMM problems. Requests prefer the
        // blocked backend and fall back to the pointwise implementation when
        // required numerical semantics are not yet supported.
        roc::host_validation::GemmRunInfo SolveGemmCPU(ContractionProblemGemm const& problem,
                                                       ContractionInputs const&      inputs,
                                                       size_t elementsToValidate);

        enum class ReferenceGemmExecution
        {
            Pointwise,
            BlockedPreferred,
            BlockedRequired,
        };

        // Translates and executes one ungrouped GEMM. No value means descriptor
        // translation or the required backend was unsupported; outputs remain
        // unchanged in that case.
        std::optional<roc::host_validation::GemmRunInfo>
            tryReferenceGemm(ContractionProblemGemm const& problem,
                             ContractionInputs const&      inputs,
                             size_t                        elementsToValidate,
                             ReferenceGemmExecution execution = ReferenceGemmExecution::Pointwise);

    } // namespace Client
} // namespace TensileLite
