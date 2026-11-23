// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/LocalKernelLibrary.hpp>

namespace TensileLite
{
    template <>
    std::shared_ptr<LocalKernelLibrary<ContractionProblemGemm, ContractionSolution>>
        globalLocalKernelLibrary<ContractionProblemGemm, ContractionSolution>()
    {
        static auto instance
            = std::make_shared<LocalKernelLibrary<ContractionProblemGemm, ContractionSolution>>();
        return instance;
    }
} // namespace TensileLite
