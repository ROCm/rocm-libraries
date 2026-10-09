// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/UserKernelLibrary.hpp>

namespace TensileLite
{
    template <>
    std::shared_ptr<UserKernelLibrary<ContractionProblemGemm, ContractionSolution>>
        globalUserKernelLibrary<ContractionProblemGemm, ContractionSolution>()
    {
        static auto instance
            = std::make_shared<UserKernelLibrary<ContractionProblemGemm, ContractionSolution>>();
        return instance;
    }
} // namespace TensileLite
