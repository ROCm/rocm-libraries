// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-gemm.hpp"
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/Tensile.hpp>

namespace hipblaslt_ext::experimental
{
    // Writes a fresh ranked parameter request. Tensile.JitGemm validates its candidates,
    // writes info.configPath, and builds only the first valid solution.
    std::string predictJitGemmConfig(const TensileLite::ContractionProblemGemm& problem,
                                     const TensileLite::Hardware&               hardware,
                                     const GenerateOptions&                     options,
                                     JitGemmInfo&                               info,
                                     const std::string&                         mxScaleFormat,
                                     const std::string&                         scaleModeA,
                                     const std::string&                         scaleModeB);
}
