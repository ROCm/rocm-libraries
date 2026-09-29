// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace TensileLite
{
    class ContractionProblemGemm;
}

namespace hipblaslt_jit
{
    // A single strided batched GEMM in Tensile's canonical index form.
    struct CanonicalGemm
    {
        bool   transA = false, transB = false;
        bool   conjugateA = false, conjugateB = false;
        size_t m = 0, n = 0, k = 0, batch = 0;
    };

    // Throws std::runtime_error when a Tensile ProblemType cannot describe the problem.
    CanonicalGemm canonicalGemm(const TensileLite::ContractionProblemGemm& problem);

    // Ordered name and JSON-literal pairs of the problem's Tensile ProblemType, as a
    // Tensile.JitGemm request spells it. Throws like canonicalGemm.
    std::vector<std::pair<std::string, std::string>>
        problemTypeFields(const TensileLite::ContractionProblemGemm& problem);
}
