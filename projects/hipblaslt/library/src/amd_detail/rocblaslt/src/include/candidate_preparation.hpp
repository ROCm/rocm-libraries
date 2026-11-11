// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>

namespace rocblaslt
{
    inline bool prepareCandidateXcc(TensileLite::ContractionProblemGemm&    problem,
                                    const TensileLite::ContractionSolution& solution,
                                    const TensileLite::Hardware&            hardware)
    {
        const bool fallback = solution.isFallbackForHW(hardware);
        // Zero restores this candidate's automatic mapping after a fallback.
        problem.setParams().setWGMXCC(fallback ? 1 : 0);
        return fallback;
    }
}
