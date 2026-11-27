// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "../../include/rocblaslt-types.h"

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

    // A rocRoller Stream-K data-parallel refusal is a bad problem for that
    // kernel. The public API reports it as invalid_value. Other runtime_errors
    // stay internal_error.
    inline rocblaslt_status statusForRocRollerStreamKDataParallel(
        TensileLite::RocRollerStreamKDataParallelError const&)
    {
        return rocblaslt_status_invalid_value;
    }

    inline rocblaslt_status statusForRocRollerStreamKBatch(
        TensileLite::RocRollerStreamKBatchError const&)
    {
        return rocblaslt_status_invalid_value;
    }
}
