// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// The eligibility checks the rocRoller-off path applies to a checked-in kernel
// once it implements a ranked tile, in a header so they can be unit-tested
// GPU-free. Neither depends on how the kernel was matched to the tile.

#include <Tensile/ContractionSolution.hpp>

#include <cstddef>

namespace rocblaslt
{
    /// Do the kernel's own hardware and problem predicates accept `problem` on
    /// `hardware`? A missing predicate accepts.
    inline bool rocRollerKernelAccepts(const TensileLite::ContractionSolution&    solution,
                                       const TensileLite::ContractionProblemGemm& problem,
                                       const TensileLite::Hardware&               hardware)
    {
        if(solution.hardwarePredicate && !(*solution.hardwarePredicate)(hardware))
            return false;
        return !solution.problemPredicate || (*solution.problemPredicate)(problem);
    }

    /// Does the kernel need no more than `maxWorkspaceBytes` of workspace for
    /// `problem`?
    inline bool rocRollerKernelFitsWorkspace(const TensileLite::ContractionSolution&    solution,
                                             const TensileLite::ContractionProblemGemm& problem,
                                             const TensileLite::Hardware&               hardware,
                                             size_t maxWorkspaceBytes)
    {
        return solution.requiredWorkspaceSize(problem, hardware) <= maxWorkspaceBytes;
    }
}
