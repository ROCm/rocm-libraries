// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocblaslt.h"

#include <string>
#include <vector>

// Heuristic queries use these when HIPBLASLT_JIT is 1 or 2. Compiled only into
// a JIT build.
namespace hipblaslt_jit
{
    // True when this process has a backend that publishes into the JIT solution
    // library. That backend replays the bundles HIPBLASLT_JIT_TEST_REPLAY lists,
    // or, when that variable is unset, the HipKittens backend of a HipKittens
    // build. A build with neither returns false.
    bool jitHeuristicLibrary();

    // One warning per process when a query would consult JIT and no library is
    // available. Mode 1 then leaves the query unchanged; mode 2 returns nothing.
    void warnJitHeuristicUnavailable();

    // Appends up to room JIT solution library indices for problem that need at
    // most workspaceLimit and whose kernels are not in excludeKernels. Looks the
    // library up first and publishes what this process's backend generates. A hit
    // does not generate. The indices start at 2^30 and run through the same path
    // as any other solution index. Returns how many were written. Zero when no
    // backend is available, the problem is outside it, or room is zero. A
    // capturing stream may return a hit and does not start a build.
    int appendJitHeuristic(rocblaslt_handle                          handle,
                           const RocblasltContractionProblem&        problem,
                           size_t                                    workspaceLimit,
                           const std::vector<std::string>&           excludeKernels,
                           rocblaslt_matmul_heuristic_result*        results,
                           int                                       room);
}
