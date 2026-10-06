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
    // True when this process has a JIT library to consult. That library is the
    // replay backend of the bundles HIPBLASLT_JIT_TEST_REPLAY lists, or, when
    // that variable is unset, the HipKittens backend of a HipKittens build.
    // A build with neither returns false.
    bool jitHeuristicLibrary();

    // One warning per process when a query would consult JIT and no library is
    // available. Mode 1 then leaves the query unchanged; mode 2 returns nothing.
    void warnJitHeuristicUnavailable();

    // Appends up to room JIT solutions for problem that need at most
    // workspaceLimit and whose kernels are not in excludeKernels. Results are
    // process-local algorithms. Returns how many were written. Zero when the
    // library is absent, the problem is outside it, or room is zero. Does not
    // compile while problem.stream is capturing; a null or legacy stream is not
    // capturing. Cached solutions are still returned.
    int appendJitHeuristic(rocblaslt_handle                          handle,
                           const RocblasltContractionProblem&        problem,
                           size_t                                    workspaceLimit,
                           const std::vector<std::string>&           excludeKernels,
                           rocblaslt_matmul_heuristic_result*        results,
                           int                                       room);
}
