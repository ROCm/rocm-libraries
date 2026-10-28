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
    class JitLibrary;

    // The process's JIT solution library, rooted at HIPBLASLT_JIT_LIBRARY_PATH or,
    // when that is unset or empty, /tmp/hipblaslt-jit-<uid> (%TEMP%\hipblaslt-jit-<user>
    // on Windows). Never destroyed, so views and adapters stay valid until exit.
    JitLibrary& processLibrary();

    // True when this process has a backend that publishes into the JIT solution
    // library. A HipKittens build uses that backend. Otherwise the backend
    // replays the committed kernels when they are compiled in. A build with
    // neither returns false.
    bool jitHeuristicLibrary();

    // One warning per process when a query would consult JIT and no library is
    // available. Mode 1 then leaves the query unchanged; mode 2 returns nothing.
    void warnJitHeuristicUnavailable();

    // Appends up to room JIT solution library indices for problem that need at
    // most workspaceLimit and whose kernels are not in excludeKernels. Looks the
    // library up first and publishes what this process's backend generates. A hit
    // does not generate. The indices are in [jitIndexBase, jitIndexEnd) and run
    // through the same path as any other solution index. Returns how many were
    // written. Zero when no backend is available, the problem is outside it, or
    // room is zero. A capturing stream may return a hit and does not start a
    // build.
    int appendJitHeuristic(rocblaslt_handle                          handle,
                           const RocblasltContractionProblem&        problem,
                           size_t                                    workspaceLimit,
                           const std::vector<std::string>&           excludeKernels,
                           rocblaslt_matmul_heuristic_result*        results,
                           int                                       room);
}
