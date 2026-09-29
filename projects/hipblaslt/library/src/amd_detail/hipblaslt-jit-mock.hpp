// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit.hpp"

// Not installed. An in-process backend for JIT tests: it replays one source
// bundle that Tensile.SingleSolution or Tensile.JitGemm wrote, without running a
// generator; the solution is still built with comgr.
namespace hipblaslt_ext::experimental::jit::mock
{
    struct Options
    {
        std::string replay; // the bundle directory
        enum class Fault
        {
            None,
            Generate, // generation fails
            Build, // the main kernel's source does not assemble
            Trap, // any generation aborts the process
        } fault = Fault::None;
    };

    // Returns NOT_SUPPORTED from getJitAlgo for problems the replayed solution
    // does not solve.
    HIPBLASLT_EXPORT hipblasStatus_t createBackend(const Options& options,
                                                   Backend&       backend,
                                                   Diagnostics&   diagnostics);
}
