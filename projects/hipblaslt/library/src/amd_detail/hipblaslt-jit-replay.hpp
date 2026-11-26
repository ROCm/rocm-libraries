// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit.hpp"
#include <memory>
#include <vector>

// Not installed. An in-process backend for JIT tests: it replays pre-generated
// TensileLite kernels without running a generator; the solutions are still
// built with comgr.
namespace hipblaslt_jit
{
    class Backend;

    struct ReplayOptions
    {
        // TensileLite source-bundle directories. Generation walks them in order
        // and returns solutions that target the device and solve the problem,
        // until the requested count is met. A bundle whose matching kernels are
        // all excluded is skipped.
        std::vector<std::string> replay;
    };

    // The replay backend; throws when a bundle cannot be read.
    std::shared_ptr<const Backend> makeReplayBackend(const ReplayOptions& options);

    // Returns NOT_SUPPORTED from getJitAlgo for problems no replayed solution
    // solves.
    HIPBLASLT_EXPORT hipblasStatus_t createReplayBackend(const ReplayOptions&        options,
                                                         std::shared_ptr<const Jit>& jit,
                                                         Diagnostics&                diagnostics);
}
