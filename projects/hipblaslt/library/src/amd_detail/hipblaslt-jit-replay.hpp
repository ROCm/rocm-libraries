// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit.hpp"

// Not installed. An in-process backend for JIT tests: it replays pre-generated
// source bundles without running a generator; the solutions are still built
// with comgr.
namespace hipblaslt_jit
{
    class Backend;
}

namespace hipblaslt_ext::experimental::jit::replay
{
    struct Options
    {
        // Bundle directories. A generation returns, in this order, those with
        // a solution that targets the device and solves the problem until they
        // hold the requested count of such solutions. It skips a bundle when
        // every such solution's kernel is excluded.
        std::vector<std::string> replay;
    };

    // The replay backend as a Jit backend; throws when a bundle cannot be read.
    std::shared_ptr<const hipblaslt_jit::Backend> makeBackend(const Options& options);

    // Returns NOT_SUPPORTED from getJitAlgo for problems no replayed solution
    // solves.
    HIPBLASLT_EXPORT hipblasStatus_t createBackend(const Options& options,
                                                   Backend&       backend,
                                                   Diagnostics&   diagnostics);
}
