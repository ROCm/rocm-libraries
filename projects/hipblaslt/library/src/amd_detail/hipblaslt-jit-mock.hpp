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

namespace hipblaslt_ext::experimental::jit::mock
{
    struct Options
    {
        // Bundle directories. A generation returns, in this order, up to the
        // requested count of those whose solution targets the device and
        // solves the problem, skipping excluded kernels.
        std::vector<std::string> replay;
        enum class Fault
        {
            None,
            Generate, // generation fails and leaves mock.log in its scratch directory
            Build, // the main kernel's source does not assemble
            Record, // generation appends its request to record and fails
            Trap, // any generation aborts the process
        } fault = Fault::None;
        std::string record;
        // The modeled contract whose predictions the mock accepts, or empty for
        // none. createBackend pairs a contract with the Origami predictor and
        // the catalog knowledge.
        std::string contract;
    };

    // The mock as a Jit backend; throws when a bundle cannot be read.
    std::shared_ptr<const hipblaslt_jit::Backend> makeBackend(const Options& options);

    // Returns NOT_SUPPORTED from getJitAlgo for problems no replayed solution
    // solves.
    HIPBLASLT_EXPORT hipblasStatus_t createBackend(const Options& options,
                                                   Backend&       backend,
                                                   Diagnostics&   diagnostics);
}
