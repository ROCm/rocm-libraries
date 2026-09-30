// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-component.hpp"
#include <memory>

// What heuristic queries use when HIPBLASLT_JIT is 1 or 2.
namespace hipblaslt_jit
{
    // The Jit for this process, built on first use: TensileLite with the tool
    // paths configured when hipBLASLt was built, each replaced by
    // HIPBLASLT_JIT_PYTHON, HIPBLASLT_JIT_TENSILE_SOURCE,
    // HIPBLASLT_JIT_PYTHONPATH or HIPBLASLT_JIT_CXX when set, publishing to
    // JitLibrary::process(). Null, with why set, when a tool is missing.
    std::shared_ptr<const Jit> processJit(Status& why);

    enum class Severity
    {
        Warning,
        Error,
    };

    // Writes "hipblaslt <warning|error>: JIT <message>" to stderr the first time
    // this process reports message, and to the hipBLASLt log every time.
    void report(Severity severity, const std::string& message);

    // "<stage> failed for <problem>: <status.message>", followed by the log path
    // when the message does not name it.
    std::string describe(const Status& status, const std::string& problem);
}
