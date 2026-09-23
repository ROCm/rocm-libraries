// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <hipblaslt/hipblaslt-jit.hpp>

namespace hipblaslt_ext::experimental::jit::tensilelite
{
    // Options belong to the TensileLite provider, not to the common JIT interface.
    struct Options
    {
        std::string pythonExecutable;
        std::string tensileSourceDirectory;
        std::string pythonPath; // Additional import paths; ';' on Windows, ':' on POSIX.
        std::string configPath; // Explicit YAML recipe.
        std::string outputPath; // Must not exist. Diagnostic .log/.cwd siblings are retained.
        std::string architecture; // Empty selects the current device.
#ifdef _WIN32
        std::string cxxCompiler    = "clang++.exe";
        std::string offloadBundler = "clang-offload-bundler.exe";
#else
        std::string cxxCompiler    = "amdclang++";
        std::string offloadBundler = "clang-offload-bundler";
#endif
    };

    // Creates a configured provider. Compilation happens when a request is submitted.
    HIPBLASLT_EXPORT hipblasStatus_t createBackend(const Options& options,
                                                   Backend&       backend,
                                                   Diagnostics&   diagnostics);
}
