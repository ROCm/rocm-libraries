// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <hipblaslt/hipblaslt.h>
#include <string>

namespace hipblaslt_ext::experimental::jit
{
    class Backend;
    struct Diagnostics;
}

namespace hipblaslt_ext::experimental::jit::tensilelite
{
    // Explicit single-solution generation options.
    struct Options
    {
        std::string pythonExecutable;
        std::string tensileSourceDirectory;
        std::string pythonPath; // Additional import paths; ';' on Windows, ':' on POSIX.
        std::string configPath; // Required explicit YAML recipe; no prediction or fallback.
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

    struct Diagnostics
    {
        std::string message;
    };

    // Generate one explicit recipe synchronously, validate it for the supplied
    // GEMM and return an algorithm for hipblasLtMatmul / hipblaslt_ext::Gemm.
    // Call on the handle's current device before stream capture. The result and
    // its helper modules remain valid until process exit; copies are local to
    // this process/device. Never persist algorithm bytes or treat them as a
    // prebuilt solution index. Workspace and stream rules of those APIs apply.
    // Empty output returns NOT_SUPPORTED; K=0 still generates beta*C. Failure
    // clears result and sets result.state. A disabled build returns NOT_SUPPORTED.
    HIPBLASLT_EXPORT hipblasStatus_t getGemmAlgo(hipblasLtHandle_t       handle,
                                                 hipblasLtMatmulDesc_t   desc,
                                                 const void*             alpha,
                                                 const void*             A,
                                                 hipblasLtMatrixLayout_t layoutA,
                                                 const void*             B,
                                                 hipblasLtMatrixLayout_t layoutB,
                                                 const void*             beta,
                                                 const void*             C,
                                                 hipblasLtMatrixLayout_t layoutC,
                                                 void*                   D,
                                                 hipblasLtMatrixLayout_t layoutD,
                                                 const Options&          options,
                                                 size_t                  maxWorkspaceBytes,
                                                 hipblasLtMatmulHeuristicResult_t& result,
                                                 Diagnostics&                      diagnostics);

    // Optional generic interface: compilation begins when the backend is submitted.
    HIPBLASLT_EXPORT hipblasStatus_t createBackend(const Options&    options,
                                                   jit::Backend&     backend,
                                                   jit::Diagnostics& diagnostics);
}
