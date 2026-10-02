// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <hipblaslt/hipblaslt.h>
#include <string>

// Not installed; exported for the same consumers as hipblaslt-jit.hpp.
namespace hipblaslt_ext::experimental::jit
{
    class Backend;
    struct Diagnostics;
}

namespace hipblaslt_jit
{
    struct ProcessBackend;
    struct Status;
}

namespace hipblaslt_ext::experimental::jit::tensilelite
{
    // Explicit single-solution generation options.
    struct Options
    {
        std::string pythonExecutable;
        std::string tensileSourceDirectory;
        std::string pythonPath; // Additional import paths; ';' on Windows, ':' on POSIX.
        // Required explicit YAML for direct getGemmAlgo. Empty requests private
        // Origami ranking only through the generic createBackend interface.
        std::string configPath;
        std::string outputPath; // Must not exist. Diagnostic .log/.cwd siblings are retained.
        std::string architecture; // Empty selects the current device.
#ifdef _WIN32
        std::string cxxCompiler = "clang++.exe";
#else
        std::string cxxCompiler = "amdclang++";
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
    // clears result and sets result.state.
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

namespace hipblaslt_ext::experimental::jit::tensilelite::detail
{
    // The process backend: TensileLite with the tool paths configured when
    // hipBLASLt was built, each replaced by HIPBLASLT_JIT_PYTHON,
    // HIPBLASLT_JIT_TENSILE_SOURCE, HIPBLASLT_JIT_PYTHONPATH or
    // HIPBLASLT_JIT_CXX when set. A Configure failure names a missing tool.
    hipblaslt_jit::Status makeProcessBackend(hipblaslt_jit::ProcessBackend& made);
}
