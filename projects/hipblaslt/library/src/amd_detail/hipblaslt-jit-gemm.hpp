// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <hipblaslt/hipblaslt.h>
#include <memory>
#include <string>
#include <vector>

// Build-only experiment, enabled by HIPBLASLT_ENABLE_JIT_GEMM.
// This header is intentionally not installed and has no stable ABI commitment.
namespace hipblaslt_ext::experimental
{
    struct GenerateOptions
    {
        std::string pythonExecutable;
        std::string tensileSourceDirectory; // Directory containing the Tensile package.
        std::string pythonPath; // Additional absolute source/build import paths (colon separated).
        std::string configPath;
        std::string outputPath; // Must not exist; ".log" and ".cwd" siblings hold diagnostics.
        std::string architecture;
        std::string cxxCompiler    = "amdclang++";
        std::string offloadBundler = "clang-offload-bundler";
    };

    struct DispatchInfo
    {
        int                      configuredGlobalSplitU = 0;
        size_t                   globalSplitU           = 0;
        std::string              accumulation;
        std::vector<std::string> kernelNames;
    };

    struct JitGemmInfo
    {
        std::string configPath, manifestPath, kernelName, prediction, error;
    };

    // Select one generated algorithm for normal hipblasLtMatmul or Gemm execution.
    // An empty configPath requests prediction; otherwise use the supplied YAML.
    // Generation is synchronous and must precede stream capture. The algorithm
    // and its private modules are retained until process exit (no eviction).
    // Copies are valid on the same device in this process; persist the YAML and
    // manifest, not the opaque algorithm or a prebuilt solution index.
    // Normal handle, workspace, stream and synchronization contracts apply.
    HIPBLASLT_EXPORT hipblasStatus_t getJitGemmAlgo(hipblasLtHandle_t       handle,
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
                                                    const GenerateOptions&  options,
                                                    size_t                  maxWorkspaceBytes,
                                                    hipblasLtMatmulHeuristicResult_t& result,
                                                    JitGemmInfo&                      info);

    // Calls on one object must be serialized. The handle, matrix storage, and workspace
    // must outlive submitted work. Scalars are captured by setProblem. Destruction waits
    // for the last submitted solution sequence before unloading the object's private HIP modules.
    class HIPBLASLT_EXPORT JitGemm
    {
    public:
        explicit JitGemm(hipblasLtHandle_t handle);
        ~JitGemm();
        JitGemm(const JitGemm&)            = delete;
        JitGemm& operator=(const JitGemm&) = delete;

        hipblasStatus_t setProblem(hipblasLtMatmulDesc_t   desc,
                                   const void*             alpha,
                                   const void*             A,
                                   hipblasLtMatrixLayout_t layoutA,
                                   const void*             B,
                                   hipblasLtMatrixLayout_t layoutB,
                                   const void*             beta,
                                   const void*             C,
                                   hipblasLtMatrixLayout_t layoutC,
                                   void*                   D,
                                   hipblasLtMatrixLayout_t layoutD);
        // Synchronously invoke the configured Python generator, validate/load its bundle,
        // and check support for the bound problem. No generation occurs in initialize/run.
        hipblasStatus_t prepare(const GenerateOptions& options, size_t& workspaceBytes);
        // Reinitialization waits for earlier work; it may bind a different stream.
        hipblasStatus_t initialize(void* workspace, size_t workspaceBytes, hipStream_t stream);
        // Enqueue on the initialized stream. Stream capture and other streams are rejected.
        hipblasStatus_t    run(hipStream_t stream);
        const std::string& lastError() const;
        const std::string& manifestPath() const;
        const std::string& kernelName() const;
        size_t             generationCount() const;
        // Valid after successful initialize; reports normal Tensile dispatch decisions.
        const DispatchInfo& dispatchInfo() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_impl;
    };
}
