// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <hipblaslt/hipblaslt-ext.hpp>
#include <memory>
#include <string>
#include <vector>

// Standalone owner used by the JIT sample. The selection API is declared in
// hipblaslt-ext.hpp; this sample-support header is not installed.
namespace hipblaslt_ext::experimental
{
    struct DispatchInfo
    {
        int                      configuredGlobalSplitU = 0;
        size_t                   globalSplitU           = 0;
        std::string              accumulation;
        std::vector<std::string> kernelNames;
    };

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
