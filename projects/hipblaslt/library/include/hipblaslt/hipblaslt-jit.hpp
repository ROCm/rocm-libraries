// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <hipblaslt/hipblaslt.h>
#include <memory>
#include <string>

namespace hipblaslt_ext::experimental::jit
{
    namespace detail
    {
        struct BackendImplementation;
        struct BackendAccess;
        struct OperationRequest;
        struct RequestAccess;
        struct CompiledSolution;
        struct SolutionAccess;
    }

    // Provider factories configure an opaque, copyable kernel generator.
    class Backend
    {
    public:
        Backend() = default;

    private:
        std::shared_ptr<const detail::BackendImplementation> implementation;
        friend struct detail::BackendAccess;
    };

    // Operation factories own the description and host scalar values. Device
    // buffers retain the execution API's normal caller lifetime requirements.
    class Request
    {
    public:
        Request() = default;

    private:
        std::shared_ptr<const detail::OperationRequest> implementation;
        friend struct detail::RequestAccess;
    };

    // Owns an executable kernel bundle, including its provider and helper modules.
    // An operation adapter converts it to an execution API's algorithm type.
    class Solution
    {
    public:
        Solution() = default;

    private:
        std::shared_ptr<const detail::CompiledSolution> implementation;
        friend struct detail::SolutionAccess;
    };

    struct Diagnostics
    {
        std::string backend;
        std::string message;
    };

    // Compile synchronously on the current HIP device (which must equal device).
    // Call before stream capture. Unsupported operation/backend pairs return
    // NOT_SUPPORTED. There is no persistent cache or reload across invocations.
    // The disabled build returns NOT_SUPPORTED and clears solution.
    HIPBLASLT_EXPORT hipblasStatus_t getJitAlgo(int            device,
                                                const Request& request,
                                                const Backend& backend,
                                                size_t         maxWorkspaceBytes,
                                                Solution&      solution,
                                                Diagnostics&   diagnostics);

    // Build the implemented GEMM request from existing hipBLASLt descriptors.
    // Other operations can add factories without changing getJitAlgo or Backend.
    HIPBLASLT_EXPORT hipblasStatus_t makeGemmRequest(hipblasLtHandle_t       handle,
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
                                                     Request&                request,
                                                     Diagnostics&            diagnostics);

    // Adapt a GEMM solution to hipblasLtMatmul / hipblaslt_ext::Gemm. Other
    // operation kinds return NOT_SUPPORTED. The resulting algorithm retains its
    // modules until process exit; copies work only on their original device in
    // this process. Never persist algorithm bytes or use them as prebuilt indices.
    // Execution preserves existing handle, workspace and stream requirements;
    // sharing a backend or solution does not relax those concurrency requirements.
    HIPBLASLT_EXPORT hipblasStatus_t getGemmAlgo(const Solution&                   solution,
                                                 hipblasLtMatmulHeuristicResult_t& result,
                                                 Diagnostics&                      diagnostics);
}
