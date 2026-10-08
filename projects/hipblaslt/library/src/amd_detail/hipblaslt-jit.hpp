// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <hipblaslt/hipblaslt.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace hipblaslt_jit
{
    class Jit;
}

// Not installed. The entry points stay exported because the JIT test binaries
// link against the shared library.
namespace hipblaslt_ext::experimental::jit
{
    namespace detail
    {
        struct OperationRequest;
        struct CompiledSolution;
    }

    // Backend factories configure an opaque, copyable solution generator.
    class Backend
    {
    public:
        Backend() = default;
        static Backend create(std::shared_ptr<const hipblaslt_jit::Jit> jit)
        {
            Backend backend;
            backend.m_jit = std::move(jit);
            return backend;
        }
        const std::shared_ptr<const hipblaslt_jit::Jit>& get() const
        {
            return m_jit;
        }

    private:
        std::shared_ptr<const hipblaslt_jit::Jit> m_jit;
    };

    // Operation factories own the description and host scalar values. Device
    // buffers retain the execution API's normal caller lifetime requirements.
    class Request
    {
    public:
        Request() = default;
        static Request create(std::shared_ptr<const detail::OperationRequest> operation)
        {
            Request request;
            request.m_operation = std::move(operation);
            return request;
        }
        const std::shared_ptr<const detail::OperationRequest>& get() const
        {
            return m_operation;
        }

    private:
        std::shared_ptr<const detail::OperationRequest> m_operation;
    };

    // Names one published JIT solution library index. An operation adapter
    // converts it to an execution API's algorithm type.
    class Solution
    {
    public:
        Solution() = default;
        static Solution create(std::shared_ptr<const detail::CompiledSolution> compiled)
        {
            Solution solution;
            solution.m_compiled = std::move(compiled);
            return solution;
        }
        const std::shared_ptr<const detail::CompiledSolution>& get() const
        {
            return m_compiled;
        }

    private:
        std::shared_ptr<const detail::CompiledSolution> m_compiled;
    };

    struct Diagnostics
    {
        std::string backend;
        std::string message;
    };

    // Look up the JIT solution library on the current HIP device (which must
    // equal device) and publish with backend when the request is not already
    // there. A hit does not generate. Returns an owned Solution for the first
    // published index. A capturing stream may return a hit and does not start a
    // build. Unsupported operation/backend pairs return NOT_SUPPORTED.
    // getGemmAlgo turns the solution into a library index.
    HIPBLASLT_EXPORT hipblasStatus_t getJitAlgo(int            device,
                                                const Request& request,
                                                const Backend& backend,
                                                size_t         maxWorkspaceBytes,
                                                Solution&      solution,
                                                Diagnostics&   diagnostics);

    // Up to count solution indices for exactly this request: those already in the
    // JIT solution library in the order they were published, then solutions
    // generated with backend and published now, best first, under the same device
    // rules as getJitAlgo. The indices persist across processes; pass them to
    // hipblaslt_ext::getAlgosFromIndex.
    HIPBLASLT_EXPORT hipblasStatus_t getLibraryAlgos(int                   device,
                                                     const Request&        request,
                                                     const Backend&        backend,
                                                     size_t                count,
                                                     size_t                maxWorkspaceBytes,
                                                     std::vector<int32_t>& indices,
                                                     Diagnostics&          diagnostics);

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

    // Adapt a GEMM solution to hipblasLtMatmul / hipblaslt_ext::Gemm. The
    // algorithm is the solution's JIT library index, from 2^30, and any process
    // that can read the library can run it. Other operation kinds return
    // NOT_SUPPORTED. Execution preserves existing handle, workspace and stream
    // requirements; sharing a backend or solution does not relax those
    // concurrency requirements.
    HIPBLASLT_EXPORT hipblasStatus_t getGemmAlgo(const Solution&                   solution,
                                                 hipblasLtMatmulHeuristicResult_t& result,
                                                 Diagnostics&                      diagnostics);
}
