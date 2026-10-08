// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-backend.hpp"
#include <hipblaslt/hipblaslt.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Not installed. The entry points stay exported because the JIT test binaries
// link against the shared library. The request is detail::GemmRequest, the
// configured generator is hipblaslt_jit::Jit, and a published result is
// detail::CompiledSolution.
namespace hipblaslt_ext::experimental::jit
{
    // Look up the JIT solution library on the current HIP device (which must
    // equal device) and publish with jit when the request is not already
    // there. A hit does not generate. Returns the CompiledSolution for the
    // first published index. A capturing stream may return a hit and does not
    // start a build. Unsupported requests return NOT_SUPPORTED.
    // getGemmAlgo turns the solution into a library index.
    HIPBLASLT_EXPORT hipblasStatus_t
        getJitAlgo(int                                                 device,
                   const std::shared_ptr<const detail::GemmRequest>&   request,
                   const std::shared_ptr<const hipblaslt_jit::Jit>&    jit,
                   size_t                                              maxWorkspaceBytes,
                   std::shared_ptr<const detail::CompiledSolution>&    solution,
                   Diagnostics&                                        diagnostics);

    // Up to count solution indices for exactly this request: those already in the
    // JIT solution library in the order they were published, then solutions
    // generated with jit and published now, best first, under the same device
    // rules as getJitAlgo. The indices persist across processes; pass them to
    // hipblaslt_ext::getAlgosFromIndex.
    HIPBLASLT_EXPORT hipblasStatus_t
        getLibraryAlgos(int                                               device,
                        const std::shared_ptr<const detail::GemmRequest>& request,
                        const std::shared_ptr<const hipblaslt_jit::Jit>&  jit,
                        size_t                                            count,
                        size_t                                            maxWorkspaceBytes,
                        std::vector<int32_t>&                             indices,
                        Diagnostics&                                      diagnostics);

    // Build the implemented GEMM request from existing hipBLASLt descriptors.
    HIPBLASLT_EXPORT hipblasStatus_t
        makeGemmRequest(hipblasLtHandle_t                              handle,
                        hipblasLtMatmulDesc_t                          desc,
                        const void*                                    alpha,
                        const void*                                    A,
                        hipblasLtMatrixLayout_t                        layoutA,
                        const void*                                    B,
                        hipblasLtMatrixLayout_t                        layoutB,
                        const void*                                    beta,
                        const void*                                    C,
                        hipblasLtMatrixLayout_t                        layoutC,
                        void*                                          D,
                        hipblasLtMatrixLayout_t                        layoutD,
                        std::shared_ptr<const detail::GemmRequest>&    request,
                        Diagnostics&                                   diagnostics);

    // Adapt a GEMM solution to hipblasLtMatmul / hipblaslt_ext::Gemm. The
    // algorithm is the solution's JIT library index, from 2^30, and any process
    // that can read the library can run it. Execution preserves existing handle,
    // workspace and stream requirements.
    HIPBLASLT_EXPORT hipblasStatus_t
        getGemmAlgo(const std::shared_ptr<const detail::CompiledSolution>& solution,
                    hipblasLtMatmulHeuristicResult_t&                      result,
                    Diagnostics&                                           diagnostics);
}
