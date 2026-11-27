// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-backend.hpp"
#include <hipblaslt/hipblaslt.h>
#include <memory>

// Not installed. Tests link these exports from the shared library.
// getJitAlgo returns one JIT solution-library index. makeGemmRequest copies a
// GEMM into GemmRequest. addHeuristicBackend registers a generator for
// HIPBLASLT_JIT heuristic queries.
namespace hipblaslt_jit
{
    HIPBLASLT_EXPORT hipblasStatus_t getJitAlgo(int                                       device,
                                                const std::shared_ptr<const GemmRequest>& request,
                                                const std::shared_ptr<const Jit>&         jit,
                                                size_t                            maxWorkspaceBytes,
                                                hipblasLtMatmulHeuristicResult_t& result,
                                                Diagnostics&                      diagnostics);

    HIPBLASLT_EXPORT void addHeuristicBackend(std::shared_ptr<const Jit> jit);

    HIPBLASLT_EXPORT hipblasStatus_t makeGemmRequest(hipblasLtHandle_t                   handle,
                                                     hipblasLtMatmulDesc_t               desc,
                                                     const void*                         alpha,
                                                     const void*                         A,
                                                     hipblasLtMatrixLayout_t             layoutA,
                                                     const void*                         B,
                                                     hipblasLtMatrixLayout_t             layoutB,
                                                     const void*                         beta,
                                                     const void*                         C,
                                                     hipblasLtMatrixLayout_t             layoutC,
                                                     void*                               D,
                                                     hipblasLtMatrixLayout_t             layoutD,
                                                     std::shared_ptr<const GemmRequest>& request,
                                                     Diagnostics& diagnostics);
}
