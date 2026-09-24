// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-gemm-tag.hpp"
#include <array>
#include <cstring>
#include <hipblaslt/hipblaslt-jit-tensilelite.hpp>
#include <memory>

namespace hipblaslt_ext::experimental::jit::tensilelite::detail
{
    // Reuse hipBLASLt's validated GEMM representation. Scalar values are owned;
    // device buffers and execution resources retain their normal caller lifetime.
    struct GemmProblem
    {
        RocblasltContractionProblem problem;
        alignas(16) std::array<unsigned char, 16> alpha{}, beta{};

        explicit GemmProblem(const RocblasltContractionProblem& source)
            : problem(source)
        {
            size_t bytes = 4;
            if(source.a_type == HIP_C_32F)
                bytes = 8;
            else if(source.a_type == HIP_C_64F)
                bytes = 16;
            else if(source.compute_type == rocblaslt_compute_f16
                    || source.compute_type == rocblaslt_compute_f16_pedantic)
                bytes = 2;
            else if(source.compute_type == rocblaslt_compute_f64
                    || source.compute_type == rocblaslt_compute_f64_pedantic)
                bytes = 8;
            if(source.alpha)
                std::memcpy(alpha.data(), source.alpha, bytes);
            if(source.beta)
                std::memcpy(beta.data(), source.beta, bytes);
            problem.alpha = source.alpha ? alpha.data() : nullptr;
            problem.beta  = source.beta ? beta.data() : nullptr;
        }
        GemmProblem(const GemmProblem&)            = delete;
        GemmProblem& operator=(const GemmProblem&) = delete;
    };

    struct Bundle;
    struct PreparedLaunch;

    std::shared_ptr<const Bundle> resolveJitAlgo(const rocblaslt_matmul_algo& algo, int device);
    rocblaslt_status              toRocStatus(hipblasStatus_t status);
    rocblaslt_status              supportJit(rocblaslt_handle             handle,
                                             const rocblaslt_matmul_algo& algo,
                                             const GemmProblem&           request,
                                             size_t&                      workspaceBytes);
    rocblaslt_status              prepareJit(rocblaslt_handle                       handle,
                                             const rocblaslt_matmul_algo&           algo,
                                             const GemmProblem&                     request,
                                             void*                                  workspace,
                                             size_t                                 workspaceBytes,
                                             hipStream_t                            stream,
                                             std::shared_ptr<const PreparedLaunch>& launch);
    rocblaslt_status              runJit(rocblaslt_handle             handle,
                                         const rocblaslt_matmul_algo& algo,
                                         const PreparedLaunch&        launch,
                                         hipStream_t                  stream,
                                         hipEvent_t                   start = nullptr,
                                         hipEvent_t                   stop  = nullptr);

    // Capture validated descriptors and host scalars before invoking TensileLite.
    rocblaslt_status createGemmProblem(rocblaslt_handle                    handle,
                                       rocblaslt_matmul_desc               desc,
                                       const void*                         alpha,
                                       const void*                         A,
                                       rocblaslt_matrix_layout             matA,
                                       const void*                         B,
                                       rocblaslt_matrix_layout             matB,
                                       const void*                         beta,
                                       const void*                         C,
                                       rocblaslt_matrix_layout             matC,
                                       void*                               D,
                                       rocblaslt_matrix_layout             matD,
                                       std::shared_ptr<const GemmProblem>& request);
}
