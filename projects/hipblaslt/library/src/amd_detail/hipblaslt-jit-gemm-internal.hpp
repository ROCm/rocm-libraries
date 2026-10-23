// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-backend.hpp"
#include "rocblaslt-types.h"
#include <array>
#include <cstring>

namespace hipblaslt_ext::experimental::jit::detail
{
    // Reuse hipBLASLt's validated GEMM representation. Device buffers and
    // execution resources retain their normal caller lifetime.
    struct HIPBLASLT_EXPORT GemmRequest
    {
        RocblasltContractionProblem problem;
        // RocblasltContractionProblem stores alpha and beta as pointers and does not own the bytes.
        // GemmRequest outlives the caller's scalars, so it copies them and points problem.alpha/beta at that copy.
        alignas(16) std::array<unsigned char, 16> alpha{}, beta{};

        explicit GemmRequest(const RocblasltContractionProblem& source)
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
        GemmRequest(const GemmRequest&)                = delete;
        GemmRequest&     operator=(const GemmRequest&) = delete;
    };

    // Captures the common descriptor translation without invoking any provider.
    rocblaslt_status createGemmRequest(rocblaslt_handle                    handle,
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
                                       std::shared_ptr<const GemmRequest>& request);
}
