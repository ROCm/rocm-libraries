// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-backend.hpp"
#include "rocblaslt-types.h"
#include <array>

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

        // Copies alpha and beta at the width of the problem's alpha and beta type.
        explicit GemmRequest(const RocblasltContractionProblem& source);
        GemmRequest(const GemmRequest&)            = delete;
        GemmRequest& operator=(const GemmRequest&) = delete;
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
