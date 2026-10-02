// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-json.hpp"
#include <cstddef>

namespace TensileLite
{
    class ContractionProblemGemm;
}

namespace hipblaslt_ext::experimental::jit::detail
{
    struct GemmRequest;
}

namespace hipblaslt_jit
{
    // The Tensile problem hipBLASLt solves for the request.
    TensileLite::ContractionProblemGemm
        lowerForJit(const hipblaslt_ext::experimental::jit::detail::GemmRequest& request);

    // A single strided batched GEMM in Tensile's canonical index form.
    struct CanonicalGemm
    {
        bool   transA = false, transB = false;
        bool   conjugateA = false, conjugateB = false;
        size_t m = 0, n = 0, k = 0, batch = 0;
    };

    // Why JIT does not implement the problem, or nullptr when it does. A
    // Tensile ProblemType has no fused all-to-all, so JIT neither generates
    // nor stores solutions for fused GEMM and all-to-all.
    const char* notImplemented(const TensileLite::ContractionProblemGemm& problem);

    // Throws std::runtime_error when a Tensile ProblemType cannot describe the
    // problem or notImplemented names a reason.
    CanonicalGemm canonicalGemm(const TensileLite::ContractionProblemGemm& problem);

    // The problem's Tensile ProblemType fields, which also key the JIT
    // solution library. Throws like canonicalGemm.
    json::Members problemTypeFields(const TensileLite::ContractionProblemGemm& problem);
}
