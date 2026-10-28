// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <cstddef>

namespace hipblaslt_ext::experimental::jit::detail
{
    struct GemmRequest;
}

namespace hipblaslt_jit
{
    // The TensileLite problem hipBLASLt solves for a rocBLASLt GEMM.
    TensileLite::ContractionProblemGemm RocblasltToTensileContractionProblemGemm(
        const hipblaslt_ext::experimental::jit::detail::GemmRequest& request);

    // A single strided batched GEMM in TensileLite's canonical index form.
    struct CanonicalGemm
    {
        bool   transA = false, transB = false;
        bool   conjugateA = false, conjugateB = false;
        size_t m = 0, n = 0, k = 0, batch = 0;
    };

    // Why JIT does not implement the problem, or nullptr when it does. A
    // TensileLite ProblemType has no fused all-to-all, so JIT neither generates
    // nor stores solutions for fused GEMM and all-to-all.
    const char* notImplemented(const TensileLite::ContractionProblemGemm& problem);

    // Throws std::runtime_error when a TensileLite ProblemType cannot describe the
    // problem or notImplemented names a reason.
    CanonicalGemm canonicalGemm(const TensileLite::ContractionProblemGemm& problem);

    // Copies the problem into TensileLite::ContractionSolution::ProblemType.
    // Throws like canonicalGemm. problemTypeKey derives the library key from it.
    TensileLite::ContractionSolution::ProblemType
        problemTypeFields(const TensileLite::ContractionProblemGemm& problem);
}
