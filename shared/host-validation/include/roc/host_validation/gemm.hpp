// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <roc/host_validation/detail/reference_gemm.hpp>
#include <utility>

namespace roc::host_validation {
struct GemmInvocation {
    explicit GemmInvocation(GemmProblem value, GemmRunOptions options = {})
        : problem(std::move(value)), execution(std::move(options)) {}

    GemmProblem problem;
    GemmRunOptions execution;
};

using GemmResult = GemmRunInfo;

inline GemmSupportInfo queryGemmSupport(const GemmInvocation& invocation) {
    if (invocation.execution.backend == GemmBackend::Automatic)
        return queryGemmSupport(invocation.problem, GemmBackend::Canonical);
    return queryGemmSupport(invocation.problem, invocation.execution.backend,
                            invocation.execution.backendImplementation);
}

inline GemmResult referenceGemm(const GemmInvocation& invocation) {
    return referenceGemm(invocation.problem, invocation.execution);
}
}  // namespace roc::host_validation
