// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "../src/detail/gemm_invocation.hpp"

namespace roc::host_numerics {
// Test-only access to the private bound invocation keeps backend conformance
// tests focused on execution details without restoring that type to the public
// API.
using GemmOperand = detail::GemmOperand;
using GemmTestSpecification = GemmSpecification;
using GemmTestCase = GemmInvocation;
using GemmTestRunInfo = GemmExecutionInfo;

struct GemmTestOutputOptions {
    std::optional<Layout> layout;
    OutputSelection selection = OutputSelection::all();
};

struct GemmTestResult {
    Tensor output;
    GemmTestRunInfo runInfo;
};

inline GemmTestRunInfo referenceGemm(const GemmTestCase& request,
                                     GemmBackend backend = GemmBackend::Automatic) {
    return detail::executeGemm(request, backend);
}

inline GemmTestResult referenceGemm(const GemmTestSpecification& problem,
                                    const GemmTestOutputOptions& output = {},
                                    GemmBackend backend = GemmBackend::Automatic) {
    const Shape outputShape{problem.a.values.shape()[0], problem.b.values.shape()[1]};
    const Layout outputLayout =
        output.layout.value_or(Layout::contiguousLastDimensionFastest(outputShape));
    Tensor destination(problem.outputType, outputLayout);
    GemmTestCase request(problem, destination, output.selection);
    GemmTestRunInfo runInfo = detail::executeGemm(request, backend);
    return {.output = std::move(destination), .runInfo = std::move(runInfo)};
}

inline GemmSupportInfo queryGemmSupportWithBlasBackend(
    const GemmTestCase& request, GemmBackend backend = GemmBackend::Automatic) {
    return detail::queryBlasGemmSupport(request, backend);
}

inline GemmTestRunInfo referenceGemmWithBlasBackend(const GemmTestCase& request,
                                                    GemmBackend backend = GemmBackend::Automatic) {
    return detail::executeBlasGemm(request, backend);
}

inline GemmTestResult referenceGemmWithBlasBackend(const GemmTestSpecification& problem,
                                                   const GemmTestOutputOptions& output = {},
                                                   GemmBackend backend = GemmBackend::Automatic) {
    const Shape outputShape{problem.a.values.shape()[0], problem.b.values.shape()[1]};
    const Layout outputLayout =
        output.layout.value_or(Layout::contiguousLastDimensionFastest(outputShape));
    Tensor destination(problem.outputType, outputLayout);
    GemmTestCase request(problem, destination, output.selection);
    GemmTestRunInfo runInfo = detail::executeBlasGemm(request, backend);
    return {.output = std::move(destination), .runInfo = std::move(runInfo)};
}
}  // namespace roc::host_numerics
