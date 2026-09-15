// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "../src/detail/gemm_invocation.hpp"
#include "../src/detail/reference_gemm.hpp"

namespace roc::host_numerics {
// Test-only access to backend selection and coverage counters. Numerical
// inputs remain the same Tensor and MatmulOptions values used by the public
// matmul API.
struct GemmTestOptions : MatmulOptions {
    explicit GemmTestOptions(ScalarType accumulator = ScalarType::Float32)
        : MatmulOptions(accumulator) {}

    OutputSelection outputSelection = OutputSelection::all();
};

struct GemmTestCase : GemmTestOptions {
    GemmTestCase(Tensor aTensor, Tensor bTensor, Tensor dTensor, ScalarType accumulator)
        : GemmTestOptions(accumulator),
          a(std::move(aTensor)),
          b(std::move(bTensor)),
          d(std::move(dTensor)) {}

    GemmTestCase(Tensor aTensor, Tensor bTensor, Tensor dTensor, const GemmTestOptions& options)
        : GemmTestOptions(options),
          a(std::move(aTensor)),
          b(std::move(bTensor)),
          d(std::move(dTensor)) {}

    Tensor a;
    Tensor b;
    Tensor d;
};

using GemmSupportInfo = detail::GemmSupportInfo;
using GemmTestRunInfo = detail::GemmExecutionInfo;

inline MatmulOptions testMatmulOptions(const GemmTestOptions& options) {
    return static_cast<const MatmulOptions&>(options);
}

inline GemmSupportInfo queryGemmSupport(const GemmTestCase& request,
                                        GemmBackend backend = GemmBackend::Automatic) {
    return detail::queryGemmSupport(
        detail::GemmInvocation(request.a, request.b, request.d, testMatmulOptions(request),
                               request.outputSelection),
        backend);
}

inline GemmTestRunInfo referenceGemm(const GemmTestCase& request,
                                     GemmBackend backend = GemmBackend::Automatic) {
    return detail::executeGemm(
        detail::GemmInvocation(request.a, request.b, request.d, testMatmulOptions(request),
                               request.outputSelection),
        backend);
}

inline GemmSupportInfo queryGemmSupportWithBlasBackend(
    const GemmTestCase& request, GemmBackend backend = GemmBackend::Automatic) {
    return detail::queryBlasGemmSupport(
        detail::GemmInvocation(request.a, request.b, request.d, testMatmulOptions(request),
                               request.outputSelection),
        backend);
}

inline GemmTestRunInfo referenceGemmWithBlasBackend(const GemmTestCase& request,
                                                    GemmBackend backend = GemmBackend::Automatic) {
    return detail::executeBlasGemm(
        detail::GemmInvocation(request.a, request.b, request.d, testMatmulOptions(request),
                               request.outputSelection),
        backend);
}

}  // namespace roc::host_numerics
