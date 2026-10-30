// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "../src/detail/gemm_invocation.hpp"
#include "../src/detail/reference_gemm.hpp"

namespace roc::hostnumerics {
using GemmTestOptions = MatmulOptions;

// Native tests inspect the same bound invocation used by the compiled backends.
struct GemmTestCase : detail::GemmInvocation {
    GemmTestCase(Tensor a, Tensor b, Tensor d, ScalarType accumulator)
        : GemmTestCase(std::move(a), std::move(b), std::move(d), MatmulOptions(accumulator)) {}

    GemmTestCase(Tensor a, Tensor b, Tensor d, const MatmulOptions& options)
        : detail::GemmInvocation(a, b, d, options, IndexRange::all(d.elementCount()),
                                 Permutation::identity(d.shape().rank())) {}
};

using GemmSupportInfo = detail::GemmSupportInfo;
using GemmTestRunInfo = detail::GemmExecutionInfo;

inline GemmSupportInfo queryGemmSupport(const GemmTestCase& request, GemmBackend backend) {
    return detail::queryGemmSupport(request, backend);
}
inline GemmTestRunInfo referenceGemm(const GemmTestCase& request, GemmBackend backend) {
    return detail::executeGemm(request, backend);
}
inline GemmSupportInfo queryGemmSupportWithBlasBackend(const GemmTestCase& request,
                                                       GemmBackend backend) {
    return detail::queryBlasGemmSupport(request, backend);
}
inline GemmTestRunInfo referenceGemmWithBlasBackend(const GemmTestCase& request,
                                                    GemmBackend backend) {
    return detail::executeBlasGemm(request, backend);
}
}  // namespace roc::hostnumerics
