// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "detail/blocked_gemm.hpp"
#include "detail/reference_gemm.hpp"

namespace roc::host_numerics {
void matmulInto(const Tensor& a, const Tensor& b, Tensor output, const MatmulOptions& options,
                OutputSelection selection, GemmBackend backend) {
    (void)detail::executeGemm(
        GemmInvocation(a, b, std::move(output), options, std::move(selection)), backend);
}

Tensor matmul(const Tensor& a, const Tensor& b, ScalarType outputType, const MatmulOptions& options,
              std::optional<Layout> outputLayout, GemmBackend backend) {
    detail::requireRank(a.shape(), 2, "matmul", "A");
    detail::requireRank(b.shape(), 2, "matmul", "B");
    if (a.shape()[1] != b.shape()[0])
        throw std::invalid_argument("matmul reduction dimension mismatch.");
    const Shape outputShape{a.shape()[0], b.shape()[1]};
    const Layout layout =
        outputLayout.value_or(Layout::contiguousLastDimensionFastest(outputShape));
    if (layout.shape() != outputShape)
        throw std::invalid_argument("matmul output layout shape mismatch.");
    Tensor output(outputType, layout);
    matmulInto(a, b, output, options, OutputSelection::all(), backend);
    return output;
}

detail::GemmSupportInfo detail::queryGemmSupport(const GemmInvocation& request,
                                                 GemmBackend backend) {
    try {
        detail::validateRuntimeGemm(request);
    } catch (const std::exception& error) {
        return {.supported = false, .reason = error.what()};
    }

    switch (backend) {
        case GemmBackend::Automatic:
        case GemmBackend::Blocked:
            return detail::queryBlockedGemmSupport(request);
        case GemmBackend::Blas:
            return {
                .supported = false,
                .reason = "The BLAS strategy requires the optional host-numerics BLAS component.",
            };
    }
    return {.supported = false, .reason = "Invalid reference GEMM backend."};
}

detail::GemmExecutionInfo detail::executeGemm(const GemmInvocation& request, GemmBackend backend) {
    if (backend == GemmBackend::Automatic) backend = GemmBackend::Blocked;

    const GemmSupportInfo requestedSupport = queryGemmSupport(request, backend);
    if (!requestedSupport) throw std::invalid_argument(requestedSupport.reason);
    return detail::runBlockedGemm(request);
}

}  // namespace roc::host_numerics
