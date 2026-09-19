// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "blas_transforming.hpp"

namespace roc::host_numerics {
void matmulIntoWithBlasBackend(const Tensor& a, const Tensor& b, Tensor output,
                               const MatmulOptions& options, OutputSelection selection) {
    (void)detail::executeBlasGemm(
        GemmInvocation(a, b, std::move(output), options, std::move(selection)),
        GemmBackend::Automatic);
}

Tensor matmulWithBlasBackend(const Tensor& a, const Tensor& b, ScalarType outputType,
                             const MatmulOptions& options, std::optional<Layout> outputLayout) {
    detail::requireRank(a.shape(), 2, "matmul", "A");
    detail::requireRank(b.shape(), 2, "matmul", "B");
    if (a.shape()[1] != b.shape()[0])
        throw std::invalid_argument("matmul reduction dimension mismatch.");
    const Shape outputShape{a.shape()[0], b.shape()[1]};
    const bool useDefaultLayout = !outputLayout.has_value();
    const Layout layout =
        outputLayout.value_or(Layout::contiguousLastDimensionFastest(outputShape));
    if (layout.shape() != outputShape)
        throw std::invalid_argument("matmul output layout shape mismatch.");
    Tensor output = useDefaultLayout && !scalarTypeInfo(outputType).isPacked()
                        ? Tensor::allocateUninitialized(outputType, layout)
                        : Tensor(outputType, layout);
    matmulIntoWithBlasBackend(a, b, output, options);
    return output;
}

GemmSupportInfo detail::queryBlasGemmSupport(const GemmInvocation& problem, GemmBackend backend) {
    if (backend == GemmBackend::Blas) return queryTransformingBlasGemmSupport(problem);
    return detail::queryGemmSupport(problem, backend);
}

detail::GemmExecutionInfo detail::executeBlasGemm(const GemmInvocation& problem,
                                                  GemmBackend backend) {
    if (backend == GemmBackend::Blas) {
        const GemmSupportInfo support = queryTransformingBlasGemmSupport(problem);
        if (!support) throw std::invalid_argument(support.reason);
        return runTransformingBlasGemm(problem);
    }
    if (backend != GemmBackend::Automatic) return detail::executeGemm(problem, backend);

    const GemmSupportInfo blasSupport = queryTransformingBlasGemmSupport(problem);
    if (blasSupport && blasSupport.preferredForAutomaticExecution)
        return runTransformingBlasGemm(problem);

    GemmExecutionInfo runInfo = detail::executeGemm(problem, GemmBackend::Automatic);
    if (!blasSupport) runInfo.fallbackReason = blasSupport.reason;
    return runInfo;
}

}  // namespace roc::host_numerics
