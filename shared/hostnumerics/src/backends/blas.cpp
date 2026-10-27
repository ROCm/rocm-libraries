// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "blas_transforming.hpp"

namespace roc::hostnumerics {
void matmulIntoWithBlasBackend(const Tensor& a, const Tensor& b, Tensor output,
                               const MatmulOptions& options, const IndexRange& indices,
                               std::span<const size_t> order) {
    (void)detail::executeBlasGemm(GemmInvocation(a, b, std::move(output), options, indices, order),
                                  GemmBackend::Automatic);
}
void matmulIntoWithBlasBackend(const Tensor& a, const Tensor& b, Tensor output,
                               const MatmulOptions& options, const std::set<size_t>& indices,
                               std::span<const size_t> order) {
    (void)detail::executeBlasGemm(GemmInvocation(a, b, std::move(output), options, indices, order),
                                  GemmBackend::Automatic);
}

Tensor matmulWithBlasBackend(const Tensor& a, const Tensor& b, ScalarType outputType,
                             const MatmulOptions& options, std::optional<Layout> outputLayout) {
    detail::requireRank(a.shape(), 2, "matmul", "A");
    detail::requireRank(b.shape(), 2, "matmul", "B");
    if (a.shape()[1] != b.shape()[0])
        throw std::runtime_error("matmul reduction dimension mismatch.");
    const Shape outputShape{a.shape()[0], b.shape()[1]};
    const bool useDefaultLayout = !outputLayout.has_value();
    const Layout layout =
        outputLayout ? *outputLayout : Layout::contiguousLastDimensionFastest(outputShape);
    if (layout.shape() != outputShape)
        throw std::runtime_error("matmul output layout shape mismatch.");
    Tensor output = useDefaultLayout && !scalarTypeInfo(outputType).isPacked()
                        ? Tensor::allocateUninitialized(outputType, layout)
                        : Tensor(outputType, layout);
    matmulIntoWithBlasBackend(a, b, output, options, IndexRange::all(output.elementCount()),
                              output.shape().indexOrderLastDimensionFastest());
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
        if (!support) throw std::runtime_error(support.reason);
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

}  // namespace roc::hostnumerics
