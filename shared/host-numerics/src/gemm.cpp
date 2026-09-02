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
namespace {
bool gemmTensorStorageOverlaps(const Tensor& left, const Tensor& right) {
    return detail::byteRangesOverlap(left.rawEncodedBackingStorage(),
                                     right.rawEncodedBackingStorage());
}

void validateOwnedGemmStorage(const GemmSpecification& problem, const Tensor& output) {
    std::vector<const Tensor*> inputs{
        &problem.a,
        &problem.b,
        &problem.c,
    };
    for (const Tensor& scale : problem.preQuantizationScalesA) inputs.push_back(&scale);
    for (const Tensor& scale : problem.preQuantizationScalesB) inputs.push_back(&scale);
    if (problem.blockScaleA) inputs.push_back(&*problem.blockScaleA);
    if (problem.blockScaleB) inputs.push_back(&*problem.blockScaleB);
    if (problem.epilogue.bias) inputs.push_back(&*problem.epilogue.bias);
    if (problem.epilogue.scaleAlpha) inputs.push_back(&*problem.epilogue.scaleAlpha);
    if (problem.epilogue.scaleA) inputs.push_back(&*problem.epilogue.scaleA);
    if (problem.epilogue.scaleB) inputs.push_back(&*problem.epilogue.scaleB);

    for (const Tensor* input : inputs) {
        if (gemmTensorStorageOverlaps(output, *input))
            throw std::invalid_argument("Owning reference GEMM output overlaps an input tensor.");
    }
}

void initializeOwnedGemmOutput(const Tensor& output, size_t requiredStorageBytes) {
    std::fill(output.rawEncodedBackingStorage().begin(),
              output.rawEncodedBackingStorage().begin() + requiredStorageBytes, std::byte{0});
    detail::forEachIndex(output.shape(), [&](std::span<const size_t> indices, size_t) {
        output.storeFrom(indices, 0.0);
    });
}
}  // namespace

detail::GemmSupportInfo detail::queryGemmSupport(const GemmInvocation& request,
                                                 GemmBackend backend) {
    try {
        detail::validateRuntimeGemm(request);
    } catch (const std::exception& error) {
        return {.supported = false, .reason = error.what()};
    }

    switch (backend) {
        case GemmBackend::Automatic:
        case GemmBackend::Pointwise:
            return {.supported = true, .reason = {}};
        case GemmBackend::Blocked:
            return detail::queryBlockedGemmSupport(request);
        case GemmBackend::Blas:
            return {
                .supported = false,
                .reason = "The BLAS strategy requires the optional host-numerics BLAS component.",
            };
        case GemmBackend::Mixed:
            return {
                .supported = false,
                .reason = "Mixed is a reporting-only GEMM backend value.",
            };
    }
    return {.supported = false, .reason = "Invalid reference GEMM backend."};
}

detail::GemmExecutionInfo detail::executeGemm(const GemmInvocation& request, GemmBackend backend) {
    std::optional<std::string> fallbackReason;
    if (backend == GemmBackend::Automatic) {
        const GemmSupportInfo blockedSupport = detail::queryBlockedGemmSupport(request);
        if (blockedSupport && blockedSupport.preferredForAutomaticExecution) {
            GemmExecutionInfo runInfo = detail::runBlockedGemm(request);
            runInfo.fallbackReason = std::move(fallbackReason);
            return runInfo;
        }
        if (!blockedSupport && !fallbackReason) fallbackReason = blockedSupport.reason;
        backend = GemmBackend::Pointwise;
    }

    const GemmSupportInfo requestedSupport = queryGemmSupport(request, backend);
    if (!requestedSupport) throw std::invalid_argument(requestedSupport.reason);
    if (backend == GemmBackend::Blocked) {
        return detail::runBlockedGemm(request);
    }

    const GemmSupportInfo pointwiseSupport = queryGemmSupport(request, GemmBackend::Pointwise);
    if (!pointwiseSupport) throw std::invalid_argument(pointwiseSupport.reason);

    GemmExecutionInfo runInfo = detail::runPointwiseGemm(request);
    runInfo.fallbackReason = std::move(fallbackReason);
    return runInfo;
}

GemmBackend referenceGemmInto(Tensor a, Tensor b, Tensor c, Tensor d, const GemmOptions& options,
                              GemmBackend backend) {
    return detail::executeGemm(detail::GemmInvocation(std::move(a), std::move(b), std::move(c),
                                                      std::move(d), options),
                               backend)
        .backendUsed;
}

Tensor referenceGemm(Tensor a, Tensor b, Tensor c, ScalarType outputType,
                     const GemmOptions& options, std::optional<Layout> outputLayout,
                     GemmBackend backend) {
    const GemmSpecification problem(std::move(a), std::move(b), std::move(c), outputType, options);
    detail::validateRuntimeGemmProblem(problem);
    const Shape outputShape{problem.a.shape()[0], problem.b.shape()[1]};
    const Layout layout =
        outputLayout.value_or(Layout::contiguousLastDimensionFastest(outputShape));
    if (layout.shape() != outputShape)
        throw std::invalid_argument("Owning reference GEMM output layout shape mismatch.");
    (void)options.outputSelection.selectedCount(outputShape.elementCount());
    const size_t requiredStorageBytes = storageBytesForLayout(outputType, layout);

    Tensor destination(outputType, layout);
    validateOwnedGemmStorage(problem, destination);
    initializeOwnedGemmOutput(destination, requiredStorageBytes);
    (void)detail::executeGemm(GemmInvocation(problem, destination, options.outputSelection),
                              backend);
    return destination;
}

}  // namespace roc::host_numerics
