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

void validateOwnedGemmStorage(const GemmProblem& problem, const Tensor& output) {
    std::vector<const Tensor*> inputs{
        &problem.a.values,
        &problem.b.values,
        &problem.c,
    };
    for (const VectorBinding& binding : problem.a.preQuantizationScales)
        inputs.push_back(&binding.values);
    for (const VectorBinding& binding : problem.b.preQuantizationScales)
        inputs.push_back(&binding.values);
    if (problem.a.blockScale) inputs.push_back(&problem.a.blockScale->values);
    if (problem.b.blockScale) inputs.push_back(&problem.b.blockScale->values);
    if (problem.epilogue.bias) inputs.push_back(&problem.epilogue.bias->values);
    if (problem.epilogue.scaleAlpha) inputs.push_back(&problem.epilogue.scaleAlpha->values);
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

GemmSupportInfo queryGemmSupport(const GemmRequest& request, GemmBackend backend) {
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

GemmRunInfo referenceGemm(const GemmRequest& request, GemmBackend backend) {
    std::optional<std::string> fallbackReason;
    if (backend == GemmBackend::Automatic) {
        const GemmSupportInfo blockedSupport = detail::queryBlockedGemmSupport(request);
        if (blockedSupport && blockedSupport.preferredForAutomaticExecution) {
            GemmRunInfo runInfo = detail::runBlockedGemm(request);
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

    GemmRunInfo runInfo = detail::runPointwiseGemm(request);
    runInfo.fallbackReason = std::move(fallbackReason);
    return runInfo;
}

GemmResult referenceGemm(const GemmProblem& problem, const GemmOutputOptions& output,
                         GemmBackend backend) {
    detail::validateRuntimeGemmProblem(problem);
    const Shape outputShape{problem.a.values.shape()[0], problem.b.values.shape()[1]};
    const Layout outputLayout =
        output.layout.value_or(Layout::contiguousLastDimensionFastest(outputShape));
    if (outputLayout.shape() != outputShape)
        throw std::invalid_argument("Owning reference GEMM output layout shape mismatch.");
    (void)output.selection.selectedCount(outputShape.elementCount());
    const size_t requiredStorageBytes = storageBytesForLayout(problem.outputType, outputLayout);

    Tensor destination(problem.outputType, outputLayout);
    validateOwnedGemmStorage(problem, destination);
    initializeOwnedGemmOutput(destination, requiredStorageBytes);
    GemmRequest request(problem, destination, output.selection);
    GemmRunInfo runInfo = referenceGemm(request, backend);
    return {.output = std::move(destination), .runInfo = std::move(runInfo)};
}

}  // namespace roc::host_numerics
