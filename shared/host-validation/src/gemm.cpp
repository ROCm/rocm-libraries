// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "detail/reference_gemm.hpp"

namespace roc::host_validation {
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

GemmSupportInfo queryGemmSupport(const GemmRequest& request, const GemmExecution& execution,
                                 const GemmBackendImplementation* backendImplementation) {
    try {
        detail::validateRuntimeGemm(request);
    } catch (const std::exception& error) {
        return {.supported = false, .reason = error.what()};
    }

    const GemmBackend backend = execution.backend;

    switch (backend) {
        case GemmBackend::Automatic:
        case GemmBackend::Pointwise:
            return {.supported = true, .reason = {}};
        case GemmBackend::Blocked:
        case GemmBackend::Blas:
            if (backendImplementation == nullptr)
                return {
                    .supported = false,
                    .reason =
                        "No implementation was supplied for the requested "
                        "runtime GEMM backend.",
                };
            if (backendImplementation->backend() != backend)
                return {
                    .supported = false,
                    .reason =
                        "The supplied runtime GEMM implementation does not "
                        "match the requested backend.",
                };
            return backendImplementation->querySupport(request);
    }
    return {.supported = false, .reason = "Invalid reference GEMM backend."};
}

GemmRunInfo referenceGemm(const GemmRequest& request, const GemmExecution& execution,
                          const GemmBackendImplementation* backendImplementation) {
    GemmBackend backend = execution.backend;
    std::optional<std::string> fallbackReason;
    if (backend == GemmBackend::Automatic) {
        if (backendImplementation != nullptr) {
            const GemmSupportInfo implementationSupport = queryGemmSupport(
                request, {.backend = backendImplementation->backend()}, backendImplementation);
            if (implementationSupport) return backendImplementation->run(request);
            fallbackReason = implementationSupport.reason;
        }
        backend = GemmBackend::Pointwise;
    }

    const GemmSupportInfo requestedSupport =
        queryGemmSupport(request, {.backend = backend}, backendImplementation);
    if (!requestedSupport) {
        if (execution.requireRequestedBackend) throw std::invalid_argument(requestedSupport.reason);
        if (backend == GemmBackend::Pointwise) throw std::invalid_argument(requestedSupport.reason);
        fallbackReason = requestedSupport.reason;
        backend = GemmBackend::Pointwise;
    } else if (backend != GemmBackend::Pointwise)
        return backendImplementation->run(request);

    const GemmSupportInfo pointwiseSupport =
        queryGemmSupport(request, {.backend = GemmBackend::Pointwise});
    if (!pointwiseSupport) throw std::invalid_argument(pointwiseSupport.reason);

    GemmRunInfo runInfo = detail::runPointwiseGemm(request);
    runInfo.fallbackReason = std::move(fallbackReason);
    return runInfo;
}

GemmResult referenceGemm(const GemmProblem& problem, const GemmOutputOptions& output,
                         const GemmExecution& execution,
                         const GemmBackendImplementation* backendImplementation) {
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
    GemmRunInfo runInfo = referenceGemm(request, execution, backendImplementation);
    return {.output = std::move(destination), .runInfo = std::move(runInfo)};
}

}  // namespace roc::host_validation
