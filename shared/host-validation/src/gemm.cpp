// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <optional>
#include <stdexcept>
#include <utility>

#include "detail/reference_gemm.hpp"

namespace roc::host_validation {
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
        case GemmBackend::Canonical:
            return {.supported = true, .reason = {}};
        case GemmBackend::Tiled:
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

GemmResult referenceGemm(const GemmRequest& request, const GemmExecution& execution,
                         const GemmBackendImplementation* backendImplementation) {
    GemmBackend backend = execution.backend;
    std::optional<std::string> fallbackReason;
    if (backend == GemmBackend::Automatic) {
        if (backendImplementation != nullptr) {
            const GemmSupportInfo implementationSupport = queryGemmSupport(
                request, {.backend = backendImplementation->backend()}, backendImplementation);
            if (implementationSupport)
                return {
                    .output = request.d.asConst(),
                    .runInfo = backendImplementation->run(request),
                };
            fallbackReason = implementationSupport.reason;
        }
        backend = GemmBackend::Canonical;
    }

    const GemmSupportInfo requestedSupport =
        queryGemmSupport(request, {.backend = backend}, backendImplementation);
    if (!requestedSupport) {
        if (execution.requireRequestedBackend) throw std::invalid_argument(requestedSupport.reason);
        if (backend == GemmBackend::Canonical) throw std::invalid_argument(requestedSupport.reason);
        fallbackReason = requestedSupport.reason;
        backend = GemmBackend::Canonical;
    } else if (backend != GemmBackend::Canonical) {
        return {
            .output = request.d.asConst(),
            .runInfo = backendImplementation->run(request),
        };
    }

    const GemmSupportInfo pointwiseSupport =
        queryGemmSupport(request, {.backend = GemmBackend::Canonical});
    if (!pointwiseSupport) throw std::invalid_argument(pointwiseSupport.reason);

    GemmRunInfo runInfo = detail::runPointwiseGemm(request);
    runInfo.fallbackReason = std::move(fallbackReason);
    return {
        .output = request.d.asConst(),
        .runInfo = std::move(runInfo),
    };
}

}  // namespace roc::host_validation
