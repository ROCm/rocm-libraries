// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <complex>
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
            const GemmExecution implementationExecution{
                .backend = backendImplementation->backend(),
                .requireRequestedBackend = true,
            };
            const GemmSupportInfo implementationSupport =
                queryGemmSupport(request, implementationExecution, backendImplementation);
            if (implementationSupport)
                return {
                    .output = request.d.asConst(),
                    .runInfo = backendImplementation->run(request),
                };
            fallbackReason = implementationSupport.reason;
        }
        backend = GemmBackend::Canonical;
    }

    const GemmExecution requestedExecution{
        .backend = backend,
        .requireRequestedBackend = execution.requireRequestedBackend,
    };
    const GemmSupportInfo requestedSupport =
        queryGemmSupport(request, requestedExecution, backendImplementation);
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

    const GemmSupportInfo canonicalSupport =
        queryGemmSupport(request, {
                                      .backend = GemmBackend::Canonical,
                                      .requireRequestedBackend = true,
                                  });
    if (!canonicalSupport) throw std::invalid_argument(canonicalSupport.reason);

    GemmRunInfo runInfo;
    switch (request.accumulatorType) {
        case ScalarType::Float16:
        case ScalarType::BFloat16:
        case ScalarType::Float32:
            runInfo = detail::referenceRuntimeCanonical<float>(request);
            break;
        case ScalarType::Float64:
            runInfo = detail::referenceRuntimeCanonical<double>(request);
            break;
        case ScalarType::Int32:
            runInfo = detail::referenceRuntimeCanonical<int32_t>(request);
            break;
        case ScalarType::ComplexFloat32:
            runInfo = detail::referenceRuntimeCanonical<std::complex<float>>(request);
            break;
        case ScalarType::ComplexFloat64:
            runInfo = detail::referenceRuntimeCanonical<std::complex<double>>(request);
            break;
        default:
            throw std::invalid_argument("Unsupported runtime reference GEMM accumulator type.");
    }
    runInfo.fallbackReason = std::move(fallbackReason);
    return {
        .output = request.d.asConst(),
        .runInfo = std::move(runInfo),
    };
}

}  // namespace roc::host_validation
