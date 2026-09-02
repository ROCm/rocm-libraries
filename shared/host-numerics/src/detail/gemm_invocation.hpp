// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <roc/host_numerics/gemm.hpp>
#include <string>
#include <utility>

namespace roc::host_numerics::detail {
struct GemmSupportInfo {
    bool supported = false;
    std::string reason;
    bool preferredForAutomaticExecution = true;

    explicit operator bool() const {
        return supported;
    }
};

// Private numerical specification used to share validation and execution code
// between owning and caller-output entry points.
struct GemmSpecification : GemmOptions {
    GemmSpecification(Tensor aTensor, Tensor bTensor, Tensor cTensor, ScalarType output,
                      ScalarType accumulator)
        : GemmSpecification(std::move(aTensor), std::move(bTensor), std::move(cTensor), output,
                            GemmOptions(accumulator)) {}

    GemmSpecification(Tensor aTensor, Tensor bTensor, Tensor cTensor, ScalarType output,
                      const GemmOptions& options)
        : GemmOptions(options),
          a(std::move(aTensor)),
          b(std::move(bTensor)),
          c(std::move(cTensor)),
          outputType(output) {}

    Tensor a;
    Tensor b;
    Tensor c;
    ScalarType outputType;
};

// Private bound execution state shared by the built-in and optional BLAS
// implementations. Public callers use referenceGemm() or referenceGemmInto().
struct GemmInvocation : GemmSpecification {
    GemmInvocation(Tensor aTensor, Tensor bTensor, Tensor cTensor, Tensor dTensor,
                   ScalarType accumulator)
        : GemmInvocation(std::move(aTensor), std::move(bTensor), std::move(cTensor),
                         std::move(dTensor), GemmOptions(accumulator)) {}

    GemmInvocation(Tensor aTensor, Tensor bTensor, Tensor cTensor, Tensor dTensor,
                   const GemmOptions& options)
        : GemmSpecification(std::move(aTensor), std::move(bTensor), std::move(cTensor),
                            dTensor.type(), options),
          d(std::move(dTensor)) {}

    GemmInvocation(GemmSpecification specification, Tensor dTensor,
                   OutputSelection selection = OutputSelection::all())
        : GemmSpecification(std::move(specification)), d(std::move(dTensor)) {
        outputSelection = std::move(selection);
    }

    Tensor d;
};

struct GemmExecutionInfo {
    GemmBackend backendUsed = GemmBackend::Pointwise;
    std::optional<std::string> fallbackReason;
    size_t outputElementsWritten = 0;
    size_t outputElementsCovered = 0;
};

GemmSupportInfo queryGemmSupport(const GemmInvocation& invocation, GemmBackend backend);
GemmExecutionInfo executeGemm(const GemmInvocation& invocation, GemmBackend backend);
GemmSupportInfo queryBlasGemmSupport(const GemmInvocation& invocation, GemmBackend backend);
GemmExecutionInfo executeBlasGemm(const GemmInvocation& invocation, GemmBackend backend);
}  // namespace roc::host_numerics::detail

// Short implementation-only aliases used by compiled source files.
namespace roc::host_numerics {
using GemmSpecification = detail::GemmSpecification;
using GemmInvocation = detail::GemmInvocation;
using GemmExecutionInfo = detail::GemmExecutionInfo;
}  // namespace roc::host_numerics
