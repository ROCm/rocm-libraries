// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <roc/host_numerics/gemm.hpp>
#include <string>
#include <utility>

namespace roc::host_numerics::detail {
// Private numerical specification used to share validation and execution code
// between owning and caller-output entry points.
struct GemmSpecification : GemmOptions {
    GemmSpecification(GemmOperand aOperand, GemmOperand bOperand, Tensor cTensor, ScalarType output,
                      ScalarType accumulator)
        : GemmSpecification(std::move(aOperand), std::move(bOperand), std::move(cTensor), output,
                            GemmOptions(accumulator)) {}

    GemmSpecification(GemmOperand aOperand, GemmOperand bOperand, Tensor cTensor, ScalarType output,
                      const GemmOptions& options)
        : GemmOptions(options),
          a(std::move(aOperand)),
          b(std::move(bOperand)),
          c(std::move(cTensor)),
          outputType(output) {}

    GemmOperand a;
    GemmOperand b;
    Tensor c;
    ScalarType outputType;
};

// Private bound execution state shared by the built-in and optional BLAS
// implementations. Public callers use referenceGemm() or referenceGemmInto().
struct GemmInvocation : GemmSpecification {
    GemmInvocation(GemmOperand aOperand, GemmOperand bOperand, Tensor cTensor, Tensor dTensor,
                   ScalarType accumulator)
        : GemmInvocation(std::move(aOperand), std::move(bOperand), std::move(cTensor),
                         std::move(dTensor), GemmOptions(accumulator)) {}

    GemmInvocation(GemmOperand aOperand, GemmOperand bOperand, Tensor cTensor, Tensor dTensor,
                   const GemmOptions& options)
        : GemmSpecification(std::move(aOperand), std::move(bOperand), std::move(cTensor),
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
