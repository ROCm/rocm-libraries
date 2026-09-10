// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

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

// Bound matrix-multiplication state shared by the built-in and optional BLAS
// implementations. Epilogue inputs are intentionally absent: callers compose
// coefficient, addend, bias, activation, and output conversion operations.
struct GemmInvocation : MatmulOptions {
    GemmInvocation(Tensor aTensor, Tensor bTensor, Tensor dTensor,
                   const MatmulOptions& options = MatmulOptions{},
                   OutputSelection selection = OutputSelection::all())
        : MatmulOptions(options),
          a(std::move(aTensor)),
          b(std::move(bTensor)),
          d(std::move(dTensor)),
          outputSelection(std::move(selection)) {}

    Tensor a;
    Tensor b;
    Tensor d;
    OutputSelection outputSelection;
};

struct GemmExecutionInfo {
    GemmBackend backendUsed = GemmBackend::Blocked;
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
using GemmInvocation = detail::GemmInvocation;
using GemmExecutionInfo = detail::GemmExecutionInfo;
}  // namespace roc::host_numerics
