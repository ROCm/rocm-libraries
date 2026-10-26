// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <roc/hostnumerics/gemm.hpp>
#include <string>
#include <utility>

#include "index_iteration.hpp"

namespace roc::hostnumerics::detail {
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
    template <typename Indices>
    GemmInvocation(Tensor aTensor, Tensor bTensor, Tensor dTensor, const MatmulOptions& options,
                   const Indices& indices, std::span<const size_t> order)
        : MatmulOptions(options),
          a(std::move(aTensor)),
          b(std::move(bTensor)),
          d(std::move(dTensor)) {
        bindIndices(*this, d.shape(), indices, order);
    }

    template <typename Indices>
    GemmInvocation(Tensor a, Tensor b, Tensor d, const MatmulOptions& options,
                   const Indices& indices, LastDimensionFastest)
        : GemmInvocation(a, b, d, options, indices, d.shape().indexOrderLastDimensionFastest()) {}

    Tensor a;
    Tensor b;
    Tensor d;
    size_t indicesCount = 0;
    std::vector<size_t> indicesOrder;
    std::function<size_t(size_t)> indexAt;
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
}  // namespace roc::hostnumerics::detail

// Short implementation-only aliases used by compiled source files.
namespace roc::hostnumerics {
using GemmInvocation = detail::GemmInvocation;
using GemmExecutionInfo = detail::GemmExecutionInfo;
}  // namespace roc::hostnumerics
