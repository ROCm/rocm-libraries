// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <roc/host_numerics/gemm.hpp>

namespace roc::host_numerics {
// Provides BLAS acceleration for direct and transformed GEMM requests.
// Runtime-typed, scaled, or compute-quantized operands are materialized as
// accumulator-typed scratch matrices only when needed. Directly compatible
// requests bypass those transformations. Supported output scaling/conversion
// is applied while writing the caller-owned output tensor. Automatic tries
// BLAS first when its cost policy prefers BLAS, then delegates to the built-in
// Blocked implementation.
GemmBackend referenceGemmIntoWithBlasBackend(Tensor a, Tensor b, Tensor c, Tensor d,
                                             const GemmOptions& options = GemmOptions{},
                                             GemmBackend backend = GemmBackend::Automatic);

Tensor referenceGemmWithBlasBackend(Tensor a, Tensor b, Tensor c, ScalarType outputType,
                                    const GemmOptions& options = GemmOptions{},
                                    std::optional<Layout> outputLayout = std::nullopt,
                                    GemmBackend backend = GemmBackend::Automatic);
}  // namespace roc::host_numerics
