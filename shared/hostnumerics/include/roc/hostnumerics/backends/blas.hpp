// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <roc/hostnumerics/gemm.hpp>

namespace roc::hostnumerics {
void matmulIntoWithBlasBackend(const Tensor& a, const Tensor& b, Tensor output,
                               const MatmulOptions& options = MatmulOptions{},
                               OutputSelection selection = OutputSelection::all());
Tensor matmulWithBlasBackend(const Tensor& a, const Tensor& b, ScalarType outputType,
                             const MatmulOptions& options = MatmulOptions{},
                             std::optional<Layout> outputLayout = std::nullopt);
}  // namespace roc::hostnumerics
