// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <roc/host_numerics/gemm.hpp>

namespace roc::host_numerics {
void matmulIntoWithBlasBackend(Tensor a, Tensor b, Tensor output,
                               const MatmulOptions& options = MatmulOptions{});
Tensor matmulWithBlasBackend(Tensor a, Tensor b, ScalarType outputType,
                             const MatmulOptions& options = MatmulOptions{},
                             std::optional<Layout> outputLayout = std::nullopt);
}  // namespace roc::host_numerics
