// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <roc/hostnumerics/gemm.hpp>

namespace roc::hostnumerics {
void matmulIntoWithBlasBackend(const Tensor& a, const Tensor& b, Tensor output,
                               const MatmulOptions& options, const IndexRange& indices,
                               std::span<const size_t> order);
void matmulIntoWithBlasBackend(const Tensor& a, const Tensor& b, Tensor output,
                               const MatmulOptions& options, const std::set<size_t>& indices,
                               std::span<const size_t> order);
Tensor matmulWithBlasBackend(const Tensor& a, const Tensor& b, ScalarType outputType,
                             const MatmulOptions& options, std::optional<Layout> outputLayout);

template <typename Indices>
void matmulIntoWithBlasBackend(const Tensor& a, const Tensor& b, Tensor output,
                               const MatmulOptions& options, const Indices& indices,
                               LastDimensionFastest) {
    matmulIntoWithBlasBackend(a, b, output, options, indices,
                              output.shape().indexOrderLastDimensionFastest());
}

}  // namespace roc::hostnumerics
