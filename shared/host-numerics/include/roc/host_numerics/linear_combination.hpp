// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <roc/host_numerics/tensor.hpp>

namespace roc::host_numerics {
// Arithmetic policy for output = alpha * x + beta * y.
struct LinearCombinationOptions {
    explicit LinearCombinationOptions(ScalarType accumulator = ScalarType::Float32)
        : accumulatorType(accumulator),
          alpha(Tensor::scalar(accumulator, 1)),
          beta(Tensor::scalar(accumulator, 1)) {}

    ScalarType accumulatorType;  // Arithmetic type before output conversion.
    Tensor alpha;                // Rank-zero X coefficient; ignored when X is absent.
    Tensor beta;                 // Rank-zero Y coefficient; ignored when Y is absent.
};

// Allocates a contiguous output tensor. At least one input must be present;
// two inputs use NumPy-style trailing-dimension broadcasting.
Tensor linearCombination(std::optional<Tensor> x, std::optional<Tensor> y, ScalarType outputType,
                         const LinearCombinationOptions& options = LinearCombinationOptions{});

// Writes every logical element of output. Exact same-layout aliasing with an
// input is allowed; differently mapped overlapping storage is rejected.
void linearCombinationInto(std::optional<Tensor> x, std::optional<Tensor> y, Tensor output,
                           const LinearCombinationOptions& options = LinearCombinationOptions{});
}  // namespace roc::host_numerics
