// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <roc/host_numerics/tensor.hpp>

namespace roc::host_numerics {
// Allocates a contiguous output and computes a numerically stabilized softmax
// over axis. The selected axis must be nonempty.
Tensor referenceSoftmax(Tensor input, size_t axis, ScalarType outputType = ScalarType::Float32,
                        ScalarType accumulatorType = ScalarType::Float32);

// Writes every logical element of output. Exact same-layout input/output
// aliasing is allowed; differently mapped overlapping storage is rejected.
void referenceSoftmaxInto(Tensor input, Tensor output, size_t axis,
                          ScalarType accumulatorType = ScalarType::Float32);
}  // namespace roc::host_numerics
