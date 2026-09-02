// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <roc/host_numerics/tensor.hpp>
#include <utility>
#include <vector>

namespace roc::host_numerics {
enum class ReductionOperation {
    Sum,              // Adds all values in each reduction slice.
    MaximumAbsolute,  // Ignores NaN magnitudes and returns the largest remaining magnitude.
};

// Allocates a contiguous output whose shape is input.shape() with axes removed.
Tensor referenceReduce(Tensor input, std::vector<size_t> axes, ReductionOperation operation,
                       ScalarType outputType, ScalarType accumulatorType);

// Writes the reduction into caller-owned output storage. Exact same-layout
// input/output aliasing is allowed; differently mapped overlap is rejected.
void referenceReduceInto(Tensor input, Tensor output, std::vector<size_t> axes,
                         ReductionOperation operation, ScalarType accumulatorType);

Tensor referenceSum(Tensor input, std::vector<size_t> axes, ScalarType outputType,
                    ScalarType accumulatorType);
void referenceSumInto(Tensor input, Tensor output, std::vector<size_t> axes,
                      ScalarType accumulatorType);

Tensor referenceMaximumAbsolute(Tensor input, ScalarType outputType, ScalarType accumulatorType);
void referenceMaximumAbsoluteInto(Tensor input, Tensor output, ScalarType accumulatorType);
}  // namespace roc::host_numerics
