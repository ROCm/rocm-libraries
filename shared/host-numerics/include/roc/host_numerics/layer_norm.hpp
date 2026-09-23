// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <optional>
#include <roc/host_numerics/tensor.hpp>

namespace roc::host_numerics {
// Numerical policy and optional affine inputs for population-variance LayerNorm.
struct LayerNormOptions {
    size_t axis = 0;                                   // Nonempty dimension normalized per slice.
    ScalarType accumulatorType = ScalarType::Float32;  // Float32 or Float64 arithmetic.
    std::optional<Tensor> gamma;                       // Optional scale vector indexed by axis.
    std::optional<Tensor> beta;                        // Optional bias vector indexed by axis.
    double epsilon = 1e-5;                             // Finite nonnegative variance offset.
};

// Scalar types to allocate for an owning LayerNorm call. Optional statistic
// types request one value per normalization slice.
struct LayerNormOutputTypes {
    ScalarType output = ScalarType::Float32;
    std::optional<ScalarType> mean;
    std::optional<ScalarType> inverseVariance;
};

// Tensors produced by LayerNorm. The statistic tensors have the input shape
// with the normalized axis removed.
struct LayerNormOutputs {
    Tensor output;
    std::optional<Tensor> mean;
    std::optional<Tensor> inverseVariance;
};

LayerNormOutputs referenceLayerNorm(Tensor input, const LayerNormOutputTypes& outputTypes = {},
                                    const LayerNormOptions& options = {});

// Writes caller-owned output tensors. output may exactly alias input with the
// same mapping; every other overlapping input/output or output/output range is
// rejected.
void referenceLayerNormInto(Tensor input, LayerNormOutputs outputs,
                            const LayerNormOptions& options = {});
}  // namespace roc::host_numerics
