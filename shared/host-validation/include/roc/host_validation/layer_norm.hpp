// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <optional>
#include <roc/host_validation/tensor.hpp>
#include <utility>

namespace roc::host_validation {
struct LayerNormProblem {
    LayerNormProblem(TensorView inputValues, MutableTensorView outputValues, size_t normalizedAxis,
                     ScalarType accumulator)
        : input(std::move(inputValues)),
          output(std::move(outputValues)),
          axis(normalizedAxis),
          accumulatorType(accumulator) {}

    TensorView input;
    MutableTensorView output;
    std::optional<MutableTensorView> mean;
    std::optional<MutableTensorView> inverseVariance;
    std::optional<TensorView> gamma;
    std::optional<TensorView> beta;
    size_t axis;
    ScalarType accumulatorType;
    double epsilon = 1e-5;
};

struct LayerNormRunInfo {
    size_t slicesComputed = 0;
    size_t elementsComputed = 0;
};

LayerNormRunInfo referenceLayerNorm(const LayerNormProblem& problem);
}  // namespace roc::host_validation
