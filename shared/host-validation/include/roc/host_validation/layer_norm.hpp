// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <optional>
#include <roc/host_validation/tensor.hpp>
#include <utility>

namespace roc::host_validation {
struct LayerNormProblem {
    LayerNormProblem(Tensor inputValues, Tensor outputValues, size_t normalizedAxis,
                     ScalarType accumulator)
        : input(std::move(inputValues)),
          output(std::move(outputValues)),
          axis(normalizedAxis),
          accumulatorType(accumulator) {}

    Tensor input;
    Tensor output;
    std::optional<Tensor> mean;
    std::optional<Tensor> inverseVariance;
    std::optional<Tensor> gamma;
    std::optional<Tensor> beta;
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
