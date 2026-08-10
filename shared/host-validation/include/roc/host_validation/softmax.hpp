// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <roc/host_validation/tensor.hpp>
#include <utility>

namespace roc::host_validation {
struct SoftmaxProblem {
    SoftmaxProblem(TensorView inputValues, MutableTensorView outputValues, size_t softmaxAxis,
                   ScalarType accumulator)
        : input(std::move(inputValues)),
          output(std::move(outputValues)),
          axis(softmaxAxis),
          accumulatorType(accumulator) {}

    TensorView input;
    MutableTensorView output;
    size_t axis;
    ScalarType accumulatorType;
};

struct SoftmaxRunInfo {
    size_t slicesComputed = 0;
    size_t elementsComputed = 0;
};

SoftmaxRunInfo referenceSoftmax(const SoftmaxProblem& problem);
}  // namespace roc::host_validation
