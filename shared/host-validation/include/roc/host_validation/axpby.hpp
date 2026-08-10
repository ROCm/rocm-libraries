// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <complex>
#include <cstddef>
#include <optional>
#include <roc/host_validation/tensor.hpp>
#include <utility>

namespace roc::host_validation {
struct AxpbyProblem {
    AxpbyProblem(std::optional<TensorView> xValues, std::optional<TensorView> yValues,
                 MutableTensorView outputValues, ScalarType accumulator)
        : x(std::move(xValues)),
          y(std::move(yValues)),
          output(std::move(outputValues)),
          accumulatorType(accumulator) {}

    std::optional<TensorView> x;
    std::optional<TensorView> y;
    MutableTensorView output;
    ScalarType accumulatorType;
    std::complex<double> alpha{1.0, 0.0};
    std::complex<double> beta{1.0, 0.0};
};

struct AxpbyRunInfo {
    size_t elementsComputed = 0;
};

AxpbyRunInfo referenceAxpby(const AxpbyProblem& problem);
}  // namespace roc::host_validation
