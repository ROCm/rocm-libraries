// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <complex>
#include <cstddef>
#include <optional>
#include <roc/host_validation/tensor.hpp>
#include <utility>

namespace roc::host_validation {
// Describes output = alpha * x + beta * y over one tensor shape. At least one
// input is present. referenceAxpby reads the present inputs and writes every
// logical element of output.
struct AxpbyProblem {
    AxpbyProblem(std::optional<Tensor> xValues, std::optional<Tensor> yValues, Tensor outputValues,
                 ScalarType accumulator)
        : x(std::move(xValues)),
          y(std::move(yValues)),
          output(std::move(outputValues)),
          accumulatorType(accumulator) {}

    std::optional<Tensor> x;               // Optional X input; shape must equal output.
    std::optional<Tensor> y;               // Optional Y input; shape must equal output.
    Tensor output;                         // Caller-owned destination written in full.
    ScalarType accumulatorType;            // Arithmetic type used before output conversion.
    std::complex<double> alpha{1.0, 0.0};  // X coefficient; ignored when X is absent.
    std::complex<double> beta{1.0, 0.0};   // Y coefficient; ignored when Y is absent.
};

// Reports the number of logical output elements written.
struct AxpbyRunInfo {
    size_t outputElementsWritten = 0;  // output.shape().elementCount().
};

AxpbyRunInfo referenceAxpby(const AxpbyProblem& problem);
}  // namespace roc::host_validation
