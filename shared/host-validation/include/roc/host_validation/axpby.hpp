// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <complex>
#include <cstddef>
#include <optional>
#include <roc/host_validation/tensor.hpp>
#include <utility>

namespace roc::host_validation {
// Reusable output = alpha * x + beta * y descriptor. It contains the inputs,
// arithmetic policy, and result scalar type, but no destination storage.
struct AxpbyProblem {
    AxpbyProblem(std::optional<Tensor> xValues, std::optional<Tensor> yValues,
                 ScalarType resultType, ScalarType accumulator)
        : x(std::move(xValues)),
          y(std::move(yValues)),
          outputType(resultType),
          accumulatorType(accumulator) {}

    std::optional<Tensor> x;               // Optional X input; X and Y shapes must match.
    std::optional<Tensor> y;               // Optional Y input; at least one input is required.
    ScalarType outputType;                 // Scalar type of the result tensor.
    ScalarType accumulatorType;            // Arithmetic type before output conversion.
    std::complex<double> alpha{1.0, 0.0};  // X coefficient; ignored when X is absent.
    std::complex<double> beta{1.0, 0.0};   // Y coefficient; ignored when Y is absent.
};

// Binds an AXPBY problem to caller-owned destination storage. Every logical
// output element is overwritten; arbitrary output layouts remain supported.
struct AxpbyRequest : AxpbyProblem {
    AxpbyRequest(std::optional<Tensor> xValues, std::optional<Tensor> yValues, Tensor outputValues,
                 ScalarType accumulator)
        : AxpbyProblem(std::move(xValues), std::move(yValues), outputValues.type(), accumulator),
          output(std::move(outputValues)) {}

    AxpbyRequest(AxpbyProblem problem, Tensor outputValues)
        : AxpbyProblem(std::move(problem)), output(std::move(outputValues)) {}

    Tensor output;
};

// Reports the number of logical output elements written.
struct AxpbyRunInfo {
    size_t outputElementsWritten = 0;  // output.shape().elementCount().
};

// Owning AXPBY result. output uses a contiguous layout; callers requiring a
// specific destination layout use AxpbyRequest.
struct AxpbyResult {
    Tensor output;
    AxpbyRunInfo runInfo;
};

AxpbyRunInfo referenceAxpby(const AxpbyRequest& request);
AxpbyResult referenceAxpby(const AxpbyProblem& problem);
}  // namespace roc::host_validation
