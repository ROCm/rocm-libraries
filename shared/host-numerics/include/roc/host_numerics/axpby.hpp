// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <optional>
#include <roc/host_numerics/tensor.hpp>
#include <utility>

namespace roc::host_numerics {
// Reusable output = alpha * x + beta * y descriptor. It contains the inputs,
// arithmetic policy, and result scalar type, but no destination storage.
struct AxpbyProblem {
    AxpbyProblem(std::optional<Tensor> xValues, std::optional<Tensor> yValues,
                 ScalarType resultType, ScalarType accumulator)
        : x(std::move(xValues)),
          y(std::move(yValues)),
          outputType(resultType),
          accumulatorType(accumulator),
          alpha(Scalar::one(accumulator)),
          beta(Scalar::one(accumulator)) {}

    std::optional<Tensor> x;     // Optional X input; X and Y shapes must match.
    std::optional<Tensor> y;     // Optional Y input; at least one input is required.
    ScalarType outputType;       // Scalar type of the result tensor.
    ScalarType accumulatorType;  // Arithmetic type before output conversion.
    Scalar alpha;                // X coefficient; ignored when X is absent.
    Scalar beta;                 // Y coefficient; ignored when Y is absent.
};

// Binds an AXPBY problem to caller-owned destination storage. Every logical
// output element is overwritten and must have a layout with provably distinct
// storage offsets. The output may exactly alias X or Y with the same type,
// layout, and complete backing-storage range; every other overlapping
// output/input backing-storage range is rejected.
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
}  // namespace roc::host_numerics
