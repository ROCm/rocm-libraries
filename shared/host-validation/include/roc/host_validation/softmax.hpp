// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <roc/host_validation/tensor.hpp>
#include <utility>

namespace roc::host_validation {
// Reusable numerically stabilized softmax descriptor. It contains the input,
// result scalar type, axis, and arithmetic policy, but no destination storage.
struct SoftmaxProblem {
    SoftmaxProblem(Tensor inputValues, ScalarType resultType, size_t softmaxAxis,
                   ScalarType accumulator)
        : input(std::move(inputValues)),
          outputType(resultType),
          axis(softmaxAxis),
          accumulatorType(accumulator) {}

    Tensor input;                // Source tensor.
    ScalarType outputType;       // Scalar type of the result tensor.
    size_t axis;                 // Nonempty dimension normalized independently per slice.
    ScalarType accumulatorType;  // Float32 or Float64 arithmetic.
};

// Binds a softmax problem to caller-owned destination storage. Every logical
// output element is overwritten; arbitrary output layouts remain supported.
struct SoftmaxRequest : SoftmaxProblem {
    SoftmaxRequest(Tensor inputValues, Tensor outputValues, size_t softmaxAxis,
                   ScalarType accumulator)
        : SoftmaxProblem(std::move(inputValues), outputValues.type(), softmaxAxis, accumulator),
          output(std::move(outputValues)) {}

    SoftmaxRequest(SoftmaxProblem problem, Tensor outputValues)
        : SoftmaxProblem(std::move(problem)), output(std::move(outputValues)) {}

    Tensor output;
};

// A slice fixes every coordinate except axis.
struct SoftmaxRunInfo {
    size_t slicesProcessed = 0;        // Product of all input extents except axis.
    size_t outputElementsWritten = 0;  // output.shape().elementCount().
};

// Owning softmax result. output uses a contiguous layout; callers requiring a
// specific destination layout use SoftmaxRequest.
struct SoftmaxResult {
    Tensor output;
    SoftmaxRunInfo runInfo;
};

SoftmaxRunInfo referenceSoftmax(const SoftmaxRequest& request);
SoftmaxResult referenceSoftmax(const SoftmaxProblem& problem);
SoftmaxResult referenceSoftmax(const SoftmaxProblem& problem,
                               const TensorStorageAllocator& allocator);
}  // namespace roc::host_validation
