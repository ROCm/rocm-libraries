// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <roc/host_validation/tensor.hpp>
#include <utility>

namespace roc::host_validation {
// Describes a numerically stabilized softmax over one tensor axis.
// referenceSoftmax reads input and writes every logical element of output.
struct SoftmaxProblem {
    SoftmaxProblem(Tensor inputValues, Tensor outputValues, size_t softmaxAxis,
                   ScalarType accumulator)
        : input(std::move(inputValues)),
          output(std::move(outputValues)),
          axis(softmaxAxis),
          accumulatorType(accumulator) {}

    Tensor input;                // Source tensor.
    Tensor output;               // Same-shape caller-owned destination written in full.
    size_t axis;                 // Nonempty dimension normalized independently per slice.
    ScalarType accumulatorType;  // Float32 or Float64 arithmetic.
};

// A slice fixes every coordinate except axis.
struct SoftmaxRunInfo {
    size_t slicesProcessed = 0;        // Product of all input extents except axis.
    size_t outputElementsWritten = 0;  // output.shape().elementCount().
};

SoftmaxRunInfo referenceSoftmax(const SoftmaxProblem& problem);
}  // namespace roc::host_validation
