// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <roc/host_validation/tensor.hpp>
#include <utility>
#include <vector>

namespace roc::host_validation {
enum class ReductionOperation {
    Sum,              // Adds all values in each reduction slice.
    MaximumAbsolute,  // Ignores NaN magnitudes and returns the largest remaining magnitude.
};

// Describes a reduction from input into the caller-owned output. Output shape
// is input shape with axes removed, preserving the remaining dimension order.
// referenceReduce writes every logical output element.
struct ReductionProblem {
    ReductionProblem(Tensor inputTensor, Tensor outputTensor, ScalarType accumulator,
                     std::vector<size_t> reductionAxes,
                     ReductionOperation reductionOperation = ReductionOperation::Sum)
        : input(std::move(inputTensor)),
          output(std::move(outputTensor)),
          accumulatorType(accumulator),
          axes(std::move(reductionAxes)),
          operation(reductionOperation) {}

    Tensor input;                  // Source tensor read once per output/reduction coordinate pair.
    Tensor output;                 // Caller-owned reduced destination written in full.
    ScalarType accumulatorType;    // Arithmetic type used before output conversion.
    std::vector<size_t> axes;      // Unique input dimensions removed by the reduction.
    ReductionOperation operation;  // Reduction applied to each output coordinate.
};

// Counts logical output writes and logical input reads. inputElementsRead is
// outputElementsWritten times the product of the reduced extents.
struct ReductionRunInfo {
    size_t outputElementsWritten = 0;  // output.shape().elementCount().
    size_t inputElementsRead = 0;      // Logical reads performed by the nested reduction loops.
};

ReductionRunInfo referenceReduce(const ReductionProblem& problem);
ReductionRunInfo referenceSum(const ReductionProblem& problem);
ReductionRunInfo referenceMaximumAbsolute(Tensor input, Tensor output, ScalarType accumulatorType);
}  // namespace roc::host_validation
