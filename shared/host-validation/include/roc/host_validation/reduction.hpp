// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <roc/host_validation/tensor.hpp>
#include <utility>
#include <vector>

namespace roc::host_validation {
enum class ReductionOperation {
    Sum,
    MaximumAbsolute,
};

struct ReductionProblem {
    ReductionProblem(Tensor inputTensor, Tensor outputTensor, ScalarType accumulator,
                     std::vector<size_t> reductionAxes,
                     ReductionOperation reductionOperation = ReductionOperation::Sum)
        : input(std::move(inputTensor)),
          output(std::move(outputTensor)),
          accumulatorType(accumulator),
          axes(std::move(reductionAxes)),
          operation(reductionOperation) {}

    Tensor input;
    Tensor output;
    ScalarType accumulatorType;
    std::vector<size_t> axes;
    ReductionOperation operation;
};

struct ReductionRunInfo {
    size_t outputElementsComputed = 0;
    size_t inputElementsRead = 0;
};

ReductionRunInfo referenceReduce(const ReductionProblem& problem);
ReductionRunInfo referenceSum(const ReductionProblem& problem);
ReductionRunInfo referenceMaximumAbsolute(Tensor input, Tensor output, ScalarType accumulatorType);
}  // namespace roc::host_validation
