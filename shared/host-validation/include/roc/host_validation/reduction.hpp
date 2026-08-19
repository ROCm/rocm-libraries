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

// Reusable reduction descriptor. The output shape is the input shape with
// axes removed, preserving the remaining dimension order.
struct ReductionProblem {
    ReductionProblem(Tensor inputTensor, ScalarType resultType, ScalarType accumulator,
                     std::vector<size_t> reductionAxes,
                     ReductionOperation reductionOperation = ReductionOperation::Sum)
        : input(std::move(inputTensor)),
          outputType(resultType),
          accumulatorType(accumulator),
          axes(std::move(reductionAxes)),
          operation(reductionOperation) {}

    Tensor input;                  // Source tensor read once per output/reduction coordinate pair.
    ScalarType outputType;         // Scalar type of the reduced result.
    ScalarType accumulatorType;    // Arithmetic type used before output conversion.
    std::vector<size_t> axes;      // Unique input dimensions removed by the reduction.
    ReductionOperation operation;  // Reduction applied to each output coordinate.
};

// Binds a reduction problem to caller-owned destination storage. Every logical
// output element is overwritten; arbitrary output layouts remain supported.
struct ReductionRequest : ReductionProblem {
    ReductionRequest(Tensor inputTensor, Tensor outputTensor, ScalarType accumulator,
                     std::vector<size_t> reductionAxes,
                     ReductionOperation reductionOperation = ReductionOperation::Sum)
        : ReductionProblem(std::move(inputTensor), outputTensor.type(), accumulator,
                           std::move(reductionAxes), reductionOperation),
          output(std::move(outputTensor)) {}

    ReductionRequest(ReductionProblem problem, Tensor outputTensor)
        : ReductionProblem(std::move(problem)), output(std::move(outputTensor)) {}

    Tensor output;
};

// Counts logical output writes and logical input reads. inputElementsRead is
// outputElementsWritten times the product of the reduced extents.
struct ReductionRunInfo {
    size_t outputElementsWritten = 0;  // output.shape().elementCount().
    size_t inputElementsRead = 0;      // Logical reads performed by the nested reduction loops.
};

// Owning reduction result. output uses a contiguous layout; callers requiring
// a specific destination layout use ReductionRequest.
struct ReductionResult {
    Tensor output;
    ReductionRunInfo runInfo;
};

ReductionRunInfo referenceReduce(const ReductionRequest& request);
ReductionResult referenceReduce(const ReductionProblem& problem);
ReductionResult referenceReduce(const ReductionProblem& problem,
                                const TensorStorageAllocator& allocator);

ReductionRunInfo referenceSum(const ReductionRequest& request);
ReductionResult referenceSum(const ReductionProblem& problem);
ReductionResult referenceSum(const ReductionProblem& problem,
                             const TensorStorageAllocator& allocator);

ReductionRunInfo referenceMaximumAbsolute(const ReductionRequest& request);
ReductionRunInfo referenceMaximumAbsolute(Tensor input, Tensor output, ScalarType accumulatorType);
ReductionResult referenceMaximumAbsolute(Tensor input, ScalarType outputType,
                                         ScalarType accumulatorType);
ReductionResult referenceMaximumAbsolute(Tensor input, ScalarType outputType,
                                         ScalarType accumulatorType,
                                         const TensorStorageAllocator& allocator);
}  // namespace roc::host_validation
