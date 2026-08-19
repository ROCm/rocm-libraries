// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <roc/host_validation/reduction.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

#include "reference_common.hpp"

namespace roc::host_validation {
namespace detail {
struct ReductionPlan {
    std::vector<bool> reducedDimensions;
    Shape outputShape;
    Shape reductionShape;
    size_t outputElements = 0;
    size_t reductionElements = 0;
};

inline ReductionPlan validateReductionProblem(const ReductionProblem& problem) {
    if (!isConcreteScalarType(problem.outputType))
        throw std::invalid_argument("Reference reduction output type is invalid.");

    const size_t inputRank = problem.input.shape().rank();
    std::vector<bool> reducedDimensions(inputRank, false);
    std::vector<size_t> reductionDimensions;
    reductionDimensions.reserve(problem.axes.size());

    for (const size_t axis : problem.axes) {
        if (axis >= inputRank)
            throw std::out_of_range("Reference reduction axis exceeds input rank.");
        if (reducedDimensions[axis])
            throw std::invalid_argument("Reference reduction axes must be unique.");
        reducedDimensions[axis] = true;
        reductionDimensions.push_back(problem.input.shape()[axis]);
    }

    std::vector<size_t> expectedOutputDimensions;
    expectedOutputDimensions.reserve(inputRank - problem.axes.size());
    for (size_t dimension = 0; dimension < inputRank; ++dimension) {
        if (!reducedDimensions[dimension])
            expectedOutputDimensions.push_back(problem.input.shape()[dimension]);
    }
    Shape outputShape(std::move(expectedOutputDimensions));
    Shape reductionShape(std::move(reductionDimensions));

    const bool complexAccumulator = problem.accumulatorType == ScalarType::ComplexFloat32 ||
                                    problem.accumulatorType == ScalarType::ComplexFloat64;
    if (problem.operation == ReductionOperation::MaximumAbsolute &&
        (complexAccumulator || isComplexScalarType(problem.input.type()) ||
         isComplexScalarType(problem.outputType)))
        throw std::invalid_argument("Maximum-absolute reduction currently requires real tensors.");
    if (!complexAccumulator && isComplexScalarType(problem.input.type()))
        throw std::invalid_argument(
            "Real reference reduction cannot consume a complex input tensor.");
    if (complexAccumulator != isComplexScalarType(problem.outputType))
        throw std::invalid_argument("Reference reduction accumulator/output complexity mismatch.");

    switch (problem.operation) {
        case ReductionOperation::Sum:
            switch (problem.accumulatorType) {
                case ScalarType::Float32:
                case ScalarType::Float64:
                case ScalarType::Int32:
                case ScalarType::ComplexFloat32:
                case ScalarType::ComplexFloat64:
                    break;
                default:
                    throw std::invalid_argument(
                        "Reference sum supports F32, F64, I32, C64, and C128 accumulators.");
            }
            break;
        case ReductionOperation::MaximumAbsolute:
            switch (problem.accumulatorType) {
                case ScalarType::Float16:
                case ScalarType::BFloat16:
                case ScalarType::Float32:
                case ScalarType::Float64:
                    break;
                default:
                    throw std::invalid_argument(
                        "Maximum-absolute reduction supports F16, BF16, F32, and F64 "
                        "accumulators.");
            }
            break;
        default:
            throw std::invalid_argument("Reference reduction operation is invalid.");
    }

    const size_t outputElements = outputShape.elementCount();
    const size_t reductionElements = reductionShape.elementCount();
    if (reductionElements != 0 &&
        outputElements > std::numeric_limits<size_t>::max() / reductionElements)
        throw std::overflow_error("Reference reduction input-read count overflow.");

    return {
        .reducedDimensions = std::move(reducedDimensions),
        .outputShape = std::move(outputShape),
        .reductionShape = std::move(reductionShape),
        .outputElements = outputElements,
        .reductionElements = reductionElements,
    };
}

inline ReductionPlan validateReductionRequest(const ReductionRequest& request) {
    ReductionPlan plan = validateReductionProblem(request);
    if (request.output.shape() != plan.outputShape)
        throw std::invalid_argument("Reference reduction output shape mismatch.");
    if (request.output.type() != request.outputType)
        throw std::invalid_argument("Reference reduction output type differs from the problem.");
    return plan;
}

template <typename Accumulator>
ReductionRunInfo referenceReductionTyped(const ReductionRequest& request,
                                         const ReductionPlan& plan) {
    const RuntimeTensorReader<Accumulator> input(request.input);
    const RuntimeTensorWriter<Accumulator> output(request.output);

    std::vector<size_t> outputCoordinates(plan.outputShape.rank());
    std::vector<size_t> reductionCoordinates(plan.reductionShape.rank());
    std::vector<size_t> inputCoordinates(request.input.shape().rank(), 0);
    for (size_t outputLinear = 0; outputLinear < plan.outputElements; ++outputLinear) {
        plan.outputShape.coordinates(outputLinear, IndexOrder::LastDimensionFastest,
                                     outputCoordinates);
        size_t outputDimension = 0;
        for (size_t inputDimension = 0; inputDimension < inputCoordinates.size();
             ++inputDimension) {
            if (!plan.reducedDimensions[inputDimension])
                inputCoordinates[inputDimension] = outputCoordinates[outputDimension++];
        }

        Accumulator result{};
        for (size_t reductionLinear = 0; reductionLinear < plan.reductionElements;
             ++reductionLinear) {
            plan.reductionShape.coordinates(reductionLinear, IndexOrder::LastDimensionFastest,
                                            reductionCoordinates);
            for (size_t axisIndex = 0; axisIndex < request.axes.size(); ++axisIndex)
                inputCoordinates[request.axes[axisIndex]] = reductionCoordinates[axisIndex];
            const Accumulator value = input(std::span<const size_t>(inputCoordinates));
            if constexpr (IsComplex<Accumulator>::value) {
                result += value;
            } else {
                if (request.operation == ReductionOperation::Sum) {
                    result += value;
                } else {
                    const Accumulator magnitude = static_cast<Accumulator>(std::abs(value));
                    if constexpr (std::is_floating_point_v<Accumulator>) {
                        if (std::isnan(magnitude)) continue;
                    }
                    result = std::max(result, magnitude);
                }
            }
        }
        output.store(std::span<const size_t>(outputCoordinates), result);
    }

    return {
        .outputElementsWritten = plan.outputElements,
        .inputElementsRead = plan.outputElements * plan.reductionElements,
    };
}
}  // namespace detail
}  // namespace roc::host_validation
