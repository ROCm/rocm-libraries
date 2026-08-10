// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <roc/host_validation/detail/reference_common.hpp>
#include <roc/host_validation/reduction.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace roc::host_validation {
namespace detail {
struct ReductionPlan {
    std::vector<bool> reducedDimensions;
    Shape reductionShape;
};

inline ReductionPlan validateReduction(const ReductionProblem& problem) {
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
    if (problem.output.shape() != Shape(expectedOutputDimensions))
        throw std::invalid_argument("Reference reduction output shape mismatch.");

    const bool complexAccumulator = problem.accumulatorType == ScalarType::ComplexFloat32 ||
                                    problem.accumulatorType == ScalarType::ComplexFloat64;
    if (problem.operation == ReductionOperation::MaximumAbsolute &&
        (complexAccumulator || isComplexScalarType(problem.input.type()) ||
         isComplexScalarType(problem.output.type())))
        throw std::invalid_argument("Maximum-absolute reduction currently requires real tensors.");
    if (!complexAccumulator && isComplexScalarType(problem.input.type()))
        throw std::invalid_argument(
            "Real reference reduction cannot consume a complex input tensor.");
    if (complexAccumulator != isComplexScalarType(problem.output.type()))
        throw std::invalid_argument("Reference reduction accumulator/output complexity mismatch.");

    if (problem.operation == ReductionOperation::Sum) {
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
    } else {
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
    }

    return {
        .reducedDimensions = std::move(reducedDimensions),
        .reductionShape = Shape(std::move(reductionDimensions)),
    };
}

inline void coordinatesFromLinear(size_t linear, const Shape& shape,
                                  std::vector<size_t>& coordinates) {
    coordinates.resize(shape.rank());
    for (size_t dimension = shape.rank(); dimension > 0; --dimension) {
        const size_t index = dimension - 1;
        coordinates[index] = linear % shape[index];
        linear /= shape[index];
    }
}

template <typename Accumulator>
ReductionRunInfo referenceReductionTyped(const ReductionProblem& problem,
                                         const ReductionPlan& plan) {
    const RuntimeTensorReader<Accumulator> input(problem.input);
    const RuntimeTensorWriter<Accumulator> output(problem.output);

    const size_t outputElements = problem.output.shape().elementCount();
    const size_t reductionElements = plan.reductionShape.elementCount();
    if (reductionElements != 0 &&
        outputElements > std::numeric_limits<size_t>::max() / reductionElements)
        throw std::overflow_error("Reference reduction input-read count overflow.");

    std::vector<size_t> outputCoordinates;
    std::vector<size_t> reductionCoordinates;
    std::vector<size_t> inputCoordinates(problem.input.shape().rank(), 0);
    for (size_t outputLinear = 0; outputLinear < outputElements; ++outputLinear) {
        coordinatesFromLinear(outputLinear, problem.output.shape(), outputCoordinates);
        size_t outputDimension = 0;
        for (size_t inputDimension = 0; inputDimension < inputCoordinates.size();
             ++inputDimension) {
            if (!plan.reducedDimensions[inputDimension])
                inputCoordinates[inputDimension] = outputCoordinates[outputDimension++];
        }

        Accumulator result{};
        for (size_t reductionLinear = 0; reductionLinear < reductionElements; ++reductionLinear) {
            coordinatesFromLinear(reductionLinear, plan.reductionShape, reductionCoordinates);
            for (size_t axisIndex = 0; axisIndex < problem.axes.size(); ++axisIndex)
                inputCoordinates[problem.axes[axisIndex]] = reductionCoordinates[axisIndex];
            const Accumulator value = input(std::span<const size_t>(inputCoordinates));
            if constexpr (IsComplex<Accumulator>::value) {
                result += value;
            } else {
                if (problem.operation == ReductionOperation::Sum) {
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
        .outputElementsComputed = outputElements,
        .inputElementsRead = outputElements * reductionElements,
    };
}
}  // namespace detail
}  // namespace roc::host_validation
