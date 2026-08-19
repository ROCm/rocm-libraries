// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstddef>
#include <optional>
#include <roc/host_validation/layer_norm.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

#include "reference_common.hpp"

namespace roc::host_validation {
namespace detail {
struct LayerNormPlan {
    Shape statisticsShape;
    size_t axisElements = 0;
    size_t slices = 0;
};

inline Shape layerNormStatisticsShape(const Shape& inputShape, size_t axis) {
    std::vector<size_t> dimensions;
    dimensions.reserve(inputShape.rank() - 1);
    for (size_t dimension = 0; dimension < inputShape.rank(); ++dimension) {
        if (dimension != axis) dimensions.push_back(inputShape[dimension]);
    }
    return Shape(std::move(dimensions));
}

inline LayerNormPlan validateLayerNormProblem(const LayerNormProblem& problem) {
    if (!isConcreteScalarType(problem.outputType))
        throw std::invalid_argument("Reference LayerNorm output type is invalid.");
    if (problem.meanType && !isConcreteScalarType(*problem.meanType))
        throw std::invalid_argument("Reference LayerNorm mean type is invalid.");
    if (problem.inverseVarianceType && !isConcreteScalarType(*problem.inverseVarianceType))
        throw std::invalid_argument("Reference LayerNorm inverse-variance type is invalid.");
    if (problem.axis >= problem.input.shape().rank())
        throw std::out_of_range("Reference LayerNorm axis exceeds input rank.");
    if (problem.input.shape()[problem.axis] == 0)
        throw std::invalid_argument("Reference LayerNorm axis must be nonempty.");
    if (!std::isfinite(problem.epsilon) || problem.epsilon < 0)
        throw std::invalid_argument("Reference LayerNorm epsilon must be finite and nonnegative.");
    if (problem.accumulatorType != ScalarType::Float32 &&
        problem.accumulatorType != ScalarType::Float64)
        throw std::invalid_argument(
            "Reference LayerNorm requires a Float32 or Float64 accumulator.");

    const size_t axisElements = problem.input.shape()[problem.axis];
    if (problem.gamma) validateRuntimeVector(*problem.gamma, axisElements, "LayerNorm", "gamma");
    if (problem.beta) validateRuntimeVector(*problem.beta, axisElements, "LayerNorm", "beta");

    return {
        .statisticsShape = layerNormStatisticsShape(problem.input.shape(), problem.axis),
        .axisElements = axisElements,
        .slices = problem.input.shape().elementCountExcluding(problem.axis),
    };
}

inline LayerNormPlan validateLayerNormRequest(const LayerNormRequest& request) {
    LayerNormPlan plan = validateLayerNormProblem(request);
    if (request.output.shape() != request.input.shape())
        throw std::invalid_argument("Reference LayerNorm input/output shapes differ.");
    if (request.output.type() != request.outputType)
        throw std::invalid_argument("Reference LayerNorm output type differs from the problem.");
    if (request.mean.has_value() != request.meanType.has_value())
        throw std::invalid_argument(
            "Reference LayerNorm mean destination does not match the problem.");
    if (request.inverseVariance.has_value() != request.inverseVarianceType.has_value())
        throw std::invalid_argument(
            "Reference LayerNorm inverse-variance destination does not match the problem.");
    if (request.mean && request.mean->shape() != plan.statisticsShape)
        throw std::invalid_argument("Reference LayerNorm mean shape mismatch.");
    if (request.mean && request.mean->type() != *request.meanType)
        throw std::invalid_argument("Reference LayerNorm mean type differs from the problem.");
    if (request.inverseVariance && request.inverseVariance->shape() != plan.statisticsShape)
        throw std::invalid_argument("Reference LayerNorm inverse-variance shape mismatch.");
    if (request.inverseVariance && request.inverseVariance->type() != *request.inverseVarianceType)
        throw std::invalid_argument(
            "Reference LayerNorm inverse-variance type differs from the problem.");
    return plan;
}

inline std::vector<size_t> layerNormSliceCoordinates(size_t sliceIndex, const Shape& shape,
                                                     size_t axis) {
    std::vector<size_t> coordinates(shape.rank(), 0);
    for (size_t dimension = shape.rank(); dimension > 0; --dimension) {
        const size_t index = dimension - 1;
        if (index == axis) continue;
        coordinates[index] = sliceIndex % shape[index];
        sliceIndex /= shape[index];
    }
    return coordinates;
}

inline std::vector<size_t> layerNormStatisticsCoordinates(std::span<const size_t> inputCoordinates,
                                                          size_t axis) {
    std::vector<size_t> coordinates;
    coordinates.reserve(inputCoordinates.size() - 1);
    for (size_t dimension = 0; dimension < inputCoordinates.size(); ++dimension) {
        if (dimension != axis) coordinates.push_back(inputCoordinates[dimension]);
    }
    return coordinates;
}

template <typename Accumulator>
LayerNormRunInfo referenceLayerNormTyped(const LayerNormRequest& request,
                                         const LayerNormPlan& plan) {
    const RuntimeTensorReader<Accumulator> input(request.input);
    const RuntimeTensorWriter<Accumulator> output(request.output);
    std::optional<RuntimeTensorWriter<Accumulator>> mean;
    std::optional<RuntimeTensorWriter<Accumulator>> inverseVariance;
    std::optional<RuntimeVectorReader<Accumulator>> gamma;
    std::optional<RuntimeVectorReader<Accumulator>> beta;
    if (request.mean) mean.emplace(*request.mean);
    if (request.inverseVariance) inverseVariance.emplace(*request.inverseVariance);
    if (request.gamma) gamma.emplace(*request.gamma);
    if (request.beta) beta.emplace(*request.beta);

    const Accumulator epsilon = static_cast<Accumulator>(request.epsilon);

    for (size_t slice = 0; slice < plan.slices; ++slice) {
        std::vector<size_t> coordinates =
            layerNormSliceCoordinates(slice, request.input.shape(), request.axis);
        Accumulator average = Accumulator(0);
        Accumulator secondMoment = Accumulator(0);
        for (size_t index = 0; index < plan.axisElements; ++index) {
            coordinates[request.axis] = index;
            const Accumulator value = input(coordinates);
            const Accumulator delta = value - average;
            average += delta / static_cast<Accumulator>(index + 1);
            const Accumulator deltaAfterUpdate = value - average;
            secondMoment += delta * deltaAfterUpdate;
        }
        const Accumulator inverse =
            Accumulator(1) /
            std::sqrt(secondMoment / static_cast<Accumulator>(plan.axisElements) + epsilon);

        if (mean || inverseVariance) {
            const std::vector<size_t> statisticsCoordinates =
                layerNormStatisticsCoordinates(coordinates, request.axis);
            if (mean) mean->store(statisticsCoordinates, average);
            if (inverseVariance) inverseVariance->store(statisticsCoordinates, inverse);
        }

        for (size_t index = 0; index < plan.axisElements; ++index) {
            coordinates[request.axis] = index;
            Accumulator value = (input(coordinates) - average) * inverse;
            if (gamma) value *= (*gamma)[index];
            if (beta) value += (*beta)[index];
            output.store(coordinates, value);
        }
    }

    return {
        .slicesProcessed = plan.slices,
        .outputElementsWritten = request.output.shape().elementCount(),
        .meanElementsWritten = request.mean ? plan.slices : 0,
        .inverseVarianceElementsWritten = request.inverseVariance ? plan.slices : 0,
    };
}
}  // namespace detail
}  // namespace roc::host_validation
