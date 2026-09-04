// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <roc/host_numerics/layer_norm.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

#include "reference_common.hpp"

namespace roc::host_numerics {
namespace detail {
struct LayerNormPlan {
    Shape statisticsShape;
    size_t axisElements = 0;
    size_t slices = 0;
};

struct LayerNormInvocation {
    Tensor input;
    LayerNormOutputs outputs;
    LayerNormOptions options;
};

inline Shape layerNormStatisticsShape(const Shape& inputShape, size_t axis) {
    std::vector<size_t> dimensions;
    dimensions.reserve(inputShape.rank() - 1);
    for (size_t dimension = 0; dimension < inputShape.rank(); ++dimension) {
        if (dimension != axis) dimensions.push_back(inputShape[dimension]);
    }
    return Shape(std::move(dimensions));
}

inline LayerNormPlan validateLayerNormArguments(const Tensor& input,
                                                const LayerNormOutputTypes& outputTypes,
                                                const LayerNormOptions& options) {
    if (!isConcreteScalarType(outputTypes.output))
        throw std::invalid_argument("Reference LayerNorm output type is invalid.");
    if (outputTypes.mean && !isConcreteScalarType(*outputTypes.mean))
        throw std::invalid_argument("Reference LayerNorm mean type is invalid.");
    if (outputTypes.inverseVariance && !isConcreteScalarType(*outputTypes.inverseVariance))
        throw std::invalid_argument("Reference LayerNorm inverse-variance type is invalid.");
    if (options.axis >= input.shape().rank())
        throw std::out_of_range("Reference LayerNorm axis exceeds input rank.");
    if (input.shape()[options.axis] == 0)
        throw std::invalid_argument("Reference LayerNorm axis must be nonempty.");
    if (!std::isfinite(options.epsilon) || options.epsilon < 0)
        throw std::invalid_argument("Reference LayerNorm epsilon must be finite and nonnegative.");
    if (options.accumulatorType != ScalarType::Float32 &&
        options.accumulatorType != ScalarType::Float64)
        throw std::invalid_argument(
            "Reference LayerNorm requires a Float32 or Float64 accumulator.");

    const size_t axisElements = input.shape()[options.axis];
    if (options.gamma) validateRuntimeVector(*options.gamma, axisElements, "LayerNorm", "gamma");
    if (options.beta) validateRuntimeVector(*options.beta, axisElements, "LayerNorm", "beta");

    return {
        .statisticsShape = layerNormStatisticsShape(input.shape(), options.axis),
        .axisElements = axisElements,
        .slices = input.shape().elementCountExcluding(options.axis),
    };
}

inline LayerNormPlan validateLayerNormInvocation(const LayerNormInvocation& invocation) {
    const LayerNormOutputTypes outputTypes{
        .output = invocation.outputs.output.type(),
        .mean =
            invocation.outputs.mean ? std::optional(invocation.outputs.mean->type()) : std::nullopt,
        .inverseVariance = invocation.outputs.inverseVariance
                               ? std::optional(invocation.outputs.inverseVariance->type())
                               : std::nullopt,
    };
    LayerNormPlan plan =
        validateLayerNormArguments(invocation.input, outputTypes, invocation.options);
    if (invocation.outputs.output.shape() != invocation.input.shape())
        throw std::invalid_argument("Reference LayerNorm input/output shapes differ.");
    if (invocation.outputs.mean && invocation.outputs.mean->shape() != plan.statisticsShape)
        throw std::invalid_argument("Reference LayerNorm mean shape mismatch.");
    if (invocation.outputs.inverseVariance &&
        invocation.outputs.inverseVariance->shape() != plan.statisticsShape)
        throw std::invalid_argument("Reference LayerNorm inverse-variance shape mismatch.");

    requireProvablyDistinctDestinationElementOffsets(invocation.outputs.output,
                                                     "Reference LayerNorm", "output");
    rejectOverlappingTensorStorageUnlessIdenticallyMapped(
        invocation.outputs.output, invocation.input,
        "Reference LayerNorm output overlaps input with a different storage mapping.");
    if (invocation.options.gamma)
        rejectOverlappingTensorStorage(invocation.outputs.output, *invocation.options.gamma,
                                       "Reference LayerNorm output overlaps gamma.");
    if (invocation.options.beta)
        rejectOverlappingTensorStorage(invocation.outputs.output, *invocation.options.beta,
                                       "Reference LayerNorm output overlaps beta.");

    const std::array<const Tensor*, 2> statistics{
        invocation.outputs.mean ? &*invocation.outputs.mean : nullptr,
        invocation.outputs.inverseVariance ? &*invocation.outputs.inverseVariance : nullptr,
    };
    for (const Tensor* statistic : statistics) {
        if (!statistic) continue;
        requireProvablyDistinctDestinationElementOffsets(*statistic, "Reference LayerNorm",
                                                         "statistic output");
        rejectOverlappingTensorStorage(*statistic, invocation.input,
                                       "Reference LayerNorm statistic output overlaps input.");
        if (invocation.options.gamma)
            rejectOverlappingTensorStorage(*statistic, *invocation.options.gamma,
                                           "Reference LayerNorm statistic output overlaps gamma.");
        if (invocation.options.beta)
            rejectOverlappingTensorStorage(*statistic, *invocation.options.beta,
                                           "Reference LayerNorm statistic output overlaps beta.");
        rejectOverlappingTensorStorage(*statistic, invocation.outputs.output,
                                       "Reference LayerNorm statistic output overlaps output.");
    }
    if (invocation.outputs.mean && invocation.outputs.inverseVariance)
        rejectOverlappingTensorStorage(*invocation.outputs.mean,
                                       *invocation.outputs.inverseVariance,
                                       "Reference LayerNorm statistic outputs overlap.");
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
void referenceLayerNormTyped(const LayerNormInvocation& invocation, const LayerNormPlan& plan) {
    const RuntimeTensorReader<Accumulator> input(invocation.input);
    const RuntimeTensorWriter<Accumulator> output(invocation.outputs.output);
    std::optional<RuntimeTensorWriter<Accumulator>> mean;
    std::optional<RuntimeTensorWriter<Accumulator>> inverseVariance;
    std::optional<RuntimeVectorReader<Accumulator>> gamma;
    std::optional<RuntimeVectorReader<Accumulator>> beta;
    if (invocation.outputs.mean) mean.emplace(*invocation.outputs.mean);
    if (invocation.outputs.inverseVariance)
        inverseVariance.emplace(*invocation.outputs.inverseVariance);
    if (invocation.options.gamma) gamma.emplace(*invocation.options.gamma);
    if (invocation.options.beta) beta.emplace(*invocation.options.beta);

    const Accumulator epsilon = static_cast<Accumulator>(invocation.options.epsilon);

    for (size_t slice = 0; slice < plan.slices; ++slice) {
        std::vector<size_t> coordinates =
            layerNormSliceCoordinates(slice, invocation.input.shape(), invocation.options.axis);
        Accumulator average = Accumulator(0);
        Accumulator secondMoment = Accumulator(0);
        for (size_t index = 0; index < plan.axisElements; ++index) {
            coordinates[invocation.options.axis] = index;
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
                layerNormStatisticsCoordinates(coordinates, invocation.options.axis);
            if (mean) mean->store(statisticsCoordinates, average);
            if (inverseVariance) inverseVariance->store(statisticsCoordinates, inverse);
        }

        for (size_t index = 0; index < plan.axisElements; ++index) {
            coordinates[invocation.options.axis] = index;
            Accumulator value = (input(coordinates) - average) * inverse;
            if (gamma) value *= (*gamma)[index];
            if (beta) value += (*beta)[index];
            output.store(coordinates, value);
        }
    }
}
}  // namespace detail
}  // namespace roc::host_numerics
