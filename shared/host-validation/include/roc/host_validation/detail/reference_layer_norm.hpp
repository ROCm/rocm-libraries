// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstddef>
#include <optional>
#include <roc/host_validation/detail/reference_common.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace roc::host_validation {
struct LayerNormProblem {
    LayerNormProblem(TensorView inputValues, MutableTensorView outputValues, size_t normalizedAxis,
                     ScalarType accumulator)
        : input(std::move(inputValues)),
          output(std::move(outputValues)),
          axis(normalizedAxis),
          accumulatorType(accumulator) {}

    TensorView input;
    MutableTensorView output;
    std::optional<MutableTensorView> mean;
    std::optional<MutableTensorView> inverseVariance;
    std::optional<TensorView> gamma;
    std::optional<TensorView> beta;
    size_t axis;
    ScalarType accumulatorType;
    double epsilon = 1e-5;
};

struct LayerNormRunInfo {
    size_t slicesComputed = 0;
    size_t elementsComputed = 0;
};

namespace detail {
inline Shape layerNormStatisticsShape(const Shape& inputShape, size_t axis) {
    std::vector<size_t> dimensions;
    dimensions.reserve(inputShape.rank() - 1);
    for (size_t dimension = 0; dimension < inputShape.rank(); ++dimension) {
        if (dimension != axis) dimensions.push_back(inputShape[dimension]);
    }
    return Shape(std::move(dimensions));
}

inline void validateLayerNorm(const LayerNormProblem& problem) {
    if (problem.input.shape() != problem.output.shape())
        throw std::invalid_argument("Reference LayerNorm input/output shapes differ.");
    if (problem.axis >= problem.input.shape().rank())
        throw std::out_of_range("Reference LayerNorm axis exceeds input rank.");
    if (problem.input.shape()[problem.axis] == 0)
        throw std::invalid_argument("Reference LayerNorm axis must be nonempty.");
    if (problem.epsilon < 0)
        throw std::invalid_argument("Reference LayerNorm epsilon must be nonnegative.");
    if (problem.accumulatorType != ScalarType::Float32 &&
        problem.accumulatorType != ScalarType::Float64)
        throw std::invalid_argument(
            "Reference LayerNorm requires a Float32 or Float64 accumulator.");

    const size_t axisElements = problem.input.shape()[problem.axis];
    if (problem.gamma) validateRuntimeVector(*problem.gamma, axisElements, "LayerNorm", "gamma");
    if (problem.beta) validateRuntimeVector(*problem.beta, axisElements, "LayerNorm", "beta");

    const Shape statisticsShape = layerNormStatisticsShape(problem.input.shape(), problem.axis);
    if (problem.mean && problem.mean->shape() != statisticsShape)
        throw std::invalid_argument("Reference LayerNorm mean shape mismatch.");
    if (problem.inverseVariance && problem.inverseVariance->shape() != statisticsShape)
        throw std::invalid_argument("Reference LayerNorm inverse-variance shape mismatch.");
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
LayerNormRunInfo referenceLayerNormTyped(const LayerNormProblem& problem) {
    const RuntimeTensorReader<Accumulator> input(problem.input);
    const RuntimeTensorWriter<Accumulator> output(problem.output);
    std::optional<RuntimeTensorWriter<Accumulator>> mean;
    std::optional<RuntimeTensorWriter<Accumulator>> inverseVariance;
    std::optional<RuntimeVectorReader<Accumulator>> gamma;
    std::optional<RuntimeVectorReader<Accumulator>> beta;
    if (problem.mean) mean.emplace(*problem.mean);
    if (problem.inverseVariance) inverseVariance.emplace(*problem.inverseVariance);
    if (problem.gamma) gamma.emplace(*problem.gamma);
    if (problem.beta) beta.emplace(*problem.beta);

    const size_t axisElements = problem.input.shape()[problem.axis];
    const size_t slices = problem.input.shape().elementCount() / axisElements;
    const Accumulator epsilon = static_cast<Accumulator>(problem.epsilon);

    for (size_t slice = 0; slice < slices; ++slice) {
        std::vector<size_t> coordinates =
            layerNormSliceCoordinates(slice, problem.input.shape(), problem.axis);
        Accumulator average = Accumulator(0);
        Accumulator secondMoment = Accumulator(0);
        for (size_t index = 0; index < axisElements; ++index) {
            coordinates[problem.axis] = index;
            const Accumulator value = input(coordinates);
            const Accumulator delta = value - average;
            average += delta / static_cast<Accumulator>(index + 1);
            const Accumulator deltaAfterUpdate = value - average;
            secondMoment += delta * deltaAfterUpdate;
        }
        const Accumulator inverse =
            Accumulator(1) /
            std::sqrt(secondMoment / static_cast<Accumulator>(axisElements) + epsilon);

        if (mean || inverseVariance) {
            const std::vector<size_t> statisticsCoordinates =
                layerNormStatisticsCoordinates(coordinates, problem.axis);
            if (mean) mean->store(statisticsCoordinates, average);
            if (inverseVariance) inverseVariance->store(statisticsCoordinates, inverse);
        }

        for (size_t index = 0; index < axisElements; ++index) {
            coordinates[problem.axis] = index;
            Accumulator value = (input(coordinates) - average) * inverse;
            if (gamma) value *= (*gamma)[index];
            if (beta) value += (*beta)[index];
            output.store(coordinates, value);
        }
    }

    return {
        .slicesComputed = slices,
        .elementsComputed = problem.input.shape().elementCount(),
    };
}
}  // namespace detail

inline LayerNormRunInfo referenceLayerNorm(const LayerNormProblem& problem) {
    detail::validateLayerNorm(problem);
    if (problem.accumulatorType == ScalarType::Float32)
        return detail::referenceLayerNormTyped<float>(problem);
    return detail::referenceLayerNormTyped<double>(problem);
}
}  // namespace roc::host_validation
