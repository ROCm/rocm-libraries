// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <roc/host_validation/detail/reference_common.hpp>
#include <stdexcept>
#include <vector>

namespace roc::host_validation {
struct SoftmaxProblem {
    SoftmaxProblem(TensorView inputValues, MutableTensorView outputValues, size_t softmaxAxis,
                   ScalarType accumulator)
        : input(std::move(inputValues)),
          output(std::move(outputValues)),
          axis(softmaxAxis),
          accumulatorType(accumulator) {}

    TensorView input;
    MutableTensorView output;
    size_t axis;
    ScalarType accumulatorType;
};

struct SoftmaxRunInfo {
    size_t slicesComputed = 0;
    size_t elementsComputed = 0;
};

namespace detail {
inline void validateSoftmax(const SoftmaxProblem& problem) {
    if (problem.input.shape() != problem.output.shape())
        throw std::invalid_argument("Reference softmax input/output shapes differ.");
    if (problem.axis >= problem.input.shape().rank())
        throw std::out_of_range("Reference softmax axis exceeds input rank.");
    if (problem.input.shape()[problem.axis] == 0)
        throw std::invalid_argument("Reference softmax axis must be nonempty.");
    if (problem.accumulatorType != ScalarType::Float32 &&
        problem.accumulatorType != ScalarType::Float64)
        throw std::invalid_argument("Reference softmax requires a Float32 or Float64 accumulator.");
}

inline std::vector<size_t> softmaxSliceCoordinates(size_t sliceIndex, const Shape& shape,
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

template <typename Accumulator>
SoftmaxRunInfo referenceSoftmaxTyped(const SoftmaxProblem& problem) {
    const RuntimeTensorReader<Accumulator> input(problem.input);
    const RuntimeTensorWriter<Accumulator> output(problem.output);
    const size_t axisElements = problem.input.shape()[problem.axis];
    const size_t slices = problem.input.shape().elementCount() / axisElements;
    std::vector<Accumulator> exponentials(axisElements);

    for (size_t slice = 0; slice < slices; ++slice) {
        std::vector<size_t> coordinates =
            softmaxSliceCoordinates(slice, problem.input.shape(), problem.axis);
        coordinates[problem.axis] = 0;
        Accumulator maximum = input(coordinates);
        for (size_t index = 1; index < axisElements; ++index) {
            coordinates[problem.axis] = index;
            maximum = std::max(maximum, input(coordinates));
        }

        Accumulator sum = Accumulator(0);
        for (size_t index = 0; index < axisElements; ++index) {
            coordinates[problem.axis] = index;
            const Accumulator value = std::exp(input(coordinates) - maximum);
            exponentials[index] = value;
            sum += value;
        }
        for (size_t index = 0; index < axisElements; ++index) {
            coordinates[problem.axis] = index;
            output.store(coordinates, exponentials[index] / sum);
        }
    }

    return {
        .slicesComputed = slices,
        .elementsComputed = problem.input.shape().elementCount(),
    };
}
}  // namespace detail

inline SoftmaxRunInfo referenceSoftmax(const SoftmaxProblem& problem) {
    detail::validateSoftmax(problem);
    if (problem.accumulatorType == ScalarType::Float32)
        return detail::referenceSoftmaxTyped<float>(problem);
    return detail::referenceSoftmaxTyped<double>(problem);
}
}  // namespace roc::host_validation
