// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <roc/host_numerics/softmax.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

#include "reference_common.hpp"

namespace roc::host_numerics {
namespace detail {
struct SoftmaxInvocation {
    Tensor input;
    Tensor output;
    size_t axis;
    ScalarType accumulatorType;
};

inline const Shape& validateSoftmaxArguments(const Tensor& input, ScalarType outputType,
                                             size_t axis, ScalarType accumulatorType) {
    if (!isConcreteScalarType(outputType))
        throw std::invalid_argument("Reference softmax output type is invalid.");
    if (axis >= input.shape().rank())
        throw std::out_of_range("Reference softmax axis exceeds input rank.");
    if (input.shape()[axis] == 0)
        throw std::invalid_argument("Reference softmax axis must be nonempty.");
    if (accumulatorType != ScalarType::Float32 && accumulatorType != ScalarType::Float64)
        throw std::invalid_argument("Reference softmax requires a Float32 or Float64 accumulator.");
    return input.shape();
}

inline void validateSoftmaxInvocation(const SoftmaxInvocation& invocation) {
    const Shape& outputShape = validateSoftmaxArguments(
        invocation.input, invocation.output.type(), invocation.axis, invocation.accumulatorType);
    if (invocation.output.shape() != outputShape)
        throw std::invalid_argument("Reference softmax input/output shapes differ.");
    requireProvablyDistinctDestinationElementOffsets(invocation.output, "Reference softmax",
                                                     "output");
    rejectOverlappingTensorStorageUnlessIdenticallyMapped(
        invocation.output, invocation.input,
        "Reference softmax output overlaps input with a different storage mapping.");
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
void referenceSoftmaxTyped(const SoftmaxInvocation& invocation) {
    const RuntimeTensorReader<Accumulator> input(invocation.input);
    const RuntimeTensorWriter<Accumulator> output(invocation.output);
    const size_t axisElements = invocation.input.shape()[invocation.axis];
    const size_t slices = invocation.input.shape().elementCountExcluding(invocation.axis);
    std::vector<Accumulator> exponentials(axisElements);

    for (size_t slice = 0; slice < slices; ++slice) {
        std::vector<size_t> coordinates =
            softmaxSliceCoordinates(slice, invocation.input.shape(), invocation.axis);
        coordinates[invocation.axis] = 0;
        Accumulator maximum = input(coordinates);
        for (size_t index = 1; index < axisElements; ++index) {
            coordinates[invocation.axis] = index;
            maximum = std::max(maximum, input(coordinates));
        }

        Accumulator sum = Accumulator(0);
        for (size_t index = 0; index < axisElements; ++index) {
            coordinates[invocation.axis] = index;
            const Accumulator value = std::exp(input(coordinates) - maximum);
            exponentials[index] = value;
            sum += value;
        }
        for (size_t index = 0; index < axisElements; ++index) {
            coordinates[invocation.axis] = index;
            output.store(coordinates, exponentials[index] / sum);
        }
    }
}
}  // namespace detail
}  // namespace roc::host_numerics
