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
inline const Shape& validateSoftmaxProblem(const SoftmaxProblem& problem) {
    if (!isConcreteScalarType(problem.outputType))
        throw std::invalid_argument("Reference softmax output type is invalid.");
    if (problem.axis >= problem.input.shape().rank())
        throw std::out_of_range("Reference softmax axis exceeds input rank.");
    if (problem.input.shape()[problem.axis] == 0)
        throw std::invalid_argument("Reference softmax axis must be nonempty.");
    if (problem.accumulatorType != ScalarType::Float32 &&
        problem.accumulatorType != ScalarType::Float64)
        throw std::invalid_argument("Reference softmax requires a Float32 or Float64 accumulator.");
    return problem.input.shape();
}

inline void validateSoftmaxRequest(const SoftmaxRequest& request) {
    const Shape& outputShape = validateSoftmaxProblem(request);
    if (request.output.shape() != outputShape)
        throw std::invalid_argument("Reference softmax input/output shapes differ.");
    if (request.output.type() != request.outputType)
        throw std::invalid_argument("Reference softmax output type differs from the problem.");
    requireProvablyDistinctDestinationElementOffsets(request.output, "Reference softmax", "output");
    rejectOverlappingTensorStorageUnlessIdenticallyMapped(
        request.output, request.input,
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
SoftmaxRunInfo referenceSoftmaxTyped(const SoftmaxRequest& request) {
    const RuntimeTensorReader<Accumulator> input(request.input);
    const RuntimeTensorWriter<Accumulator> output(request.output);
    const size_t axisElements = request.input.shape()[request.axis];
    const size_t slices = request.input.shape().elementCountExcluding(request.axis);
    std::vector<Accumulator> exponentials(axisElements);

    for (size_t slice = 0; slice < slices; ++slice) {
        std::vector<size_t> coordinates =
            softmaxSliceCoordinates(slice, request.input.shape(), request.axis);
        coordinates[request.axis] = 0;
        Accumulator maximum = input(coordinates);
        for (size_t index = 1; index < axisElements; ++index) {
            coordinates[request.axis] = index;
            maximum = std::max(maximum, input(coordinates));
        }

        Accumulator sum = Accumulator(0);
        for (size_t index = 0; index < axisElements; ++index) {
            coordinates[request.axis] = index;
            const Accumulator value = std::exp(input(coordinates) - maximum);
            exponentials[index] = value;
            sum += value;
        }
        for (size_t index = 0; index < axisElements; ++index) {
            coordinates[request.axis] = index;
            output.store(coordinates, exponentials[index] / sum);
        }
    }

    return {
        .slicesProcessed = slices,
        .outputElementsWritten = request.output.shape().elementCount(),
    };
}
}  // namespace detail
}  // namespace roc::host_numerics
