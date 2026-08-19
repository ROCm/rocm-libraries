// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <complex>
#include <cstddef>
#include <optional>
#include <roc/host_validation/axpby.hpp>
#include <stdexcept>
#include <utility>

#include "reference_common.hpp"

namespace roc::host_validation {
namespace detail {
inline const Shape& axpbyOutputShape(const AxpbyProblem& problem) {
    if (!problem.x && !problem.y)
        throw std::invalid_argument("Reference AXPBY requires at least one input tensor.");
    if (problem.x && problem.y && problem.x->shape() != problem.y->shape())
        throw std::invalid_argument("Reference AXPBY input shapes differ.");
    return problem.x ? problem.x->shape() : problem.y->shape();
}

template <typename Accumulator>
inline void validateAxpbyScalars(const AxpbyProblem& problem) {
    if (problem.x) (void)runtimeScalar<Accumulator>(problem.alpha, "alpha");
    if (problem.y) (void)runtimeScalar<Accumulator>(problem.beta, "beta");
}

inline const Shape& validateAxpbyProblem(const AxpbyProblem& problem) {
    const Shape& outputShape = axpbyOutputShape(problem);
    if (!isConcreteScalarType(problem.outputType))
        throw std::invalid_argument("Reference AXPBY output type is invalid.");

    switch (problem.accumulatorType) {
        case ScalarType::Float32:
            validateAxpbyScalars<float>(problem);
            break;
        case ScalarType::Float64:
            validateAxpbyScalars<double>(problem);
            break;
        case ScalarType::ComplexFloat32:
            validateAxpbyScalars<std::complex<float>>(problem);
            break;
        case ScalarType::ComplexFloat64:
            validateAxpbyScalars<std::complex<double>>(problem);
            break;
        default:
            throw std::invalid_argument("Unsupported reference AXPBY accumulator type.");
    }
    return outputShape;
}

inline void validateAxpbyRequest(const AxpbyRequest& request) {
    const Shape& outputShape = validateAxpbyProblem(request);
    if (request.output.shape() != outputShape)
        throw std::invalid_argument("Reference AXPBY input/output shapes differ.");
    if (request.output.type() != request.outputType)
        throw std::invalid_argument("Reference AXPBY output type differs from the problem.");
}

template <typename Accumulator>
AxpbyRunInfo referenceAxpbyTyped(const AxpbyRequest& request) {
    std::optional<RuntimeTensorReader<Accumulator>> x;
    std::optional<RuntimeTensorReader<Accumulator>> y;
    if (request.x) x.emplace(*request.x);
    if (request.y) y.emplace(*request.y);
    const RuntimeTensorWriter<Accumulator> output(request.output);
    const Accumulator alpha =
        x ? runtimeScalar<Accumulator>(request.alpha, "alpha") : Accumulator(0);
    const Accumulator beta = y ? runtimeScalar<Accumulator>(request.beta, "beta") : Accumulator(0);

    detail::forEachIndex(request.output.shape(), [&](std::span<const size_t> indices, size_t) {
        Accumulator value = Accumulator(0);
        if (x) value += alpha * (*x)(indices);
        if (y) value += beta * (*y)(indices);
        output.store(indices, value);
    });
    return {.outputElementsWritten = request.output.shape().elementCount()};
}
}  // namespace detail
}  // namespace roc::host_validation
