// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <complex>
#include <cstddef>
#include <optional>
#include <roc/host_validation/detail/reference_common.hpp>
#include <stdexcept>
#include <utility>

namespace roc::host_validation {
struct AxpbyProblem {
    AxpbyProblem(std::optional<TensorView> xValues, std::optional<TensorView> yValues,
                 MutableTensorView outputValues, ScalarType accumulator)
        : x(std::move(xValues)),
          y(std::move(yValues)),
          output(std::move(outputValues)),
          accumulatorType(accumulator) {}

    std::optional<TensorView> x;
    std::optional<TensorView> y;
    MutableTensorView output;
    ScalarType accumulatorType;
    std::complex<double> alpha{1.0, 0.0};
    std::complex<double> beta{1.0, 0.0};
};

struct AxpbyRunInfo {
    size_t elementsComputed = 0;
};

namespace detail {
inline void validateAxpby(const AxpbyProblem& problem) {
    if (!problem.x && !problem.y)
        throw std::invalid_argument("Reference AXPBY requires at least one input tensor.");
    if (problem.x && problem.x->shape() != problem.output.shape())
        throw std::invalid_argument("Reference AXPBY X/output shapes differ.");
    if (problem.y && problem.y->shape() != problem.output.shape())
        throw std::invalid_argument("Reference AXPBY Y/output shapes differ.");

    switch (problem.accumulatorType) {
        case ScalarType::Float32:
        case ScalarType::Float64:
        case ScalarType::ComplexFloat32:
        case ScalarType::ComplexFloat64:
            return;
        default:
            throw std::invalid_argument("Unsupported reference AXPBY accumulator type.");
    }
}

template <typename Accumulator>
AxpbyRunInfo referenceAxpbyTyped(const AxpbyProblem& problem) {
    std::optional<RuntimeTensorReader<Accumulator>> x;
    std::optional<RuntimeTensorReader<Accumulator>> y;
    if (problem.x) x.emplace(*problem.x);
    if (problem.y) y.emplace(*problem.y);
    const RuntimeTensorWriter<Accumulator> output(problem.output);
    const Accumulator alpha =
        x ? runtimeScalar<Accumulator>(problem.alpha, "alpha") : Accumulator(0);
    const Accumulator beta = y ? runtimeScalar<Accumulator>(problem.beta, "beta") : Accumulator(0);

    detail::forEachIndex(problem.output.shape(), [&](std::span<const size_t> indices, size_t) {
        Accumulator value = Accumulator(0);
        if (x) value += alpha * (*x)(indices);
        if (y) value += beta * (*y)(indices);
        output.store(indices, value);
    });
    return {.elementsComputed = problem.output.shape().elementCount()};
}
}  // namespace detail

inline AxpbyRunInfo referenceAxpby(const AxpbyProblem& problem) {
    detail::validateAxpby(problem);
    switch (problem.accumulatorType) {
        case ScalarType::Float32:
            return detail::referenceAxpbyTyped<float>(problem);
        case ScalarType::Float64:
            return detail::referenceAxpbyTyped<double>(problem);
        case ScalarType::ComplexFloat32:
            return detail::referenceAxpbyTyped<std::complex<float>>(problem);
        case ScalarType::ComplexFloat64:
            return detail::referenceAxpbyTyped<std::complex<double>>(problem);
        default:
            throw std::invalid_argument("Unsupported reference AXPBY accumulator type.");
    }
}
}  // namespace roc::host_validation
