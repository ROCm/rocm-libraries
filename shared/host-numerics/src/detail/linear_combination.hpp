// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <complex>
#include <cstddef>
#include <optional>
#include <roc/host_numerics/linear_combination.hpp>
#include <stdexcept>
#include <utility>

#include "reference_common.hpp"

namespace roc::host_numerics {
namespace detail {
struct LinearCombinationInvocation {
    std::optional<Tensor> x;
    std::optional<Tensor> y;
    Tensor output;
    LinearCombinationOptions options;
};

inline const Shape& linearCombinationOutputShape(const std::optional<Tensor>& x,
                                                 const std::optional<Tensor>& y) {
    if (!x && !y)
        throw std::invalid_argument("Linear combination requires at least one input tensor.");
    if (x && y && x->shape() != y->shape())
        throw std::invalid_argument("Linear combination input shapes differ.");
    return x ? x->shape() : y->shape();
}

template <typename Accumulator>
inline void validateLinearCombinationScalars(const std::optional<Tensor>& x,
                                             const std::optional<Tensor>& y,
                                             const LinearCombinationOptions& options) {
    if (x) (void)runtimeScalar<Accumulator>(options.alpha, "alpha");
    if (y) (void)runtimeScalar<Accumulator>(options.beta, "beta");
}

inline const Shape& validateLinearCombinationArguments(const std::optional<Tensor>& x,
                                                       const std::optional<Tensor>& y,
                                                       ScalarType outputType,
                                                       const LinearCombinationOptions& options) {
    const Shape& outputShape = linearCombinationOutputShape(x, y);
    if (!isConcreteScalarType(outputType))
        throw std::invalid_argument("Linear combination output type is invalid.");

    switch (options.accumulatorType) {
        case ScalarType::Float32:
            validateLinearCombinationScalars<float>(x, y, options);
            break;
        case ScalarType::Float64:
            validateLinearCombinationScalars<double>(x, y, options);
            break;
        case ScalarType::ComplexFloat32:
            validateLinearCombinationScalars<std::complex<float>>(x, y, options);
            break;
        case ScalarType::ComplexFloat64:
            validateLinearCombinationScalars<std::complex<double>>(x, y, options);
            break;
        default:
            throw std::invalid_argument("Unsupported linear combination accumulator type.");
    }
    return outputShape;
}

inline void validateLinearCombinationInvocation(const LinearCombinationInvocation& invocation) {
    const Shape& outputShape = validateLinearCombinationArguments(
        invocation.x, invocation.y, invocation.output.type(), invocation.options);
    if (invocation.output.shape() != outputShape)
        throw std::invalid_argument("Linear combination input/output shapes differ.");
    requireProvablyDistinctDestinationElementOffsets(invocation.output, "Linear combination",
                                                     "output");
    if (invocation.x)
        rejectOverlappingTensorStorageUnlessIdenticallyMapped(
            invocation.output, *invocation.x,
            "Linear combination output overlaps X with a different storage mapping.");
    if (invocation.y)
        rejectOverlappingTensorStorageUnlessIdenticallyMapped(
            invocation.output, *invocation.y,
            "Linear combination output overlaps Y with a different storage mapping.");
}

template <typename Accumulator>
void linearCombinationTyped(const LinearCombinationInvocation& invocation) {
    std::optional<RuntimeTensorReader<Accumulator>> x;
    std::optional<RuntimeTensorReader<Accumulator>> y;
    if (invocation.x) x.emplace(*invocation.x);
    if (invocation.y) y.emplace(*invocation.y);
    const RuntimeTensorWriter<Accumulator> output(invocation.output);
    const Accumulator alpha =
        x ? runtimeScalar<Accumulator>(invocation.options.alpha, "alpha") : Accumulator(0);
    const Accumulator beta =
        y ? runtimeScalar<Accumulator>(invocation.options.beta, "beta") : Accumulator(0);

    detail::forEachIndex(invocation.output.shape(), [&](std::span<const size_t> indices, size_t) {
        Accumulator value = Accumulator(0);
        if (x) value += alpha * (*x)(indices);
        if (y) value += beta * (*y)(indices);
        output.store(indices, value);
    });
}
}  // namespace detail
}  // namespace roc::host_numerics
