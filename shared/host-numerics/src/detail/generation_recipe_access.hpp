// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <roc/host_numerics/generation.hpp>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "generation_primitives.hpp"
#include "generation_values.hpp"

namespace roc::host_numerics::detail {
struct GenerationRecipeAccess {
    using Component = GenerationRecipe::Component;

    static uint64_t rawGenerationValue(const Component& component, uint64_t seed, uint64_t domain,
                                       std::span<const size_t> indices, const Shape& shape,
                                       size_t logicalIndex, ScalarType destinationType) {
        return std::visit(
            [&](const auto& pattern) -> uint64_t {
                using Pattern = std::remove_cvref_t<decltype(pattern)>;
                if constexpr (std::is_same_v<Pattern, Component::RawConstantPattern>) {
                    return pattern.parameters.bits;
                } else if constexpr (std::is_same_v<Pattern, Component::UniformRawIntegerPattern>) {
                    return static_cast<uint64_t>(static_cast<int64_t>(
                        indexedUniformInteger(seed, domain, logicalIndex, pattern.parameters.lower,
                                              pattern.parameters.upper)));
                } else if constexpr (std::is_same_v<Pattern,
                                                    Component::UniformFiniteEncodedValuePattern>) {
                    return uniformFiniteEncodedValue(destinationType, seed, domain, logicalIndex);
                } else if constexpr (std::is_same_v<Pattern, Component::RandomRawBitsPattern>) {
                    return counterRandom(seed, domain, logicalIndex);
                } else if constexpr (std::is_same_v<Pattern,
                                                    Component::RawSerialDimensionPattern>) {
                    if (pattern.parameters.dimension >= shape.rank())
                        throw std::out_of_range("Generation dimension exceeds tensor rank.");
                    return static_cast<uint64_t>(indices[pattern.parameters.dimension]);
                } else {
                    throw std::invalid_argument(
                        "Requested generation pattern does not produce raw storage.");
                }
            },
            component.pattern_);
    }

    static double baseGenerationValue(const Component& component, uint64_t seed, uint64_t domain,
                                      std::span<const size_t> indices, const Shape& shape,
                                      size_t logicalIndex, ScalarType destinationType) {
        return std::visit(
            [&](const auto& pattern) -> double {
                using Pattern = std::remove_cvref_t<decltype(pattern)>;
                if constexpr (std::is_same_v<Pattern, Component::ZeroPattern>) {
                    return 0.0;
                } else if constexpr (std::is_same_v<Pattern, Component::ConstantPattern>) {
                    return pattern.parameters.value;
                } else if constexpr (std::is_same_v<Pattern, Component::ChoicePattern>) {
                    if (pattern.parameters.values.empty())
                        throw std::invalid_argument("Choice generation requires values.");
                    return pattern.parameters.values[counterRandom(seed, domain, logicalIndex) %
                                                     pattern.parameters.values.size()];
                } else if constexpr (std::is_same_v<Pattern, Component::UniformIntegerPattern>) {
                    return static_cast<double>(indexedUniformInteger(seed, domain, logicalIndex,
                                                                     pattern.parameters.lower,
                                                                     pattern.parameters.upper));
                } else if constexpr (std::is_same_v<Pattern,
                                                    Component::AbsoluteUniformIntegerPattern>) {
                    return std::abs(static_cast<double>(
                        indexedUniformInteger(seed, domain, logicalIndex, pattern.parameters.lower,
                                              pattern.parameters.upper)));
                } else if constexpr (std::is_same_v<Pattern, Component::UniformRealPattern>) {
                    if (pattern.parameters.lower > pattern.parameters.upper)
                        throw std::invalid_argument(
                            "Uniform-real lower bound exceeds upper bound.");
                    const double unit = indexedUniformUnit(seed, domain, logicalIndex);
                    return pattern.parameters.lower +
                           unit * (pattern.parameters.upper - pattern.parameters.lower);
                } else if constexpr (std::is_same_v<Pattern, Component::NormalPattern>) {
                    constexpr double twoPi = 6.28318530717958647692528676655900576;
                    const double first = indexedUniformUnit(seed, domain, 2 * logicalIndex);
                    const double second = indexedUniformUnit(seed, domain, 2 * logicalIndex + 1);
                    const double standardNormal =
                        std::sqrt(-2.0 * std::log(first)) * std::cos(twoPi * second);
                    return pattern.parameters.mean +
                           pattern.parameters.standardDeviation * standardNormal;
                } else if constexpr (std::is_same_v<Pattern, Component::SinePattern>) {
                    return std::sin(static_cast<double>(logicalIndex));
                } else if constexpr (std::is_same_v<Pattern, Component::CosinePattern>) {
                    return std::cos(static_cast<double>(logicalIndex));
                } else if constexpr (std::is_same_v<Pattern, Component::AbsoluteSinePattern>) {
                    return std::abs(std::sin(static_cast<double>(logicalIndex)));
                } else if constexpr (std::is_same_v<Pattern, Component::AbsoluteCosinePattern>) {
                    return std::abs(std::cos(static_cast<double>(logicalIndex)));
                } else if constexpr (std::is_same_v<Pattern, Component::SerialIndexPattern>) {
                    return static_cast<double>(logicalIndex);
                } else if constexpr (std::is_same_v<Pattern, Component::SerialDimensionPattern>) {
                    if (pattern.parameters.dimension >= shape.rank())
                        throw std::out_of_range("Generation dimension exceeds tensor rank.");
                    return static_cast<double>(indices[pattern.parameters.dimension]);
                } else if constexpr (std::is_same_v<Pattern,
                                                    Component::AffineIndexRemainderPattern>) {
                    const auto& parameters = pattern.parameters;
                    if (parameters.dimensionCoefficients.size() != shape.rank())
                        throw std::invalid_argument(
                            "Affine-index coefficient count must match the tensor rank.");
                    if (parameters.positiveDivisor <= 0)
                        throw std::invalid_argument(
                            "Affine-index remainder divisor must be positive.");

                    int64_t value = parameters.offset;
                    for (size_t dimension = 0; dimension < shape.rank(); ++dimension) {
                        const int64_t coefficient = parameters.dimensionCoefficients[dimension];
                        if (indices[dimension] >
                            static_cast<size_t>(std::numeric_limits<int64_t>::max()))
                            throw std::overflow_error("Affine-index coordinate exceeds Int64.");
                        const int64_t index = static_cast<int64_t>(indices[dimension]);
                        int64_t term = 0;
                        if (coefficient > 0) {
                            if (index > std::numeric_limits<int64_t>::max() / coefficient)
                                throw std::overflow_error("Affine-index multiplication overflow.");
                            term = coefficient * index;
                        } else if (coefficient < 0) {
                            if (coefficient == std::numeric_limits<int64_t>::min()) {
                                if (index > 1)
                                    throw std::overflow_error(
                                        "Affine-index multiplication overflow.");
                                term = index == 0 ? 0 : std::numeric_limits<int64_t>::min();
                            } else {
                                const int64_t magnitude = -coefficient;
                                if (index > std::numeric_limits<int64_t>::max() / magnitude)
                                    throw std::overflow_error(
                                        "Affine-index multiplication overflow.");
                                term = -(magnitude * index);
                            }
                        }

                        if ((term > 0 && value > std::numeric_limits<int64_t>::max() - term) ||
                            (term < 0 && value < std::numeric_limits<int64_t>::min() - term))
                            throw std::overflow_error("Affine-index addition overflow.");
                        value += term;
                    }
                    return static_cast<double>(value % parameters.positiveDivisor);
                } else if constexpr (std::is_same_v<Pattern, Component::IdentityPattern>) {
                    if (shape.rank() < 2)
                        throw std::invalid_argument(
                            "Identity generation requires rank at least two.");
                    return indices[0] == indices[1] ? 1.0 : 0.0;
                } else if constexpr (std::is_same_v<Pattern,
                                                    Component::CheckerboardUniformIntegerPattern>) {
                    double value = static_cast<double>(
                        indexedUniformInteger(seed, domain, logicalIndex, pattern.parameters.lower,
                                              pattern.parameters.upper));
                    size_t parity = 0;
                    for (const size_t index : indices) parity ^= index;
                    return (parity & 1U) == 0 ? -value : value;
                } else if constexpr (std::is_same_v<Pattern, Component::TypeMaximumPattern>) {
                    return typeMaximum(destinationType);
                } else if constexpr (std::is_same_v<Pattern, Component::TypeLowestPattern>) {
                    return typeLowest(destinationType);
                } else if constexpr (std::is_same_v<Pattern,
                                                    Component::TypeDenormalMinimumPattern>) {
                    return typeDenormalMinimum(destinationType);
                } else if constexpr (std::is_same_v<Pattern,
                                                    Component::TypeDenormalMaximumPattern>) {
                    return typeDenormalMaximum(destinationType);
                } else if constexpr (std::is_same_v<Pattern, Component::TypeNaNPattern>) {
                    return typeNaN(destinationType);
                } else if constexpr (std::is_same_v<Pattern, Component::TypeInfinityPattern>) {
                    return typeInfinity(destinationType, false);
                } else if constexpr (std::is_same_v<Pattern,
                                                    Component::TypeNegativeInfinityPattern>) {
                    return typeInfinity(destinationType, true);
                } else if constexpr (std::is_same_v<Pattern, Component::TypeNegativeZeroPattern>) {
                    return -0.0;
                } else if constexpr (std::is_same_v<Pattern, Component::UniformTypeRangePattern>) {
                    const double maximum = typeMaximum(destinationType);
                    const double unit = indexedUniformUnit(seed, domain, logicalIndex);
                    return maximum * (2.0 * unit - 1.0);
                } else if constexpr (std::is_same_v<Pattern,
                                                    Component::RandomEncodedExponentPattern>) {
                    return randomEncodedExponentValue(pattern.parameters, seed, domain,
                                                      logicalIndex, destinationType);
                } else {
                    throw std::invalid_argument("Raw generation requires encoded storage output.");
                }
            },
            component.pattern_);
    }

    static double generationValue(const Component& component, uint64_t seed, uint64_t domain,
                                  std::span<const size_t> indices, const Shape& shape,
                                  size_t logicalIndex, ScalarType destinationType) {
        if (component.zeroOutsideMainDiagonal_) {
            if (shape.rank() < 2)
                throw std::invalid_argument("Main-diagonal generation requires rank at least two.");
            if (indices[0] != indices[1]) return 0.0;
        }
        double value = baseGenerationValue(component, seed, domain, indices, shape, logicalIndex,
                                           destinationType);
        switch (component.unaryTransform_) {
            case Component::UnaryTransform::None:
                break;
            case Component::UnaryTransform::Absolute:
                value = std::abs(value);
                break;
            case Component::UnaryTransform::Sine:
                value = std::sin(value);
                break;
            case Component::UnaryTransform::Cosine:
                value = std::cos(value);
                break;
        }
        value = value * component.affineValue_.scale + component.affineValue_.offset;

        if (component.alternatingSign_.has_value()) {
            size_t parity = 0;
            for (const size_t dimension : component.alternatingSign_->dimensions) {
                if (dimension >= shape.rank())
                    throw std::out_of_range(
                        "Alternating-sign generation dimension exceeds tensor rank.");
                parity ^= indices[dimension];
            }
            if (((parity & 1U) != 0) == component.alternatingSign_->negativeWhenOdd) value = -value;
        }
        return value;
    }

    static const GenerationRecipe::BoundComponent& realComponent(const GenerationRecipe& recipe) {
        return *std::visit(
            [](const auto& policy) -> const GenerationRecipe::BoundComponent* {
                using Policy = std::remove_cvref_t<decltype(policy)>;
                if constexpr (std::is_same_v<Policy, GenerationRecipe::RealOnlyPolicy>)
                    return &policy.real;
                else if constexpr (std::is_same_v<Policy, GenerationRecipe::ReplicatedPolicy>)
                    return &policy.value;
                else
                    return &policy.real;
            },
            recipe.complexPolicy_);
    }

    static uint64_t randomDomain(const GenerationRecipe::BoundComponent& component) {
        return component.randomDomain;
    }

    static double generatedNumericalValue(const GenerationRecipe& recipe,
                                          std::span<const size_t> indices, const Shape& shape,
                                          size_t logicalIndex, ScalarType destinationType) {
        const GenerationRecipe::BoundComponent& component = realComponent(recipe);
        if (component.component.isRaw())
            throw std::invalid_argument(
                "Raw generation recipe does not produce a numerical value.");
        return generationValue(component.component, recipe.settings_.seed, randomDomain(component),
                               indices, shape, logicalIndex, destinationType);
    }

    static uint64_t generatedRawValue(const GenerationRecipe& recipe,
                                      std::span<const size_t> indices, const Shape& shape,
                                      size_t logicalIndex, ScalarType destinationType) {
        const GenerationRecipe::BoundComponent& component = realComponent(recipe);
        if (!component.component.isRaw())
            throw std::invalid_argument("Numerical generation recipe does not produce raw bits.");
        return rawGenerationValue(component.component, recipe.settings_.seed,
                                  randomDomain(component), indices, shape, logicalIndex,
                                  destinationType);
    }

    static void writeRawGeneration(Tensor destination, const GenerationRecipe& recipe,
                                   const GenerationRecipe::BoundComponent& bound,
                                   std::span<const size_t> indices, size_t logicalIndex) {
        const uint16_t bits = scalarTypeInfo(destination.type()).storageBits;
        if (bits > 64)
            throw std::invalid_argument("Raw generation supports scalar encodings up to 64 bits.");

        const ptrdiff_t elementOffset = destination.layout().elementOffset(indices);
        const uint64_t raw =
            rawGenerationValue(bound.component, recipe.settings_.seed, randomDomain(bound), indices,
                               destination.shape(), logicalIndex, destination.type());
        const uint64_t offsetBits = bitOffset(destination.type(), elementOffset);
        if (bits <= 32) {
            writePackedBits(destination.rawEncodedBackingStorage(), offsetBits, bits,
                            static_cast<uint32_t>(raw));
        } else {
            writeNative<uint64_t>(destination.rawEncodedBackingStorage(),
                                  static_cast<size_t>(offsetBits / 8), raw);
        }
    }

    static void generateElement(Tensor destination, const GenerationRecipe& recipe,
                                std::span<const size_t> indices, size_t logicalIndex) {
        const bool complexOutput =
            scalarTypeInfo(destination.type()).category == ScalarCategory::Complex;
        const GenerationRecipe::BoundComponent* real = &realComponent(recipe);

        if (!complexOutput) {
            if (real->component.isRaw()) {
                writeRawGeneration(destination, recipe, *real, indices, logicalIndex);
                return;
            }
            destination.storeFrom(
                indices, generatedNumericalValue(recipe, indices, destination.shape(), logicalIndex,
                                                 destination.type()));
            return;
        }

        std::visit(
            [&](const auto& policy) {
                using Policy = std::remove_cvref_t<decltype(policy)>;
                if constexpr (std::is_same_v<Policy, GenerationRecipe::RealOnlyPolicy>) {
                    if (policy.real.component.isRaw())
                        throw std::invalid_argument(
                            "Raw generation does not support complex output.");
                    const double value = generationValue(
                        policy.real.component, recipe.settings_.seed, randomDomain(policy.real),
                        indices, destination.shape(), logicalIndex, destination.type());
                    destination.storeFrom(indices, std::complex<double>(value, 0.0));
                } else if constexpr (std::is_same_v<Policy, GenerationRecipe::ReplicatedPolicy>) {
                    if (policy.value.component.isRaw())
                        throw std::invalid_argument(
                            "Raw generation does not support complex output.");
                    const double value = generationValue(
                        policy.value.component, recipe.settings_.seed, randomDomain(policy.value),
                        indices, destination.shape(), logicalIndex, destination.type());
                    destination.storeFrom(indices, std::complex<double>(value, value));
                } else {
                    if (policy.real.component.isRaw() || policy.imaginary.component.isRaw())
                        throw std::invalid_argument(
                            "Raw generation does not support complex output.");
                    const double realValue = generationValue(
                        policy.real.component, recipe.settings_.seed, randomDomain(policy.real),
                        indices, destination.shape(), logicalIndex, destination.type());
                    const double imaginaryValue =
                        generationValue(policy.imaginary.component, recipe.settings_.seed,
                                        randomDomain(policy.imaginary), indices,
                                        destination.shape(), logicalIndex, destination.type());
                    destination.storeFrom(indices, std::complex<double>(realValue, imaginaryValue));
                }
            },
            recipe.complexPolicy_);
    }
};

inline void generateElement(Tensor destination, const GenerationRecipe& recipe,
                            std::span<const size_t> indices, size_t logicalIndex) {
    GenerationRecipeAccess::generateElement(destination, recipe, indices, logicalIndex);
}
}  // namespace roc::host_numerics::detail
