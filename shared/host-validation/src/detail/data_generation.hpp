// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <roc/host_validation/generation.hpp>
#include <span>
#include <stdexcept>
#include <vector>

namespace roc::host_validation {
namespace detail {
inline double indexedUniformUnit(uint64_t seed, uint64_t domain, uint64_t index) {
    constexpr double inverseTwoTo53 = 1.0 / 9007199254740992.0;
    const uint64_t mantissa = counterRandom(seed, domain, index) >> 11;
    return (static_cast<double>(mantissa) + 0.5) * inverseTwoTo53;
}

inline ScalarType generationComponentType(ScalarType type) {
    if (type == ScalarType::ComplexFloat32) return ScalarType::Float32;
    if (type == ScalarType::ComplexFloat64) return ScalarType::Float64;
    return type;
}

inline double typeMaximum(ScalarType requestedType) {
    const ScalarType type = generationComponentType(requestedType);
    switch (type) {
        case ScalarType::Boolean:
            return 1.0;
        case ScalarType::UInt8:
            return std::numeric_limits<uint8_t>::max();
        case ScalarType::Int8:
            return std::numeric_limits<int8_t>::max();
        case ScalarType::UInt16:
            return std::numeric_limits<uint16_t>::max();
        case ScalarType::Int16:
            return std::numeric_limits<int16_t>::max();
        case ScalarType::UInt32:
            return std::numeric_limits<uint32_t>::max();
        case ScalarType::Int32:
            return std::numeric_limits<int32_t>::max();
        case ScalarType::Float16:
            return decodeFloat16(0x7bffU);
        case ScalarType::BFloat16:
            return decodeBFloat16(0x7f7fU);
        case ScalarType::Float32:
            return std::numeric_limits<float>::max();
        case ScalarType::Float64:
            return std::numeric_limits<double>::max();
        case ScalarType::Float4E2M1:
        case ScalarType::Float6E2M3:
        case ScalarType::Float6E3M2:
        case ScalarType::Float8E4M3:
        case ScalarType::Float8E5M2:
        case ScalarType::Float8E4M3Fnuz:
        case ScalarType::Float8E5M2Fnuz:
        case ScalarType::E5M3:
        case ScalarType::E4M3: {
            const BinaryFloatFormat format = binaryFloatFormat(type);
            return decodeBinaryFloat(type, format.maximumPositiveFiniteRaw);
        }
        case ScalarType::Int4:
            return 7.0;
        case ScalarType::Int12:
            return 2047.0;
        case ScalarType::E8M0:
            return decodeE8M0(0xfeU);
        case ScalarType::UInt64:
        case ScalarType::Int64:
            throw std::invalid_argument(
                "Type-derived generation does not represent 64-bit integer extrema through "
                "double.");
        case ScalarType::ComplexFloat32:
        case ScalarType::ComplexFloat64:
        case ScalarType::Count:
            break;
    }
    throw std::invalid_argument("Unsupported scalar type for maximum generation.");
}

inline double typeLowest(ScalarType requestedType) {
    const ScalarType type = generationComponentType(requestedType);
    switch (type) {
        case ScalarType::Boolean:
        case ScalarType::UInt8:
        case ScalarType::UInt16:
        case ScalarType::UInt32:
        case ScalarType::UInt64:
        case ScalarType::E8M0:
        case ScalarType::E5M3:
        case ScalarType::E4M3:
            return 0.0;
        case ScalarType::Int8:
            return std::numeric_limits<int8_t>::min();
        case ScalarType::Int16:
            return std::numeric_limits<int16_t>::min();
        case ScalarType::Int32:
            return std::numeric_limits<int32_t>::min();
        case ScalarType::Int4:
            return -8.0;
        case ScalarType::Int12:
            return -2048.0;
        case ScalarType::Float16:
        case ScalarType::BFloat16:
        case ScalarType::Float32:
        case ScalarType::Float64:
        case ScalarType::Float4E2M1:
        case ScalarType::Float6E2M3:
        case ScalarType::Float6E3M2:
        case ScalarType::Float8E4M3:
        case ScalarType::Float8E5M2:
        case ScalarType::Float8E4M3Fnuz:
        case ScalarType::Float8E5M2Fnuz:
            return -typeMaximum(type);
        case ScalarType::Int64:
            throw std::invalid_argument(
                "Type-derived generation does not represent Int64 minimum through double.");
        case ScalarType::ComplexFloat32:
        case ScalarType::ComplexFloat64:
        case ScalarType::Count:
            break;
    }
    throw std::invalid_argument("Unsupported scalar type for lowest-value generation.");
}

inline double typeDenormalMinimum(ScalarType requestedType) {
    const ScalarType type = generationComponentType(requestedType);
    switch (type) {
        case ScalarType::Float16:
            return decodeFloat16(0x0001U);
        case ScalarType::BFloat16:
            return decodeBFloat16(0x0001U);
        case ScalarType::Float32:
            return std::numeric_limits<float>::denorm_min();
        case ScalarType::Float64:
            return std::numeric_limits<double>::denorm_min();
        case ScalarType::Float4E2M1:
        case ScalarType::Float6E2M3:
        case ScalarType::Float6E3M2:
        case ScalarType::Float8E4M3:
        case ScalarType::Float8E5M2:
        case ScalarType::Float8E4M3Fnuz:
        case ScalarType::Float8E5M2Fnuz:
        case ScalarType::E5M3:
        case ScalarType::E4M3:
            return decodeBinaryFloat(type, 1U);
        default:
            throw std::invalid_argument("Requested scalar type has no denormal minimum.");
    }
}

inline double typeDenormalMaximum(ScalarType requestedType) {
    const ScalarType type = generationComponentType(requestedType);
    switch (type) {
        case ScalarType::Float16:
            return decodeFloat16(0x03ffU);
        case ScalarType::BFloat16:
            return decodeBFloat16(0x007fU);
        case ScalarType::Float32:
            return std::bit_cast<float>(0x007fffffU);
        case ScalarType::Float64:
            return std::bit_cast<double>(0x000fffffffffffffULL);
        case ScalarType::Float4E2M1:
        case ScalarType::Float6E2M3:
        case ScalarType::Float6E3M2:
        case ScalarType::Float8E4M3:
        case ScalarType::Float8E5M2:
        case ScalarType::Float8E4M3Fnuz:
        case ScalarType::Float8E5M2Fnuz:
        case ScalarType::E5M3:
        case ScalarType::E4M3: {
            const BinaryFloatFormat format = binaryFloatFormat(type);
            return decodeBinaryFloat(type, (1U << format.mantissaBits) - 1U);
        }
        default:
            throw std::invalid_argument("Requested scalar type has no denormal maximum.");
    }
}

inline double typeNaN(ScalarType type) {
    type = generationComponentType(type);
    if (!scalarTypeInfo(type).supportsNaN)
        throw std::invalid_argument("Requested scalar type has no NaN encoding.");
    return std::numeric_limits<double>::quiet_NaN();
}

inline double typeInfinity(ScalarType type, bool negative) {
    type = generationComponentType(type);
    if (!scalarTypeInfo(type).supportsInfinity)
        throw std::invalid_argument("Requested scalar type has no infinity encoding.");
    return negative ? -std::numeric_limits<double>::infinity()
                    : std::numeric_limits<double>::infinity();
}

inline double randomEncodedExponentValue(
    const RandomEncodedExponentGenerationParameters& parameters, uint64_t seed, uint64_t domain,
    size_t logicalIndex, ScalarType destinationType) {
    ScalarType type = parameters.sourceType.has_value()
                          ? generationComponentType(*parameters.sourceType)
                          : generationComponentType(destinationType);
    const ScalarTypeInfo& info = scalarTypeInfo(type);
    if (info.exponentBits == 0)
        throw std::invalid_argument(
            "Random encoded-exponent generation requires a floating-point encoding.");
    const int lowerExponent = parameters.lowerUnbiasedExponent;
    const int upperExponent = parameters.upperUnbiasedExponent;
    if (lowerExponent > upperExponent)
        throw std::invalid_argument("Random encoded-exponent lower bound exceeds upper bound.");

    const uint64_t randomBits = counterRandom(seed, domain, logicalIndex);
    const int exponent =
        static_cast<int>(randomBits % static_cast<uint64_t>(upperExponent - lowerExponent + 1)) +
        lowerExponent;
    const uint64_t exponentMask = ((uint64_t{1} << info.exponentBits) - 1U) << info.mantissaBits;
    const uint64_t encoded =
        (randomBits & ~exponentMask) |
        ((static_cast<uint64_t>(exponent + info.exponentBias) << info.mantissaBits) & exponentMask);
    const uint64_t storageMask = info.storageBits == 64 ? std::numeric_limits<uint64_t>::max()
                                                        : ((uint64_t{1} << info.storageBits) - 1U);
    const uint64_t raw = encoded & storageMask;

    switch (type) {
        case ScalarType::Float64:
            return std::bit_cast<double>(raw);
        case ScalarType::Float32:
            return std::bit_cast<float>(static_cast<uint32_t>(raw));
        case ScalarType::Float16:
            return decodeFloat16(static_cast<uint16_t>(raw));
        case ScalarType::BFloat16:
            return decodeBFloat16(static_cast<uint16_t>(raw));
        case ScalarType::Float4E2M1:
        case ScalarType::Float6E2M3:
        case ScalarType::Float6E3M2:
        case ScalarType::Float8E4M3:
        case ScalarType::Float8E5M2:
        case ScalarType::Float8E4M3Fnuz:
        case ScalarType::Float8E5M2Fnuz:
        case ScalarType::E5M3:
        case ScalarType::E4M3:
            return decodeBinaryFloat(type, static_cast<uint32_t>(raw));
        case ScalarType::E8M0:
            return decodeE8M0(static_cast<uint8_t>(raw));
        default:
            throw std::invalid_argument("Random encoded-exponent source type is unsupported.");
    }
}

struct GenerationRecipeAccess {
    using Component = GenerationRecipe::Component;

    static uint64_t rawGenerationValue(const Component& component, uint64_t seed, uint64_t domain,
                                       std::span<const size_t> indices, const Shape& shape,
                                       size_t logicalIndex) {
        return std::visit(
            [&](const auto& pattern) -> uint64_t {
                using Pattern = std::remove_cvref_t<decltype(pattern)>;
                if constexpr (std::is_same_v<Pattern, Component::RawConstantPattern>) {
                    return pattern.parameters.bits;
                } else if constexpr (std::is_same_v<Pattern, Component::UniformRawIntegerPattern>) {
                    return static_cast<uint64_t>(static_cast<int64_t>(
                        indexedUniformInteger(seed, domain, logicalIndex, pattern.parameters.lower,
                                              pattern.parameters.upper)));
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
                } else if constexpr (std::is_same_v<Pattern, Component::CandidateSetPattern>) {
                    if (pattern.parameters.values.empty())
                        throw std::invalid_argument("Candidate-set generation requires values.");
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

    static void writeRawGeneration(Tensor destination, const GenerationRecipe& recipe,
                                   const GenerationRecipe::BoundComponent& bound,
                                   std::span<const size_t> indices, size_t logicalIndex) {
        const uint16_t bits = scalarTypeInfo(destination.type()).storageBits;
        if (bits > 64)
            throw std::invalid_argument("Raw generation supports scalar encodings up to 64 bits.");

        const ptrdiff_t elementOffset = destination.layout().elementOffset(indices);
        const uint64_t raw =
            rawGenerationValue(bound.component, recipe.settings_.seed, bound.randomDomain, indices,
                               destination.shape(), logicalIndex);
        const uint64_t offsetBits = bitOffset(destination.type(), elementOffset);
        if (bits <= 32) {
            writePackedBits(destination.storage(), offsetBits, bits, static_cast<uint32_t>(raw));
        } else {
            writeNative<uint64_t>(destination.storage(), static_cast<size_t>(offsetBits / 8), raw);
        }
    }

    static void generateElement(Tensor destination, const GenerationRecipe& recipe,
                                std::span<const size_t> indices, size_t logicalIndex) {
        const bool complexOutput =
            scalarTypeInfo(destination.type()).category == ScalarCategory::Complex;
        const GenerationRecipe::BoundComponent* real = std::visit(
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

        if (!complexOutput) {
            if (real->component.isRaw()) {
                writeRawGeneration(destination, recipe, *real, indices, logicalIndex);
                return;
            }
            destination.storeFrom(
                indices,
                generationValue(real->component, recipe.settings_.seed, real->randomDomain, indices,
                                destination.shape(), logicalIndex, destination.type()));
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
                        policy.real.component, recipe.settings_.seed, policy.real.randomDomain,
                        indices, destination.shape(), logicalIndex, destination.type());
                    destination.storeFrom(indices, std::complex<double>(value, 0.0));
                } else if constexpr (std::is_same_v<Policy, GenerationRecipe::ReplicatedPolicy>) {
                    if (policy.value.component.isRaw())
                        throw std::invalid_argument(
                            "Raw generation does not support complex output.");
                    const double value = generationValue(
                        policy.value.component, recipe.settings_.seed, policy.value.randomDomain,
                        indices, destination.shape(), logicalIndex, destination.type());
                    destination.storeFrom(indices, std::complex<double>(value, value));
                } else {
                    if (policy.real.component.isRaw() || policy.imaginary.component.isRaw())
                        throw std::invalid_argument(
                            "Raw generation does not support complex output.");
                    const double realValue = generationValue(
                        policy.real.component, recipe.settings_.seed, policy.real.randomDomain,
                        indices, destination.shape(), logicalIndex, destination.type());
                    const double imaginaryValue =
                        generationValue(policy.imaginary.component, recipe.settings_.seed,
                                        policy.imaginary.randomDomain, indices, destination.shape(),
                                        logicalIndex, destination.type());
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
}  // namespace detail
}  // namespace roc::host_validation
