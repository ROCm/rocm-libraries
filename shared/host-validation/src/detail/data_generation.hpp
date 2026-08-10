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
inline size_t logicalLinearIndex(std::span<const size_t> indices, const Shape& shape,
                                 LogicalIndexOrder order) {
    size_t result = 0;
    size_t stride = 1;
    if (order == LogicalIndexOrder::FirstDimensionFastest) {
        for (size_t dimension = 0; dimension < shape.rank(); ++dimension) {
            result += indices[dimension] * stride;
            stride *= shape[dimension];
        }
    } else {
        for (size_t dimension = shape.rank(); dimension > 0; --dimension) {
            const size_t index = dimension - 1;
            result += indices[index] * stride;
            stride *= shape[index];
        }
    }
    return result;
}

inline std::vector<size_t> logicalCoordinates(size_t linearIndex, const Shape& shape,
                                              LogicalIndexOrder order) {
    if (linearIndex >= shape.elementCount())
        throw std::out_of_range("Generation logical index exceeds tensor shape.");

    std::vector<size_t> indices(shape.rank(), 0);
    if (order == LogicalIndexOrder::FirstDimensionFastest) {
        for (size_t dimension = 0; dimension < shape.rank(); ++dimension) {
            indices[dimension] = linearIndex % shape[dimension];
            linearIndex /= shape[dimension];
        }
    } else {
        for (size_t dimension = shape.rank(); dimension > 0; --dimension) {
            const size_t index = dimension - 1;
            indices[index] = linearIndex % shape[index];
            linearIndex /= shape[index];
        }
    }
    return indices;
}

inline double indexedUniformUnit(uint64_t seed, uint64_t stream, uint64_t index) {
    constexpr double inverseTwoTo53 = 1.0 / 9007199254740992.0;
    const uint64_t mantissa = counterRandom(seed, stream, index) >> 11;
    return (static_cast<double>(mantissa) + 0.5) * inverseTwoTo53;
}

inline bool isRawGenerationPattern(GenerationPattern pattern) {
    return pattern == GenerationPattern::RawConstant ||
           pattern == GenerationPattern::UniformRawInteger ||
           pattern == GenerationPattern::RandomRawBits ||
           pattern == GenerationPattern::RawSerialDimension;
}

inline uint64_t rawGenerationValue(const GenerationPatternSpec& spec, uint64_t seed,
                                   std::span<const size_t> indices, const Shape& shape,
                                   size_t logicalIndex) {
    switch (spec.pattern) {
        case GenerationPattern::RawConstant:
            return static_cast<uint64_t>(static_cast<int64_t>(spec.parameter0));
        case GenerationPattern::UniformRawInteger:
            return static_cast<uint64_t>(static_cast<int64_t>(indexedUniformInteger(
                seed, spec.stream, logicalIndex, static_cast<int>(spec.parameter0),
                static_cast<int>(spec.parameter1))));
        case GenerationPattern::RandomRawBits:
            return counterRandom(seed, spec.stream, logicalIndex);
        case GenerationPattern::RawSerialDimension:
            if (spec.dimension >= shape.rank())
                throw std::out_of_range("Generation dimension exceeds tensor rank.");
            return static_cast<uint64_t>(indices[spec.dimension]);
        default:
            throw std::invalid_argument(
                "Requested generation pattern does not produce raw storage.");
    }
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
        case ScalarType::E5M3: {
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
        case ScalarType::E5M3: {
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

inline double randomEncodedExponentValue(const GenerationPatternSpec& spec, uint64_t seed,
                                         size_t logicalIndex, ScalarType destinationType) {
    ScalarType type = spec.sourceType == ScalarType::Count
                          ? generationComponentType(destinationType)
                          : generationComponentType(spec.sourceType);
    const ScalarTypeInfo& info = scalarTypeInfo(type);
    if (info.exponentBits == 0)
        throw std::invalid_argument(
            "Random encoded-exponent generation requires a floating-point encoding.");
    const int lowerExponent = static_cast<int>(spec.parameter0);
    const int upperExponent = static_cast<int>(spec.parameter1);
    if (lowerExponent > upperExponent)
        throw std::invalid_argument("Random encoded-exponent lower bound exceeds upper bound.");

    const uint64_t randomBits = counterRandom(seed, spec.stream, logicalIndex);
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
            return decodeBinaryFloat(type, static_cast<uint32_t>(raw));
        case ScalarType::E8M0:
            return decodeE8M0(static_cast<uint8_t>(raw));
        default:
            throw std::invalid_argument("Random encoded-exponent source type is unsupported.");
    }
}

inline double baseGenerationValue(const GenerationPatternSpec& spec, uint64_t seed,
                                  std::span<const size_t> indices, const Shape& shape,
                                  size_t logicalIndex, ScalarType destinationType) {
    switch (spec.pattern) {
        case GenerationPattern::Zero:
            return 0.0;
        case GenerationPattern::Constant:
            return spec.parameter0;
        case GenerationPattern::CandidateSet:
            if (spec.candidates.empty())
                throw std::invalid_argument("Candidate-set generation requires values.");
            return spec.candidates[counterRandom(seed, spec.stream, logicalIndex) %
                                   spec.candidates.size()];
        case GenerationPattern::UniformInteger:
            return static_cast<double>(indexedUniformInteger(seed, spec.stream, logicalIndex,
                                                             static_cast<int>(spec.parameter0),
                                                             static_cast<int>(spec.parameter1)));
        case GenerationPattern::AbsoluteUniformInteger:
            return std::abs(static_cast<double>(indexedUniformInteger(
                seed, spec.stream, logicalIndex, static_cast<int>(spec.parameter0),
                static_cast<int>(spec.parameter1))));
        case GenerationPattern::UniformReal: {
            if (spec.parameter0 > spec.parameter1)
                throw std::invalid_argument("Uniform-real lower bound exceeds upper bound.");
            const double unit = indexedUniformUnit(seed, spec.stream, logicalIndex);
            return spec.parameter0 + unit * (spec.parameter1 - spec.parameter0);
        }
        case GenerationPattern::Normal: {
            constexpr double twoPi = 6.28318530717958647692528676655900576;
            const double first = indexedUniformUnit(seed, spec.stream, 2 * logicalIndex);
            const double second = indexedUniformUnit(seed, spec.stream, 2 * logicalIndex + 1);
            const double standardNormal =
                std::sqrt(-2.0 * std::log(first)) * std::cos(twoPi * second);
            return spec.parameter0 + spec.parameter1 * standardNormal;
        }
        case GenerationPattern::Sine:
            return std::sin(static_cast<double>(logicalIndex));
        case GenerationPattern::Cosine:
            return std::cos(static_cast<double>(logicalIndex));
        case GenerationPattern::AbsoluteSine:
            return std::abs(std::sin(static_cast<double>(logicalIndex)));
        case GenerationPattern::AbsoluteCosine:
            return std::abs(std::cos(static_cast<double>(logicalIndex)));
        case GenerationPattern::SerialIndex:
            return static_cast<double>(logicalIndex);
        case GenerationPattern::SerialDimension:
            if (spec.dimension >= shape.rank())
                throw std::out_of_range("Generation dimension exceeds tensor rank.");
            return static_cast<double>(indices[spec.dimension]);
        case GenerationPattern::AffineIndexRemainder: {
            if (spec.dimensionCoefficients.size() != shape.rank())
                throw std::invalid_argument(
                    "Affine-index coefficient count must match the tensor rank.");
            if (spec.remainderDivisor <= 0)
                throw std::invalid_argument("Affine-index remainder divisor must be positive.");

            int64_t value = spec.affineOffset;
            for (size_t dimension = 0; dimension < shape.rank(); ++dimension) {
                const int64_t coefficient = spec.dimensionCoefficients[dimension];
                if (indices[dimension] > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
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
                            throw std::overflow_error("Affine-index multiplication overflow.");
                        term = index == 0 ? 0 : std::numeric_limits<int64_t>::min();
                    } else {
                        const int64_t magnitude = -coefficient;
                        if (index > std::numeric_limits<int64_t>::max() / magnitude)
                            throw std::overflow_error("Affine-index multiplication overflow.");
                        term = -(magnitude * index);
                    }
                }

                if ((term > 0 && value > std::numeric_limits<int64_t>::max() - term) ||
                    (term < 0 && value < std::numeric_limits<int64_t>::min() - term))
                    throw std::overflow_error("Affine-index addition overflow.");
                value += term;
            }
            return static_cast<double>(value % spec.remainderDivisor);
        }
        case GenerationPattern::Identity:
            if (shape.rank() < 2)
                throw std::invalid_argument("Identity generation requires rank at least two.");
            return indices[0] == indices[1] ? 1.0 : 0.0;
        case GenerationPattern::CheckerboardUniformInteger: {
            double value = static_cast<double>(indexedUniformInteger(
                seed, spec.stream, logicalIndex, static_cast<int>(spec.parameter0),
                static_cast<int>(spec.parameter1)));
            size_t parity = 0;
            for (const size_t index : indices) parity ^= index;
            return (parity & 1) == 0 ? -value : value;
        }
        case GenerationPattern::TypeMaximum:
            return typeMaximum(destinationType);
        case GenerationPattern::TypeLowest:
            return typeLowest(destinationType);
        case GenerationPattern::TypeDenormalMinimum:
            return typeDenormalMinimum(destinationType);
        case GenerationPattern::TypeDenormalMaximum:
            return typeDenormalMaximum(destinationType);
        case GenerationPattern::TypeNaN:
            return typeNaN(destinationType);
        case GenerationPattern::TypeInfinity:
            return typeInfinity(destinationType, false);
        case GenerationPattern::TypeNegativeInfinity:
            return typeInfinity(destinationType, true);
        case GenerationPattern::TypeNegativeZero:
            return -0.0;
        case GenerationPattern::UniformTypeRange: {
            const double maximum = typeMaximum(destinationType);
            const double unit = indexedUniformUnit(seed, spec.stream, logicalIndex);
            return maximum * (2.0 * unit - 1.0);
        }
        case GenerationPattern::RandomEncodedExponent:
            return randomEncodedExponentValue(spec, seed, logicalIndex, destinationType);
        case GenerationPattern::RawConstant:
        case GenerationPattern::UniformRawInteger:
        case GenerationPattern::RandomRawBits:
        case GenerationPattern::RawSerialDimension:
            throw std::invalid_argument("Raw generation requires encoded storage output.");
    }
    throw std::invalid_argument("Unsupported GenerationPattern.");
}

inline double generationValue(const GenerationPatternSpec& spec, uint64_t seed,
                              std::span<const size_t> indices, const Shape& shape,
                              size_t logicalIndex, ScalarType destinationType) {
    double value = baseGenerationValue(spec, seed, indices, shape, logicalIndex, destinationType);
    switch (spec.transform) {
        case GenerationTransform::None:
            break;
        case GenerationTransform::Absolute:
            value = std::abs(value);
            break;
        case GenerationTransform::Sine:
            value = std::sin(value);
            break;
        case GenerationTransform::Cosine:
            value = std::cos(value);
            break;
    }
    value = value * spec.valueScale + spec.valueOffset;

    if (!spec.alternatingDimensions.empty()) {
        size_t parity = 0;
        for (const size_t dimension : spec.alternatingDimensions) {
            if (dimension >= shape.rank())
                throw std::out_of_range(
                    "Alternating-sign generation dimension exceeds tensor rank.");
            parity ^= indices[dimension];
        }
        if ((parity & 1U) == (spec.negativeParity & 1U)) value = -value;
    }
    return value;
}

inline void generateElement(MutableTensorView destination, const GenerationOptions& options,
                            std::span<const size_t> indices, size_t logicalIndex) {
    const bool complexOutput =
        scalarTypeInfo(destination.type()).category == ScalarCategory::Complex;
    if (isRawGenerationPattern(options.real.pattern)) {
        if (complexOutput)
            throw std::invalid_argument("Raw generation does not support complex output.");
        const uint16_t bits = scalarTypeInfo(destination.type()).storageBits;
        if (bits > 64)
            throw std::invalid_argument("Raw generation supports scalar encodings up to 64 bits.");

        const ptrdiff_t elementOffset = destination.layout().elementOffset(indices);
        const uint64_t raw = rawGenerationValue(options.real, options.seed, indices,
                                                destination.shape(), logicalIndex);
        const uint64_t offsetBits = bitOffset(destination.type(), elementOffset);
        if (bits <= 32) {
            writePackedBits(destination.storage(), offsetBits, bits, static_cast<uint32_t>(raw));
        } else {
            writeNative<uint64_t>(destination.storage(), static_cast<size_t>(offsetBits / 8), raw);
        }
        return;
    }

    const double real = generationValue(options.real, options.seed, indices, destination.shape(),
                                        logicalIndex, destination.type());
    if (complexOutput) {
        const double imaginary =
            generationValue(options.imaginary, options.seed, indices, destination.shape(),
                            logicalIndex, destination.type());
        destination.storeFrom(indices, std::complex<double>(real, imaginary));
    } else {
        destination.storeFrom(indices, real);
    }
}
}  // namespace detail
}  // namespace roc::host_validation
