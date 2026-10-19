// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "detail/generation_values.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <roc/hostnumerics/generation.hpp>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "detail/generation_primitives.hpp"

namespace roc::hostnumerics::detail {
ScalarType generationComponentType(ScalarType type) {
    if (type == ScalarType::ComplexFloat32) return ScalarType::Float32;
    if (type == ScalarType::ComplexFloat64) return ScalarType::Float64;
    return type;
}

double typeMaximum(ScalarType requestedType) {
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
        case ScalarType::E8M0:
        case ScalarType::E8M0Zero:
            return decodeE8M0(0xfeU);
        case ScalarType::UInt64:
        case ScalarType::Int64:
            throw std::runtime_error(
                "Type-derived generation does not represent 64-bit integer extrema through "
                "double.");
        case ScalarType::ComplexFloat32:
        case ScalarType::ComplexFloat64:
        case ScalarType::Count:
            break;
    }
    throw std::runtime_error("Unsupported scalar type for maximum generation.");
}

double typeLowest(ScalarType requestedType) {
    const ScalarType type = generationComponentType(requestedType);
    switch (type) {
        case ScalarType::Boolean:
        case ScalarType::UInt8:
        case ScalarType::UInt16:
        case ScalarType::UInt32:
        case ScalarType::UInt64:
        case ScalarType::E8M0:
        case ScalarType::E8M0Zero:
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
            throw std::runtime_error(
                "Type-derived generation does not represent Int64 minimum through double.");
        case ScalarType::ComplexFloat32:
        case ScalarType::ComplexFloat64:
        case ScalarType::Count:
            break;
    }
    throw std::runtime_error("Unsupported scalar type for lowest-value generation.");
}

double typeDenormalMinimum(ScalarType requestedType) {
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
            throw std::runtime_error("Requested scalar type has no denormal minimum.");
    }
}

double typeDenormalMaximum(ScalarType requestedType) {
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
            throw std::runtime_error("Requested scalar type has no denormal maximum.");
    }
}

double typeNaN(ScalarType type) {
    type = generationComponentType(type);
    if (!scalarTypeInfo(type).supportsNaN)
        throw std::runtime_error("Requested scalar type has no NaN encoding.");
    return std::numeric_limits<double>::quiet_NaN();
}

double typeInfinity(ScalarType type, bool negative) {
    type = generationComponentType(type);
    if (!scalarTypeInfo(type).supportsInfinity)
        throw std::runtime_error("Requested scalar type has no infinity encoding.");
    return negative ? -std::numeric_limits<double>::infinity()
                    : std::numeric_limits<double>::infinity();
}

PreparedEncodedExponent prepareEncodedExponent(
    const RandomEncodedExponentGenerationParameters& parameters, ScalarType destinationType) {
    const ScalarType type =
        generationComponentType(parameters.sourceType.value_or(destinationType));
    const auto& info = scalarTypeInfo(type);
    if (info.exponentBits == 0)
        throw std::runtime_error(
            "Random encoded-exponent generation requires a floating-point encoding.");
    if (parameters.lowerUnbiasedExponent > parameters.upperUnbiasedExponent)
        throw std::runtime_error("Random encoded-exponent lower bound exceeds upper bound.");
    const auto decode = visitScalarType(type, []<typename Tag>() -> double (*)(uint64_t) {
        constexpr auto encoding = Tag::type;
        if constexpr (encoding == ScalarType::Float64)
            return +[](uint64_t raw) { return std::bit_cast<double>(raw); };
        else if constexpr (encoding == ScalarType::Float32)
            return +[](uint64_t raw) -> double {
                return std::bit_cast<float>(static_cast<uint32_t>(raw));
            };
        else if constexpr (encoding == ScalarType::Float16)
            return +[](uint64_t raw) -> double {
                return decodeFloat16(static_cast<uint16_t>(raw));
            };
        else if constexpr (encoding == ScalarType::BFloat16)
            return +[](uint64_t raw) -> double {
                return decodeBFloat16(static_cast<uint16_t>(raw));
            };
        else if constexpr (encoding == ScalarType::E8M0)
            return +[](uint64_t raw) -> double { return decodeE8M0(static_cast<uint8_t>(raw)); };
        else if constexpr (encoding == ScalarType::E8M0Zero)
            return +[](uint64_t raw) -> double {
                return decodeE8M0Zero(static_cast<uint8_t>(raw));
            };
        else if constexpr (scalarTypeInfo(encoding).category == ScalarCategory::FloatingPoint)
            return +[](uint64_t raw) -> double {
                return decodeBinaryFloatKnown<Tag::type>(static_cast<uint32_t>(raw));
            };
        else
            throw std::runtime_error("Random encoded-exponent source type is unsupported.");
    });
    return {
        .lowerExponent = parameters.lowerUnbiasedExponent,
        .exponentCount =
            static_cast<uint64_t>(static_cast<int64_t>(parameters.upperUnbiasedExponent) -
                                  parameters.lowerUnbiasedExponent) +
            1,
        .exponentBias = info.exponentBias,
        .mantissaBits = info.mantissaBits,
        .exponentMask = ((uint64_t{1} << info.exponentBits) - 1U) << info.mantissaBits,
        .storageMask = info.storageBits == 64 ? std::numeric_limits<uint64_t>::max()
                                              : ((uint64_t{1} << info.storageBits) - 1U),
        .decode = decode,
    };
}

double randomEncodedExponentValue(const RandomEncodedExponentGenerationParameters& parameters,
                                  uint64_t seed, uint64_t domain, size_t logicalIndex,
                                  ScalarType destinationType) {
    return prepareEncodedExponent(parameters,
                                  destinationType)(counterRandom(seed, domain, logicalIndex));
}

std::span<const uint8_t> finiteEncodedValues(ScalarType type) {
    const ScalarTypeInfo& info = scalarTypeInfo(type);
    if (info.category != ScalarCategory::FloatingPoint || info.storageBits > 8)
        throw std::runtime_error(
            "Uniform finite encoded generation requires a floating-point encoding of at most "
            "eight bits.");

    static const std::array<std::vector<uint8_t>, scalarTypeCount> candidatesByType = [] {
        std::array<std::vector<uint8_t>, scalarTypeCount> candidates;
        for (size_t typeIndex = 0; typeIndex < scalarTypeCount; ++typeIndex) {
            const ScalarType candidateType = static_cast<ScalarType>(typeIndex);
            const ScalarTypeInfo& candidateInfo = scalarTypeInfo(candidateType);
            if (candidateInfo.category != ScalarCategory::FloatingPoint ||
                candidateInfo.storageBits > 8)
                continue;
            const uint32_t encodingCount = uint32_t{1} << candidateInfo.storageBits;
            for (uint32_t raw = 0; raw < encodingCount; ++raw) {
                const double value = candidateType == ScalarType::E8M0
                                         ? decodeE8M0(static_cast<uint8_t>(raw))
                                     : candidateType == ScalarType::E8M0Zero
                                         ? decodeE8M0Zero(static_cast<uint8_t>(raw))
                                         : decodeBinaryFloat(candidateType, raw);
                if (std::isfinite(value))
                    candidates[typeIndex].push_back(static_cast<uint8_t>(raw));
            }
        }
        return candidates;
    }();

    const std::vector<uint8_t>& candidates = candidatesByType[static_cast<size_t>(type)];
    if (candidates.empty()) throw std::runtime_error("Scalar type has no finite encoded values.");
    return candidates;
}

uint64_t uniformFiniteEncodedValue(ScalarType type, uint64_t seed, uint64_t domain,
                                   size_t logicalIndex) {
    const std::span<const uint8_t> candidates = finiteEncodedValues(type);
    return candidates[counterRandom(seed, domain, logicalIndex) % candidates.size()];
}
}  // namespace roc::hostnumerics::detail
