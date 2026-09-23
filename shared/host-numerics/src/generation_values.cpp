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
#include <roc/host_numerics/generation.hpp>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "detail/generation_primitives.hpp"

namespace roc::host_numerics::detail {
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
            throw std::invalid_argument(
                "Type-derived generation does not represent Int64 minimum through double.");
        case ScalarType::ComplexFloat32:
        case ScalarType::ComplexFloat64:
        case ScalarType::Count:
            break;
    }
    throw std::invalid_argument("Unsupported scalar type for lowest-value generation.");
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
            throw std::invalid_argument("Requested scalar type has no denormal minimum.");
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
            throw std::invalid_argument("Requested scalar type has no denormal maximum.");
    }
}

double typeNaN(ScalarType type) {
    type = generationComponentType(type);
    if (!scalarTypeInfo(type).supportsNaN)
        throw std::invalid_argument("Requested scalar type has no NaN encoding.");
    return std::numeric_limits<double>::quiet_NaN();
}

double typeInfinity(ScalarType type, bool negative) {
    type = generationComponentType(type);
    if (!scalarTypeInfo(type).supportsInfinity)
        throw std::invalid_argument("Requested scalar type has no infinity encoding.");
    return negative ? -std::numeric_limits<double>::infinity()
                    : std::numeric_limits<double>::infinity();
}

double randomEncodedExponentValue(const RandomEncodedExponentGenerationParameters& parameters,
                                  uint64_t seed, uint64_t domain, size_t logicalIndex,
                                  ScalarType destinationType) {
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
        case ScalarType::E8M0Zero:
            return decodeE8M0Zero(static_cast<uint8_t>(raw));
        default:
            throw std::invalid_argument("Random encoded-exponent source type is unsupported.");
    }
}

std::span<const uint8_t> finiteEncodedValues(ScalarType type) {
    const ScalarTypeInfo& info = scalarTypeInfo(type);
    if ((info.category != ScalarCategory::FloatingPoint &&
         info.category != ScalarCategory::Scale) ||
        info.storageBits > 8)
        throw std::invalid_argument(
            "Uniform finite encoded generation requires a floating-point encoding of at most "
            "eight bits.");

    static const std::array<std::vector<uint8_t>, scalarTypeCount> candidatesByType = [] {
        std::array<std::vector<uint8_t>, scalarTypeCount> candidates;
        for (size_t typeIndex = 0; typeIndex < scalarTypeCount; ++typeIndex) {
            const ScalarType candidateType = static_cast<ScalarType>(typeIndex);
            const ScalarTypeInfo& candidateInfo = scalarTypeInfo(candidateType);
            if ((candidateInfo.category != ScalarCategory::FloatingPoint &&
                 candidateInfo.category != ScalarCategory::Scale) ||
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
    if (candidates.empty())
        throw std::invalid_argument("Scalar type has no finite encoded values.");
    return candidates;
}

uint64_t uniformFiniteEncodedValue(ScalarType type, uint64_t seed, uint64_t domain,
                                   size_t logicalIndex) {
    const std::span<const uint8_t> candidates = finiteEncodedValues(type);
    return candidates[counterRandom(seed, domain, logicalIndex) % candidates.size()];
}
}  // namespace roc::host_numerics::detail
