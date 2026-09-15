// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "detail/comparison_metrics.hpp"

#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace roc::host_numerics::detail {
double valueMagnitude(const ComparisonValue& value) {
    return value.complex ? std::hypot(value.real, value.imaginary) : std::abs(value.real);
}

uint64_t orderedFloatingEncoding(uint64_t raw, uint32_t bitCount) {
    if (bitCount == 0 || bitCount > 64)
        throw std::invalid_argument("Floating encoding width must be in [1, 64].");
    const uint64_t mask =
        bitCount == 64 ? std::numeric_limits<uint64_t>::max() : (uint64_t{1} << bitCount) - 1;
    const uint64_t sign = uint64_t{1} << (bitCount - 1);
    raw &= mask;
    return (raw & sign) != 0 ? (~raw + 1) & mask : raw | sign;
}

double encodedUlpDistance(double exact, double approximation, ScalarType type) {
    if (exact == approximation) return 0.0;
    if (!std::isfinite(exact) || !std::isfinite(approximation))
        return std::numeric_limits<double>::infinity();

    uint64_t exactEncoding = 0;
    uint64_t approximationEncoding = 0;
    switch (type) {
        case ScalarType::Float32:
        case ScalarType::ComplexFloat32:
            exactEncoding =
                orderedFloatingEncoding(std::bit_cast<uint32_t>(static_cast<float>(exact)), 32);
            approximationEncoding = orderedFloatingEncoding(
                std::bit_cast<uint32_t>(static_cast<float>(approximation)), 32);
            break;
        case ScalarType::Float64:
        case ScalarType::ComplexFloat64:
            exactEncoding = orderedFloatingEncoding(std::bit_cast<uint64_t>(exact), 64);
            approximationEncoding =
                orderedFloatingEncoding(std::bit_cast<uint64_t>(approximation), 64);
            break;
        case ScalarType::Float16:
            exactEncoding = orderedFloatingEncoding(encodeFloat16(static_cast<float>(exact)), 16);
            approximationEncoding =
                orderedFloatingEncoding(encodeFloat16(static_cast<float>(approximation)), 16);
            break;
        case ScalarType::BFloat16:
            exactEncoding = orderedFloatingEncoding(encodeBFloat16(static_cast<float>(exact)), 16);
            approximationEncoding =
                orderedFloatingEncoding(encodeBFloat16(static_cast<float>(approximation)), 16);
            break;
        case ScalarType::Float8E4M3:
        case ScalarType::Float8E5M2:
        case ScalarType::Float8E4M3Fnuz:
        case ScalarType::Float8E5M2Fnuz:
        case ScalarType::Float6E2M3:
        case ScalarType::Float6E3M2:
        case ScalarType::Float4E2M1:
        case ScalarType::E5M3:
        case ScalarType::E4M3: {
            const BinaryFloatFormat format = binaryFloatFormat(type);
            const uint64_t exactRaw = encodeBinaryFloat(type, static_cast<float>(exact));
            const uint64_t approximationRaw =
                encodeBinaryFloat(type, static_cast<float>(approximation));
            exactEncoding =
                format.hasSign ? orderedFloatingEncoding(exactRaw, format.totalBits) : exactRaw;
            approximationEncoding =
                format.hasSign ? orderedFloatingEncoding(approximationRaw, format.totalBits)
                               : approximationRaw;
            break;
        }
        case ScalarType::Boolean:
        case ScalarType::UInt8:
        case ScalarType::Int8:
        case ScalarType::UInt16:
        case ScalarType::Int16:
        case ScalarType::UInt32:
        case ScalarType::Int32:
        case ScalarType::UInt64:
        case ScalarType::Int64:
        case ScalarType::Int4:
            return std::abs(exact - approximation);
        default:
            return ulpDistance(exact, approximation, ulpMantissaBits(type));
    }

    return static_cast<double>(exactEncoding >= approximationEncoding
                                   ? exactEncoding - approximationEncoding
                                   : approximationEncoding - exactEncoding);
}

double ulpDistanceForType(double exact, double approximation, ScalarType type,
                          UlpComparisonMode mode) {
    const auto category = scalarTypeInfo(type).category;
    if (category == ScalarCategory::Boolean || category == ScalarCategory::SignedInteger ||
        category == ScalarCategory::UnsignedInteger)
        return std::abs(exact - approximation);
    if (mode == UlpComparisonMode::EncodedDistance)
        return detail::encodedUlpDistance(exact, approximation, type);
    return ulpDistance(exact, approximation, ulpMantissaBits(type));
}
}  // namespace roc::host_numerics::detail
