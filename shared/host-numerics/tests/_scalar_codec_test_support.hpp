// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <roc/host_numerics/scalar.hpp>
#include <roc/host_numerics/tensor.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "test_support.hpp"

#pragma once

namespace scalar_codec_test {
namespace {
using roc::host_numerics::BFloat16Rounding;
using roc::host_numerics::convertScalar;
using roc::host_numerics::IntegerOverflow;
using roc::host_numerics::IntegerRounding;
using roc::host_numerics::isConcreteScalarType;
using roc::host_numerics::Layout;
using roc::host_numerics::nativeScalarType;
using roc::host_numerics::ScalarCategory;
using roc::host_numerics::ScalarConversionOptions;
using roc::host_numerics::ScalarType;
using roc::host_numerics::scalarTypeCount;
using roc::host_numerics::scalarTypeInfo;
using roc::host_numerics::scalarTypeInfos;
using roc::host_numerics::Shape;
using roc::host_numerics::Tensor;
using roc::host_numerics::visitScalarType;
using roc::host_numerics::test::require;
using roc::host_numerics::test::requireThrows;

class RoundingModeRestore {
   public:
    RoundingModeRestore() : m_original(std::fegetround()) {}

    ~RoundingModeRestore() {
        if (m_original != -1) std::fesetround(m_original);
    }

   private:
    int m_original;
};

uint16_t bytesToUint16(std::span<const std::byte> bytes) {
    return static_cast<uint16_t>(std::to_integer<uint8_t>(bytes[0])) |
           static_cast<uint16_t>(std::to_integer<uint8_t>(bytes[1])) << 8;
}

uint32_t rawMask(ScalarType type) {
    const uint16_t bits = scalarTypeInfo(type).storageBits;
    return bits == 32 ? std::numeric_limits<uint32_t>::max() : (1U << bits) - 1U;
}

Tensor tensorFromRaw(ScalarType type, uint32_t raw) {
    const size_t bytes = (scalarTypeInfo(type).storageBits + 7) / 8;
    std::vector<std::byte> storage(bytes);
    for (size_t index = 0; index < bytes; ++index)
        storage[index] = static_cast<std::byte>((raw >> (index * 8)) & 0xffU);
    return Tensor::takeOwnershipOfEncodedBackingStorage(
        type, Layout::contiguousLastDimensionFastest(Shape{1}), std::move(storage));
}

uint32_t tensorRaw(const Tensor& tensor) {
    uint32_t raw = 0;
    for (size_t index = 0; index < tensor.rawEncodedBackingStorage().size(); ++index)
        raw |= static_cast<uint32_t>(
                   std::to_integer<uint8_t>(tensor.rawEncodedBackingStorage()[index]))
               << (index * 8);
    return raw & rawMask(tensor.type());
}

uint32_t encodeRaw(ScalarType type, float value) {
    Tensor tensor(type, Shape{1});
    tensor.storeFrom({0}, value);
    return tensorRaw(tensor);
}

struct ExpectedBinaryFormat {
    uint8_t exponentBits;
    uint8_t mantissaBits;
    int exponentBias;
    uint8_t totalBits;
    bool hasSign;
};

// Keep the test oracle independent of the production tag traits so adding a format requires an
// explicit expectation rather than copying the implementation's metadata path.
ExpectedBinaryFormat expectedFormat(ScalarType type) {
    switch (type) {
        case ScalarType::Float4E2M1:
            return {2, 1, 1, 4, true};
        case ScalarType::Float6E2M3:
            return {2, 3, 1, 6, true};
        case ScalarType::Float6E3M2:
            return {3, 2, 3, 6, true};
        case ScalarType::Float8E4M3:
            return {4, 3, 7, 8, true};
        case ScalarType::Float8E5M2:
            return {5, 2, 15, 8, true};
        case ScalarType::Float8E4M3Fnuz:
            return {4, 3, 8, 8, true};
        case ScalarType::Float8E5M2Fnuz:
            return {5, 2, 16, 8, true};
        case ScalarType::E5M3:
            return {5, 3, 15, 8, false};
        case ScalarType::E4M3:
            return {4, 3, 7, 7, false};
        default:
            throw std::invalid_argument("No expected binary format.");
    }
}

// This is likewise an independent encoding oracle; replacing it with visitScalarType would make
// exhaustive tests agree with the implementation by construction.
bool expectedNaN(ScalarType type, uint32_t raw) {
    switch (type) {
        case ScalarType::Float8E4M3:
            return (raw & 0x7fU) == 0x7fU;
        case ScalarType::Float8E5M2:
            return (raw & 0x7fU) > 0x7cU;
        case ScalarType::Float8E4M3Fnuz:
        case ScalarType::Float8E5M2Fnuz:
            return raw == 0x80U;
        case ScalarType::E5M3:
        case ScalarType::E8M0:
        case ScalarType::E8M0Zero:
            return raw == 0xffU;
        case ScalarType::E4M3:
            return (raw & 0x7fU) == 0x7fU;
        default:
            return false;
    }
}

bool expectedInfinity(ScalarType type, uint32_t raw) {
    return type == ScalarType::Float8E5M2 && (raw & 0x7fU) == 0x7cU;
}

float expectedBinaryDecode(ScalarType type, uint32_t raw) {
    if (expectedNaN(type, raw)) return std::numeric_limits<float>::quiet_NaN();

    const auto format = expectedFormat(type);
    const uint32_t signMask = format.hasSign ? 1U << (format.totalBits - 1U) : 0U;
    const bool negative = format.hasSign && (raw & signMask) != 0;
    if (expectedInfinity(type, raw))
        return negative ? -std::numeric_limits<float>::infinity()
                        : std::numeric_limits<float>::infinity();

    const uint32_t payloadMask = (1U << format.totalBits) - 1U;
    const uint32_t magnitude = format.hasSign ? raw & (signMask - 1U) : raw & payloadMask;
    const uint32_t exponentMask = (1U << format.exponentBits) - 1U;
    const uint32_t mantissaMask = (1U << format.mantissaBits) - 1U;
    const uint32_t exponent = (magnitude >> format.mantissaBits) & exponentMask;
    const uint32_t mantissa = magnitude & mantissaMask;
    const float fraction =
        static_cast<float>(mantissa) / static_cast<float>(1U << format.mantissaBits);
    const float positive =
        exponent == 0
            ? std::ldexp(fraction, 1 - format.exponentBias)
            : std::ldexp(1.0f + fraction, static_cast<int>(exponent) - format.exponentBias);
    return negative ? -positive : positive;
}

}  // namespace

void testScalarTypeInfoContract();
void testExhaustiveBinaryFormat(roc::host_numerics::ScalarType type);
void testIntegerConversionPrimitives();
void testNearestEvenIgnoresHostRoundingMode();
void testIntegerCodecPolicies();
void testScalarEncodings();
}  // namespace scalar_codec_test
