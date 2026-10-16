// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <roc/hostnumerics/generation.hpp>
#include <span>

#include "generation_primitives.hpp"

namespace roc::hostnumerics::detail {
inline double uniformUnitFromMantissa(uint64_t mantissa) {
    constexpr double inverseTwoTo53 = 1.0 / 9007199254740992.0;
    return (static_cast<double>(mantissa) + 0.5) * inverseTwoTo53;
}

inline double indexedUniformUnit(uint64_t seed, uint64_t domain, uint64_t index) {
    const uint64_t mantissa = counterRandom(seed, domain, index) >> 11;
    return uniformUnitFromMantissa(mantissa);
}
// Prepared once for a validated source encoding and exponent interval.
struct PreparedEncodedExponent {
    int lowerExponent = 0;
    uint64_t exponentCount = 0;
    int exponentBias = 0;
    int mantissaBits = 0;
    uint64_t exponentMask = 0;
    uint64_t storageMask = 0;
    double (*decode)(uint64_t) = nullptr;

    double operator()(uint64_t randomBits) const {
        const int64_t exponent = static_cast<int64_t>(randomBits % exponentCount) + lowerExponent;
        const uint64_t encoded =
            (randomBits & ~exponentMask) |
            ((static_cast<uint64_t>(exponent + exponentBias) << mantissaBits) & exponentMask);
        return decode(encoded & storageMask);
    }
};
PreparedEncodedExponent prepareEncodedExponent(
    const RandomEncodedExponentGenerationParameters& parameters, ScalarType destinationType);
ScalarType generationComponentType(ScalarType type);
std::span<const uint8_t> finiteEncodedValues(ScalarType type);
double typeMaximum(ScalarType requestedType);
double typeLowest(ScalarType requestedType);
double typeDenormalMinimum(ScalarType requestedType);
double typeDenormalMaximum(ScalarType requestedType);
double typeNaN(ScalarType type);
double typeInfinity(ScalarType type, bool negative);
double randomEncodedExponentValue(const RandomEncodedExponentGenerationParameters& parameters,
                                  uint64_t seed, uint64_t domain, size_t logicalIndex,
                                  ScalarType destinationType);
uint64_t uniformFiniteEncodedValue(ScalarType type, uint64_t seed, uint64_t domain,
                                   size_t logicalIndex);
}  // namespace roc::hostnumerics::detail
