// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <type_traits>

#include "comparison_common.hpp"

namespace roc::host_numerics::detail {
double valueMagnitude(const ComparisonValue& value);
uint64_t orderedFloatingEncoding(uint64_t raw, uint32_t bitCount);
double encodedUlpDistance(double exact, double approximation, ScalarType type);
double ulpDistanceForType(double exact, double approximation, ScalarType type,
                          UlpComparisonMode mode);

template <typename Left, typename Right>
bool integralValuesEqual(Left left, Right right) {
    static_assert(std::is_integral_v<Left> && std::is_integral_v<Right>);
    if constexpr (std::is_signed_v<Left>) {
        if (left < 0) {
            if constexpr (!std::is_signed_v<Right>) return false;
        }
    }
    if constexpr (std::is_signed_v<Right>) {
        if (right < 0) {
            if constexpr (!std::is_signed_v<Left>) return false;
        }
    }
    if constexpr (std::is_signed_v<Left> == std::is_signed_v<Right>)
        return left == right;
    else
        return static_cast<uintmax_t>(left) == static_cast<uintmax_t>(right);
}

struct ExactRealEvidence {
    ComponentResult component;
    long double observedMagnitude = 0.0L;
    long double expectedMagnitude = 0.0L;
    long double difference = 0.0L;
};
}  // namespace roc::host_numerics::detail
